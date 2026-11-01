"""Frozen black/white annotation panels on an exact decoded-frame timeline.

This module plans and paints panels only. It does not encode video, claim media
admission, start a queue, truncate text or synthesize a constant frame rate.
An encoder must preserve the admitted proxy's real timestamps and full pixels.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import copy
import hashlib
import io
import struct
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import schema as S
from .professional_schema import validate_media
from .errors import SchemaInvalid

VERSION = "annotation-panel-v2-word-boundaries"
MAX_FONT_BYTES = 64 * 1024 * 1024
MAX_FRAME_COUNT = 2_000_000
MAX_ANNOTATIONS = 20_000
MAX_PANEL_TEXT = 32_000
MAX_CANVAS_PIXELS = 32 * 1024 * 1024
FIELDS = ("label", "text", "interval", "track", "tags", "review_status")


def _fail(code, message):
    raise SchemaInvalid(code, message)


def _cancel(cancelled):
    """Poll cooperative control without hiding its typed deadline/lease errors."""
    if cancelled is not None and cancelled():
        raise InterruptedError("Annotation layout was cancelled")


def _integer(value, low, high, field):
    if type(value) is not int or not low <= value <= high:
        _fail("INVALID_RENDER_RECIPE", "Invalid " + field)
    return value


def normalize_recipe(recipe):
    allowed={"version","position","panel_extent_px","body_font_px","title_font_px","padding_px",
             "line_spacing_px","fields","tracks","font_sha256","font_license","encoding"}
    if not isinstance(recipe,dict) or set(recipe)!=allowed:
        _fail("INVALID_RENDER_RECIPE","Render recipe has missing or unknown fields")
    if recipe["version"]!=1 or type(recipe["version"]) is not int or recipe["position"] not in ("top","bottom","left","right"):
        _fail("INVALID_RENDER_RECIPE","Unsupported layout version or position")
    for key,low,high in (("panel_extent_px",64,4096),("body_font_px",12,96),("title_font_px",12,128),("padding_px",8,256),("line_spacing_px",0,64)):
        _integer(recipe[key],low,high,key)
    for key,maximum in (("fields",len(FIELDS)),("tracks",256)):
        values=recipe[key]
        if not isinstance(values,list) or len(values)>maximum or any(not isinstance(v,str) or not v for v in values) or len(set(values))!=len(values):
            _fail("INVALID_RENDER_RECIPE","Invalid selected " + key)
    if not recipe["fields"] or any(f not in FIELDS for f in recipe["fields"]):
        _fail("INVALID_RENDER_RECIPE","Select at least one supported display field")
    S.require_hash(recipe["font_sha256"],"font_sha256")
    if not isinstance(recipe["font_license"],str) or not recipe["font_license"].strip() or len(recipe["font_license"])>1024:
        _fail("INVALID_RENDER_RECIPE","A font source/license declaration is required")
    enc=recipe["encoding"]
    if not isinstance(enc,dict) or set(enc)!={"video_codec","pixel_format","crf","preset","audio_codec","audio_bitrate_kbps","color_policy"}:
        _fail("INVALID_RENDER_RECIPE","Encoding parameters must be explicit")
    if enc["video_codec"]!="libx264" or enc["pixel_format"]!="yuv420p" or enc["preset"] not in ("veryfast","fast","medium","slow") or enc["audio_codec"] not in ("aac","copy","none"):
        _fail("UNSUPPORTED_RENDER_ENCODING","Unsupported encoding configuration")
    if enc['color_policy']!='preserve_declared_proxy':_fail("UNSUPPORTED_RENDER_ENCODING","Unknown proxy color tags must not be fabricated")
    _integer(enc["crf"],0,51,"crf");_integer(enc["audio_bitrate_kbps"],32,320,"audio_bitrate_kbps")
    return copy.deepcopy(recipe)


def _cmap(data):
    """Read bounded Unicode SFNT cmap4/12; reject collections/unknown formats."""
    if len(data)<12 or data[:4] not in (b"\x00\x01\x00\x00",b"OTTO",b"true"):
        _fail("UNSUPPORTED_RENDER_FONT","Use a single TrueType/OpenType font, not a collection")
    count=struct.unpack_from(">H",data,4)[0]
    if count>4096 or 12+count*16>len(data):
        _fail("INVALID_RENDER_FONT","Font table directory is invalid")
    tables={}
    for i in range(count):
        tag,_,offset,length=struct.unpack_from(">4sIII",data,12+i*16)
        if tag in tables or offset+length>len(data):_fail("INVALID_RENDER_FONT","Font table boundary is invalid")
        tables[tag]=(offset,length)
    if b"cmap" not in tables:_fail("UNSUPPORTED_RENDER_FONT","Font lacks a Unicode character map")
    offset,length=tables[b"cmap"];end=offset+length
    if length<4:_fail("INVALID_RENDER_FONT","Character map header is invalid")
    count=struct.unpack_from(">H",data,offset+2)[0]
    if count>4096 or offset+4+count*8>end:_fail("INVALID_RENDER_FONT","Character map directory is invalid")
    supported=set();recognized=False
    for i in range(count):
        platform,encoding,sub=struct.unpack_from(">HHI",data,offset+4+i*8)
        if platform!=0 and not (platform==3 and encoding in (1,10)):continue
        p=offset+sub
        if p+2>end:_fail("INVALID_RENDER_FONT","Character map subtable escapes its table")
        fmt=struct.unpack_from(">H",data,p)[0]
        if fmt==12:
            if p+16>end:_fail("INVALID_RENDER_FONT","Character map12 header is invalid")
            size=struct.unpack_from(">I",data,p+4)[0];groups=struct.unpack_from(">I",data,p+12)[0]
            if groups>100_000 or p+size>end or 16+groups*12>size:_fail("INVALID_RENDER_FONT","Character map12 exceeds its bound")
            previous=-1
            for j in range(groups):
                a,z,g=struct.unpack_from(">III",data,p+16+j*12)
                if a>z or a<=previous or z>0x10FFFF:_fail("INVALID_RENDER_FONT","Character map12 range is invalid")
                supported.update(range(a+(g==0),z+1));previous=z
            recognized=True
        elif fmt==4:
            if p+14>end:_fail("INVALID_RENDER_FONT","Character map4 header is invalid")
            size=struct.unpack_from(">H",data,p+2)[0];segs=struct.unpack_from(">H",data,p+6)[0]//2
            if not segs or p+size>end or 16+segs*8>size:_fail("INVALID_RENDER_FONT","Character map4 exceeds its bound")
            previous=-1
            for j in range(segs):
                z=struct.unpack_from(">H",data,p+14+2*j)[0]
                a=struct.unpack_from(">H",data,p+16+2*segs+2*j)[0]
                delta=struct.unpack_from(">h",data,p+16+4*segs+2*j)[0]
                rpos=p+16+6*segs+2*j;ro=struct.unpack_from(">H",data,rpos)[0]
                if a>z or a<=previous:_fail("INVALID_RENDER_FONT","Character map4 ranges overlap")
                for c in range(a,z+1):
                    if c==65535:continue
                    if ro:
                        loc=rpos+ro+2*(c-a)
                        if loc+2>p+size:_fail("INVALID_RENDER_FONT","Character map4 glyph escapes its subtable")
                        raw=struct.unpack_from(">H",data,loc)[0];glyph=((raw+delta)&65535) if raw else 0
                    else:glyph=(c+delta)&65535
                    if glyph:supported.add(c)
                previous=z
            recognized=True
    if not recognized:_fail("UNSUPPORTED_RENDER_FONT","Font has no supported Unicode cmap4/12")
    return frozenset(supported)


class FontFace:
    """Verified immutable font bytes, never silently replaced by a fallback."""
    def __init__(self,path,expected_sha256):
        S.require_hash(expected_sha256,"font_sha256")
        p=Path(path)
        if not p.is_file() or p.stat().st_size>MAX_FONT_BYTES:
            _fail("RENDER_FONT_UNAVAILABLE","Font is missing or exceeds64MiB")
        self.data=p.read_bytes();self.sha256=hashlib.sha256(self.data).hexdigest()
        if self.sha256!=expected_sha256:_fail("RENDER_FONT_CHANGED","Font hash differs from frozen configuration")
        try:self.coverage=_cmap(self.data)
        except struct.error as exc:raise SchemaInvalid("INVALID_RENDER_FONT","Font structure is invalid") from exc
        self._sizes={}
        self.font(16)

    def font(self,size):
        if size not in self._sizes:
            try:self._sizes[size]=ImageFont.truetype(io.BytesIO(self.data),size)
            except (OSError,ValueError) as exc:raise SchemaInvalid("UNSUPPORTED_RENDER_FONT","Pillow cannot load this font") from exc
        return self._sizes[size]

    def check(self,text):
        if len(text)>MAX_PANEL_TEXT:_fail("RENDER_TEXT_LIMIT","Selected panel text exceeds32000 characters")
        for c in text:
            if c=='\n':continue
            if unicodedata.category(c) in ("Cc","Cf","Cs") or 0xFE00<=ord(c)<=0xFE0F:
                _fail("UNSUPPORTED_RENDER_TEXT","Control, joiner or variation-sequence layout is unsupported")
            if ord(c) not in self.coverage:
                _fail("RENDER_MISSING_GLYPH","Configured font lacks a selected text glyph")


def _clusters(text):
    result=[]
    for c in text:
        if unicodedata.combining(c) and result:result[-1]+=c
        else:result.append(c)
    return result


def _width(font,text):
    bbox=font.getbbox(text,anchor="lt")
    return max(float(font.getlength(text)),bbox[2]-bbox[0])


CLOSING_PUNCTUATION='，。！？；：、）》】”’％%)]},.;:!?'
OPENING_PUNCTUATION='（《【“‘([{'


def _ascii_cluster(cluster):
    # Treat non-space ASCII runs (words, numbers and URLs) as a preferred
    # unbroken unit. Combining accents stay attached to their base cluster.
    return bool(cluster) and ord(cluster[0])<128 and not cluster[0].isspace()


def wrap_text(text,face,size,width,*,cancelled=None):
    """Greedy full-text wrapping with explicit, bounded typography rules.

    Fitting ASCII runs remain intact. An overwide run may break between
    clusters; a Chinese/ASCII closing mark may not start an automatic line,
    nor an opening mark end it. No final-line balancing moves characters back
    across these guards. This is not a complete Unicode line-break engine.
    Original spaces and authored paragraph breaks are never stripped.
    """
    _cancel(cancelled)
    face.check(text);font=face.font(size);result=[]
    if not isinstance(width,(int,float)) or isinstance(width,bool) or not 0<width<=8192:
        _fail('RENDER_GLYPH_TOO_WIDE','Text area width must be positive and bounded')
    for paragraph in text.split('\n'):
        _cancel(cancelled)
        clusters=_clusters(paragraph);count=len(clusters);lines=[]
        if not count:
            result.append('');continue
        previous=[];last=-1
        for index,cluster in enumerate(clusters):
            if index%64==0:_cancel(cancelled)
            previous.append(last)
            if not cluster.isspace():last=index
        previous.append(last)
        following=[count]*(count+1);next_visible=count
        for index in range(count-1,-1,-1):
            if index%64==0:_cancel(cancelled)
            if not clusters[index].isspace():next_visible=index
            following[index]=next_visible
        ascii_runs={};run_widths={};index=0
        while index<count:
            _cancel(cancelled)
            if not _ascii_cluster(clusters[index]):index+=1;continue
            beginning=index
            while index<count and _ascii_cluster(clusters[index]):
                if index%64==0:_cancel(cancelled)
                ascii_runs[index]=beginning;index+=1
            run_widths[beginning]=_width(font,''.join(clusters[beginning:index]))

        def punctuation_boundary(start,end):
            left=previous[end];right=following[end]
            return not (left>=start and clusters[left][0] in OPENING_PUNCTUATION
                        or right<count and clusters[right][0] in CLOSING_PUNCTUATION)

        start=0;measurements=0
        while start<count:
            _cancel(cancelled)
            first=following[start]
            if first<count and clusters[first][0] in CLOSING_PUNCTUATION:
                _fail('RENDER_LINE_BREAK_UNREPRESENTABLE','Closing punctuation cannot begin a rendered line')
            end=start;line=''
            while end<count:
                if measurements%64==0:_cancel(cancelled)
                candidate=line+clusters[end];measurements+=1
                if _width(font,candidate)>width:break
                line=candidate;end+=1
            if end==start:
                _fail('RENDER_GLYPH_TOO_WIDE','A glyph cannot fit the configured panel width')
            cut=None
            # Prefer a natural boundary. Splitting an ASCII run is a fallback
            # only when no whole fitting unit can finish this line.
            for boundary in range(end,start,-1):
                if boundary%64==0:_cancel(cancelled)
                if not punctuation_boundary(start,boundary):continue
                if boundary==count or not (_ascii_cluster(clusters[boundary-1]) and _ascii_cluster(clusters[boundary])):
                    cut=boundary;break
            if cut is None:
                for boundary in range(end,start,-1):
                    if boundary%64==0:_cancel(cancelled)
                    if not punctuation_boundary(start,boundary):continue
                    if (boundary<count and _ascii_cluster(clusters[boundary-1]) and _ascii_cluster(clusters[boundary])
                            and run_widths[ascii_runs[boundary]]<=width):continue
                    cut=boundary;break
            if cut is None:
                _fail('RENDER_LINE_BREAK_UNREPRESENTABLE','Punctuation group cannot fit without an invalid line break')
            lines.append(''.join(clusters[start:cut]));start=cut
        if ''.join(lines)!=paragraph or any(_width(font,line)>width for line in lines):
            _fail('RENDER_LAYOUT_ERROR','Text wrapping failed full-content preservation')
        result.extend(lines)
    _cancel(cancelled)
    return result


def _seconds(value):
    value=S.rational_value(value)
    # Human labels round to milliseconds; frame membership never uses this text.
    sign='-' if value<0 else '';value=abs(value)
    milliseconds=(value.numerator*2000+value.denominator)//(value.denominator*2)
    return f'{sign}{milliseconds//1000}.{milliseconds%1000:03d}'


def _field(annotation,field):
    if field in ("label","text"):return annotation[field]
    if field=="track":return 'Track: '+annotation['track']
    if field=="tags":return 'Tags: '+', '.join(annotation['tags'])
    if field=="review_status":return 'Review: '+annotation['review_status']
    if annotation['type']=='segment':return _seconds(annotation['start']['seconds'])+' - '+_seconds(annotation['end']['seconds'])+' s'
    if annotation['type']=='event':return 'Event: '+_seconds(annotation['at']['seconds'])+' s'
    return 'Whole clip'


def _panel_entries(annotations,recipe,face,size,*,cancelled=None):
    w,h=size;pad=recipe['padding_px'];width=w-2*pad;y=pad;entries=[]
    if width<=0 or h<=2*pad:_fail("RENDER_TEXT_OVERFLOW","Panel padding leaves no text area")
    for annotation in annotations:
        _cancel(cancelled)
        for field in recipe['fields']:
            _cancel(cancelled)
            text=_field(annotation,field)
            fontsize=recipe['title_font_px'] if field=='label' else recipe['body_font_px']
            font=face.font(fontsize)
            for line in wrap_text(text,face,fontsize,width,cancelled=cancelled):
                _cancel(cancelled)
                box=font.getbbox(line or ' ',anchor='lt');ink_height=box[3]-box[1]
                line_height=max(ink_height,fontsize)+recipe['line_spacing_px']
                if y+max(ink_height,fontsize)>h-pad:_fail("RENDER_TEXT_OVERFLOW","Full selected text does not fit; enlarge panel or select fewer fields")
                entries.append({'text':line,'source_field':field,'annotation_id':annotation['id'],'font_px':fontsize,'x':pad-box[0],'y':y,'ink_width':box[2]-box[0],'ink_height':ink_height})
                y+=line_height
            y+=recipe['line_spacing_px']
        y+=recipe['body_font_px']//2
    return entries


def _prepare_plan(task,media,frame_rows,recipe,*,font_path=None,font=None,playback_size,frame_map_sha256,cancelled=None):
    """Freeze full-field panels and exact frame-index spans from verified rows.

    The caller must supply the actual proxy dimensions and verified immutable
    frame-map rows, not average FPS. Selected temporal bounds are left-closed,
    right-open at actual display timestamps; events map to their display frame.
    """
    _cancel(cancelled)
    validate_media(media);recipe=normalize_recipe(recipe)
    S.require_hash(frame_map_sha256,"frame_map_sha256")
    S.require_id(task.get('id'));S.require_hash(task.get('content_sha256'))
    if task.get('media_id')!=media['id'] or type(task.get('revision')) is not int or task['revision']<0:
        _fail("INVALID_RENDER_TASK","Task media or revision binding is invalid")
    if not isinstance(task.get('annotations'),list) or len(task['annotations'])>MAX_ANNOTATIONS:
        _fail("RENDER_LIMIT_EXCEEDED","Too many annotations for render planning")
    if not isinstance(playback_size,(tuple,list)) or len(playback_size)!=2:
        _fail("INVALID_RENDER_GEOMETRY","Actual proxy dimensions are required")
    sw,sh=[_integer(v,2,8192,'playback dimension') for v in playback_size]
    if sw%2 or sh%2:_fail("UNSUPPORTED_RENDER_GEOMETRY","The selected yuv420 encoder needs even proxy dimensions")
    face=font or FontFace(font_path,recipe['font_sha256'])
    if face.sha256!=recipe['font_sha256']:_fail("RENDER_FONT_CHANGED","Font differs from recipe")
    extent=recipe['panel_extent_px']
    if extent%2:_fail("INVALID_RENDER_GEOMETRY","Panel extent must be even for yuv420 output")
    horizontal=recipe['position'] in ('left','right')
    canvas=[sw+extent,sh] if horizontal else [sw,sh+extent]
    if max(canvas)>8192 or canvas[0]*canvas[1]>MAX_CANVAS_PIXELS:
        _fail("RENDER_LIMIT_EXCEEDED","Canvas exceeds render pixel budget")
    source_rect=[extent if recipe['position']=='left' else 0,extent if recipe['position']=='top' else 0,sw,sh]
    panel_rect=[sw if recipe['position']=='right' else 0,sh if recipe['position']=='bottom' else 0,extent if horizontal else sw,sh if horizontal else extent]
    times=[];previous_source=None;timeline_hash=hashlib.sha256()
    for i,row in enumerate(frame_rows):
        if i%64==0:_cancel(cancelled)
        if i>=MAX_FRAME_COUNT:_fail("RENDER_LIMIT_EXCEEDED","Frame map exceeds supported count")
        if not isinstance(row,dict) or set(row)!={'frame_index','source_pts','proxy_pts','seconds'} or row['frame_index']!=str(i):
            _fail("INVALID_FRAME_MAP","Render frame map is not contiguous")
        for key in ('source_pts','proxy_pts'):
            if not isinstance(row[key],str) or len(row[key])>40 or str(int(row[key]))!=row[key]:_fail("INVALID_FRAME_MAP","Frame PTS must be canonical integers")
        t=S.rational_value(row['seconds'])
        if S.rational(t)!=row['seconds'] or t!=int(row['proxy_pts'])*S.rational_value(media['proxy_time_base']):_fail("INVALID_FRAME_MAP","Frame timestamp differs from proxy PTS")
        source_pts=int(row['source_pts'])
        if (times and (t<=times[-1] or source_pts<=previous_source)) or abs(source_pts*S.rational_value(media['source_time_base'])-S.rational_value(media['common_origin'])-t)>min(S.rational_value(media['source_time_base']),S.rational_value(media['proxy_time_base'])):
            _fail("INVALID_FRAME_MAP","Source/proxy frame clocks disagree")
        times.append(t);previous_source=source_pts;timeline_hash.update(S.canonical_json(row)+b'\n')
    if not times or str(len(times))!=media['frame_count'] or times[0]!=S.rational_value(media['video_start']) or times[-1]>=S.rational_value(media['video_end']):
        _fail("INVALID_FRAME_MAP","Frame count/active endpoint differs from media")
    _cancel(cancelled)
    selected=[];seen=set();events={0:{'start':[],'end':[]},len(times):{'start':[],'end':[]}}
    for annotation in task['annotations']:
        _cancel(cancelled)
        if recipe['tracks'] and annotation['track'] not in recipe['tracks']:continue
        ident=S.require_id(annotation['id'])
        if ident in seen:_fail("INVALID_RENDER_TASK","Duplicate selected annotation identifier")
        seen.add(ident)
        kind=annotation['type']
        if kind=='whole_clip':start,end=0,len(times)
        elif kind=='event':
            at=S.rational_value(annotation['at']['seconds'])
            if not times[0]<=at<S.rational_value(media['video_end']):_fail("RENDER_TIME_OUTSIDE_VIDEO","Selected event has no active video frame")
            start=bisect_right(times,at)-1;end=start+1
        elif kind=='segment':
            a,b=[S.rational_value(annotation[key]['seconds']) for key in ('start','end')]
            if not times[0]<=a<b<=S.rational_value(media['video_end']):_fail("RENDER_TIME_OUTSIDE_VIDEO","Selected segment exceeds active video interval")
            start,end=bisect_left(times,a),bisect_left(times,b)
            if start==end:_fail("RENDER_INTERVAL_WITHOUT_FRAME","Selected interval contains no frame display timestamp")
        else:_fail("UNSUPPORTED_RENDER_ANNOTATION","Unsupported selected annotation type")
        selected.append(copy.deepcopy(annotation))
        events.setdefault(start,{'start':[],'end':[]})['start'].append(ident)
        events.setdefault(end,{'start':[],'end':[]})['end'].append(ident)
    if recipe['tracks'] and set(recipe['tracks'])-set(a['track'] for a in task['annotations']):
        _fail("RENDER_TRACK_NOT_FOUND","Selected track does not exist")
    order={a['id']:i for i,a in enumerate(selected)};byid={a['id']:a for a in selected}
    active=set();panels={};spans=[];points=sorted(events)
    for index,start in enumerate(points[:-1]):
        _cancel(cancelled)
        active.difference_update(events[start]['end']);active.update(events[start]['start'])
        ids=sorted(active,key=order.__getitem__)
        panel_id=S.make_id('panel',S.sha256_text(ids))
        if panel_id not in panels:
            entries=_panel_entries([byid[ident] for ident in ids],recipe,face,panel_rect[2:],cancelled=cancelled)
            panels[panel_id]={'annotation_ids':ids,'entries':entries}
        spans.append({'start_frame':str(start),'end_frame_exclusive':str(points[index+1]),'panel_id':panel_id})
    plan={'kind':'traceannotate.render-layout','version':1,'renderer_version':VERSION,
          'binding':{'task_id':task['id'],'revision':task['revision'],'task_content_sha256':task['content_sha256'],
                     'media_id':media['id'],'source_sha256':media['source']['sha256'],'playback_sha256':media['playback']['sha256'],
                     'frame_map_artifact_id':media['frame_map_artifact'],'frame_map_sha256':frame_map_sha256,
                     'frame_timeline_sha256':timeline_hash.hexdigest(),'font_sha256':face.sha256},
          'recipe':recipe,'geometry':{'canvas':canvas,'source_rect':source_rect,'panel_rect':panel_rect,'source_crop':False,'source_scale':False},
          'frame_count':media['frame_count'],'proxy_time_base':copy.deepcopy(media['proxy_time_base']),
          'video_start':copy.deepcopy(media['video_start']),'video_end':copy.deepcopy(media['video_end']),
          'panels':panels,'spans':spans,'selected_annotations':selected,
          'scope':'Frozen full selected fields on exact decoded-frame indices; font license and annotation truth are declarations, not independently certified.',
          'time_policy':'segment membership uses frame display PTS in [start,end); events occupy their containing display frame; whole_clip covers active video frames',
          'time_display':'Human labels round to the nearest millisecond; exact rational anchors and frame membership are unchanged.',
          'readability':'Full text on short shots may require pause; source timing is not lengthened.'}
    _cancel(cancelled)
    plan['content_sha256']=S.sha256_text(plan)
    _cancel(cancelled)
    return plan


def prepare_plan(task,media,frame_rows,recipe,*,font_path=None,font=None,playback_size,frame_map_sha256,cancelled=None):
    if cancelled is not None and not callable(cancelled):
        _fail("INVALID_RENDER_CALLBACK","Layout cancellation must be callable")
    try:
        return _prepare_plan(task,media,frame_rows,recipe,font_path=font_path,font=font,
                             playback_size=playback_size,frame_map_sha256=frame_map_sha256,cancelled=cancelled)
    except SchemaInvalid:
        raise
    except (TypeError,ValueError,KeyError,OverflowError,AttributeError) as exc:
        raise SchemaInvalid("INVALID_RENDER_PLAN_INPUT","Render input is malformed or unsupported") from exc


def _verify_plan(plan):
    if not isinstance(plan,dict) or plan.get('kind')!='traceannotate.render-layout' or plan.get('version')!=1:
        _fail("INVALID_RENDER_PLAN","Unsupported render plan")
    S.require_hash(plan.get('content_sha256'))
    if S.sha256_text({k:v for k,v in plan.items() if k!='content_sha256'})!=plan['content_sha256']:
        _fail("RENDER_PLAN_CHANGED","Frozen render plan bytes changed")


def panel_for_frame(plan,frame_index):
    _verify_plan(plan)
    _integer(frame_index,0,int(plan['frame_count'])-1,'frame index')
    starts=[int(s['start_frame']) for s in plan['spans']]
    return plan['spans'][bisect_right(starts,frame_index)-1]['panel_id']


def render_panel(plan,panel_id,*,font_path=None,font=None):
    _verify_plan(plan)
    if panel_id not in plan['panels']:_fail("RENDER_PANEL_NOT_FOUND","No panel in frozen plan")
    face=font or FontFace(font_path,plan['binding']['font_sha256'])
    if face.sha256!=plan['binding']['font_sha256']:_fail("RENDER_FONT_CHANGED","Font bytes differ from frozen plan")
    image=Image.new('RGB',tuple(plan['geometry']['panel_rect'][2:]),'black');draw=ImageDraw.Draw(image)
    for entry in plan['panels'][panel_id]['entries']:
        draw.text((entry['x'],entry['y']),entry['text'],font=face.font(entry['font_px']),fill='white',anchor='lt')
    return image


def composite_frame(plan,frame,frame_index,*,font_path=None,font=None):
    """Reference composition for previews; full encoder service remains separate."""
    _verify_plan(plan)
    if frame.size!=tuple(plan['geometry']['source_rect'][2:]):_fail("RENDER_MEDIA_CHANGED","Decoded frame geometry differs from frozen plan")
    output=Image.new('RGB',tuple(plan['geometry']['canvas']),'black')
    output.paste(frame.convert('RGB'),tuple(plan['geometry']['source_rect'][:2]))
    output.paste(render_panel(plan,panel_for_frame(plan,frame_index),font_path=font_path,font=font),tuple(plan['geometry']['panel_rect'][:2]))
    return output
