"""Verify a frozen annotation-video candidate without database publication.

One-frame PNG inputs repeat on the admitted video's exact frame timeline. No
ASS centisecond conversion, FPS synthesis, source crop or source scaling occurs.
Only a fully decoded, verified candidate obtains a ``validated`` directory.
"""
from __future__ import annotations

from fractions import Fraction
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

from PIL import Image, ImageChops, ImageStat

from ..temporal import preparation as P
from ..temporal.media import stream_sha256
from . import render_layout as L, schema as S
from .errors import Conflict, DependencyUnavailable, NotFound, SchemaInvalid, StorageFull, WorkspaceError

VERSION = "panel-overlay-exact-pts-v1"
MAX_UNIQUE_PANELS = 128
MAX_SPANS = 256
MAX_BOUNDARY_SAMPLES = 2 * MAX_SPANS
MAX_FILTER_PIXELS = 256 * 1024 * 1024
MAX_AUDIO_STREAMS = 8
MAX_PANEL_MAE = 4.0
DEFAULT_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
MIN_FREE_RESERVE = 256 * 1024 * 1024
COLOR_KEYS = ("color_primaries", "color_transfer", "color_space", "color_range")


def _write(path, value):
    with path.open("xb") as stream:
        stream.write(S.canonical_json(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def _safe(path, *, missing=False):
    for part in [path, *path.parents]:
        try:
            info=part.lstat()
        except FileNotFoundError:
            if missing and part==path:continue
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0) & 0x400:
            raise SchemaInvalid("UNSAFE_RENDER_PATH", "Render paths must not traverse symlinks or reparse points")


def _space(parent, required):
    if shutil.disk_usage(parent).free < required:
        raise StorageFull("STORAGE_FULL", "Insufficient reserved space for bounded rendering", retryable=True,
                          details={"required_free_bytes":required})


def _check_output_size(path, limit):
    """FFmpeg -fs is an early-stop hint; only this final check enforces cap."""
    actual=path.stat().st_size
    if actual>limit:
        raise WorkspaceError('RENDER_OUTPUT_LIMIT','Rendered candidate exceeds the exact final byte cap',
                             details={'maximum_bytes':str(limit),'actual_bytes':str(actual)})
    return actual


def _storage_failure_details(exc, work, target):
    """Prefer OS errno; classify bounded, actual tool diagnostics conservatively.

    FFmpeg has no structured errno channel through the existing runner. A
    diagnostic marker is explicitly reported as a tool declaration, not proof
    that this process observed a kernel ENOSPC. Other tool failures stay generic.
    """
    current=exc;seen=set()
    while current is not None and id(current) not in seen and len(seen)<8:
        seen.add(id(current))
        if isinstance(current,OSError) and (current.errno in (errno.ENOSPC,getattr(errno,'EDQUOT',errno.ENOSPC)) or getattr(current,'winerror',None) in (39,112)):
            return {'resource_detection':'os_errno','errno':current.errno,'winerror':getattr(current,'winerror',None)}
        current=current.__cause__ or current.__context__
    if not isinstance(exc,P.MediaPreparationError):return None
    try:
        free=shutil.disk_usage(target).free
    except OSError:free=None
    if free is not None and free<MIN_FREE_RESERVE:
        return {'resource_detection':'reserve_below_threshold_after_tool_failure','free_bytes':str(free),'reserve_bytes':str(MIN_FREE_RESERVE)}
    if work is None:return None
    try:
        log=work/'commands.jsonl'
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-65536));lines=stream.read().splitlines()
        record=json.loads(lines[-1]);name=record.get('stderr_file')
        if not isinstance(name,str) or re.fullmatch(r'[a-z][a-z0-9-]{0,80}\.stderr',name) is None:return None
        path=work/name
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-16384));tail=stream.read().decode('utf-8','replace')
    except (OSError,ValueError,TypeError,KeyError,IndexError):return None
    markers=(
        ('tool_reported_errno',r'(?i)\b(?:errno|error number)\s*(?:[=:]\s*)?-?(?:28|122)\b|\bwinerror\s*(?:[=:]\s*)?(?:39|112)\b'),
        ('tool_reported_space_marker',r'(?im)(?:^|:\s*)(?:ENOSPC|EDQUOT)(?:\s*[:\]]|\s*$)'),
        ('tool_reported_space_message',r'(?im)(?:^|:\s*)(?:No space left on device|Disk quota exceeded)\s*[.;]?\s*$'),
    )
    for basis,pattern in markers:
        if re.search(pattern,tail):
            return {'resource_detection':basis,'diagnostic_file':name,'kernel_errno_independently_observed':False,'free_bytes':None if free is None else str(free)}
    return None


def _plan(plan):
    L._verify_plan(plan)
    if L.normalize_recipe(plan['recipe'])!=plan['recipe']:
        raise SchemaInvalid("INVALID_RENDER_PLAN", "Recipe differs from frozen normalized configuration")
    count=int(plan['frame_count']);panels=plan['panels'];spans=plan['spans']
    if not 1<=count<=L.MAX_FRAME_COUNT or not isinstance(panels,dict) or not 1<=len(panels)<=MAX_UNIQUE_PANELS or not isinstance(spans,list) or not 1<=len(spans)<=MAX_SPANS:
        raise SchemaInvalid("RENDER_LIMIT_EXCEEDED", "Panel or span count exceeds the bounded encoder")
    canvas=plan['geometry']['canvas'];rect=plan['geometry']['panel_rect'];source=plan['geometry']['source_rect']
    if any(type(v) is not int or not 0<=v<=8192 for v in canvas+rect+source) or len(canvas)!=2 or len(rect)!=4 or len(source)!=4:
        raise SchemaInvalid("INVALID_RENDER_PLAN", "Invalid canvas geometry")
    if plan['geometry'].get('source_crop') is not False or plan['geometry'].get('source_scale') is not False:
        raise SchemaInvalid("UNSUPPORTED_RENDER_PLAN", "Source cropping or scaling is unsupported")
    w,h=canvas;sw,sh=source[2:];extent=plan['recipe']['panel_extent_px'];pos=plan['recipe']['position']
    expected_canvas=[sw+extent,sh] if pos in ('left','right') else [sw,sh+extent]
    expected_source=[extent if pos=='left' else 0,extent if pos=='top' else 0,sw,sh]
    expected_panel=[sw if pos=='right' else 0,sh if pos=='bottom' else 0,extent if pos in ('left','right') else sw,sh if pos in ('left','right') else extent]
    if canvas!=expected_canvas or source!=expected_source or rect!=expected_panel or min(w,h,sw,sh)<=0 or w%2 or h%2 or w*h>L.MAX_CANVAS_PIXELS:
        raise SchemaInvalid("INVALID_RENDER_PLAN", "Geometry does not preserve the complete source rectangle")
    if len(panels)*(w*h+rect[2]*rect[3])>MAX_FILTER_PIXELS:
        raise SchemaInvalid("RENDER_LIMIT_EXCEEDED", "Decoded panel/filter pixel budget exceeded")
    previous=0;boundaries={0,count-1}
    for span in spans:
        a,b=int(span['start_frame']),int(span['end_frame_exclusive'])
        if a!=previous or not a<b<=count or span['panel_id'] not in panels:
            raise SchemaInvalid("INVALID_RENDER_PLAN", "Render spans do not cover each frame exactly once")
        previous=b;boundaries.update((a,max(0,a-1)))
    if previous!=count:raise SchemaInvalid("INVALID_RENDER_PLAN", "Render spans do not reach the final frame")
    return sorted(boundaries)


def _balanced_sum(terms):
    """Keep FFmpeg expression recursion logarithmic, retaining every term.

    A flat a+b+c chain creates a left-deep parser tree and fails before the
    supported boundary count on actual FFmpeg builds. Parenthesized halves
    preserve the exact sum with at most ceil(log2(term_count)) additions deep.
    These bounded terms are generated internally, never supplied by a client.
    """
    if (not isinstance(terms, (list, tuple)) or not 1 <= len(terms) <= MAX_BOUNDARY_SAMPLES
            or any(not isinstance(term, str) or not term or len(term) > 128 for term in terms)):
        raise SchemaInvalid("INVALID_RENDER_PLAN", "Invalid bounded filter expression terms")
    def build(start, end):
        if end - start == 1:
            return terms[start]
        middle = (start + end) // 2
        return "(" + build(start, middle) + "+" + build(middle, end) + ")"
    return build(0, len(terms))


def _frame_select_expression(boundaries):
    """Select all sorted exact frame indices, including each transition side."""
    if (not isinstance(boundaries, (list, tuple)) or not 1 <= len(boundaries) <= MAX_BOUNDARY_SAMPLES
            or any(type(index) is not int or not 0 <= index < L.MAX_FRAME_COUNT for index in boundaries)
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise SchemaInvalid("INVALID_RENDER_PLAN", "Boundary samples must be bounded increasing exact frame indices")
    return _balanced_sum([f"eq(n\\,{index})" for index in boundaries])


def _encode_command(ffmpeg, source, panels, plan, video, audios, summary, graph, output, limit, progress_file, graph_option):
    base=P._time_base(video);timescale=base.denominator
    if timescale>2147483647:
        raise SchemaInvalid("UNSUPPORTED_RENDER_TIMEBASE", "Proxy timescale exceeds MP4 support")
    last=int(summary['last']*base*timescale);hold=int(summary['last_duration']*base*timescale)
    final_hold=f"setts=pts=PTS:dts=DTS:duration=if(eq(PTS\\,{last})\\,{hold}\\,DURATION)"
    enc=plan['recipe']['encoding']
    args=[ffmpeg,'-hide_banner','-v','error','-nostdin','-n','-xerror','-copyts',
          '-protocol_whitelist','file,pipe','-err_detect','explode','-noautorotate','-i',str(source)]
    for path in panels:
        args+=['-f','image2','-pattern_type','none','-framerate','1','-i',str(path)]
    args+=['-filter_complex_threads','1',graph_option,str(graph),'-map','[rendered]']
    if enc['audio_codec']!='none':
        for stream in audios:args+=['-map',f"0:{stream['index']}"]
    args+=['-c:v','libx264','-bf','0','-threads:v','2','-pix_fmt','yuv420p','-crf',str(enc['crf']),'-preset',enc['preset'],
           '-fps_mode:v','passthrough','-enc_time_base:v',f'1/{timescale}','-video_track_timescale',str(timescale),'-bsf:v',final_hold]
    for key,option in (('color_primaries','-color_primaries'),('color_transfer','-color_trc'),('color_space','-colorspace'),('color_range','-color_range')):
        if video.get(key) not in (None,'unknown','unspecified'):args += [option,str(video[key])]
    if enc['audio_codec']!='none':
        for i,stream in enumerate(audios):
            args += [f'-c:a:{i}',enc['audio_codec']]
            if enc['audio_codec']=='aac':args += [f'-b:a:{i}',str(enc['audio_bitrate_kbps'])+'k',f'-ar:a:{i}',str(stream['sample_rate'])]
            language=stream.get('tags',{}).get('language')
            if isinstance(language,str) and len(language)<=128:args += [f'-metadata:s:a:{i}','language='+language]
    args+=['-map_metadata','-1','-map_chapters','-1','-avoid_negative_ts','disabled','-movflags','+faststart',
           '-progress',str(progress_file),'-fs',str(limit),str(output)]
    if os.name=='nt' and sum(len(a)+3 for a in args)>28000:
        raise SchemaInvalid("RENDER_LIMIT_EXCEEDED", "Render input paths exceed the safe Windows command budget")
    return args


def _panel_arguments(work, paths):
    """Use a safe ancestor for a short, child-local cwd and bounded paths."""
    # CreateProcess on Windows may reject a long cwd even with a long-path
    # prefix. Choose the nearest ancestor short enough for a child process;
    # every relative input must still resolve underneath the private work dir.
    cwd=work
    if os.name=='nt':
        cwd=next((p for p in (work,*work.parents) if len(str(p).encode('utf-16-le'))//2<=200),None)
        if cwd is None:raise SchemaInvalid('RENDER_LIMIT_EXCEEDED','No usable short working directory exists')
    _safe(cwd);_safe(work)
    result=[]
    for path in paths:
        _safe(path)
        if path.parent != work or re.fullmatch(r'panel-[0-9]{3}\.png',path.name) is None:
            raise SchemaInvalid('UNSAFE_RENDER_PATH','Panel is outside the private render work directory')
        result.append(str(path.relative_to(cwd)))
    return cwd,result


def _graph(plan, panel_ids, video_index):
    w,h=plan['geometry']['canvas'];sx,sy=plan['geometry']['source_rect'][:2];x,y=plan['geometry']['panel_rect'][:2]
    base=S.rational_value(plan['proxy_time_base'])
    lines=[f"[0:{video_index}]settb=1/{base.denominator},pad={w}:{h}:{sx}:{sy}:black[base0]"]
    for i,ident in enumerate(panel_ids,1):
        terms=[f"gte(n,{int(s['start_frame'])})*lt(n,{int(s['end_frame_exclusive'])})" for s in plan['spans'] if s['panel_id']==ident]
        lines.append(f"[base{i-1}][{i}:v]overlay=x={x}:y={y}:eof_action=repeat:repeatlast=1:shortest=0:enable='{_balanced_sum(terms)}'[base{i}]")
    lines.append(f"[base{len(panel_ids)}]format=yuv420p[rendered]")
    return ';\n'.join(lines)+'\n'


def render_video(playback_path, plan, output_dir, *, font_path=None, font=None,
                 cancelled=None, progress=None, max_output_bytes=DEFAULT_OUTPUT_BYTES):
    """Return a small verified manifest; never write to a workspace database.

    ``plan`` must come from ``render_layout.prepare_plan`` on the caller's
    store-validated immutable snapshot. The caller owns scheduling/publication.
    A new output directory retains tool evidence on failure; only successful
    validation creates its ``validated`` child. The source is never modified.
    Progress reports stages and actual encoder frame counters, not guessed ETA.
    """
    if cancelled is not None and not callable(cancelled) or progress is not None and not callable(progress):
        raise SchemaInvalid("INVALID_RENDER_CALLBACK", "Render callbacks must be callable")
    if type(max_output_bytes) is not int or not 1024*1024<=max_output_bytes<=8*1024*1024*1024:
        raise SchemaInvalid("INVALID_RENDER_BUDGET", "Output byte budget must be1MiB through8GiB")
    target=Path(output_dir).absolute();source=Path(playback_path).absolute();work=None;stage='preflight';last_progress=0.0;last_space_check=0.0
    frozen=copy.deepcopy(plan)
    try:
        boundaries=_plan(frozen)
        _safe(source);_safe(target,missing=True)
        if os.path.lexists(target):raise Conflict("OUTPUT_EXISTS", "Render output directory must be new")
        if not source.is_file():raise NotFound("NOT_FOUND", "Immutable playback object is unavailable")
        missing=[tool for tool in ('ffmpeg','ffprobe') if not shutil.which(tool)]
        if missing:raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "Render tools are unavailable",details={'tools':missing})
        # Reserve the final accepted-file cap, not an estimated compression
        # ratio. FFmpeg -fs may overshoot: strict size checks still reject it.
        panel_pixels=frozen['geometry']['panel_rect'][2]*frozen['geometry']['panel_rect'][3]
        required=max_output_bytes+MIN_FREE_RESERVE+len(frozen['panels'])*panel_pixels*4
        _space(target.parent,required)
        target.mkdir();work=target/'work';work.mkdir();project=work/'project';project.mkdir()
        _write(project/'render-plan.json',frozen)
        ffmpeg,ffprobe=P._tool('ffmpeg'),P._tool('ffprobe')
        def check():
            nonlocal last_progress,last_space_check
            if cancelled is not None and cancelled():return True
            now=time.monotonic()
            if now-last_space_check>=0.25:
                _space(target,MIN_FREE_RESERVE);last_space_check=now
            if progress is not None and now-last_progress>=0.25:
                value={'stage':stage,'frames_processed':None}
                path=work/'encode-progress.txt'
                if stage=='encode' and path.is_file():
                    with path.open('rb') as stream:
                        stream.seek(max(0,path.stat().st_size-8192));tail=stream.read().decode('utf-8','replace')
                    frames=[line.partition('=')[2] for line in tail.splitlines() if line.startswith('frame=')]
                    if frames and frames[-1].isdigit():value['frames_processed']=int(frames[-1])
                try:progress(value)
                except Exception as exc:raise WorkspaceError('RENDER_CALLBACK_FAILED','Render progress callback failed') from exc
                last_progress=now
            return False
        def tick(value):
            nonlocal stage,last_progress,last_space_check
            stage=value;last_progress=0.0;last_space_check=0.0;P._cancel(check)
        tick('verify_input')
        before=stream_sha256(source,cancelled=check)
        if before!=frozen['binding']['playback_sha256']:
            raise Conflict('MEDIA_CHANGED','Playback object differs from frozen render binding')
        for name,tool in (('ffmpeg',ffmpeg),('ffprobe',ffprobe)):
            P._run([tool,'-version'],work,name+'-version',check,strict=False)
        version=(work/'ffmpeg-version.stdout').read_text('utf-8',errors='replace').splitlines()[0]
        match=re.search(r'ffmpeg version (?:n)?(\d+)[.\-]',version)
        if match is None or int(match[1])<6:
            raise DependencyUnavailable('UNSUPPORTED_RENDER_TOOLCHAIN','Rendering requires an identified FFmpeg6 or newer toolchain')
        graph_option='-/filter_complex' if int(match[1])>=7 else '-filter_complex_script'
        probe=P._probe(ffprobe,source,work,'input-probe',check)
        videos=[s for s in probe['streams'] if s.get('codec_type')=='video' and not s.get('disposition',{}).get('attached_pic')]
        audios=[s for s in probe['streams'] if s.get('codec_type')=='audio']
        if len(videos)!=1 or len(audios)>MAX_AUDIO_STREAMS:raise SchemaInvalid('UNSUPPORTED_RENDER_MEDIA','Exactly one video and at most8 audio tracks are supported')
        video=videos[0];base=P._time_base(video);enc=frozen['recipe']['encoding']
        hdr=video.get('color_transfer') in ('smpte2084','arib-std-b67') or any(any(token in str(s.get('side_data_type','')).lower() for token in ('dovi','mastering display','content light')) for s in video.get('side_data_list',[]))
        if hdr:raise SchemaInvalid('UNSUPPORTED_RENDER_HDR','HDR needs a separately verified color pipeline')
        if video.get('codec_name')!='h264' or video.get('pix_fmt')!='yuv420p' or video.get('color_range')=='pc':
            raise SchemaInvalid('UNSUPPORTED_RENDER_MEDIA','Only the admitted8-bit limited/unknown-range H264 proxy is supported')
        if [video.get('width'),video.get('height')]!=frozen['geometry']['source_rect'][2:] or base!=S.rational_value(frozen['proxy_time_base']):
            raise Conflict('MEDIA_CHANGED','Proxy geometry or clock differs from frozen plan')
        sar=video.get('sample_aspect_ratio')
        if sar not in ('1:1','1/1') or any(Fraction(str(s.get('rotation',0)))%360 for s in video.get('side_data_list',[]) if 'rotation' in s) or Fraction(str(video.get('tags',{}).get('rotate',0)))%360:
            raise SchemaInvalid('UNSUPPORTED_RENDER_GEOMETRY','Proxy must have square pixels and applied rotation')
        if enc['audio_codec']=='copy' and any(s.get('codec_name')!='aac' for s in audios):
            raise SchemaInvalid('UNSUPPORTED_RENDER_AUDIO_COPY','Current MP4 copy verification supports AAC only')
        for audio in audios:
            if not 1<=int(audio.get('channels',0))<=8 or int(audio.get('sample_rate',0)) not in (7350,8000,11025,12000,16000,22050,24000,32000,44100,48000,64000,88200,96000):
                raise SchemaInvalid('UNSUPPORTED_RENDER_AUDIO','Unsupported channel count or sample rate')
        frames=P._frames(ffprobe,source,video['index'],work,'input-video-frames',check)
        summary=P._verified_video_summary(ffprobe,source,video['index'],frames,base,work,'input-video-packets',check)
        if str(summary['count'])!=frozen['frame_count'] or summary['first']*base!=S.rational_value(frozen['video_start']) or (summary['last']+summary['last_duration'])*base!=S.rational_value(frozen['video_end']):
            raise Conflict('MEDIA_CHANGED','Actual video count or active endpoint differs from frozen plan')
        input_audio=[]
        for i,audio in enumerate(audios):
            rows=P._frames(ffprobe,source,audio['index'],work,f'input-audio-{i}',check)
            input_audio.append(P._audio_summary(rows,P._time_base(audio),int(audio['sample_rate']),check))
        tick('panels');face=font or L.FontFace(font_path,frozen['binding']['font_sha256'])
        if face.sha256!=frozen['binding']['font_sha256']:raise Conflict('RENDER_FONT_CHANGED','Font differs from render plan')
        panel_ids=[];panel_paths=[]
        for i,(ident,panel) in enumerate(frozen['panels'].items()):
            P._cancel(check)
            if not panel['entries']:continue # padding already supplies black
            path=work/f'panel-{i:03d}.png';L.render_panel(frozen,ident,font=face).save(path)
            panel_ids.append(ident);panel_paths.append(path)
        graph=work/'filtergraph.txt';graph.write_text(_graph(frozen,panel_ids,video['index']),encoding='utf-8')
        candidate=project/'annotation.mp4'
        # Use paths relative to a validated ancestor only for generated panels.
        # Source, output, filter and verification paths remain absolute, and
        # a child-local cwd avoids process-global chdir and concurrency races.
        encode_cwd,panel_arguments=_panel_arguments(work,panel_paths)
        command=_encode_command(ffmpeg,source,panel_arguments,frozen,video,audios,summary,graph,candidate,max_output_bytes,work/'encode-progress.txt',graph_option)
        tick('encode');P._run(command,work,'encode',check,cwd=encode_cwd)
        _check_output_size(candidate,max_output_bytes)
        tick('verify_result')
        P._run([ffmpeg,'-hide_banner','-v','error','-nostdin','-xerror','-err_detect','explode','-protocol_whitelist','file,pipe','-i',str(candidate),'-map','0:v','-map','0:a?','-fps_mode:v','passthrough','-enc_time_base:v','demux','-f','null','-'],work,'full-decode',check)
        output_probe=P._probe(ffprobe,candidate,work,'output-probe',check)
        outvideo=[s for s in output_probe['streams'] if s.get('codec_type')=='video'];outaudio=[s for s in output_probe['streams'] if s.get('codec_type')=='audio']
        if len(outvideo)!=1 or len(outaudio)!=(0 if enc['audio_codec']=='none' else len(audios)):
            raise WorkspaceError('RENDER_STREAM_MISMATCH','Output stream count differs from explicit configuration')
        changed=outvideo[0];outbase=P._time_base(changed)
        if [changed.get('width'),changed.get('height')]!=frozen['geometry']['canvas'] or changed.get('codec_name')!='h264' or changed.get('pix_fmt')!='yuv420p':
            raise WorkspaceError('RENDER_VIDEO_MISMATCH','Output video differs from configured canvas or codec')
        color={key:{'proxy':video.get(key),'output':changed.get(key),'status':'unknown' if video.get(key) in (None,'unknown','unspecified') else 'preserved'} for key in COLOR_KEYS}
        if any(value['status']=='preserved' and value['proxy']!=value['output'] for value in color.values()):
            raise WorkspaceError('RENDER_COLOR_MISMATCH','Declared proxy color tags changed')
        output_frames=P._frames(ffprobe,candidate,changed['index'],work,'output-video-frames',check)
        outsummary=P._verified_video_summary(ffprobe,candidate,changed['index'],output_frames,outbase,work,'output-video-packets',check)
        output_rows=iter(P._frame_rows(output_frames));count=0
        for original in P._frame_rows(frames):
            P._cancel(check);converted=next(output_rows,None)
            if converted is None or original['pts']*base!=converted['pts']*outbase:
                raise WorkspaceError('RENDER_PTS_MISMATCH','Rendered video altered a displayed frame timestamp')
            count+=1
        if next(output_rows,None) is not None or count!=outsummary['count'] or count!=summary['count'] or summary['duration']!=outsummary['duration']:
            raise WorkspaceError('RENDER_PTS_MISMATCH','Rendered frame count or final hold changed')
        audio_checks=[]
        for i,(original,converted) in enumerate(zip(audios,outaudio)):
            rate=int(original['sample_rate']);changed_rate=int(converted.get('sample_rate',0))
            if converted.get('codec_name')!='aac' or converted.get('channels')!=original.get('channels') or changed_rate!=rate:
                raise WorkspaceError('RENDER_AUDIO_MISMATCH','Rendered audio channels, codec or sample rate changed')
            rows=P._frames(ffprobe,candidate,converted['index'],work,f'output-audio-{i}',check)
            actual=P._audio_summary(rows,P._time_base(converted),rate,check);expected=input_audio[i]
            tolerance=Fraction(0) if enc['audio_codec']=='copy' else Fraction(2048,rate)+Fraction(2,1000)
            if abs(actual['start']-expected['start'])>tolerance or abs(actual['end']-expected['end'])>tolerance or abs(actual['samples']-expected['samples'])>(0 if enc['audio_codec']=='copy' else 2048):
                raise WorkspaceError('RENDER_AUDIO_TIMING_MISMATCH','Rendered audio samples or endpoints exceeded declared tolerance')
            pcm_hashes=[]
            if enc['audio_codec']=='copy':
                for label,path,stream in (('input',source,original),('output',candidate,converted)):
                    raw=P._run([ffmpeg,'-hide_banner','-v','error','-nostdin','-xerror','-protocol_whitelist','file,pipe','-i',str(path),'-map',f"0:{stream['index']}",'-c:a','pcm_s32le','-f','hash','-hash','sha256','-'],work,f'{label}-pcm-{i}',check)
                    pcm_hashes.append(raw.read_text('utf-8').strip())
                if pcm_hashes[0]!=pcm_hashes[1] or not pcm_hashes[0].startswith('SHA256='):
                    raise WorkspaceError('RENDER_AUDIO_COPY_MISMATCH','Copied AAC did not retain identical decoded PCM')
            audio_checks.append({'track':i,'source_stream':original['index'],'output_stream':converted['index'],'sample_rate':rate,'channels':original['channels'],
                                 'input_samples':expected['samples'],'output_samples':actual['samples'],'input_start':S.rational(expected['start']),'input_end':S.rational(expected['end']),
                                 'output_start':S.rational(actual['start']),'output_end':S.rational(actual['end']),'tolerance':S.rational(tolerance),'pcm_sha256':pcm_hashes[0][7:] if pcm_hashes else None})
        tick('verify_panels');x,y,pw,ph=frozen['geometry']['panel_rect'];select=_frame_select_expression(boundaries)
        P._run([ffmpeg,'-hide_banner','-v','error','-nostdin','-n','-xerror','-protocol_whitelist','file,pipe','-i',str(candidate),'-vf',f'select={select},crop={pw}:{ph}:{x}:{y}',
                '-fps_mode:v','passthrough','-c:v','png','-threads:v','1',str(work/'boundary-%06d.png')],work,'panel-boundaries',check)
        panels_cache={};panel_checks=[]
        for i,n in enumerate(boundaries,1):
            P._cancel(check);ident=L.panel_for_frame(frozen,n)
            if ident not in panels_cache:panels_cache[ident]=L.render_panel(frozen,ident,font=face)
            with Image.open(work/f'boundary-{i:06d}.png') as image:
                actual=image.convert('RGB')
                if actual.size!=(pw,ph):raise WorkspaceError('RENDER_PANEL_MISMATCH','Extracted panel has incorrect dimensions')
                error=sum(ImageStat.Stat(ImageChops.difference(actual,panels_cache[ident])).mean)/3
            if error>MAX_PANEL_MAE:raise WorkspaceError('RENDER_PANEL_MISMATCH','Encoded panel differs from full frozen text; reduce CRF if compression is excessive')
            panel_checks.append({'frame_index':str(n),'panel_id':ident,'mae':error})
        if len(list(work.glob('boundary-*.png')))!=len(boundaries):raise WorkspaceError('RENDER_PANEL_MISMATCH','Boundary sample count differs')
        tick('verify_source_unchanged')
        after=stream_sha256(source,cancelled=check)
        if after!=before:raise Conflict('MEDIA_CHANGED','Playback object changed during rendering')
        verification={'kind':'traceannotate.render-verification','version':1,'full_decode_exit':0,'frame_count':str(count),'all_frame_pts_exact':True,'final_hold_exact':True,
                      'variable_frame_rate':summary['variable'],'input_frame_rows_sha256':stream_sha256(frames,cancelled=check),'output_frame_rows_sha256':stream_sha256(output_frames,cancelled=check),
                      'audio':audio_checks,'color':color,'panel_maximum_mae':MAX_PANEL_MAE,'panel_boundaries':panel_checks}
        _write(project/'verification.json',verification)
        audit=[]
        for path in sorted(work.iterdir()):
            P._cancel(check)
            if path.is_file():audit.append({'file':path.name,'size_bytes':str(path.stat().st_size),'sha256':stream_sha256(path,cancelled=check)})
        _write(project/'audit.json',{'kind':'traceannotate.render-audit','version':1,'scope':'Retained local work diagnostics; hashes are descriptive and are not instructions to read arbitrary paths',
                                   'renderer_version':VERSION,'files':audit})
        files=[]
        for path in sorted(project.iterdir()):files.append({'file':path.name,'size_bytes':str(path.stat().st_size),'sha256':stream_sha256(path,cancelled=check)})
        _check_output_size(candidate,max_output_bytes)
        manifest={'kind':'traceannotate.rendered-video','version':1,'status':'verified_not_workspace_published','renderer_version':VERSION,'directory':'validated',
                  'binding':copy.deepcopy(frozen['binding']),'layout_sha256':frozen['content_sha256'],'encoding':copy.deepcopy(enc),'files':files,
                  'output_budget':{'maximum_final_bytes':str(max_output_bytes),'ffmpeg_fs_policy':'early_stop_threshold_may_overshoot; any final overshoot is rejected'},
                  'video':{'file':'annotation.mp4','frame_count':str(count),'time_base':S.rational(outbase),'start':S.rational(outsummary['first']*outbase),'end':S.rational((outsummary['last']+outsummary['last_duration'])*outbase),'canvas':frozen['geometry']['canvas']},
                  'validation':{'full_decode_exit':0,'all_frame_pts_exact':True,'boundary_samples':len(boundaries),'audio_tracks':len(outaudio),'audio_policy':'AAC copy with identical decoded PCM' if enc['audio_codec']=='copy' else 'Lossy AAC re-encoding; no gain filters; timing/sample count verified' if enc['audio_codec']=='aac' else 'Audio intentionally omitted'},
                  'limitations':['Annotation truth, audio listening and color fidelity are not certified by rendering.','H264 video is lossy; complete source geometry is preserved, not byte-identical encoded pixels.','Unknown proxy color tags remain unknown; no BT.709 assignment is invented.','Local work diagnostics remain outside the validated payload; root publisher owns durable artifact references.']}
        manifest['content_sha256']=S.sha256_text(manifest)
        _write(project/'project.json',manifest)
        tick('publish_candidate');P._publish_new(project,target/'validated',check)
        return manifest
    except BaseException as exc:
        resource=_storage_failure_details(exc,work,target)
        mapped=exc
        if resource is not None:
            mapped=StorageFull('STORAGE_FULL','Rendering encountered a storage limit; original diagnostics were retained',retryable=True,details=resource)
        elif isinstance(exc,WorkspaceError):pass
        elif isinstance(exc,InterruptedError):mapped=WorkspaceError('CANCELLED','Rendering was cancelled')
        elif isinstance(exc,FileNotFoundError):mapped=NotFound('NOT_FOUND','A required local render file is unavailable')
        elif isinstance(exc,FileExistsError):mapped=Conflict('OUTPUT_EXISTS','Rendering refuses existing outputs')
        elif isinstance(exc,OSError):mapped=WorkspaceError('RENDER_IO_ERROR','Rendering could not access its local files')
        elif isinstance(exc,(P.MediaPreparationError,ValueError,TypeError,KeyError,OverflowError)):
            mapped=WorkspaceError('RENDER_FAILED','Render validation or local tool execution failed',details={'stage':stage})
        if work is not None:
            failure={'status':'not_validated','stage':stage,'exception':type(exc).__name__,'code':getattr(mapped,'code','RENDER_FAILED'),
                     'original_code':getattr(exc,'code',None),'resource_detection':resource}
            try:_write(target/'failure.json',failure)
            except OSError:pass
        if mapped is exc:raise
        raise mapped from exc
