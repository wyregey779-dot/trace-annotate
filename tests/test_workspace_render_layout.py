"""Exact/VFR selection, literal text and explicit layout rejection."""
from __future__ import annotations

import copy
import hashlib
import unicodedata
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageFont
import pytest

from trace_annotate.workspace import render_layout as R
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import SchemaInvalid


@pytest.fixture(scope='module')
def face():
    for candidate in ('DejaVuSans.ttf','arial.ttf','LiberationSans-Regular.ttf'):
        try:
            loaded=ImageFont.truetype(candidate,16)
            path=Path(loaded.path)
            if path.is_file():return R.FontFace(path,hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:pass
    pytest.skip('A real local Unicode TTF/OTF is needed for Pillow glyph/layout checks')


def inputs(face, position='bottom'):
    rows=[{'frame_index':str(i),'source_pts':str(p),'proxy_pts':str(p),'seconds':S.rational(Fraction(p,30))} for i,p in enumerate((0,1,3,6))]
    media={'id':'med-layout','source':{'sha256':'1'*64,'size_bytes':'100','display_name':'source.mp4'},
           'playback':{'sha256':'2'*64,'size_bytes':'200','display_name':'playback.mp4'},
           'duration':S.rational('3/10'),'video_start':S.rational(0),'video_end':S.rational('3/10'),
           'frame_count':'4','frame_map_artifact':S.make_id('art','3'*64),'source_time_base':S.rational('1/30'),
           'proxy_time_base':S.rational('1/30'),'common_origin':S.rational(0),'source_video_stream':0,
           'source_width':160,'source_height':240,'pixel_aspect_ratio':S.rational(1),'rotation_degrees':0,
           'coordinate_policy':'source_decoded_pixels_with_explicit_transform'}
    annotation={'id':'ann-a','type':'segment','label':'First {literal} \\N','text':'Complete text: -42, 1,234, braces{} and \\slash.',
                'start':{'seconds':S.rational('1/30')},'end':{'seconds':S.rational('1/10')},
                'track':'shots','tags':['one','two'],'review_status':'unreviewed'}
    task={'id':'tsk-layout','revision':5,'content_sha256':'4'*64,'media_id':media['id'],'annotations':[annotation]}
    recipe={'version':1,'position':position,'panel_extent_px':256,'body_font_px':12,'title_font_px':14,
            'padding_px':12,'line_spacing_px':3,'fields':['label','text'],'tracks':['shots'],
            'font_sha256':face.sha256,'font_license':'Local system font used only for raster tests; no font distributed',
            'encoding':{'video_codec':'libx264','pixel_format':'yuv420p','crf':19,'preset':'fast','audio_codec':'aac','audio_bitrate_kbps':192,'color_policy':'preserve_declared_proxy'}}
    return task,media,rows,recipe


def plan(face, position='bottom', mutate=None):
    t,m,rows,r=inputs(face,position)
    if mutate:mutate(t,m,rows,r)
    return R.prepare_plan(t,m,rows,r,font=face,playback_size=(160,240),frame_map_sha256='3'*64)


@pytest.mark.parametrize('position',['top','bottom','left','right'])
def test_four_positions_keep_source_pixels_full_and_literal_text(face,position):
    p=plan(face,position)
    source=Image.new('RGB',(160,240),(12,64,127))
    output=R.composite_frame(p,source,1,font=face)
    x,y,w,h=p['geometry']['source_rect']
    assert output.crop((x,y,x+w,y+h)).tobytes()==source.tobytes()
    entries=p['panels'][R.panel_for_frame(p,1)]['entries']
    for field in ('label','text'):
        assert ''.join(e['text'] for e in entries if e['source_field']==field)==p['selected_annotations'][0][field]
    panel=R.render_panel(p,R.panel_for_frame(p,1),font=face)
    assert panel.getpixel((0,0))==(0,0,0) and panel.getextrema()[0][1]==255
    assert all(e['font_px']==(14 if e['source_field']=='label' else 12) for e in entries)


def test_half_open_selection_uses_real_vfr_pts_not_average_fps(face):
    p=plan(face)
    assert p['panels'][R.panel_for_frame(p,0)]['annotation_ids']==[]
    assert p['panels'][R.panel_for_frame(p,1)]['annotation_ids']==['ann-a']
    assert p['panels'][R.panel_for_frame(p,2)]['annotation_ids']==[]
    assert p['panels'][R.panel_for_frame(p,3)]['annotation_ids']==[]


def test_overlap_event_and_whole_clip_not_silently_dropped(face):
    def add(t,m,rows,r):
        a=copy.deepcopy(t['annotations'][0]);a.update(id='ann-b',label='Second');a['start']={'seconds':S.rational('1/10')};a['end']={'seconds':S.rational('3/10')}
        e={'id':'ann-event','type':'event','label':'Point','text':'event','at':{'seconds':S.rational('1/8')},'track':'shots','tags':[],'review_status':'unreviewed'}
        w={'id':'ann-all','type':'whole_clip','label':'All','text':'whole','track':'shots','tags':[],'review_status':'unreviewed'}
        t['annotations'] += [a,e,w];r['fields']=['label'];r['panel_extent_px']=512
    p=plan(face,mutate=add)
    assert p['panels'][R.panel_for_frame(p,2)]['annotation_ids']==['ann-b','ann-event','ann-all']
    assert p['panels'][R.panel_for_frame(p,3)]['annotation_ids']==['ann-b','ann-all']


def test_track_selection_is_exact_and_other_tracks_remain_unselected(face):
    def add(t,m,rows,r):
        a=copy.deepcopy(t['annotations'][0]);a.update(id='ann-other',track='other',text='Other track')
        t['annotations'].append(a)
    p=plan(face,mutate=add)
    assert [a['id'] for a in p['selected_annotations']]==['ann-a']
    with pytest.raises(SchemaInvalid,match='does not exist'):
        plan(face,mutate=lambda t,m,rows,r:r.update(tracks=['missing']))


def test_layout_is_deterministic_and_source_inputs_remain_unchanged(face):
    t,m,rows,r=inputs(face);before=copy.deepcopy((t,m,rows,r))
    a=R.prepare_plan(t,m,rows,r,font=face,playback_size=(160,240),frame_map_sha256='3'*64)
    b=R.prepare_plan(t,m,rows,r,font=face,playback_size=(160,240),frame_map_sha256='3'*64)
    assert a==b and (t,m,rows,r)==before
    assert a['binding']['task_content_sha256']=='4'*64 and a['binding']['frame_map_sha256']=='3'*64
    c=plan(face,mutate=lambda t,m,rows,r:r['encoding'].update(crf=20))
    assert c['content_sha256']!=a['content_sha256']


@pytest.mark.parametrize('mutation',[
 lambda t,m,rows,r:r.update(body_font_px=0),lambda t,m,rows,r:r.update(unknown=True),
 lambda t,m,rows,r:r.update(fields=['text','text']),lambda t,m,rows,r:r.update(fields=[]),
 lambda t,m,rows,r:r.update(position='overlay'),lambda t,m,rows,r:r.update(panel_extent_px=255),
 lambda t,m,rows,r:r.update(font_sha256='9'*64),lambda t,m,rows,r:r['encoding'].update(video_codec='unknown'),
 lambda t,m,rows,r:r.update(font_license=''),lambda t,m,rows,r:r.update(tracks=[None]),
])
def test_invalid_recipe_is_structured_rejection(face,mutation):
    with pytest.raises(SchemaInvalid):plan(face,mutate=mutation)


@pytest.mark.parametrize('mutation',[
 lambda t,m,rows,r:rows[1].update(frame_index='2'),lambda t,m,rows,r:rows[1].update(proxy_pts='0'),
 lambda t,m,rows,r:rows[1].update(source_pts='banana'),lambda t,m,rows,r:rows.pop(),
 lambda t,m,rows,r:t['annotations'][0].update(end={'seconds':S.rational('1/30')}),
 lambda t,m,rows,r:t['annotations'][0].update(start={'seconds':S.rational('1/20')},end={'seconds':S.rational('3/40')}),
])
def test_invalid_map_or_interval_cannot_fall_back_to_fps(face,mutation):
    with pytest.raises(SchemaInvalid):plan(face,mutate=mutation)


def test_long_or_missing_glyph_text_is_rejected_without_font_shrink(face):
    with pytest.raises(SchemaInvalid,match='does not fit'):
        plan(face,mutate=lambda t,m,rows,r:t['annotations'][0].update(text='Long full text '*300))
    with pytest.raises(SchemaInvalid) as missing:
        plan(face,mutate=lambda t,m,rows,r:t['annotations'][0].update(text='missing\U0010FFFF'))
    assert missing.value.code=='RENDER_MISSING_GLYPH'
    with pytest.raises(SchemaInvalid):R.wrap_text('A',face,24,1)
    assert R.wrap_text('first\nsecond',face,12,200)==['first','second']


def test_combining_marks_remain_attached_and_time_labels_are_readable(face):
    text='e\u0301 complete words e\u0301 complete words e\u0301'
    lines=R.wrap_text(text,face,18,95)
    assert ''.join(lines)==text
    assert all(not line or not unicodedata.combining(line[0]) for line in lines)
    assert R._seconds(S.rational('58366667/1000000'))=='58.367'
    assert R._seconds(S.rational('286/5'))=='57.200'
    assert R._seconds(S.rational('1/2000'))=='0.001'


def test_mutated_plan_font_or_proxy_geometry_is_not_accepted(face,tmp_path):
    p=plan(face);changed=copy.deepcopy(p);changed['recipe']['body_font_px']=13
    with pytest.raises(SchemaInvalid,match='changed'):R.panel_for_frame(changed,1)
    with pytest.raises(SchemaInvalid):R.composite_frame(p,Image.new('RGB',(240,160)),1,font=face)
    font=tmp_path/'font.ttf';font.write_bytes(face.data+b'changed')
    with pytest.raises(SchemaInvalid,match='hash differs'):R.FontFace(font,face.sha256)
    with pytest.raises(SchemaInvalid):R.FontFace(tmp_path/'missing.ttf',face.sha256)
    with pytest.raises(SchemaInvalid):R._cmap(b'not a font')



def test_layout_can_cancel_after_frame_map_before_any_text(face,monkeypatch):
    t,m,original,r=inputs(face)
    stopped=False
    def rows():
        nonlocal stopped
        yield from original
        stopped=True
    def unwanted(*args,**kwargs):
        pytest.fail('Layout continued into text after frame-map cancellation')
    monkeypatch.setattr(R,'_panel_entries',unwanted)
    with pytest.raises(InterruptedError):
        R.prepare_plan(t,m,rows(),r,font=face,playback_size=(160,240),frame_map_sha256='3'*64,
                       cancelled=lambda:stopped)


def test_long_text_wrap_polls_cancellation_before_finishing_all_glyphs(face,monkeypatch):
    original=R._width;calls=0
    def count(font,text):
        nonlocal calls
        calls+=1
        return original(font,text)
    monkeypatch.setattr(R,'_width',count)
    with pytest.raises(InterruptedError):
        R.wrap_text('Long cancellable text '*100,face,12,160,cancelled=lambda:calls>=5)
    assert 5<=calls<200


def test_layout_final_return_checks_cancellation_and_preserves_timeout(face,monkeypatch):
    from trace_annotate.workspace.errors import WorkspaceError
    t,m,rows,r=inputs(face);original=R.S.sha256_text;stopped=False
    def digest(value):
        nonlocal stopped
        result=original(value)
        if isinstance(value,dict) and value.get('kind')=='traceannotate.render-layout':stopped=True
        return result
    def timeout():
        if stopped:raise WorkspaceError('RENDER_TIMEOUT','Controlled deadline at final plan hash')
        return False
    monkeypatch.setattr(R.S,'sha256_text',digest)
    with pytest.raises(WorkspaceError) as caught:
        R.prepare_plan(t,m,rows,r,font=face,playback_size=(160,240),frame_map_sha256='3'*64,cancelled=timeout)
    assert caught.value.code=='RENDER_TIMEOUT'


def test_layout_rejects_noncallable_cancellation(face):
    t,m,rows,r=inputs(face)
    with pytest.raises(SchemaInvalid) as caught:
        R.prepare_plan(t,m,rows,r,font=face,playback_size=(160,240),frame_map_sha256='3'*64,cancelled=True)
    assert caught.value.code=='INVALID_RENDER_CALLBACK'


@pytest.fixture(scope='module')
def bundled_face():
    # The distributed CJK font, not a host-font substitution, reproduced the
    # actual chapter/shot failures seen during the147-second video review.
    path=Path(R.__file__).parent/'fonts/NotoSansCJKsc-Regular.otf'
    return R.FontFace(path,'2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b')


CHAPTER_TEXT='从室内楼梯通道转入舞台、茶具及观众席，章末出现 Day 2 转场与博物馆外景。'
SHOT_TEXT='宽景呈现一列高速列车向站台方向驶近，列车车身、站台顶棚及站台边缘共同入镜；片段作为返程收尾，未确认具体车次。'


def assert_legal_lines(text,lines,face,size,width):
    assert ''.join(lines)==text
    assert all(R._width(face.font(size),line)<=width for line in lines)
    assert all(not line.strip() or line.lstrip()[0] not in R.CLOSING_PUNCTUATION for line in lines)
    assert all(not line.strip() or line.rstrip()[-1] not in R.OPENING_PUNCTUATION for line in lines)
    assert all(not line or not unicodedata.combining(line[0]) for line in lines)


@pytest.mark.parametrize('width',[676,688])
@pytest.mark.parametrize('text',[CHAPTER_TEXT,SHOT_TEXT])
def test_actual_chapter_and_train_text_have_whole_day_and_no_comma_first_line(bundled_face,width,text):
    lines=R.wrap_text(text,bundled_face,28,width)
    assert_legal_lines(text,lines,bundled_face,28,width)
    if text==CHAPTER_TEXT:
        assert any('Day 2' in line for line in lines)
        assert not any(line.endswith('D') or line.startswith('ay') for line in lines)


@pytest.mark.parametrize('text,tokens,width',[
    ('中文前缀 Caption 中文后缀',['Caption'],150),
    ('成本 -42，金额 1,234.56 元，比例 12.5%。',['-42','1,234.56','12.5%'],240),
    ('访问 https://example.com/a?x=1 后返回',['https://example.com/a?x=1'],440),
    ('画面（红色组件），测试【括号】与《标题》。',[],100),
    ('first  second   third',['first','second','third'],130),
    ('e\u0301 中文 e\u0301 and cafe\u0301',['e\u0301','cafe\u0301'],120),
])
def test_word_numeric_url_and_punctuation_guards_preserve_exact_text(bundled_face,text,tokens,width):
    lines=R.wrap_text(text,bundled_face,24,width)
    assert_legal_lines(text,lines,bundled_face,24,width)
    for token in tokens:
        assert R._width(bundled_face.font(24),token)<=width
        assert any(token in line for line in lines)


def test_overwide_url_can_emergency_break_without_loss_or_font_change(bundled_face):
    text='https://example.com/very-long-path?item=1234&offset=-42'
    lines=R.wrap_text(text,bundled_face,24,100)
    assert len(lines)>2
    assert_legal_lines(text,lines,bundled_face,24,100)


def test_authored_paragraphs_and_spaces_survive_without_rebalancing(bundled_face):
    text='第一段 Day 2。\n\n  第二段 -42 与 1,234.56。  '
    expected=[]
    for paragraph in text.split('\n'):
        lines=R.wrap_text(paragraph,bundled_face,24,200)
        assert_legal_lines(paragraph,lines,bundled_face,24,200)
        expected.extend(lines)
    assert R.wrap_text(text,bundled_face,24,200)==expected


@pytest.mark.parametrize('text,width',[
    ('中，',28),('（中',28),('，中',100),('中（',100),
])
def test_unrepresentable_punctuation_group_is_explicit_not_looped_or_shrunk(bundled_face,text,width):
    with pytest.raises(SchemaInvalid) as caught:
        R.wrap_text(text,bundled_face,28,width)
    assert caught.value.code=='RENDER_LINE_BREAK_UNREPRESENTABLE'


def test_fitting_ascii_word_is_not_split_to_hide_punctuation_overflow(bundled_face):
    width=R._width(bundled_face.font(28),'Caption')
    with pytest.raises(SchemaInvalid) as caught:
        R.wrap_text('Caption，',bundled_face,28,width)
    assert caught.value.code=='RENDER_LINE_BREAK_UNREPRESENTABLE'


def test_explicit_layout_generation_changes_with_word_boundary_behavior():
    assert R.VERSION=='annotation-panel-v2-word-boundaries'
