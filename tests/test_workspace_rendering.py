"""Real CFR/VFR encoding, exact panels and failure/cancellation barriers."""
from __future__ import annotations

import copy
import errno
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import ImageFont
import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.temporal.media import stream_sha256
from trace_annotate.workspace import render_layout as L, rendering as R, schema as S
from trace_annotate.workspace.errors import WorkspaceError


@pytest.fixture(scope='module')
def face():
    for candidate in ('DejaVuSans.ttf','arial.ttf','LiberationSans-Regular.ttf'):
        try:
            loaded=ImageFont.truetype(candidate,16);path=Path(loaded.path)
            if path.is_file():return L.FontFace(path,hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:pass
    pytest.skip('A local Unicode TTF/OTF is needed; no test font is distributed')


@pytest.fixture(scope='module')
def clips(tmp_path_factory):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):pytest.skip('Real FFmpeg/ffprobe are required')
    root=tmp_path_factory.mktemp('real-render-inputs');result={}
    for kind in ('cfr','vfr'):
        work=root/kind;work.mkdir();source=work/'source.mp4'
        expression='N' if kind=='cfr' else 'if(lt(N,2),N,if(lt(N,4),2+(N-2)*2,6+(N-4)*3))'
        duration='0.6' if kind=='cfr' else '1.0';last=500 if kind=='cfr' else 900
        command=['ffmpeg','-hide_banner','-v','error','-nostdin','-n','-f','lavfi','-i','testsrc2=size=160x120:rate=10:duration=0.6',
                 '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration='+duration,
                 '-vf','setpts='+expression.replace(',','\\,'),'-c:v','libx264','-bf','0','-pix_fmt','yuv420p','-preset','veryfast','-crf','18',
                 '-fps_mode:v','passthrough','-enc_time_base:v','1/1000','-video_track_timescale','1000',
                 '-bsf:v',f'setts=pts=PTS:dts=DTS:duration=if(eq(PTS\\,{last})\\,100\\,DURATION)',
                 '-c:a','aac','-b:a','96k',str(source)]
        P._run(command,work,'generate',None)
        probe=P._probe('ffprobe',source,work,'probe',None);video=next(s for s in probe['streams'] if s['codec_type']=='video');base=P._time_base(video)
        raw=P._frames('ffprobe',source,video['index'],work,'frames',None);summary=P._verified_video_summary('ffprobe',source,video['index'],raw,base,work,'packets',None)
        rows=[{'frame_index':str(i),'source_pts':str(row['pts']),'proxy_pts':str(row['pts']),'seconds':S.rational(row['pts']*base)} for i,row in enumerate(P._frame_rows(raw))]
        assert summary['count']==6 and summary['variable']==(kind=='vfr')
        sha=stream_sha256(source);end=(summary['last']+summary['last_duration'])*base
        media={'id':'med-render','source':{'sha256':sha,'size_bytes':str(source.stat().st_size),'display_name':'source.mp4'},
               'playback':{'sha256':sha,'size_bytes':str(source.stat().st_size),'display_name':'playback.mp4'},'duration':S.rational(end),
               'video_start':S.rational(summary['first']*base),'video_end':S.rational(end),'frame_count':str(summary['count']),
               'frame_map_artifact':S.make_id('art','3'*64),'source_time_base':S.rational(base),'proxy_time_base':S.rational(base),'common_origin':S.rational(0),
               'source_video_stream':video['index'],'source_width':160,'source_height':120,'pixel_aspect_ratio':S.rational(1),'rotation_degrees':0,
               'coordinate_policy':'source_decoded_pixels_with_explicit_transform'}
        boundary=S.rational_value(rows[2]['seconds'])
        def segment(ident,label,a,b):return {'id':ident,'type':'segment','label':label,'text':'Full '+label,'start':{'seconds':S.rational(a)},'end':{'seconds':S.rational(b)},'track':'shots','tags':[],'review_status':'unreviewed'}
        annotations=[segment('ann-first','First',0,boundary),segment('ann-last','Last',boundary,end),
                     {'id':'ann-point','type':'event','label':'Event','text':'Exact event','at':{'seconds':S.rational(boundary)},'track':'shots','tags':[],'review_status':'unreviewed'},
                     {'id':'ann-all','type':'whole_clip','label':'Whole','text':'All frames','track':'shots','tags':[],'review_status':'unreviewed'}]
        task={'id':'tsk-render','revision':2,'media_id':media['id'],'content_sha256':'4'*64,'annotations':annotations}
        result[kind]=(source,task,media,rows)
    return result


def make_plan(face,clip,position='bottom',audio='copy'):
    source,task,media,rows=clip
    recipe={'version':1,'position':position,'panel_extent_px':256,'body_font_px':12,'title_font_px':14,'padding_px':8,'line_spacing_px':0,
            'fields':['label','text'],'tracks':['shots'],'font_sha256':face.sha256,'font_license':'Local test font, not distributed',
            'encoding':{'video_codec':'libx264','pixel_format':'yuv420p','crf':18,'preset':'veryfast','audio_codec':audio,'audio_bitrate_kbps':96,'color_policy':'preserve_declared_proxy'}}
    return L.prepare_plan(task,media,rows,recipe,font=face,playback_size=(160,120),frame_map_sha256='3'*64)


@pytest.mark.parametrize('kind',['cfr','vfr'])
@pytest.mark.parametrize('position',['top','bottom','left','right'])
def test_real_four_positions_cfr_vfr_all_frames_audio_and_boundaries(face,clips,tmp_path,kind,position):
    clip=clips[kind];plan=make_plan(face,clip,position)
    manifest=R.render_video(clip[0],plan,tmp_path/'render',font=face,max_output_bytes=8*1024*1024)
    output=tmp_path/'render/validated'
    assert manifest['status']=='verified_not_workspace_published'
    assert manifest['video']['frame_count']=='6' and manifest['validation']['all_frame_pts_exact']
    verification=json.loads((output/'verification.json').read_text('utf-8'))
    assert verification['variable_frame_rate']==(kind=='vfr') and verification['all_frame_pts_exact'] and verification['final_hold_exact']
    assert verification['audio'][0]['input_samples']==verification['audio'][0]['output_samples']
    assert verification['audio'][0]['pcm_sha256']
    assert {row['frame_index'] for row in verification['panel_boundaries']}=={'0','1','2','3','5'}
    assert max(row['mae'] for row in verification['panel_boundaries'])<R.MAX_PANEL_MAE
    for file in manifest['files']:
        path=output/file['file'];assert path.stat().st_size==int(file['size_bytes']) and stream_sha256(path)==file['sha256']
    assert S.sha256_text({k:v for k,v in manifest.items() if k!='content_sha256'})==manifest['content_sha256']


@pytest.mark.parametrize('audio',['aac','none'])
def test_explicit_audio_transcode_or_omission_is_disclosed(face,clips,tmp_path,audio):
    plan=make_plan(face,clips['vfr'],audio=audio)
    manifest=R.render_video(clips['vfr'][0],plan,tmp_path/'render',font=face,max_output_bytes=8*1024*1024)
    assert manifest['validation']['audio_tracks']==(0 if audio=='none' else 1)
    assert ('intentionally omitted' if audio=='none' else 'Lossy AAC') in manifest['validation']['audio_policy']


def test_cancelled_real_owned_encoder_has_no_validated_result_and_leaves_other_process(face,clips,tmp_path):
    baseline={p['pid'] for p in P.live_media_processes()}
    unrelated=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def cancelled():return any(p['name']=='encode' and p['pid'] not in baseline for p in P.live_media_processes())
    try:
        with pytest.raises(WorkspaceError) as error:R.render_video(clips['vfr'][0],make_plan(face,clips['vfr']),tmp_path/'cancel',font=face,cancelled=cancelled,max_output_bytes=8*1024*1024)
        assert error.value.code=='CANCELLED' and not (tmp_path/'cancel/validated').exists()
        assert unrelated.poll() is None
        commands=[json.loads(line) for line in (tmp_path/'cancel/work/commands.jsonl').read_text('utf-8').splitlines()]
        encoded=next(row for row in commands if row['name']=='encode')
        assert encoded['cancelled_or_interrupted'] and encoded['exit_code'] is not None
        assert encoded['pid'] not in {p['pid'] for p in P.live_media_processes()}
    finally:
        unrelated.terminate();unrelated.wait(timeout=5)


def test_real_ffmpeg_error_has_no_validated_result(face,clips,tmp_path,monkeypatch):
    run=P._run
    def broken(command,directory,name,cancelled,**kwargs):
        if name=='encode':command=command[:-1]+['-traceannotate_invalid_option',command[-1]]
        return run(command,directory,name,cancelled,**kwargs)
    monkeypatch.setattr(P,'_run',broken)
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'failed',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='RENDER_FAILED' and not (tmp_path/'failed/validated').exists()
    assert (tmp_path/'failed/work/encode.stderr').stat().st_size>0


def test_changed_source_after_encode_is_rejected_before_publication(face,clips,tmp_path):
    source=tmp_path/'mutable-test-source.mp4';shutil.copyfile(clips['cfr'][0],source)
    changed=False
    def progress(value):
        nonlocal changed
        if value['stage']=='verify_result' and not changed:
            with source.open('ab') as stream:stream.write(b'fault-injected-after-encode')
            changed=True
    with pytest.raises(WorkspaceError) as error:R.render_video(source,make_plan(face,clips['cfr']),tmp_path/'changed',font=face,progress=progress,max_output_bytes=8*1024*1024)
    assert changed and error.value.code=='MEDIA_CHANGED' and not (tmp_path/'changed/validated').exists()


def test_space_existing_outputs_bad_font_and_bad_hash_are_not_accepted(face,clips,tmp_path,monkeypatch):
    clip=clips['cfr'];plan=make_plan(face,clip)
    existing=tmp_path/'existing';existing.mkdir();canary=existing/'user.txt';canary.write_text('preserve','utf-8')
    with pytest.raises(WorkspaceError) as error:R.render_video(clip[0],plan,existing,font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='OUTPUT_EXISTS' and canary.read_text('utf-8')=='preserve'
    changed=copy.deepcopy(plan);changed['binding']['playback_sha256']='8'*64;changed['content_sha256']=S.sha256_text({k:v for k,v in changed.items() if k!='content_sha256'})
    with pytest.raises(WorkspaceError) as error:R.render_video(clip[0],changed,tmp_path/'hash',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='MEDIA_CHANGED' and not (tmp_path/'hash/validated').exists()
    monkeypatch.setattr(R.shutil,'disk_usage',lambda path:shutil._ntuple_diskusage(0,0,0))
    with pytest.raises(WorkspaceError) as error:R.render_video(clip[0],plan,tmp_path/'space',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='STORAGE_FULL' and not (tmp_path/'space').exists()


def test_panel_count_budget_and_hdr_are_explicit_refusals(face,clips,tmp_path,monkeypatch):
    plan=make_plan(face,clips['cfr']);first=next(iter(plan['panels'].values()))
    plan['panels']={f'panel-{i}':copy.deepcopy(first) for i in range(R.MAX_UNIQUE_PANELS+1)}
    plan['content_sha256']=S.sha256_text({k:v for k,v in plan.items() if k!='content_sha256'})
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],plan,tmp_path/'panels',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='RENDER_LIMIT_EXCEEDED' and not (tmp_path/'panels').exists()
    probe=P._probe
    def hdr(*args,**kwargs):
        result=probe(*args,**kwargs)
        if args[3]=='input-probe':next(s for s in result['streams'] if s['codec_type']=='video')['color_transfer']='smpte2084'
        return result
    monkeypatch.setattr(P,'_probe',hdr)
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'hdr',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='UNSUPPORTED_RENDER_HDR' and not (tmp_path/'hdr/validated').exists()


def test_cancellation_at_publication_barrier_has_no_validated_payload(face,clips,tmp_path):
    stopped=False
    def progress(value):
        nonlocal stopped
        if value['stage']=='publish_candidate':stopped=True
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'cancel-publish',font=face,
                                                               progress=progress,cancelled=lambda:stopped,max_output_bytes=8*1024*1024)
    assert error.value.code=='CANCELLED' and stopped and not (tmp_path/'cancel-publish/validated').exists()


def test_actual_nonzero_video_audio_timeline_and_all_audio_tracks_survive(face,clips,tmp_path):
    work=tmp_path/'offset-input';work.mkdir();source=work/'offset.mp4'
    P._run(['ffmpeg','-hide_banner','-v','error','-nostdin','-n','-copyts','-itsoffset','0.25','-i',str(clips['cfr'][0]),
            '-map','0:v','-map','0:a','-map','0:a','-c','copy','-avoid_negative_ts','disabled',str(source)],work,'make-offset',None)
    probe=P._probe('ffprobe',source,work,'probe',None);video=next(s for s in probe['streams'] if s['codec_type']=='video');base=P._time_base(video)
    raw=P._frames('ffprobe',source,video['index'],work,'frames',None);summary=P._verified_video_summary('ffprobe',source,video['index'],raw,base,work,'packets',None)
    rows=[{'frame_index':str(i),'source_pts':str(row['pts']),'proxy_pts':str(row['pts']),'seconds':S.rational(row['pts']*base)} for i,row in enumerate(P._frame_rows(raw))]
    start=summary['first']*base;end=(summary['last']+summary['last_duration'])*base;assert start>0
    media=copy.deepcopy(clips['cfr'][2]);sha=stream_sha256(source)
    media.update(source={'sha256':sha,'size_bytes':str(source.stat().st_size),'display_name':'offset.mp4'},playback={'sha256':sha,'size_bytes':str(source.stat().st_size),'display_name':'offset.mp4'},
                 video_start=S.rational(start),video_end=S.rational(end),duration=S.rational(end),source_time_base=S.rational(base),proxy_time_base=S.rational(base))
    task=copy.deepcopy(clips['cfr'][1]);task['annotations']=[{'id':'ann-all','type':'whole_clip','label':'All','text':'Nonzero timeline','track':'shots','tags':[],'review_status':'unreviewed'}]
    plan=make_plan(face,(source,task,media,rows))
    result=R.render_video(source,plan,tmp_path/'offset-render',font=face,max_output_bytes=8*1024*1024)
    assert S.rational_value(result['video']['start'])==start and result['validation']['audio_tracks']==2
    verification=json.loads((tmp_path/'offset-render/validated/verification.json').read_text('utf-8'))
    assert len(verification['audio'])==2 and all(row['pcm_sha256'] for row in verification['audio'])


def test_strict_final_output_cap_rejects_even_one_excess_byte(tmp_path):
    path=tmp_path/'candidate.bin';cap=1024*1024
    with path.open('wb') as stream:stream.truncate(cap)
    assert R._check_output_size(path,cap)==cap
    with path.open('r+b') as stream:stream.truncate(cap+1)
    with pytest.raises(WorkspaceError) as error:R._check_output_size(path,cap)
    assert error.value.code=='RENDER_OUTPUT_LIMIT' and error.value.details['actual_bytes']==str(cap+1)


@pytest.mark.parametrize('exception,expected',[(OSError(errno.ENOSPC,'非英文写入错误'),'os_errno'),(OSError(getattr(errno,'EDQUOT',errno.ENOSPC),'配额'),'os_errno'),(OSError(errno.EIO,'input/output'),None)])
def test_storage_classification_uses_structured_errno_without_tools(tmp_path,exception,expected):
    result=R._storage_failure_details(exception,None,tmp_path)
    assert (None if result is None else result['resource_detection'])==expected


@pytest.mark.parametrize('diagnostic,expected',[
    ('OSError: [Errno 28] 本次注入的非英文错误\n','tool_reported_errno'),
    ('Error number -28\n','tool_reported_errno'),
    ('Error writing trailer: No space left on device\n','tool_reported_space_message'),
    ('ENOSPC: fault-injection\n','tool_reported_space_marker'),
    ('Error opening input file C:/media/No space left on device.mp4\n',None),
    ('Invalid option, decoder failure\n',None),
])
def test_storage_classification_reads_controlled_diagnostic_conservatively(tmp_path,monkeypatch,diagnostic,expected):
    (tmp_path/'encode.stderr').write_text(diagnostic,'utf-8')
    (tmp_path/'commands.jsonl').write_text(json.dumps({'name':'encode','stderr_file':'encode.stderr','exit_code':1})+'\n','utf-8')
    monkeypatch.setattr(R.shutil,'disk_usage',lambda path:shutil._ntuple_diskusage(10**12,0,10**12))
    result=R._storage_failure_details(P.MediaPreparationError('encode failed'),tmp_path,tmp_path)
    assert (None if result is None else result['resource_detection'])==expected
    if result:assert result['kernel_errno_independently_observed'] is False


def test_runtime_reserve_loss_stops_actual_owned_encoder(face,clips,tmp_path,monkeypatch):
    original_run=P._run;armed=False;started=time.monotonic()
    def run(command,directory,name,cancelled,**kwargs):
        nonlocal armed
        if name=='encode':
            # Loop only the injected test input: a six-frame clip fits inside
            # FFmpeg's realtime initial burst and cannot exercise a250ms poll.
            armed=True;index=command.index('-i');command=command[:index]+['-re','-stream_loop','-1']+command[index:]
        return original_run(command,directory,name,cancelled,**kwargs)
    def free(path):
        low=armed and any(p['name']=='encode' for p in P.live_media_processes())
        return shutil._ntuple_diskusage(10**12,0,R.MIN_FREE_RESERVE-1 if low else 10**12)
    monkeypatch.setattr(P,'_run',run);monkeypatch.setattr(R.shutil,'disk_usage',free)
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'runtime-space',font=face,
                                                               cancelled=lambda:time.monotonic()-started>5,max_output_bytes=8*1024*1024)
    assert error.value.code=='STORAGE_FULL' and not (tmp_path/'runtime-space/validated').exists()
    failure=json.loads((tmp_path/'runtime-space/failure.json').read_text('utf-8'));assert failure['code']=='STORAGE_FULL'
    commands=[json.loads(line) for line in (tmp_path/'runtime-space/work/commands.jsonl').read_text('utf-8').splitlines()]
    encoded=next(row for row in commands if row['name']=='encode')
    assert encoded['cancelled_or_interrupted'] and encoded['exit_code'] is not None


def test_post_encode_structured_write_enospc_has_storage_full_receipt(face,clips,tmp_path,monkeypatch):
    original_write=R._write
    def full(path,value):
        if path.name=='verification.json':raise OSError(errno.ENOSPC,'非英文写入错误注入')
        return original_write(path,value)
    monkeypatch.setattr(R,'_write',full)
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'write-full',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='STORAGE_FULL' and error.value.details['resource_detection']=='os_errno'
    assert not (tmp_path/'write-full/validated').exists()
    failure=json.loads((tmp_path/'write-full/failure.json').read_text('utf-8'))
    assert failure['exception']=='OSError' and failure['code']=='STORAGE_FULL'


def test_controlled_child_enospc_diagnostic_maps_without_requiring_english(face,clips,tmp_path,monkeypatch):
    original_run=P._run
    def injected(command,directory,name,cancelled,**kwargs):
        if name=='encode':command=[sys.executable,'-B','-c',"import os,sys; os.write(2, 'OSError: [Errno 28] 故障注入\\n'.encode('utf-8')); sys.exit(1)"]
        return original_run(command,directory,name,cancelled,**kwargs)
    monkeypatch.setattr(P,'_run',injected)
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'tool-full',font=face,max_output_bytes=8*1024*1024)
    assert error.value.code=='STORAGE_FULL' and error.value.details['resource_detection']=='tool_reported_errno'
    assert error.value.details['kernel_errno_independently_observed'] is False
    assert not (tmp_path/'tool-full/validated').exists()
    assert '[Errno 28]' in (tmp_path/'tool-full/work/encode.stderr').read_text('utf-8')


def test_actual_encoded_candidate_one_byte_over_cap_is_not_validated(face,clips,tmp_path,monkeypatch):
    original_run=P._run;cap=1024*1024
    def overshoot(command,directory,name,cancelled,**kwargs):
        result=original_run(command,directory,name,cancelled,**kwargs)
        if name=='encode':
            # Fault injection models -fs/muxer overshoot after the real encode;
            # a single excess byte must fail before further validation.
            with Path(command[-1]).open('ab') as stream:stream.truncate(cap+1)
        return result
    monkeypatch.setattr(P,'_run',overshoot)
    with pytest.raises(WorkspaceError) as error:R.render_video(clips['cfr'][0],make_plan(face,clips['cfr']),tmp_path/'overshoot',font=face,max_output_bytes=cap)
    assert error.value.code=='RENDER_OUTPUT_LIMIT' and error.value.details['actual_bytes']==str(cap+1)
    assert not (tmp_path/'overshoot/validated').exists()
    assert (tmp_path/'overshoot/work/project/annotation.mp4').stat().st_size==cap+1


@pytest.mark.parametrize('boundaries',[
    None,[],[True],[0.0],['0'],[-1],[L.MAX_FRAME_COUNT],[1,0],[0,0],
    list(range(R.MAX_BOUNDARY_SAMPLES+1)),
])
def test_boundary_expression_refuses_lossy_unordered_or_over_budget_indices(boundaries):
    with pytest.raises(WorkspaceError) as error:
        R._frame_select_expression(boundaries)
    assert error.value.code=='INVALID_RENDER_PLAN'


@pytest.mark.parametrize('terms',[None,[],[''],[None],[True],['x'*129],['1']*(R.MAX_BOUNDARY_SAMPLES+1)])
def test_balanced_filter_sum_refuses_invalid_or_unbounded_terms(terms):
    with pytest.raises(WorkspaceError) as error:
        R._balanced_sum(terms)
    assert error.value.code=='INVALID_RENDER_PLAN'


@pytest.mark.parametrize('count',[1,2,3,156,R.MAX_BOUNDARY_SAMPLES])
def test_boundary_expression_retains_every_term_with_logarithmic_depth(count):
    import math
    import re
    boundaries=list(range(count))
    expression=R._frame_select_expression(boundaries)
    selected=[int(value) for value in re.findall(r'eq\(n\\,([0-9]+)\)',expression)]
    assert selected==boundaries
    depth=maximum=0
    for char in expression:
        if char=='(':depth+=1;maximum=max(maximum,depth)
        elif char==')':depth-=1
        assert depth>=0
    assert depth==0 and maximum<=math.ceil(math.log2(count))+1
    assert expression.count('+')==count-1
    assert R.MAX_BOUNDARY_SAMPLES==2*R.MAX_SPANS


def test_maximum_span_plan_keeps_both_sides_of_all_boundaries(face,clips):
    plan=make_plan(face,clips['cfr'])
    ident=next(iter(plan['panels']))
    plan['frame_count']=str(2*R.MAX_SPANS)
    plan['spans']=[{'start_frame':str(2*i),'end_frame_exclusive':str(2*i+2),'panel_id':ident}
                   for i in range(R.MAX_SPANS)]
    plan['content_sha256']=S.sha256_text({k:v for k,v in plan.items() if k!='content_sha256'})
    boundaries=R._plan(plan)
    assert boundaries==list(range(R.MAX_BOUNDARY_SAMPLES))
    graph=R._graph(plan,[ident],0)
    assert graph.count('gte(n,')==R.MAX_SPANS
    # The same repeated-panel enable expression also uses balanced addition;
    # a dense track must not merely move the old parser failure to encoding.
    assert "enable='(" in graph and ')+' in graph


@pytest.mark.parametrize('selected_count',[156,R.MAX_BOUNDARY_SAMPLES])
def test_actual_ffmpeg_selects_all_dense_boundaries_with_exact_indices(tmp_path,selected_count):
    if not shutil.which('ffmpeg'):pytest.skip('Real FFmpeg is required for expression parser regression')
    boundaries=list(range(0,2*selected_count,2))
    expression=R._frame_select_expression(boundaries)
    # The source has an actual timestamp for each frame. Framemd5 retains
    # output PTS, so a parser success or matching count alone cannot hide a
    # dropped, duplicate or wrong selected frame. Only a tiny16x16 source is
    # needed to reach the real parser's previous156-term failure.
    raw=P._run(['ffmpeg','-hide_banner','-v','error','-nostdin','-xerror','-f','lavfi',
        '-i',f'testsrc2=size=16x16:rate=30:duration={2*selected_count/30}',
        '-vf','select='+expression,'-fps_mode:v','passthrough','-enc_time_base:v','1/30',
        '-c:v','rawvideo','-threads:v','1','-f','framemd5','-'],tmp_path,'dense-boundary-selector',None)
    lines=raw.read_text('utf-8').splitlines()
    records=[line.split(',') for line in lines if line and not line.startswith('#')]
    assert len(records)==selected_count
    assert [int(row[2]) for row in records]==boundaries
    assert all(int(row[3])==1 and int(row[4])==384 for row in records)
    assert '#tb 0: 1/30' in lines
    commands=[json.loads(line) for line in (tmp_path/'commands.jsonl').read_text('utf-8').splitlines()]
    assert commands[-1]['exit_code']==0
    assert (tmp_path/'dense-boundary-selector.stderr').is_file()
