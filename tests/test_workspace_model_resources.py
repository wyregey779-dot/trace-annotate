"""Admission, owned descendants and historical model-job compatibility."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from trace_annotate.workspace import model_worker as M
from trace_annotate.workspace import resource_budget as R
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace import vision_jobs as V
from tests.test_workspace_vision_jobs import clip, store, task, submit


def settings():
    return dict(R.policy(), wait_seconds=.1, frame_seconds=.1)


def test_memory_measurement_is_real_positive_os_value():
    assert R.memory_available() > 0


def test_low_memory_never_launches_a_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'memory_available', lambda: 1)
    monkeypatch.setattr(R.subprocess, 'Popen', lambda *a, **k: pytest.fail('Worker must not start'))
    with pytest.raises(WorkspaceError) as error:
        R.run_owned([sys.executable, '-c', 'pass'], tmp_path/'execution',
                    settings=settings(), cancelled=lambda:False, timeout=5)
    assert error.value.code == 'INSUFFICIENT_MEMORY' and error.value.retryable
    receipt=json.loads((tmp_path/'execution/process-receipt.json').read_text('utf-8'))
    assert 'wrapper_pid' not in receipt


def test_unreadable_memory_is_explicit_retryable(tmp_path,monkeypatch):
    def unavailable():
        raise OSError('OS query failed')
    monkeypatch.setattr(R,'memory_available',unavailable)
    with pytest.raises(WorkspaceError) as error:
        with R.admit(settings(),cancelled=lambda:False):
            pytest.fail('Must not admit')
    assert error.value.code=='MEMORY_STATUS_UNAVAILABLE' and error.value.retryable


def test_cancel_wait_is_bounded(monkeypatch):
    monkeypatch.setattr(R,'memory_available',lambda:1)
    started=time.monotonic()
    with pytest.raises(WorkspaceError) as error:
        with R.admit(settings(),cancelled=lambda:time.monotonic()-started>.05):
            pytest.fail('Must not admit')
    assert error.value.code=='CANCELLED' and time.monotonic()-started<1


def test_cross_workspace_model_gate_is_process_wide(monkeypatch):
    monkeypatch.setattr(R,'memory_available',lambda:10*R.GIB)
    entered=threading.Event()
    leave=threading.Event()
    def first():
        with R.admit(settings(),cancelled=lambda:False):
            entered.set()
            leave.wait(2)
    thread=threading.Thread(target=first)
    thread.start()
    assert entered.wait(1)
    try:
        with pytest.raises(WorkspaceError) as error:
            with R.admit(settings(),cancelled=lambda:False):
                pytest.fail('Two workspaces cannot overlap model phases')
        assert error.value.code=='MODEL_RESOURCE_BUSY'
    finally:
        leave.set()
        thread.join(2)
    with R.admit(settings(),cancelled=lambda:False):
        pass


def worker_command(script,marker):
    package=str(Path(__file__).parents[1])
    bootstrap=("import os,sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
               "from trace_annotate.workspace.resource_budget import join_owned_job;"
               "join_owned_job(os.environ.get('TRACE_ANNOTATE_OWNED_MODEL_JOB',''));"
               "runpy.run_path(sys.argv.pop(1),run_name='__main__')")
    return [sys.executable,'-I','-B','-c',bootstrap,package,str(script),str(marker)]


def alive(pid):
    if os.name=='nt':
        import ctypes
        kernel=R._windows_api()
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        kernel.OpenProcess.restype=ctypes.c_void_p
        process=kernel.OpenProcess(0x1000,False,pid)
        if not process:
            return False
        code=ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(process),ctypes.byref(code))) and code.value==259
        finally:
            kernel.CloseHandle(process)
    try:
        if sys.platform.startswith('linux'):
            try:
                status=Path('/proc',str(pid),'stat').read_text('ascii').rsplit(')',1)[1].split()[0]
            except (OSError,IndexError):
                status=None
            if status=='Z':
                return False
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize('mode',['cancel','timeout','low_memory'])
def test_owned_real_worker_and_grandchild_are_reaped(tmp_path,monkeypatch,mode):
    marker=tmp_path/'pids.json'
    script=tmp_path/'tree.py'
    script.write_text("import json,os,subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(60)'])\n"
        "open(sys.argv[1],'w').write(json.dumps([os.getpid(),p.pid]))\n"
        "time.sleep(60)\n",encoding='utf-8')
    monkeypatch.setattr(R,'memory_available',lambda:1 if mode=='low_memory' and marker.exists() else 10*R.GIB)
    with pytest.raises(WorkspaceError) as error:
        R.run_owned(worker_command(script,marker),tmp_path/'run',settings=dict(settings(),wait_seconds=1),
                    cancelled=lambda:mode=='cancel' and marker.exists(),timeout=1)
    assert error.value.code=={'cancel':'CANCELLED','timeout':'MODEL_TIMEOUT','low_memory':'INSUFFICIENT_MEMORY'}[mode]
    assert marker.exists()
    pids=json.loads(marker.read_text())
    deadline=time.monotonic()+3
    while any(alive(pid) for pid in pids) and time.monotonic()<deadline:
        time.sleep(.05)
    assert not any(alive(pid) for pid in pids)
    receipt=json.loads((tmp_path/'run/process-receipt.json').read_text('utf-8'))
    assert receipt['wrapper_pid'] and receipt['actual_exit_code'] is not None


def test_missing_optional_model_keeps_not_run_without_worker(tmp_path,monkeypatch):
    monkeypatch.setenv('TRACE_ANNOTATE_CLIP_CACHE',str(tmp_path/'empty-clip'))
    monkeypatch.setenv('TRACE_ANNOTATE_VLM_CACHE',str(tmp_path/'empty-vlm'))
    config=M.configuration()
    monkeypatch.setattr(R,'run_owned',lambda *a,**k:pytest.fail('Absent weights require no process'))
    result=M.run_phase('clip',config,[{'frame_index':'0','path':'unused','sha256':'a'*64}],tmp_path,cancelled=lambda:False)
    assert result['results'][0]['result']['status']=='not_run'


def test_invalid_resource_policy_cannot_silently_lower_reserve(monkeypatch):
    monkeypatch.setenv('TRACE_ANNOTATE_MODEL_RESERVE_MIB','1')
    with pytest.raises(WorkspaceError) as error:
        R.policy()
    assert error.value.code=='INVALID_MODEL_RESOURCE_POLICY'


def fake_phase(module,config,images,work,*,cancelled):
    return {'identity':M._empty_identity(module,config),'available':False,
            'results':[{'result':{'status':'not_run','reason':'deterministic missing model'}} for image in images]}


def test_normal_pipeline_uses_separate_phases_and_releases_models(store,monkeypatch):
    calls=[]
    def phase(module,*args,**kwargs):
        calls.append(module)
        return fake_phase(module,*args,**kwargs)
    def forbidden(*args,**kwargs):
        pytest.fail('Parent must not construct heavy adapters')
    monkeypatch.setattr(M,'run_phase',phase)
    monkeypatch.setattr(V,'CLIPExtractor',forbidden)
    monkeypatch.setattr(V,'VLMExtractor',forbidden)
    jobs=V.VisionJobs(store,clip_model_path='missing-clip',vlm_model_path='missing-vlm')
    queued=submit(jobs)['jobs'][0]
    try:
        assert jobs.run_pending()[0]['state']=='succeeded'
        result=jobs.result(queued['id'])
        assert calls==['clip','vlm']
        assert len(result['model_runs'])==13
        assert jobs.options()['capabilities']['clip']=='unavailable'
        assert all(frame['clip']['status']=='not_run' for frame in result['keyframes'])
    finally:
        assert jobs.close()['stopped']
    assert jobs._extractors is None


def test_resource_failure_has_no_candidate_and_explicit_retry_preserves_attempts(store,monkeypatch):
    before=task(store)
    def unavailable(*args,**kwargs):
        raise WorkspaceError('INSUFFICIENT_MEMORY','Measured memory reserve unavailable',retryable=True)
    monkeypatch.setattr(M,'run_phase',unavailable)
    jobs=V.VisionJobs(store,clip_model_path='missing-clip',vlm_model_path='missing-vlm')
    queued=submit(jobs)['jobs'][0]
    try:
        failed=jobs.run_pending()[0]
        assert failed['state']=='failed' and failed['error_code']=='INSUFFICIENT_MEMORY'
        assert failed['commit'] is None and task(store)==before
        monkeypatch.setattr(M,'run_phase',fake_phase)
        jobs.queue.control_job(queued['id'],'retry_failed',command_id='retry-low-memory')
        done=jobs.run_pending()[0]
        assert done['state']=='succeeded' and done['attempt']==2
        assert jobs.result(queued['id'])
        assert task(store)==before
    finally:
        jobs.close()


def test_old_frozen_jobs_validate_but_cannot_silently_execute_new_policy(store):
    jobs=V.VisionJobs(store,clip_model_path='missing-clip',vlm_model_path='missing-vlm')
    current=task(store)
    parameters=V._parameters(include_model_resources=False,optional_providers=jobs._optional_config())
    batch=jobs.queue.create_batch([{'task_id':current['id'],'expected_revision':current['revision'],
        'expected_sha256':current['content_sha256']}],operation=V.OPERATION,processor_version=V.VERSION,
        parameters=parameters,command_id='old-resource-contract')
    try:
        assert V._validate_job(batch['jobs'][0])==parameters
        failed=jobs.run_pending(batch_id=batch['id'])[0]
        assert failed['state']=='failed' and failed['error_code']=='VISION_RESUBMIT_REQUIRED'
    finally:
        jobs.close()


def test_completed_pre_resource_candidate_keeps_read_and_apply_paths(store,monkeypatch):
    jobs=V.VisionJobs(store,clip_model_path='missing-clip',vlm_model_path='missing-vlm')
    current=task(store)
    parameters=V._parameters(include_model_resources=False,optional_providers=jobs._optional_config())
    batch=jobs.queue.create_batch([{'task_id':current['id'],'expected_revision':current['revision'],
        'expected_sha256':current['content_sha256']}],operation=V.OPERATION,processor_version=V.VERSION,
        parameters=parameters,command_id='old-completed-resource-contract')
    original=V._job_identity
    def historical_identity(**kwargs):
        return original(**dict(kwargs,include_model_resources=False))
    try:
        with monkeypatch.context() as historical:
            # The fixture uses real decoding and queue commit, while freezing
            # the historical identity shape; model output remains not_run.
            historical.setattr(V,'_job_identity',historical_identity)
            historical.setattr(M,'run_phase',fake_phase)
            assert jobs.run_pending()[0]['state']=='succeeded'
        old=jobs.result(batch['jobs'][0]['id'])
        assert 'model_resources' not in old['job_identity']['config']
        assert len(old['model_runs'])==13
        jobs.apply(batch['jobs'][0]['id'],command_id='apply-old-resource-candidate',
            expected_revision=current['revision'],expected_sha256=current['content_sha256'],request_body=b'{}')
        assert jobs.result(batch['jobs'][0]['id'])==old
        assert all(a['review_status']=='unreviewed' for a in task(store)['annotations'])
    finally:
        jobs.close()


def test_parent_checks_exact_worker_checkpoint_and_revision(tmp_path,monkeypatch):
    monkeypatch.setenv('TRACE_ANNOTATE_CLIP_CACHE',str(tmp_path/'clip'))
    monkeypatch.setenv('TRACE_ANNOTATE_VLM_CACHE',str(tmp_path/'vlm'))
    checkpoint=tmp_path/'clip/models--example/snapshots/exact/weights.bin'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'exact immutable checkpoint')
    config=M.configuration()
    identity={'weights_path':str(checkpoint),'weights_sha256':M._digest(checkpoint),'selected_revision':'exact'}
    M._validate_identity('clip',config,identity,True)
    for changed in [dict(identity,weights_sha256='a'*64),dict(identity,selected_revision='another'),
                    dict(identity,weights_path=str(tmp_path/'other.bin'))]:
        with pytest.raises(WorkspaceError) as error:
            M._validate_identity('clip',config,changed,True)
        assert error.value.code=='MODEL_WORKER_PROTOCOL'


def test_digest_rechecks_same_size_bytes_with_restored_mtime(tmp_path):
    path=tmp_path/'checkpoint.bin'
    path.write_bytes(b'first-version')
    before=path.stat()
    expected=M._digest(path)
    path.write_bytes(b'other-version')
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert path.stat().st_size==before.st_size
    assert M._digest(path)!=expected


def test_frozen_policy_and_inventory_reject_malformed_or_lowered_values(tmp_path,monkeypatch):
    import copy
    monkeypatch.setenv('TRACE_ANNOTATE_CLIP_CACHE',str(tmp_path/'clip'))
    monkeypatch.setenv('TRACE_ANNOTATE_VLM_CACHE',str(tmp_path/'vlm'))
    config=M.configuration()
    invalid=[]
    lowered=copy.deepcopy(config)
    lowered['policy']['reserve_bytes']=1
    invalid.append(lowered)
    escaped=copy.deepcopy(config)
    escaped['clip']['inventory']=[{'file':'../weights.bin','bytes':1,'sha256':'a'*64}]
    invalid.append(escaped)
    for value in invalid:
        with pytest.raises(WorkspaceError) as error:
            M.validate_configuration(value)
        assert error.value.code=='SCHEMA_INVALID'


def test_selected_checkpoint_load_errors_cannot_masquerade_as_absent_weights():
    from types import SimpleNamespace
    adapter=SimpleNamespace(model=None,load_error='incompatible layer shape',
        code_identity={'weights_path':'selected.bin','weights_sha256':'a'*64})
    with pytest.raises(M.ModelLoadFailed,match='incompatible layer'):
        M._check_loaded_adapter(adapter)
    adapter.load_error='DefaultCPUAllocator cannot allocate memory'
    with pytest.raises(MemoryError):
        M._check_loaded_adapter(adapter)
    adapter.load_error='custom model has no verifiable local checkpoint'
    adapter.code_identity={'weights_path':None,'weights_sha256':None}
    assert M._check_loaded_adapter(adapter)==adapter.code_identity


@pytest.mark.parametrize('available',[False,'false'])
def test_worker_cannot_report_success_without_a_loaded_model(tmp_path,monkeypatch,available):
    monkeypatch.setenv('TRACE_ANNOTATE_CLIP_CACHE',str(tmp_path/'clip'))
    monkeypatch.setenv('TRACE_ANNOTATE_VLM_CACHE',str(tmp_path/'vlm'))
    config=M.configuration()
    images=[{'frame_index':'0','path':'decoded.png','sha256':'a'*64}]
    monkeypatch.setattr(M,'_unavailable',lambda *a:None)
    def inconsistent(command,*a,**k):
        output=Path(command[-1])
        output.write_text(json.dumps({'contract':M.CONTRACT,'module':'clip','available':available,
            'identity':M._empty_identity('clip',config),
            'results':[dict(images[0],input_sha256=images[0]['sha256'],result={'status':'success'})]}),'utf-8')
        return {'status':'completed'}
    monkeypatch.setattr(R,'run_owned',inconsistent)
    with pytest.raises(WorkspaceError) as error:
        M.run_phase('clip',config,images,tmp_path,cancelled=lambda:False)
    assert error.value.code=='MODEL_WORKER_PROTOCOL'


@pytest.mark.parametrize('cooperative',[False,True],ids=['uncooperative-owner','cooperative-cancel'])
def test_server_close_reports_vision_owner_and_preserves_store_until_actual_exit(store,tmp_path,monkeypatch,cooperative):
    from trace_annotate.workspace import server as HTTP
    from trace_annotate.workspace.store import WorkspaceStore
    entered,release,cancel_recorded=threading.Event(),threading.Event(),threading.Event()
    monkeypatch.setenv('TRACE_ANNOTATE_CLIP_CACHE',str(tmp_path/'empty-close-clip'))
    monkeypatch.setenv('TRACE_ANNOTATE_VLM_CACHE',str(tmp_path/'empty-close-vlm'))
    # Keep the real VisionJobs runner, queue ownership and server shutdown;
    # no optional model construction is needed for this lifecycle boundary.
    monkeypatch.setattr(V.VisionJobs,'start',lambda self:self)
    monkeypatch.setattr(V.VisionJobs,'_load_extractors',lambda self,thresholds:None)
    server=HTTP.make_server(store)
    assert server.analysis_jobs.close(timeout=5)["stopped"]
    assert server.render_jobs.close(timeout=5)["stopped"]
    current=task(store)
    worker=None
    def blocked_scene_stage(*args,**kwargs):
        entered.set()
        # Cooperative control waits for the real queue cancellation transaction;
        # the other mode models a native stage ignoring cancellation temporarily.
        assert (cancel_recorded if cooperative else release).wait(10)
        raise WorkspaceError('CANCELLED','Controlled vision stage stopped after cancellation')
    monkeypatch.setattr(V,'_run_scenes',blocked_scene_stage)
    original_cancel=server.queue.cancel
    def cancel_and_signal(job_id):
        result=original_cancel(job_id)
        cancel_recorded.set()
        return result
    monkeypatch.setattr(server.queue,'cancel',cancel_and_signal)
    queued=server.vision_jobs.submit(current['id'],command_id='vision-close-owner',
        expected_revision=current['revision'],expected_sha256=current['content_sha256'])['jobs'][0]
    try:
        worker=threading.Thread(target=server.vision_jobs.run_pending)
        worker.start()
        assert entered.wait(10)
        monkeypatch.setattr(HTTP,'CLOSE_WORKER_SECONDS',2.0 if cooperative else .01)
        if cooperative:
            server.server_close()
            assert server.shutdown_report['vision_worker']['stopped'] is True
        else:
            with pytest.raises(WorkspaceError) as busy_server:
                server.server_close()
            assert busy_server.value.code=='WORKSPACE_BUSY'
            assert 'Vision worker' in str(busy_server.value)
            assert server.shutdown_report['vision_worker']['stopped'] is False
            assert worker.is_alive() and store.lock._handle is not None
            with pytest.raises(WorkspaceError) as busy_store:
                store.close()
            assert busy_store.value.code=='WORKSPACE_BUSY'
            assert task(store)==current
            # A failed close must retain the actual OS single-writer lock.
            with pytest.raises(WorkspaceError) as locked:
                WorkspaceStore(store.root).open()
            assert locked.value.code=='PROJECT_LOCKED'
    finally:
        release.set()
        if worker is not None:
            worker.join(10)
        monkeypatch.setattr(HTTP,'CLOSE_WORKER_SECONDS',2.0)
        server.server_close()
    assert worker is not None and not worker.is_alive()
    assert cancel_recorded.is_set()
    assert server.shutdown_report['vision_worker']['stopped'] is True
    terminal=server.queue.get_job(queued['id'])
    assert terminal['state']=='cancelled' and terminal['commit'] is None
    assert task(store)==current
    store.close()
    store.open()
    assert task(store)==current
    assert server.queue.get_job(queued['id'])['state']=='cancelled'
