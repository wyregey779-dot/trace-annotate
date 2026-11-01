"""Explicit cancelled-job resumption preserves frozen input and physical fences.

Actual SQLite/backup/loopback HTTP are exercised. Controlled JSON processors
test queue semantics, not semantic annotation or measured analysis quality.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import shutil
import subprocess
import threading

import pytest

from trace_annotate.workspace import backup as B
from trace_annotate.workspace.analysis_jobs import AnalysisJobs
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.queue import DurableQueue, WorkerContext
from trace_annotate.workspace.server import make_server
from trace_annotate.workspace.store import WorkspaceStore
from tests.test_workspace_server import request

API='/api/v1/projects/current'


@pytest.fixture(scope='module')
def resume_snapshot(tmp_path_factory):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):pytest.skip('Real media admission requires FFmpeg/ffprobe')
    root=tmp_path_factory.mktemp('cancel-resume-base');clip=root/'clip.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-v','error','-nostdin','-n','-f','lavfi','-i',
                    'color=c=blue:s=32x32:r=5:d=0.4','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],
                   check=True,capture_output=True,timeout=30)
    store=WorkspaceStore(root/'workspace');store.create()
    try:
        store.import_source(clip)
        snapshot=root/'snapshot';B.backup_workspace(store,snapshot)
    finally:store.close()
    return snapshot


@pytest.fixture()
def store(tmp_path,resume_snapshot):
    path=tmp_path/'workspace';B.restore_workspace(resume_snapshot,path)
    opened=WorkspaceStore(path);opened.open()
    try:yield opened
    finally:opened.close()


def current(store):return store.get_task(store.list_tasks()[0]['task_id'])


def enqueue(queue,store,number=1):
    return queue.enqueue(current(store)['id'],operation='controlled-resume',processor_version='test-v1',
                         parameters={'threshold':0.3,'tracks':['main'],'number':number})


def frozen(store,job_id):
    return store.conn.execute('SELECT input,idempotency_key FROM jobs WHERE id=?',(job_id,)).fetchone()


def error_code(expected,call):
    with pytest.raises(WorkspaceError) as error:call()
    assert error.value.code==expected


def finish(queue,job,value=None):
    return queue.finish(job['id'],job['lease_generation'],job['worker_id'],{'controlled_candidate':True} if value is None else value)


def edit_task(store):
    task=current(store)
    annotation={'id':'ann-user-draft','type':'whole_clip','label':'User draft','text':'Keep -42 and 1,234 unchanged.',
                'track':'main','tags':[],'attributes':{},'evidence_ids':[],
                'provenance':{'origin':'manual','actor_id':'user','job_id':None}}
    store.apply_task_command(command_id='cmd-user-edit',task_id=task['id'],expected_revision=task['revision'],
        expected_sha256=task['content_sha256'],operation='add_annotation',payload={'annotation':annotation},request_body=b'cmd-user-edit')
    return current(store)


def test_queued_cancel_resume_is_same_job_same_task_and_frozen_input(store):
    queue=DurableQueue(store);job=enqueue(queue,store);before=frozen(store,job['id']);task=current(store)
    queue.control_job(job['id'],'cancel',command_id='cmd-cancel')
    assert queue.claim('worker-before-resume') is None
    result=queue.control_job(job['id'],'retry_cancelled',command_id='cmd-resume')
    assert result=={'job_id':job['id'],'action':'retry_cancelled','state':'queued'}
    assert frozen(store,job['id'])==before
    seen=[]
    def processor(value,context):
        seen.append(value);context.heartbeat();return {'controlled_candidate':True}
    done=queue.execute_once(processor,worker_id='worker-resume')
    assert done['id']==job['id'] and done['task_id']==task['id'] and done['state']=='succeeded'
    assert seen[0]['task_snapshot']==task and seen[0]['parameters']==job['parameters']
    assert store.conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==1
    assert store.conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]==1
    assert store.conn.execute('SELECT COUNT(*) FROM batches').fetchone()[0]==1
    assert store.conn.execute('SELECT COUNT(*) FROM job_commits').fetchone()[0]==1
    assert current(store)==task and queue.claim('worker-no-duplicate') is None


def test_replayed_resume_never_requeues_a_later_cancellation(store):
    queue=DurableQueue(store);job=enqueue(queue,store)
    queue.cancel(job['id']);receipt=queue.control_job(job['id'],'retry_cancelled',command_id='cmd-resume-once')
    first=queue.claim('worker-first');queue.cancel(job['id']);queue.acknowledge_cancel(job['id'],first['lease_generation'],'worker-first')
    assert queue.control_job(job['id'],'retry_cancelled',command_id='cmd-resume-once')==receipt
    assert queue.get_job(job['id'])['state']=='cancelled' and queue.claim('worker-replay') is None
    error_code('IDEMPOTENCY_CONFLICT',lambda:queue.control_job(job['id'],'cancel',command_id='cmd-resume-once'))
    queue.control_job(job['id'],'retry_cancelled',command_id='cmd-resume-second')
    second=queue.claim('worker-second');finish(queue,second)
    assert second['attempt']==2 and second['lease_generation']>first['lease_generation']
    assert queue.control_job(job['id'],'retry_cancelled',command_id='cmd-resume-second')['state']=='queued'
    assert queue.get_job(job['id'])['state']=='succeeded' and queue.claim('worker-after-success') is None
    assert store.conn.execute('SELECT COUNT(*) FROM job_commits').fetchone()[0]==1


def test_new_enqueue_of_same_cancelled_spec_does_not_implicitly_resume(store):
    queue=DurableQueue(store);job=enqueue(queue,store);queue.cancel(job['id'])
    repeated=enqueue(queue,store)
    assert repeated['id']==job['id'] and repeated['state']=='cancelled'
    assert queue.claim('worker-no-implicit-resume') is None
    queue.control_job(job['id'],'retry_cancelled',command_id='cmd-explicit')
    assert queue.claim('worker-explicit')['id']==job['id']


def test_resume_preserves_old_revision_after_user_edit_and_flags_candidate_stale(store):
    queue=DurableQueue(store);job=enqueue(queue,store);before=frozen(store,job['id']);queue.cancel(job['id'])
    changed=edit_task(store);queue.retry_cancelled(job['id']);claimed=queue.claim('worker-old-input')
    assert claimed['input_revision']==0 and claimed['task_snapshot']==job['task_snapshot']
    assert claimed['input_sha256']==job['input_sha256']!=changed['content_sha256']
    assert frozen(store,job['id'])==before
    commit=finish(queue,claimed)
    assert commit['input_still_current'] is False and commit['current_revision']==changed['revision']
    assert current(store)==changed and current(store)['annotations'][0]['text']=='Keep -42 and 1,234 unchanged.'


@pytest.mark.parametrize('state',['queued','running','cancel_requested','failed','succeeded'])
def test_only_cancelled_can_resume_and_failed_retry_semantics_are_unchanged(store,state):
    queue=DurableQueue(store);job=enqueue(queue,store)
    if state!='queued':
        claim=queue.claim('worker-prepare-state')
        if state=='cancel_requested':queue.cancel(job['id'])
        elif state=='failed':queue.fail(job['id'],claim['lease_generation'],'worker-prepare-state')
        elif state=='succeeded':finish(queue,claim)
    before=queue.get_job(job['id']);raw=frozen(store,job['id'])
    error_code('INVALID_JOB_STATE',lambda:queue.control_job(job['id'],'retry_cancelled',command_id='cmd-invalid-resume'))
    assert queue.get_job(job['id'])==before and frozen(store,job['id'])==raw
    error_code('COMMAND_UNKNOWN',lambda:store.get_command_receipt('cmd-invalid-resume'))
    if state=='failed':assert queue.retry_failed(job['id'])['state']=='queued'


def test_retry_failed_still_refuses_cancelled_jobs(store):
    queue=DurableQueue(store);job=enqueue(queue,store);queue.cancel(job['id'])
    error_code('INVALID_JOB_STATE',lambda:queue.control_job(job['id'],'retry_failed',command_id='cmd-wrong-action'))
    assert queue.get_job(job['id'])['state']=='cancelled'


def test_late_old_generation_cannot_heartbeat_fail_or_commit_after_resume(store):
    queue=DurableQueue(store);job=enqueue(queue,store);old=queue.claim('worker-old');context=WorkerContext(queue,old)
    queue.cancel(job['id']);queue.acknowledge_cancel(job['id'],old['lease_generation'],'worker-old');queue.retry_cancelled(job['id'])
    assert context.cancelled()
    for call in (lambda:context.heartbeat(),lambda:finish(queue,old),lambda:queue.fail(job['id'],old['lease_generation'],'worker-old')):
        error_code('STALE_LEASE',call)
    # Reusing the worker identity isolates generation fencing from the separate
    # worker-id check; an old generation must still be rejected while running.
    new=queue.claim('worker-old');assert new['lease_generation']>old['lease_generation']
    error_code('STALE_LEASE',lambda:context.heartbeat())
    error_code('STALE_LEASE',lambda:finish(queue,old))
    finish(queue,new);error_code('STALE_LEASE',lambda:finish(queue,old))
    assert store.conn.execute('SELECT COUNT(*) FROM job_commits').fetchone()[0]==1
    assert [a['state'] for a in queue.get_job(job['id'])['attempts']]==['cancelled','succeeded']


@pytest.mark.parametrize('transition',['acknowledged','expired'])
def test_active_cancelled_attempt_blocks_same_job_at_max2_but_allows_other_job(store,transition):
    now=[1000.0];queue=DurableQueue(store,max_concurrency=2,lease_seconds=5,clock=lambda:now[0])
    old=enqueue(queue,store,1);other=enqueue(queue,store,2)
    batches=queue.list_batches();old_batch=next(b['id'] for b in batches if b['jobs'][0]['id']==old['id']);other_batch=next(b['id'] for b in batches if b['jobs'][0]['id']==other['id'])
    entered,release=threading.Event(),threading.Event();snapshots=[]
    def slow(job,context):
        snapshots.append(job);entered.set();assert release.wait(timeout=10);return {'late_old_candidate':True}
    with ThreadPoolExecutor(max_workers=1) as executor:
        running=executor.submit(queue.execute_once,slow,worker_id='worker-old-active')
        try:
            assert entered.wait(timeout=5);claimed=snapshots[0]
            queue.cancel(old['id'])
            if transition=='expired':now[0]+=6;queue.recover()
            else:queue.acknowledge_cancel(old['id'],claimed['lease_generation'],'worker-old-active')
            assert queue.get_job(old['id'])['state']=='cancelled';queue.retry_cancelled(old['id'])
            next_job=queue.claim('worker-other',other_batch)
            assert next_job is not None and next_job['id']==other['id']
            finish(queue,next_job)
            assert queue.claim('worker-must-not-reenter',old_batch) is None and not running.done()
            assert queue.get_job(old['id'])['state']=='queued' and queue.get_job(old['id'])['attempt']==1
        finally:release.set()
        assert running.result(timeout=5)['state']=='queued'
    new=queue.claim('worker-after-exit',old_batch);assert new['id']==old['id'] and new['lease_generation']>claimed['lease_generation']
    finish(queue,new)
    assert [a['state'] for a in queue.get_job(old['id'])['attempts']]==['cancelled','succeeded']


@pytest.mark.parametrize('mode',['reopen','backup_restore'])
def test_cancelled_job_survives_reopen_or_restore_and_resumes_frozen_input(store,tmp_path,mode):
    queue=DurableQueue(store);job=enqueue(queue,store);queue.control_job(job['id'],'cancel',command_id='cmd-cancel-persist')
    before=frozen(store,job['id']);changed=edit_task(store)
    root=store.root
    if mode=='backup_restore':
        snapshot=tmp_path/'cancelled-backup';B.backup_workspace(store,snapshot)
        root=tmp_path/'cancelled-restored';result=B.restore_workspace(snapshot,root)
        assert result['jobs_restarted'] is False and result['commands_replayed'] is False
    else:store.close()
    recovered=WorkspaceStore(root);recovered.open()
    try:
        resumed=DurableQueue(recovered);assert resumed.get_job(job['id'])['state']=='cancelled'
        assert resumed.claim('worker-no-auto-restart') is None
        resumed.control_job(job['id'],'retry_cancelled',command_id='cmd-persist-resume')
        assert frozen(recovered,job['id'])==before
        claimed=resumed.claim('worker-restored');commit=finish(resumed,claimed)
        assert commit['input_still_current'] is False and current(recovered)==changed
        assert recovered.get_command_receipt('cmd-cancel-persist')['response']['state']=='cancelled'
    finally:recovered.close()


@contextmanager
def serving(store,monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(AnalysisJobs,'start',lambda self:self)
        server=make_server(store)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    try:yield server
    finally:
        server.shutdown();thread.join(timeout=10);server.server_close();assert not thread.is_alive()


def test_actual_http_cancelled_control_matches_reanalysis_button_contract(store,monkeypatch):
    task=current(store)
    with serving(store,monkeypatch) as server:
        submit={'command_id':'cmd-http-submit','expected_revision':task['revision'],'expected_sha256':task['content_sha256'],
                'stream_index':None,'window':None,'subject_annotation_id':None}
        response=request(server,'POST',API+'/tasks/'+task['id']+'/analysis-jobs',submit)
        assert response[0]==202;job=response[2]['jobs'][0];path=API+'/jobs/'+job['id']+'/control';before=frozen(store,job['id'])
        assert request(server,'POST',path,{'command_id':'cmd-http-cancel','action':'cancel'})[2]['state']=='cancelled'
        edit_task(store)
        body={'command_id':'cmd-http-resume','action':'retry_cancelled'}
        first=request(server,'POST',path,body);assert first[0]==200 and first[2]['state']=='queued'
        repeated=request(server,'POST',path,body);assert repeated[0]==first[0] and repeated[2]==first[2]
        assert frozen(store,job['id'])==before and server.queue.get_job(job['id'])['input_revision']==0
        request(server,'POST',path,{'command_id':'cmd-http-cancel-again','action':'cancel'})
        replay=request(server,'POST',path,body);assert replay[0]==200 and replay[2]==first[2]
        assert server.queue.get_job(job['id'])['state']=='cancelled'
        invalid=request(server,'POST',path,{'command_id':'cmd-http-wrong','action':'retry_failed'})
        assert invalid[0]==409 and invalid[2]['error']['code']=='INVALID_JOB_STATE'
        malformed=request(server,'POST',path,{'command_id':'cmd-http-bad-fields','action':'retry_cancelled','expected_revision':1})
        assert malformed[0]==422 and malformed[2]['error']['code']=='SCHEMA_INVALID'
        page=request(server,'GET','/')[2].decode('utf-8')
        assert '重新分析' in page and "controlAnalysisJob(job.id,'retry_cancelled')" in page
        assert store.conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==1
