"""The legacy integrity CLI must not consume a different processor's jobs."""
from tests.test_workspace_cli import cli, cli_video, cli_project, create_batch, payload, typed_error
from trace_annotate.workspace.analysis_jobs import AnalysisJobs
from trace_annotate.workspace.store import WorkspaceStore


def analysis_batch(project, task_id):
    store=WorkspaceStore(project);store.open()
    try:
        task=store.get_task(task_id)
        return AnalysisJobs(store).submit(task_id,command_id='cmd-cli-analysis',
            expected_revision=task['revision'],expected_sha256=task['content_sha256'])
    finally:store.close()


def test_integrity_cli_refuses_an_explicit_analysis_batch_before_claim(cli,cli_project):
    project,task_id=cli_project
    analysis=analysis_batch(project,task_id)
    typed_error(cli('batch','run','--project',project,'--batch-id',analysis['id']),
                'UNSUPPORTED_PROCESSOR')
    after=payload(cli('batch','show','--project',project,'--batch-id',analysis['id']))
    assert after['jobs'][0]['state']=='queued'
    assert after['jobs'][0]['attempt']==0
    assert after['jobs'][0]['commit'] is None


def test_integrity_cli_runs_its_own_jobs_without_failing_queued_analysis(cli,cli_project):
    project,task_id=cli_project
    analysis=analysis_batch(project,task_id)
    verification=create_batch(cli,project,task_id)
    result=payload(cli('batch','run','--project',project),expected=1)
    assert result['skipped_batch_ids']==[analysis['id']]
    assert len(result['executed'])==1 and result['executed'][0]['state']=='succeeded'
    assert result['executed'][0]['operation']=='verify-media'
    after=payload(cli('batch','show','--project',project,'--batch-id',analysis['id']))
    assert after['jobs'][0]['state']=='queued' and after['jobs'][0]['attempt']==0
    completed=payload(cli('batch','show','--project',project,'--batch-id',verification['id']))
    assert completed['jobs'][0]['state']=='succeeded'
