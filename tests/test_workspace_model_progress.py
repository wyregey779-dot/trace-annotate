"""Progress publication survives transient Windows read/replace sharing conflicts."""
import errno
import copy
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from trace_annotate.workspace import model_worker as M
from trace_annotate.workspace import resource_budget as R
from trace_annotate.workspace.errors import WorkspaceError
from tests.test_workspace_vision_jobs import clip, store, task


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_replace_conflicts_preserve_atomic_progress(tmp_path, monkeypatch, winerror):
    destination = tmp_path / "progress.json"
    old = {"phase": "previous", "started_at": 1}
    destination.write_text(json.dumps(old), "utf-8")
    replace = Path.replace
    calls = []
    def contended(source, target):
        calls.append(1)
        if len(calls) <= 3:
            assert json.loads(destination.read_text("utf-8")) == old
            error = PermissionError("simulated Windows sharing conflict")
            error.winerror = winerror
            raise error
        return replace(source, target)
    monkeypatch.setattr(Path, "replace", contended)
    monkeypatch.setattr(M.time, "sleep", lambda seconds: None)
    assert M._publish_progress(tmp_path / "result.json", "frame", "83") == 3
    value = json.loads(destination.read_text("utf-8"))
    assert value["phase"] == "frame" and value["frame_index"] == "83"
    assert not (tmp_path / "progress.pending").exists()


def test_persistent_sharing_conflict_is_bounded_and_retains_pending_diagnostic(tmp_path, monkeypatch):
    attempts, sleeps = [], []
    def blocked(source, target):
        attempts.append(1)
        error = PermissionError("persistent Windows sharing conflict")
        error.winerror = 5
        raise error
    monkeypatch.setattr(Path, "replace", blocked)
    monkeypatch.setattr(M.time, "sleep", sleeps.append)
    with pytest.raises(M.ModelProgressFailed):
        M._publish_progress(tmp_path / "result.json", "finalizing")
    assert len(attempts) == M.PROGRESS_REPLACE_ATTEMPTS
    assert sleeps == [M.PROGRESS_RETRY_SECONDS] * (M.PROGRESS_REPLACE_ATTEMPTS - 1)
    assert json.loads((tmp_path / "progress.pending").read_text("utf-8"))["phase"] == "finalizing"
    assert not (tmp_path / "result.json").exists()


def test_unrelated_filesystem_error_fails_immediately(tmp_path, monkeypatch):
    def full(source, target):
        raise OSError(errno.ENOSPC, "No space left")
    monkeypatch.setattr(Path, "replace", full)
    monkeypatch.setattr(M.time, "sleep", lambda seconds: pytest.fail("Disk errors must not be retried"))
    with pytest.raises(OSError) as error:
        M._publish_progress(tmp_path / "result.json", "frame", "1")
    assert error.value.errno == errno.ENOSPC


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics required")
def test_real_windows_reader_release_allows_progress_publication(tmp_path):
    destination = tmp_path / "progress.json"
    destination.write_text('{"phase":"previous"}', "utf-8")
    reader = destination.open("rb")
    released = threading.Event()
    def release():
        time.sleep(.15)
        reader.close()
        released.set()
    worker = threading.Thread(target=release)
    worker.start()
    try:
        retries = M._publish_progress(tmp_path / "result.json", "frame", "4")
        assert retries > 0
        assert json.loads(destination.read_text("utf-8"))["frame_index"] == "4"
    finally:
        reader.close()
        worker.join(timeout=2)
    assert released.is_set()


def test_progress_failure_exit_keeps_typed_queue_error_and_process_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "memory_available", lambda: 10 * R.GIB)
    settings = R.policy()
    bootstrap = ("import os,sys;sys.path.insert(0,sys.argv[1]);"
                 "from trace_annotate.workspace.resource_budget import join_owned_job;"
                 "join_owned_job(os.environ.get('TRACE_ANNOTATE_OWNED_MODEL_JOB',''));sys.exit(78)")
    with pytest.raises(WorkspaceError) as error:
        R.run_owned([sys.executable, "-I", "-B", "-c", bootstrap, str(Path(M.__file__).parents[2])],
                    tmp_path / "execution", settings=settings, cancelled=lambda: False, timeout=5)
    assert error.value.code == "MODEL_PROGRESS_FAILED"
    assert error.value.details["exit_code"] == 78
    receipt = json.loads((tmp_path / "execution/process-receipt.json").read_text("utf-8"))
    assert receipt["status"] == "failed" and receipt["actual_exit_code"] == 78


def test_older_model_implementation_keeps_completed_results_and_requires_new_execution(store, monkeypatch):
    from trace_annotate.workspace import vision_jobs as V
    from tests.test_workspace_model_resources import fake_phase
    jobs = V.VisionJobs(store, clip_model_path="missing-clip", vlm_model_path="missing-vlm")
    current = task(store)
    legacy = copy.deepcopy(jobs._model_config())
    legacy["implementation"]["workspace/model_worker.py"] = "0" * 64
    try:
        with monkeypatch.context() as patch:
            # Real decoding/queue commits with a controlled older implementation
            # descriptor. The unrelated absent model phase stays explicitly not_run.
            patch.setattr(jobs, "_model_config", lambda: copy.deepcopy(legacy))
            patch.setattr(M, "run_phase", fake_phase)
            completed = jobs.submit(current["id"], command_id="older-implementation-completed",
                expected_revision=current["revision"], expected_sha256=current["content_sha256"])
            assert jobs.run_pending()[0]["state"] == "succeeded"
        job_id = completed["jobs"][0]["id"]
        prior = jobs.result(job_id)
        pending = jobs.queue.create_batch([{"task_id": current["id"],
            "expected_revision": current["revision"], "expected_sha256": current["content_sha256"]}],
            operation=V.OPERATION, processor_version=V.VERSION, command_id="older-implementation-queued",
            parameters=V._parameters(model_resources=legacy, optional_providers=jobs._optional_config(),
                window={"start": V.S.rational(0), "end": V.S.rational(1)}))
        rejected = jobs.run_pending(batch_id=pending["id"])[0]
        assert rejected["state"] == "failed" and rejected["error_code"] == "VISION_RESUBMIT_REQUIRED"
        jobs.apply(job_id, command_id="apply-older-implementation", expected_revision=current["revision"],
                   expected_sha256=current["content_sha256"], request_body=b"{}")
        assert jobs.result(job_id) == prior
        assert all(a["review_status"] == "unreviewed" for a in task(store)["annotations"])
    finally:
        jobs.close()
