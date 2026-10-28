"""Real SQLite queue tests with controlled JSON processors, not annotation QA.

Media admission uses a tiny generated clip. Processor outputs are deliberately
synthetic; these tests establish persistence and race semantics only.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.queue import DurableQueue, read_queue_snapshot
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def queue_clip(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for queue admission fixture")
    clip = tmp_path_factory.mktemp("queue-media") / "clip.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=32x32:r=5:d=0.4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=30)
    return clip


@pytest.fixture()
def store(tmp_path, queue_clip):
    project = WorkspaceStore(tmp_path / "project")
    project.create()
    try:
        project.import_source(queue_clip)
        yield project
    finally:
        project.close()


def task(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


def item(store):
    value = task(store)
    return {"task_id": value["id"], "expected_revision": value["revision"],
            "expected_sha256": store.task_digest(value)}


def enqueue(queue, store, **kw):
    return queue.enqueue(task(store)["id"], operation="synthetic-candidate",
                         processor_version="test-v1", **kw)


def finish(queue, claimed, value=None):
    return queue.finish(claimed["id"], claimed["lease_generation"], claimed["worker_id"],
                        {"candidate": "controlled-test"} if value is None else value)


def code(expected, call):
    with pytest.raises(WorkspaceError) as error:
        call()
    assert error.value.code == expected


def test_enqueue_freezes_input_deduplicates_and_records_command(store):
    queue = DurableQueue(store)
    config = {"threshold": 0.3, "tracks": ["main"]}
    first = queue.create_batch([item(store)], operation="probe", processor_version="v1",
                               parameters=config, command_id="cmd-batch")
    config["tracks"].append("changed-after-enqueue")
    original = first["jobs"][0]
    assert original["parameters"] == {"threshold": 0.3, "tracks": ["main"]}
    assert original["task_snapshot"] == task(store)
    assert original["media_snapshot"]["id"] == task(store)["media_id"]
    repeated = queue.create_batch([item(store)], operation="probe", processor_version="v1",
                                  parameters=original["parameters"], command_id="cmd-batch")
    assert repeated["id"] == first["id"]
    other = queue.create_batch([item(store)], operation="probe", processor_version="v1",
                               parameters=original["parameters"])
    assert other["jobs"][0]["id"] == original["id"]
    assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    receipt = store.get_command_receipt("cmd-batch")
    assert receipt["state"] == "committed"
    assert receipt["response"]["batch_id"] == first["id"]
    queue.pause(first["id"])
    replay = queue.create_batch([item(store)], operation="probe", processor_version="v1",
                                parameters=original["parameters"], command_id="cmd-batch")
    assert replay == first
    assert queue.get_batch(first["id"])["dispatch_paused"] is True
    code("IDEMPOTENCY_CONFLICT", lambda: queue.create_batch([item(store)], operation="probe",
         processor_version="v2", parameters={}, command_id="cmd-batch"))


@pytest.mark.parametrize("parameters", [{"bad": float("nan")}, {"bad": float("inf")},
                                        {"bad": 2 ** 70}, {1: "coerced-key"}, {"bad": (1, 2)}])
def test_noncanonical_parameters_rejected_without_queue_rows(store, parameters):
    queue = DurableQueue(store)
    code("SCHEMA_INVALID", lambda: enqueue(queue, store, parameters=parameters))
    assert queue.list_batches() == []
    assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_stale_input_and_duplicate_task_reject_entire_batch(store):
    queue = DurableQueue(store)
    old = item(store)
    old["expected_revision"] += 1
    code("REVISION_CONFLICT", lambda: queue.create_batch([old], operation="probe", processor_version="v1"))
    wrong_hash = item(store)
    wrong_hash["expected_sha256"] = "0" * 64
    code("MEDIA_CHANGED", lambda: queue.create_batch([wrong_hash], operation="probe", processor_version="v1"))
    code("SCHEMA_INVALID", lambda: queue.create_batch([item(store), item(store)], operation="probe", processor_version="v1"))
    assert queue.list_batches() == []


def test_pause_blocks_dispatch_but_allows_running_finish(store):
    queue = DurableQueue(store)
    batch = queue.create_batch([item(store)], operation="probe", processor_version="v1")
    queue.pause(batch["id"])
    assert queue.claim("worker-1") is None
    queue.resume(batch["id"])
    claimed = queue.claim("worker-1")
    queue.pause(batch["id"])
    commit = finish(queue, claimed)
    assert commit["disposition"] == "candidate"
    assert queue.get_job(claimed["id"])["state"] == "succeeded"


def test_cancel_queued_and_running_requires_acknowledgement(store):
    queue = DurableQueue(store)
    first = enqueue(queue, store, parameters={"n": 1})
    assert queue.cancel(first["id"])["state"] == "cancelled"
    assert queue.claim("worker-1") is None
    second = enqueue(queue, store, parameters={"n": 2})
    claimed = queue.claim("worker-1")
    assert claimed["id"] == second["id"]
    assert queue.cancel(second["id"])["state"] == "cancel_requested"
    heartbeat = queue.heartbeat(claimed["id"], claimed["lease_generation"], "worker-1")
    assert heartbeat["cancel_requested"] is True
    code("CANCELLED", lambda: finish(queue, claimed))
    assert queue.get_job(claimed["id"])["commit"] is None
    assert queue.acknowledge_cancel(claimed["id"], claimed["lease_generation"], "worker-1")["state"] == "cancelled"
    code("INVALID_JOB_STATE", lambda: queue.retry_failed(first["id"]))


def test_finish_then_cancel_preserves_exactly_one_commit(store):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    first = finish(queue, claimed)
    assert finish(queue, claimed) == first
    code("ALREADY_COMMITTED", lambda: queue.cancel(claimed["id"]))
    code("IDEMPOTENCY_CONFLICT", lambda: finish(queue, claimed, {"different": True}))
    code("STALE_LEASE", lambda: queue.finish(claimed["id"], claimed["lease_generation"] + 1, "worker-1", {}))
    code("SCHEMA_INVALID", lambda: queue.finish(claimed["id"], True, "worker-1", {}))
    assert store.conn.execute("SELECT COUNT(*) FROM job_commits").fetchone()[0] == 1
    assert store.conn.execute("SELECT COUNT(*) FROM durable_commits WHERE id=?", (first["id"],)).fetchone()[0] == 1
    artifact = store.root / first["artifact"]["relative_path"]
    assert json.loads(artifact.read_text(encoding="utf-8")) == {"candidate": "controlled-test"}


def test_human_edit_during_worker_preserves_draft_and_candidate(store):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    initial = task(store)
    annotation = {"id": "manual-note", "type": "whole_clip", "label": "Human label",
                  "text": "Keep this exact text.", "track": "main", "tags": [], "attributes": {},
                  "evidence_ids": [], "provenance": {"origin": "manual", "actor_id": "user", "job_id": None}}
    store.apply_task_command(command_id="manual-edit", task_id=initial["id"], expected_revision=initial["revision"],
                             expected_sha256=store.task_digest(initial), operation="add_annotation",
                             payload={"annotation": annotation}, request_body=b'{"command_id":"manual-edit"}')
    human = task(store)
    commit = finish(queue, claimed, {"proposed_label": "Automated candidate"})
    assert commit["disposition"] == "candidate"
    assert commit["input_still_current"] is False
    assert commit["current_revision"] == human["revision"]
    assert task(store) == human
    assert task(store)["annotations"][0]["text"] == "Keep this exact text."


def test_restart_invalidates_lease_and_requeues_only_uncommitted(store):
    queue = DurableQueue(store)
    enqueue(queue, store, parameters={"n": 1})
    done = queue.claim("worker-old")
    commit = finish(queue, done)
    enqueue(queue, store, parameters={"n": 2})
    interrupted = queue.claim("worker-old")
    root = store.root
    store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        recovered = DurableQueue(reopened)
        assert recovered.get_job(done["id"])["commit"]["id"] == commit["id"]
        assert recovered.get_job(done["id"])["state"] == "succeeded"
        assert recovered.get_job(interrupted["id"])["state"] == "queued"
        new = recovered.claim("worker-new")
        assert new["id"] == interrupted["id"]
        assert new["attempt"] == 2
        assert new["lease_generation"] > interrupted["lease_generation"]
        code("STALE_LEASE", lambda: finish(recovered, interrupted))
        finish(recovered, new)
        assert [a["state"] for a in recovered.get_job(new["id"])["attempts"]] == ["interrupted", "succeeded"]
    finally:
        reopened.close()


def test_wrapper_does_not_recover_live_session_and_heartbeat_extends_lease(store):
    now = [1000.0]
    queue = DurableQueue(store, lease_seconds=5, clock=lambda: now[0])
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    assert DurableQueue(store, clock=lambda: now[0]).get_job(claimed["id"])["state"] == "running"
    now[0] += 4
    queue.heartbeat(claimed["id"], claimed["lease_generation"], "worker-1")
    now[0] += 2
    assert queue.recover() == []
    now[0] += 4
    assert queue.recover()[0]["state"] == "failed"
    code("STALE_LEASE", lambda: finish(queue, claimed))
    queue.retry_failed(claimed["id"])
    new = queue.claim("worker-2")
    assert new["attempt"] == 2
    finish(queue, new)


def test_cancel_survives_restart_without_dispatch(store):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-old")
    queue.cancel(claimed["id"])
    root = store.root
    store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        recovered = DurableQueue(reopened)
        assert recovered.get_job(claimed["id"])["state"] == "cancelled"
        assert recovered.claim("worker-new") is None
    finally:
        reopened.close()


@pytest.mark.parametrize("window", ["before_artifact", "after_artifact", "after_job_commit", "before_transaction_commit"])
def test_commit_fault_windows_rollback_and_retry_once(store, monkeypatch, window):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    before_artifacts = store.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    if window == "before_artifact":
        original = store._put_artifact
        def fail_before_artifact(data):
            raise OSError("injected artifact write failure")
        monkeypatch.setattr(store, "_put_artifact", fail_before_artifact)
    else:
        target = {"after_artifact": "job_commits", "after_job_commit": "durable_commits",
                  "before_transaction_commit": "jobs"}[window]
        event = "UPDATE OF state" if target == "jobs" else "INSERT"
        condition = " WHEN NEW.state='succeeded'" if target == "jobs" else ""
        store.conn.execute(f"CREATE TEMP TRIGGER fail_window BEFORE {event} ON {target}{condition} "
                           "BEGIN SELECT RAISE(ABORT, 'injected commit fault'); END")
    with pytest.raises(Exception, match="injected"):
        finish(queue, claimed)
    assert queue.get_job(claimed["id"])["state"] == "running"
    assert queue.get_job(claimed["id"])["commit"] is None
    assert store.conn.execute("SELECT COUNT(*) FROM job_commits").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM durable_commits").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == before_artifacts
    if window == "before_artifact":
        monkeypatch.setattr(store, "_put_artifact", original)
    else:
        store.conn.execute("DROP TRIGGER fail_window")
    commit = finish(queue, claimed)
    assert finish(queue, claimed) == commit
    assert store.conn.execute("SELECT COUNT(*) FROM job_commits").fetchone()[0] == 1


def test_fifth_window_lost_response_after_commit_reopens_without_duplicate(store):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    committed = []
    def lose_response():
        committed.append(finish(queue, claimed))
        raise ConnectionError("response lost after commit")
    with pytest.raises(ConnectionError):
        lose_response()
    root = store.root
    store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        recovered = DurableQueue(reopened)
        assert recovered.claim("worker-new") is None
        assert finish(recovered, claimed) == committed[0]
        assert reopened.conn.execute("SELECT COUNT(*) FROM job_commits").fetchone()[0] == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("crash_window", ["after_artifact", "after_commit"])
def test_process_exit_recovery_distinguishes_orphan_from_commit(store, crash_window):
    queue = DurableQueue(store)
    queued = enqueue(queue, store)
    root = store.root
    store.close()
    script = """
import os
from pathlib import Path
import sys
from trace_annotate.workspace.store import WorkspaceStore
from trace_annotate.workspace.queue import DurableQueue
store = WorkspaceStore(Path(sys.argv[1]))
store.open()
queue = DurableQueue(store)
job = queue.claim('crashing-worker')
if sys.argv[2] == 'after_artifact':
    original = store._put_artifact
    def publish_and_exit(data):
        original(data)
        os._exit(72)
    store._put_artifact = publish_and_exit
queue.finish(job['id'], job['lease_generation'], job['worker_id'], {'crash_fixture': True})
os._exit(73)
"""
    crashed = subprocess.run([sys.executable, "-c", script, str(root), crash_window],
                             cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, timeout=30)
    assert crashed.returncode == (72 if crash_window == "after_artifact" else 73), crashed.stderr
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        recovered = DurableQueue(reopened)
        job = recovered.get_job(queued["id"])
        assert job["state"] == ("queued" if crash_window == "after_artifact" else "succeeded")
        if crash_window == "after_artifact":
            assert job["commit"] is None
            assert reopened.conn.execute("SELECT COUNT(*) FROM job_commits").fetchone()[0] == 0
            retried = recovered.claim("new-worker")
            finish(recovered, retried)
        else:
            assert recovered.claim("new-worker") is None
        assert reopened.conn.execute("SELECT COUNT(*) FROM job_commits").fetchone()[0] == 1
    finally:
        reopened.close()


def test_concurrent_claims_obey_shared_cap(store):
    queue = DurableQueue(store, max_concurrency=2)
    for number in range(4):
        enqueue(queue, store, parameters={"n": number})
    wrapper = DurableQueue(store)
    gate = threading.Barrier(4)
    def claim(number):
        gate.wait(timeout=5)
        return (queue if number % 2 else wrapper).claim("worker-" + str(number))
    with ThreadPoolExecutor(max_workers=4) as executor:
        claimed = list(executor.map(claim, range(4)))
    active = [job for job in claimed if job is not None]
    assert len(active) == 2
    assert len({job["id"] for job in active}) == 2
    for job in active:
        finish(queue, job)
    assert len(queue.run_pending(lambda job, context: {"next": True})) == 2
    code("CONFIGURATION_CONFLICT", lambda: DurableQueue(store, max_concurrency=3))


def test_runner_isolates_exception_and_retries_same_job_with_attempt_history(store):
    queue = DurableQueue(store)
    first = enqueue(queue, store, parameters={"n": 1})
    second = enqueue(queue, store, parameters={"n": 2})
    def processor(job, context):
        context.heartbeat()
        if job["id"] == first["id"]:
            raise ValueError("private processor diagnostic")
        return {"synthetic": True}
    results = queue.run_pending(processor)
    assert [job["state"] for job in results] == ["failed", "succeeded"]
    assert results[0]["error_code"] == "PROCESSOR_FAILED"
    assert results[1]["id"] == second["id"]
    queue.retry_failed(first["id"])
    retried = queue.execute_once(lambda job, context: {"retried": True})
    assert retried["id"] == first["id"]
    assert retried["attempt"] == 2
    assert [a["state"] for a in retried["attempts"]] == ["failed", "succeeded"]


def test_running_processor_releases_database_for_cancel_race(store):
    queue = DurableQueue(store)
    queued = enqueue(queue, store)
    entered, proceed = threading.Event(), threading.Event()
    def processor(job, context):
        entered.set()
        assert proceed.wait(timeout=5)
        context.raise_if_cancelled()
        pytest.fail("Cancelled processor must not continue")
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(queue.execute_once, processor)
        try:
            assert entered.wait(timeout=5)
            assert queue.cancel(queued["id"])["state"] == "cancel_requested"
        finally:
            proceed.set()
        assert running.result(timeout=5)["state"] == "cancelled"
    assert queue.get_job(queued["id"])["commit"] is None


def test_expired_processor_retains_physical_slot_until_exit(store):
    now = [1000.0]
    queue = DurableQueue(store, lease_seconds=5, clock=lambda: now[0])
    old = enqueue(queue, store, parameters={"n": 1})
    enqueue(queue, store, parameters={"n": 2})
    entered, release = threading.Event(), threading.Event()
    def slow_processor(job, context):
        entered.set()
        assert release.wait(timeout=5)
        return {"late": True}
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(queue.execute_once, slow_processor)
        try:
            assert entered.wait(timeout=5)
            now[0] += 6
            assert queue.claim("another-worker") is None
            assert queue.get_job(old["id"])["state"] == "failed"
        finally:
            release.set()
        assert running.result(timeout=5)["state"] == "failed"
    assert queue.execute_once(lambda job, context: {"next": True})["state"] == "succeeded"
    assert queue.get_job(old["id"])["commit"] is None


def test_processor_input_mutation_cannot_change_commit_identity(store):
    queue = DurableQueue(store)
    queued = enqueue(queue, store)
    def processor(job, context):
        job["id"] = "different-job"
        job["lease_generation"] += 100
        job["task_snapshot"]["annotations"].append({"unvalidated": True})
        return {"synthetic": True}
    result = queue.execute_once(processor)
    assert result["id"] == queued["id"]
    assert result["state"] == "succeeded"
    assert task(store)["annotations"] == []


def test_commit_wins_cancel_race_at_transaction_boundary(store, monkeypatch):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    artifact_entered, release_artifact, cancel_started = threading.Event(), threading.Event(), threading.Event()
    original = store._put_artifact
    def write_artifact(data):
        value = original(data)
        artifact_entered.set()
        assert release_artifact.wait(timeout=5)
        return value
    monkeypatch.setattr(store, "_put_artifact", write_artifact)
    def request_cancel():
        cancel_started.set()
        return queue.cancel(claimed["id"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        finishing = executor.submit(finish, queue, claimed)
        try:
            assert artifact_entered.wait(timeout=5)
            cancelling = executor.submit(request_cancel)
            assert cancel_started.wait(timeout=5)
        finally:
            release_artifact.set()
        assert finishing.result(timeout=5)["disposition"] == "candidate"
        with pytest.raises(WorkspaceError) as error:
            cancelling.result(timeout=5)
        assert error.value.code == "ALREADY_COMMITTED"


def test_control_receipt_replay_cannot_cancel_later_retry(store):
    queue = DurableQueue(store)
    enqueue(queue, store)
    claimed = queue.claim("worker-1")
    queue.fail(claimed["id"], claimed["lease_generation"], "worker-1")
    response = queue.control_job(claimed["id"], "retry_failed", command_id="retry-once")
    again = queue.claim("worker-2")
    queue.fail(again["id"], again["lease_generation"], "worker-2")
    assert queue.control_job(claimed["id"], "retry_failed", command_id="retry-once") == response
    assert queue.get_job(claimed["id"])["state"] == "failed"
    code("IDEMPOTENCY_CONFLICT", lambda: queue.control_job(claimed["id"], "cancel", command_id="retry-once"))


def test_batch_retry_only_failed_and_storage_full_pauses_dispatch(store):
    queue = DurableQueue(store)
    batch = queue.create_batch([item(store)], operation="probe", processor_version="v1")
    claimed = queue.claim("worker-1")
    queue.fail(claimed["id"], claimed["lease_generation"], "worker-1", "STORAGE_FULL")
    assert queue.get_batch(batch["id"])["dispatch_paused"] is True
    result = queue.control_batch(batch["id"], "retry_failed", command_id="retry-failed")
    assert result["outcomes"][0]["state"] == "queued"
    assert queue.claim("worker-2") is None
    queue.control_batch(batch["id"], "resume_dispatch", command_id="resume-batch")
    done = queue.execute_once(lambda job, context: {"result": "synthetic"})
    assert done["state"] == "succeeded"
    result = queue.control_batch(batch["id"], "cancel", command_id="cancel-complete")
    assert result["outcomes"][0]["error_code"] == "ALREADY_COMMITTED"


def test_snapshot_without_queue_does_not_create_tables_or_change_revision(store):
    tables = store.conn.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall()
    changes = store.conn.total_changes
    revision = store.conn.execute("SELECT value FROM meta WHERE key='workspace_revision'").fetchone()[0]
    assert read_queue_snapshot(store) == {"batches": [], "commits": []}
    assert store.conn.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall() == tables
    assert store.conn.total_changes == changes
    assert store.conn.execute("SELECT value FROM meta WHERE key='workspace_revision'").fetchone()[0] == revision


def test_snapshot_expired_lease_is_pure_and_revision_tracks_queue_changes(store):
    now = [1000.0]
    queue = DurableQueue(store, lease_seconds=5, clock=lambda: now[0])
    def revision():
        return int(store.conn.execute("SELECT value FROM meta WHERE key='workspace_revision'").fetchone()[0])
    start_revision = revision()
    enqueue(queue, store)
    queued_revision = revision()
    assert queued_revision > start_revision
    claimed = queue.claim("worker-1")
    claimed_revision = revision()
    assert claimed_revision > queued_revision
    now[0] += 6
    changes = store.conn.total_changes
    view = read_queue_snapshot(store)
    assert view["batches"][0]["jobs"][0]["state"] == "running"
    assert view["batches"][0]["jobs"][0]["lease_generation"] == claimed["lease_generation"]
    assert revision() == claimed_revision
    assert store.conn.total_changes == changes
    assert queue.snapshot() == view
    assert store.conn.total_changes == changes
    queue.recover()
    assert revision() > claimed_revision
    assert read_queue_snapshot(store)["batches"][0]["jobs"][0]["state"] == "failed"


def test_claim_and_physical_slot_registration_share_one_transaction(store, monkeypatch):
    now = [1000.0]
    queue = DurableQueue(store, lease_seconds=5, clock=lambda: now[0])
    enqueue(queue, store, parameters={"n": 1})
    enqueue(queue, store, parameters={"n": 2})
    claimed, release = threading.Event(), threading.Event()
    original = queue.claim
    processor_calls = []
    def pause_after_claim(worker_id, batch_id=None):
        value = original(worker_id, batch_id)
        if worker_id == "paused-worker":
            claimed.set()
            assert release.wait(timeout=5)
        return value
    def processor(job, context):
        processor_calls.append(job["worker_id"])
        return {"synthetic": True}
    monkeypatch.setattr(queue, "claim", pause_after_claim)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(queue.execute_once, processor, worker_id="paused-worker")
        try:
            assert claimed.wait(timeout=5)
            # If claim were committed before slot registration, this lock could
            # be acquired and another dispatcher could exceed the physical cap.
            unexpectedly_unlocked = store._mutex.acquire(blocking=False)
            if unexpectedly_unlocked:
                store._mutex.release()
            assert unexpectedly_unlocked is False
            now[0] += 6
            second = executor.submit(queue.execute_once, processor, worker_id="other-worker")
        finally:
            release.set()
        assert first.result(timeout=5)["state"] == "failed"
        second.result(timeout=5)
    assert "paused-worker" not in processor_calls
