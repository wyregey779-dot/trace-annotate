"""Real durable job/child/slot boundaries; controlled outputs test queue logic.

The actual meter is exercised by test_real_playback_job. Controlled processors
in other tests establish persistence/cancellation, not audiovisual quality.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
import threading
import time

import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.workspace import analysis_jobs as J
from trace_annotate.workspace import analysis_service as A
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import Conflict, WorkspaceError
from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.slots import get_prepare_slot
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for admitted-playback fixtures")
    source = tmp_path_factory.mktemp("analysis-jobs-source") / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=blue:s=32x32:r=10:d=0.6",
                    "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=0.6",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)],
                   check=True, capture_output=True, timeout=30)
    return source


@pytest.fixture
def store(tmp_path, clip):
    project = WorkspaceStore(tmp_path / "workspace")
    project.create()
    try:
        project.import_source(clip)
        yield project
    finally:
        project.close()


def task(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


def submit(jobs, *, command_id="analyze-first", **kwargs):
    current = task(jobs.store)
    return jobs.submit(current["id"], command_id=command_id, expected_revision=current["revision"],
                       expected_sha256=jobs.store.task_digest(current), **kwargs)


def fake_analysis(source, media, output, **kwargs):
    assert kwargs["time_domain"] == "playback_seconds"
    output.mkdir(parents=True)
    (output / "private-log.txt").write_text(str(source), encoding="utf-8")
    values = [A._value("fixture", "controlled queue-only output", basis="declared")]
    reports = [A._report("technical_declared", media, "playback_seconds", None,
                         {"id": "controlled-test", "version": "1", "parameters": {}}, values)]
    if kwargs.get("window") is not None:
        reports.append(A._report("loudness", media, "playback_seconds", None,
            {"id": "controlled-test", "version": "1", "parameters": {}},
            [A._value("integrated_lufs", None, "LUFS", reason="controlled_test_unavailable")],
            kwargs["window"], kwargs["subject_annotation_id"]))
    return {"reports": reports}


def fail_code(code, call):
    with pytest.raises(WorkspaceError) as exc:
        call()
    assert exc.value.code == code


def test_real_playback_job_is_hash_bound_candidate_and_does_not_edit_task(store):
    jobs = J.AnalysisJobs(store)
    initial = task(store)
    queued = submit(jobs)["jobs"][0]
    assert queued["parameters"]["time_domain"] == "playback_seconds"
    fail_code("ANALYSIS_NOT_READY", lambda: jobs.result(queued["id"]))
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded"
    result = jobs.result(queued["id"])
    assert result["file_sha256"] == store.get_media(initial["media_id"])["playback"]["sha256"]
    assert result["scope"] == "registered_playback_only; original_source_not_measured; candidate_not_applied"
    assert all(report["time_domain"] == "playback_seconds" for report in result["reports"])
    assert any(report["kind"] == "loudness" for report in result["reports"])
    assert result["evidence"] and all(item["relative_path"].startswith("tmp/analysis/jobs/") for item in result["evidence"])
    assert str(store.root) not in json.dumps(result)
    assert task(store) == initial
    assert jobs.close()["stopped"]


def test_submit_is_idempotent_after_task_edit_and_shutdown(store):
    jobs = J.AnalysisJobs(store)
    current = task(store)
    arguments = dict(command_id="receipt", expected_revision=current["revision"], expected_sha256=store.task_digest(current))
    first = jobs.submit(current["id"], **arguments)
    add_segment(store)
    jobs.close()
    assert jobs.submit(current["id"], **arguments) == first
    fail_code("WORKSPACE_CLOSING", lambda: submit(jobs, command_id="new-after-close"))
    fail_code("IDEMPOTENCY_CONFLICT", lambda: jobs.submit(current["id"], stream_index=0, **arguments))


@pytest.mark.parametrize("kwargs", [{"stream_index": True}, {"stream_index": -1},
    {"window": {"start": S.rational(0), "end": S.rational(0)}},
    {"window": {"start": S.rational(0), "end": S.rational(999)}},
    {"subject_annotation_id": "not-a-segment"}])
def test_invalid_requests_leave_no_jobs_or_receipts(store, kwargs):
    jobs = J.AnalysisJobs(store)
    with pytest.raises(WorkspaceError):
        submit(jobs, **kwargs)
    assert jobs.queue.list_batches() == []
    assert store.conn.execute("SELECT COUNT(*) FROM command_receipts WHERE command_id='analyze-first'").fetchone()[0] == 0


def test_only_analysis_batches_are_dispatched_and_unrelated_jobs_stay_queued(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    other = jobs.queue.enqueue(task(store)["id"], operation="verify-media", processor_version="1")
    own = submit(jobs)["jobs"][0]
    assert [row["id"] for row in jobs.run_pending()] == [own["id"]]
    assert jobs.queue.get_job(other["id"])["state"] == "queued"
    fail_code("UNSUPPORTED_PROCESSOR", lambda: jobs.result(other["id"]))
    other_batch = next(batch for batch in jobs.queue.list_batches() if batch["jobs"][0]["id"] == other["id"])
    fail_code("UNSUPPORTED_PROCESSOR", lambda: jobs.run_pending(batch_id=other_batch["id"]))


def test_failure_retry_uses_unique_attempt_and_no_duplicate_task(store, monkeypatch):
    calls = []
    def failing_once(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 1:
            args[2].mkdir(parents=True)
            (args[2] / "failed.txt").write_text("kept")
            raise WorkspaceError("ANALYSIS_FAILED", "Controlled transient failure")
        return fake_analysis(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", failing_once)
    jobs = J.AnalysisJobs(store)
    queued = submit(jobs)["jobs"][0]
    assert jobs.run_pending()[0]["state"] == "failed"
    jobs.queue.control_job(queued["id"], "retry_failed", command_id="retry")
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded" and done["attempt"] == 2
    assert calls[0] != calls[1] and (calls[0] / "failed.txt").is_file()
    assert len(store.list_tasks()) == 1
    assert len(done["attempts"]) == 2 and jobs.result(queued["id"])


def test_queued_cancel_and_reopen_dispatch_only_remaining_jobs(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    first = submit(jobs)["jobs"][0]
    second = submit(jobs, command_id="second", stream_index=0)["jobs"][0]
    jobs.queue.cancel(first["id"])
    assert jobs.close()["stopped"]
    root = store.root
    store.close()
    reopened = WorkspaceStore(root); reopened.open()
    try:
        restored = J.AnalysisJobs(reopened)
        assert [r["id"] for r in restored.run_pending()] == [second["id"]]
        assert restored.queue.get_job(first["id"])["state"] == "cancelled"
        assert restored.result(second["id"])
        restored.close()
    finally:
        reopened.close()


def test_heartbeat_keeps_actual_one_second_queue_lease_alive(store, monkeypatch):
    def slow(*args, **kwargs):
        until = time.monotonic() + 1.6
        while time.monotonic() < until:
            assert not kwargs["cancelled"]()
            time.sleep(.02)
        return fake_analysis(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", slow)
    queue = DurableQueue(store, lease_seconds=1)
    jobs = J.AnalysisJobs(store, queue=queue)
    queued = submit(jobs)["jobs"][0]
    started = time.time()
    assert jobs.run_pending()[0]["state"] == "succeeded"
    assert queue.get_job(queued["id"])["heartbeat_at"] > started + 1


def test_lost_renewal_stops_processor_without_a_candidate(store, monkeypatch):
    def slow(*args, **kwargs):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            if kwargs["cancelled"]():
                raise InterruptedError("cancelled")
            time.sleep(.01)
        pytest.fail("Renewal failure must fence the processor")
    monkeypatch.setattr(A, "analyze_media", slow)
    queue = DurableQueue(store, lease_seconds=1)
    jobs = J.AnalysisJobs(store, queue=queue)
    submit(jobs)
    monkeypatch.setattr(queue, "heartbeat", lambda *a: (_ for _ in ()).throw(Conflict("STALE_LEASE", "Injected renewal failure")))
    done = jobs.run_pending()[0]
    assert done["state"] == "failed" and done["error_code"] == "LEASE_LOST" and done["commit"] is None


def controlled_child_analysis(source, media, output, **kwargs):
    output.mkdir(parents=True)
    P._run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-re", "-f", "lavfi", "-i",
            "anullsrc=r=48000:cl=stereo", "-t", "30", "-f", "null", "-"],
           output, "controlled-analysis-owned-child", kwargs["cancelled"])
    pytest.fail("Controlled child must be cancelled before completion")


@pytest.mark.parametrize("action", ["cancel", "close", "external_close"])
def test_running_owned_ffmpeg_is_stopped_and_no_false_success(store, monkeypatch, action):
    monkeypatch.setattr(A, "analyze_media", controlled_child_analysis)
    closing = threading.Event()
    jobs = J.AnalysisJobs(store, closing=closing)
    queued = submit(jobs)["jobs"][0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(jobs.run_pending)
        deadline = time.monotonic() + 5
        owned = []
        while time.monotonic() < deadline:
            owned = [p for p in P.live_media_processes() if p["name"] == "controlled-analysis-owned-child"]
            if owned:
                break
            time.sleep(.01)
        assert owned
        if action == "cancel":
            jobs.queue.cancel(queued["id"])
        elif action == "close":
            assert jobs.close(timeout=5)["stopped"]
        else:
            closing.set()
        done = running.result(timeout=6)[0]
    assert done["state"] == "cancelled" and done["commit"] is None
    assert not ({p["pid"] for p in owned} & {p["pid"] for p in P.live_media_processes()})
    commands = list((store.tmp_dir / "analysis/jobs" / queued["id"]).glob("*/commands.jsonl"))
    assert commands
    log = json.loads(commands[0].read_text("utf-8").splitlines()[-1])
    assert log["cancelled_or_interrupted"] and log["exit_code"] is not None
    assert jobs.close()["stopped"]


def test_shared_import_slot_keeps_unclaimed_analysis_queued_and_close_interrupts_wait(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store, slot_wait_seconds=10)
    own = submit(jobs)["jobs"][0]
    slot = get_prepare_slot(store.root)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with slot.hold():
            pending = executor.submit(jobs.run_pending)
            time.sleep(.1)
            assert jobs.queue.get_job(own["id"])["state"] == "queued"
            assert jobs.close(timeout=2)["stopped"]
            assert pending.result(timeout=2) == []
    assert jobs.queue.get_job(own["id"])["attempt"] == 0


def test_two_managers_share_single_physical_decode_slot(store, monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()
    def slow(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
        try:
            time.sleep(.1)
            return fake_analysis(*args, **kwargs)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(A, "analyze_media", slow)
    queue = DurableQueue(store, max_concurrency=2)
    first = J.AnalysisJobs(store, queue=queue)
    second = J.AnalysisJobs(store, queue=queue)
    one = submit(first)
    two = submit(first, command_id="second", stream_index=0)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(owner.run_pending, batch_id=batch["id"])
                   for owner, batch in ((first, one), (second, two))]
        assert all(f.result(timeout=5)[0]["state"] == "succeeded" for f in futures)
    assert peak == 1 and get_prepare_slot(store.root).stats()["peak"] == 1


@pytest.mark.parametrize("corruption", ["bytes", "durable", "path", "oversize", "missing", "directory"])
def test_result_reads_only_verified_bounded_registered_commit(store, monkeypatch, corruption):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    queued = submit(jobs)["jobs"][0]
    done = jobs.run_pending()[0]
    artifact = done["commit"]["artifact"]
    path = store.root / artifact["relative_path"]
    if corruption == "bytes":
        path.write_bytes(b"{}")
    elif corruption == "missing":
        path.unlink()
    elif corruption == "directory":
        path.unlink(); path.mkdir()
    elif corruption == "durable":
        store.conn.execute("UPDATE durable_commits SET body='{}' WHERE id=?", (done["commit"]["id"],))
    elif corruption == "path":
        store.conn.execute("UPDATE artifacts SET rel_path='../outside.json' WHERE id=?", (artifact["id"],))
    else:
        store.conn.execute("UPDATE artifacts SET size_bytes=? WHERE id=?", (J.MAX_RESULT_BYTES + 1, artifact["id"]))
    fail_code("RECOVERY_REQUIRED", lambda: jobs.result(queued["id"]))


def test_background_drainer_starts_once_and_truthfully_closes(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store, poll_seconds=.02)
    own = submit(jobs)["jobs"][0]
    jobs.start(); thread = jobs._drainer
    jobs.start()
    assert jobs._drainer is thread
    deadline = time.monotonic() + 5
    while jobs.queue.get_job(own["id"])["state"] != "succeeded" and time.monotonic() < deadline:
        time.sleep(.01)
    assert jobs.result(own["id"])
    assert jobs.close(timeout=2)["stopped"] and not thread.is_alive()


def add_segment(store):
    current = task(store)
    def boundary(index):
        return {"seconds": S.rational(index / 10)}
    ann = {"id": "segment-test", "type": "segment", "label": "kept", "text": "", "track": "main", "tags": [],
           "attributes": {}, "evidence_ids": [], "provenance": {"origin": "manual", "actor_id": "local-user", "job_id": None},
           "start": boundary(0), "end": boundary(3)}
    return store.apply_task_command(command_id="add-segment", task_id=current["id"], expected_revision=current["revision"],
        expected_sha256=store.task_digest(current), operation="add_annotation", payload={"annotation": ann}, request_body=b"segment")


def test_apply_bundle_is_explicit_atomic_and_replay_survives_later_edit(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs)["jobs"][0]
    jobs.run_pending()
    initial = task(store)
    arguments = dict(command_id="save-analysis", expected_revision=initial["revision"],
                     expected_sha256=store.task_digest(initial), request_body=b"explicit-save")
    saved = jobs.apply(own["id"], **arguments)
    assert saved["revision"] == initial["revision"] + 1
    assert task(store)["extensions"]["professional"]["analysis_reports"] == jobs.result(own["id"])["reports"]
    add_segment(store)
    assert jobs.apply(own["id"], **arguments) == saved
    fail_code("REVISION_CONFLICT", lambda: jobs.apply(own["id"], **dict(arguments, command_id="new-stale")))


def test_changed_subject_refuses_candidate_without_changing_window_or_annotations(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    add_segment(store)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs, window={"start": S.rational(0), "end": S.rational(.3)}, subject_annotation_id="segment-test")["jobs"][0]
    jobs.run_pending()
    current = task(store)
    store.apply_task_command(command_id="move-boundary", task_id=current["id"], expected_revision=current["revision"],
        expected_sha256=store.task_digest(current), operation="update_annotation",
        payload={"annotation_id": "segment-test", "changes": {"end": {"seconds": S.rational(.4)}}}, request_body=b"boundary")
    current = task(store)
    fail_code("ANALYSIS_STALE", lambda: jobs.apply(own["id"], command_id="stale-report", expected_revision=current["revision"],
        expected_sha256=store.task_digest(current), request_body=b"must-refuse"))
    assert task(store) == current and "professional" not in current.get("extensions", {})


def test_shutdown_timeout_never_claims_uncooperative_runner_stopped(store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return fake_analysis(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", slow)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs)["jobs"][0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(jobs.run_pending)
        assert entered.wait(timeout=3)
        try:
            outcome = jobs.close(timeout=.02)
            assert outcome["stopped"] is False and own["id"] in outcome["active_jobs"]
        finally:
            release.set()
        assert running.result(timeout=3)[0]["state"] == "cancelled"
    assert jobs.close(timeout=2)["stopped"]


def test_previous_session_running_job_recovers_through_existing_queue_rules(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs)["jobs"][0]
    previous = jobs.queue.claim("old-session")
    assert previous["id"] == own["id"]
    root = store.root
    store.close()
    reopened = WorkspaceStore(root); reopened.open()
    try:
        restored = J.AnalysisJobs(reopened)
        recovered = restored.queue.get_job(own["id"])
        assert recovered["state"] == "queued" and recovered["lease_generation"] > previous["lease_generation"]
        done = restored.run_pending()[0]
        assert done["state"] == "succeeded" and done["attempt"] == 2
        assert [row["state"] for row in done["attempts"]] == ["interrupted", "succeeded"]
        assert restored.result(own["id"])
        restored.close()
    finally:
        reopened.close()


def test_registered_playback_corruption_is_refused_before_analysis(store, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Corrupt playback must not reach an analyzer")
    monkeypatch.setattr(A, "analyze_media", forbidden)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs)["jobs"][0]
    media = store.get_media(task(store)["media_id"])
    path = store.root / ("objects/" + media["playback"]["sha256"])
    with path.open("ab") as out:
        out.write(b"controlled mutation")
    done = jobs.run_pending()[0]
    assert done["state"] == "failed" and done["error_code"] == "MEDIA_CHANGED" and done["commit"] is None


def test_frozen_task_snapshot_tampering_cannot_produce_trusted_candidate(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs)["jobs"][0]
    row = store.conn.execute("SELECT input FROM jobs WHERE id=?", (own["id"],)).fetchone()
    body = json.loads(row[0])
    body["task_snapshot"]["extensions"] = {"untrusted": "changed after enqueue"}
    store.conn.execute("UPDATE jobs SET input=? WHERE id=?", (S.canonical_json(body).decode(), own["id"]))
    done = jobs.run_pending()[0]
    assert done["state"] == "failed" and done["error_code"] == "RECOVERY_REQUIRED" and done["commit"] is None


def test_original_source_job_parameters_cannot_relabel_proxy(store, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Unsupported domain must be refused before measuring proxy")
    monkeypatch.setattr(A, "analyze_media", forbidden)
    jobs = J.AnalysisJobs(store)
    jobs.queue.enqueue(task(store)["id"], operation=J.OPERATION, processor_version=J.VERSION,
        parameters={"stream_index": None, "window": None, "subject_annotation_id": None, "time_domain": "original_source_seconds"})
    done = jobs.run_pending()[0]
    assert done["state"] == "failed" and done["error_code"] == "SCHEMA_INVALID" and done["commit"] is None


def test_analysis_and_actual_synchronous_import_share_decode_slot(store, monkeypatch, tmp_path):
    alternate = tmp_path / "alternate.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=32x32:r=5:d=0.4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(alternate)],
                   check=True, capture_output=True, timeout=30)
    entered, release = threading.Event(), threading.Event()
    def held(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return fake_analysis(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", held)
    jobs = J.AnalysisJobs(store)
    submit(jobs)
    with ThreadPoolExecutor(max_workers=2) as executor:
        analysis = executor.submit(jobs.run_pending)
        assert entered.wait(timeout=3)
        importing = executor.submit(store.import_source, alternate)
        try:
            time.sleep(.1)
            assert not importing.done() and len(store.list_tasks()) == 1
        finally:
            release.set()
        assert analysis.result(timeout=5)[0]["state"] == "succeeded"
        assert importing.result(timeout=15)["task_id"]
    assert len(store.list_tasks()) == 2 and get_prepare_slot(store.root).stats()["peak"] == 1


def test_subject_window_mismatch_rolls_back_enqueue(store):
    add_segment(store)
    jobs = J.AnalysisJobs(store)
    fail_code("SCHEMA_INVALID", lambda: submit(jobs, window={"start": S.rational(0), "end": S.rational(.4)},
                                                subject_annotation_id="segment-test"))
    assert jobs.queue.list_batches() == []


def test_analysis_candidate_is_kept_by_gc_after_reopen(store, monkeypatch):
    monkeypatch.setattr(A, "analyze_media", fake_analysis)
    jobs = J.AnalysisJobs(store)
    own = submit(jobs)["jobs"][0]
    done = jobs.run_pending()[0]
    artifact_id = done["commit"]["artifact"]["id"]
    assert artifact_id not in {row["id"] for row in store.unbound_artifacts()["artifacts"]}
    root = store.root
    jobs.close()
    store.close()
    reopened = WorkspaceStore(root); reopened.open()
    try:
        restored = J.AnalysisJobs(reopened)
        reopened.sweep_unbound_artifacts(apply=True)
        assert restored.result(own["id"])
        restored.close()
    finally:
        reopened.close()
