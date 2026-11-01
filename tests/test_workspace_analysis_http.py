"""Actual loopback analysis jobs: measured playback candidates and atomic saves.

Short media is encoded and measured by the installed FFmpeg tools. Gates only
control dispatch/cancellation boundaries; they never supply fabricated results.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.workspace import analysis_service as A
from trace_annotate.workspace import professional_schema as PS
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.analysis_jobs import AnalysisJobs
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.http_media import resolve_media_object
from trace_annotate.workspace.server import make_server, TOKEN_HEADER
from trace_annotate.workspace.store import WorkspaceStore
from tests.test_workspace_server import request, assert_error

API = "/api/v1/projects/current"


@pytest.fixture(scope="module")
def measured_clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Real objective media analysis requires FFmpeg and ffprobe")
    source = tmp_path_factory.mktemp("analysis-http-source") / "stereo.mov"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc2=duration=2:size=32x32:rate=10", "-f", "lavfi", "-i",
                    "aevalsrc=0.5*sin(2*PI*997*t)|0.25*sin(2*PI*499*t):s=48000:d=2:c=stereo",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "pcm_s24le", str(source)], check=True, capture_output=True, timeout=30)
    return source


@pytest.fixture()
def analysis_store(tmp_path, measured_clip):
    store = WorkspaceStore(tmp_path / "workspace")
    store.create()
    store.import_source(measured_clip)
    try:
        yield store
    finally:
        store.close()


@contextmanager
def serving(store, monkeypatch=None, *, automatic=False):
    # A disabled auto-start exposes stable queued states; run_pending below
    # still invokes the real production processor and FFmpeg measurement.
    if automatic:
        server = make_server(store)
    else:
        assert monkeypatch is not None
        with monkeypatch.context() as patch:
            patch.setattr(AnalysisJobs, "start", lambda self: self)
            server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        assert not thread.is_alive()


def task(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


def submission(store, command_id="cmd-analysis", **changes):
    current = task(store)
    body = {"command_id": command_id, "expected_revision": current["revision"],
            "expected_sha256": current["content_sha256"], "stream_index": None,
            "window": None, "subject_annotation_id": None}
    body.update(changes)
    return API + "/tasks/" + current["id"] + "/analysis-jobs", body


def apply_body(store, command_id="cmd-apply", **changes):
    current = task(store)
    result = {"command_id": command_id, "expected_revision": current["revision"],
              "expected_sha256": current["content_sha256"], "operation": "apply_analysis_job", "payload": {}}
    result.update(changes)
    return result


def submitted(server, store, command_id="cmd-analysis", **changes):
    path, body = submission(store, command_id, **changes)
    response = request(server, "POST", path, body)
    assert response[0] == 202, response
    batch = response[2]
    assert batch["batch_id"] == batch["id"] and len(batch["jobs"]) == 1
    job = batch["jobs"][0]
    assert job["id"] == job["job_id"]
    assert job["state"] == "queued"
    return job, path, body, batch


def completed(server, store, command_id="cmd-analysis", **changes):
    job, path, body, batch = submitted(server, store, command_id, **changes)
    runs = server.analysis_jobs.run_pending(batch_id=batch["id"])
    assert len(runs) == 1 and runs[0]["state"] == "succeeded", runs
    response = request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result")
    assert response[0] == 200, response
    return job, response[2]


def store_reports(store, *, where="bulk-analysis"):
    current = task(store)
    media = store.get_media(current["media_id"])
    playback, _, _ = resolve_media_object(store, media["id"], "playback_path")
    reports = A.analyze_media(playback, media, store.root.parent / where,
                              time_domain="playback_seconds")["reports"]
    assert len(reports) >= 4
    return reports


def test_bulk_reports_commit_once_and_replay_exactly(analysis_store):
    store = analysis_store
    before = task(store)
    reports = store_reports(store)
    arguments = dict(command_id="cmd-bulk", task_id=before["id"], expected_revision=before["revision"],
                     expected_sha256=before["content_sha256"], operation="set_analysis_reports",
                     payload={"reports": reports}, request_body=b"actual-ffmpeg-bulk")
    first = store.apply_task_command(**arguments)["response"]
    assert first["revision"] == before["revision"] + 1 and first["report_count"] == len(reports)
    after = task(store)
    assert after["annotations"] == before["annotations"]
    assert after["extensions"]["professional"]["analysis_reports"] == reports
    assert store.apply_task_command(**arguments)["response"] == first
    assert task(store) == after
    assert store.conn.execute("SELECT COUNT(*) FROM task_revisions WHERE task_id=?", (after["id"],)).fetchone()[0] == 2


@pytest.mark.parametrize("invalid", ["wrong_file", "duplicate", "identity_collision"])
def test_bulk_report_failure_is_atomic_even_after_a_valid_first_report(analysis_store, invalid):
    store = analysis_store
    reports = store_reports(store)
    if invalid == "identity_collision":
        current = task(store)
        store.apply_task_command(command_id="cmd-existing-report", task_id=current["id"], expected_revision=current["revision"],
            expected_sha256=current["content_sha256"], operation="set_analysis_report", payload={"report": reports[-1]}, request_body=b"seed")
    before = task(store)
    values = copy.deepcopy(reports)
    if invalid == "wrong_file":
        values[-1]["file_sha256"] = "0" * 64
        values[-1]["input_digest"] = PS.analysis_input_digest(values[-1])
    elif invalid == "duplicate":
        values.append(copy.deepcopy(values[0]))
    else:
        values[-1]["method"]["parameters"]["changed-result"] = True
        values[-1]["input_digest"] = PS.analysis_input_digest(values[-1])
    with pytest.raises(WorkspaceError):
        store.apply_task_command(command_id="cmd-atomic-invalid", task_id=before["id"], expected_revision=before["revision"],
            expected_sha256=before["content_sha256"], operation="set_analysis_reports", payload={"reports": values}, request_body=b"invalid")
    assert task(store) == before
    with pytest.raises(WorkspaceError) as missing:
        store.get_command_receipt("cmd-atomic-invalid")
    assert missing.value.code == "COMMAND_UNKNOWN"


def test_http_analysis_candidate_is_not_saved_until_explicit_atomic_apply(analysis_store, monkeypatch):
    store = analysis_store
    before = task(store)
    media = store.get_media(before["media_id"])
    with serving(store, monkeypatch) as server:
        status = request(server, "GET", API + "/tasks/" + before["id"] + "/analysis-status")
        assert status[0] == 200 and status[2]["reports"] == [] and status[2]["assessment"] == []
        job, candidate = completed(server, store)
        assert task(store) == before
        assert candidate["kind"] == "traceannotate.analysis-candidate"
        assert candidate["file_sha256"] == media["playback"]["sha256"]
        assert candidate["file_sha256"] != media["source"]["sha256"]
        assert candidate["time_domain"] == "playback_seconds"
        assert candidate["semantic_listening"] == "not_performed"
        assert "original_source_not_measured" in candidate["scope"]
        assert candidate["input_revision"] == before["revision"] and candidate["input_sha256"] == before["content_sha256"]
        for report in candidate["reports"]:
            PS.validate_analysis(report, media)
            assert report["file_sha256"] == media["playback"]["sha256"]
        audio = next(r for r in candidate["reports"] if r["kind"] == "loudness")
        assert next(v for v in audio["values"] if v["name"] == "sample_peak_dbfs")["status"] == "available"
        body = apply_body(store)
        applied = request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", body)
        assert applied[0] == 200 and applied[2]["report_count"] == len(candidate["reports"]), applied
        assert applied[2]["revision"] == before["revision"] + 1
        assert request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", body)[2] == applied[2]
        after = task(store)
        assert after["annotations"] == before["annotations"]
        assert after["extensions"]["professional"]["analysis_reports"] == candidate["reports"]
        exported = request(server, "GET", API + "/tasks/" + after["id"] + "/export?format=professional")
        assert exported[0] == 200 and exported[2]["analyses"] == candidate["reports"]
        assert all(item["status"] == "fresh" for item in exported[2]["analysis_assessment"])
        assert request(server, "GET", API + "/tasks/" + after["id"])[2] == after
        status = request(server, "GET", API + "/tasks/" + after["id"] + "/analysis-status")
        assert status[0] == 200 and status[2]["reports"] == candidate["reports"]
        assert status[2]["revision"] == after["revision"] and status[2]["content_sha256"] == after["content_sha256"]
        assert all(r["time_domain"] == "playback_seconds" for r in status[2]["reports"])
        assert all(a["status"] == "fresh" for a in status[2]["assessment"])


def test_http_analysis_submission_response_loss_replays_one_job_and_receipt(analysis_store, monkeypatch):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        path, body = submission(store)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("POST", path, body=json.dumps(body).encode(), headers={TOKEN_HEADER: server.token,
                "Origin": server.origin, "Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == 202
            response.close()
        finally:
            connection.close()
        receipt = request(server, "GET", API + "/commands/" + body["command_id"])
        assert receipt[0] == 200 and receipt[2]["state"] == "committed"
        replay = request(server, "POST", path, body)
        assert replay[0] == 202 and replay[2] == receipt[2]["response"]
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        changed = dict(body, stream_index=0)
        assert_error(request(server, "POST", path, changed), 409, "IDEMPOTENCY_CONFLICT")


def test_http_analysis_not_ready_cancelled_job_has_no_result_or_task_mutation(analysis_store, monkeypatch):
    store = analysis_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        job, _, _, _ = submitted(server, store)
        resultpath = API + "/jobs/" + job["id"] + "/analysis-result"
        assert_error(request(server, "GET", resultpath), 409, "ANALYSIS_NOT_READY")
        control = {"command_id": "cmd-cancel", "action": "cancel"}
        response = request(server, "POST", API + "/jobs/" + job["id"] + "/control", control)
        assert response[0] == 200 and response[2]["state"] == "cancelled"
        assert request(server, "POST", API + "/jobs/" + job["id"] + "/control", control)[2] == response[2]
        assert server.analysis_jobs.run_pending() == []
        assert_error(request(server, "GET", resultpath), 409, "ANALYSIS_NOT_READY")
        assert_error(request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_body(store)), 409, "ANALYSIS_NOT_READY")
        assert task(store) == before


@pytest.mark.parametrize("endpoint", ["submit", "result", "apply"])
def test_analysis_http_routes_preserve_private_token_and_exact_origin(analysis_store, monkeypatch, endpoint):
    store = analysis_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        create, body = submission(store)
        method, path = "POST", create
        if endpoint == "result":
            method, path, body = "GET", API + "/jobs/job-missing/analysis-result", None
        elif endpoint == "apply":
            path, body = API + "/jobs/job-missing/apply-analysis", apply_body(store)
        assert_error(request(server, method, path, body, authorized=False), 403, "INVALID_TOKEN")
        assert_error(request(server, method, path, body, headers={TOKEN_HEADER: "wrong"}), 403, "INVALID_TOKEN")
        assert_error(request(server, method, path, body, headers={"Origin": "https://attacker.example"}), 403, "FORBIDDEN_ORIGIN")
        assert_error(request(server, method, path, body, headers={"Host": "attacker.example"}), 403, "FORBIDDEN_ORIGIN")
        assert task(store) == before
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("change", [
    {"source_path": "C:/Users/private/secret.mp4"}, {"time_domain": "original_source_seconds"},
    {"stream_index": True}, {"stream_index": -1}, {"expected_revision": True},
    {"subject_annotation_id": "ann-absent"},
    {"window": {"start": {"n": "1", "d": "1"}, "end": {"n": "0", "d": "1"}}},
    {"window": {"start": {"n": "0", "d": "1"}, "end": {"n": "99", "d": "1"}}},
])
def test_analysis_submission_rejects_untrusted_paths_domains_and_invalid_bounds_without_enqueue(analysis_store, monkeypatch, change):
    store = analysis_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        path, body = submission(store, **change)
        assert_error(request(server, "POST", path, body), 422, "SCHEMA_INVALID")
        assert task(store) == before
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert_error(request(server, "GET", API + "/commands/" + body["command_id"]), 404, "COMMAND_UNKNOWN")


@pytest.mark.parametrize("raw", [b'{"command_id":"a","command_id":"b"}', b'{"stream_index":NaN}', b'[]', b'null'])
def test_analysis_submission_strict_json_and_object_envelope(analysis_store, monkeypatch, raw):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        path, _ = submission(store)
        result = request(server, "POST", path, raw)
        assert result[0] == (400 if raw.startswith(b'{') else 422), result
        assert result[2]["error"]["code"] == ("INVALID_JSON" if raw.startswith(b'{') else "SCHEMA_INVALID")
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_analysis_enqueue_and_apply_require_correct_current_cas(analysis_store, monkeypatch):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        path, body = submission(store)
        assert_error(request(server, "POST", path, dict(body, expected_revision=99)), 409, "REVISION_CONFLICT")
        assert_error(request(server, "POST", path, dict(body, expected_sha256="0" * 64)), 409, "MEDIA_CHANGED")
        job, candidate = completed(server, store)
        first = task(store)
        changed = request(server, "POST", API + "/tasks/" + first["id"] + "/commands", {
            "command_id": "cmd-unrelated-label", "expected_revision": first["revision"], "expected_sha256": first["content_sha256"],
            "operation": "add_annotation", "payload": {"annotation": {"id": "ann-note", "type": "whole_clip", "label": "review-note"}}})
        assert changed[0] == 200
        stale = apply_body(store, expected_revision=first["revision"], expected_sha256=first["content_sha256"])
        applypath = API + "/jobs/" + job["id"] + "/apply-analysis"
        assert_error(request(server, "POST", applypath, stale), 409, "REVISION_CONFLICT")
        # A fresh explicit CAS can save file reports after unrelated edits; the
        # candidate keeps its original input revision/hash rather than lying.
        fresh = request(server, "POST", applypath, apply_body(store))
        assert fresh[0] == 200 and fresh[2]["revision"] == first["revision"] + 2
        assert request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result")[2] == candidate
        assert task(store)["annotations"][0]["id"] == "ann-note"


def test_analysis_bound_segment_change_rejects_entire_bundle_until_remeasured(analysis_store, monkeypatch):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        before = task(store)
        response = request(server, "POST", API + "/tasks/" + before["id"] + "/commands", {
            "command_id": "cmd-segment", "expected_revision": before["revision"], "expected_sha256": before["content_sha256"],
            "operation": "add_annotation", "payload": {"annotation": {"id": "ann-subject", "type": "segment", "label": "audio",
                "start": {"seconds": S.rational(0)}, "end": {"seconds": S.rational(1)}}}})
        assert response[0] == 200
        window = {"start": S.rational(0), "end": S.rational(1)}
        job, _ = completed(server, store, subject_annotation_id="ann-subject", window=window)
        before = task(store)
        assert request(server, "POST", API + "/tasks/" + before["id"] + "/commands", {
            "command_id": "cmd-change-boundary", "expected_revision": before["revision"], "expected_sha256": before["content_sha256"],
            "operation": "update_annotation", "payload": {"annotation_id": "ann-subject", "changes": {"end": {"seconds": S.rational("1.5")}}}})[0] == 200
        after_edit = task(store)
        assert_error(request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_body(store)), 409, "ANALYSIS_STALE")
        assert task(store) == after_edit and "professional" not in task(store)["extensions"]
        assert_error(request(server, "GET", API + "/commands/cmd-apply"), 404, "COMMAND_UNKNOWN")


def test_analysis_candidate_object_tamper_fails_closed_without_applying_any_report(analysis_store, monkeypatch):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        job, _ = completed(server, store)
        before = task(store)
        committed = server.queue.get_job(job["id"])["commit"]
        artifact = store.root / committed["artifact"]["relative_path"]
        original = artifact.read_bytes()
        try:
            artifact.write_bytes(original + b" ")
            assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result"), 500, "RECOVERY_REQUIRED")
            assert_error(request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_body(store)), 500, "RECOVERY_REQUIRED")
            assert task(store) == before
            assert_error(request(server, "GET", API + "/commands/cmd-apply"), 404, "COMMAND_UNKNOWN")
        finally:
            artifact.write_bytes(original)


@pytest.mark.parametrize("change", [{"operation": "set_analysis_reports"}, {"payload": {"reports": []}}, {"document": {}}, {"task_id": "tsk-other"}])
def test_analysis_apply_never_accepts_client_report_override_or_wrong_command(analysis_store, monkeypatch, change):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        job, _ = completed(server, store)
        before = task(store)
        response = request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_body(store, **change))
        assert response[0] == 422 and response[2]["error"]["code"] in {"SCHEMA_INVALID", "UNSUPPORTED_OPERATION"}, response
        assert task(store) == before


def test_analysis_running_cancel_is_cooperative_and_never_creates_a_candidate(analysis_store, monkeypatch):
    store = analysis_store
    before = task(store)
    entered, release = threading.Event(), threading.Event()
    original = A.analyze_media
    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(10), "test release gate was not reached"
        return original(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", gated)
    with serving(store, monkeypatch) as server:
        job, _, _, batch = submitted(server, store)
        errors = []
        def run():
            try:
                server.analysis_jobs.run_pending(batch_id=batch["id"])
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert entered.wait(10)
            result = request(server, "POST", API + "/jobs/" + job["id"] + "/control", {"command_id": "cmd-cancel-running", "action": "cancel"})
            assert result[0] == 200 and result[2]["state"] == "cancel_requested"
        finally:
            release.set()
            worker.join(timeout=10)
        assert not worker.is_alive() and not errors
        status, _, terminal = request(server, "GET", API + "/jobs/" + job["id"])
        assert status == 200 and terminal["state"] == "cancelled" and terminal["commit"] is None
        assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result"), 409, "ANALYSIS_NOT_READY")
        assert task(store) == before and not P.live_media_processes()


def test_analysis_transient_failure_retry_measures_real_media_once_and_keeps_one_job(analysis_store, monkeypatch):
    store = analysis_store
    original = A.analyze_media
    attempts = []
    def fail_once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise WorkspaceError("ANALYSIS_FAILED", "Injected pre-measurement transient dependency failure", retryable=True)
        return original(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", fail_once)
    with serving(store, monkeypatch) as server:
        job, _, _, batch = submitted(server, store)
        failed = server.analysis_jobs.run_pending(batch_id=batch["id"])
        assert failed[0]["state"] == "failed" and failed[0]["error_code"] == "ANALYSIS_FAILED"
        before = task(store)
        retry = {"command_id": "cmd-retry", "action": "retry_failed"}
        response = request(server, "POST", API + "/jobs/" + job["id"] + "/control", retry)
        assert response[0] == 200 and response[2]["state"] == "queued"
        assert request(server, "POST", API + "/jobs/" + job["id"] + "/control", retry)[2] == response[2]
        runs = server.analysis_jobs.run_pending(batch_id=batch["id"])
        assert runs[0]["state"] == "succeeded" and runs[0]["attempt"] == 2
        candidate = request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result")
        assert candidate[0] == 200
        assert any(report["kind"] == "loudness" and report["status"] == "available" for report in candidate[2]["reports"])
        assert task(store) == before
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert len(attempts) == 2


def test_analysis_automatic_drainer_reaches_a_real_candidate_and_closes(analysis_store):
    store = analysis_store
    with serving(store, automatic=True) as server:
        path, body = submission(store)
        result = request(server, "POST", path, body)
        assert result[0] == 202
        ident = result[2]["jobs"][0]["id"]
        deadline = time.monotonic() + 20
        until_next_poll = threading.Event()
        while True:
            status, _, job = request(server, "GET", API + "/jobs/" + ident)
            assert status == 200
            if job["state"] in {"succeeded", "failed", "cancelled"}:
                break
            assert time.monotonic() < deadline, job
            until_next_poll.wait(.02)
        assert job["state"] == "succeeded", job
        assert request(server, "GET", API + "/jobs/" + ident + "/analysis-result")[0] == 200
    assert server.analysis_jobs.close(timeout=1)["stopped"]
    assert not P.live_media_processes()


def test_analysis_status_distinguishes_source_and_playback_and_reassesses_bound_windows(analysis_store, measured_clip, monkeypatch):
    store = analysis_store
    before = task(store)
    media = store.get_media(before["media_id"])
    source_result = A.analyze_media(measured_clip, media, store.root.parent / "original-source-report",
                                     time_domain="admitted_source_seconds")
    source_report = next(r for r in source_result["reports"] if r["kind"] == "loudness")
    assert source_report["file_sha256"] == media["source"]["sha256"]
    with serving(store, monkeypatch) as server:
        seed = {"command_id": "cmd-source-report", "expected_revision": before["revision"],
                "expected_sha256": before["content_sha256"], "report": source_report}
        assert request(server, "POST", API + "/tasks/" + before["id"] + "/analysis-reports", seed)[0] == 200
        current = task(store)
        assert request(server, "POST", API + "/tasks/" + current["id"] + "/commands", {
            "command_id": "cmd-status-segment", "expected_revision": current["revision"], "expected_sha256": current["content_sha256"],
            "operation": "add_annotation", "payload": {"annotation": {"id": "ann-status", "type": "segment", "label": "original label",
                "start": {"seconds": S.rational(0)}, "end": {"seconds": S.rational(1)}}}})[0] == 200
        job, candidate = completed(server, store, window={"start": S.rational(0), "end": S.rational(1)}, subject_annotation_id="ann-status")
        assert request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_body(store))[0] == 200
        original_saved = copy.deepcopy(task(store)["extensions"]["professional"]["analysis_reports"])
        url = API + "/tasks/" + current["id"] + "/analysis-status"
        status = request(server, "GET", url)
        assert status[0] == 200
        assert {r["time_domain"] for r in status[2]["reports"]} == {"admitted_source_seconds", "playback_seconds"}
        assert all(a["status"] == "fresh" for a in status[2]["assessment"])
        for changes, expected in [({"label": "new label"}, "fresh"), ({"end": {"seconds": S.rational("1.5")}}, "stale")]:
            current = task(store)
            changed = request(server, "POST", API + "/tasks/" + current["id"] + "/commands", {
                "command_id": "cmd-status-" + expected, "expected_revision": current["revision"], "expected_sha256": current["content_sha256"],
                "operation": "update_annotation", "payload": {"annotation_id": "ann-status", "changes": changes}})
            assert changed[0] == 200
            status = request(server, "GET", url)
            current = task(store)
            assert status[2]["reports"] == original_saved
            assert status[2]["revision"] == current["revision"] and status[2]["content_sha256"] == current["content_sha256"]
            assessed = {a["id"]: a for a in status[2]["assessment"]}
            for report in candidate["reports"]:
                assert assessed[report["id"]]["status"] == (expected if report["subject_annotation_id"] else "fresh")
            assert assessed[source_report["id"]]["status"] == "fresh"


def test_analysis_close_reports_a_still_running_owner_until_it_actually_stops(analysis_store, monkeypatch):
    store = analysis_store
    entered, release = threading.Event(), threading.Event()
    original = A.analyze_media
    def deliberately_blocked(*args, **kwargs):
        # Deliberately do not honor cancellation while held: a close result
        # must reveal this owner, not claim that a stop request killed it.
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", deliberately_blocked)
    with serving(store, monkeypatch) as server:
        job, _, _, batch = submitted(server, store)
        worker = threading.Thread(target=server.analysis_jobs.run_pending, kwargs={"batch_id": batch["id"]})
        worker.start()
        try:
            assert entered.wait(10)
            closing = server.analysis_jobs.close(timeout=0)
            assert closing["stopped"] is False and closing["active_jobs"] == [job["id"]]
            assert worker.is_alive()
            # The caller still owns an OPEN store until the actual worker exit.
            assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
            path, body = submission(store, command_id="cmd-after-closing")
            assert_error(request(server, "POST", path, body), 409, "WORKSPACE_CLOSING")
            # Both lifecycle layers must preserve the still-owned DB/lock. A
            # stopped=False report cannot be followed by silent store closure.
            server.shutdown()
            with monkeypatch.context() as patch:
                patch.setattr("trace_annotate.workspace.server.CLOSE_WORKER_SECONDS", .01)
                with pytest.raises(WorkspaceError) as busy_server:
                    server.server_close()
            assert busy_server.value.code == "WORKSPACE_BUSY"
            assert server.shutdown_report["analysis_worker"]["stopped"] is False
            with pytest.raises(WorkspaceError) as busy_store:
                store.close()
            assert busy_store.value.code == "WORKSPACE_BUSY"
            assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        finally:
            release.set()
            worker.join(timeout=10)
        assert not worker.is_alive()
        closed = server.analysis_jobs.close(timeout=5)
        assert closed["stopped"] is True and closed["active_jobs"] == []
        terminal = server.queue.get_job(job["id"])
        assert terminal["state"] == "cancelled" and terminal["commit"] is None
        assert not P.live_media_processes()


def test_analysis_server_shutdown_waits_for_cooperative_runner_and_records_actual_stop(analysis_store, monkeypatch):
    store = analysis_store
    entered = threading.Event()
    original = A.analyze_media
    def cancellable_gate(*args, **kwargs):
        entered.set()
        next_poll = threading.Event()
        deadline = time.monotonic() + 10
        while not kwargs["cancelled"]():
            assert time.monotonic() < deadline
            next_poll.wait(.01)
        return original(*args, **kwargs)
    monkeypatch.setattr(A, "analyze_media", cancellable_gate)
    worker = None
    with serving(store, monkeypatch) as server:
        job, _, _, batch = submitted(server, store)
        worker = threading.Thread(target=server.analysis_jobs.run_pending, kwargs={"batch_id": batch["id"]})
        worker.start()
        assert entered.wait(10)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert server.shutdown_report["analysis_worker"]["stopped"] is True
    assert server.shutdown_report["analysis_worker"]["active_jobs"] == []
    assert server.queue.get_job(job["id"])["state"] == "cancelled"
    assert not P.live_media_processes()


@pytest.mark.parametrize("save", [False, True])
def test_analysis_jobs_and_saved_reports_survive_real_store_reopen(analysis_store, monkeypatch, save):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        job, path, submitted_body, batch = submitted(server, store)
        apply_request = None
        applied = None
        if save:
            assert server.analysis_jobs.run_pending(batch_id=batch["id"])[0]["state"] == "succeeded"
            apply_request = apply_body(store)
            applied = request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_request)
            assert applied[0] == 200
        saved = copy.deepcopy(task(store))
    store.close()
    store.open()
    with serving(store, monkeypatch) as server:
        assert task(store) == saved
        replay = request(server, "POST", path, submitted_body)
        assert replay[0] == 202 and replay[2] == batch
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        if not save:
            assert server.queue.get_job(job["id"])["state"] == "queued"
            assert server.analysis_jobs.run_pending(batch_id=batch["id"])[0]["state"] == "succeeded"
        else:
            assert request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_request)[2] == applied[2]
            assert task(store) == saved
        candidate = request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result")
        assert candidate[0] == 200
        assert candidate[2]["time_domain"] == "playback_seconds"
        assert all(Path(item["relative_path"]).is_absolute() is False for item in candidate[2]["evidence"])


def test_analysis_routes_reject_oversize_and_unsupported_query_without_dispatch(analysis_store, monkeypatch):
    store = analysis_store
    with serving(store, monkeypatch) as server:
        path, _ = submission(store)
        oversized = b'{"padding":"' + b"x" * (1024 * 1024) + b'"}'
        assert_error(request(server, "POST", path, oversized), 413, "LIMIT_EXCEEDED")
        # The established server path policy rejects query-bearing routes
        # outside its explicit query allow-list as NOT_FOUND.
        assert_error(request(server, "POST", path + "?source=C:/secret.mp4", b"{}"), 404, "NOT_FOUND")
        assert_error(request(server, "GET", API + "/jobs/job-missing/analysis-result?path=../../secret"), 404, "NOT_FOUND")
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_analysis_routes_do_not_dispatch_unrelated_processor_jobs(analysis_store, monkeypatch):
    store = analysis_store
    current = task(store)
    with serving(store, monkeypatch) as server:
        response = request(server, "POST", API + "/batches", {"command_id": "cmd-unrelated", "items": [{"task_id": current["id"],
            "expected_revision": current["revision"], "expected_sha256": current["content_sha256"]}], "operation": "other-processor", "processor_version": "1", "parameters": {}})
        assert response[0] == 200
        job = response[2]["jobs"][0]
        assert server.analysis_jobs.run_pending() == []
        assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/analysis-result"), 409, "UNSUPPORTED_PROCESSOR")
        assert_error(request(server, "POST", API + "/jobs/" + job["id"] + "/apply-analysis", apply_body(store)), 409, "UNSUPPORTED_PROCESSOR")
        assert task(store) == current
