"""Read-only task report and explicit-loss export routes (G01/G09 surfaces).

These routes must describe structure only, never certify semantic review, and
must carry the binding revision/content digest so a stale export is detectable.
"""
from __future__ import annotations

import json

from tests.test_workspace_server import api, assert_error, command, http_video, request


def _task_path(store, suffix):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    return f"/api/v1/projects/current/tasks/{task['id']}/{suffix}"


def test_report_on_empty_task_reports_structure_only(api):
    server, store = api
    status, headers, body = request(server, "GET", _task_path(store, "report"))
    assert status == 200, body
    assert body["kind"] == "traceannotate.task-structural-report"
    assert body["scope"] == "structure_and_video_interval_statistics_only"
    assert body["semantic_review"] == "not_performed"
    assert body["checks"]["status"] == "passed"
    assert body["counts"]["annotations"] == 0
    # Unannotated duration must appear as uncovered, never as Idle.
    assert any("Gaps mean unannotated" in item for item in body["limitations"])
    assert body["binding"]["task_id"] == store.list_tasks()[0]["task_id"]
    assert headers["Cache-Control"] == "no-store"


def test_report_after_commit_counts_segment_and_coverage(api):
    server, store = api
    path, body = command(store)
    assert request(server, "POST", path, body)[0] == 200
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200
    assert report["counts"]["segments"] == 1
    track = report["tracks"][0]
    assert track["segment_count"] == 1
    # The committed segment spans [0,1) on the active clock.
    assert track["covered_seconds"]["n"] in ("1",) or float(track["covered_seconds"]["n"]) > 0


def test_export_json_carries_binding_and_loss_disclosure(api):
    server, store = api
    path, body = command(store)
    request(server, "POST", path, body)
    status, headers, payload = request(server, "GET", _task_path(store, "export?format=json"))
    assert status == 200
    task = store.get_task(store.list_tasks()[0]["task_id"])
    assert headers["X-TraceAnnotate-Task-Revision"] == str(task["revision"])
    assert headers["X-TraceAnnotate-Content-SHA256"] == task["content_sha256"]
    assert "No media bytes" in headers["X-TraceAnnotate-Export-Losses"]
    assert payload["kind"] == "traceannotate.task-snapshot"
    assert payload["binding"]["task_id"] == task["id"]
    assert payload["task"]["annotations"][0]["label"] == "working"


def test_export_csv_and_srt_roundtrip_one_segment(api):
    server, store = api
    path, body = command(store)
    request(server, "POST", path, body)
    status, headers, csv_body = request(server, "GET", _task_path(store, "export?format=csv"))
    assert status == 200 and "text/csv" in headers["Content-Type"]
    text = csv_body.decode("utf-8-sig")
    assert "task_id" in text.splitlines()[0] and "ann-first" in text
    status, headers, srt_body = request(server, "GET", _task_path(store, "export?format=srt"))
    assert status == 200
    srt = srt_body.decode("utf-8")
    assert "-->" in srt and "reviewed" in srt


def test_export_rejects_missing_or_unknown_format_and_unauthorized(api):
    server, store = api
    task_url = "/api/v1/projects/current/tasks/" + store.list_tasks()[0]["task_id"]
    assert_error(request(server, "GET", task_url + "/export"), 400, "INVALID_QUERY")
    assert_error(request(server, "GET", task_url + "/export?format=parquet"),
                 422, "UNSUPPORTED_EXPORT_FORMAT")
    assert_error(request(server, "GET", task_url + "/export?format=json", authorized=False),
                 403, "INVALID_TOKEN")
