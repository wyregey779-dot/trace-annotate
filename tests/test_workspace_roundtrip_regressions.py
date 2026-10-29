"""Actual HTTP import/storage/restart/export regressions, without task patching."""
from __future__ import annotations

import copy
from fractions import Fraction
import pytest

from tests.test_workspace_server import api, command, http_video, request, running
from trace_annotate.workspace.legacy_adapter import temporal_import_payload, export_temporal_document
from trace_annotate.workspace.schema import rational, rational_value
from trace_annotate.workspace.store import WorkspaceStore
from trace_annotate.workspace.frames import FrameService
from trace_annotate.workspace import reporting


def _document(store, *, metadata=True):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    media = store.get_media(task["media_id"])
    doc = {"kind": "temporal", "schema_version": 2,
           # Identity is the hash, not the original display name.
           "clip": "源文件重命名😀.mp4",
           "duration": float(rational_value(media["duration"])),
           "media_size": int(media["source"]["size_bytes"]),
           "media_sha256": media["source"]["sha256"],
           "segments": [{"id": "中文ID-😀", "start": 0.1, "end": 0.9,
                         "label": "位置 -12,345.67", "text": "第一行\n第二行😀",
                         "track": "轨道 A", "tags": ["重复", "重复", "=1"],
                         "review_status": "unreviewed"}]}
    if metadata:
        doc["metadata"] = {"nested": {"number": -12345.67, "truth": True, "nothing": None,
                                     "items": ["中文", {"x": "-001.20"}, []]},
                           "ordered": ["b", "a", "b"], "empty": {}}
    return doc


def _import(server, store, doc, ident):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    url = f"/api/v1/projects/current/tasks/{task['id']}/import-temporal"
    return request(server, "POST", url, {"document": doc, "command_id": ident,
        "expected_revision": task["revision"], "expected_sha256": task["content_sha256"]})


def _export(server, store, format="temporal"):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    return request(server, "GET", f"/api/v1/projects/current/tasks/{task['id']}/export?format={format}")


@pytest.mark.parametrize("metadata_mode", ["nested", "empty", "absent"])
def test_http_metadata_and_source_envelope_survive_close_reopen(tmp_path, http_video, metadata_mode):
    root = tmp_path / "project"
    store = WorkspaceStore(root)
    store.create()
    store.import_source(http_video)
    doc = _document(store, metadata=metadata_mode != "absent")
    if metadata_mode == "empty":
        doc["metadata"] = {}
    original = copy.deepcopy(doc)
    try:
        with running(store) as server:
            status, _, receipt = _import(server, store, doc, "cmd-import-roundtrip")
            assert status == 200, receipt
            saved = store.get_task(store.list_tasks()[0]["task_id"])
            assert saved["legacy_document_metadata"] == doc.get("metadata", {})
            assert saved["extensions"]["temporal_source"]["metadata_present"] == ("metadata" in doc)
            assert store.task_digest(saved) == saved["content_sha256"]
    finally:
        store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        with running(reopened) as server:
            status, headers, exported = _export(server, reopened)
            assert status == 200, exported
            assert exported == original
            assert "Lossless" not in headers["X-TraceAnnotate-Export-Losses"]
            assert "not a complete workspace backup" in headers["X-TraceAnnotate-Export-Losses"]
            task = reopened.get_task(reopened.list_tasks()[0]["task_id"])
            assert headers["X-TraceAnnotate-Content-SHA256"] == task["content_sha256"]
            status, _, snapshot = _export(server, reopened, "json")
            assert status == 200
            assert snapshot["task"]["legacy_document_metadata"] == original.get("metadata", {})
            assert snapshot["task"]["extensions"]["temporal_source"]["clip"] == original["clip"]
    finally:
        reopened.close()
    assert doc == original


def test_import_replacement_clears_previous_metadata_without_retaining_other_document_data(api):
    server, store = api
    first = _document(store)
    assert _import(server, store, first, "cmd-first-metadata")[0] == 200
    second = _document(store, metadata=False)
    second["clip"] = "second.mp4"
    status, _, result = _import(server, store, second, "cmd-second-no-metadata")
    assert status == 200, result
    status, _, exported = _export(server, store)
    assert status == 200 and exported == second
    assert "metadata" not in exported


def test_later_human_edit_and_idempotent_import_keep_metadata_in_same_revision(api):
    server, store = api
    doc = _document(store)
    before = store.get_task(store.list_tasks()[0]["task_id"])
    url = f"/api/v1/projects/current/tasks/{before['id']}/import-temporal"
    body = {"document": doc, "command_id": "cmd-metadata-idempotent",
            "expected_revision": before["revision"], "expected_sha256": before["content_sha256"]}
    first = request(server, "POST", url, body)
    assert first[0] == 200, first[2]
    current = store.get_task(before["id"])
    path, change = command(store, "cmd-human-after-import")
    change.update(operation="update_annotation", payload={
        "annotation_id": current["annotations"][0]["id"], "changes": {"label": "人工修改"}})
    assert request(server, "POST", path, change)[0] == 200
    edited = store.get_task(before["id"])
    assert edited["legacy_document_metadata"] == doc["metadata"]
    assert request(server, "POST", url, body)[0] == 200
    assert store.get_task(before["id"]) == edited
    status, _, exported = _export(server, store)
    assert status == 200
    assert exported["metadata"] == doc["metadata"]
    assert exported["segments"][0]["label"] == "人工修改"


def test_temporal_export_discloses_non_segment_omissions_over_http(api):
    server, store = api
    for ident, kind in [("ann-point", "event"), ("ann-whole", "whole_clip")]:
        path, body = command(store, "cmd-" + ident)
        ann = body["payload"]["annotation"]
        ann.update(id=ident, type=kind)
        del ann["start"], ann["end"]
        if kind == "event":
            ann["at"] = {"seconds": rational(1)}
        status, _, result = request(server, "POST", path, body)
        assert status == 200, result
    before = store.get_task(store.list_tasks()[0]["task_id"])
    status, headers, projected = _export(server, store)
    assert status == 200, projected
    assert projected["segments"] == []
    loss = headers["X-TraceAnnotate-Export-Losses"]
    assert "event=1" in loss and "whole_clip=1" in loss and "Lossless" not in loss
    assert store.get_task(before["id"]) == before
    status, _, snapshot = _export(server, store, "json")
    assert status == 200 and snapshot["task"]["annotations"] == before["annotations"]


def test_temporal_export_discloses_rational_precision_and_csv_metadata_loss(api):
    server, store = api
    doc = _document(store)
    assert _import(server, store, doc, "cmd-metadata")[0] == 200
    task = store.get_task(store.list_tasks()[0]["task_id"])
    path, body = command(store, "cmd-third-time")
    body.update(operation="update_annotation", payload={"annotation_id": task["annotations"][0]["id"],
                "changes": {"start": {"seconds": rational(Fraction(1, 3))}}})
    assert request(server, "POST", path, body)[0] == 200
    status, headers, _ = _export(server, store)
    assert status == 200
    assert "Rational times rounded to JSON decimal numbers: 1" in headers["X-TraceAnnotate-Export-Losses"]
    status, headers, _ = _export(server, store, "csv")
    assert status == 200
    assert "excludes task/document metadata" in headers["X-TraceAnnotate-Export-Losses"]


def test_unknown_temporal_fields_reject_without_changing_saved_metadata(api):
    server, store = api
    doc = _document(store)
    assert _import(server, store, doc, "cmd-known")[0] == 200
    before = store.get_task(store.list_tasks()[0]["task_id"])
    unknown = copy.deepcopy(doc)
    unknown["undocumented_field"] = {"must_not_disappear": True}
    status, _, error = _import(server, store, unknown, "cmd-unknown")
    assert status == 422, error
    assert error["error"]["code"] == "INVALID_TEMPORAL_DOCUMENT"
    assert store.get_task(before["id"]) == before


def test_source_payload_deep_copies_input_and_archives_historical_reviews():
    media = {"duration": rational(2), "source": {"display_name": "a.mp4", "sha256": "a" * 64, "size_bytes": "12"}}
    doc = {"kind": "temporal", "schema_version": 2, "clip": "source.mp4", "duration": 2,
           "media_size": 12, "media_sha256": "a" * 64, "metadata": {"nested": ["original"]},
           "segments": [{"id": "seg-original", "start": 0, "end": 1, "label": "label", "text": "",
                         "track": "main", "tags": [], "review_status": "reviewed"}]}
    payload = temporal_import_payload(doc, media)
    doc["metadata"]["nested"].append("mutated after conversion")
    assert payload["legacy_document_metadata"] == {"nested": ["original"]}
    assert payload["legacy_document_envelope"]["source_reviews"] == [
        {"annotation_id": "seg-original", "legacy_id": "seg-original", "review_status": "reviewed"}]
    # Source declaration and admitted duration differ: preserve a still-valid
    # source declaration instead of silently rewriting the original envelope.
    task = {"annotations": payload["annotations"], "legacy_document_metadata": payload["legacy_document_metadata"],
            "extensions": {"temporal_source": payload["legacy_document_envelope"]}}
    exported = export_temporal_document(task, dict(media, duration=rational(Fraction(21, 10))))
    assert exported["duration"] == 2
    assert exported["clip"] == "source.mp4"


def test_frame_anchor_loss_is_disclosed_and_original_is_kept_in_json(api):
    server, store = api
    task = store.get_task(store.list_tasks()[0]["task_id"])
    rows = FrameService(store).page(task["media_id"])["frames"]
    path, body = command(store, "cmd-frame-projection")
    annotation = body["payload"]["annotation"]
    for name, row in [("start", rows[1]), ("end", rows[9])]:
        annotation[name] = {"seconds": row["seconds"], "anchor": {
            key: row[key] for key in ("frame_index", "source_pts", "proxy_pts")}}
    status, _, result = request(server, "POST", path, body)
    assert status == 200, result
    status, headers, temporal = _export(server, store)
    assert status == 200, temporal
    assert "Omitted frame anchors: 2" in headers["X-TraceAnnotate-Export-Losses"]
    assert temporal["segments"][0]["start"] == 0.1
    status, _, snapshot = _export(server, store, "json")
    assert status == 200
    assert snapshot["task"]["annotations"][0]["start"] == annotation["start"]
    assert snapshot["task"]["annotations"][0]["end"] == annotation["end"]


def test_historical_review_provenance_is_archived_not_new_approval(api):
    server, store = api
    doc = _document(store)
    doc["segments"][0]["review_status"] = "reviewed"
    status, _, result = _import(server, store, doc, "cmd-historical-review")
    assert status == 200, result
    task = store.get_task(store.list_tasks()[0]["task_id"])
    source = task["extensions"]["temporal_source"]
    assert source["source_reviews"][0]["review_status"] == "reviewed"
    assert source["source_reviews"][0]["legacy_id"] == doc["segments"][0]["id"]
    assert task["annotations"][0]["review_status"] == "unreviewed"
    status, headers, exported = _export(server, store)
    assert status == 200, exported
    assert exported["segments"][0]["review_status"] == "unreviewed"
    assert "Historical review provenance remains" in headers["X-TraceAnnotate-Export-Losses"]


@pytest.mark.parametrize("corruption", ["reviews", "duration", "clip", "presence", "unknown", "media_hash"])
def test_direct_import_command_rejects_invalid_source_envelope_without_writing(api, corruption):
    server, store = api
    before = store.get_task(store.list_tasks()[0]["task_id"])
    media = store.get_media(before["media_id"])
    payload = temporal_import_payload(_document(store), media)
    source = payload["legacy_document_envelope"]
    if corruption == "reviews":
        source["source_reviews"] = [None]
    elif corruption == "duration":
        source["duration"] = "not-a-duration"
    elif corruption == "clip":
        source["clip"] = "../../escape.mp4"
    elif corruption == "presence":
        source["metadata_present"] = "false"
    elif corruption == "unknown":
        source["unknown_extension"] = {"cannot_ignore": True}
    else:
        source["media_sha256"] = "0" * 64
    path, body = command(store, "cmd-corrupt-envelope")
    body.update(operation="import_temporal", payload=payload)
    status, _, error = request(server, "POST", path, body)
    assert status == 422, error
    assert error["error"]["code"] == (
        "MEDIA_IDENTITY_MISMATCH" if corruption == "media_hash" else "INVALID_TEMPORAL_ENVELOPE")
    assert store.get_task(before["id"]) == before


def test_report_distinguishes_current_review_from_historical_flags_without_mutating_task(api):
    server, store = api
    assert _import(server, store, _document(store), "cmd-review-report-import")[0] == 200
    before = store.get_task(store.list_tasks()[0]["task_id"])
    historical = copy.deepcopy(before)
    historical["annotations"][0]["review_status"] = "reviewed"
    old_bytes = copy.deepcopy(historical)
    qualified = reporting._review_qualification(store, historical, historical["annotations"])
    assert qualified["content_bound_reviewed"] == 0
    assert qualified["historical_or_unbound_reviewed"] == 1
    assert historical == old_bytes and store.get_task(before["id"]) == before
    path, body = command(store, "cmd-review-report-approve")
    body.update(operation="review_annotation", payload={"annotation_id": before["annotations"][0]["id"], "reviewed": True})
    assert request(server, "POST", path, body)[0] == 200
    url = f"/api/v1/projects/current/tasks/{before['id']}/report"
    status, _, report = request(server, "GET", url)
    assert status == 200
    assert report["semantic_review"] == "not_performed"
    assert report["review_qualification"]["content_bound_reviewed"] == 1
    assert report["review_qualification"]["historical_or_unbound_reviewed"] == 0
    path, body = command(store, "cmd-review-report-change")
    body.update(operation="update_annotation", payload={"annotation_id": before["annotations"][0]["id"],
                "changes": {"text": "a later human content change"}})
    assert request(server, "POST", path, body)[0] == 200
    status, _, report = request(server, "GET", url)
    assert status == 200
    assert report["review_qualification"]["content_bound_reviewed"] == 0
    assert report["review_qualification"]["unreviewed"] == 1


def test_temporal_rejects_unrepresentable_saved_fields_with_typed_error_not_500(api):
    server, store = api
    path, body = command(store, "cmd-empty-tag")
    # Workspace accepts this string; the older temporal format requires every
    # tag to be nonempty. A projection must reject, never silently filter it.
    body["payload"]["annotation"]["tags"] = ["", "kept"]
    status, _, result = request(server, "POST", path, body)
    assert status == 200, result
    before = store.get_task(store.list_tasks()[0]["task_id"])
    status, _, error = _export(server, store)
    assert status == 422, error
    assert error["error"]["code"] == "TEMPORAL_PROJECTION_UNSUPPORTED"
    assert store.get_task(before["id"]) == before
    status, _, snapshot = _export(server, store, "json")
    assert status == 200 and snapshot["task"] == before
