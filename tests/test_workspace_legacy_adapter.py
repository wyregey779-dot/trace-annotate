"""Test bidirectional adapter between temporal format and workspace schema."""
from fractions import Fraction
import copy
import json
from pathlib import Path
import sys
import pytest

from tests.test_workspace_server import api, command, http_video, request
from trace_annotate.temporal.schema import validate_document, require_valid_document
from trace_annotate.workspace.legacy_adapter import import_temporal_document, export_temporal_document
from trace_annotate.workspace.reporting import export_task
from trace_annotate.workspace.schema import rational, rational_value
from trace_annotate.workspace import schema as S


SAMPLE_TEMPORAL_V2 = {
    "kind": "temporal",
    "schema_version": 2,
    "clip": "sample.mp4",
    "duration": 10.0,
    "media_size": 1048576,
    "media_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "segments": [
        {
            "id": "seg-1",
            "start": 0.5,
            "end": 2.5,
            "label": "Action A",
            "text": "First action description",
            "tags": ["tag1", "tag2"],
            "track": "main",
            "review_status": "reviewed",
        },
        {
            "id": "seg-2",
            "start": 2.0,
            "end": 5.0,
            "label": "Action B (Overlapping)",
            "text": "Second action on same track",
            "tags": ["overlap"],
            "track": "main",
            "review_status": "unreviewed",
        },
        {
            "id": "custom-track-seg",
            "start": 1.0,
            "end": 4.0,
            "label": "Action C",
            "text": "Secondary track action",
            "tags": [],
            "track": "sub_track",
            "review_status": "reviewed",
        },
    ],
    "metadata": {"annotator": "tester", "notes": "sample test"},
}


def test_import_and_export_roundtrip():
    media = {
        "duration": rational(10.0),
        "size_bytes": 1048576,
        "source": {
            "display_name": "sample.mp4",
            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        },
    }
    # 1. Convert temporal doc to workspace annotations
    annotations = import_temporal_document(SAMPLE_TEMPORAL_V2, media)
    assert len(annotations) == 3
    assert annotations[0]["id"] == "seg-1"
    assert annotations[0]["review_status"] == "reviewed"
    assert annotations[0]["start"] == {"seconds": {"n": "1", "d": "2"}}
    assert annotations[0]["end"] == {"seconds": {"n": "5", "d": "2"}}

    # 2. Convert back to temporal doc
    task = {"annotations": annotations, "metadata": SAMPLE_TEMPORAL_V2["metadata"]}
    exported = export_temporal_document(task, media, schema_version=2)
    assert exported["kind"] == "temporal"
    assert exported["schema_version"] == 2
    assert exported["clip"] == "sample.mp4"
    assert exported["duration"] == 10.0
    assert exported["media_sha256"] == SAMPLE_TEMPORAL_V2["media_sha256"]
    assert exported["metadata"] == SAMPLE_TEMPORAL_V2["metadata"]

    # Compare segments
    assert len(exported["segments"]) == 3
    s1 = next(s for s in exported["segments"] if s["id"] == "seg-1")
    assert s1["start"] == 0.5
    assert s1["end"] == 2.5
    assert s1["label"] == "Action A"
    assert s1["text"] == "First action description"
    assert s1["review_status"] == "reviewed"
    assert s1["tags"] == ["tag1", "tag2"]
    assert s1["track"] == "main"


def test_import_preserves_non_ascii_legacy_id():
    media = {
        "duration": rational(10.0),
        "source": {"display_name": "video.mp4", "sha256": "0" * 64},
    }
    doc = {
        "kind": "temporal",
        "schema_version": 2,
        "clip": "video.mp4",
        "duration": 10.0,
        "media_size": 100,
        "media_sha256": "0" * 64,
        "segments": [
            {
                "id": "中文ID_123",  # Non-ASCII, not a standard workspace ID
                "start": 1.0,
                "end": 2.0,
                "label": "Test",
                "text": "",
                "tags": [],
                "track": "main",
                "review_status": "unreviewed",
            }
        ],
    }
    annotations = import_temporal_document(doc, media)
    assert len(annotations) == 1
    # ID was mapped to valid ASCII
    assert annotations[0]["id"].startswith("seg-")
    assert annotations[0]["legacy_id"] == "中文ID_123"

    # Export restores legacy ID
    task = {"annotations": annotations}
    exported = export_temporal_document(task, media)
    assert exported["segments"][0]["id"] == "中文ID_123"


def test_replace_annotations_in_store(api):
    server, store = api
    tasks = store.list_tasks()
    task_id = tasks[0]["task_id"]
    task_before = store.get_task(task_id)

    # Prepare replacement annotations
    media = store.get_media(task_before["media_id"])
    new_annotations = [
        {
            "id": "seg-alpha",
            "type": "segment",
            "start": {"seconds": rational(0.5)},
            "end": {"seconds": rational(1.5)},
            "label": "Replaced Segment",
            "text": "Description",
            "tags": ["alpha"],
            "track": "main",
            "review_status": "reviewed",
            "attributes": {},
            "evidence_ids": [],
            "provenance": {"origin": "manual", "actor_id": "user", "job_id": None},
        }
    ]

    path, body = command(store, "cmd-replace")
    body["operation"] = "replace_annotations"
    body["payload"] = {"annotations": new_annotations}
    status, _, receipt = request(server, "POST", path, body)
    assert status == 200, receipt
    assert receipt["applied"] is True

    # Check updated task in store
    task_after = store.get_task(task_id)
    assert task_after["revision"] == task_before["revision"] + 1
    assert len(task_after["annotations"]) == 1
    assert task_after["annotations"][0]["id"] == "seg-alpha"
    # Carrying a reviewed flag in an ordinary replacement is not a new review.
    assert task_after["annotations"][0]["review_status"] == "unreviewed"
    review_path, review_body = command(store, "cmd-review-replaced-segment")
    review_body.update(operation="review_annotation", payload={"annotation_id": "seg-alpha", "reviewed": True})
    status, _, review_receipt = request(server, "POST", review_path, review_body)
    assert status == 200, review_receipt
    reviewed = store.get_task(task_id)["annotations"][0]
    assert reviewed["review_status"] == "reviewed"
    assert reviewed["review_binding"]["command_id"] == "cmd-review-replaced-segment"

    # Export as temporal format via server route
    export_url = f"/api/v1/projects/current/tasks/{task_id}/export?format=temporal"
    status, headers, data = request(server, "GET", export_url)
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    exported_doc = data if isinstance(data, dict) else json.loads(data.decode("utf-8"))
    assert exported_doc["kind"] == "temporal"
    assert len(exported_doc["segments"]) == 1
    assert exported_doc["segments"][0]["id"] == "seg-alpha"
    assert exported_doc["segments"][0]["review_status"] == "reviewed"
    assert exported_doc["segments"][0]["label"] == "Replaced Segment"


def test_real_102_items_roundtrip():
    source_path = Path(r"D:\099\trace-real-assignment-20261027\full-delivery\修复版浏览器最终导出.json")
    if not source_path.is_file():
        pytest.skip("Real 102 delivery file not found on disk")
    doc = json.loads(source_path.read_text(encoding="utf-8"))
    media = {
        "duration": rational(doc["duration"]),
        "size_bytes": doc["media_size"],
        "source": {
            "display_name": doc["clip"],
            "sha256": doc["media_sha256"],
            "size_bytes": str(doc["media_size"]),
        },
    }
    annotations = import_temporal_document(doc, media)
    assert len(annotations) == 102
    task = {"annotations": annotations, "metadata": doc.get("metadata", {})}
    exported = export_temporal_document(task, media, schema_version=2)
    assert len(exported["segments"]) == 102
    for orig, exp in zip(doc["segments"], exported["segments"]):
        assert orig["id"] == exp["id"]
        assert abs(orig["start"] - exp["start"]) < 1e-6
        assert abs(orig["end"] - exp["end"]) < 1e-6
        assert orig["label"] == exp["label"]
        assert orig["text"] == exp["text"]
        assert orig["tags"] == exp["tags"]
        assert orig["track"] == exp["track"]
        assert orig["review_status"] == exp["review_status"]


def test_post_import_temporal_endpoint(api):
    server, store = api
    tasks = store.list_tasks()
    task_id = tasks[0]["task_id"]
    task_before = store.get_task(task_id)
    media = store.get_media(task_before["media_id"])

    doc = {
        "kind": "temporal",
        "schema_version": 2,
        "clip": media["source"]["display_name"],
        "duration": float(rational_value(media["duration"])),
        "media_size": int(media["source"]["size_bytes"]),
        "media_sha256": media["source"]["sha256"],
        "segments": [
            {
                "id": "imported-seg-1",
                "start": 0.1,
                "end": 0.9,
                "label": "Imported Label",
                "text": "Imported Text",
                "tags": ["imp1"],
                "track": "main",
                "review_status": "reviewed",
            }
        ],
    }

    url = f"/api/v1/projects/current/tasks/{task_id}/import-temporal"
    task_before = store.get_task(task_id)
    status, _, res = request(server, "POST", url, {
        "document": doc,
        "command_id": "cmd-import-1",
        "expected_revision": task_before["revision"],
        "expected_sha256": task_before["content_sha256"],
    })
    assert status == 200, res
    assert res["applied"] is True
    assert res["imported_count"] == 1

    task_after = store.get_task(task_id)
    assert len(task_after["annotations"]) == 1
    assert task_after["annotations"][0]["id"] == "imported-seg-1"
    assert task_after["annotations"][0]["review_status"] == "unreviewed"
    assert task_after["extensions"]["temporal_source"]["source_reviews"] == [
        {"annotation_id": "imported-seg-1", "legacy_id": "imported-seg-1", "review_status": "reviewed"}]
    # Imported source approval is retained as history. A current approval needs
    # an explicit action bound to this task's current annotation content.
    review_path, review_body = command(store, "cmd-review-imported-segment")
    review_body.update(operation="review_annotation", payload={"annotation_id": "imported-seg-1", "reviewed": True})
    status, _, review_receipt = request(server, "POST", review_path, review_body)
    assert status == 200, review_receipt
    reviewed = store.get_task(task_id)["annotations"][0]
    assert reviewed["review_status"] == "reviewed"
    assert reviewed["review_binding"]["command_id"] == "cmd-review-imported-segment"


def _doc(media, **over):
    doc = {
        "kind": "temporal", "schema_version": 2,
        "clip": media["source"]["display_name"],
        "duration": float(rational_value(media["duration"])),
        "media_size": int(media["source"]["size_bytes"]),
        "media_sha256": media["source"]["sha256"],
        "segments": [{"id": "seg-a", "start": 0.1, "end": 0.9, "label": "A",
                      "text": "t", "tags": [], "track": "main",
                      "review_status": "unreviewed"}],
    }
    doc.update(over)
    return doc


def test_import_temporal_rejects_missing_command_identity(api):
    server, store = api
    task_id = store.list_tasks()[0]["task_id"]
    media = store.get_media(store.get_task(task_id)["media_id"])
    doc = _doc(media)
    url = f"/api/v1/projects/current/tasks/{task_id}/import-temporal"
    # No command_id at all -> rejected (no invented idempotency key).
    status, _, res = request(server, "POST", url, doc)
    assert status == 422, res
    assert res["error"]["code"] == "IMPORT_REQUIRES_COMMAND_ID"
    task_before = store.get_task(task_id)
    # Declared hash does not match this task's media -> rejected, task untouched.
    bad = _doc(media, media_sha256="0" * 64)
    status, _, res = request(server, "POST", url, {
        "document": bad, "command_id": "cmd-bad-hash",
        "expected_revision": task_before["revision"],
        "expected_sha256": task_before["content_sha256"],
    })
    assert status == 422, res
    assert res["error"]["code"] == "MEDIA_IDENTITY_MISMATCH"
    assert len(store.get_task(task_id)["annotations"]) == len(task_before["annotations"])
    # Declared size mismatch -> rejected.
    badsize = _doc(media, media_size=int(media["source"]["size_bytes"]) + 999)
    status, _, res = request(server, "POST", url, {
        "document": badsize, "command_id": "cmd-bad-size",
        "expected_revision": task_before["revision"],
        "expected_sha256": task_before["content_sha256"],
    })
    assert status == 422, res
    assert res["error"]["code"] == "MEDIA_IDENTITY_MISMATCH"


def test_import_temporal_replay_is_idempotent(api):
    server, store = api
    task_id = store.list_tasks()[0]["task_id"]
    media = store.get_media(store.get_task(task_id)["media_id"])
    doc = _doc(media)
    url = f"/api/v1/projects/current/tasks/{task_id}/import-temporal"
    task_before = store.get_task(task_id)
    body = {
        "document": doc, "command_id": "cmd-replay-1",
        "expected_revision": task_before["revision"],
        "expected_sha256": task_before["content_sha256"],
    }
    s1, _, r1 = request(server, "POST", url, body)
    assert s1 == 200, r1
    rev_after_first = store.get_task(task_id)["revision"]
    # Re-post the SAME command id (e.g. response lost) -> same receipt, no second apply.
    s2, _, r2 = request(server, "POST", url, body)
    assert s2 == 200, r2
    assert store.get_task(task_id)["revision"] == rev_after_first
    assert len(store.get_task(task_id)["annotations"]) == 1


@pytest.mark.parametrize("reserve_suffixes", [False, True])
def test_legacy_id_collisions_terminate_and_preserve_original_ids(reserve_suffixes):
    """Valid user IDs must not hold the import transaction in an endless loop."""
    raw = "非ASCII标注"
    base = S.make_id("seg", raw)
    ids = ([base, base + "-1", base + "-2", raw] if reserve_suffixes else
           [base, S.make_id("seg", raw + "2"), raw])
    doc = copy.deepcopy(SAMPLE_TEMPORAL_V2)
    doc["segments"] = [dict(doc["segments"][0], id=ident) for ident in ids]
    assert validate_document(doc) == []
    media = {"duration": rational(10), "source": {"display_name": doc["clip"],
             "sha256": doc["media_sha256"], "size_bytes": str(doc["media_size"])}}
    # A bounded diagnostic makes the old endless-loop regression fail without
    # stranding a test thread or requiring the runner to kill a child process.
    previous_trace = sys.gettrace()
    remaining = 20_000
    def bounded_import(frame, event, arg):
        nonlocal remaining
        if event == "line" and frame.f_code is import_temporal_document.__code__:
            remaining -= 1
            if remaining <= 0:
                raise AssertionError("Valid legacy ID collision import did not terminate")
        return bounded_import
    try:
        sys.settrace(bounded_import)
        annotations = import_temporal_document(doc, media)
    finally:
        sys.settrace(previous_trace)
    assert len({ann["id"] for ann in annotations}) == len(ids)
    assert all(S.ID_RE.fullmatch(ann["id"]) for ann in annotations)
    assert import_temporal_document(doc, media) == annotations
    exported = export_temporal_document({"annotations": annotations}, media)
    assert [seg["id"] for seg in exported["segments"]] == ids

