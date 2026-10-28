"""Test bidirectional adapter between temporal format and workspace schema."""
from fractions import Fraction
import json
from pathlib import Path
import pytest

from tests.test_workspace_server import api, command, http_video, request
from trace_annotate.temporal.schema import validate_document, require_valid_document
from trace_annotate.workspace.legacy_adapter import import_temporal_document, export_temporal_document
from trace_annotate.workspace.reporting import export_task
from trace_annotate.workspace.schema import rational, rational_value


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
    assert task_after["annotations"][0]["review_status"] == "reviewed"

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
    status, _, res = request(server, "POST", url, doc)
    assert status == 200, res
    assert res["applied"] is True
    assert res["imported_count"] == 1

    task_after = store.get_task(task_id)
    assert len(task_after["annotations"]) == 1
    assert task_after["annotations"][0]["id"] == "imported-seg-1"
    assert task_after["annotations"][0]["review_status"] == "reviewed"
