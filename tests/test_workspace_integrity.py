"""Storage failure boundaries; these are not success-path smoke tests."""
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture
def store(tmp_path):
    value = WorkspaceStore(tmp_path / "workspace")
    value.create()
    # Isolate persistence from codecs. Media admission is tested independently.
    media = {"id": "med-test", "duration": S.rational(2),
             "video_start": S.rational(0), "video_end": S.rational(2),
             "source": {"sha256": "a" * 64, "size_bytes": "12"}}
    value.conn.execute("INSERT INTO media VALUES(?,?)", (media["id"], json.dumps(media)))
    task = {"id": "tsk-test", "media_id": media["id"], "template_id": "generic-segment",
            "template_version": 1, "revision": 0, "annotations": [], "extensions": {}}
    task["content_sha256"] = (value.task_digest(task) if hasattr(value, "task_digest")
                               else S.task_content_sha256(task))
    value.conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?)", (task["id"], media["id"], "generic-segment", 1, 0))
    value.conn.execute("INSERT INTO task_revisions VALUES(?,?,?,?,?,?,?)",
                       ("rev-test", task["id"], 0, json.dumps(task), task["content_sha256"], None, "test"))
    yield value
    value.close()


def command(store, **overrides):
    task = store.get_task("tsk-test")
    args = dict(command_id="cmd-test", task_id=task["id"], expected_revision=task["revision"],
                expected_sha256=task["content_sha256"], operation="add_annotation",
                payload={"annotation": {"id": "ann-test", "type": "whole_clip", "label": "review", "text": "review"}},
                request_body=b"{}")
    args.update(overrides)
    return store.apply_task_command(**args)


def test_commit_failure_rolls_back_revision_and_receipt(store):
    store.conn.execute("CREATE TRIGGER fail_commit BEFORE INSERT ON durable_commits BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError):
        command(store)
    assert store.get_task("tsk-test")["revision"] == 0
    assert store.conn.execute("SELECT revision FROM tasks").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM command_receipts").fetchone()[0] == 0


def test_idempotency_is_bound_to_semantic_request_even_with_same_raw_bytes(store):
    command(store)
    with pytest.raises(WorkspaceError) as error:
        command(store, payload={"annotation": {"id": "ann-other", "type": "whole_clip"}})
    assert error.value.code == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("bad", ["../outside", "..\\outside", "C:\\outside", "bad/name", "trailing\n"])
def test_export_rejects_escape_names(store, bad):
    with pytest.raises(WorkspaceError):
        store.export_snapshot(bad)


def test_export_does_not_replace_existing_snapshot(store):
    path = store.export_snapshot("stable")
    original = path.read_bytes()
    command(store)
    with pytest.raises(WorkspaceError):
        store.export_snapshot("stable")
    assert path.read_bytes() == original


def test_store_reads_work_on_request_thread(store):
    results = []
    def read():
        try:
            results.append(store.get_task("tsk-test")["id"])
        except Exception as error:
            results.append(error)
    thread = threading.Thread(target=read)
    thread.start()
    thread.join(5)
    assert results == ["tsk-test"]


def test_artifact_corruption_is_rejected_without_leaking_lock(tmp_path):
    value = WorkspaceStore(tmp_path / "corrupt")
    value.create()
    artifact = value._put_artifact(b"original")
    value.close()
    (value.root / artifact["relative_path"]).write_bytes(b"tampered")
    for _ in range(2):
        reopened = WorkspaceStore(value.root)
        try:
            with pytest.raises(WorkspaceError) as error:
                reopened.open()
            assert error.value.code == "RECOVERY_REQUIRED"
        finally:
            reopened.close()


def test_invalid_identifier_returns_typed_error():
    for value in ("bad\n", "../x", None):
        with pytest.raises(WorkspaceError):
            S.require_id(value)


def test_rational_preserves_large_decimal_integer_precision():
    assert S.rational({"n": "9007199254740993", "d": "2"}) == {"n": "9007199254740993", "d": "2"}


def test_task_hash_changes_with_revision(store):
    task = store.get_task("tsk-test")
    first = store.task_digest(task)
    task["revision"] += 1
    assert store.task_digest(task) != first


def test_annotation_outside_media_is_not_committed(store):
    with pytest.raises(WorkspaceError):
        command(store, payload={"annotation": {"id": "ann-outside", "type": "segment", "label": "test",
                                                "start": {"seconds": S.rational(1)},
                                                "end": {"seconds": S.rational(3)}}})
    assert store.get_task("tsk-test")["revision"] == 0
