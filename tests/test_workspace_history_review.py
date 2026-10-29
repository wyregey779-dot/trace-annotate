"""Conditional history and content-bound review regressions (real store)."""
import copy
import json

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture
def store(tmp_path):
    value = WorkspaceStore(tmp_path / "project")
    value.create()
    # Deliberately isolate transactions from decoding; real media/HTTP/UI gates
    # run separately and must not claim this fixture is media admission proof.
    media = {"id": "med-history", "duration": S.rational(3),
             "source": {"sha256": "a" * 64, "size_bytes": "12"}}
    value.conn.execute("INSERT INTO media VALUES(?,?)", (media["id"], json.dumps(media)))
    task = {"id": "tsk-history", "media_id": media["id"], "template_id": "generic-segment",
            "template_version": 1, "revision": 0, "annotations": [], "extensions": {}}
    task["content_sha256"] = value.task_digest(task)
    value.conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?)", (task["id"], media["id"], "generic-segment", 1, 0))
    value.conn.execute("INSERT INTO task_revisions VALUES(?,?,?,?,?,?,?)",
                       ("rev-initial", task["id"], 0, json.dumps(task), task["content_sha256"], None, "test"))
    try:
        yield value
    finally:
        value.close()


def submit(store, command_id, operation, payload, **overrides):
    task = store.get_task("tsk-history")
    args = dict(command_id=command_id, task_id=task["id"], expected_revision=task["revision"],
                expected_sha256=task["content_sha256"], operation=operation,
                payload=payload, request_body=S.canonical_json(payload))
    args.update(overrides)
    return store.apply_task_command(**args)


def ann(ident="ann-a", label="original"):
    return {"id": ident, "type": "segment", "label": label,
            "start": {"seconds": S.rational(0)}, "end": {"seconds": S.rational(1)}}


def test_revert_rejects_refreshed_revision_after_other_tab_edit(store):
    submit(store, "cmd-add", "add_annotation", {"annotation": ann()})
    submit(store, "cmd-a-edit", "update_annotation", {"annotation_id": "ann-a", "changes": {"label": "A"}})
    submit(store, "cmd-b-add", "add_annotation", {"annotation": ann("ann-b", "B")})
    before = store.get_task("tsk-history")
    with pytest.raises(WorkspaceError) as error:
        submit(store, "cmd-unsafe-undo", "revert_command", {"target_command_id": "cmd-a-edit"})
    assert error.value.code == "HISTORY_DIVERGED"
    assert store.get_task("tsk-history") == before
    with pytest.raises(WorkspaceError) as missing:
        store.get_command_receipt("cmd-unsafe-undo")
    assert missing.value.code == "COMMAND_UNKNOWN"


def test_revert_and_redo_are_durable_idempotent_commands(store):
    added = submit(store, "cmd-add", "add_annotation", {"annotation": ann()})["response"]
    undone = submit(store, "cmd-undo", "revert_command", {"target_command_id": "cmd-add"})["response"]
    assert not store.get_task("tsk-history")["annotations"]
    submit(store, "cmd-redo", "revert_command", {"target_command_id": "cmd-undo"})
    assert store.get_task("tsk-history")["annotations"][0]["label"] == "original"
    replay = submit(store, "cmd-undo", "revert_command", {"target_command_id": "cmd-add"},
                    expected_revision=added["revision"], expected_sha256=added["content_sha256"])
    assert replay["idempotent_replay"] and replay["response"] == undone
    assert store.get_task("tsk-history")["revision"] == 3


def test_review_cannot_be_inherited_by_boundary_button_or_api(store):
    submit(store, "cmd-add", "add_annotation", {"annotation": ann()})
    submit(store, "cmd-review", "review_annotation", {"annotation_id": "ann-a", "reviewed": True})
    reviewed = copy.deepcopy(store.get_task("tsk-history")["annotations"][0])
    assert reviewed["review_status"] == "reviewed" and reviewed["review_binding"]
    submit(store, "cmd-boundary", "update_annotation", {"annotation_id": "ann-a", "changes": {
        "end": {"seconds": S.rational(3)}, "review_status": "reviewed"}})
    changed = store.get_task("tsk-history")["annotations"][0]
    assert changed["review_status"] == "unreviewed"
    assert "review_binding" not in changed
    submit(store, "cmd-revert-boundary", "revert_command", {"target_command_id": "cmd-boundary"})
    restored = store.get_task("tsk-history")["annotations"][0]
    assert restored["end"] == reviewed["end"]
    assert restored["review_status"] == "unreviewed"


def test_legacy_review_flag_does_not_create_current_review(store):
    legacy = ann()
    legacy["review_status"] = "reviewed"
    submit(store, "cmd-legacy", "add_annotation", {"annotation": legacy})
    assert store.get_task("tsk-history")["annotations"][0]["review_status"] == "unreviewed"


def test_multiple_undos_and_redos_follow_protected_logical_history(store):
    submit(store, "cmd-first", "add_annotation", {"annotation": ann()})
    submit(store, "cmd-second", "update_annotation", {"annotation_id": "ann-a", "changes": {"label": "second"}})
    submit(store, "cmd-undo-second", "revert_command", {"target_command_id": "cmd-second"})
    submit(store, "cmd-undo-first", "revert_command", {"target_command_id": "cmd-first"})
    assert not store.get_task("tsk-history")["annotations"]
    submit(store, "cmd-redo-first", "revert_command", {"target_command_id": "cmd-undo-first"})
    submit(store, "cmd-redo-second", "revert_command", {"target_command_id": "cmd-undo-second"})
    assert store.get_task("tsk-history")["annotations"][0]["label"] == "second"


def test_undo_review_does_not_inherit_the_review_being_undone(store):
    submit(store, "cmd-add", "add_annotation", {"annotation": ann()})
    submit(store, "cmd-review", "review_annotation", {"annotation_id": "ann-a", "reviewed": True})
    submit(store, "cmd-unreview", "revert_command", {"target_command_id": "cmd-review"})
    assert store.get_task("tsk-history")["annotations"][0]["review_status"] == "unreviewed"


def test_corrupt_historical_revision_cannot_be_promoted_to_current(store):
    submit(store, "cmd-add", "add_annotation", {"annotation": ann()})
    submit(store, "cmd-edit", "update_annotation", {"annotation_id": "ann-a", "changes": {"label": "edited"}})
    store.conn.execute("UPDATE task_revisions SET body=json_set(body,'$.annotations[0].label','tampered') WHERE task_id=? AND revision=1", ("tsk-history",))
    before = store.get_task("tsk-history")
    with pytest.raises(WorkspaceError) as error:
        submit(store, "cmd-undo", "revert_command", {"target_command_id": "cmd-edit"})
    assert error.value.code == "RECOVERY_REQUIRED"
    assert store.get_task("tsk-history") == before
