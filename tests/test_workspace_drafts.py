"""Persist complete rescue documents without applying annotations or pending commands."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.drafts import KIND, WorkspaceDrafts
from trace_annotate.workspace.errors import Conflict, NotFound, WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture()
def setup(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        candidate = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe"
        ffmpeg = str(candidate) if candidate.is_file() else None
    assert ffmpeg, "FFmpeg is required for real workspace rescue persistence tests"
    source = tmp_path / "真实 草稿源.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
                    "testsrc=duration=2:size=160x120:rate=10", "-pix_fmt", "yuv420p", str(source)],
                   check=True, capture_output=True, timeout=30)
    store = WorkspaceStore(tmp_path / "project")
    store.create()
    imported = store.import_source(source)
    task = store.get_task(imported["task_id"])
    media = store.get_media(task["media_id"])
    body = {"kind": KIND, "schema_version": 1, "project_id": store._project_id(),
            "task_id": task["id"], "media_id": media["id"],
            "source_sha256": media["source"]["sha256"], "playback_sha256": media["playback"]["sha256"],
            "base_revision": task["revision"], "base_sha256": task["content_sha256"],
            "current_form": {"label": "未保存的中文 😀", "start": "1/10", "end": "1", "reviewed": True},
            "anchors": {"start": None, "end": None}, "dirty": True, "selected_id": None,
            "task_snapshot": task, "pending_command": None,
            "form_base": {"label": ""}, "anchor_base": {}, "saved_at": "2026-10-29T00:00:00Z"}
    drafts = WorkspaceDrafts(store)
    request = dict(draft_id="draft-a", editor_session_id="editor-a", task_id=task["id"],
                   media_id=media["id"], expected_draft_revision=0, body=body)
    try:
        yield store, drafts, request, source
    finally:
        store.close()


def test_draft_full_roundtrip_reopen_does_not_change_task_or_receipts(setup):
    store, drafts, request, source = setup
    before = store.get_snapshot()
    saved = drafts.save(**request)
    assert saved["draft_revision"] == 1 and saved["body"] == request["body"]
    assert saved["base_conflict"] is False
    assert store.get_snapshot() == before
    original_bytes = source.read_bytes()
    store.close()
    store.open()
    reopened = WorkspaceDrafts(store).get("draft-a")
    assert reopened == saved
    assert store.get_snapshot() == before
    assert source.read_bytes() == original_bytes


def test_revision_cas_and_clean_draft_archiving_without_delete(setup):
    _, drafts, request, _ = setup
    drafts.save(**request)
    with pytest.raises(Conflict) as stale:
        drafts.save(**request)
    assert stale.value.code == "DRAFT_REVISION_CONFLICT"
    request["expected_draft_revision"] = 1
    request["body"]["dirty"] = False
    saved = drafts.save(**request)
    assert saved["draft_revision"] == 2
    assert drafts.list() == []
    assert [item["draft_id"] for item in drafts.list(include_clean=True)] == ["draft-a"]
    assert drafts.get("draft-a")["body"]["current_form"]["reviewed"] is True


def test_same_draft_id_cannot_be_claimed_by_another_editor(setup):
    _, drafts, request, _ = setup
    drafts.save(**request)
    another = copy.deepcopy(request)
    another.update(editor_session_id="editor-b", expected_draft_revision=1)
    with pytest.raises(Conflict) as conflict:
        drafts.save(**another)
    assert conflict.value.code == "DRAFT_OWNER_CONFLICT"
    assert drafts.get("draft-a")["editor_session_id"] == "editor-a"


def test_separate_sessions_and_tasks_do_not_overwrite_each_other(setup):
    store, drafts, request, source = setup
    drafts.save(**request)
    second = copy.deepcopy(request)
    second.update(draft_id="draft-b", editor_session_id="editor-b")
    second["body"]["current_form"]["label"] = "第二页独立草稿"
    drafts.save(**second)
    new_task = store.get_task(store.import_source(source)["task_id"])
    third = copy.deepcopy(request)
    third.update(draft_id="draft-c", task_id=new_task["id"])
    third["body"].update(task_id=new_task["id"], base_revision=new_task["revision"],
                         base_sha256=new_task["content_sha256"], task_snapshot=new_task)
    drafts.save(**third)
    assert len(drafts.list(task_id=request["task_id"])) == 2
    assert [r["draft_id"] for r in drafts.list(task_id=new_task["id"])] == ["draft-c"]
    assert drafts.get("draft-a")["body"]["current_form"]["label"] != drafts.get("draft-b")["body"]["current_form"]["label"]


def test_concurrent_editor_saves_have_exactly_one_cas_winner(setup):
    _, drafts, request, _ = setup
    drafts.save(**request)
    barrier = threading.Barrier(2)
    def compete(label):
        contender = copy.deepcopy(request)
        contender["expected_draft_revision"] = 1
        contender["body"]["current_form"]["label"] = label
        barrier.wait(timeout=10)
        try:
            return drafts.save(**contender)
        except Conflict as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(compete, ("left", "right")))
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert "DRAFT_REVISION_CONFLICT" in outcomes
    assert drafts.get("draft-a")["draft_revision"] == 2


def test_stale_offline_base_is_preserved_for_merge_without_annotation_apply(setup):
    store, drafts, request, _ = setup
    task = store.get_task(request["task_id"])
    annotation = {"id": "ann-current", "type": "segment", "label": "另一页已经提交",
                  "start": {"seconds": S.rational("0.1")}, "end": {"seconds": S.rational("1")}}
    command = dict(command_id="cmd-new-content", task_id=task["id"], expected_revision=task["revision"],
                   expected_sha256=task["content_sha256"], operation="add_annotation", payload={"annotation": annotation})
    store.apply_task_command(**command, request_body=S.canonical_json(command))
    current = store.get_task(task["id"])
    request["body"]["pending_command"] = {"command": {"command_id": "cmd-never-execute",
        "task_id": task["id"], "operation": "delete_annotation", "payload": {"annotation_id": "ann-current"}}}
    saved = drafts.save(**request)
    assert saved["base_conflict"] is True and saved["current_task_revision"] == 1
    assert saved["body"]["base_revision"] == 0
    assert store.get_task(task["id"]) == current
    with pytest.raises(NotFound):
        store.get_command_receipt("cmd-never-execute")


@pytest.mark.parametrize("field,value", [
    ("project_id", "proj-other"), ("task_id", "tsk-other"), ("media_id", "med-other"),
    ("source_sha256", "0" * 64), ("playback_sha256", "0" * 64),
    ("base_sha256", "0" * 64), ("base_revision", 99),
    ("draft_id", "draft-other"), ("editor_session_id", "editor-other")])
def test_mismatched_rescue_identity_is_rejected_without_saving(setup, field, value):
    store, drafts, request, _ = setup
    before = store.get_snapshot()
    request["body"][field] = value
    with pytest.raises(WorkspaceError):
        drafts.save(**request)
    assert drafts.list() == []
    assert store.get_snapshot() == before


@pytest.mark.parametrize("field,value", [("kind", "other"), ("schema_version", 2),
    ("current_form", []), ("anchors", []), ("dirty", "yes"), ("pending_command", []),
    ("base_revision", True), ("task_snapshot", {"id": "wrong"}), ("token", "must-not-persist")])
def test_malformed_rescue_is_rejected(setup, field, value):
    _, drafts, request, _ = setup
    request["body"][field] = value
    with pytest.raises(WorkspaceError):
        drafts.save(**request)
    assert drafts.list() == []


def test_non_json_cyclic_deep_and_oversized_rescue_is_rejected(setup):
    _, drafts, request, _ = setup
    for bad in (float("nan"), b"bytes", {"not-a-string-key": {1: "no"}}, set()):
        altered = copy.deepcopy(request)
        altered["body"]["extra"] = bad
        with pytest.raises(WorkspaceError):
            drafts.save(**altered)
    cyclic = copy.deepcopy(request)
    cyclic["body"]["cycle"] = cyclic["body"]
    with pytest.raises(WorkspaceError):
        drafts.save(**cyclic)
    deep = copy.deepcopy(request)
    value = {}
    deep["body"]["extra"] = value
    for _ in range(40):
        value["nested"] = {}
        value = value["nested"]
    with pytest.raises(WorkspaceError):
        drafts.save(**deep)
    huge = copy.deepcopy(request)
    huge["body"]["extra"] = "x" * (4 * 1024 * 1024)
    with pytest.raises(WorkspaceError):
        drafts.save(**huge)
    assert drafts.list() == []


def test_pending_command_for_other_task_is_rejected(setup):
    _, drafts, request, _ = setup
    request["body"]["pending_command"] = {"command": {"task_id": "tsk-other"}}
    with pytest.raises(Conflict):
        drafts.save(**request)
    assert drafts.list() == []


def test_corrupt_saved_rescue_is_not_returned_as_valid(setup):
    store, drafts, request, _ = setup
    drafts.save(**request)
    with store.transaction() as connection:
        connection.execute("UPDATE drafts SET body=? WHERE draft_id=?", ('{"broken":true}', "draft-a"))
    with pytest.raises(WorkspaceError) as corrupt:
        drafts.get("draft-a")
    assert corrupt.value.code == "RECOVERY_REQUIRED"
