"""Tests for the persistent workspace (W01/W02 first slice).

These call the real WorkspaceStore implementation: project init, media import,
revisioned annotation commands, idempotency, conflict detection, reopen and
snapshot export.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import Conflict, NotFound, WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


FFMPEG = shutil.which("ffmpeg") or str(
    Path(os.environ.get("LOCALAPPDATA", "")) /
    "Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" /
    "ffmpeg-9.0.1-full_build/bin/ffmpeg.exe")


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    if not Path(FFMPEG).exists():
        pytest.skip("ffmpeg unavailable")
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=160x120:rate=10",
         "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture()
def project(tmp_path: Path, tiny_video: Path):
    root = tmp_path / "proj"
    store = WorkspaceStore(root)
    store.create()
    store.import_source(tiny_video)
    store.close()
    return root


def reopen(root: Path) -> WorkspaceStore:
    s = WorkspaceStore(root)
    s.open()
    return s


def test_project_creates_layout(project: Path):
    assert (project / "workspace.sqlite3").exists()
    assert (project / "objects").is_dir()


def test_import_admits_media_and_task(project: Path):
    s = reopen(project)
    try:
        tasks = s.list_tasks()
        assert len(tasks) == 1
        media = s.get_media(tasks[0]["media_id"])
        assert media["source"]["sha256"]
        assert int(media["frame_count"]) >= 1
    finally:
        s.close()


def _segment() -> dict:
    return {
        "id": "ann-1", "type": "segment", "label": "working",
        "text": "seg", "tags": ["t"], "track": "main", "attributes": {},
        "provenance": {"origin": "manual", "actor_id": "user", "job_id": None},
        "evidence_ids": [],
        "start": {"seconds": S.rational(0.0)},
        "end": {"seconds": S.rational(1.0)},
    }


def test_add_reopen_persists_and_export(project: Path):
    s = reopen(project)
    tid = s.list_tasks()[0]["task_id"]
    t0 = s.get_task(tid)
    resp = s.apply_task_command(
        command_id="c1", task_id=tid, expected_revision=0,
        expected_sha256=t0["content_sha256"], operation="add_annotation",
        payload={"annotation": _segment()}, request_body=b"{}")
    assert resp["response"]["revision"] == 1
    out = s.export_snapshot()
    s.close()

    s2 = reopen(project)
    t1 = s2.get_task(tid)
    assert t1["revision"] == 1
    assert t1["annotations"][0]["label"] == "working"
    snap = json.loads(out.read_text(encoding="utf-8"))
    assert snap["kind"] == "traceannotate-workspace"
    assert snap["schema_version"] == 3
    s2.close()


def test_idempotent_replay_returns_same_receipt(project: Path):
    s = reopen(project)
    tid = s.list_tasks()[0]["task_id"]
    t0 = s.get_task(tid)
    kw = dict(task_id=tid, expected_revision=0, expected_sha256=t0["content_sha256"],
              operation="add_annotation", payload={"annotation": _segment()}, request_body=b"{}")
    r1 = s.apply_task_command(command_id="dup", **kw)
    r2 = s.apply_task_command(command_id="dup", **kw)
    assert r1["response"]["revision"] == 1
    assert r2["idempotent_replay"] is True
    assert r2["response"]["revision"] == 1
    s.close()


def test_stale_revision_conflict_and_keeps_draft(project: Path):
    s = reopen(project)
    tid = s.list_tasks()[0]["task_id"]
    t0 = s.get_task(tid)
    s.apply_task_command(command_id="a", task_id=tid, expected_revision=0,
                         expected_sha256=t0["content_sha256"], operation="add_annotation",
                         payload={"annotation": _segment()}, request_body=b"{}")
    with pytest.raises(Conflict) as exc:
        s.apply_task_command(command_id="b", task_id=tid, expected_revision=0,
                             expected_sha256=t0["content_sha256"], operation="add_annotation",
                             payload={"annotation": _segment()}, request_body=b"{}")
    assert exc.value.code == "REVISION_CONFLICT"
    s.close()


def test_same_content_rename_relates_by_hash(project: Path, tmp_path: Path):
    s = reopen(project)
    tid = s.list_tasks()[0]["task_id"]
    media_id = s.list_tasks()[0]["media_id"]
    original_hash = s.get_media(media_id)["source"]["sha256"]
    s.close()
    renamed = tmp_path / "renamed.mp4"
    src = next(tmp_path.glob("clip.mp4"))
    shutil.copy2(src, renamed)
    s2 = reopen(project)
    r = s2.import_source(renamed, display_name="renamed.mp4")
    # same bytes -> same media_id identity
    assert r["media_id"] == media_id
    s2.close()


def test_missing_source_rejected(project: Path):
    s = reopen(project)
    with pytest.raises(NotFound):
        s.import_source(project / "nope.mp4")
    s.close()
