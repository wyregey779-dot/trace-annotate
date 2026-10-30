"""R07 backup/restore safety: path traversal rejection and closure integrity."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from trace_annotate.workspace.backup import (
    BACKUP_MANIFEST, SNAPSHOT_DB, backup_workspace, restore_workspace)
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.store import WorkspaceStore

FFMPEG = shutil.which("ffmpeg") or str(
    Path(__import__("os").environ.get("LOCALAPPDATA", "")) /
    "Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" /
    "ffmpeg-9.0.1-full_build/bin/ffmpeg.exe")


@pytest.fixture()
def source_video(tmp_path: Path) -> Path:
    if not Path(FFMPEG).exists():
        pytest.skip("ffmpeg unavailable")
    out = tmp_path / "incoming.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                     "-i", "testsrc=duration=1:size=160x120:rate=10",
                     "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def _failed_intake_project(tmp_path: Path, source: Path) -> WorkspaceStore:
    store = WorkspaceStore(tmp_path / "proj")
    store.create()
    q = AsyncIntakes(store)
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not media" * 8)
    upload = store.tmp_dir / "up.tmp"
    shutil.copy2(bad, upload)
    q.enqueue(upload, display_name="broken.mp4")
    q.run_pending()
    rows = store.list_intakes()
    assert q.get(rows[0]["id"])["state"] == "failed"
    return store


def test_restore_rejects_relative_path_traversal(tmp_path, source_video):
    store = _failed_intake_project(tmp_path, source_video)
    manifest = backup_workspace(store, tmp_path / "backup")
    store.close()

    # Forge a malicious backup that asks restore to write outside its root.
    evil = tmp_path / "evil-backup"
    shutil.copytree(tmp_path / "backup", evil)
    sentinel = tmp_path / "escape.txt"
    sentinel.write_text("outside")
    data = json.loads((evil / BACKUP_MANIFEST).read_text(encoding="utf-8"))
    data["retryable_sources"][0]["relative_path"] = "../escape/probe.source"
    (evil / BACKUP_MANIFEST).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(Exception) as exc:
        restore_workspace(evil, tmp_path / "restored")
    msg = str(exc.value)
    code = getattr(exc.value, "code", "")
    assert "UNSAFE_PATH" in msg or "closure" in msg or code in {"UNSAFE_PATH", "BACKUP_INTEGRITY_FAILED"}
    # the sentinel outside the restore root must be untouched
    assert sentinel.read_text() == "outside"
    assert not (tmp_path / "escape").exists() or not (tmp_path / "escape" / "probe.source").exists()


def test_restore_rejects_duplicate_intake_in_manifest(tmp_path, source_video):
    store = _failed_intake_project(tmp_path, source_video)
    backup_workspace(store, tmp_path / "backup")
    store.close()
    evil = tmp_path / "evil2"
    shutil.copytree(tmp_path / "backup", evil)
    data = json.loads((evil / BACKUP_MANIFEST).read_text(encoding="utf-8"))
    src = dict(data["retryable_sources"][0])
    data["retryable_sources"].append(src)  # duplicate id
    (evil / BACKUP_MANIFEST).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(Exception):
        restore_workspace(evil, tmp_path / "restored2")


def test_valid_backup_restore_roundtrip_keeps_failed_source(tmp_path, source_video):
    store = _failed_intake_project(tmp_path, source_video)
    rec = store.list_intakes()[0]["id"]
    backup_workspace(store, tmp_path / "backup")
    store.close()
    restored = restore_workspace(tmp_path / "backup", tmp_path / "restored")
    assert restored["state"] == "complete"
    reopened = WorkspaceStore(tmp_path / "restored")
    reopened.open()
    q = AsyncIntakes(reopened)
    row = q.get(rec)
    assert row["state"] == "failed" and row["staged_present"] is True
    # retry works after restore (source was carried back)
    again = q.retry_failed(rec)
    assert again["state"] == "queued"
    reopened.close()


def test_reordering_retryable_sources_in_manifest_still_restores(tmp_path, source_video):
    """P01: entry order is not part of the closure semantics."""
    # Build a project with TWO failed intakes.
    store = WorkspaceStore(tmp_path / "proj")
    store.create()
    q = AsyncIntakes(store)
    for name in ("broken1.bin", "broken2.bin"):
        bad = tmp_path / name
        bad.write_bytes(b"not media " + name.encode())
        upload = store.tmp_dir / (name + ".tmp")
        shutil.copy2(bad, upload)
        q.enqueue(upload, display_name=name)
    q.run_pending()
    rows = store.list_intakes()
    assert len(rows) == 2 and all(r["state"] == "failed" for r in rows)
    manifest = backup_workspace(store, tmp_path / "backup")
    store.close()
    assert len(manifest["retryable_sources"]) == 2

    # Reverse the order of retryable_sources in a copy of the backup.
    shutil.copytree(tmp_path / "backup", tmp_path / "reordered")
    data = json.loads((tmp_path / "reordered" / BACKUP_MANIFEST).read_text(encoding="utf-8"))
    data["retryable_sources"] = list(reversed(data["retryable_sources"]))
    (tmp_path / "reordered" / BACKUP_MANIFEST).write_text(json.dumps(data), encoding="utf-8")

    receipt = restore_workspace(tmp_path / "reordered", tmp_path / "restored")
    assert receipt["state"] == "complete"
    reopened = WorkspaceStore(tmp_path / "restored")
    reopened.open()
    assert len(reopened.list_intakes()) == 2
    reopened.close()
