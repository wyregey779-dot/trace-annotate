"""Async staged admission (T08): enqueue, single prepare, cancel, retry, recovery."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from trace_annotate.temporal.media import stream_sha256
from trace_annotate.workspace.errors import Conflict, NotFound
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


@pytest.fixture()
def store(tmp_path: Path):
    s = WorkspaceStore(tmp_path / "proj")
    s.create()
    yield s
    s.close()


def _staging_upload(store: WorkspaceStore, source: Path) -> Path:
    copy = store.tmp_dir / "browser-upload.tmp"
    shutil.copy2(source, copy)
    return copy


def test_enqueue_then_run_creates_task(store, source_video):
    q = AsyncIntakes(store)
    upload = _staging_upload(store, source_video)
    receipt = q.enqueue(upload, display_name="incoming.mp4")
    assert receipt["state"] == "queued"
    assert receipt["task_id"] is None
    results = q.run_pending()
    assert len(results) == 1
    done = results[0]
    assert done["state"] == "succeeded"
    assert done["task_id"] and done["media_id"]
    task = store.get_task(done["task_id"])
    assert task["revision"] == 0
    # a second drain has nothing to do -> exactly one task per intake
    assert q.run_pending() == []
    assert len(store.list_tasks()) == 1


def test_enqueue_idempotent_replay_returns_same_receipt(store, source_video):
    q = AsyncIntakes(store)
    upload = _staging_upload(store, source_video)
    r1 = q.enqueue(upload, display_name="a.mp4", command_id="cmd-up-1")
    upload2 = _staging_upload(store, source_video)
    r2 = q.enqueue(upload2, display_name="a.mp4", command_id="cmd-up-1")
    assert r1["id"] == r2["id"]
    # same command id but different declared bytes -> conflict, not a second intake
    upload3 = _staging_upload(store, source_video)
    with pytest.raises(Conflict):
        q.enqueue(upload3, display_name="a.mp4", command_id="cmd-up-1",
                  expected_sha256="0" * 64)


def test_declared_hash_mismatch_rejected_before_staging(store, source_video):
    q = AsyncIntakes(store)
    upload = _staging_upload(store, source_video)
    with pytest.raises(Conflict) as exc:
        q.enqueue(upload, display_name="x.mp4", expected_sha256="f" * 64)
    assert exc.value.code == "MEDIA_IDENTITY_MISMATCH"
    assert store.list_intakes() == [] or all(i["task_id"] is None for i in store.list_intakes())


def test_cancel_queued_intake(store, source_video):
    q = AsyncIntakes(store)
    upload = _staging_upload(store, source_video)
    rec = q.enqueue(upload, display_name="c.mp4")
    out = q.request_cancel(rec["id"])
    assert out["state"] == "cancelled" and out["error_code"] == "CANCELLED"
    assert q.run_pending() == []
    assert store.list_tasks() == []


def test_interrupted_preparing_marked_failed_on_restart(store, source_video):
    q = AsyncIntakes(store)
    upload = _staging_upload(store, source_video)
    rec = q.enqueue(upload, display_name="z.mp4")
    # simulate a previous process that died mid-preparation
    store.conn.execute("UPDATE intakes SET state='preparing', worker_id='old', session_id='old' WHERE id=?",
                       (rec["id"],))
    q2 = AsyncIntakes(store)
    row = q2.get(rec["id"])
    assert row["state"] == "failed" and row["error_code"] == "INTERRUPTED"
    assert row["task_id"] is None
    # explicit retry puts it back in the queue and then succeeds
    again = q2.retry_failed(rec["id"])
    assert again["state"] == "queued"
    done = q2.run_pending()[0]
    assert done["state"] == "succeeded" and done["task_id"]


def test_bad_media_fails_and_can_be_retried(store, tmp_path):
    q = AsyncIntakes(store)
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a real media file at all" * 4)
    copy = store.tmp_dir / "up-bad.tmp"
    shutil.copy2(bad, copy)
    rec = q.enqueue(copy, display_name="broken.mp4")
    done = q.run_pending()[0]
    assert done["state"] == "failed"
    assert done["error_code"] in {"UNSUPPORTED_MEDIA", "MEDIA_IO_ERROR", "ADMISSION_FAILED"}
    # staged input is preserved for an explicit retry, not silently discarded
    assert q.get(rec["id"])["staged_present"] is True
    again = q.retry_failed(rec["id"])
    assert again["state"] == "queued"
    rerun = q.run_pending()[0]
    assert rerun["state"] == "failed"
