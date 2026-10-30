"""R03/R06 reliability: atomic admission commit, crash barriers, lease fencing."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from trace_annotate.workspace.errors import Conflict
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


def _upload(store: WorkspaceStore, source: Path) -> Path:
    copy = store.tmp_dir / "browser-upload.tmp"
    shutil.copy2(source, copy)
    return copy


def test_admission_is_atomic_task_and_intake(store, source_video):
    """media/task/revision0 and intake=succeeded commit in one transaction."""
    q = AsyncIntakes(store)
    rec = q.enqueue(_upload(store, source_video), display_name="a.mp4")
    done = q.run_pending()[0]
    assert done["state"] == "succeeded"
    assert done["task_id"]
    row = q.get(rec["id"])
    assert row["task_id"] == done["task_id"] and row["media_id"] == done["media_id"]
    # task and revision0 exist, and there is exactly one task for this intake
    task = store.get_task(done["task_id"])
    assert task["revision"] == 0
    assert len(store.list_tasks()) == 1


def test_idempotent_finish_does_not_create_second_task(store, source_video):
    q = AsyncIntakes(store)
    rec = q.enqueue(_upload(store, source_video), display_name="b.mp4")
    done = q.run_pending()[0]
    # Simulate the worker calling finish() again after a restart / duplicate loop.
    again = q.finish(rec["id"], 0, "no-such-worker", {
        "media_id": "med-fake", "task_id": "tsk-fake"})
    assert again["task_id"] == done["task_id"]
    tasks = store.list_tasks()
    assert len(tasks) == 1
    assert store.get_task(done["task_id"]) is not None


def test_interrupted_preparing_has_no_orphan_task(store, source_video):
    q = AsyncIntakes(store)
    rec = q.enqueue(_upload(store, source_video), display_name="c.mp4")
    with store.transaction() as conn:
        conn.execute("UPDATE intakes SET state='preparing', worker_id='old' WHERE id=?",
                     (rec["id"],))
    q2 = AsyncIntakes(store)
    row = q2.get(rec["id"])
    assert row["state"] == "failed" and row["task_id"] is None
    assert store.list_tasks() == []
    # retry recovers to exactly one task
    q2.retry_failed(rec["id"])
    done = q2.run_pending()[0]
    assert done["state"] == "succeeded"
    assert len(store.list_tasks()) == 1


def test_stale_worker_cannot_finish_new_generation(store, source_video):
    q = AsyncIntakes(store)
    rec = q.enqueue(_upload(store, source_video), display_name="d.mp4")
    claimed = q.claim("worker-a")
    # lease lost; row returns to queue and worker-b reclaims
    with store.transaction() as conn:
        conn.execute("UPDATE intakes SET state='queued' WHERE id=?", (rec["id"],))
    q.claim("worker-b")
    with pytest.raises(Conflict):
        q.finish(rec["id"], claimed["generation"], "worker-a",
                 {"media_id": "x", "task_id": "y"})
