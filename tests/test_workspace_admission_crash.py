"""P02: a real process crash at each admission barrier leaves no half-write.

The child process (``tests/support/crash_admission.py``) runs a real
preparation and then exits with ``os._exit`` at one of three barriers:

* ``before_tx`` - after the prepared objects are published, before the final
  transaction opens;
* ``in_tx`` - inside the uncommitted final transaction, after the media row;
* ``after_tx`` - after the commit, before the reply is finished.

For every barrier: the reopened workspace is consistent (SQLite integrity and
foreign keys), a retry produces at most one task, and revision 0 is whole.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "support" / "crash_admission.py"
EXIT_CODES = {"before_tx": 70, "in_tx": 71, "after_tx": 72}


def _ffmpeg() -> str:
    return require_media_tools()[0]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("p02-clips") / "clip.mp4"
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=10",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, timeout=300)
    return out


def _counts(store: WorkspaceStore) -> dict:
    with store.transaction():
        return {table: int(store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("intakes", "media", "tasks", "task_revisions",
                              "artifacts", "admission_completions")}


def _assert_consistent(store: WorkspaceStore) -> None:
    with store.transaction():
        assert store.conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert store.conn.execute("PRAGMA foreign_key_check").fetchall() == []


def _prepare_project(tmp_path: Path, clip: Path) -> tuple[Path, str]:
    store = WorkspaceStore(tmp_path / "proj")
    store.create()
    upload = store.tmp_dir / "crash.mp4"
    shutil.copy2(clip, upload)
    accepted = AsyncIntakes(store).enqueue(upload, display_name="crash.mp4")
    store.close()
    return tmp_path / "proj", accepted["id"]


def _run_child(project: Path, point: str, marker: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, str(RUNNER), str(REPO_ROOT), str(project),
                           point, str(marker)],
                          capture_output=True, text=True, timeout=600, env=env)


@pytest.mark.parametrize("point", ["before_tx", "in_tx", "after_tx"])
def test_a_real_crash_at_an_admission_barrier_leaves_no_half_written_task(tmp_path, clip, point):
    project, intake_id = _prepare_project(tmp_path, clip)
    marker = tmp_path / "marker.json"
    completed = _run_child(project, point, marker)
    assert completed.returncode == EXIT_CODES[point], (completed.returncode, completed.stderr[-2000:])
    state = json.loads(marker.read_text(encoding="utf-8"))
    assert state["crashed"] == point, state
    if point == "before_tx":
        # The crash really happened after a real preparation published objects.
        assert state["prepared"] is True and state["objects"] == 4, state

    store = WorkspaceStore(project)
    store.open()
    try:
        intakes = AsyncIntakes(store)  # a new session recovers the dead attempt
        row = intakes.get(intake_id)
        _assert_consistent(store)
        counts = _counts(store)

        if point == "after_tx":
            # The commit is the last thing that happened: one task, one rev0.
            assert row["state"] == "succeeded" and row["task_id"], row
            assert row["completion"]["task_id"] == row["task_id"]
            assert counts["tasks"] == 1 and counts["task_revisions"] == 1
            assert counts["media"] == 1 and counts["admission_completions"] == 1
            task = store.get_task(row["task_id"])
            assert task["revision"] == 0 and task["annotations"] == []
            assert task["media_id"] == row["media_id"]
            # An explicit retry is refused, and a duplicate finish cannot add a task.
            with pytest.raises(WorkspaceError) as exc:
                intakes.retry_failed(intake_id)
            assert exc.value.code == "INVALID_STATE"
            assert intakes.run_pending() == []
            assert _counts(store)["tasks"] == 1
        else:
            # Nothing from the final transaction survived: no media, no task.
            assert row["state"] == "failed" and row["error_code"] == "INTERRUPTED", row
            assert row["task_id"] is None and row["media_id"] is None
            assert row["staged_present"] is True
            assert row["completion"] is None
            assert counts["tasks"] == 0 and counts["task_revisions"] == 0
            assert counts["media"] == 0 and counts["admission_completions"] == 0
            # The explicit retry produces exactly one whole task.
            intakes.retry_failed(intake_id)
            intakes.run_pending()
            done = intakes.get(intake_id)
            assert done["state"] == "succeeded" and done["task_id"], done
            counts = _counts(store)
            assert counts["tasks"] == 1 and counts["task_revisions"] == 1
            assert counts["media"] == 1 and counts["admission_completions"] == 1
            task = store.get_task(done["task_id"])
            assert task["revision"] == 0 and task["media_id"] == done["media_id"]
            _assert_consistent(store)
    finally:
        store.close()


def test_objects_published_before_a_crash_are_measured_and_swept_safely(tmp_path, clip):
    project, intake_id = _prepare_project(tmp_path, clip)
    marker = tmp_path / "marker.json"
    completed = _run_child(project, "before_tx", marker)
    assert completed.returncode == EXIT_CODES["before_tx"], completed.stderr[-2000:]

    store = WorkspaceStore(project)
    store.open()
    try:
        # Four objects were published but no media binds them yet.
        measured = store.unbound_artifacts()
        assert measured["count"] == 4, measured
        assert all(item["relative_path"].startswith("objects/") for item in measured["artifacts"])
        assert all((store.root / item["relative_path"]).is_file() for item in measured["artifacts"])
        dry = store.sweep_unbound_artifacts()
        assert dry["applied"] is False and dry["removed"] == 0
        assert store.unbound_artifacts()["count"] == 4

        # A retry commits a whole media/task; the new preparation binds its own
        # objects and leaves the crashed attempt's bytes measurable, not bound.
        intakes = AsyncIntakes(store)
        intakes.retry_failed(intake_id)
        intakes.run_pending()
        done = intakes.get(intake_id)
        assert done["state"] == "succeeded", done
        media = store.get_media(done["media_id"])
        with store.transaction():
            bindings = json.loads(store.conn.execute(
                "SELECT value FROM meta WHERE key=?", (f"media_objects:{media['id']}",)).fetchone()[0])
        bound = {reference["id"] for reference in bindings.values()}
        assert len(bound) == 4
        assert all((store.root / f"objects/{reference['sha256']}").is_file()
                   for reference in bindings.values())
        leftover = store.unbound_artifacts()
        assert leftover["count"] <= 4
        assert not (bound & {item["id"] for item in leftover["artifacts"]})

        # Disposal removes exactly the measured leftovers and nothing bound.
        swept = store.sweep_unbound_artifacts(apply=True)
        assert swept["removed"] == leftover["count"], swept
        assert store.unbound_artifacts()["count"] == 0
        assert all((store.root / f"objects/{reference['sha256']}").is_file()
                   for reference in bindings.values())
        # The task still opens and still binds the same media.
        task = store.get_task(done["task_id"])
        assert task["media_id"] == media["id"] and task["revision"] == 0
        _assert_consistent(store)
    finally:
        store.close()


def test_a_sweep_removes_only_objects_that_are_still_unbound(tmp_path, clip):
    project, intake_id = _prepare_project(tmp_path, clip)
    marker = tmp_path / "marker.json"
    assert _run_child(project, "before_tx", marker).returncode == EXIT_CODES["before_tx"]

    store = WorkspaceStore(project)
    store.open()
    try:
        measured = store.unbound_artifacts()
        assert measured["count"] == 4
        # Keep one object "bound" by an unrelated media record: a sweep must
        # leave it alone even though the crash orphaned it in this project.
        keep = measured["artifacts"][0]
        with store.transaction():
            store.conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                               ("media_objects:med-keep", json.dumps(
                                   {"playback_path": {"id": keep["id"]}})))
        swept = store.sweep_unbound_artifacts(apply=True)
        assert swept["applied"] is True and swept["removed"] == 3, swept
        assert store.unbound_artifacts()["count"] == 0
        assert (store.root / keep["relative_path"]).is_file(), "a bound object was removed"
        with store.transaction():
            remaining = store.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        assert remaining == 1
        # The staged source and the intake row are untouched by the sweep.
        intakes = AsyncIntakes(store)
        row = intakes.get(intake_id)
        assert row["state"] == "failed" and row["staged_present"] is True
        intakes.retry_failed(intake_id)
        intakes.run_pending()
        assert intakes.get(intake_id)["state"] == "succeeded"
    finally:
        store.close()
