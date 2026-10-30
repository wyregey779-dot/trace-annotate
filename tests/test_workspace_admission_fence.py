"""P02: cancellation, stale fences and finish binding at the admission boundary.

The invariants under test:

* a user cancellation that wins before the final transaction settles the intake
  as ``cancelled`` with no task; a commit that wins is reported as success and a
  later cancel changes nothing;
* a late success or a late failure from an older generation, a foreign worker,
  a foreign session or an expired lease is refused by the fence and never
  overwrites the row the new worker owns;
* ``finish`` refuses a nonexistent task, a task that belongs to another media,
  and any attempt to rebind an already succeeded intake, so a fake result can
  never be written as success;
* a later human edit survives every late or duplicate admission message.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools


def _ffmpeg() -> str:
    return require_media_tools()[0]


def _video(path: Path, *, seconds: float, size: str) -> Path:
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size={size}:rate=10",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=600)
    return path


@pytest.fixture(scope="module")
def slow_clip(tmp_path_factory) -> Path:
    return _video(tmp_path_factory.mktemp("p02-slow") / "slow.mp4", seconds=180, size="640x360")


@pytest.fixture(scope="module")
def fast_clip(tmp_path_factory) -> Path:
    return _video(tmp_path_factory.mktemp("p02-fast") / "fast.mp4", seconds=1, size="160x120")


@pytest.fixture()
def store(tmp_path) -> WorkspaceStore:
    value = WorkspaceStore(tmp_path / "proj")
    value.create()
    yield value
    try:
        value.close()
    except Exception:
        pass


def _upload(store: WorkspaceStore, source: Path, name: str) -> str:
    target = store.tmp_dir / name
    shutil.copy2(source, target)
    return AsyncIntakes(store).enqueue(target, display_name=name)["id"]


def _wait(predicate, *, budget=120.0, interval=0.01, message="condition not reached"):
    deadline = time.monotonic() + budget
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"{message} within {budget:.0f}s (last={last!r})")


def _counts(store: WorkspaceStore) -> dict:
    with store.transaction():
        return {table: int(store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("media", "tasks", "task_revisions", "admission_completions")}


def _run_in_thread(intakes: AsyncIntakes, worker: str):
    result = {}

    def target():
        try:
            result["value"] = intakes.run_pending(worker_id=worker)
        except BaseException as exc:  # surfaced by the test
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, result


# ---------------------------------------------------------------------------
# cancel_wins / commit_wins
# ---------------------------------------------------------------------------
def test_a_cancel_that_wins_leaves_a_settled_cancellation(store, slow_clip):
    intake_id = _upload(store, slow_clip, "slow.mp4")
    intakes = AsyncIntakes(store)
    thread, result = _run_in_thread(intakes, "worker-cancel")
    _wait(lambda: intakes.get(intake_id)["state"] == "preparing",
          message="preparation never started")
    receipt = intakes.request_cancel(intake_id)
    assert receipt["state"] in {"preparing", "cancel_requested", "cancelled"}, receipt
    thread.join(timeout=180)
    assert "error" not in result, result

    row = intakes.get(intake_id)
    assert row["state"] == "cancelled" and row["error_code"] == "CANCELLED", row
    assert row["task_id"] is None and row["media_id"] is None
    assert row["completion"] is None
    assert row["staged_present"] is True
    assert _counts(store) == {"media": 0, "tasks": 0, "task_revisions": 0,
                              "admission_completions": 0}
    # A cancelled intake is terminal: it cannot be retried into a task.
    with pytest.raises(WorkspaceError) as exc:
        intakes.retry_failed(intake_id)
    assert exc.value.code == "INVALID_STATE"


def test_a_commit_that_wins_is_not_undone_by_a_late_cancel(store, fast_clip):
    intake_id = _upload(store, fast_clip, "fast.mp4")
    intakes = AsyncIntakes(store)
    intakes.run_pending()
    done = intakes.get(intake_id)
    assert done["state"] == "succeeded", done
    receipt = intakes.request_cancel(intake_id)
    assert receipt["state"] == "succeeded" and receipt["task_id"] == done["task_id"]
    row = intakes.get(intake_id)
    assert row == done
    assert _counts(store) == {"media": 1, "tasks": 1, "task_revisions": 1,
                              "admission_completions": 1}


# ---------------------------------------------------------------------------
# late_success / late_failure / worker_mismatch / session_mismatch / lease_lost
# ---------------------------------------------------------------------------
def _claimed_then_reclaimed(store: WorkspaceStore, clip: Path) -> tuple[AsyncIntakes, dict, dict]:
    """Claim with one worker, force the row back to the queue, claim again."""
    intake_id = _upload(store, clip, "fence.mp4")
    intakes = AsyncIntakes(store)
    first = intakes.claim("worker-old")
    assert first is not None and first["id"] == intake_id
    with store.transaction() as conn:
        conn.execute("UPDATE intakes SET state='queued' WHERE id=?", (intake_id,))
    second = intakes.claim("worker-new")
    assert second is not None and second["generation"] > first["generation"]
    return intakes, first, second


def test_a_late_success_from_an_old_generation_is_refused(store, fast_clip):
    intakes, first, second = _claimed_then_reclaimed(store, fast_clip)
    # A real committed task from another intake, used as a plausible payload.
    other_id = _upload(store, fast_clip, "other.mp4")
    intakes.run_pending()
    other = intakes.get(other_id)
    assert other["state"] == "succeeded"
    before = intakes.get(second["id"])
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(second["id"], first["generation"], "worker-old",
                       {"media_id": other["media_id"], "task_id": other["task_id"]})
    assert exc.value.code == "STALE_LEASE"
    assert intakes.get(second["id"]) == before, "a stale success changed the row"
    assert _counts(store)["tasks"] == 1


def test_a_late_failure_from_an_old_generation_is_refused(store, fast_clip):
    intakes, first, second = _claimed_then_reclaimed(store, fast_clip)
    before = intakes.get(second["id"])
    returned = intakes.fail(second["id"], first["generation"], "worker-old", "INTERRUPTED")
    assert returned["state"] == before["state"] == "preparing"
    assert returned["error_code"] is None
    assert returned["worker_id"] == "worker-new"
    assert returned["generation"] == second["generation"]


def test_a_foreign_worker_or_session_cannot_finish(store, fast_clip):
    intake_id = _upload(store, fast_clip, "foreign.mp4")
    intakes = AsyncIntakes(store)
    claimed = intakes.claim("worker-owner")
    assert claimed is not None
    result = {"media_id": S.make_id("med", "0" * 64), "task_id": S.make_id("tsk", "x")}
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, claimed["generation"], "worker-thief", result)
    assert exc.value.code == "STALE_LEASE"
    original_session = intakes.session_id
    try:
        intakes.session_id = "intake-someoneelse"
        with pytest.raises(WorkspaceError) as exc:
            intakes.finish(intake_id, claimed["generation"], "worker-owner", result)
        assert exc.value.code == "STALE_LEASE"
    finally:
        intakes.session_id = original_session
    row = intakes.get(intake_id)
    assert row["state"] == "preparing" and row["task_id"] is None


def test_an_expired_lease_cannot_finish(store, fast_clip):
    intake_id = _upload(store, fast_clip, "expired.mp4")
    intakes = AsyncIntakes(store)
    claimed = intakes.claim("worker-slow")
    assert claimed is not None
    with store.transaction() as conn:
        conn.execute("UPDATE intakes SET lease_expires_at=0 WHERE id=?", (intake_id,))
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, claimed["generation"], "worker-slow",
                       {"media_id": S.make_id("med", "0" * 64), "task_id": S.make_id("tsk", "x")})
    assert exc.value.code == "STALE_LEASE"
    assert intakes.get(intake_id)["state"] == "preparing"


# ---------------------------------------------------------------------------
# FINISH_BINDING_VALIDATION
# ---------------------------------------------------------------------------
def test_finish_refuses_a_task_that_does_not_exist(store, fast_clip):
    intake_id = _upload(store, fast_clip, "ghost.mp4")
    intakes = AsyncIntakes(store)
    claimed = intakes.claim("worker-ghost")
    assert claimed is not None
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, claimed["generation"], "worker-ghost",
                       {"media_id": S.make_id("med", "0" * 64), "task_id": "tsk-ghost000000"})
    assert exc.value.code == "TASK_NOT_FOUND"
    row = intakes.get(intake_id)
    assert row["state"] == "preparing" and row["task_id"] is None


def test_finish_refuses_a_task_bound_to_another_media(store, fast_clip):
    real_id = _upload(store, fast_clip, "real.mp4")
    intakes = AsyncIntakes(store)
    intakes.run_pending()
    real = intakes.get(real_id)
    assert real["state"] == "succeeded"

    intake_id = _upload(store, fast_clip, "mismatch.mp4")
    claimed = intakes.claim("worker-mismatch")
    assert claimed is not None and claimed["id"] == intake_id
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, claimed["generation"], "worker-mismatch",
                       {"media_id": S.make_id("med", "1" * 64), "task_id": real["task_id"]})
    assert exc.value.code == "FINISH_BINDING_MISMATCH"
    row = intakes.get(intake_id)
    assert row["state"] == "preparing" and row["task_id"] is None
    assert _counts(store)["tasks"] == 1


def test_an_existing_success_binding_is_never_changed(store, fast_clip):
    intake_id = _upload(store, fast_clip, "bound.mp4")
    intakes = AsyncIntakes(store)
    intakes.run_pending()
    done = intakes.get(intake_id)
    assert done["state"] == "succeeded"
    # The same binding is an idempotent confirmation.
    assert intakes.finish(intake_id, 0, "any-worker",
                          {"media_id": done["media_id"], "task_id": done["task_id"]}) == done
    # A different binding is refused and changes nothing.
    other_id = _upload(store, fast_clip, "bound-other.mp4")
    intakes.run_pending()
    other = intakes.get(other_id)
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, 0, "any-worker",
                       {"media_id": other["media_id"], "task_id": other["task_id"]})
    assert exc.value.code == "FINISH_BINDING_MISMATCH"
    assert intakes.get(intake_id) == done


# ---------------------------------------------------------------------------
# human_edit_unchanged
# ---------------------------------------------------------------------------
def test_a_late_admission_message_cannot_overwrite_a_human_edit(store, fast_clip):
    intake_id = _upload(store, fast_clip, "edited.mp4")
    intakes = AsyncIntakes(store)
    intakes.run_pending()
    done = intakes.get(intake_id)
    assert done["state"] == "succeeded"
    task = store.get_task(done["task_id"])
    store.apply_task_command(
        command_id="cmd-human-edit", task_id=task["id"], expected_revision=0,
        expected_sha256=task["content_sha256"], operation="add_annotation",
        payload={"annotation": {
            "id": "ann-human", "type": "segment", "label": "working", "text": "human",
            "tags": [], "track": "main", "attributes": {},
            "provenance": {"origin": "manual", "actor_id": "user", "job_id": None},
            "evidence_ids": [],
            "start": {"seconds": S.rational(0.0)}, "end": {"seconds": S.rational(1.0)},
        }}, request_body=b"{}")
    edited = store.get_task(task["id"])
    assert edited["revision"] == 1

    # Duplicate finish and a stale failure both leave the edit alone.
    assert intakes.finish(intake_id, 0, "worker-old",
                          {"media_id": done["media_id"], "task_id": done["task_id"]})["state"] == "succeeded"
    intakes.fail(intake_id, 0, "worker-old", "INTERRUPTED")
    after = store.get_task(task["id"])
    assert after == edited
    assert after["annotations"][0]["id"] == "ann-human"
    assert intakes.get(intake_id)["task_id"] == done["task_id"]
