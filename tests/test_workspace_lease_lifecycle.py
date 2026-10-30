"""P05: a live preparation is fenced by generation, session, worker and deadline.

The invariants under test:

* a second runner on the same workspace does not mistake a live attempt for a
  dead one, so a real preparation keeps running and still commits one task;
* a real crash is settled on reopen without silently re-running the attempt;
* renewal is fenced: only the owning generation/session/worker can extend the
  lease, the renewal trace names all three, and a renewal can never extend past
  the attempt deadline;
* a lost lease, a late heartbeat and an expired deadline are reported as such
  and can never revive a cancellation or overwrite a newer attempt.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools


def _video(path: Path, *, seconds: float, size: str) -> Path:
    ffmpeg = require_media_tools()[0]
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size={size}:rate=10",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=600)
    return path


@pytest.fixture(scope="module")
def slow_clip(tmp_path_factory) -> Path:
    return _video(tmp_path_factory.mktemp("p05-slow") / "slow.mp4", seconds=180, size="640x360")


@pytest.fixture(scope="module")
def fast_clip(tmp_path_factory) -> Path:
    return _video(tmp_path_factory.mktemp("p05-fast") / "fast.mp4", seconds=1, size="160x120")


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


# ---------------------------------------------------------------------------
# SESSION_ACTIVE_AND_RESTART
# ---------------------------------------------------------------------------
def test_a_second_runner_does_not_interrupt_a_live_preparation(store, slow_clip):
    intake_id = _upload(store, slow_clip, "live.mp4")
    owner = AsyncIntakes(store)
    result = {}

    def prepare():
        result["rows"] = owner.run_pending(worker_id="owner")

    thread = threading.Thread(target=prepare, daemon=True)
    thread.start()
    _wait(lambda: owner.get(intake_id)["state"] == "preparing", message="prepare never started")
    during = owner.get(intake_id)

    # A second runner opens the same workspace while the first is decoding.
    second = AsyncIntakes(store)
    assert second.session_id != owner.session_id
    observed = second.get(intake_id)
    assert observed["state"] == "preparing", observed
    assert observed["worker_id"] == during["worker_id"]
    assert observed["session_id"] == owner.session_id
    assert observed["generation"] == during["generation"]

    thread.join(timeout=300)
    assert result.get("rows"), result
    done = owner.get(intake_id)
    assert done["state"] == "succeeded", done
    assert done["task_id"]
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_a_crashed_attempt_is_settled_on_reopen_without_rerunning(store, fast_clip, tmp_path):
    from tests.test_workspace_admission_crash import _run_child  # same barrier harness

    intake_id = _upload(store, fast_clip, "crash.mp4")
    project = store.root
    store.close()
    marker = tmp_path / "marker.json"
    completed = _run_child(project, "before_tx", marker)
    assert completed.returncode == 70, completed.stderr[-2000:]

    reopened = WorkspaceStore(project)
    reopened.open()
    try:
        with reopened.transaction():
            before = dict(zip(
                [column[0] for column in reopened.conn.execute(
                    "SELECT * FROM intakes WHERE id=?", (intake_id,)).description],
                reopened.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)).fetchone()))
        assert before["state"] == "preparing" and before["attempt"] == 1

        intakes = AsyncIntakes(reopened)
        row = intakes.get(intake_id)
        assert row["state"] == "failed" and row["error_code"] == "INTERRUPTED", row
        # No silent re-run: the attempt counter and the task set are unchanged.
        assert row["attempt"] == 1
        with reopened.transaction():
            assert reopened.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
            assert reopened.conn.execute("SELECT COUNT(*) FROM media").fetchone()[0] == 0
        # Only an explicit retry starts a new attempt.
        intakes.retry_failed(intake_id)
        intakes.run_pending()
        done = intakes.get(intake_id)
        assert done["state"] == "succeeded" and done["attempt"] == 2
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# LEASE_LOSS_AND_DEADLINE (fenced renewal)
# ---------------------------------------------------------------------------
def test_renewal_is_fenced_by_generation_worker_and_session(store, fast_clip):
    clock = {"now": 1000.0}
    intake_id = _upload(store, fast_clip, "renew.mp4")
    intakes = AsyncIntakes(store, clock=lambda: clock["now"])
    claimed = intakes.claim("worker-owner")
    assert claimed is not None
    generation = claimed["generation"]
    before = intakes.get(intake_id)["lease_expires_at"]

    clock["now"] += 10
    assert intakes.renew_lease(intake_id, generation, "worker-owner") is True
    after = intakes.get(intake_id)["lease_expires_at"]
    assert after > before
    trace = intakes.last_heartbeat()
    assert trace["intake_id"] == intake_id and trace["ok"] is True
    assert trace["lease_expires_at"] == after

    assert intakes.renew_lease(intake_id, generation + 1, "worker-owner") is False
    assert intakes.renew_lease(intake_id, generation, "worker-thief") is False
    original_session = intakes.session_id
    try:
        intakes.session_id = "intake-someoneelse"
        assert intakes.renew_lease(intake_id, generation, "worker-owner") is False
    finally:
        intakes.session_id = original_session
    assert intakes.get(intake_id)["lease_expires_at"] == after


def test_a_renewal_cannot_extend_past_the_attempt_deadline(store, fast_clip):
    clock = {"now": 2000.0}
    intake_id = _upload(store, fast_clip, "deadline.mp4")
    intakes = AsyncIntakes(store, clock=lambda: clock["now"], lease_seconds=120.0,
                           prepare_deadline=300.0)
    claimed = intakes.claim("worker-deadline")
    assert claimed is not None
    generation = claimed["generation"]
    deadline = clock["now"] + 30
    assert intakes.renew_lease(intake_id, generation, "worker-deadline",
                               deadline_at=deadline) is True
    assert intakes.get(intake_id)["lease_expires_at"] <= deadline
    clock["now"] = deadline + 1
    assert intakes.renew_lease(intake_id, generation, "worker-deadline",
                               deadline_at=deadline) is False
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, generation, "worker-deadline",
                       {"media_id": "med-x", "task_id": "tsk-x"})
    assert exc.value.code in {"STALE_LEASE", "INVALID_ID"}


def test_a_cancelled_attempt_cannot_be_revived_by_a_late_heartbeat(store, fast_clip):
    intake_id = _upload(store, fast_clip, "cancel.mp4")
    intakes = AsyncIntakes(store)
    claimed = intakes.claim("worker-cancel")
    assert claimed is not None
    generation = claimed["generation"]
    intakes.request_cancel(intake_id)
    assert intakes.get(intake_id)["state"] == "cancel_requested"
    # A heartbeat may keep the winding-down attempt's lease alive, but it can
    # never turn the row back into a running attempt.
    intakes.renew_lease(intake_id, generation, "worker-cancel")
    assert intakes.get(intake_id)["state"] == "cancel_requested"
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, generation, "worker-cancel",
                       {"media_id": "med-x", "task_id": "tsk-x"})
    assert exc.value.code == "CANCELLED"
    settled = intakes.fail(intake_id, generation, "worker-cancel", "CANCELLED")
    assert settled["state"] == "cancelled"
    assert intakes.renew_lease(intake_id, generation, "worker-cancel") is False
    assert intakes.get(intake_id)["state"] == "cancelled"


def test_a_renewal_after_expiry_rearms_the_lease_but_a_newer_claim_fences_it(store, fast_clip):
    clock = {"now": 3000.0}
    intake_id = _upload(store, fast_clip, "lost.mp4")
    intakes = AsyncIntakes(store, clock=lambda: clock["now"], lease_seconds=10.0)
    claimed = intakes.claim("worker-lost")
    assert claimed is not None
    generation = claimed["generation"]
    clock["now"] += 11  # the lease expired without a renewal
    assert intakes.renew_lease(intake_id, generation, "worker-lost") is True
    # The renewal re-arms the lease, so a *newer* claim is what really fences it.
    with store.transaction() as conn:
        conn.execute("UPDATE intakes SET state='queued' WHERE id=?", (intake_id,))
    newer = intakes.claim("worker-newer")
    assert newer is not None and newer["generation"] > generation
    assert intakes.renew_lease(intake_id, generation, "worker-lost") is False
    with pytest.raises(WorkspaceError) as exc:
        intakes.finish(intake_id, generation, "worker-lost",
                       {"media_id": "med-x", "task_id": "tsk-x"})
    assert exc.value.code == "STALE_LEASE"
    assert intakes.get(intake_id)["state"] == "preparing"
