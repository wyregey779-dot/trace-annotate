"""Deadline/lease fences remain truthful when a store lock delays observation."""
from contextlib import contextmanager

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes, _LeaseHeartbeat
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture
def setup_intake(tmp_path):
    store = WorkspaceStore(tmp_path / "project")
    store.create()
    now = [100.0]
    queue = AsyncIntakes(store, lease_seconds=30.0, prepare_deadline=0.6,
                         clock=lambda: now[0])
    source = store.tmp_dir / "incoming.tmp"
    source.write_bytes(b"deterministic staged bytes")
    record = queue.enqueue(source, display_name="fixture.mp4")
    try:
        yield store, queue, now, record
    finally:
        store.close()


@pytest.mark.parametrize("expiry_offset, expected", [
    (0.0, "DEADLINE_EXCEEDED"),
    (1.0, "DEADLINE_EXCEEDED"),
    (-0.1, "LEASE_LOST"),
])
def test_cancel_observation_after_lock_wait_distinguishes_earlier_lease_loss(
        setup_intake, monkeypatch, expiry_offset, expected):
    store, queue, now, _ = setup_intake
    # No background renewal: the exact lock wait and timestamp below control
    # the race without sleeping or depending on runner scheduling.
    monkeypatch.setattr(_LeaseHeartbeat, "start", lambda self: self)
    original_transaction = store.transaction

    def controlled_admission(claimed, *, cancelled, deadline_at, **kwargs):
        with original_transaction() as conn:
            conn.execute("UPDATE intakes SET lease_expires_at=? WHERE id=?",
                         (deadline_at + expiry_offset, claimed["id"]))

        @contextmanager
        def delayed_transaction():
            with original_transaction() as conn:
                # Reproduce a lock wait crossing the deadline between the
                # cancellation callback's entry and its protected row read.
                now[0] = deadline_at + 0.1
                yield conn

        with monkeypatch.context() as patch:
            patch.setattr(store, "transaction", delayed_transaction)
            assert cancelled() is True
        raise WorkspaceError("CANCELLED", "injected cooperative interruption")

    monkeypatch.setattr(store, "admit_staged_intake", controlled_admission)
    result = queue.run_once(worker_id="race-worker")
    assert result["state"] == "failed"
    assert result["error_code"] == expected
    assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("changed_column,new_value", [
    ("generation", 99), ("worker_id", "new-worker"), ("session_id", "new-session"),
])
def test_lost_owner_cannot_overwrite_a_new_generation_worker_or_session(
        setup_intake, monkeypatch, changed_column, new_value):
    store, queue, _, _ = setup_intake
    monkeypatch.setattr(_LeaseHeartbeat, "start", lambda self: self)

    def competing_admission(claimed, *, cancelled, **kwargs):
        # The column is selected from the fixed parametrization above, never
        # from user input. The old worker then observes its ownership loss.
        with store.transaction() as conn:
            conn.execute("UPDATE intakes SET " + changed_column + "=? WHERE id=?",
                         (new_value, claimed["id"]))
        assert cancelled() is True
        raise WorkspaceError("CANCELLED", "injected ownership change")

    monkeypatch.setattr(store, "admit_staged_intake", competing_admission)
    result = queue.run_once(worker_id="old-worker")
    assert result["state"] == "preparing"
    assert result[changed_column] == new_value
    assert result["error_code"] is None
    assert queue.last_heartbeat()["stop_reason"] == "lease_lost"
    assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("expiry_offset, expected", [
    (0.0, "DEADLINE_EXCEEDED"),
    (1.0, "DEADLINE_EXCEEDED"),
    (-0.1, "STALE_LEASE"),
])
def test_final_admission_fence_distinguishes_deadline_from_earlier_lease_expiry(
        setup_intake, monkeypatch, expiry_offset, expected):
    store, queue, now, record = setup_intake
    claimed = queue.claim("worker")
    deadline_at = now[0] + 0.6
    # Existing media exercises the real admission commit fence without running
    # a decoder; no task may be inserted for any of these expired attempts.
    media = {"id": S.make_id("med", claimed["source_sha256"]),
             "source": {"sha256": claimed["source_sha256"]}}
    with store.transaction() as conn:
        conn.execute("INSERT INTO media VALUES(?,?)", (media["id"], S.canonical_json(media).decode("utf-8")))
        conn.execute("UPDATE intakes SET lease_expires_at=? WHERE id=?",
                     (deadline_at + expiry_offset, record["id"]))

    original_transaction = store.transaction
    count = [0]

    @contextmanager
    def delayed_final_fence():
        with original_transaction() as conn:
            count[0] += 1
            if count[0] == 2:
                now[0] = deadline_at + 0.1
            yield conn

    monkeypatch.setattr(store, "transaction", delayed_final_fence)
    with pytest.raises(WorkspaceError) as error:
        store.admit_staged_intake(claimed, cancelled=lambda: False,
                                  generation=claimed["generation"], worker_id="worker",
                                  session_id=queue.session_id, clock=lambda: now[0],
                                  deadline_at=deadline_at)
    assert error.value.code == expected
    assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM admission_completions").fetchone()[0] == 0
