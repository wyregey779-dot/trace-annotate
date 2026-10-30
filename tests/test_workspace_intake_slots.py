"""P04/P05/P06: single cancellable prepare slot, fenced lease, ordered shutdown.

These tests assert behaviour, not implementation shape:

* a queued intake keeps **no** lease while it waits for the decoder (P04);
* real ffmpeg/ffprobe subprocesses never overlap inside one process (P04);
* a long prepare is kept alive by a fenced heartbeat that cannot pass the
  attempt deadline, and a stale generation cannot renew or commit (P05);
* closing stops new admission and new claims, interrupts a slot wait, and
  reports whether the worker actually stopped (P06).

Media-process concurrency is measured by counting live ``subprocess``
invocations made by the preparation module, i.e. real ffmpeg/ffprobe processes,
not Python threads.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.temporal import preparation
from trace_annotate.workspace import media as media_module
from trace_annotate.workspace.errors import Conflict, WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.slots import PrepareSlot, SlotClosed, SlotTimeout, get_prepare_slot
from trace_annotate.workspace.store import WorkspaceStore

FFMPEG = shutil.which("ffmpeg")


def _make_video(path: Path, *, seconds: float, pattern: str, size: str = "160x120") -> Path:
    if FFMPEG is None:
        pytest.skip("ffmpeg is required for intake slot tests")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"{pattern}=duration={seconds}:size={size}:rate=10",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture()
def store(tmp_path: Path):
    s = WorkspaceStore(tmp_path / "proj")
    s.create()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def upload(store: WorkspaceStore, tmp_path: Path):
    def _stage(name: str, pattern: str = "testsrc", seconds: float = 1.0) -> Path:
        source = _make_video(tmp_path / f"{name}.mp4", seconds=seconds, pattern=pattern)
        target = store.tmp_dir / f"upload-{name}.tmp"
        shutil.copy2(source, target)
        return target
    return _stage


@pytest.fixture()
def process_counter(monkeypatch):
    """Count live ffmpeg/ffprobe processes actually spawned by preparation."""
    state = {"live": 0, "peak": 0, "spawned": 0}
    guard = threading.Lock()
    original = preparation._run

    def wrapped(*args, **kwargs):
        with guard:
            state["live"] += 1
            state["spawned"] += 1
            state["peak"] = max(state["peak"], state["live"])
        try:
            return original(*args, **kwargs)
        finally:
            with guard:
                state["live"] -= 1

    monkeypatch.setattr(preparation, "_run", wrapped)
    return state


# -- P04: slot semantics ---------------------------------------------------
def test_slot_is_reentrant_for_the_owning_thread():
    slot = PrepareSlot()
    slot.acquire()
    assert slot.depth == 1
    slot.acquire()          # the media layer re-enters for the same intake
    assert slot.depth == 2
    slot.release()
    assert slot.depth == 1
    assert slot.stats()["peak"] == 1
    slot.release()
    assert slot.depth == 0


def test_slot_wait_honours_cancellation_within_budget():
    slot = PrepareSlot()
    slot.acquire()
    cancelled = threading.Event()
    outcome = {}

    def waiter():
        started = time.monotonic()
        try:
            slot.acquire(cancelled=cancelled.is_set)
            outcome["result"] = "acquired"
        except InterruptedError:
            outcome["result"] = "cancelled"
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.15)
    cancelled.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome["result"] == "cancelled"
    # Cancellation must be observed within a few poll intervals, not the lease.
    assert outcome["elapsed"] < 2.0
    slot.release()


def test_slot_wait_stops_when_workspace_closes():
    slot = PrepareSlot()
    slot.acquire()
    closing = threading.Event()
    outcome = {}

    def waiter():
        try:
            slot.acquire(closing=closing.is_set)
            outcome["result"] = "acquired"
        except SlotClosed:
            outcome["result"] = "closed"

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.1)
    closing.set()
    thread.join(timeout=5)
    assert outcome["result"] == "closed"
    slot.release()


def test_slot_wait_timeout_does_not_take_the_slot():
    slot = PrepareSlot()
    slot.acquire()
    outcome = {}

    def waiter():
        try:
            slot.acquire(timeout=0.2)
            outcome["result"] = "acquired"
        except SlotTimeout:
            outcome["result"] = "timeout"
        outcome["depth"] = slot.depth

    thread = threading.Thread(target=waiter)
    thread.start()
    thread.join(timeout=5)
    assert outcome == {"result": "timeout", "depth": 0}, outcome
    slot.release()
    # The failed waiter must not have consumed the slot for anyone else.
    slot.acquire(timeout=0.2)
    assert slot.depth == 1
    slot.release()


# -- P04: a waiting intake must not burn its lease -------------------------
def test_queued_intake_keeps_no_lease_while_the_slot_is_busy(store):
    """The core P04 defect: claim happened before the slot was reserved."""
    q = AsyncIntakes(store)
    bad = store.tmp_dir / "not-media.bin"
    bad.write_bytes(b"not media" * 64)
    rec = q.enqueue(bad, display_name="waiting.mp4")

    slot = get_prepare_slot(store.root)
    slot.acquire()
    outcome = {}

    def worker():
        outcome["result"] = q.run_once(worker_id="intake-worker")

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        # While the slot is held the row must stay untouched: no claim, no
        # attempt bump, no lease expiry counting down against the user.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            row = q.get(rec["id"])
            assert row["state"] == "queued", row
            assert row["worker_id"] is None, row
            assert row["lease_expires_at"] is None, row
            assert row["attempt"] == 0, row
            time.sleep(0.05)
        assert thread.is_alive(), "worker should still be waiting for the slot"
    finally:
        slot.release()
    thread.join(timeout=60)
    assert not thread.is_alive()
    final = q.get(rec["id"])
    # The row was claimed only after the slot became free, then failed honestly.
    assert final["state"] == "failed"
    assert final["attempt"] == 1
    assert final["worker_id"] is None


def test_two_runners_never_overlap_media_processes(store, upload, process_counter):
    q = AsyncIntakes(store)
    for index, pattern in enumerate(("testsrc", "smptebars", "rgbtestsrc")):
        q.enqueue(upload(f"clip{index}", pattern=pattern, seconds=1.0),
                  display_name=f"clip{index}.mp4")

    get_prepare_slot(store.root).reset_stats()
    results = []
    guard = threading.Lock()

    def runner(name: str):
        local = q.run_pending(worker_id=name)
        with guard:
            results.extend(local)

    threads = [threading.Thread(target=runner, args=(f"runner-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 3, results
    assert {row["state"] for row in results} == {"succeeded"}
    # Real evidence: at most one ffmpeg/ffprobe child existed at any instant.
    assert process_counter["peak"] == 1, process_counter
    assert get_prepare_slot(store.root).stats()["peak"] == 1


def test_sync_and_async_prepare_share_one_slot(store, upload, process_counter):
    """A synchronous CLI import and the async drainer must not decode together."""
    from trace_annotate.workspace.media import prepare_workspace_media

    q = AsyncIntakes(store)
    q.enqueue(upload("async", pattern="testsrc", seconds=1.0), display_name="async.mp4")
    get_prepare_slot(store.root).reset_stats()

    errors = []

    def async_side():
        try:
            q.run_pending(worker_id="async-worker")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=async_side)
    thread.start()
    try:
        sync_source = upload("sync", pattern="smptebars", seconds=1.0)
        target = store.tmp_dir / "sync-work"
        prepare_workspace_media(sync_source, target)
    finally:
        thread.join(timeout=180)

    assert not errors, errors
    assert process_counter["peak"] == 1, process_counter
    assert get_prepare_slot(store.root).stats()["peak"] == 1


# -- P05: fenced renewal and a hard attempt deadline -----------------------
def test_long_prepare_is_kept_alive_by_fenced_renewal(store, upload, monkeypatch):
    q = AsyncIntakes(store, lease_seconds=1.0, prepare_deadline=60.0)
    q.enqueue(upload("long", pattern="testsrc", seconds=1.0), display_name="long.mp4")

    real = media_module.prepare_workspace_media

    def slow_prepare(source, output_dir, **kwargs):
        # Deliberate delay so the attempt outlives the original 1s lease.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            check = kwargs.get("cancelled")
            if check is not None and check():
                raise WorkspaceError("CANCELLED", "cancelled during slow prepare")
            time.sleep(0.05)
        return real(source, output_dir, **kwargs)

    monkeypatch.setattr(media_module, "prepare_workspace_media", slow_prepare)
    results = q.run_pending(worker_id="long-worker")
    assert len(results) == 1
    assert results[0]["state"] == "succeeded", results[0]
    report = q.last_heartbeat()
    assert report["renewals"] >= 1, report
    assert report["lease_lost"] is False, report
    assert report["deadline_exceeded"] is False, report
    assert report["elapsed_seconds"] > 1.0, report


def test_renewal_never_extends_past_the_attempt_deadline(store):
    clock = {"now": 1000.0}
    q = AsyncIntakes(store, lease_seconds=120.0, clock=lambda: clock["now"])
    bad = store.tmp_dir / "x.bin"
    bad.write_bytes(b"x" * 16)
    rec = q.enqueue(bad, display_name="x.mp4")
    claimed = q.claim("w1")
    assert claimed["id"] == rec["id"]
    generation = claimed["generation"]
    deadline_at = clock["now"] + 10.0
    assert q.renew_lease(rec["id"], generation, "w1", deadline_at=deadline_at) is True
    row = q.get(rec["id"])
    assert row["lease_expires_at"] <= deadline_at
    # A stale generation may not renew, so an old attempt cannot resurrect.
    assert q.renew_lease(rec["id"], generation - 1, "w1") is False
    # A different worker/session may not renew either.
    assert q.renew_lease(rec["id"], generation, "w2") is False


def test_deadline_exceeded_fails_the_attempt(store, upload, monkeypatch):
    q = AsyncIntakes(store, lease_seconds=30.0, prepare_deadline=0.6)
    q.enqueue(upload("slow", pattern="testsrc", seconds=1.0), display_name="slow.mp4")

    def stalled(source, output_dir, **kwargs):
        check = kwargs.get("cancelled")
        for _ in range(200):
            if check is not None and check():
                raise WorkspaceError("CANCELLED", "stopped by deadline")
            time.sleep(0.02)
        raise AssertionError("deadline was never observed")

    monkeypatch.setattr(media_module, "prepare_workspace_media", stalled)
    results = q.run_pending(worker_id="deadline-worker")
    assert len(results) == 1
    assert results[0]["state"] == "failed", results[0]
    assert results[0]["error_code"] == "DEADLINE_EXCEEDED", results[0]
    report = q.last_heartbeat()
    assert report["stop_reason"] == "deadline", report
    # Whichever observer noticed first, the attempt must have been allowed to run
    # up to its deadline and must not have been killed early or extended past it.
    assert report["renewals"] >= 1, report
    assert report["elapsed_seconds"] >= 0.5, report
    assert report["elapsed_seconds"] < 5.0, report


def test_stale_generation_cannot_commit(store, upload):
    q = AsyncIntakes(store)
    q.enqueue(upload("stale", pattern="testsrc", seconds=1.0), display_name="stale.mp4")
    claimed = q.claim("w1")
    with pytest.raises(Conflict) as excinfo:
        q.finish(claimed["id"], claimed["generation"] + 1, "w1",
                 {"media_id": "med-x", "task_id": "tsk-x"})
    assert excinfo.value.code == "STALE_LEASE"
    # And a lease that already expired may not commit either.
    with store.transaction() as conn:
        conn.execute("UPDATE intakes SET lease_expires_at=0 WHERE id=?", (claimed["id"],))
    with pytest.raises(Conflict) as excinfo:
        q.finish(claimed["id"], claimed["generation"], "w1",
                 {"media_id": "med-x", "task_id": "tsk-x"})
    assert excinfo.value.code == "STALE_LEASE"


def test_retry_uses_an_attempt_scoped_work_directory(store, upload):
    q = AsyncIntakes(store)
    rec = q.enqueue(upload("retry", pattern="testsrc", seconds=1.0), display_name="retry.mp4")
    # Force a first failure by making the first prepare explode.
    original = media_module.prepare_workspace_media
    calls = {"n": 0}

    def flaky(source, output_dir, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "attempt-one.evidence").write_text("first attempt", encoding="utf-8")
            raise WorkspaceError("ADMISSION_FAILED", "synthetic first-attempt failure")
        return original(source, output_dir, **kwargs)

    import trace_annotate.workspace.store as store_module
    store_module_prepare = original
    try:
        media_module.prepare_workspace_media = flaky
        first = q.run_pending(worker_id="w1")
    finally:
        media_module.prepare_workspace_media = store_module_prepare
    assert first[0]["state"] == "failed", first[0]
    assert first[0]["attempt"] == 1
    work_dirs = sorted(p.name for p in store.tmp_dir.iterdir() if p.is_dir() and p.name.startswith(rec["id"]))
    assert any(name.endswith("-a1-g1") for name in work_dirs), work_dirs
    # The first attempt's evidence survives; a retry gets its own directory.
    assert (store.tmp_dir / f"{rec['id']}-a1-g1" / "attempt-one.evidence").is_file()
    q.retry_failed(rec["id"])
    second = q.run_pending(worker_id="w1")
    assert second[0]["state"] == "succeeded", second[0]
    # ``attempt`` counts real prepares: the retry is attempt 2, not attempt 3.
    assert second[0]["attempt"] == 2, second[0]
    assert (store.tmp_dir / f"{rec['id']}-a1-g1" / "attempt-one.evidence").is_file()
    retry_dirs = [p.name for p in store.tmp_dir.iterdir()
                  if p.is_dir() and p.name.startswith(rec["id"]) and p.name != f"{rec['id']}-a1-g1"]
    assert retry_dirs, "the retry must prepare in its own attempt-scoped directory"


# -- P06: ordered shutdown -------------------------------------------------
def test_close_refuses_new_admission_and_claims(store, upload):
    closing = threading.Event()
    q = AsyncIntakes(store, closing=closing)
    rec = q.enqueue(upload("before-close", pattern="testsrc", seconds=1.0), display_name="a.mp4")
    report = q.close(timeout=5)
    assert report["stopped"] is True, report
    assert q.claim("w1") is None
    assert q.run_once("w1") is None
    with pytest.raises(Conflict) as excinfo:
        q.enqueue(upload("after-close", pattern="smptebars", seconds=1.0), display_name="b.mp4")
    assert excinfo.value.code == "WORKSPACE_CLOSING"
    # Accepted work is preserved for the next process, not silently cancelled.
    assert q.get(rec["id"])["state"] == "queued"


def test_close_interrupts_a_slot_wait_promptly(store):
    closing = threading.Event()
    q = AsyncIntakes(store, closing=closing)
    slot = get_prepare_slot(store.root)
    slot.acquire()
    started = time.monotonic()
    outcome = {}

    def worker():
        outcome["result"] = q.run_once(worker_id="w1")
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.2)
    report = q.close(timeout=5)
    thread.join(timeout=10)
    try:
        assert not thread.is_alive()
        assert outcome["result"] is None
        assert outcome["elapsed"] < 5.0
        assert report["stopped"] is True, report
    finally:
        slot.release()


def test_close_reports_a_worker_that_does_not_stop(store, upload, monkeypatch):
    q = AsyncIntakes(store, lease_seconds=30.0, prepare_deadline=120.0)
    q.enqueue(upload("stuck", pattern="testsrc", seconds=1.0), display_name="stuck.mp4")

    release = threading.Event()

    def ignores_cancel(source, output_dir, **kwargs):
        release.wait(timeout=30)
        raise WorkspaceError("CANCELLED", "released")

    monkeypatch.setattr(media_module, "prepare_workspace_media", ignores_cancel)
    thread = threading.Thread(target=q.run_pending, kwargs={"worker_id": "stuck-worker"})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and q.active_intake() is None:
            time.sleep(0.02)
        report = q.close(timeout=0.5)
        assert report["stopped"] is False, report
        assert report["active_intake"] is not None
    finally:
        release.set()
        thread.join(timeout=30)
    assert q.close(timeout=5)["stopped"] is True


# -- P04: every real entry point shares the workspace slot -----------------
def test_error_path_releases_the_slot(store, upload, monkeypatch):
    """A prepare that raises must not keep the workspace slot."""
    q = AsyncIntakes(store)
    q.enqueue(upload("boom", pattern="testsrc", seconds=1.0), display_name="boom.mp4")

    def explode(source, output_dir, **kwargs):
        raise WorkspaceError("ADMISSION_FAILED", "synthetic failure")

    monkeypatch.setattr(media_module, "prepare_workspace_media", explode)
    results = q.run_pending(worker_id="w1")
    assert results[0]["state"] == "failed"
    slot = get_prepare_slot(store.root)
    assert slot.depth == 0
    assert slot.stats()["held"] is False
    # A later unit of work must still be able to take the slot.
    monkeypatch.undo()
    q.retry_failed(results[0]["id"])
    again = q.run_pending(worker_id="w1")
    assert again[0]["state"] == "succeeded", again[0]


def test_two_workspaces_prepare_independently(tmp_path, process_counter):
    """An unrelated workspace must not be blocked by another workspace's decode."""
    first = WorkspaceStore(tmp_path / "ws-a")
    second = WorkspaceStore(tmp_path / "ws-b")
    first.create()
    second.create()
    try:
        slot_a = get_prepare_slot(first.root)
        slot_a.acquire()
        started = time.monotonic()
        target = second.tmp_dir / "independent-work"
        source = _make_video(tmp_path / "independent.mp4", seconds=1.0, pattern="testsrc")
        from trace_annotate.workspace.media import prepare_workspace_media
        prepare_workspace_media(source, target, slot_key=second.root, slot_timeout=30)
        elapsed = time.monotonic() - started
        # The second workspace completed while the first workspace's slot was held.
        assert elapsed < 30.0, elapsed
        assert (target / "project.json").is_file()
        assert slot_a.stats()["held"] is True
        slot_a.release()
    finally:
        first.close()
        second.close()


def test_cli_import_and_async_worker_share_one_slot(store, upload, process_counter):
    """store.import_source (CLI) and the async drainer must not decode together."""
    q = AsyncIntakes(store)
    q.enqueue(upload("async2", pattern="smptebars", seconds=1.0), display_name="async2.mp4")
    get_prepare_slot(store.root).reset_stats()
    errors = []

    def async_side():
        try:
            q.run_pending(worker_id="async-worker")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=async_side)
    thread.start()
    try:
        store.import_source(upload("cli", pattern="rgbtestsrc", seconds=1.0),
                            display_name="cli.mp4")
    finally:
        thread.join(timeout=180)
    assert not errors, errors
    assert process_counter["peak"] == 1, process_counter
    assert get_prepare_slot(store.root).stats()["peak"] == 1


def test_http_import_and_async_worker_share_one_slot(store, upload, process_counter):
    """The synchronous HTTP upload path funnels through the same workspace slot."""
    import http.client
    import json as _json
    from trace_annotate.workspace.server import TOKEN_HEADER, make_server

    q = AsyncIntakes(store)
    q.enqueue(upload("async3", pattern="testsrc", seconds=1.0), display_name="async3.mp4")
    httpd = make_server(store)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01},
                              daemon=True)
    thread.start()
    get_prepare_slot(store.root).reset_stats()
    errors = []

    def async_side():
        try:
            httpd.async_intakes.run_pending(worker_id="async-worker")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    drain = threading.Thread(target=async_side)
    drain.start()
    try:
        raw = upload("http", pattern="smptebars", seconds=1.0).read_bytes()
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=120)
        conn.request("POST", "/api/v1/projects/current/imports", body=raw,
                     headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                              "Content-Type": "application/octet-stream",
                              "Content-Length": str(len(raw)),
                              "X-TraceAnnotate-Filename": "http.mp4"})
        response = conn.getresponse()
        payload = _json.loads(response.read())
        conn.close()
    finally:
        drain.join(timeout=180)
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()
    assert not errors, errors
    assert response.status == 200, payload
    assert payload["task_id"]
    assert process_counter["peak"] == 1, process_counter
    assert get_prepare_slot(store.root).stats()["peak"] == 1
