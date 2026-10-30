"""P06: closing the workspace must end work in a defined order.

The invariants under test:

* close refuses new admission before a byte of upload is accepted;
* close interrupts an active prepare as a retryable ``INTERRUPTED`` failure and
  lets the owned media child exit, without turning it into a user cancellation;
* queued work that was already accepted survives the close;
* a settled cancellation stays cancelled when close arrives afterwards;
* close is idempotent, cheap on an empty queue, and reports whether the worker
  really stopped;
* a worker that survives into a reopened workspace cannot commit a late result.
"""
from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.temporal.preparation import live_media_processes
from trace_annotate.workspace.errors import Conflict, WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.server import TOKEN_HEADER, make_server
from trace_annotate.workspace.slots import get_prepare_slot
from trace_annotate.workspace.store import WorkspaceStore

FFMPEG = shutil.which("ffmpeg")


def _video(path: Path, *, seconds: float = 1.0, pattern: str = "testsrc",
           size: str = "320x240") -> Path:
    if FFMPEG is None:
        pytest.skip("ffmpeg is required")
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


def _stage(store: WorkspaceStore, tmp_path: Path, name: str, **kwargs) -> Path:
    source = _video(tmp_path / f"{name}.mp4", **kwargs)
    target = store.tmp_dir / f"upload-{name}.tmp"
    shutil.copy2(source, target)
    return target


def _wait(predicate, budget: float = 60.0, message: str = "condition not reached"):
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(message)


def test_close_refuses_an_http_upload_before_the_body_is_accepted(store, tmp_path):
    httpd = make_server(store)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01},
                              daemon=True)
    thread.start()
    try:
        httpd.closing.set()
        raw = _stage(store, tmp_path, "late").read_bytes()
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
        conn.request("POST", "/api/v1/projects/current/imports", body=raw,
                     headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                              "Content-Type": "application/octet-stream",
                              "Content-Length": str(len(raw)),
                              "X-TraceAnnotate-Filename": "late.mp4"})
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()
        assert response.status == 409, payload
        assert payload["error"]["code"] == "WORKSPACE_CLOSING", payload
        # No intake may exist: refusing must not leave a half-created row.
        assert httpd.async_intakes.list() == []
        with store.transaction():
            count = store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def test_close_during_prepare_interrupts_and_keeps_queued(store, tmp_path):
    q = AsyncIntakes(store, lease_seconds=30.0, prepare_deadline=300.0)
    running = q.enqueue(_stage(store, tmp_path, "running", seconds=30.0, size="854x480",
                               pattern="smptebars"), display_name="running.mp4")
    waiting = q.enqueue(_stage(store, tmp_path, "waiting", pattern="testsrc"),
                        display_name="waiting.mp4")

    outcome = {}

    def worker():
        outcome["result"] = q.run_once(worker_id="w1")

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        _wait(lambda: q.get(running["id"])["state"] == "preparing",
              message="prepare never started")
        child = _wait(lambda: [p for p in live_media_processes()
                               if running["id"] in " ".join(p.get("argv") or [])] or None,
                      message="no owned media child was associated with the intake")[0]
        assert child["parent_pid"] == __import__("os").getpid()
        report = q.close(timeout=60)
        assert report["stopped"] is True, report
    finally:
        thread.join(timeout=90)
    assert not thread.is_alive()

    final = q.get(running["id"])
    assert final["state"] == "failed", final
    assert final["error_code"] == "INTERRUPTED", final
    assert final["task_id"] is None
    # The child really exited rather than being abandoned.
    _wait(lambda: not _alive(child["pid"]), budget=60,
          message="the media child survived the shutdown")
    # Accepted but not started work survives for the next process.
    assert q.get(waiting["id"])["state"] == "queued"
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def _alive(pid: int) -> bool:
    import os
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_close_after_a_settled_cancellation_keeps_it_cancelled(store, tmp_path):
    q = AsyncIntakes(store, lease_seconds=30.0, prepare_deadline=300.0)
    rec = q.enqueue(_stage(store, tmp_path, "cancelling", seconds=30.0, size="854x480",
                           pattern="smptebars"), display_name="cancelling.mp4")
    thread = threading.Thread(target=q.run_once, kwargs={"worker_id": "w1"})
    thread.start()
    try:
        _wait(lambda: q.get(rec["id"])["state"] == "preparing", message="prepare never started")
        q.request_cancel(rec["id"])
        assert q.get(rec["id"])["state"] == "cancel_requested"
        report = q.close(timeout=60)
        assert report["stopped"] is True, report
    finally:
        thread.join(timeout=90)
    final = q.get(rec["id"])
    assert final["state"] == "cancelled", final
    assert final["error_code"] == "CANCELLED", final
    assert final["task_id"] is None


def test_repeat_close_and_empty_queue_are_safe(store):
    q = AsyncIntakes(store)
    first = q.close(timeout=5)
    assert first["stopped"] is True
    started = time.monotonic()
    second = q.close(timeout=5)
    elapsed = time.monotonic() - started
    assert second["stopped"] is True
    assert elapsed < 2.0, elapsed
    assert q.claim("w1") is None
    assert q.run_pending("w1") == []


def test_late_result_after_close_and_reopen_cannot_commit(store, tmp_path):
    q = AsyncIntakes(store)
    rec = q.enqueue(_stage(store, tmp_path, "late-result"), display_name="late.mp4")
    claimed = q.claim("w1")
    generation = claimed["generation"]
    q.close(timeout=5)
    # A new process opens the same workspace: the interrupted row is settled.
    reopened = AsyncIntakes(store)
    assert reopened.get(rec["id"])["state"] == "failed"
    assert reopened.get(rec["id"])["error_code"] == "INTERRUPTED"
    with pytest.raises(Conflict) as excinfo:
        q.finish(rec["id"], generation, "w1",
                 {"media_id": "med-late", "task_id": "tsk-late"})
    assert excinfo.value.code == "STALE_LEASE"
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_close_interrupts_a_synchronous_import_as_interrupted(store, tmp_path):
    source = _stage(store, tmp_path, "sync-close", seconds=30.0, size="854x480",
                    pattern="smptebars")
    closing = threading.Event()
    outcome = {}

    def run():
        try:
            store.import_source(source, display_name="sync-close.mp4",
                                closing=closing.is_set)
            outcome["result"] = "succeeded"
        except WorkspaceError as exc:
            outcome["error"] = exc.code

    thread = threading.Thread(target=run)
    thread.start()
    try:
        _wait(lambda: any(row["state"] == "running" for row in store.list_intakes()),
              message="the synchronous import never started")
        closing.set()
    finally:
        thread.join(timeout=120)
    assert outcome.get("error") == "INTERRUPTED", outcome
    rows = store.list_intakes()
    assert len(rows) == 1
    assert rows[0]["state"] == "failed", rows[0]
    assert rows[0]["error_code"] == "INTERRUPTED", rows[0]
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    # The slot was released even though the import failed.
    assert get_prepare_slot(store.root).stats()["held"] is False
