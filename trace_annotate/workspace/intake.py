"""Async, durable media admission (T08).

The old synchronous ``store.import_source`` path is unchanged. This module adds a
second route in which the request side only:

  1. fully receives the upload bytes,
  2. fsyncs them and verifies the whole-file sha256 and size,
  3. moves the bytes into a workspace-owned staged directory,
  4. persists a ``queued`` intake and returns immediately (HTTP 202).

A runner then prepares at most one intake at a time (conservative single
prepare), honors cooperative cancellation, persists failures for explicit
retry, and on restart records interrupted intakes as failed instead of silently
re-running them. The staged source is the controlled object; file names and
duration are hints only.

Ordering rules that matter for reliability (P04/P05/P06):

* The single physical prepare slot is reserved **before** an intake is claimed,
  so a row that is still waiting for the decoder stays ``queued`` and does not
  burn its lease.
* A claimed intake is kept alive by a fenced heartbeat (generation + worker +
  session) that can never extend the attempt's total deadline.
* Closing the workspace stops new admission, stops new claims, interrupts a
  slot wait, and reports whether the worker actually stopped instead of
  silently joining for a fixed number of seconds.
"""
from __future__ import annotations

import datetime as dt
import errno
import json
import os
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..temporal.media import stream_sha256
from . import schema as S
from .errors import Conflict, LimitExceeded, NotFound, SchemaInvalid, StorageFull, WorkspaceError
from .slots import SlotClosed, SlotTimeout, get_prepare_slot
from .store import _safe, _new_id

MAX_STAGED_BYTES = 512 * 1024 * 1024
TERMINAL = {"succeeded", "failed", "cancelled"}

# Frozen operational budgets (instruction 039/115). Each value is a wait or
# work bound with a stated unit; none of them is an unbounded wait.
DEFAULT_LEASE_SECONDS = 120.0        # seconds a claim stays valid without renewal
DEFAULT_LEASE_RENEW_FRACTION = 3.0   # renew every lease/3 seconds
DEFAULT_PREPARE_DEADLINE = 1800.0    # seconds of total prepare work per attempt
DEFAULT_SLOT_WAIT_SECONDS = 900.0    # seconds a worker may wait for the decode slot
DEFAULT_CLOSE_SECONDS = 60.0         # seconds shutdown waits for the worker to stop


def _row(cursor: Any) -> Optional[Dict[str, Any]]:
    value = cursor.fetchone()
    if value is None:
        return None
    return dict(zip((column[0] for column in cursor.description), value))


def _rows(cursor: Any) -> List[Dict[str, Any]]:
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class _LeaseHeartbeat:
    """Fenced lease renewal that can never extend the attempt deadline."""

    def __init__(self, owner: "AsyncIntakes", intake_id: str, generation: int,
                 worker_id: str, *, deadline_at: float, interval: float):
        self._owner = owner
        self._intake_id = intake_id
        self._generation = generation
        self._worker_id = worker_id
        self._deadline_at = deadline_at
        self._interval = max(0.01, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.lease_lost = False
        self.deadline_exceeded = False
        self.renewals = 0
        self.last_error: Optional[str] = None

    def start(self) -> "_LeaseHeartbeat":
        self._thread = threading.Thread(target=self._loop, name="intake-heartbeat", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            now = self._owner.clock()
            if now >= self._deadline_at:
                self.deadline_exceeded = True
                return
            try:
                renewed = self._owner.renew_lease(
                    self._intake_id, self._generation, self._worker_id,
                    deadline_at=self._deadline_at)
            except Exception as exc:  # noqa: BLE001 - a failed renewal is a lost lease
                self.last_error = type(exc).__name__
                self.lease_lost = True
                return
            if not renewed:
                self.lease_lost = True
                return
            self.renewals += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval * 4))

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "_LeaseHeartbeat":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


class AsyncIntakes:
    """Durable staged-upload queue sharing the store's single-writer boundary."""

    def __init__(self, store: Any, *, lease_seconds: float = DEFAULT_LEASE_SECONDS,
                 clock: Callable[[], float] = time.time,
                 prepare_deadline: float = DEFAULT_PREPARE_DEADLINE,
                 slot_wait_seconds: float = DEFAULT_SLOT_WAIT_SECONDS,
                 closing: Optional[threading.Event] = None):
        if (isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float))
                or not lease_seconds > 0):
            raise SchemaInvalid("SCHEMA_INVALID", "lease_seconds must be positive")
        if (isinstance(prepare_deadline, bool) or not isinstance(prepare_deadline, (int, float))
                or not prepare_deadline > 0):
            raise SchemaInvalid("SCHEMA_INVALID", "prepare_deadline must be positive")
        if (isinstance(slot_wait_seconds, bool) or not isinstance(slot_wait_seconds, (int, float))
                or not slot_wait_seconds > 0):
            raise SchemaInvalid("SCHEMA_INVALID", "slot_wait_seconds must be positive")
        self.store, self.clock, self.lease_seconds = store, clock, float(lease_seconds)
        self.prepare_deadline = float(prepare_deadline)
        self.slot_wait_seconds = float(slot_wait_seconds)
        self.session_id = "intake-" + secrets.token_hex(8)
        self.staged_root = store.tmp_dir / "intakes"
        self._closing = closing if closing is not None else threading.Event()
        self._active_lock = threading.Lock()
        self._active_intake: Optional[str] = None
        self._closed = False
        self._last_heartbeat: Dict[str, Any] = {}
        with store.transaction():
            self.staged_root.mkdir(parents=True, exist_ok=True)
            _safe(self.staged_root)
            # A previous process that died mid-preparation must not silently
            # re-run; the input is preserved and marked interrupted for review.
            store.conn.execute(
                "UPDATE intakes SET state='failed', error_code='INTERRUPTED', worker_id=NULL, "
                "session_id=NULL, lease_expires_at=NULL WHERE state='preparing' AND task_id IS NULL")
            # A row left in cancel_requested by a dead process is a settled
            # cancellation, not something a new worker should resume.
            store.conn.execute(
                "UPDATE intakes SET state='cancelled', error_code='CANCELLED', worker_id=NULL, "
                "session_id=NULL, lease_expires_at=NULL WHERE state='cancel_requested'")

    # -- closing -----------------------------------------------------------
    def is_closing(self) -> bool:
        return self._closing.is_set()

    def active_intake(self) -> Optional[str]:
        with self._active_lock:
            return self._active_intake

    def close(self, *, timeout: float = DEFAULT_CLOSE_SECONDS) -> Dict[str, Any]:
        """Stop accepting/claiming and wait for the active prepare to stop.

        Returns an explicit liveness report. Callers must not close the store
        while ``stopped`` is false: a live worker still uses the database.
        """
        self._closing.set()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self.active_intake() is None:
                self._closed = True
                return {"stopped": True, "timeout_seconds": float(timeout),
                        "active_intake": None, "closing": True}
            time.sleep(0.02)
        self._closed = True
        return {"stopped": False, "timeout_seconds": float(timeout),
                "active_intake": self.active_intake(), "closing": True}

    # -- views -------------------------------------------------------------
    def get(self, intake_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            row = _row(self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                raise NotFound("INTAKE_NOT_FOUND", "Intake not found")
            row["staged_present"] = (self.staged_root / intake_id / "source").is_file()
            return row

    def list(self) -> List[Dict[str, Any]]:
        with self.store.transaction():
            return _rows(self.store.conn.execute("SELECT * FROM intakes ORDER BY rowid"))

    # -- request side: stage and return immediately -----------------------
    def enqueue(self, source_path: Path, *, display_name: str,
                template_id: str = "generic-segment", template_version: int = 1,
                command_id: Optional[str] = None,
                expected_sha256: Optional[str] = None,
                expected_size: Optional[int] = None,
                max_bytes: int = MAX_STAGED_BYTES) -> Dict[str, Any]:
        source = Path(source_path)
        S.require_id(template_id)
        if type(template_version) is not int or template_version < 1:
            raise SchemaInvalid("SCHEMA_INVALID", "template_version must be a positive integer")
        if not isinstance(display_name, str) or not display_name.strip():
            raise SchemaInvalid("SCHEMA_INVALID", "display_name is required")
        S.canonical_json(display_name)
        if not source.is_file():
            raise NotFound("SOURCE_MISSING", "Uploaded source is not a regular file")
        size = source.stat().st_size
        if size == 0 or size > max_bytes:
            raise LimitExceeded("LIMIT_EXCEEDED", "Upload size is outside the accepted range")
        digest = stream_sha256(source)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise Conflict("MEDIA_IDENTITY_MISMATCH", "Upload sha256 does not match the declared value")
        if expected_size is not None and size != expected_size:
            raise Conflict("MEDIA_IDENTITY_MISMATCH", "Upload size does not match the declared value")
        request = {"kind": "intake.enqueue", "display_name": display_name,
                   "template_id": template_id, "template_version": template_version,
                   "sha256": digest, "size": size}
        request_sha = S.sha256_bytes(S.canonical_json(request))
        with self.store.transaction():
            # A replay of an already-accepted command must still return its
            # original receipt after shutdown started; only new work is refused.
            replay = None
            if command_id:
                S.require_id(command_id)
                existing = _row(self.store.conn.execute(
                    "SELECT request_sha256,response FROM command_receipts WHERE command_id=?", (command_id,)))
                if existing is not None:
                    if existing["request_sha256"] != request_sha:
                        raise Conflict("IDEMPOTENCY_CONFLICT", "command_id was used for a different intake")
                    return json.loads(existing["response"])
                replay = existing
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace is shutting down and accepts no new uploads")
            if not self.store.conn.execute("SELECT 1 FROM templates WHERE id=? AND version=?",
                                           (template_id, template_version)).fetchone():
                raise NotFound("TEMPLATE_NOT_FOUND", "Requested template does not exist")
            intake_id = _new_id("int")
            group_id = _new_id("grp")
            target_dir = self.staged_root / intake_id
            target_dir.mkdir(parents=True, exist_ok=True)
            staged = target_dir / "source"
            # Move the caller's temp upload into the controlled staging area.
            try:
                os.replace(source, staged)
            except OSError:
                shutil.copy2(source, staged)
            _safe(staged)
            if stream_sha256(staged) != digest or staged.stat().st_size != size:
                raise Conflict("MEDIA_CHANGED", "Staged bytes do not match the verified upload")
            self.store.conn.execute(
                "INSERT INTO intakes(id,group_id,state,display_name,created_at,source_sha256,source_size,"
                "job_id,template_id,template_version,staged_path,attempt,generation,request_sha256) "
                "VALUES(?,?,'queued',?,?,?,?,null,?,?,?,0,0,?)",
                (intake_id, group_id, display_name, _now(), digest, size,
                 template_id, template_version, f"tmp/intakes/{intake_id}/source", request_sha))
            response = self.get(intake_id)
            response["staged_present"] = True
            response["status"] = "queued"
            if command_id:
                self.store.conn.execute(
                    "INSERT INTO command_receipts(command_id,task_id,request_sha256,state,response,created_at) "
                    "VALUES(?,NULL,?,'committed',?,?)",
                    (command_id, request_sha, S.canonical_json(response).decode("utf-8"), _now()))
            return response

    # -- control -----------------------------------------------------------
    def request_cancel(self, intake_id: str, *, command_id: Optional[str] = None) -> Dict[str, Any]:
        request = {"kind": "intake.cancel", "intake_id": intake_id}
        with self.store.transaction():
            if command_id:
                existing = _row(self.store.conn.execute(
                    "SELECT request_sha256,response FROM command_receipts WHERE command_id=?", (command_id,)))
                req_sha = S.sha256_bytes(S.canonical_json(request))
                if existing is not None:
                    if existing["request_sha256"] != req_sha:
                        raise Conflict("IDEMPOTENCY_CONFLICT", "command_id was reused")
                    return json.loads(existing["response"])
            row = _row(self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                raise NotFound("INTAKE_NOT_FOUND", "Intake not found")
            if row["state"] in TERMINAL:
                response = self.get(intake_id)
                if command_id:
                    req_sha = S.sha256_bytes(S.canonical_json(request))
                    self.store.conn.execute(
                        "INSERT INTO command_receipts(command_id,task_id,request_sha256,state,response,created_at) "
                        "VALUES(?,NULL,?,'committed',?,?)",
                        (command_id, req_sha, S.canonical_json(response).decode("utf-8"), _now()))
                return response
            if row["state"] == "queued":
                self.store.conn.execute("UPDATE intakes SET state='cancelled',error_code='CANCELLED' WHERE id=?",
                                        (intake_id,))
            else:  # preparing: cooperative; the worker fences its own output
                self.store.conn.execute("UPDATE intakes SET state='cancel_requested' WHERE id=?", (intake_id,))
            response = self.get(intake_id)
            if command_id:
                req_sha = S.sha256_bytes(S.canonical_json(request))
                self.store.conn.execute(
                    "INSERT INTO command_receipts(command_id,task_id,request_sha256,state,response,created_at) "
                    "VALUES(?,NULL,?,'committed',?,?)",
                    (command_id, req_sha, S.canonical_json(response).decode("utf-8"), _now()))
            return response

    def retry_failed(self, intake_id: str, *, command_id: Optional[str] = None) -> Dict[str, Any]:
        request = {"kind": "intake.retry", "intake_id": intake_id}
        req_sha = S.sha256_bytes(S.canonical_json(request))
        with self.store.transaction():
            if command_id:
                S.require_id(command_id)
                existing = _row(self.store.conn.execute(
                    "SELECT request_sha256,response FROM command_receipts WHERE command_id=?", (command_id,)))
                if existing is not None:
                    if existing["request_sha256"] != req_sha:
                        raise Conflict("IDEMPOTENCY_CONFLICT", "command_id was reused")
                    return json.loads(existing["response"])
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace is shutting down and accepts no new work")
            row = _row(self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                raise NotFound("INTAKE_NOT_FOUND", "Intake not found")
            if row["state"] != "failed":
                raise Conflict("INVALID_STATE", "Only failed intakes can be retried")
            if not (self.staged_root / intake_id / "source").is_file():
                raise NotFound("SOURCE_MISSING", "Staged upload was removed; re-upload the file")
            # ``attempt`` counts real prepare attempts and is bumped by
            # ``claim``. A retry must not pre-increment it, or the editor would
            # report "attempt 3" for what is actually the second prepare.
            self.store.conn.execute(
                "UPDATE intakes SET state='queued', error_code=NULL, "
                "worker_id=NULL, session_id=NULL, lease_expires_at=NULL WHERE id=?", (intake_id,))
            response = self.get(intake_id)
            if command_id:
                self.store.conn.execute(
                    "INSERT INTO command_receipts(command_id,task_id,request_sha256,state,response,created_at) "
                    "VALUES(?,NULL,?,'committed',?,?)",
                    (command_id, req_sha, S.canonical_json(response).decode("utf-8"), _now()))
            return response

    # -- worker side -------------------------------------------------------
    def claim(self, worker_id: str) -> Optional[Dict[str, Any]]:
        S.require_id(worker_id)
        if self.is_closing():
            return None
        with self.store.transaction():
            if self.is_closing():
                return None
            row = _row(self.store.conn.execute(
                "SELECT * FROM intakes WHERE state='queued' ORDER BY rowid LIMIT 1"))
            if row is None:
                return None
            now = self.clock()
            self.store.conn.execute(
                "UPDATE intakes SET state='preparing', attempt=attempt+1, generation=generation+1, "
                "worker_id=?, session_id=?, lease_expires_at=? WHERE id=?",
                (worker_id, self.session_id, now + self.lease_seconds, row["id"]))
            claimed = self.get(row["id"])
        with self._active_lock:
            self._active_intake = row["id"]
        return claimed

    def renew_lease(self, intake_id: str, generation: int, worker_id: str, *,
                    deadline_at: Optional[float] = None) -> bool:
        """Fenced renewal; refuses to extend past the attempt deadline."""
        with self.store.transaction() as conn:
            now = self.clock()
            expires = now + self.lease_seconds
            if deadline_at is not None:
                expires = min(expires, float(deadline_at))
            if expires <= now:
                return False
            cursor = conn.execute(
                "UPDATE intakes SET lease_expires_at=? WHERE id=? AND generation=? AND worker_id=? "
                "AND session_id=? AND state IN ('preparing','cancel_requested')",
                (expires, intake_id, generation, worker_id, self.session_id))
            renewed = cursor.rowcount == 1
        self._last_heartbeat = {"intake_id": intake_id, "generation": generation,
                                "worker_id": worker_id, "renewed_at": now,
                                "lease_expires_at": expires, "ok": renewed}
        return renewed

    def last_heartbeat(self) -> Dict[str, Any]:
        return dict(self._last_heartbeat)

    def _assert_lease(self, row: Dict[str, Any], generation: int, worker_id: str) -> None:
        if (row["generation"] != generation or row["worker_id"] != worker_id
                or row["session_id"] != self.session_id or row["state"] not in ("preparing", "cancel_requested")
                or (row["lease_expires_at"] or 0) <= self.clock()):
            raise Conflict("STALE_LEASE", "Intake lease is no longer valid")

    def finish(self, intake_id: str, generation: int, worker_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        with self.store.transaction():
            row = _row(self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                raise NotFound("INTAKE_NOT_FOUND", "Intake not found")
            if row["state"] == "succeeded" and row["task_id"]:
                return self.get(intake_id)  # idempotent double-commit
            self._assert_lease(row, generation, worker_id)
            if row["state"] == "cancel_requested":
                raise Conflict("CANCELLED", "Cancellation was requested before admission committed")
            self.store.conn.execute(
                "UPDATE intakes SET state='succeeded', media_id=?, task_id=?, error_code=NULL, "
                "worker_id=NULL, session_id=NULL, lease_expires_at=NULL WHERE id=?",
                (result["media_id"], result["task_id"], intake_id))
            return self.get(intake_id)

    def fail(self, intake_id: str, generation: int, worker_id: str, error_code: str) -> Dict[str, Any]:
        if not (isinstance(error_code, str) and error_code.replace("_", "").isalnum() and error_code.isupper()):
            error_code = "ADMISSION_FAILED"
        with self.store.transaction():
            row = _row(self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                raise NotFound("INTAKE_NOT_FOUND", "Intake not found")
            # An old attempt must not overwrite a newer generation / a row that
            # has already reached a terminal or been re-claimed by another worker.
            if row["generation"] != generation or row["worker_id"] != worker_id:
                return self.get(intake_id)
            if row["state"] not in ("preparing", "cancel_requested"):
                return self.get(intake_id)
            terminal_state = "cancelled" if row["state"] == "cancel_requested" else "failed"
            self.store.conn.execute(
                "UPDATE intakes SET state=?, error_code=?, worker_id=NULL, session_id=NULL, "
                "lease_expires_at=NULL WHERE id=?",
                (terminal_state, "CANCELLED" if terminal_state == "cancelled" else error_code, intake_id))
            return self.get(intake_id)

    def run_once(self, worker_id: str = "intake-local") -> Optional[Dict[str, Any]]:
        """Reserve the decode slot, then claim and prepare exactly one intake."""
        slot = get_prepare_slot(self.store.root)
        # Reserve the single physical prepare slot BEFORE claiming so a row that
        # is still waiting for the decoder stays ``queued`` and keeps no lease.
        try:
            slot.acquire(closing=self.is_closing, timeout=self.slot_wait_seconds)
        except (SlotClosed, SlotTimeout):
            return None
        try:
            claimed = self.claim(worker_id)
            if claimed is None:
                return None
            try:
                return self._prepare_claimed(claimed, worker_id)
            finally:
                with self._active_lock:
                    self._active_intake = None
        finally:
            slot.release()

    def _prepare_claimed(self, claimed: Dict[str, Any], worker_id: str) -> Dict[str, Any]:
        intake_id = claimed["id"]
        generation = claimed["generation"]
        started_at = self.clock()
        deadline_at = started_at + self.prepare_deadline
        state = {"reason": None}
        # Renew well inside the lease, and always poll several times before the
        # attempt deadline so the heartbeat itself observes the overrun instead
        # of relying on the worker's own check.
        interval = max(0.01, min(self.lease_seconds / DEFAULT_LEASE_RENEW_FRACTION,
                                 self.prepare_deadline / 4.0))
        heartbeat = _LeaseHeartbeat(self, intake_id, generation, worker_id,
                                    deadline_at=deadline_at, interval=interval)

        def _stop(reason: str) -> bool:
            # Record the first reason actually observed. A later condition must
            # not relabel an earlier user cancellation as a deadline abort, and
            # vice versa; the persisted error_code has to stay truthful.
            if state["reason"] is None:
                state["reason"] = reason
            return True

        def is_cancelled() -> bool:
            if self.is_closing():
                # A shutdown is not a user cancellation: the attempt is
                # interrupted, the staged source is kept and the row becomes a
                # retryable failure instead of a settled cancellation.
                return _stop("closing")
            if heartbeat.lease_lost:
                return _stop("lease_lost")
            if heartbeat.deadline_exceeded or self.clock() >= deadline_at:
                return _stop("deadline")
            with self.store.transaction():
                row = _row(self.store.conn.execute(
                    "SELECT state,generation,worker_id,lease_expires_at FROM intakes WHERE id=?",
                    (intake_id,)))
            if row is None:
                return _stop("cancelled")
            if row["state"] == "cancel_requested":
                return _stop("cancelled")
            if (row["generation"] != generation or row["worker_id"] != worker_id
                    or (row["lease_expires_at"] or 0) <= self.clock()):
                return _stop("lease_lost")
            return False

        try:
            with heartbeat:
                result = self.store.admit_staged_intake(
                    claimed, cancelled=is_cancelled, generation=generation, worker_id=worker_id,
                    session_id=self.session_id, clock=self.clock, deadline_at=deadline_at)
            return self.finish(intake_id, generation, worker_id, result)
        except Exception as exc:  # noqa: BLE001 - persisted as a failed intake
            code = getattr(exc, "code", None)
            if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
                code = "STORAGE_FULL"
            elif state["reason"] is not None:
                # The stop reason is authoritative: preparation surfaces every
                # cooperative abort as CANCELLED, which would otherwise hide a
                # deadline overrun or a lost lease from the operator.
                code = {"closing": "INTERRUPTED", "deadline": "DEADLINE_EXCEEDED",
                        "lease_lost": "LEASE_LOST", "cancelled": "CANCELLED"}[state["reason"]]
            elif code is None:
                code = "ADMISSION_FAILED"
            return self.fail(intake_id, generation, worker_id, code)
        finally:
            self._last_heartbeat = {
                "intake_id": intake_id, "generation": generation, "worker_id": worker_id,
                "renewals": heartbeat.renewals, "lease_lost": heartbeat.lease_lost,
                "deadline_exceeded": heartbeat.deadline_exceeded,
                "stop_reason": state["reason"],
                "heartbeat_alive": heartbeat.alive(),
                "started_at": started_at, "deadline_at": deadline_at,
                "elapsed_seconds": round(self.clock() - started_at, 3),
            }

    def run_pending(self, worker_id: str = "intake-local") -> List[Dict[str, Any]]:
        results = []
        while True:
            value = self.run_once(worker_id)
            if value is None:
                return results
            results.append(value)
