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
from .store import _safe, _new_id

MAX_STAGED_BYTES = 512 * 1024 * 1024
TERMINAL = {"succeeded", "failed", "cancelled"}


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


class AsyncIntakes:
    """Durable staged-upload queue sharing the store's single-writer boundary."""

    def __init__(self, store: Any, *, lease_seconds: float = 120,
                 clock: Callable[[], float] = time.time):
        if (isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float))
                or not lease_seconds > 0):
            raise SchemaInvalid("SCHEMA_INVALID", "lease_seconds must be positive")
        self.store, self.clock, self.lease_seconds = store, clock, float(lease_seconds)
        self.session_id = "intake-" + secrets.token_hex(8)
        self.staged_root = store.tmp_dir / "intakes"
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
            if command_id:
                S.require_id(command_id)
                existing = _row(self.store.conn.execute(
                    "SELECT request_sha256,response FROM command_receipts WHERE command_id=?", (command_id,)))
                if existing is not None:
                    if existing["request_sha256"] != request_sha:
                        raise Conflict("IDEMPOTENCY_CONFLICT", "command_id was used for a different intake")
                    return json.loads(existing["response"])
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
                return self.get(intake_id)
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

    def retry_failed(self, intake_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            row = _row(self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                raise NotFound("INTAKE_NOT_FOUND", "Intake not found")
            if row["state"] != "failed":
                raise Conflict("INVALID_STATE", "Only failed intakes can be retried")
            if not (self.staged_root / intake_id / "source").is_file():
                raise NotFound("SOURCE_MISSING", "Staged upload was removed; re-upload the file")
            self.store.conn.execute(
                "UPDATE intakes SET state='queued', error_code=NULL, attempt=attempt+1, "
                "worker_id=NULL, session_id=NULL, lease_expires_at=NULL WHERE id=?", (intake_id,))
            return self.get(intake_id)

    # -- worker side -------------------------------------------------------
    def claim(self, worker_id: str) -> Optional[Dict[str, Any]]:
        S.require_id(worker_id)
        with self.store.transaction():
            row = _row(self.store.conn.execute(
                "SELECT * FROM intakes WHERE state='queued' ORDER BY rowid LIMIT 1"))
            if row is None:
                return None
            now = self.clock()
            self.store.conn.execute(
                "UPDATE intakes SET state='preparing', attempt=attempt+1, generation=generation+1, "
                "worker_id=?, session_id=?, lease_expires_at=? WHERE id=?",
                (worker_id, self.session_id, now + self.lease_seconds, row["id"]))
            return self.get(row["id"])

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
        """Claim and prepare one intake; returns the final intake view or None."""
        claimed = self.claim(worker_id)
        if claimed is None:
            return None
        intake_id = claimed["id"]
        generation = claimed["generation"]
        cancelled_flag = {"cancelled": False}

        def is_cancelled() -> bool:
            with self.store.transaction():
                row = _row(self.store.conn.execute(
                    "SELECT state,lease_expires_at FROM intakes WHERE id=?", (intake_id,)))
            if row is None:
                return True
            if row["state"] == "cancel_requested":
                cancelled_flag["cancelled"] = True
            return cancelled_flag["cancelled"]

        try:
            result = self.store.admit_staged_intake(
                claimed, cancelled=is_cancelled,
                generation=generation, worker_id=worker_id, session_id=self.session_id)
            return self.finish(intake_id, generation, worker_id, result)
        except Exception as exc:  # noqa: BLE001 - persisted as a failed intake
            code = getattr(exc, "code", None) or ("CANCELLED" if cancelled_flag["cancelled"] else "ADMISSION_FAILED")
            if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
                code = "STORAGE_FULL"
            return self.fail(intake_id, generation, worker_id, code)

    def run_pending(self, worker_id: str = "intake-local") -> List[Dict[str, Any]]:
        results = []
        while True:
            value = self.run_once(worker_id)
            if value is None:
                return results
            results.append(value)
