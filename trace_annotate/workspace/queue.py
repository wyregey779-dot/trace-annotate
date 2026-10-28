"""Durable, bounded dispatch for jobs on already admitted workspace tasks.

Processors run outside the SQLite lock and return JSON candidate artifacts. A
successful job never means a candidate was reviewed or applied to a task. Intake
admission and processor-specific result validation are separate responsibilities.
"""
from __future__ import annotations

import datetime as dt
import errno
import json
import math
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from . import schema as S
from .errors import Conflict, LimitExceeded, NotFound, SchemaInvalid, WorkspaceError


MAX_CONCURRENCY = 8
MAX_BATCH_ITEMS = 1000
MAX_RESULT_BYTES = 16 * 1024 * 1024
_DDL = (
    "CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, configuration_sha256 TEXT NOT NULL, "
    "dispatch_paused INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, "
    "input TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, "
    "lease_generation INTEGER NOT NULL DEFAULT 0, worker_id TEXT, session_id TEXT, "
    "lease_expires_at REAL, heartbeat_at REAL, error_code TEXT, created_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS batch_jobs (batch_id TEXT NOT NULL REFERENCES batches(id), "
    "job_id TEXT NOT NULL REFERENCES jobs(id), ordinal INTEGER NOT NULL, "
    "PRIMARY KEY(batch_id,job_id))",
    "CREATE TABLE IF NOT EXISTS attempts (job_id TEXT NOT NULL REFERENCES jobs(id), "
    "attempt INTEGER NOT NULL, lease_generation INTEGER NOT NULL, worker_id TEXT NOT NULL, "
    "state TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT, error_code TEXT, "
    "diagnostic_artifact_id TEXT, "
    "PRIMARY KEY(job_id,attempt))",
    "CREATE TABLE IF NOT EXISTS job_commits (job_id TEXT PRIMARY KEY REFERENCES jobs(id), "
    "id TEXT NOT NULL UNIQUE, body TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS jobs_dispatch ON jobs(state,created_at)",
)


def _json(value: Any) -> bytes:
    """Reject lossy Python-to-JSON coercions as well as non-finite numbers."""
    def check(item: Any) -> None:
        if item is None or isinstance(item, (str, bool)):
            return
        if isinstance(item, (int, float)):
            if not math.isfinite(item) or abs(item) > S.MAX_SAFE_INTEGER:
                raise ValueError("number is outside the JSON contract")
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("unsupported JSON value")

    try:
        check(value)
        return S.canonical_json(value)
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise SchemaInvalid("SCHEMA_INVALID", "Expected finite, lossless JSON data") from exc


def _ident(value: Any, field: str) -> str:
    if not isinstance(value, str) or S.ID_RE.fullmatch(value) is None:
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid " + field)
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid " + field)
    return value


def _row(cursor: Any) -> Optional[Dict[str, Any]]:
    value = cursor.fetchone()
    return dict(zip((column[0] for column in cursor.description), value)) if value else None


def _read_job(store: Any, job_id: str, *, attempts: bool = False) -> Dict[str, Any]:
    value = _row(store.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)))
    if value is None:
        raise NotFound("JOB_NOT_FOUND", "Job not found")
    value.update(json.loads(value.pop("input")))
    value["job_id"] = value["id"]
    commit = store.conn.execute("SELECT body FROM job_commits WHERE job_id=?", (job_id,)).fetchone()
    value["commit"] = json.loads(commit[0]) if commit else None
    if attempts:
        cursor = store.conn.execute("SELECT * FROM attempts WHERE job_id=? ORDER BY attempt", (job_id,))
        names = [column[0] for column in cursor.description]
        value["attempts"] = [dict(zip(names, row)) for row in cursor]
    return value


def _read_batch(store: Any, batch_id: str) -> Dict[str, Any]:
    batch = _row(store.conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)))
    if batch is None:
        raise NotFound("BATCH_NOT_FOUND", "Batch not found")
    batch["batch_id"] = batch["id"]
    batch["dispatch_paused"] = bool(batch["dispatch_paused"])
    ids = store.conn.execute("SELECT job_id FROM batch_jobs WHERE batch_id=? ORDER BY ordinal", (batch_id,)).fetchall()
    batch["jobs"] = [_read_job(store, row[0], attempts=True) for row in ids]
    batch["counts"] = {state: sum(job["state"] == state for job in batch["jobs"])
                       for state in ("queued", "running", "cancel_requested", "cancelled", "failed", "succeeded")}
    return batch


def read_queue_snapshot(store: Any) -> Dict[str, Any]:
    """Read a consistent queue view without initialization or lease recovery.

    Exporting a workspace must not create a queue, change its revision, or mark
    an expired worker failed. Queue startup/dispatch owns those state changes.
    """
    with store.transaction():
        expected = {"batches", "jobs", "batch_jobs", "attempts", "job_commits"}
        existing = {row[0] for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        present = expected & existing
        if not present:
            return {"batches": [], "commits": []}
        if present != expected:
            raise WorkspaceError("RECOVERY_REQUIRED", "Queue tables are incomplete")
        ids = store.conn.execute("SELECT id FROM batches ORDER BY rowid").fetchall()
        return {"batches": [_read_batch(store, row[0]) for row in ids],
                "commits": [json.loads(row[0]) for row in store.conn.execute("SELECT body FROM job_commits ORDER BY rowid")]}


class WorkerContext:
    """Cooperative control; processors must check at their own safe boundaries."""

    def __init__(self, queue: "DurableQueue", job: Dict[str, Any]):
        self.queue = queue
        self.job_id = job["id"]
        self.lease_generation = job["lease_generation"]
        self.worker_id = job["worker_id"]

    def cancelled(self) -> bool:
        with self.queue.store.transaction():
            job = self.queue._job(self.job_id)
            return (job["state"] != "running" or
                    job["lease_generation"] != self.lease_generation or
                    job["worker_id"] != self.worker_id or
                    job["lease_expires_at"] <= self.queue.clock())

    def heartbeat(self) -> Dict[str, Any]:
        return self.queue.heartbeat(self.job_id, self.lease_generation, self.worker_id)

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise WorkspaceError("CANCELLED", "Job cancelled or lease no longer valid")


class DurableQueue:
    """A queue sharing the store's single-writer transaction boundary.

    The persisted concurrency limit is shared by all queue wrappers. Constructing
    a wrapper for a newly opened store recovers leases from previous sessions;
    constructing another wrapper for the same store leaves live workers alone.
    """

    def __init__(self, store: Any, *, max_concurrency: Optional[int] = None,
                 lease_seconds: float = 60, clock: Callable[[], float] = time.time):
        if max_concurrency is not None and (type(max_concurrency) is not int or
                                           not 1 <= max_concurrency <= MAX_CONCURRENCY):
            raise SchemaInvalid("SCHEMA_INVALID", "max_concurrency must be between 1 and 8")
        if (isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or
                not math.isfinite(lease_seconds) or not 1 <= lease_seconds <= 3600):
            raise SchemaInvalid("SCHEMA_INVALID", "lease_seconds must be between 1 and 3600")
        self.store, self.clock, self.lease_seconds = store, clock, lease_seconds
        with store.transaction():
            for statement in _DDL:
                store.conn.execute(statement)
            store._install_revision_triggers(("batches", "jobs", "attempts", "batch_jobs"))
            config = store.conn.execute("SELECT value FROM meta WHERE key='queue_max_concurrency'").fetchone()
            if config:
                self.max_concurrency = int(config[0])
                if max_concurrency is not None and max_concurrency != self.max_concurrency:
                    raise Conflict("CONFIGURATION_CONFLICT", "Queue concurrency is already configured")
            else:
                self.max_concurrency = max_concurrency or 1
                store.conn.execute("INSERT INTO meta(key,value) VALUES('queue_max_concurrency',?)",
                                   (str(self.max_concurrency),))
            if getattr(store, "_queue_connection", None) is not store.conn:
                store._queue_session_id = "session-" + uuid.uuid4().hex
                store._queue_connection = store.conn
                store._queue_active_processors = set()
            self.session_id = store._queue_session_id
            self.recover(restart=True)

    def _now(self) -> str:
        return dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).isoformat()

    def _job(self, job_id: str) -> Dict[str, Any]:
        return _read_job(self.store, job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            return _read_job(self.store, job_id, attempts=True)

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            return _read_batch(self.store, batch_id)

    def list_batches(self) -> List[Dict[str, Any]]:
        with self.store.transaction():
            ids = self.store.conn.execute("SELECT id FROM batches ORDER BY rowid").fetchall()
            return [self.get_batch(row[0]) for row in ids]

    def snapshot(self) -> Dict[str, Any]:
        return read_queue_snapshot(self.store)

    def _receipt(self, command_id: Optional[str], request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if command_id is None:
            return None
        _ident(command_id, "command_id")
        row = self.store.conn.execute("SELECT request_sha256,response FROM command_receipts WHERE command_id=?",
                                      (command_id,)).fetchone()
        if row is None:
            return None
        if row[0] != S.sha256_bytes(_json(request)):
            raise Conflict("IDEMPOTENCY_CONFLICT", "command_id already has different content")
        return json.loads(row[1])

    def _save_receipt(self, command_id: Optional[str], request: Dict[str, Any], response: Dict[str, Any]) -> None:
        if command_id is not None:
            self.store.conn.execute("INSERT INTO command_receipts(command_id,task_id,request_sha256,state,response,created_at) "
                                    "VALUES(?,NULL,?,'committed',?,?)",
                                    (command_id, S.sha256_bytes(_json(request)), _json(response).decode("utf-8"), self._now()))

    def create_batch(self, items: List[Dict[str, Any]], *, operation: str,
                     processor_version: str, parameters: Optional[Dict[str, Any]] = None,
                     command_id: Optional[str] = None) -> Dict[str, Any]:
        _text(operation, "operation")
        _text(processor_version, "processor_version")
        parameters = {} if parameters is None else parameters
        if not isinstance(parameters, dict):
            raise SchemaInvalid("SCHEMA_INVALID", "parameters must be a JSON object")
        if not isinstance(items, list) or not items or len(items) > MAX_BATCH_ITEMS:
            raise SchemaInvalid("SCHEMA_INVALID", "A batch needs between 1 and 1000 items")
        request = json.loads(_json({"kind": "queue.create_batch", "items": items, "operation": operation,
                                   "processor_version": processor_version, "parameters": parameters}))
        with self.store.transaction():
            receipt = self._receipt(command_id, request)
            if receipt is not None:
                return receipt
            frozen, seen = [], set()
            for item in request["items"]:
                if not isinstance(item, dict) or set(item) != {"task_id", "expected_revision", "expected_sha256"}:
                    raise SchemaInvalid("SCHEMA_INVALID", "Every item needs task_id, expected_revision and expected_sha256")
                task_id = _ident(item["task_id"], "task_id")
                if task_id in seen:
                    raise SchemaInvalid("SCHEMA_INVALID", "A batch cannot contain a task twice")
                seen.add(task_id)
                revision = item["expected_revision"]
                digest = item["expected_sha256"]
                if type(revision) is not int or revision < 0 or not isinstance(digest, str) or S.HASH_RE.fullmatch(digest) is None:
                    raise SchemaInvalid("SCHEMA_INVALID", "Invalid input revision or hash")
                task = self.store.get_task(task_id)
                if task["revision"] != revision:
                    raise Conflict("REVISION_CONFLICT", "Task changed before enqueue")
                if self.store.task_digest(task) != digest:
                    raise Conflict("MEDIA_CHANGED", "Task digest changed before enqueue")
                spec = {"task_id": task_id, "operation": operation, "input_revision": revision,
                        "input_sha256": digest, "processor_version": processor_version, "parameters": request["parameters"]}
                key = S.sha256_bytes(_json(spec))
                spec["task_snapshot"] = task
                spec["media_snapshot"] = self.store.get_media(task["media_id"])
                frozen.append((key, spec))
            batch_id = "batch-" + uuid.uuid4().hex
            config_hash = S.sha256_text(sorted(key for key, _ in frozen))
            self.store.conn.execute("INSERT INTO batches(id,configuration_sha256,created_at) VALUES(?,?,?)",
                                    (batch_id, config_hash, self._now()))
            for ordinal, (key, spec) in enumerate(frozen):
                job_id = "job-" + key
                self.store.conn.execute("INSERT OR IGNORE INTO jobs(id,idempotency_key,input,state,created_at) VALUES(?,?,?,'queued',?)",
                                        (job_id, key, _json(spec).decode("utf-8"), self._now()))
                self.store.conn.execute("INSERT INTO batch_jobs(batch_id,job_id,ordinal) VALUES(?,?,?)",
                                        (batch_id, job_id, ordinal))
            response = self.get_batch(batch_id)
            self._save_receipt(command_id, request, response)
            return response

    def enqueue(self, task_id: str, *, operation: str, processor_version: str,
                parameters: Optional[Dict[str, Any]] = None, expected_revision: Optional[int] = None,
                expected_sha256: Optional[str] = None, command_id: Optional[str] = None) -> Dict[str, Any]:
        with self.store.transaction():
            task = self.store.get_task(task_id)
            batch = self.create_batch([{"task_id": task_id,
                                       "expected_revision": task["revision"] if expected_revision is None else expected_revision,
                                       "expected_sha256": self.store.task_digest(task) if expected_sha256 is None else expected_sha256}],
                                      operation=operation, processor_version=processor_version,
                                      parameters=parameters, command_id=command_id)
            return batch["jobs"][0]

    def claim(self, worker_id: str, batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        _ident(worker_id, "worker_id")
        with self.store.transaction():
            self.recover()
            if batch_id is not None:
                self.get_batch(batch_id)
            active = self.store.conn.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('running','cancel_requested')").fetchone()[0]
            # An expired lease fences output, but cannot forcibly stop Python
            # code. Retain the process-local slot until its runner actually exits.
            for active_id, generation in self.store._queue_active_processors:
                running = self._job(active_id)
                if running["state"] not in ("running", "cancel_requested") or running["lease_generation"] != generation:
                    active += 1
            if active >= self.max_concurrency:
                return None
            sql = ("SELECT j.id FROM jobs j WHERE j.state='queued' AND EXISTS "
                   "(SELECT 1 FROM batch_jobs bj JOIN batches b ON b.id=bj.batch_id "
                   "WHERE bj.job_id=j.id AND b.dispatch_paused=0")
            args = ()
            if batch_id is not None:
                sql += " AND b.id=?"
                args = (batch_id,)
            row = self.store.conn.execute(sql + ") ORDER BY j.rowid LIMIT 1", args).fetchone()
            if row is None:
                return None
            now = self.clock()
            self.store.conn.execute("UPDATE jobs SET state='running',attempt=attempt+1,lease_generation=lease_generation+1, "
                                    "worker_id=?,session_id=?,heartbeat_at=?,lease_expires_at=?,error_code=NULL WHERE id=?",
                                    (worker_id, self.session_id, now, now + self.lease_seconds, row[0]))
            job = self._job(row[0])
            self.store.conn.execute("INSERT INTO attempts(job_id,attempt,lease_generation,worker_id,state,started_at) "
                                    "VALUES(?,?,?,?,'running',?)",
                                    (job["id"], job["attempt"], job["lease_generation"], worker_id, self._now()))
            return job

    def _lease(self, job_id: str, lease_generation: int, worker_id: str) -> Dict[str, Any]:
        job = self._job(job_id)
        if (type(lease_generation) is not int or job["lease_generation"] != lease_generation or
                job["worker_id"] != worker_id or job["session_id"] != self.session_id or
                job["state"] not in ("running", "cancel_requested") or
                job["lease_expires_at"] <= self.clock()):
            raise Conflict("STALE_LEASE", "Worker lease is no longer valid")
        return job

    def heartbeat(self, job_id: str, lease_generation: int, worker_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            job = self._lease(job_id, lease_generation, worker_id)
            now = self.clock()
            self.store.conn.execute("UPDATE jobs SET heartbeat_at=?,lease_expires_at=? WHERE id=?",
                                    (now, now + self.lease_seconds, job_id))
            return {"cancel_requested": job["state"] == "cancel_requested", "lease_expires_at": now + self.lease_seconds}

    def _end(self, job: Dict[str, Any], state: str, error_code: Optional[str] = None) -> None:
        self.store.conn.execute("UPDATE jobs SET state=?,error_code=?,lease_expires_at=NULL WHERE id=?",
                                (state, error_code, job["id"]))
        self.store.conn.execute("UPDATE attempts SET state=?,ended_at=?,error_code=? WHERE job_id=? AND attempt=?",
                                (state, self._now(), error_code, job["id"], job["attempt"]))

    def finish(self, job_id: str, lease_generation: int, worker_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if type(lease_generation) is not int or lease_generation < 1:
            raise SchemaInvalid("SCHEMA_INVALID", "Invalid lease_generation")
        _ident(worker_id, "worker_id")
        if not isinstance(result, dict):
            raise SchemaInvalid("SCHEMA_INVALID", "Processor result must be a JSON object")
        data = _json(result)
        if len(data) > MAX_RESULT_BYTES:
            raise LimitExceeded("LIMIT_EXCEEDED", "Candidate result exceeds 16 MiB")
        with self.store.transaction():
            existing = self._job(job_id)
            if existing["commit"] is not None:
                commit = existing["commit"]
                if commit["lease_generation"] != lease_generation or commit["worker_id"] != worker_id:
                    raise Conflict("STALE_LEASE", "Result belongs to an old worker lease")
                if commit["artifact"]["sha256"] != S.sha256_bytes(data):
                    raise Conflict("IDEMPOTENCY_CONFLICT", "Committed result cannot be replaced")
                return commit
            job = self._lease(job_id, lease_generation, worker_id)
            if job["state"] == "cancel_requested":
                raise Conflict("CANCELLED", "Cancellation was accepted before result commit")
            current = self.store.get_task(job["task_id"])
            artifact = self.store._put_artifact(data)
            commit = {"id": "commit-" + uuid.uuid4().hex, "job_id": job_id, "task_id": job["task_id"],
                      "input_revision": job["input_revision"], "input_sha256": job["input_sha256"],
                      "current_revision": current["revision"], "current_sha256": self.store.task_digest(current),
                      "input_still_current": (current["revision"] == job["input_revision"] and
                                              self.store.task_digest(current) == job["input_sha256"]),
                      "lease_generation": lease_generation, "worker_id": worker_id,
                      "processor_version": job["processor_version"], "idempotency_key": job["idempotency_key"],
                      "disposition": "candidate", "artifact": artifact, "created_at": self._now()}
            body = _json(commit).decode("utf-8")
            self.store.conn.execute("INSERT INTO job_commits(job_id,id,body) VALUES(?,?,?)", (job_id, commit["id"], body))
            self.store.conn.execute("INSERT INTO durable_commits(id,body) VALUES(?,?)", (commit["id"], body))
            self._end(job, "succeeded")
            return commit

    def fail(self, job_id: str, lease_generation: int, worker_id: str,
             error_code: str = "PROCESSOR_FAILED", *, diagnostic: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(error_code, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", error_code) is None:
            raise SchemaInvalid("SCHEMA_INVALID", "Invalid error_code")
        with self.store.transaction():
            job = self._lease(job_id, lease_generation, worker_id)
            if diagnostic is not None and error_code != "STORAGE_FULL":
                artifact = self.store._put_artifact(_json(diagnostic))
                self.store.conn.execute("UPDATE attempts SET diagnostic_artifact_id=? WHERE job_id=? AND attempt=?",
                                        (artifact["id"], job_id, job["attempt"]))
            self._end(job, "cancelled" if job["state"] == "cancel_requested" else "failed", error_code)
            if error_code == "STORAGE_FULL":
                self.store.conn.execute("UPDATE batches SET dispatch_paused=1")
            return self.get_job(job_id)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            job = self._job(job_id)
            if job["commit"] is not None:
                raise Conflict("ALREADY_COMMITTED", "The job result is already committed")
            if job["state"] in ("queued", "failed"):
                self.store.conn.execute("UPDATE jobs SET state='cancelled',error_code='CANCELLED' WHERE id=?", (job_id,))
            elif job["state"] == "running":
                self.store.conn.execute("UPDATE jobs SET state='cancel_requested' WHERE id=?", (job_id,))
            return self.get_job(job_id)

    def acknowledge_cancel(self, job_id: str, lease_generation: int, worker_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            job = self._lease(job_id, lease_generation, worker_id)
            if job["state"] != "cancel_requested":
                raise Conflict("INVALID_JOB_STATE", "Cancellation was not requested")
            self._end(job, "cancelled", "CANCELLED")
            return self.get_job(job_id)

    def pause(self, batch_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            self.get_batch(batch_id)
            self.store.conn.execute("UPDATE batches SET dispatch_paused=1 WHERE id=?", (batch_id,))
            return self.get_batch(batch_id)

    def resume(self, batch_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            self.get_batch(batch_id)
            self.store.conn.execute("UPDATE batches SET dispatch_paused=0 WHERE id=?", (batch_id,))
            return self.get_batch(batch_id)

    def retry_failed(self, job_id: str) -> Dict[str, Any]:
        with self.store.transaction():
            job = self._job(job_id)
            if job["state"] != "failed":
                raise Conflict("INVALID_JOB_STATE", "Only failed jobs can be retried")
            self.store.conn.execute("UPDATE jobs SET state='queued',worker_id=NULL,session_id=NULL,lease_expires_at=NULL, "
                                    "error_code=NULL WHERE id=?", (job_id,))
            return self.get_job(job_id)

    def control_batch(self, batch_id: str, action: str, *, command_id: Optional[str] = None) -> Dict[str, Any]:
        if action not in ("pause_dispatch", "resume_dispatch", "cancel", "retry_failed"):
            raise SchemaInvalid("SCHEMA_INVALID", "Unsupported batch control")
        request = {"kind": "queue.control_batch", "batch_id": batch_id, "action": action}
        with self.store.transaction():
            receipt = self._receipt(command_id, request)
            if receipt is not None:
                return receipt
            batch = self.get_batch(batch_id)
            outcomes = []
            if action in ("pause_dispatch", "resume_dispatch"):
                (self.pause if action == "pause_dispatch" else self.resume)(batch_id)
            else:
                for job in batch["jobs"]:
                    try:
                        if action == "cancel":
                            value = self.cancel(job["id"])
                        elif job["state"] == "failed":
                            value = self.retry_failed(job["id"])
                        else:
                            outcomes.append({"job_id": job["id"], "status": "unchanged", "state": job["state"]})
                            continue
                        outcomes.append({"job_id": job["id"], "status": "accepted", "state": value["state"]})
                    except Conflict as exc:
                        outcomes.append({"job_id": job["id"], "status": "rejected", "error_code": exc.code})
            result = {"batch_id": batch_id, "action": action, "outcomes": outcomes}
            self._save_receipt(command_id, request, result)
            return result

    def control_job(self, job_id: str, action: str, *, command_id: Optional[str] = None) -> Dict[str, Any]:
        if action not in ("cancel", "retry_failed"):
            raise SchemaInvalid("SCHEMA_INVALID", "Unsupported job control")
        request = {"kind": "queue.control_job", "job_id": job_id, "action": action}
        with self.store.transaction():
            receipt = self._receipt(command_id, request)
            if receipt is not None:
                return receipt
            value = (self.cancel if action == "cancel" else self.retry_failed)(job_id)
            result = {"job_id": job_id, "action": action, "state": value["state"]}
            self._save_receipt(command_id, request, result)
            return result

    def recover(self, *, restart: bool = False) -> List[Dict[str, Any]]:
        """Recover old sessions; expired live-session leases fail until explicit retry."""
        with self.store.transaction():
            rows = self.store.conn.execute("SELECT id FROM jobs WHERE state IN ('running','cancel_requested','queued','failed')").fetchall()
            changes = []
            for (job_id,) in rows:
                job = self._job(job_id)
                if job["commit"] is not None:
                    self._end(job, "succeeded")
                    changes.append({"job_id": job_id, "state": "succeeded"})
                    continue
                if job["state"] not in ("running", "cancel_requested"):
                    continue
                old_session = job["session_id"] != self.session_id
                expired = job["lease_expires_at"] <= self.clock()
                if not ((restart and old_session) or expired):
                    continue
                state = "cancelled" if job["state"] == "cancel_requested" else ("queued" if old_session else "failed")
                self._end(job, "cancelled" if state == "cancelled" else "interrupted", "LEASE_LOST")
                self.store.conn.execute("UPDATE jobs SET state=?,lease_generation=lease_generation+1,worker_id=NULL,session_id=NULL, "
                                        "lease_expires_at=NULL,error_code=? WHERE id=?", (state, "LEASE_LOST", job_id))
                changes.append({"job_id": job_id, "state": state})
            return changes

    def execute_once(self, processor: Callable[[Dict[str, Any], WorkerContext], Dict[str, Any]], *,
                     worker_id: str = "worker-local", batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Execute one claim; callers may create at most the configured worker count.

        A stalled processor cannot be killed safely by this API. It must implement
        bounded subprocess waits and cooperative checks; lease expiry fences its
        output, and recovery records failure instead of claiming work completed.
        """
        with self.store.transaction():
            job = self.claim(worker_id, batch_id)
            if job is None:
                return None
            context = WorkerContext(self, job)
            slot = (job["id"], job["lease_generation"])
            self.store._queue_active_processors.add(slot)
        try:
            context.raise_if_cancelled()
            # A processor receives its own copy so edits to its input dictionary
            # cannot change the lease identity used by the runner's commit path.
            result = processor(json.loads(_json(job)), context)
            self.finish(job["id"], job["lease_generation"], worker_id, result)
        except Exception as exc:
            with self.store.transaction():
                current = self._job(job["id"])
                if current["commit"] is not None:
                    return self.get_job(job["id"])
                if (current["lease_generation"] != job["lease_generation"] or
                        current["worker_id"] != worker_id or current["state"] not in ("running", "cancel_requested")):
                    return self.get_job(job["id"])
                if current["lease_expires_at"] <= self.clock():
                    self.recover()
                else:
                    code = exc.code if isinstance(exc, WorkspaceError) else "PROCESSOR_FAILED"
                    if isinstance(exc, OSError) and (exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", 122)) or
                                                      getattr(exc, "winerror", None) == 112):
                        code = "STORAGE_FULL"
                    self.fail(job["id"], job["lease_generation"], worker_id, code,
                              diagnostic={"stage": "processor_or_commit", "exception_type": type(exc).__name__,
                                          "error_code": code, "job_id": job["id"], "attempt": job["attempt"],
                                          "input_revision": job["input_revision"], "input_sha256": job["input_sha256"]})
        finally:
            with self.store.transaction():
                self.store._queue_active_processors.discard(slot)
        return self.get_job(job["id"])

    def run_pending(self, processor: Callable[[Dict[str, Any], WorkerContext], Dict[str, Any]], *,
                    worker_id: str = "worker-local", batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Drain currently dispatchable work sequentially, isolating item failures."""
        results = []
        while True:
            result = self.execute_once(processor, worker_id=worker_id, batch_id=batch_id)
            if result is None:
                return results
            results.append(result)
