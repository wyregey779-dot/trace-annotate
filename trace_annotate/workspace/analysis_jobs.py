"""Durable playback-only objective analysis, sharing the ordinary queue/slot.

An analysis job creates a candidate. It never edits annotations or claims to
measure the original source: only the registered immutable playback is present
in a workspace. Saving any report remains an explicit revision/CAS command.
"""
from __future__ import annotations

import json
import math
import stat
import threading
from pathlib import PurePosixPath

from ..temporal.media import stream_sha256
from . import analysis_service as A
from . import professional_schema as PS
from . import schema as S
from .errors import Conflict, SchemaInvalid, WorkspaceError
from .http_media import resolve_media_object
from .queue import DurableQueue, MAX_RESULT_BYTES
from .slots import SlotClosed, SlotTimeout, get_prepare_slot
from .store import _safe

OPERATION = "analyze-media"
VERSION = "1"
RESULT_KIND = "traceannotate.analysis-candidate"
DOMAIN = "playback_seconds"


def _parameters(stream_index=None, window=None, subject_annotation_id=None):
    if stream_index is not None and (type(stream_index) is not int or stream_index < 0):
        raise SchemaInvalid("SCHEMA_INVALID", "stream_index must be a nonnegative integer")
    if subject_annotation_id is not None:
        S.require_id(subject_annotation_id)
        if window is None:
            raise SchemaInvalid("SCHEMA_INVALID", "An annotation subject requires an interval")
    if window is not None:
        if (not isinstance(window, dict) or set(window) != {"start", "end"}
                or any(S.rational(window[key]) != window[key] for key in ("start", "end"))
                or not 0 <= S.rational_value(window["start"]) < S.rational_value(window["end"])):
            raise SchemaInvalid("SCHEMA_INVALID", "window must be a reduced rational half-open interval")
    return json.loads(S.canonical_json({"stream_index": stream_index, "window": window,
        "subject_annotation_id": subject_annotation_id, "time_domain": DOMAIN}))


def _validate_job(job):
    if job["operation"] != OPERATION or job["processor_version"] != VERSION:
        raise WorkspaceError("UNSUPPORTED_PROCESSOR", "No installed analysis processor matches this frozen job")
    parameters = job["parameters"]
    if (not isinstance(parameters, dict) or set(parameters) != {
            "stream_index", "window", "subject_annotation_id", "time_domain"}
            or parameters["time_domain"] != DOMAIN
            or parameters != _parameters(parameters["stream_index"], parameters["window"],
                                         parameters["subject_annotation_id"])):
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid frozen analysis parameters")
    media, task = job["media_snapshot"], job["task_snapshot"]
    spec = {key: job[key] for key in ("task_id", "operation", "input_revision", "input_sha256", "processor_version", "parameters")}
    if S.sha256_text(spec) != job["idempotency_key"] or job["id"] != "job-" + job["idempotency_key"]:
        raise WorkspaceError("RECOVERY_REQUIRED", "Analysis configuration identity is inconsistent")
    PS.validate_media(media)
    if task["id"] != job["task_id"] or task["media_id"] != media["id"] or task["revision"] != job["input_revision"]:
        raise WorkspaceError("RECOVERY_REQUIRED", "Analysis snapshot identity is inconsistent")
    window = parameters["window"]
    if window is not None and S.rational_value(window["end"]) > S.rational_value(media["duration"]):
        raise SchemaInvalid("SCHEMA_INVALID", "Analysis interval exceeds playback duration")
    subject = parameters["subject_annotation_id"]
    if subject is not None:
        annotation = next((a for a in task["annotations"] if a["id"] == subject), None)
        if (annotation is None or annotation["type"] != "segment" or window != {
                "start": annotation["start"]["seconds"], "end": annotation["end"]["seconds"]}):
            raise SchemaInvalid("SCHEMA_INVALID", "Analysis interval must match its frozen segment subject")
    return parameters


class AnalysisJobs:
    """One optional drainer; all wrappers still share durable leases and decoding.

    close() reports actual runner liveness. The caller must keep the store open
    until stopped is true. Queued jobs survive shutdown and resume on reopening.
    """

    def __init__(self, store, *, queue=None, closing=None, poll_seconds=.1, slot_wait_seconds=900):
        for value in (poll_seconds, slot_wait_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SchemaInvalid("SCHEMA_INVALID", "Analysis waits must be finite positive seconds")
        if closing is not None and not isinstance(closing, threading.Event):
            raise SchemaInvalid("SCHEMA_INVALID", "closing must be a threading.Event")
        self.store = store
        self.queue = queue if queue is not None else DurableQueue(store)
        if self.queue.store is not store:
            raise SchemaInvalid("SCHEMA_INVALID", "Analysis queue belongs to another workspace")
        self._external_closing = closing
        self._closing = threading.Event()
        self._condition = threading.Condition()
        self._runners = set()
        self._active_jobs = set()
        self._heartbeats = set()
        self._drainer = None
        self._drainer_error = None
        self.poll_seconds, self.slot_wait_seconds = float(poll_seconds), float(slot_wait_seconds)

    def is_closing(self):
        return self._closing.is_set() or (self._external_closing is not None and self._external_closing.is_set())

    def submit(self, task_id, *, command_id, expected_revision, expected_sha256,
               stream_index=None, window=None, subject_annotation_id=None):
        S.require_id(command_id)
        parameters = _parameters(stream_index, window, subject_annotation_id)
        items = [{"task_id": task_id, "expected_revision": expected_revision, "expected_sha256": expected_sha256}]
        request = {"kind": "queue.create_batch", "items": items, "operation": OPERATION,
                   "processor_version": VERSION, "parameters": parameters}
        with self.store.transaction():
            # A lost response remains replayable even after closing or an edit.
            replay = self.queue._receipt(command_id, request)
            if replay is not None:
                return replay
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace accepts no new analysis jobs")
            batch = self.queue.create_batch(items, operation=OPERATION, processor_version=VERSION,
                                             parameters=parameters, command_id=command_id)
            for job in batch["jobs"]:
                _validate_job(job)
            return batch

    def _batches(self, batch_id=None):
        batches = [self.queue.get_batch(batch_id)] if batch_id is not None else self.queue.list_batches()
        selected = []
        for batch in batches:
            matching = [job["operation"] == OPERATION for job in batch["jobs"]]
            if any(matching) and not all(matching):
                raise WorkspaceError("RECOVERY_REQUIRED", "Mixed-operation analysis batch is inconsistent")
            if matching and all(matching):
                if not batch["dispatch_paused"] and any(job["state"] == "queued" for job in batch["jobs"]):
                    selected.append(batch["id"])
            elif batch_id is not None:
                raise Conflict("UNSUPPORTED_PROCESSOR", "Batch is not an analysis batch")
        return selected

    def _processor(self, job, context):
        parameters = _validate_job(job)
        self._validate_snapshot(job)
        stop = threading.Event()
        lost = threading.Event()

        def renew():
            try:
                while not stop.wait(min(5.0, self.queue.lease_seconds / 3.0)):
                    try:
                        if context.heartbeat()["cancel_requested"]:
                            return
                    except Exception:
                        lost.set()
                        return
            finally:
                with self._condition:
                    self._heartbeats.discard(threading.current_thread())
                    self._condition.notify_all()

        def cancelled():
            if self.is_closing():
                try:
                    self.queue.cancel(job["id"])
                except Conflict as exc:
                    if exc.code != "ALREADY_COMMITTED":
                        raise
                return True
            if lost.is_set():
                raise Conflict("LEASE_LOST", "Analysis lease renewal failed")
            return context.cancelled()

        with self._condition:
            self._active_jobs.add(job["id"])
        heartbeat = threading.Thread(target=renew, name="analysis-heartbeat", daemon=True)
        with self._condition:
            self._heartbeats.add(heartbeat)
        heartbeat.start()
        try:
            if cancelled():
                raise WorkspaceError("CANCELLED", "Analysis was cancelled before resolving playback")
            # Resolving the current binding verifies its whole hash. The frozen
            # snapshot must agree too; a proxy is never relabelled as original.
            source, digest, size = resolve_media_object(self.store, job["media_snapshot"]["id"], "playback_path")
            if (digest, str(size)) != (job["media_snapshot"]["playback"]["sha256"],
                                       job["media_snapshot"]["playback"]["size_bytes"]):
                raise Conflict("MEDIA_CHANGED", "Playback binding differs from the frozen analysis input")
            relative = f"tmp/analysis/jobs/{job['id']}/{job['attempt']}-{job['lease_generation']}"
            work = self.store.root / relative
            _safe(work)
            result = A.analyze_media(source, job["media_snapshot"], work, cancelled=cancelled,
                stream_index=parameters["stream_index"], window=parameters["window"],
                subject_annotation_id=parameters["subject_annotation_id"], time_domain=DOMAIN)
            evidence = []
            for path in sorted(work.iterdir()):
                if cancelled():
                    raise WorkspaceError("CANCELLED", "Analysis cancelled while hashing evidence")
                _safe(path)
                if not stat.S_ISREG(path.stat().st_mode):
                    raise WorkspaceError("RECOVERY_REQUIRED", "Analysis evidence is not a regular file")
                evidence.append({"relative_path": relative + "/" + path.name,
                                 "sha256": stream_sha256(path, cancelled=cancelled),
                                 "size_bytes": str(path.stat().st_size)})
            candidate = {"kind": RESULT_KIND, "version": 1, "processor_version": VERSION,
                "job_id": job["id"], "task_id": job["task_id"], "media_id": job["media_snapshot"]["id"],
                "input_revision": job["input_revision"], "input_sha256": job["input_sha256"],
                "file_sha256": digest, "time_domain": DOMAIN, "reports": result["reports"],
                "evidence": evidence, "semantic_listening": "not_performed", "source_unchanged": True,
                "scope": "registered_playback_only; original_source_not_measured; candidate_not_applied"}
            self._validate_candidate(candidate, job)
            context.raise_if_cancelled()
            return candidate
        except InterruptedError as exc:
            raise WorkspaceError("CANCELLED", "Objective analysis was cancelled") from exc
        finally:
            stop.set()
            heartbeat.join(timeout=6)
            with self._condition:
                self._active_jobs.discard(job["id"])
                self._condition.notify_all()

    def run_pending(self, *, batch_id=None, worker_id="analysis-local"):
        with self._condition:
            if self.is_closing():
                return []
            token = object()
            self._runners.add(token)
        results = []
        slot = get_prepare_slot(self.store.root)
        try:
            for ident in self._batches(batch_id):
                while not self.is_closing():
                    try:
                        with slot.hold(closing=self.is_closing, timeout=self.slot_wait_seconds):
                            if self.is_closing():
                                break
                            result = self.queue.execute_once(self._processor, batch_id=ident, worker_id=worker_id)
                    except (SlotClosed, SlotTimeout):
                        return results
                    if result is None:
                        break
                    results.append(result)
            return results
        finally:
            with self._condition:
                self._runners.discard(token)
                self._condition.notify_all()

    def start(self):
        with self._condition:
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Cannot start a closing analysis drainer")
            if self._drainer is not None and self._drainer.is_alive():
                return self
            def drain():
                try:
                    while not self.is_closing():
                        self.run_pending()
                        self._closing.wait(self.poll_seconds)
                except Exception as exc:
                    self._drainer_error = type(exc).__name__
                finally:
                    with self._condition:
                        self._condition.notify_all()
            self._drainer = threading.Thread(target=drain, name="analysis-drainer", daemon=True)
            self._drainer.start()
        return self

    def close(self, *, timeout=60):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise SchemaInvalid("SCHEMA_INVALID", "close timeout must be finite nonnegative seconds")
        import time
        self._closing.set()
        with self._condition:
            ids = list(self._active_jobs)
        for ident in ids:
            try:
                self.queue.cancel(ident)
            except Conflict as exc:
                if exc.code != "ALREADY_COMMITTED":
                    raise
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._runners or self._heartbeats or (self._drainer is not None and self._drainer.is_alive()):
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._condition.wait(min(left, .05))
            return {"stopped": not self._runners and not self._heartbeats and not (self._drainer is not None and self._drainer.is_alive()),
                    "active_jobs": sorted(self._active_jobs), "closing": True, "drainer_error": self._drainer_error}

    def _validate_candidate(self, value, job):
        parameters = _validate_job(job)
        fixed = {"kind": RESULT_KIND, "version": 1, "processor_version": VERSION,
            "job_id": job["id"], "task_id": job["task_id"], "media_id": job["media_snapshot"]["id"],
            "input_revision": job["input_revision"], "input_sha256": job["input_sha256"],
            "file_sha256": job["media_snapshot"]["playback"]["sha256"], "time_domain": DOMAIN,
            "semantic_listening": "not_performed", "source_unchanged": True,
            "scope": "registered_playback_only; original_source_not_measured; candidate_not_applied"}
        if (not isinstance(value, dict) or set(value) != set(fixed) | {"reports", "evidence"}
                or any(value[key] != item for key, item in fixed.items())
                or not isinstance(value["reports"], list) or not 1 <= len(value["reports"]) <= 64
                or not isinstance(value["evidence"], list) or len(value["evidence"]) > 256):
            raise WorkspaceError("RECOVERY_REQUIRED", "Analysis candidate identity is inconsistent")
        for report in value["reports"]:
            PS.validate_analysis(report, job["media_snapshot"])
            if (report["kind"] not in {"technical_declared", "loudness", "frame_timing"}
                    or report["time_domain"] != DOMAIN or report["file_sha256"] != fixed["file_sha256"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis report is not bound to playback")
            if report["kind"] == "loudness" and (report["window"] != parameters["window"] or
                    report["subject_annotation_id"] != parameters["subject_annotation_id"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis report interval is inconsistent")
        prefix = f"tmp/analysis/jobs/{job['id']}/{job['attempt']}-{job['lease_generation']}/"
        paths = set()
        for evidence in value["evidence"]:
            if (not isinstance(evidence, dict) or set(evidence) != {"relative_path", "sha256", "size_bytes"}
                    or not isinstance(evidence["relative_path"], str) or not evidence["relative_path"].startswith(prefix)):
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis evidence descriptor is invalid")
            path = evidence["relative_path"]
            name = path[len(prefix):]
            if (not name or PurePosixPath(name).name != name or "\\" in name or ":" in name or name in {".", ".."}
                    or path in paths or not isinstance(evidence["sha256"], str) or S.HASH_RE.fullmatch(evidence["sha256"]) is None
                    or not isinstance(evidence["size_bytes"], str) or not evidence["size_bytes"].isdigit()):
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis evidence path or identity is invalid")
            paths.add(path)
        return value

    def _validate_snapshot(self, job):
        with self.store.transaction():
            if (self.store.get_media(job["media_snapshot"]["id"]) != job["media_snapshot"]
                    or self.store.task_digest(job["task_snapshot"]) != job["input_sha256"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis frozen input was altered")

    def result(self, job_id):
        S.require_id(job_id)
        try:
            return self._read_result(job_id)
        except (OSError, ValueError, KeyError, TypeError, RecursionError, SchemaInvalid) as exc:
            raise WorkspaceError("RECOVERY_REQUIRED", "Analysis candidate or commit is corrupt") from exc

    def _read_result(self, job_id):
        S.require_id(job_id)
        with self.store.transaction():
            job = self.queue.get_job(job_id)
            if job["operation"] != OPERATION:
                raise Conflict("UNSUPPORTED_PROCESSOR", "Job is not an analysis job")
            if job["state"] != "succeeded" or job["commit"] is None:
                raise Conflict("ANALYSIS_NOT_READY", "Analysis has no committed successful candidate")
            _validate_job(job)
            self._validate_snapshot(job)
            commit = job["commit"]
            durable = self.store.conn.execute("SELECT body FROM durable_commits WHERE id=?", (commit["id"],)).fetchone()
            descriptor = commit["artifact"]
            row = self.store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE id=?", (descriptor["id"],)).fetchone()
            if (durable is None or json.loads(durable[0]) != commit or commit["job_id"] != job_id
                    or commit["disposition"] != "candidate" or row is None or row[0] != "objects/" + row[1]
                    or any(commit[key] != job[key] for key in ("task_id", "input_revision", "input_sha256",
                                                               "processor_version", "idempotency_key", "lease_generation", "worker_id"))
                    or descriptor != {"id": descriptor["id"], "relative_path": row[0], "sha256": row[1], "size_bytes": str(row[2])}
                    or not 0 < row[2] <= MAX_RESULT_BYTES):
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis commit or artifact binding is inconsistent")
            path = self.store.root / row[0]
            _safe(path)
            try:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_size != row[2]:
                    raise ValueError("Candidate is not the registered regular file")
                with path.open("rb") as stream:
                    data = stream.read(MAX_RESULT_BYTES + 1)
                if len(data) != row[2] or S.sha256_bytes(data) != row[1]:
                    raise ValueError("Candidate identity mismatch")
                value = json.loads(data)
                if S.canonical_json(value) != data:
                    raise ValueError("Candidate is not canonical finite JSON")
                try:
                    self._validate_candidate(value, job)
                except SchemaInvalid as exc:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Analysis candidate schema is corrupt") from exc
                return value
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise WorkspaceError("RECOVERY_REQUIRED", "Analysis candidate is unavailable or corrupt") from exc

    def apply(self, job_id, *, command_id, expected_revision, expected_sha256, request_body):
        """Commit only server-loaded verified reports, atomically and explicitly.

        Whole-file reports survive unrelated label edits. Segment-associated
        reports must still match their subject's exact current interval; the
        existing bulk command performs report validation under the same CAS.
        """
        with self.store.transaction():
            result = self.result(job_id)
            arguments = dict(command_id=command_id, task_id=result["task_id"],
                expected_revision=expected_revision, expected_sha256=expected_sha256,
                operation="set_analysis_reports", payload={"reports": result["reports"]}, request_body=request_body)
            # Delegate receipt identity checks to the single command journal.
            # A later boundary edit must not invalidate an already saved reply.
            if self.store.conn.execute("SELECT 1 FROM command_receipts WHERE command_id=?", (command_id,)).fetchone():
                return self.store.apply_task_command(**arguments)["response"]
            current = self.store.get_task(result["task_id"])
            media = self.store.get_media(current["media_id"])
            if media["id"] != result["media_id"] or media["playback"]["sha256"] != result["file_sha256"]:
                raise Conflict("MEDIA_CHANGED", "Analysis candidate does not match the task playback")
            for report in result["reports"]:
                if (report["subject_annotation_id"] is not None and
                        PS.derived_analysis_status(current, media, dict(report, status="available"))["status"] == "stale"):
                    raise Conflict("ANALYSIS_STALE", "Analysis segment subject changed; request a new analysis")
            return self.store.apply_task_command(**arguments)["response"]
