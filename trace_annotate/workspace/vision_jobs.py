"""Durable playback-only vision candidate jobs.

A vision job runs local extractors (scene cuts, face, OCR, CLIP) against the
registered immutable playback and returns a candidate artifact. It never edits
annotations, never claims to measure the original source, and never treats a
model result as reviewed ground truth. Saving candidates as task annotations
remains an explicit revision/CAS command with model_candidate provenance.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from ..config import MetricThresholds
from ..extractors.clip_engine import CLIPExtractor
from ..extractors.faces import FaceExtractor
from ..extractors.ocr import OCRExtractor
from ..extractors.scene import SceneDetector, SceneDetectionError
from ..extractors.shot_attributes import classify_framing
from ..extractors.vlm import VLMExtractor
from ..temporal.media import stream_sha256
from . import professional_schema as PS
from . import schema as S
from . import vision_sampling as VS
from .semantic_schema import SEMANTIC_VERSION
from .errors import Conflict, SchemaInvalid, WorkspaceError
from .http_media import resolve_media_object
from .queue import DurableQueue, MAX_RESULT_BYTES
from .slots import SlotClosed, SlotTimeout, get_prepare_slot
from .store import _safe

OPERATION = "vision-media"
VERSION = "1"
RESULT_KIND = "traceannotate.vision-candidate"
DOMAIN = "playback_seconds"
MAX_SCENES = 512
MAX_KEYFRAMES = 512
MAX_FRAME_BYTES = 16 * 1024 * 1024
FRAME_QUALITY = 3
CANDIDATE_LABEL_PREFIX = "镜头候选"
CANDIDATE_TRACK = "镜头"


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' positive' if positive else ''}")
    return result


def _parameters(window: Optional[Dict[str, Any]] = None,
                subject_annotation_id: Optional[str] = None) -> Dict[str, Any]:
    if subject_annotation_id is not None:
        S.require_id(subject_annotation_id)
        if window is None:
            raise SchemaInvalid("SCHEMA_INVALID", "An annotation subject requires an interval")
    if window is not None:
        if (not isinstance(window, dict) or set(window) != {"start", "end"}
                or any(S.rational(window[key]) != window[key] for key in ("start", "end"))
                or not 0 <= S.rational_value(window["start"]) < S.rational_value(window["end"])):
            raise SchemaInvalid("SCHEMA_INVALID", "window must be a reduced rational half-open interval")
    return json.loads(S.canonical_json({"window": window, "subject_annotation_id": subject_annotation_id,
                                        "time_domain": DOMAIN}))


def _validate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if job["operation"] != OPERATION or job["processor_version"] != VERSION:
        raise WorkspaceError("UNSUPPORTED_PROCESSOR", "No installed vision processor matches this frozen job")
    parameters = job["parameters"]
    if (not isinstance(parameters, dict) or set(parameters) != {"window", "subject_annotation_id", "time_domain"}
            or parameters["time_domain"] != DOMAIN
            or parameters != _parameters(parameters["window"], parameters["subject_annotation_id"])):
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid frozen vision parameters")
    media, task = job["media_snapshot"], job["task_snapshot"]
    spec = {key: job[key] for key in ("task_id", "operation", "input_revision", "input_sha256", "processor_version", "parameters")}
    if S.sha256_text(spec) != job["idempotency_key"] or job["id"] != "job-" + job["idempotency_key"]:
        raise WorkspaceError("RECOVERY_REQUIRED", "Vision configuration identity is inconsistent")
    PS.validate_media(media)
    if task["id"] != job["task_id"] or task["media_id"] != media["id"] or task["revision"] != job["input_revision"]:
        raise WorkspaceError("RECOVERY_REQUIRED", "Vision snapshot identity is inconsistent")
    window = parameters["window"]
    if window is not None and S.rational_value(window["end"]) > S.rational_value(media["duration"]):
        raise SchemaInvalid("SCHEMA_INVALID", "Vision interval exceeds playback duration")
    subject = parameters["subject_annotation_id"]
    if subject is not None:
        annotation = next((a for a in task["annotations"] if a["id"] == subject), None)
        if (annotation is None or annotation["type"] != "segment" or window != {
                "start": annotation["start"]["seconds"], "end": annotation["end"]["seconds"]}):
            raise SchemaInvalid("SCHEMA_INVALID", "Vision interval must match its frozen segment subject")
    return parameters


def _seconds_ratio(seconds: float) -> Dict[str, str]:
    value = Fraction(seconds).limit_denominator(1_000_000)
    return {"n": str(value.numerator), "d": str(value.denominator)}


def _frame_timestamp(seconds: float) -> str:
    """Format a ffmpeg -ss timestamp from seconds with microsecond precision."""
    return f"{float(seconds):.6f}"


def _extract_keyframe(ffmpeg: str, video_path: Path, seconds: float, output: Path) -> None:
    """Decode exactly one frame at a verified timestamp into a JPEG file."""
    if output.exists():
        raise WorkspaceError("RECOVERY_REQUIRED", "Vision frame output already exists")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "error",
               "-xerror", "-err_detect", "explode", "-ss", _frame_timestamp(seconds),
               "-i", str(video_path), "-map", "0:v:0", "-frames:v", "1",
               "-q:v", str(FRAME_QUALITY), "-y", str(output)]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=60, encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceError("VISION_FRAME_FAILED", f"Keyframe extraction could not complete: {exc}") from exc
    if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        detail = result.stderr.strip()[-2000:] or "no diagnostic output"
        raise WorkspaceError("VISION_FRAME_FAILED", f"Keyframe extraction failed: {detail}")
    if output.stat().st_size > MAX_FRAME_BYTES:
        raise WorkspaceError("ANALYSIS_LIMIT_EXCEEDED", "Vision keyframe exceeds the frame size budget")


def _run_scenes(scene_detector: SceneDetector, video_path: Path,
                thresholds: Any) -> Dict[str, Any]:
    try:
        segments = scene_detector.detect_scenes_ffmpeg(str(video_path))
    except (SceneDetectionError, ValueError, OSError) as exc:
        return {"status": "failed", "error": str(exc), "scenes": []}
    if len(segments) > MAX_SCENES:
        return {"status": "failed", "error": f"scene count exceeds {MAX_SCENES}", "scenes": []}
    return {"status": "success", "scenes": segments, "error": None}


def _run_face(frame: Any, face_extractor: Optional[FaceExtractor]) -> Dict[str, Any]:
    if face_extractor is None:
        return {"status": "not_run", "reason": "face_weights_unavailable"}
    try:
        result = face_extractor.detect_faces(frame)
    except Exception as exc:  # A vision helper must never crash the whole job.
        return {"status": "failed", "error": str(exc)}
    return result


def _shot_framing(frame: Any, face_result: Dict[str, Any]) -> Dict[str, Any]:
    """Framing from the face boxes this project actually measured.

    A face-only ratio is explicitly downgraded and is never called a close-up
    merely because the face is large. Without a face result the framing is
    ``not_applicable``; it is not guessed from the frame.
    """
    if face_result.get("status") != "success":
        return {"status": "not_run", "framing": None,
                "reason": "face extractor did not report boxes"}
    faces = face_result.get("faces_detected") or []
    if not faces:
        return {"status": "success", "framing": "不适用", "subjects": 0,
                "reason": "画面内没有可用主体框", "confidence": 0.0,
                "uncertainty": "未检测到主体不等于画面中不存在主体"}
    height, width = frame.shape[:2]
    boxes = []
    for face in faces:
        x, y, box_width, box_height = [float(value) for value in face["box"]]
        boxes.append((x, y, x + box_width, y + box_height))
    result = classify_framing(boxes, (width, height), subject_kind="face")
    result["status"] = "success"
    return result


def _run_ocr(frame: Any, ocr_extractor: Optional[OCRExtractor]) -> Dict[str, Any]:
    if ocr_extractor is None or ocr_extractor.engine is None:
        return {"status": "not_run", "reason": "ocr_runtime_unavailable"}
    try:
        result = ocr_extractor.extract_text(frame)
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    result["status"] = "success"
    return result


def _run_clip(frame: Any, clip_extractor: Optional[CLIPExtractor]) -> Dict[str, Any]:
    if clip_extractor is None:
        return {"status": "not_run", "reason": "clip_weights_unavailable"}
    try:
        result = clip_extractor.classify(frame)
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    return result


def _run_vlm(frame: Any, vlm_extractor: Optional[VLMExtractor]) -> Dict[str, Any]:
    if vlm_extractor is None:
        return {"status": "not_run", "reason": "vlm_weights_unavailable", "caption": ""}
    try:
        result = vlm_extractor.caption(frame)
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "caption": ""}
    return result


class VisionJobs:
    """One optional drainer sharing durable leases; wrappers never write tasks.

    close() reports actual runner liveness. Queued jobs survive shutdown and
    resume on reopening. A successful job is a candidate, never a review.
    """

    def __init__(self, store: Any, *, queue: Optional[DurableQueue] = None,
                 closing: Optional[threading.Event] = None, poll_seconds: float = .1,
                 slot_wait_seconds: float = 900, face_model_path: Optional[str] = None,
                 clip_model_path: Optional[str] = None, vlm_model_path: Optional[str] = None):
        for value in (poll_seconds, slot_wait_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SchemaInvalid("SCHEMA_INVALID", "Vision waits must be finite positive seconds")
        if closing is not None and not isinstance(closing, threading.Event):
            raise SchemaInvalid("SCHEMA_INVALID", "closing must be a threading.Event")
        self.store = store
        self.queue = queue if queue is not None else DurableQueue(store)
        if self.queue.store is not store:
            raise SchemaInvalid("SCHEMA_INVALID", "Vision queue belongs to another workspace")
        self._external_closing = closing
        self._closing = threading.Event()
        self._condition = threading.Condition()
        self._runners = set()
        self._active_jobs = set()
        self._heartbeats = set()
        self._drainer = None
        self._drainer_error = None
        self.poll_seconds, self.slot_wait_seconds = float(poll_seconds), float(slot_wait_seconds)
        self.face_model_path = face_model_path
        self.clip_model_path = clip_model_path
        self.vlm_model_path = vlm_model_path
        self._extractors = None
        self._extractor_error = None

    def is_closing(self) -> bool:
        return self._closing.is_set() or (self._external_closing is not None and self._external_closing.is_set())

    def options(self) -> Dict[str, Any]:
        """Report status without loading weights.

        A capability query must not import torch/transformers or touch model
        caches: opening the editor used to block for minutes on a cold start.
        Extractor instances are built lazily by the first job, so before that
        the per-adapter state is the honest string "not_loaded" rather than
        False. After loading, "loaded" and "unavailable" distinguish a real
        adapter from a missing dependency or weight.
        """
        loaded = self._extractors is not None

        def adapter_state(key: str) -> str:
            if not loaded:
                return "not_loaded"
            instance = self._extractors.get(key) if self._extractors else None
            if key == "face":
                ready = instance is not None and instance.detector is not None
            elif key == "ocr":
                ready = instance is not None and instance.engine is not None
            else:
                ready = instance is not None and instance.model is not None
            return "loaded" if ready else "unavailable"

        capabilities = {
            "scene": bool(shutil.which("ffmpeg")),
            "face": adapter_state("face"),
            "ocr": adapter_state("ocr"),
            "clip": adapter_state("clip"),
            "vlm": adapter_state("vlm"),
        }
        return {"operation": OPERATION, "version": VERSION, "domain": DOMAIN,
                "capabilities": capabilities, "loaded": loaded,
                "extractor_error": self._extractor_error,
                "closing": self.is_closing()}

    def submit(self, task_id: str, *, command_id: str, expected_revision: int,
               expected_sha256: str, window: Optional[Dict[str, Any]] = None,
               subject_annotation_id: Optional[str] = None) -> Dict[str, Any]:
        S.require_id(command_id)
        parameters = _parameters(window, subject_annotation_id)
        items = [{"task_id": task_id, "expected_revision": expected_revision, "expected_sha256": expected_sha256}]
        request = {"kind": "queue.create_batch", "items": items, "operation": OPERATION,
                   "processor_version": VERSION, "parameters": parameters}
        with self.store.transaction():
            replay = self.queue._receipt(command_id, request)
            if replay is not None:
                return replay
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace accepts no new vision jobs")
            batch = self.queue.create_batch(items, operation=OPERATION, processor_version=VERSION,
                                             parameters=parameters, command_id=command_id)
            for job in batch["jobs"]:
                _validate_job(job)
            return batch

    def submit_many(self, items: List[Dict[str, Any]], *, command_id: str,
                    window: Optional[Dict[str, Any]] = None,
                    subject_annotation_id: Optional[str] = None) -> Dict[str, Any]:
        """Freeze several tasks into one vision batch with independent items.

        Each item keeps its own revision and content digest, so a failure in one
        task never rewrites another task's result. The batch is not a cross-task
        transaction: results commit per job, and the same command_id replays the
        same receipt instead of creating a second set of candidates.
        """
        S.require_id(command_id)
        parameters = _parameters(window, subject_annotation_id)
        if not isinstance(items, list) or not items:
            raise SchemaInvalid("SCHEMA_INVALID", "Vision batch needs at least one task")
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != {"task_id", "expected_revision", "expected_sha256"}:
                raise SchemaInvalid("SCHEMA_INVALID", "Vision items need task_id, expected_revision and expected_sha256")
            S.require_id(item["task_id"])
            if item["task_id"] in seen:
                raise SchemaInvalid("SCHEMA_INVALID", "Duplicate task in one vision batch")
            seen.add(item["task_id"])
            normalized.append({"task_id": item["task_id"],
                               "expected_revision": item["expected_revision"],
                               "expected_sha256": item["expected_sha256"]})
        request = {"kind": "queue.create_batch", "items": normalized, "operation": OPERATION,
                   "processor_version": VERSION, "parameters": parameters}
        with self.store.transaction():
            replay = self.queue._receipt(command_id, request)
            if replay is not None:
                return replay
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace accepts no new vision jobs")
            batch = self.queue.create_batch(normalized, operation=OPERATION, processor_version=VERSION,
                                            parameters=parameters, command_id=command_id)
            for job in batch["jobs"]:
                _validate_job(job)
            return batch

    def _batches(self, batch_id: Optional[str] = None) -> List[str]:
        batches = [self.queue.get_batch(batch_id)] if batch_id is not None else self.queue.list_batches()
        selected = []
        for batch in batches:
            matching = [job["operation"] == OPERATION for job in batch["jobs"]]
            if any(matching) and not all(matching):
                raise WorkspaceError("RECOVERY_REQUIRED", "Mixed-operation vision batch is inconsistent")
            if matching and all(matching):
                if not batch["dispatch_paused"] and any(job["state"] == "queued" for job in batch["jobs"]):
                    selected.append(batch["id"])
            elif batch_id is not None:
                raise Conflict("UNSUPPORTED_PROCESSOR", "Batch is not a vision batch")
        return selected

    def _load_extractors(self, thresholds: Any) -> None:
        """Build extractor instances once; missing weights degrade to not_run.

        Each extractor swallows its own load failure internally (model=None),
        so one broken adapter never disables the other extractors.
        """
        if self._extractors is not None:
            return
        try:
            face = FaceExtractor(model_path=self.face_model_path, thresholds=thresholds)
            ocr = OCRExtractor(thresholds=thresholds)
            clip = CLIPExtractor(model_name_or_path=self.clip_model_path, thresholds=thresholds)
            vlm = VLMExtractor(model_path=self.vlm_model_path)
            self._extractors = {"face": face, "ocr": ocr, "clip": clip, "vlm": vlm}
        except Exception as exc:
            self._extractor_error = str(exc)
            self._extractors = {"face": None, "ocr": None, "clip": None, "vlm": None}

    def _processor(self, job: Dict[str, Any], context: Any) -> Dict[str, Any]:
        parameters = _validate_job(job)
        self._validate_snapshot(job)
        stop = threading.Event()
        lost = threading.Event()

        def renew() -> None:
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

        def cancelled() -> bool:
            if self.is_closing():
                try:
                    self.queue.cancel(job["id"])
                except Conflict as exc:
                    if exc.code != "ALREADY_COMMITTED":
                        raise
                return True
            if lost.is_set():
                raise Conflict("LEASE_LOST", "Vision lease renewal failed")
            return context.cancelled()

        with self._condition:
            self._active_jobs.add(job["id"])
        heartbeat = threading.Thread(target=renew, name="vision-heartbeat", daemon=True)
        with self._condition:
            self._heartbeats.add(heartbeat)
        heartbeat.start()
        try:
            if cancelled():
                raise WorkspaceError("CANCELLED", "Vision was cancelled before resolving playback")
            source, digest, size = resolve_media_object(self.store, job["media_snapshot"]["id"], "playback_path")
            if (digest, str(size)) != (job["media_snapshot"]["playback"]["sha256"],
                                       job["media_snapshot"]["playback"]["size_bytes"]):
                raise Conflict("MEDIA_CHANGED", "Playback binding differs from the frozen vision input")
            relative = f"tmp/vision/jobs/{job['id']}/{job['attempt']}-{job['lease_generation']}"
            work = self.store.root / relative
            _safe(work)
            work.mkdir(parents=True, exist_ok=False)
            try:
                ffmpeg = _find_ffmpeg()
                thresholds = MetricThresholds()
                self._load_extractors(thresholds)
                extractors = self._extractors or {"face": None, "ocr": None, "clip": None}
                detector = SceneDetector(thresholds=thresholds)
                scene_result = _run_scenes(detector, Path(source), thresholds)
                keyframes: List[Dict[str, Any]] = []
                window = parameters["window"]
                scenes = scene_result["scenes"]
                if window is not None:
                    start = S.rational_value(window["start"])
                    end = S.rational_value(window["end"])
                    scenes = [scene for scene in scenes
                              if scene["end"] > start and scene["start"] < end]
                if scene_result["status"] == "success" and scenes:
                    boundary_seconds = [scene["start"] for scene in scenes]
                    boundaries = VS.boundary_indices(self.store, job["media_snapshot"]["id"], boundary_seconds)
                    plan = VS.plan_samples(
                        self.store, job["media_snapshot"]["id"], window=window,
                        uniform=min(8, max(1, len(scenes))), boundary_frames=boundaries,
                        budget=VS.SamplingBudget(max_frames=MAX_KEYFRAMES))
                    decoded = VS.decode_samples(
                        self.store, job["media_snapshot"]["id"], plan, work / "samples",
                        cancelled=cancelled, ffmpeg=ffmpeg)
                    sampling = {"frame_map_sha256": plan["frame_map_sha256"],
                                "truncated": plan["truncated"],
                                "planned": plan["planned"], "returned": plan["returned"],
                                "coverage": plan["coverage"],
                                "fingerprint": VS.samples_fingerprint(plan)}
                    for item in decoded["decoded"]:
                        if cancelled():
                            raise WorkspaceError("CANCELLED", "Vision cancelled while analysing samples")
                        seconds = S.rational_value(item["seconds"])
                        scene_index = next((index for index, scene in enumerate(scenes)
                                            if scene["start"] <= seconds < scene["end"]), None)
                        if scene_index is None:
                            continue
                        frame = _load_frame(Path(item["path"]))
                        face_result = _run_face(frame, extractors["face"])
                        framing_result = _shot_framing(frame, face_result)
                        ocr_result = _run_ocr(frame, extractors["ocr"])
                        clip_result = _run_clip(frame, extractors["clip"])
                        vlm_result = _run_vlm(frame, extractors["vlm"])
                        keyframes.append({
                            "scene_index": scene_index,
                            "frame_index": item["frame_index"],
                            "source_pts": item["source_pts"],
                            "proxy_pts": item["proxy_pts"],
                            "roles": item["roles"],
                            "timestamp": _seconds_ratio(seconds),
                            "evidence_sha256": item["sha256"],
                            "face": face_result,
                            "framing": framing_result,
                            "ocr": ocr_result,
                            "clip": clip_result,
                            "vlm": vlm_result,
                        })
                else:
                    sampling = None
                extractor_status = {
                    "scene": scene_result["status"],
                    "face": _aggregate_status(keyframes, "face"),
                    "ocr": _aggregate_status(keyframes, "ocr"),
                    "clip": _aggregate_status(keyframes, "clip"),
                    "vlm": _aggregate_status(keyframes, "vlm"),
                }
                if self._extractor_error is not None:
                    extractor_status["extractor_error"] = self._extractor_error
                evidence = []
                for path in sorted(work.rglob("*")):
                    if cancelled():
                        raise WorkspaceError("CANCELLED", "Vision cancelled while hashing evidence")
                    _safe(path)
                    if path.is_dir():
                        continue
                    if not stat.S_ISREG(path.stat().st_mode):
                        raise WorkspaceError("RECOVERY_REQUIRED", "Vision evidence is not a regular file")
                    nested = path.relative_to(work).as_posix()
                    evidence.append({"relative_path": relative + "/" + nested,
                                     "sha256": stream_sha256(path, cancelled=cancelled),
                                     "size_bytes": str(path.stat().st_size)})
                candidate = {"kind": RESULT_KIND, "version": 1, "processor_version": VERSION,
                    "job_id": job["id"], "task_id": job["task_id"], "media_id": job["media_snapshot"]["id"],
                    "input_revision": job["input_revision"], "input_sha256": job["input_sha256"],
                    "file_sha256": digest, "time_domain": DOMAIN, "window": window,
                    "subject_annotation_id": parameters["subject_annotation_id"],
                    "scenes": scenes, "keyframes": keyframes, "sampling": sampling,
                    "extractor_status": extractor_status, "evidence": evidence,
                    "semantic_listening": "not_performed", "source_unchanged": True,
                    "scope": "registered_playback_only; original_source_not_measured; candidate_not_applied"}
                self._validate_candidate(candidate, job)
                context.raise_if_cancelled()
                return candidate
            finally:
                # Evidence has already been hashed; the work directory remains
                # referenced by immutable artifact bindings under the store root.
                pass
        except InterruptedError as exc:
            raise WorkspaceError("CANCELLED", "Vision was cancelled") from exc
        finally:
            stop.set()
            heartbeat.join(timeout=6)
            with self._condition:
                self._active_jobs.discard(job["id"])
                self._condition.notify_all()

    def run_pending(self, *, batch_id: Optional[str] = None, worker_id: str = "vision-local") -> List[Dict[str, Any]]:
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

    def start(self) -> "VisionJobs":
        with self._condition:
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Cannot start a closing vision drainer")
            if self._drainer is not None and self._drainer.is_alive():
                return self

            def drain() -> None:
                try:
                    while not self.is_closing():
                        self.run_pending()
                        self._closing.wait(self.poll_seconds)
                except Exception as exc:
                    self._drainer_error = type(exc).__name__
                finally:
                    with self._condition:
                        self._condition.notify_all()

            self._drainer = threading.Thread(target=drain, name="vision-drainer", daemon=True)
            self._drainer.start()
        return self

    def close(self, *, timeout: float = 60) -> Dict[str, Any]:
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

    def _validate_candidate(self, value: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        parameters = _validate_job(job)
        fixed = {"kind": RESULT_KIND, "version": 1, "processor_version": VERSION,
            "job_id": job["id"], "task_id": job["task_id"], "media_id": job["media_snapshot"]["id"],
            "input_revision": job["input_revision"], "input_sha256": job["input_sha256"],
            "file_sha256": job["media_snapshot"]["playback"]["sha256"], "time_domain": DOMAIN,
            "semantic_listening": "not_performed", "source_unchanged": True,
            "scope": "registered_playback_only; original_source_not_measured; candidate_not_applied"}
        expected = set(fixed) | {"window", "subject_annotation_id", "scenes", "keyframes",
                                 "extractor_status", "evidence"}
        if (not isinstance(value, dict) or not expected <= set(value)
                or set(value) - expected - {"sampling"}
                or any(value[key] != item for key, item in fixed.items())
                or value["window"] != parameters["window"]
                or value["subject_annotation_id"] != parameters["subject_annotation_id"]):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision candidate identity is inconsistent")
        sampling = value.get("sampling")
        if sampling is not None and (not isinstance(sampling, dict)
                or set(sampling) != {"frame_map_sha256", "truncated", "planned", "returned",
                                     "coverage", "fingerprint"}
                or not isinstance(sampling["frame_map_sha256"], str)
                or not isinstance(sampling["fingerprint"], str)):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision sampling descriptor is inconsistent")
        if (not isinstance(value["scenes"], list) or len(value["scenes"]) > MAX_SCENES
                or not isinstance(value["keyframes"], list) or len(value["keyframes"]) > MAX_KEYFRAMES):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision candidate scene list is inconsistent")
        for scene in value["scenes"]:
            if (not isinstance(scene, dict) or set(scene) != {"start", "end", "duration"}
                    or any(not isinstance(scene[key], (int, float)) or not math.isfinite(scene[key]) for key in scene)):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision scene descriptor is invalid")
            if not 0 <= scene["start"] < scene["end"]:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision scene interval is invalid")
        prefix = f"tmp/vision/jobs/{job['id']}/{job['attempt']}-{job['lease_generation']}/"
        paths = set()
        for evidence in value["evidence"]:
            if (not isinstance(evidence, dict) or set(evidence) != {"relative_path", "sha256", "size_bytes"}
                    or not isinstance(evidence["relative_path"], str) or not evidence["relative_path"].startswith(prefix)):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision evidence descriptor is invalid")
            path = evidence["relative_path"]
            name = path[len(prefix):]
            parts = PurePosixPath(name).parts
            if (not name or any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts)
                    or path in paths or not isinstance(evidence["sha256"], str) or S.HASH_RE.fullmatch(evidence["sha256"]) is None
                    or not isinstance(evidence["size_bytes"], str) or not evidence["size_bytes"].isdigit()):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision evidence path or identity is invalid")
            paths.add(path)
        return value

    def _validate_snapshot(self, job: Dict[str, Any]) -> None:
        with self.store.transaction():
            if (self.store.get_media(job["media_snapshot"]["id"]) != job["media_snapshot"]
                    or self.store.task_digest(job["task_snapshot"]) != job["input_sha256"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision frozen input was altered")

    def job_status(self, job_id: str) -> Dict[str, Any]:
        """Read-only progress for one vision job, including a terminal error."""
        S.require_id(job_id)
        with self.store.transaction():
            job = self.queue.get_job(job_id)
            if job["operation"] != OPERATION:
                raise Conflict("UNSUPPORTED_PROCESSOR", "Job is not a vision job")
            attempt = (job.get("attempts") or [{}])[-1] if job.get("attempts") else {}
            return {"job_id": job_id, "task_id": job["task_id"], "state": job["state"],
                    "attempt": job.get("attempt"), "processor_version": job["processor_version"],
                    "input_revision": job["input_revision"], "input_sha256": job["input_sha256"],
                    "cancel_requested": job["state"] == "cancel_requested",
                    "error_code": job.get("error_code") or attempt.get("error_code"),
                    "has_candidate": job.get("commit") is not None}

    def result(self, job_id: str) -> Dict[str, Any]:
        S.require_id(job_id)
        try:
            return self._read_result(job_id)
        except (OSError, ValueError, KeyError, TypeError, RecursionError, SchemaInvalid) as exc:
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision candidate or commit is corrupt") from exc

    def _read_result(self, job_id: str) -> Dict[str, Any]:
        S.require_id(job_id)
        with self.store.transaction():
            job = self.queue.get_job(job_id)
            if job["operation"] != OPERATION:
                raise Conflict("UNSUPPORTED_PROCESSOR", "Job is not a vision job")
            if job["state"] != "succeeded" or job["commit"] is None:
                raise Conflict("VISION_NOT_READY", "Vision has no committed successful candidate")
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
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision commit or artifact binding is inconsistent")
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
                    raise WorkspaceError("RECOVERY_REQUIRED", "Vision candidate schema is corrupt") from exc
                return value
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision candidate is unavailable or corrupt") from exc

    def apply(self, job_id: str, *, command_id: str, expected_revision: int,
              expected_sha256: str, request_body: bytes) -> Dict[str, Any]:
        """Commit server-loaded vision candidates as model_candidate annotations.

        Every annotation carries provenance origin=model_candidate and an
        unreviewed review status; a completed local review is refused by the
        professional schema. Upgrading to human-reviewed ground truth is a
        separate explicit command, never an automatic side effect. The sampled
        keyframe PNGs are registered as immutable artifacts and pinned to the
        candidate annotation as evidence so the claim trail survives export/import.
        """
        from .professional_store import apply_vision_candidates

        with self.store.transaction():
            result = self.result(job_id)
            annotations = _candidate_annotations(result)
            _attach_candidate_evidence(self.store, result, annotations)
            arguments = dict(store=self.store, command_id=command_id, task_id=result["task_id"],
                expected_revision=expected_revision, expected_sha256=expected_sha256,
                candidates=annotations, source_job_id=result["job_id"], request_body=request_body)
            if self.store.conn.execute("SELECT 1 FROM command_receipts WHERE command_id=?", (command_id,)).fetchone():
                return self.store.apply_task_command(command_id=command_id, task_id=result["task_id"],
                    expected_revision=expected_revision, expected_sha256=expected_sha256,
                    operation="import_vision_candidates", payload={"candidates": annotations,
                    "source_job_id": result["job_id"]}, request_body=request_body)["response"]
            return apply_vision_candidates(**arguments)


def _find_ffmpeg() -> str:
    import shutil
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise WorkspaceError("DEPENDENCY_UNAVAILABLE", "ffmpeg is required for vision keyframe extraction")
    return executable


def _load_frame(path: Path) -> Any:
    from ..core.hasher import safe_load_image
    frame = safe_load_image(str(path))
    if frame is None:
        raise WorkspaceError("VISION_FRAME_FAILED", "Keyframe could not be decoded for analysis")
    return frame


def _aggregate_status(keyframes: List[Dict[str, Any]], extractor: str) -> str:
    if not keyframes:
        return "skipped"
    statuses = [frame[extractor].get("status", "failed") for frame in keyframes]
    if all(status == "success" for status in statuses):
        return "success"
    if all(status == "not_run" for status in statuses):
        return "not_run"
    if any(status == "failed" for status in statuses):
        return "failed"
    return "partial"


def _evidence_preview(source: Path) -> Path:
    """Create a small JPEG preview of one sampled keyframe.

    Full keyframe samples are hundreds of kilobytes each. A compact preview keeps
    the professional exchange document inside its bounded HTTP envelope and lets
    dozens of candidates travel together. It is a lossy, resampled preview of the
    exact sampled frame, never a caption or a synthesised image; the full sample
    remains in the job workspace.
    """
    from PIL import Image

    handle, name = tempfile.mkstemp(prefix="vision-sample-", suffix=".jpg")
    os.close(handle)
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((240, 240), Image.LANCZOS)
            image.save(name, "JPEG", quality=78, optimize=True)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return Path(name)


def _attach_candidate_evidence(store, candidate: Dict[str, Any],
                               annotations: List[Dict[str, Any]]) -> None:
    """Create and pin compact previews of sampled keyframes as evidence.

    The vision job workspace is temporary. A candidate that is applied to the
    task must own durable evidence, otherwise cleanup or export would leave a
    claim with an empty evidence trail. Each full sample is verified byte-for-byte
    against the job candidate, then a 240px JPEG preview is registered as an
    immutable artifact so the portable document stays inside its byte budget.
    """
    samples = {item["sha256"]: item for item in candidate.get("evidence") or []}
    frames = candidate.get("keyframes") or []
    for index, annotation in enumerate(annotations):
        scene_frames = [frame for frame in frames if frame.get("scene_index") == index]
        evidence_ids: List[str] = []
        frame_artifacts: Dict[str, str] = {}
        # One representative sampled keyframe per candidate keeps the portable
        # professional document inside its byte budget while still grounding the
        # keyframe claim in a real immutable PNG.
        for frame in scene_frames[:1]:
            digest = frame.get("evidence_sha256")
            descriptor = samples.get(digest)
            if descriptor is None:
                raise WorkspaceError("VISION_EVIDENCE_UNAVAILABLE",
                                     "Candidate keyframe has no registered sample")
            source = store.root / descriptor["relative_path"]
            _safe(source)
            if (not source.is_file() or source.stat().st_size != int(descriptor["size_bytes"])
                    or stream_sha256(source) != digest):
                raise WorkspaceError("VISION_EVIDENCE_UNAVAILABLE",
                                     "Candidate sample is missing or changed")
            preview = _evidence_preview(source)
            try:
                artifact = store._put_file(preview)
            finally:
                try:
                    preview.unlink()
                except OSError:
                    pass
            store.bind_artifact_reference(artifact, owner_kind="evidence",
                owner_id=annotation["id"], role="vision-sample-preview", owner_revision="0")
            evidence_ids.append(artifact["id"])
            frame_artifacts[str(frame["frame_index"])] = artifact["id"]
        annotation["evidence_ids"] = evidence_ids
        for claim in annotation["attributes"]["semantic"]["claims"]:
            if claim["method"]["id"] in {"keyframe-sampling-local", "shot-framing-local"}:
                frame_index = str(claim["method"]["parameters"]["frame_index"])
                if frame_index in frame_artifacts:
                    claim["evidence_ids"] = [frame_artifacts[frame_index]]


def _candidate_claims(candidate: Dict[str, Any], index: int, scene: Dict[str, Any],
                      start: Dict[str, str], end: Dict[str, str],
                      keyframe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build versioned, observation-only claims for one candidate.

    The claims describe what the local detector actually returned: a time
    interval and (when available) the sampled frame identity. They are
    deliberately not interpretations, and they stay inside a model_candidate
    annotation that a human must still review.
    """
    job_id = candidate["job_id"]
    window = {"start": {"seconds": start}, "end": {"seconds": end}}
    claims = [{
        "id": S.make_id("clm", f"{job_id}:{index}:scene-boundary"),
        "claim_type": "observable",
        "statement": (f"本地镜头边界算法测得区间 {float(scene['start']):.6f}–"
                      f"{float(scene['end']):.6f} 秒；边界仍待人工复核。"),
        "subject_ref": None,
        "time_window": window,
        "evidence_ids": [],
        "method": {"id": "shot-boundary-local", "version": str(candidate.get("processor_version") or "1"),
                   "parameters": {"detector": "local-scene-cut", "scene_index": index}},
        "model_run_id": None,
        "score": None,
        "score_kind": "none",
        "uncertainty_reason": "算法候选；未进行人工边界复核",
    }]
    if keyframe is not None:
        claims.append({
            "id": S.make_id("clm", f"{job_id}:{index}:keyframe"),
            "claim_type": "observable",
            "statement": (f"抽样关键帧 #{keyframe['frame_index']}（源 PTS "
                          f"{keyframe['source_pts']}）位于该镜头内。"),
            "subject_ref": None,
            "time_window": window,
            "evidence_ids": [],
            "method": {"id": "keyframe-sampling-local", "version": "1",
                       "parameters": {"frame_index": keyframe["frame_index"],
                                      "roles": list(keyframe.get("roles") or [])}},
            "model_run_id": None,
            "score": None,
            "score_kind": "none",
            "uncertainty_reason": "抽样帧只代表该时刻，不覆盖镜头内全部画面",
        })
        framing = keyframe.get("framing") or {}
        if (framing.get("status") == "success"
                and framing.get("framing") in {"远景", "全景", "中景", "近景", "特写"}):
            measurements = framing.get("measurements") or []
            ratio = measurements[0].get("height_ratio") if measurements else None
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:framing"),
                "claim_type": "measured",
                "statement": (f"按关键帧 #{keyframe['frame_index']} 的人脸框高度占比"
                              f"测得景别为{framing['framing']}（仅人脸框，需人物框复核）。"),
                "subject_ref": None,
                "time_window": window,
                "evidence_ids": [],
                "method": {"id": "shot-framing-local", "version": "1",
                           "parameters": {"framing": framing["framing"],
                                          "subject_kind": "face",
                                          "face_height_ratio": ratio,
                                          "frame_index": keyframe["frame_index"]}},
                "model_run_id": None,
                "score": float(framing.get("confidence") or 0.0),
                "score_kind": "rule_score",
                "uncertainty_reason": framing.get("uncertainty")
                or "未检测到主体不等于画面中不存在主体",
            })
    return {"version": SEMANTIC_VERSION, "claims": claims}


def _candidate_annotations(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map a vision candidate into model_candidate segment annotations.

    Only time boundaries and objective extractor evidence are carried over.
    A VLM caption is copied in as a text *draft* when present, but the
    annotation stays provenance=model_candidate and review_status=unreviewed:
    the human reviews and upgrades each candidate before it becomes
    authoritative. A caption is never treated as a fact.
    """
    annotations = []
    for index, scene in enumerate(candidate["scenes"]):
        start = _seconds_ratio(float(scene["start"]))
        end = _seconds_ratio(float(scene["end"]))
        keyframe = next((frame for frame in candidate["keyframes"]
                         if frame["scene_index"] == index), None)
        tags = ["vision-candidate", "auto-scene-cut"]
        text_draft = ""
        if keyframe is not None:
            face = keyframe.get("face", {})
            if face.get("status") == "success" and face.get("face_count"):
                tags.append("vision-face")
                if face.get("crowd_level"):
                    tags.append(face["crowd_level"])
            ocr = keyframe.get("ocr", {})
            if ocr.get("status") == "success" and ocr.get("ocr_readable"):
                tags.append("vision-text")
            clip = keyframe.get("clip", {})
            if clip.get("status") == "success":
                if clip.get("indoor_outdoor"):
                    tags.append(clip["indoor_outdoor"])
                if clip.get("screen_type") and clip["screen_type"] != "无屏":
                    tags.append(clip["screen_type"])
            vlm = keyframe.get("vlm", {})
            if vlm.get("status") == "success" and vlm.get("caption"):
                text_draft = vlm["caption"]
                tags.append("vision-caption")
            framing = keyframe.get("framing") or {}
            if (framing.get("status") == "success"
                    and framing.get("framing") in {"远景", "全景", "中景", "近景", "特写"}):
                tags.append("vision-framing-" + framing["framing"])
        annotation = {
            "id": S.make_id("ann", candidate["job_id"] + ":" + str(index)),
            "type": "segment",
            "label": f"{CANDIDATE_LABEL_PREFIX} {index + 1}",
            "text": text_draft,
            "tags": tags,
            "track": CANDIDATE_TRACK,
            "attributes": {"semantic": _candidate_claims(candidate, index, scene, start, end, keyframe)},
            "evidence_ids": [],
            "provenance": {"origin": "model_candidate", "actor_id": "vision-worker",
                           "job_id": candidate["job_id"]},
            "review_status": "unreviewed",
            "start": {"seconds": start},
            "end": {"seconds": end},
        }
        annotations.append(annotation)
    return annotations
