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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import MetricThresholds
from ..extractors.actions import (ACTION_VOCAB, detect_translation_candidates,
                                  merge_candidates, provider_coverage)
from ..extractors.camera_motion import analyse_motion
from ..extractors.expressive_cues import acoustic_claim, measure_acoustic_window
from ..extractors.scene_change import detect_content_changes, minimum_samples
from ..extractors.clip_engine import CLIPExtractor
from ..extractors.faces import FaceExtractor
from ..extractors.ocr import OCRExtractor
from ..extractors.scene import SceneDetector, SceneDetectionError
from ..extractors.shot_attributes import classify_framing
from ..extractors.shot_boundaries import (classify_scene_cut, detect_shot_boundaries,
                                          TRANSITION_KINDS)
from ..extractors.vlm import VLMExtractor
from ..temporal.media import stream_sha256
from . import professional_schema as PS
from . import schema as S
from . import semantic_schema as SS
from . import vision_sampling as VS
from .semantic_schema import SEMANTIC_VERSION
from .errors import Conflict, SchemaInvalid, WorkspaceError
from .http_media import resolve_media_object
from .queue import DurableQueue, MAX_RESULT_BYTES
from .slots import SlotClosed, SlotTimeout, get_prepare_slot
from .store import _safe

OPERATION = "vision-media"
VERSION = "1.1"
RESULT_KIND = "traceannotate.vision-candidate"
DOMAIN = "playback_seconds"
MAX_SCENES = 512
MAX_KEYFRAMES = 512
BOUNDARY_MAX_FRAMES = MAX_KEYFRAMES * 4
CONTENT_MIN_SAMPLES = minimum_samples()
CONTENT_UNIFORM = 24
CONTENT_UNIFORM_LONG = 96
CONTENT_TARGET_GAP = 32
CONTENT_MAX_FRAMES = CONTENT_UNIFORM_LONG + 2
MAX_CONTENT_FRAMES = MAX_SCENES * CONTENT_MAX_FRAMES
CONTENT_EVIDENCE_CONTRACT = "decoded-content-adaptive-v2"
MAX_FRAME_BYTES = 16 * 1024 * 1024
FRAME_QUALITY = 3
CANDIDATE_LABEL_PREFIX = "镜头候选"
CANDIDATE_TRACK = "镜头"
MOTION_LABELS = {"static", "pan_left", "pan_right", "tilt_up", "tilt_down",
                 "zoom_in", "zoom_out", "handheld_like"}
BOUNDARY_RADIUS = 8
TRANSITION_LABELS = {"cut", "fade_in", "fade_out", "dissolve", "flash",
                     "unknown_transition"}


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' positive' if positive else ''}")
    return result


def _content_uniform(span_frames: int) -> int:
    """Scale content sampling with shot length instead of a fixed 26-frame cap."""
    if span_frames <= 0:
        return CONTENT_UNIFORM
    return max(CONTENT_UNIFORM,
               min(CONTENT_UNIFORM_LONG, math.ceil(span_frames / CONTENT_TARGET_GAP)))


def _content_span(bounds: Sequence[int]) -> int:
    """A window clipped to one admitted frame still spans one frame, not zero."""
    if not bounds:
        return 0
    if len(bounds) == 1:
        return 1
    return max(1, bounds[1] - bounds[0] + 1)


def _content_overlap(start: Any, end: Any) -> bool:
    """A scene clipped out of the requested window has no frame to sample."""
    return S.rational_value(end) > S.rational_value(start)


def _content_clip(start: Any, end: Any, duration: Any) -> Tuple[Any, Any]:
    """Clamp a scene to the registered playback duration before sampling it."""
    if S.rational_value(start) > S.rational_value(duration):
        start = duration
    if S.rational_value(end) > S.rational_value(duration):
        end = duration
    return start, end


def _job_identity(*, content_evidence_contract: Optional[str] = CONTENT_EVIDENCE_CONTRACT) -> Dict[str, Any]:
    thresholds = MetricThresholds()
    config = {
        "scene_threshold": thresholds.scene_threshold,
        "scene_min_duration_s": thresholds.scene_min_duration_s,
        "scene_cap_seconds": thresholds.scene_cap_seconds,
        "face_min_confidence": thresholds.face_min_confidence,
        "face_min_height_ratio": thresholds.face_min_height_ratio,
        "ocr_min_height_ratio": thresholds.ocr_min_height_ratio,
        "ocr_min_confidence": thresholds.ocr_min_confidence,
        "clip_binary_confidence": thresholds.clip_binary_confidence,
        "clip_multiclass_confidence": thresholds.clip_multiclass_confidence,
        "clip_entity_mismatch_cutoff": thresholds.clip_entity_mismatch_cutoff,
        "sampling": {"max_frames": MAX_KEYFRAMES, "boundary_radius": BOUNDARY_RADIUS},
        "content_sampling": {"uniform": CONTENT_UNIFORM,
                             "uniform_long": CONTENT_UNIFORM_LONG,
                             "target_gap_frames": CONTENT_TARGET_GAP,
                             "max_frames": CONTENT_MAX_FRAMES,
                             "min_samples": CONTENT_MIN_SAMPLES},
    }
    if content_evidence_contract is not None:
        config["content_sampling"]["evidence_contract"] = content_evidence_contract
    return {"pipeline": "vision-assist-v1", "processor_version": VERSION,
            "config_sha256": S.sha256_text(config), "config": config,
            "model_identity": "declared_at_run_not_at_config_query",
            "capability_identity": "ffmpeg+local_extractors; weights_sha256_recorded_per_run"}


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
                                        "time_domain": DOMAIN, "job_identity": _job_identity()}))


def _validate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if job["operation"] != OPERATION or job["processor_version"] != VERSION:
        raise WorkspaceError("UNSUPPORTED_PROCESSOR", "No installed vision processor matches this frozen job")
    parameters = job["parameters"]
    if (not isinstance(parameters, dict)
            or set(parameters) != {"window", "subject_annotation_id", "time_domain", "job_identity"}
            or parameters["time_domain"] != DOMAIN):
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid frozen vision parameters")
    current = _parameters(parameters["window"], parameters["subject_annotation_id"])
    # Read/apply the exact pre-evidence (6e52961) contract without accepting
    # arbitrary frozen configurations. New submissions always use current.
    legacy = dict(current, job_identity=_job_identity(content_evidence_contract=None))
    if parameters != current and parameters != legacy:
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


def _subject_tracks(keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Associate sampled face boxes into short per-scene subject tracks.

    Only boxes actually measured by the face extractor are used. A track is a
    measurement of the sampled window, not an identity, and no track is invented
    for a frame with no box. IoU association is per scene so a cut never links
    two different subjects.
    """
    tracks: List[Dict[str, Any]] = []
    per_scene: Dict[int, List[Dict[str, Any]]] = {}
    for keyframe in sorted(keyframes, key=lambda item: (item["scene_index"], float(item["timestamp"]["n"]) / float(item["timestamp"]["d"]))):
        face = keyframe.get("face") or {}
        if face.get("status") != "success":
            continue
        faces = face.get("faces_detected") or []
        if not faces:
            continue
        largest = max(faces, key=lambda item: float(item["box"][2]) * float(item["box"][3]))
        x, y, box_width, box_height = [float(value) for value in largest["box"]]
        box = (x, y, x + box_width, y + box_height)
        seconds = float(keyframe["timestamp"]["n"]) / float(keyframe["timestamp"]["d"])
        scene_index = int(keyframe["scene_index"])
        candidates = per_scene.setdefault(scene_index, [])
        best = None
        best_iou = 0.0
        for track in candidates:
            previous = track["points"][-1]["box"]
            iou = _box_iou(previous, box)
            if iou > best_iou:
                best, best_iou = track, iou
        if best is None or best_iou < 0.2:
            best = {"subject_id": f"scene{scene_index}-subject{len(candidates) + 1}",
                    "scene_index": scene_index, "kind": "face", "points": []}
            candidates.append(best)
            tracks.append(best)
        best["points"].append({"box": [round(value, 2) for value in box],
                               "frame_index": keyframe["frame_index"], "seconds": seconds})
    return tracks


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_first = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    area_second = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _action_candidates(keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure walk/run from measured subject-box tracks; leave other classes not_run."""
    coverage = provider_coverage("subject-translation-v1")
    tracks = _subject_tracks(keyframes)
    candidates: List[Dict[str, Any]] = []
    track_results = []
    for track in tracks:
        points = track["points"]
        result = detect_translation_candidates(points, subject_id=track["subject_id"], min_samples=3)
        track_results.append({"subject_id": track["subject_id"],
                              "scene_index": track["scene_index"],
                              "kind": track["kind"], "samples": len(points),
                              "status": result.get("status"),
                              "reason": result.get("reason"),
                              "metrics": result.get("metrics"),
                              "candidate_count": len(result.get("candidates") or [])})
        for item in result.get("candidates") or []:
            item["provider"] = "subject-translation-v1"
            item["scene_index"] = track["scene_index"]
            candidates.append(item)
    for track in tracks:
        for point in track["points"]:
            point.pop("box", None)
    merged = merge_candidates(candidates)
    if not tracks:
        status, reason = "not_run", "no_subject_box_track_in_sampled_frames"
    elif not any(item["samples"] >= 3 for item in track_results):
        status, reason = "not_run", "fewer_than_three_box_samples_per_subject"
    else:
        status, reason = "success", None
    return {
        "status": status, "provider": "subject-translation-v1",
        "supported": list(coverage.get("supported") or []),
        "unsupported": list(coverage.get("unsupported") or []),
        "reason": reason,
        "tracks": track_results,
        "candidates": merged,
        "measurement_basis": "主体框中心位移/主体高度/时间；仅走、跑两类；拿起/放下/打开/关闭/递交无本地 provider",
        "uncertainty": "仅由抽样帧主体框位移推出，动作类别与主体身份需人工确认",
    }


def _measure_audio(source: Path, ffmpeg: str, work: Path) -> Dict[str, Any]:
    """Decode the first real audio stream and measure RMS/f0/active intervals."""
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", str(source),
               "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
               "-f", "f32le", "pipe:1"]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "not_run", "reason": f"audio_decode_failed:{type(exc).__name__}",
                "measurement": None}
    if completed.returncode != 0 or not completed.stdout:
        return {"status": "not_run", "reason": "no_decodable_audio_track",
                "detail": (completed.stderr or b"").decode("utf-8", "replace")[-300:],
                "measurement": None}
    import array
    import wave

    samples = array.array("f")
    samples.frombytes(completed.stdout[: len(completed.stdout) - (len(completed.stdout) % 4)])
    if not samples:
        return {"status": "not_run", "reason": "audio_stream_empty", "measurement": None}
    sample_rate = 16000
    measurement = measure_acoustic_window(samples, sample_rate, origin_seconds=0.0)
    if measurement.get("status") != "success":
        return {"status": "not_run", "reason": measurement.get("reason"),
                "measurement": measurement}
    audio_dir = work / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "acoustic.wav"
    pcm = array.array("h")
    for value in samples:
        scaled = int(round(float(value) * 32767.0))
        pcm.append(max(-32768, min(32767, scaled)))
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    audio_path = audio_dir / "acoustic.m4a"
    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(wav_path),
         "-c:a", "aac", "-b:a", "32k", "-ac", "1", str(audio_path)],
        capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not audio_path.is_file() or audio_path.stat().st_size == 0:
        return {"status": "not_run",
                "reason": "acoustic_evidence_encode_failed:" +
                          (completed.stderr or "empty output").strip()[:200],
                "measurement": measurement}
    wav_path.unlink()
    return {"status": "success", "reason": None, "audio_relative_path": "audio/acoustic.m4a",
            "audio_sha256": stream_sha256(audio_path),
            "stream_selection": "0:a:0 (first audio stream; video stream ignored)",
            "sample_rate": sample_rate, "samples": len(samples),
            "measurement": measurement}


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
        if parameters["job_identity"] != _job_identity():
            raise WorkspaceError("VISION_RESUBMIT_REQUIRED",
                                 "Legacy vision configuration is read/apply-only; resubmit a new vision job")
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
                scene_frames: Dict[int, List[Dict[str, Any]]] = {}
                motion_by_scene: Dict[int, Dict[str, Any]] = {}

                def flush_scene(index: int) -> None:
                    samples = scene_frames.pop(index, None)
                    if not samples or len(samples) < 2:
                        return
                    motion_by_scene[index] = analyse_motion(samples)
                    for keyframe in keyframes:
                        if keyframe["scene_index"] == index:
                            keyframe["motion"] = motion_by_scene[index]

                active_scene: Optional[int] = None
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
                    for item in sorted(decoded["decoded"], key=lambda entry: int(entry["frame_index"])):
                        if cancelled():
                            raise WorkspaceError("CANCELLED", "Vision cancelled while analysing samples")
                        seconds = S.rational_value(item["seconds"])
                        scene_index = next((index for index, scene in enumerate(scenes)
                                            if scene["start"] <= seconds < scene["end"]), None)
                        if scene_index is None:
                            continue
                        if active_scene is not None and scene_index != active_scene:
                            flush_scene(active_scene)
                        active_scene = scene_index
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
                        scene_frames.setdefault(scene_index, []).append({
                            "frame_index": item["frame_index"], "seconds": seconds,
                            "frame": frame})
                else:
                    sampling = None
                for index in list(scene_frames):
                    flush_scene(index)
                content_sampling: Dict[str, Any] = {
                    "status": "not_run", "reason": "scene_extraction_not_run"}
                content_frames: List[Dict[str, Any]] = []
                if sampling is not None:
                    try:
                        scene_results = []
                        for scene_index, scene in enumerate(scenes):
                            content_start = S.rational_value(_seconds_ratio(float(scene["start"])))
                            content_end = S.rational_value(_seconds_ratio(float(scene["end"])))
                            if window is not None:
                                content_start = max(content_start, S.rational_value(window["start"]))
                                content_end = min(content_end, S.rational_value(window["end"]))
                            content_start, content_end = _content_clip(
                                content_start, content_end, job["media_snapshot"]["duration"])
                            content_window = {"start": S.rational(content_start), "end": S.rational(content_end)}
                            if not _content_overlap(content_start, content_end):
                                scene_results.append({
                                    "scene_index": scene_index,
                                    "window": content_window,
                                    "frame_map_sha256": None,
                                    "fingerprint": None,
                                    "params": None,
                                    "planned": 0,
                                    "returned": 0,
                                    "truncated": False,
                                    "span_frames": 0,
                                    "uniform": 0,
                                    "status": "empty_scene",
                                    "provided_samples": 0,
                                    "required_samples": CONTENT_MIN_SAMPLES,
                                    "message": "场景与请求窗口无重叠帧，未采样",
                                    "max_gap_frames": 0,
                                    "changes": 0,
                                    "filtered": 0,
                                })
                                continue
                            bounds = VS.boundary_indices(
                                self.store, job["media_snapshot"]["id"],
                                [content_start, content_end])
                            span_frames = _content_span(bounds)
                            uniform = _content_uniform(span_frames)
                            try:
                                plan_c = VS.plan_samples(
                                    self.store, job["media_snapshot"]["id"],
                                    window=content_window,
                                    uniform=uniform, include_edges=True,
                                    budget=VS.SamplingBudget(max_frames=uniform + 2,
                                                             max_uniform=max(uniform, VS.MAX_UNIFORM),
                                                             boundary_radius=0))
                            except SchemaInvalid as exc:
                                scene_results.append({
                                    "scene_index": scene_index,
                                    "window": content_window,
                                    "frame_map_sha256": None,
                                    "fingerprint": None,
                                    "params": None,
                                    "planned": 0,
                                    "returned": 0,
                                    "truncated": False,
                                    "span_frames": span_frames,
                                    "uniform": uniform,
                                    "status": "empty_scene",
                                    "provided_samples": 0,
                                    "required_samples": CONTENT_MIN_SAMPLES,
                                    "message": str(exc),
                                    "max_gap_frames": 0,
                                    "changes": 0,
                                    "filtered": 0,
                                })
                                continue
                            decoded_c = VS.decode_samples(
                                self.store, job["media_snapshot"]["id"], plan_c,
                                work / "content" / str(scene_index),
                                cancelled=cancelled, ffmpeg=ffmpeg)
                            samples = []
                            for item in decoded_c["decoded"]:
                                if cancelled():
                                    raise WorkspaceError(
                                        "CANCELLED",
                                        "Vision cancelled while analysing content changes")
                                samples.append({"frame_index": item["frame_index"],
                                                "seconds": float(S.rational_value(item["seconds"])),
                                                "frame": _load_frame(Path(item["path"]))})
                                content_frames.append({
                                    "scene_index": scene_index,
                                    "frame_index": item["frame_index"],
                                    "seconds": item["seconds"],
                                    "source_pts": item["source_pts"],
                                    "proxy_pts": item["proxy_pts"],
                                    "evidence_sha256": item["sha256"],
                                })
                            analysis = detect_content_changes(samples)
                            indices = [int(item["frame_index"]) for item in decoded_c["decoded"]]
                            max_gap = max((right - left for left, right
                                           in zip(indices, indices[1:])), default=0)
                            detail = {
                                "status": analysis.get("status", "success"),
                                "provided_samples": analysis.get("provided_samples", len(samples)),
                                "required_samples": analysis.get("required_samples",
                                                                  CONTENT_MIN_SAMPLES),
                                "message": analysis.get("message"),
                                "max_gap_frames": max_gap,
                            }
                            for keyframe in keyframes:
                                if keyframe["scene_index"] == scene_index:
                                    keyframe["content_sampling"] = detail
                                    if "content_changes" not in keyframe:
                                        keyframe["content_changes"] = analysis["changes"]
                                        keyframe["content_filtered"] = analysis["filtered"]
                            scene_results.append({
                                "scene_index": scene_index,
                                "window": content_window,
                                "frame_map_sha256": plan_c["frame_map_sha256"],
                                "fingerprint": VS.samples_fingerprint(plan_c),
                                "params": analysis["params"],
                                "planned": plan_c["planned"],
                                "returned": len(decoded_c["decoded"]),
                                "truncated": plan_c["truncated"],
                                "span_frames": span_frames,
                                "uniform": uniform,
                                "status": detail["status"],
                                "provided_samples": detail["provided_samples"],
                                "required_samples": detail["required_samples"],
                                "message": detail["message"],
                                "max_gap_frames": max_gap,
                                "changes": len(analysis["changes"]),
                                "filtered": len(analysis["filtered"]),
                            })
                        insufficient = [item for item in scene_results
                                        if item["status"] == "insufficient_samples"]
                        content_sampling = {
                            "status": "insufficient_samples" if insufficient else "success",
                            "scenes": len(scenes),
                            "planned": sum(item["planned"] for item in scene_results),
                            "returned": sum(item["returned"] for item in scene_results),
                            "truncated": any(item["truncated"] for item in scene_results),
                            "required_samples": CONTENT_MIN_SAMPLES,
                            "uniform": CONTENT_UNIFORM,
                            "uniform_long": CONTENT_UNIFORM_LONG,
                            "target_gap_frames": CONTENT_TARGET_GAP,
                            "insufficient_scenes": len(insufficient),
                            "max_gap_frames": max((item["max_gap_frames"]
                                                     for item in scene_results), default=0),
                            "changes": sum(item["changes"] for item in scene_results),
                            "filtered": sum(item["filtered"] for item in scene_results),
                            "scene_results": scene_results,
                        }
                    except Exception as exc:
                        content_sampling = {"status": "failed",
                                            "error": type(exc).__name__,
                                            "message": str(exc)[:240]}
                actions_sampling = _action_candidates(keyframes)
                try:
                    acoustics_sampling = _measure_audio(Path(source), ffmpeg, work)
                except Exception as exc:
                    acoustics_sampling = {"status": "failed", "reason": type(exc).__name__,
                                          "message": str(exc)[:240], "measurement": None}
                boundary_frames: List[Dict[str, Any]] = []
                boundary_sampling: Dict[str, Any] = {
                    "status": "not_run",
                    "reason": "no_scene_boundary_in_window",
                }
                if scene_result["status"] == "success" and len(scenes) > 1:
                    try:
                        targets: List[tuple[int, int]] = []
                        for scene_index in range(1, len(scenes)):
                            index = VS.boundary_indices(
                                self.store, job["media_snapshot"]["id"],
                                [scenes[scene_index]["start"]])[0]
                            targets.append((index, scene_index))
                        plan_b = VS.plan_samples(
                            self.store, job["media_snapshot"]["id"], uniform=0,
                            boundary_frames=[item[0] for item in targets],
                            include_edges=False,
                            budget=VS.SamplingBudget(max_frames=BOUNDARY_MAX_FRAMES,
                                                     boundary_radius=BOUNDARY_RADIUS))
                        decoded_b = VS.decode_samples(
                            self.store, job["media_snapshot"]["id"], plan_b,
                            work / "boundaries", cancelled=cancelled, ffmpeg=ffmpeg)
                        boundary_sampling = {
                            "status": "success",
                            "frame_map_sha256": plan_b["frame_map_sha256"],
                            "truncated": plan_b["truncated"],
                            "planned": plan_b["planned"],
                            "returned": plan_b["returned"],
                            "radius": BOUNDARY_RADIUS,
                            "fingerprint": VS.samples_fingerprint(plan_b),
                        }
                        boundary_samples: Dict[int, List[Dict[str, Any]]] = {}
                        detected: Dict[tuple, tuple[int, Dict[str, Any]]] = {}
                        for item in decoded_b["decoded"]:
                            if cancelled():
                                raise WorkspaceError("CANCELLED",
                                                     "Vision cancelled while analysing boundaries")
                            frame_index = int(item["frame_index"])
                            scene_indices = [scene for target, scene in targets
                                             if abs(frame_index - target) <= BOUNDARY_RADIUS]
                            if not scene_indices:
                                continue
                            seconds = float(S.rational_value(item["seconds"]))
                            boundary_frames.append({
                                "scene_index": scene_indices[0],
                                "scene_indices": scene_indices,
                                "frame_index": item["frame_index"],
                                "seconds": seconds,
                                "evidence_sha256": item["sha256"],
                            })
                            sample = {"frame_index": item["frame_index"],
                                      "seconds": seconds,
                                      "source_pts": item["source_pts"],
                                      "proxy_pts": item["proxy_pts"],
                                      "frame": _load_frame(Path(item["path"]))}
                            for scene_index in scene_indices:
                                boundary_samples.setdefault(scene_index, []).append(dict(sample))
                        scene_events: Dict[int, List[Dict[str, Any]]] = {}
                        for scene_index, samples in boundary_samples.items():
                            if len(samples) < 2:
                                continue
                            result = detect_shot_boundaries(samples)
                            for event in result["events"]:
                                key = (event.get("kind"), event.get("anchor_frame"),
                                       (event.get("interval") or {}).get("start_frame"),
                                       (event.get("interval") or {}).get("end_frame"))
                                detected.setdefault(key, (scene_index, event))
                            scene_events[scene_index] = list(result["events"])
                        target_frames = {scene: index for index, scene in targets}
                        final_events: Dict[int, List[Dict[str, Any]]] = {}
                        for scene_index, events in scene_events.items():
                            target = target_frames.get(scene_index)
                            has_cut = any(
                                event.get("kind") == "cut"
                                and target is not None
                                and abs(int(event.get("anchor_frame") or -1) - target) <= 2
                                for event in events)
                            if has_cut or target is None:
                                final_events[scene_index] = events
                                continue
                            confirmation = classify_scene_cut(
                                boundary_samples.get(scene_index, []),
                                candidate_frame=target)
                            if confirmation is None:
                                final_events[scene_index] = events
                                continue
                            final_events[scene_index] = [confirmation]
                        for scene_index, events in final_events.items():
                            first = next((frame for frame in keyframes
                                          if frame["scene_index"] == scene_index), None)
                            if first is None:
                                continue
                            for event in events:
                                first.setdefault("boundary_events", []).append(event)
                                detected[(scene_index, event.get("kind"),
                                          event.get("anchor_frame"))] = (scene_index, event)
                        boundary_sampling["scene_events"] = {
                            str(scene): len(final_events[scene]) for scene in sorted(final_events)}
                        boundary_sampling["scenes_with_events"] = sorted({scene for scene, _ in detected.values()})
                        boundary_sampling["event_count"] = len(detected)
                        boundary_sampling["overlap_mapping"] = "each decoded boundary frame is reused by every scene window that covers it"
                    except Exception as exc:
                        boundary_sampling = {"status": "failed",
                                             "error": type(exc).__name__,
                                             "message": str(exc)[:240]}
                        boundary_frames = []
                extractor_status = {
                    "scene": scene_result["status"],
                    "face": _aggregate_status(keyframes, "face"),
                    "ocr": _aggregate_status(keyframes, "ocr"),
                    "clip": _aggregate_status(keyframes, "clip"),
                    "vlm": _aggregate_status(keyframes, "vlm"),
                }
                motion_status = "success" if any(
                    frame.get("motion", {}).get("status") == "success" for frame in keyframes) else "not_run"
                module_status = dict(extractor_status)
                module_status.update({
                    "boundary": boundary_sampling.get("status", "not_run"),
                    "framing": extractor_status["face"],
                    "motion": motion_status,
                    "transition": ("success" if any(frame.get("boundary_events") for frame in keyframes)
                                   else boundary_sampling.get("status", "not_run")),
                    "actions": actions_sampling.get("status", "not_run"),
                    "content": content_sampling.get("status", "not_run"),
                    "expressive": "not_run",
                    "acoustics": acoustics_sampling.get("status", "not_run"),
                })
                module_status["actions_reason"] = actions_sampling.get("reason")
                module_status["actions_supported"] = actions_sampling.get("supported")
                module_status["actions_unsupported"] = actions_sampling.get("unsupported")
                module_status["expressive_reason"] = "no_landmark_provider_in_this_build"
                module_status["content_reason"] = (content_sampling.get("message")
                                                   or content_sampling.get("error"))
                module_status["acoustics_reason"] = acoustics_sampling.get("reason")
                if self._extractor_error is not None:
                    extractor_status["extractor_error"] = self._extractor_error
                model_runs = _build_model_runs(
                    job, extractors, keyframes, scene_result,
                    boundary_sampling, boundary_frames, module_status,
                    digest, window, actions_sampling, acoustics_sampling,
                    content_sampling, content_frames)
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
                    "boundary_frames": boundary_frames, "boundary_sampling": boundary_sampling,
                    "content_sampling": content_sampling,
                    "content_frames": content_frames,
                    "actions_sampling": _jsonable(actions_sampling),
                    "acoustics_sampling": _jsonable(acoustics_sampling),
                    "extractor_status": extractor_status, "module_status": module_status,
                    "model_runs": model_runs,
                    "job_identity": parameters["job_identity"], "evidence": evidence,
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
                                 "extractor_status", "module_status", "job_identity",
                                 "model_runs", "evidence", "boundary_frames", "boundary_sampling"}
        expected = expected | {"content_sampling", "actions_sampling", "acoustics_sampling"}
        if (not isinstance(value, dict) or not expected <= set(value)
                or set(value) - expected - {"sampling", "content_frames"}
                or any(value[key] != item for key, item in fixed.items())
                or value["window"] != parameters["window"]
                or value["job_identity"] != parameters["job_identity"]
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
        if (not isinstance(value["boundary_frames"], list)
                or len(value["boundary_frames"]) > BOUNDARY_MAX_FRAMES):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision boundary frame list is inconsistent")
        for frame in value["boundary_frames"]:
            base = {"scene_index", "frame_index", "seconds", "evidence_sha256"}
            if (not isinstance(frame, dict)
                    or set(frame) not in (base, base | {"scene_indices"})
                    or type(frame["scene_index"]) is not int
                    or not isinstance(frame["frame_index"], str)
                    or not frame["frame_index"].isdigit()
                    or not isinstance(frame["seconds"], (int, float))
                    or not math.isfinite(frame["seconds"])
                    or not isinstance(frame["evidence_sha256"], str)
                    or S.HASH_RE.fullmatch(frame["evidence_sha256"]) is None
                    or ("scene_indices" in frame and
                        (not isinstance(frame["scene_indices"], list)
                         or any(type(item) is not int or item < 0 or item >= len(value["scenes"])
                                for item in frame["scene_indices"])
                         or not frame["scene_indices"]))):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision boundary frame descriptor is invalid")
        if value["boundary_sampling"] is not None and not isinstance(value["boundary_sampling"], dict):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision boundary sampling descriptor is inconsistent")
        # Optional for immutable candidates produced before content provenance
        # was recorded. New candidates always carry the decoded frame identity.
        content_frames = value.get("content_frames", [])
        requires_content = (parameters["job_identity"]["config"]["content_sampling"].get("evidence_contract")
                            == CONTENT_EVIDENCE_CONTRACT)
        if requires_content and "content_frames" not in value:
            raise WorkspaceError("RECOVERY_REQUIRED", "Current vision contract requires decoded content frames")
        if not isinstance(content_frames, list) or len(content_frames) > MAX_CONTENT_FRAMES:
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision content frame list is inconsistent")
        content_keys = set()
        for frame in content_frames:
            if (not isinstance(frame, dict)
                    or set(frame) != {"scene_index", "frame_index", "seconds", "source_pts",
                                      "proxy_pts", "evidence_sha256"}
                    or type(frame["scene_index"]) is not int
                    or not 0 <= frame["scene_index"] < len(value["scenes"])
                    or not isinstance(frame["frame_index"], str) or not frame["frame_index"].isdigit()
                    or not isinstance(frame["evidence_sha256"], str)
                    or S.HASH_RE.fullmatch(frame["evidence_sha256"]) is None):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision content frame descriptor is invalid")
            try:
                if S.rational(frame["seconds"]) != frame["seconds"] or S.rational_value(frame["seconds"]) < 0:
                    raise ValueError("invalid frame time")
                seconds = S.rational_value(frame["seconds"])
                scene = value["scenes"][frame["scene_index"]]
                if not (S.rational_value(_seconds_ratio(scene["start"])) <= seconds
                        < S.rational_value(_seconds_ratio(scene["end"]))):
                    raise ValueError("content sample outside its scene")
                if value["window"] is not None and not (S.rational_value(value["window"]["start"]) <= seconds
                                                        < S.rational_value(value["window"]["end"])):
                    raise ValueError("content sample outside requested window")
            except (TypeError, ValueError, WorkspaceError) as exc:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision content frame time is invalid") from exc
            identity = (frame["scene_index"], frame["frame_index"])
            if identity in content_keys:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision content frame identities are duplicated")
            content_keys.add(identity)
        model_runs = value["model_runs"]
        if not isinstance(model_runs, list) or len(model_runs) > SS.MAX_MODEL_RUNS:
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision model-run list is inconsistent")
        run_ids = set()
        for run in model_runs:
            checked = SS.validate_model_run(run)
            if checked["id"] in run_ids:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision model-run identities are duplicated")
            run_ids.add(checked["id"])
        content_run = next((run for run in model_runs
                            if run["id"] == _module_run_id(job["id"], "content")), None)
        if requires_content and content_run is None:
            raise WorkspaceError("RECOVERY_REQUIRED", "Current vision contract requires a content model run")
        if content_run is not None:
            if ("content_frames" not in value or not isinstance(value["content_sampling"], dict)
                    or content_run["input_sha256"] != S.sha256_text(_content_input(
                        value["file_sha256"], value["window"], value["content_sampling"], content_frames))):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision content input fingerprint is inconsistent")
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
        evidence_hashes = {item["sha256"] for item in value["evidence"]}
        if any(frame["evidence_sha256"] not in evidence_hashes for frame in content_frames):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision content frame has no registered sample")
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
                    "job_identity": job["parameters"].get("job_identity"),
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
                candidates=annotations, source_job_id=result["job_id"],
                model_runs=result.get("model_runs") or [], request_body=request_body)
            if self.store.conn.execute("SELECT 1 FROM command_receipts WHERE command_id=?", (command_id,)).fetchone():
                return self.store.apply_task_command(command_id=command_id, task_id=result["task_id"],
                    expected_revision=expected_revision, expected_sha256=expected_sha256,
                    operation="import_vision_candidates", payload={"candidates": annotations,
                    "source_job_id": result["job_id"],
                    "model_runs": result.get("model_runs") or []}, request_body=request_body)["response"]
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


def _jsonable(value: Any) -> Any:
    """Convert extractor output to strict JSON without silently stringifying.

    Some detectors expose numpy scalars or arrays. They are converted with
    ``item``/``tolist``; an unknown object is a schema error rather than a
    model-run digest over a Python repr.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        if abs(value) > S.MAX_SAFE_INTEGER:
            return str(value)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaInvalid("SCHEMA_INVALID", "Model output contains a non-finite number")
        return float(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise SchemaInvalid("SCHEMA_INVALID", "Model output is not JSON-serializable")


def _module_run_id(job_id: str, module: str) -> str:
    return S.make_id("run", f"{job_id}:{module}")


def _package_version(distribution: str) -> str:
    try:
        from importlib.metadata import version
        return version(distribution)
    except Exception:  # noqa: BLE001 - missing metadata is reported, never guessed
        return "unavailable"


def _weights_digest(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    return stream_sha256(Path(path))


def _first_module_reason(keyframes: List[Dict[str, Any]], module: str) -> Optional[str]:
    for frame in keyframes:
        result = frame.get(module) or {}
        if result.get("status") != "success":
            reason = result.get("reason") or result.get("error")
            if reason:
                return str(reason)[:500]
            if result.get("status") == "not_run":
                return f"{module}_runtime_unavailable"
    return None


def _run_status(value: str) -> str:
    if value == "success":
        return "success"
    if value == "not_run" or value == "skipped":
        return "not_run"
    return "failed"


def _model_run_record(job_id: str, module: str, *, provider: str, model_name: str,
                      model_version: str, weights_sha256: Optional[str], license: str,
                      preprocessing: Dict[str, Any], parameters: Dict[str, Any],
                      input_value: Any, output_value: Any, status: str,
                      failure_reason: Optional[str], code_version: str,
                      score_kind: str = "none") -> Dict[str, Any]:
    normalized = _run_status(status)
    reason = None if normalized == "success" else (str(failure_reason)[:500]
                                                  if failure_reason else "module did not report success")
    output_sha = None
    if output_value is not None and normalized != "not_run":
        output_sha = S.sha256_text(_jsonable(output_value))
    if normalized == "success" and output_sha is None:
        raise SchemaInvalid("SCHEMA_INVALID", "A successful model run requires an output digest")
    return {
        "id": _module_run_id(job_id, module),
        "provider": str(provider)[:120],
        "model_name": str(model_name)[:200],
        "model_version": str(model_version)[:120],
        "weights_sha256": weights_sha256,
        "license": str(license)[:200],
        "preprocessing": _jsonable(preprocessing),
        "parameters": _jsonable(parameters),
        "input_sha256": S.sha256_text(_jsonable(input_value)),
        "output_sha256": output_sha,
        "code_version": str(code_version)[:120],
        "status": normalized,
        "failure_reason": reason,
        "score_kind": score_kind,
    }


def _keyframe_input(keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"basis": "decoded_keyframe_samples",
            "frames": [{"frame_index": frame["frame_index"],
                        "evidence_sha256": frame["evidence_sha256"]}
                       for frame in keyframes]}


def _frame_output(keyframes: List[Dict[str, Any]], module: str) -> Dict[str, Any]:
    return {"module": module,
            "frames": [{"frame_index": frame["frame_index"],
                        "scene_index": frame["scene_index"],
                        "result": _jsonable(frame.get(module))}
                       for frame in keyframes]} 


def _content_input(digest: str, window: Optional[Dict[str, Any]],
                   sampling: Dict[str, Any], frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fingerprint actual decoded inputs, independently of general keyframes."""
    return {"basis": "decoded_content_samples", "file_sha256": digest,
            "window": window, "time_domain": DOMAIN,
            "scenes": [{key: scene.get(key) for key in
                        ("scene_index", "window", "frame_map_sha256", "fingerprint", "params")}
                       for scene in sampling.get("scene_results", [])],
            "frames": frames}


def _build_model_runs(job: Dict[str, Any], extractors: Dict[str, Any],
                      keyframes: List[Dict[str, Any]], scene_result: Dict[str, Any],
                      boundary_sampling: Optional[Dict[str, Any]],
                      boundary_frames: List[Dict[str, Any]],
                      module_status: Dict[str, Any], digest: str,
                      window: Optional[Dict[str, Any]],
                      actions_sampling: Optional[Dict[str, Any]] = None,
                      acoustics_sampling: Optional[Dict[str, Any]] = None,
                      content_sampling: Optional[Dict[str, Any]] = None,
                      content_frames: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Record one real run per executed module, including honest non-success.

    Weights are hashed only when a local file exists. A failed or not-run
    adapter still gets a record so the export can distinguish "did not run"
    from "ran and produced this". Rule-based detectors are recorded too: their
    identity is the frozen threshold set, not a fabricated model weight.
    """
    job_id = job["id"]
    thresholds = MetricThresholds()
    frame_input = _keyframe_input(keyframes)
    runs: List[Dict[str, Any]] = []

    def extractor_identity(module: str, fallback: str) -> Dict[str, Any]:
        instance = extractors.get(module)
        if instance is None:
            return {"provider": "unavailable", "model_name": fallback,
                    "model_version": "unavailable", "weights_path": None,
                    "license": "not declared"}
        return instance.code_identity

    face_identity = extractor_identity("face", "YuNet")
    runs.append(_model_run_record(
        job_id, "face", provider=face_identity.get("provider", "opencv"),
        model_name=face_identity.get("model_name", "YuNet"),
        model_version=face_identity.get("model_version", "unknown"),
        weights_sha256=_weights_digest(face_identity.get("weights_path")),
        license=face_identity.get("license", "not declared"),
        preprocessing={"input": "BGR frame as decoded",
                       "detector_kind": face_identity.get("detector_kind"),
                       "detector_input": "320x320" if face_identity.get("detector_kind") == "yunet" else "native"},
        parameters={"face_min_confidence": thresholds.face_min_confidence},
        input_value=frame_input, output_value=_frame_output(keyframes, "face"),
        status=module_status.get("face", "not_run"),
        failure_reason=_first_module_reason(keyframes, "face")
        or ("no sampled keyframes" if not keyframes else None),
        code_version=f"vision_jobs={VERSION}; opencv={_package_version('opencv-python')}",
        score_kind="confidence"))

    ocr_identity = extractor_identity("ocr", "RapidOCR")
    runs.append(_model_run_record(
        job_id, "ocr", provider=ocr_identity.get("provider", "RapidOCR"),
        model_name=ocr_identity.get("model_name", "RapidOCR default det+cls+rec"),
        model_version=ocr_identity.get("model_version", "unavailable"),
        weights_sha256=_weights_digest(ocr_identity.get("weights_path")),
        license=ocr_identity.get("license", "Apache-2.0"),
        preprocessing={"input": "BGR frame as decoded", "orientation": "as-shot"},
        parameters={"ocr_min_confidence": thresholds.ocr_min_confidence,
                    "ocr_min_height_ratio": thresholds.ocr_min_height_ratio},
        input_value=frame_input, output_value=_frame_output(keyframes, "ocr"),
        status=module_status.get("ocr", "not_run"),
        failure_reason=_first_module_reason(keyframes, "ocr")
        or ("no sampled keyframes" if not keyframes else None),
        code_version=f"vision_jobs={VERSION}; rapidocr={ocr_identity.get('model_version', 'unavailable')}",
        score_kind="confidence"))

    clip_identity = extractor_identity("clip", "ViT-B-32")
    runs.append(_model_run_record(
        job_id, "clip", provider=clip_identity.get("provider", "open_clip"),
        model_name=clip_identity.get("model_name", "ViT-B-32"),
        model_version=clip_identity.get("pretrained", "openai"),
        weights_sha256=_weights_digest(clip_identity.get("weights_path")),
        license=clip_identity.get("license", "MIT (open_clip); pretrained weights terms not declared locally"),
        preprocessing={"input": "BGR frame converted to RGB",
                       "transform": "open_clip pretrained eval transform",
                       "offline": clip_identity.get("offline"),
                       "downloads_allowed": clip_identity.get("downloads_allowed")},
        parameters={"clip_binary_confidence": thresholds.clip_binary_confidence,
                    "clip_multiclass_confidence": thresholds.clip_multiclass_confidence,
                    "clip_entity_mismatch_cutoff": thresholds.clip_entity_mismatch_cutoff,
                    "force_quick_gelu": clip_identity.get("force_quick_gelu")},
        input_value=frame_input, output_value=_frame_output(keyframes, "clip"),
        status=module_status.get("clip", "not_run"),
        failure_reason=_first_module_reason(keyframes, "clip")
        or ("no sampled keyframes" if not keyframes else None),
        code_version=f"vision_jobs={VERSION}; open_clip_torch={_package_version('open_clip_torch')}",
        score_kind="softmax"))

    vlm_identity = extractor_identity("vlm", "microsoft/Florence-2-base")
    runs.append(_model_run_record(
        job_id, "vlm", provider=vlm_identity.get("provider", "transformers"),
        model_name=vlm_identity.get("model_path", "microsoft/Florence-2-base"),
        model_version=vlm_identity.get("revision", "unpinned"),
        weights_sha256=_weights_digest(vlm_identity.get("weights_path")),
        license=vlm_identity.get("license", "MIT (Florence-2); local weight terms not declared"),
        preprocessing={"input": "BGR frame converted to RGB PIL image",
                       "trust_remote_code": vlm_identity.get("trust_remote_code"),
                       "offline": vlm_identity.get("offline"),
                       "downloads_allowed": vlm_identity.get("downloads_allowed")},
        parameters={"task_prompt": "CAPTION", "max_new_tokens": 192,
                    "num_beams": 3, "do_sample": False},
        input_value=frame_input, output_value=_frame_output(keyframes, "vlm"),
        status=module_status.get("vlm", "not_run"),
        failure_reason=_first_module_reason(keyframes, "vlm")
        or ("no sampled keyframes" if not keyframes else None),
        code_version=f"vision_jobs={VERSION}; transformers={_package_version('transformers')}",
        score_kind="none"))

    runs.append(_model_run_record(
        job_id, "scene", provider="trace-annotate", model_name="scene-cut-ffmpeg",
        model_version=VERSION, weights_sha256=None,
        license="Project source; FFmpeg build license applies to the bundled binary",
        preprocessing={"sampler": "ffmpeg lavfi.scene_score", "time_domain": DOMAIN},
        parameters={"scene_threshold": thresholds.scene_threshold,
                    "scene_min_duration_s": thresholds.scene_min_duration_s},
        input_value={"file_sha256": digest, "window": window,
                     "time_domain": DOMAIN},
        output_value={"scenes": _jsonable(scene_result.get("scenes") or []),
                      "status": scene_result.get("status")},
        status=scene_result.get("status", "failed"),
        failure_reason=scene_result.get("error") or ("no scenes detected" if not scene_result.get("scenes") else None),
        code_version=f"vision_jobs={VERSION}; ffmpeg=system",
        score_kind="rule_score"))

    runs.append(_model_run_record(
        job_id, "motion", provider="trace-annotate", model_name="camera-motion-v1",
        model_version="1", weights_sha256=None, license="Project source",
        preprocessing={"input": "adjacent sampled frames",
                       "grayscale": True, "analysis_width": 96},
        parameters={"min_points": 15, "min_inliers": 12,
                    "static_pixels": 2.0,
                    "scale_threshold": 0.02,
                    "rotation_threshold": 1.5},
        input_value=frame_input,
        output_value={"module": "motion",
                      "frames": [{"frame_index": frame["frame_index"],
                                  "summary": _jsonable((frame.get("motion") or {}).get("summary")),
                                  "status": (frame.get("motion") or {}).get("status")}
                                 for frame in keyframes]},
        status=module_status.get("motion", "not_run"),
        failure_reason=("fewer than two sampled frames per scene" if module_status.get("motion") == "not_run"
                        else None),
        code_version=f"vision_jobs={VERSION}; camera_motion=1",
        score_kind="rule_score"))

    content = content_sampling or {"status": "not_run", "reason": "content_sampling_not_recorded"}
    runs.append(_model_run_record(
        job_id, "content", provider="trace-annotate", model_name="scene-change-v1",
        model_version="1", weights_sha256=None, license="Project source",
        preprocessing={"input": "per-scene decoded content samples", "time_domain": DOMAIN,
                       "sampling": {"uniform": CONTENT_UNIFORM, "max_frames": CONTENT_MAX_FRAMES}},
        parameters=detect_content_changes([])["params"],
        input_value=_content_input(digest, window, content, content_frames or []),
        output_value={"sampling": content,
                      "changes": [{"scene_index": frame["scene_index"],
                                   "changes": frame.get("content_changes", []),
                                   "filtered": frame.get("content_filtered", [])}
                                  for frame in keyframes if "content_changes" in frame]},
        status=content.get("status", "not_run"),
        failure_reason=content.get("reason") or content.get("message") or content.get("error")
        or (f"insufficient_samples: {content.get('insufficient_scenes', 0)} scenes need "
            f"{content.get('required_samples', CONTENT_MIN_SAMPLES)} samples"
            if content.get("status") == "insufficient_samples" else None),
        code_version=f"vision_jobs={VERSION}; scene_change=1", score_kind="rule_score"))

    runs.append(_model_run_record(
        job_id, "actions", provider="trace-annotate", model_name="subject-translation-v1",
        model_version="1", weights_sha256=None, license="Project source",
        preprocessing={"input": "face-box tracks built from sampled keyframes per scene",
                       "time_domain": DOMAIN},
        parameters={"walk_band_heights_per_s": [0.15, 1.2],
                    "run_band_heights_per_s": [1.2, 6.0],
                    "supported": list(module_status.get("actions_supported") or []),
                    "unsupported": list(module_status.get("actions_unsupported") or [])},
        input_value=frame_input,
        output_value={"module": "actions",
                      "tracks": _jsonable((actions_sampling or {}).get("tracks") or []),
                      "candidates": _jsonable((actions_sampling or {}).get("candidates") or []),
                      "coverage": {"supported": module_status.get("actions_supported"),
                                   "unsupported": module_status.get("actions_unsupported")}},
        status=module_status.get("actions", "not_run"),
        failure_reason=module_status.get("actions_reason"),
        code_version=f"vision_jobs={VERSION}; actions=1",
        score_kind="rule_score"))

    acoustics = acoustics_sampling or {}
    runs.append(_model_run_record(
        job_id, "acoustics", provider="trace-annotate", model_name="acoustic-window-v1",
        model_version="1", weights_sha256=None, license="Project source",
        preprocessing={"stream": acoustics.get("stream_selection"),
                       "sample_rate": acoustics.get("sample_rate"),
                       "channels": 1, "sample_format": "float32 decoded to PCM16 evidence"},
        parameters={"f0_min_hz": 60.0, "f0_max_hz": 500.0,
                    "frame_hop_seconds": 0.02, "silence_dbfs": -45.0},
        input_value={"file_sha256": digest, "module": "acoustics"},
        output_value={"module": "acoustics",
                      "measurement": _jsonable(acoustics.get("measurement")),
                      "audio_sha256": acoustics.get("audio_sha256")},
        status=module_status.get("acoustics", "not_run"),
        failure_reason=acoustics.get("reason"),
        code_version=f"vision_jobs={VERSION}; expressive_cues=1; ffmpeg={_package_version('static-ffmpeg')}",
        score_kind="rule_score"))

    runs.append(_model_run_record(
        job_id, "boundary", provider="trace-annotate", model_name="shot-boundaries-v1",
        model_version="1", weights_sha256=None, license="Project source",
        preprocessing={"input": "boundary-radius sampled frames",
                       "analysis_width": 96, "hist_bins": 32},
        parameters={"boundary_radius": BOUNDARY_RADIUS,
                    "cut_threshold": 0.30, "gradual_low": 0.05,
                    "strong_cut": 0.60, "flash_ratio": 0.5,
                    "min_gradual_frames": 3, "dissolve_residual": 0.20,
                    "dissolve_far": 0.15, "black_level": 10,
                    "white_level": 248, "neighbour_radius": 6},
        input_value={"boundary_frames": [
            {"frame_index": frame["frame_index"],
             "seconds": frame["seconds"],
             "evidence_sha256": frame["evidence_sha256"]}
            for frame in boundary_frames]},
        output_value={"module": "boundary",
                      "events": [_jsonable(event)
                                 for frame in keyframes
                                 for event in (frame.get("boundary_events") or [])]},
        status=module_status.get("boundary", "not_run"),
        failure_reason=((boundary_sampling or {}).get("message")
                        or (boundary_sampling or {}).get("error")
                        or ("fewer than two scenes; no boundary sample"
                            if module_status.get("boundary") == "not_run" else None)),
        code_version=f"vision_jobs={VERSION}; shot_boundaries=1",
        score_kind="rule_score"))
    return runs


def _evidence_preview(source: Path) -> Path:
    """Create a small JPEG preview of one sampled keyframe.

    Full keyframe samples are hundreds of kilobytes each. A compact preview keeps
    the professional exchange document inside its bounded HTTP envelope and lets
    dozens of candidates travel together. It is a lossy, resampled preview of the
    exact sampled frame, never a caption or a synthesised image; the full sample
    remains in the job workspace.
    """
    from PIL import Image

    handle, name = tempfile.mkstemp(prefix="vision-sample-", suffix=".jpg", dir=source.parent)
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
    keyframes = candidate.get("keyframes") or []
    boundary_frames = candidate.get("boundary_frames") or []
    content_frames = candidate.get("content_frames") or []
    frames = list(keyframes) + list(boundary_frames) + list(content_frames)
    for index, annotation in enumerate(annotations):
        scene_frames = [frame for frame in frames if frame.get("scene_index") == index]
        transition_frames = [frame for frame in boundary_frames
                             if frame.get("scene_index") == index]
        evidence_ids: List[str] = []
        frame_artifacts: Dict[str, str] = {}
        sample_artifacts: Dict[Tuple[str, str], str] = {}
        # One representative sampled keyframe per candidate keeps the portable
        # professional document inside its byte budget while still grounding the
        # keyframe claim in a real immutable PNG. A camera-motion claim needs both
        # ends of its comparison, so both sampled frames are previewed.
        selected = list(scene_frames[:1])
        has_motion = any(claim["method"]["id"] == "camera-motion-local"
                         for claim in annotation["attributes"]["semantic"]["claims"])
        if has_motion:
            for claim in annotation["attributes"]["semantic"]["claims"]:
                if claim["method"]["id"] != "camera-motion-local":
                    continue
                for key in ("from_frame", "to_frame"):
                    wanted = str(claim["method"]["parameters"].get(key))
                    match = next((frame for frame in scene_frames
                                  if str(frame.get("frame_index")) == wanted), None)
                    if match is not None and match not in selected:
                        selected.append(match)
        has_transition = any(claim["method"]["id"] == "shot-boundaries-v1"
                             for claim in annotation["attributes"]["semantic"]["claims"])
        if has_transition:
            for claim in annotation["attributes"]["semantic"]["claims"]:
                if claim["method"]["id"] != "shot-boundaries-v1":
                    continue
                for key in ("previous_frame", "interval_start_frame", "anchor_frame"):
                    wanted = str(claim["method"]["parameters"].get(key))
                    match = next((frame for frame in transition_frames
                                  if str(frame.get("frame_index")) == wanted), None)
                    if match is not None and match not in selected:
                        selected.append(match)
        for claim in annotation["attributes"]["semantic"]["claims"]:
            if claim["method"]["id"] != "scene-change-v1":
                continue
            for sample in claim["method"]["parameters"].get("sample_frames", []):
                match = next((frame for frame in content_frames
                              if frame["scene_index"] == index
                              and frame["frame_index"] == sample["frame_index"]
                              and frame["evidence_sha256"] == sample["evidence_sha256"]), None)
                if match is None:
                    raise WorkspaceError("VISION_EVIDENCE_UNAVAILABLE",
                                         "Content change has no matching decoded content sample")
                if not any(str(frame["frame_index"]) == str(match["frame_index"])
                           and frame["evidence_sha256"] == match["evidence_sha256"]
                           for frame in selected):
                    selected.append(match)
        for frame in selected:
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
            if artifact["id"] not in evidence_ids:
                evidence_ids.append(artifact["id"])
            # General keyframe claims retain their own first-selected sample.
            # A content/boundary decode of the same index can have different
            # bytes and must be verified and referenced independently.
            frame_artifacts.setdefault(str(frame["frame_index"]), artifact["id"])
            sample_artifacts[(str(frame["frame_index"]), digest)] = artifact["id"]
        acoustic_claims = [claim for claim in annotation["attributes"]["semantic"]["claims"]
                           if claim["method"]["id"] == "acoustic-window-v1"]
        if acoustic_claims:
            descriptor = next((item for item in candidate.get("evidence") or []
                               if item["relative_path"].endswith("/audio/acoustic.m4a")), None)
            if descriptor is None:
                raise WorkspaceError("VISION_EVIDENCE_UNAVAILABLE",
                                     "Acoustic claim has no decoded audio evidence")
            source = store.root / descriptor["relative_path"]
            _safe(source)
            if (not source.is_file() or source.stat().st_size != int(descriptor["size_bytes"])
                    or stream_sha256(source) != descriptor["sha256"]):
                raise WorkspaceError("VISION_EVIDENCE_UNAVAILABLE",
                                     "Acoustic evidence is missing or changed")
            artifact = store._put_file(source)
            store.bind_artifact_reference(artifact, owner_kind="evidence",
                owner_id=annotation["id"], role="vision-acoustic-audio", owner_revision="0")
            evidence_ids.append(artifact["id"])
            for claim in acoustic_claims:
                claim["evidence_ids"] = [artifact["id"]]
        annotation["evidence_ids"] = evidence_ids
        for claim in annotation["attributes"]["semantic"]["claims"]:
            if claim["method"]["id"] == "camera-motion-local":
                ids = [frame_artifacts.get(str(claim["method"]["parameters"].get("from_frame"))),
                       frame_artifacts.get(str(claim["method"]["parameters"].get("to_frame")))]
                ids = [item for item in ids if item]
                if ids:
                    claim["evidence_ids"] = ids
            elif claim["method"]["id"] == "shot-boundaries-v1":
                ids = []
                for key in ("previous_frame", "interval_start_frame", "anchor_frame"):
                    value = claim["method"]["parameters"].get(key)
                    if value is None:
                        continue
                    frame = next((item for item in transition_frames
                                  if str(item["frame_index"]) == str(value)), None)
                    artifact = (sample_artifacts.get((str(value), frame["evidence_sha256"]))
                                if frame is not None else None)
                    if artifact and artifact not in ids:
                        ids.append(artifact)
                if ids:
                    claim["evidence_ids"] = ids
            elif claim["method"]["id"] in {"keyframe-sampling-local", "shot-framing-local"}:
                frame_index = str(claim["method"]["parameters"]["frame_index"])
                if frame_index in frame_artifacts:
                    claim["evidence_ids"] = [frame_artifacts[frame_index]]
            elif claim["method"]["id"] == "scene-change-v1":
                claim["evidence_ids"] = list(dict.fromkeys(
                    sample_artifacts[(sample["frame_index"], sample["evidence_sha256"])]
                    for sample in claim["method"]["parameters"].get("sample_frames", [])))


def _frame_seconds(candidate: Dict[str, Any], frame_index: Any) -> Optional[float]:
    if frame_index is None:
        return None
    frames = (list(candidate.get("content_frames") or [])
              + list(candidate.get("boundary_frames") or []) + list(candidate.get("keyframes") or []))
    for frame in frames:
        if str(frame.get("frame_index")) == str(frame_index):
            value = frame.get("seconds", frame.get("timestamp"))
            try:
                return float(S.rational_value(value)) if isinstance(value, dict) else float(value)
            except (TypeError, ValueError):
                return None
    return None


def _content_change_context(candidate: Dict[str, Any], scene_index: int,
                            change: Dict[str, Any], fallback: Dict[str, Any]):
    """Use the decoded content timeline, never a nominal-FPS reconstruction."""
    frames = sorted((frame for frame in candidate.get("content_frames", [])
                     if frame["scene_index"] == scene_index), key=lambda frame: int(frame["frame_index"]))
    start = next((frame for frame in frames if frame["frame_index"] == str(change.get("start_frame"))), None)
    end = next((frame for frame in frames if frame["frame_index"] == str(change.get("end_frame"))), None)
    if start is None or end is None or S.rational_value(start["seconds"]) >= S.rational_value(end["seconds"]):
        return fallback, [], False
    before = [frame for frame in frames if int(frame["frame_index"]) < int(start["frame_index"])]
    selected = ([before[-1]] if before else []) + [start, end]
    samples = [{key: frame[key] for key in ("frame_index", "seconds", "evidence_sha256")}
               for frame in selected]
    return ({"start": {"seconds": start["seconds"]}, "end": {"seconds": end["seconds"]}}, samples, True)


def _event_time_window(candidate: Dict[str, Any], event: Dict[str, Any],
                       fallback: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Use the detector's event interval, never the whole scene window.

    A gradual transition has an explicit interval. A cut normally has no
    interval, so the two sampled frames immediately around the anchor provide a
    one-frame bracket. Only when no frame time exists does the claim fall back
    to the scene context window, and the caller records that in uncertainty.
    """
    interval = event.get("interval") or {}
    try:
        interval_start = float(interval["start_seconds"])
        interval_end = float(interval["end_seconds"])
    except (KeyError, TypeError, ValueError):
        interval_start = interval_end = None
    if interval_start is not None and interval_end is not None and interval_start < interval_end:
        return ({"start": {"seconds": _seconds_ratio(interval_start)},
                 "end": {"seconds": _seconds_ratio(interval_end)}}, True)
    anchor_seconds = _frame_seconds(candidate, event.get("anchor_frame"))
    previous_seconds = _frame_seconds(candidate, event.get("previous_frame"))
    if anchor_seconds is not None and previous_seconds is not None and previous_seconds < anchor_seconds:
        return ({"start": {"seconds": _seconds_ratio(previous_seconds)},
                 "end": {"seconds": _seconds_ratio(anchor_seconds)}}, True)
    if anchor_seconds is not None:
        later = [float(frame["seconds"]) for frame in (candidate.get("boundary_frames") or [])
                 if str(frame.get("frame_index")) != str(event.get("anchor_frame"))
                 and frame.get("seconds") is not None]
        later = [value for value in later if value > anchor_seconds]
        if later:
            return ({"start": {"seconds": _seconds_ratio(anchor_seconds)},
                     "end": {"seconds": _seconds_ratio(min(later))}}, True)
    return fallback, False


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
        "model_run_id": _module_run_id(job_id, "scene"),
        "score": None,
        "score_kind": "none",
        "measurement": None,
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
            "measurement": None,
            "uncertainty_reason": "抽样帧只代表该时刻，不覆盖镜头内全部画面",
        })
        framing = keyframe.get("framing") or {}
        if (framing.get("status") == "success"
                and framing.get("framing") in {"远景", "全景", "中景", "近景", "特写"}):
            measurements = framing.get("measurements") or []
            ratio = measurements[0].get("height_ratio") if measurements else None
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:framing"),
                "claim_type": "measured" if isinstance(ratio, (int, float)) else "observable",
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
                "model_run_id": _module_run_id(job_id, "face"),
                "score": float(framing.get("confidence") or 0.0),
                "score_kind": "rule_score",
                "measurement": ({"value": float(ratio), "unit": "ratio", "basis": "measured",
                                 "analysis_window": window}
                                if isinstance(ratio, (int, float)) else None),
                "uncertainty_reason": framing.get("uncertainty")
                or "未检测到主体不等于画面中不存在主体",
            })
        motion = keyframe.get("motion") or {}
        summary = motion.get("summary") or {}
        if (motion.get("status") == "success"
                and summary.get("class") in MOTION_LABELS):
            pairs = [pair for pair in (motion.get("pairs") or [])
                     if pair.get("class") not in (None, "not_run")]
            confidence = min((float(pair.get("confidence") or 0.0) for pair in pairs), default=0.0)
            first_pair = pairs[0] if pairs else {}
            speed = summary.get("mean_speed_px_s")
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:motion"),
                "claim_type": "measured" if isinstance(speed, (int, float)) else "observable",
                "statement": (f"该镜头相邻抽样帧之间的画面全局运动为"
                              f"{summary['class']}（仅抽样帧对比，不代表全程运镜）。"),
                "subject_ref": None,
                "time_window": window,
                "evidence_ids": [],
                "method": {"id": "camera-motion-local", "version": "1",
                           "parameters": {"class": summary["class"],
                                          "agreement": summary.get("agreement"),
                                          "pairs": summary.get("pairs"),
                                          "mean_speed_px_s": speed,
                                          "frame_index": keyframe.get("frame_index"),
                                          "from_frame": first_pair.get("from_frame"),
                                          "to_frame": first_pair.get("to_frame")}},
                "model_run_id": _module_run_id(job_id, "motion"),
                "score": round(confidence, 3),
                "score_kind": "rule_score",
                "measurement": ({"value": float(speed), "unit": "px/s", "basis": "measured",
                                 "analysis_window": window}
                                if isinstance(speed, (int, float)) else None),
                "uncertainty_reason": summary.get("uncertainty")
                or (f"仅 {len(pairs)} 对相邻抽样帧，不能证明整段镜头的运动方式" if len(pairs) < 2
                    else "相邻抽样帧分类不一致，需人工复核"),
            })
        for change in keyframe.get("content_changes") or []:
            content_window, sample_frames, exact = _content_change_context(candidate, index, change, window)
            content_run = _module_run_id(job_id, "content")
            if not any(run.get("id") == content_run for run in candidate.get("model_runs", [])):
                content_run = None
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:content:{change.get('start_frame')}"),
                "claim_type": "observable",
                "statement": (f"镜头内部 #{change.get('start_frame')}–"
                              f"#{change.get('end_frame')} 帧出现持续画面内容变化"
                              "（边缘结构与色度同时改变，且新状态在后续抽样帧保持；需人工复核）。"),
                "subject_ref": None,
                "time_window": content_window,
                "evidence_ids": [],
                "method": {"id": "scene-change-v1", "version": "1",
                           "parameters": {"start_frame": change.get("start_frame"),
                                          "end_frame": change.get("end_frame"),
                                           "sample_frames": sample_frames,
                                          "metrics": change.get("metrics"),
                                          "sustained": change.get("sustained")}},
                "model_run_id": content_run,
                "score": float(change.get("confidence") or 0.0),
                "score_kind": "rule_score",
                "measurement": None,
                "uncertainty_reason": (change.get("uncertainty")
                    or "仅按镜头内均匀抽样帧判断，未逐帧验证；短时遮挡或字幕可能被过滤")
                    + ("；旧候选未记录内容采样时间与证据，时间窗仅为镜头上下文" if not exact else ""),
            })
        actions = candidate.get("actions_sampling") or {}
        for number, action in enumerate(actions.get("candidates") or []):
            window_seconds = action.get("window_seconds") or {}
            try:
                left = float(window_seconds.get("start"))
                right = float(window_seconds.get("end"))
            except (TypeError, ValueError):
                continue
            if right < float(scene["start"]) or left > float(scene["end"]):
                continue
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:action:{number}"),
                "claim_type": "measured",
                "statement": (f"主体框位移测得候选动作“{action.get('action')}”"
                              f"（{left:.3f}–{right:.3f} 秒）；动作类别与主体身份需人工确认。"),
                "subject_ref": action.get("subject_id"),
                "time_window": {"start": {"seconds": _seconds_ratio(left)},
                                "end": {"seconds": _seconds_ratio(right)}},
                "evidence_ids": [],
                "method": {"id": "subject-translation-v1", "version": "1",
                           "parameters": {"action": action.get("action"),
                                          "subject_id": action.get("subject_id"),
                                          "metrics": action.get("metrics"),
                                          "walk_band": [0.15, 1.2],
                                          "run_band": [1.2, 6.0]}},
                "model_run_id": _module_run_id(job_id, "actions"),
                "score": float(action.get("score") or 0.0),
                "score_kind": "rule_score",
                "measurement": {"value": float((action.get("metrics") or {}).get("speed_heights_per_s") or 0.0),
                                "unit": "subject_heights/s", "basis": "measured",
                                "analysis_window": {"start": {"seconds": _seconds_ratio(left)},
                                                    "end": {"seconds": _seconds_ratio(right)}}},
                "uncertainty_reason": action.get("uncertainty")
                or "仅由主体框位移推出，动作类别需人工确认",
            })
        acoustics = candidate.get("acoustics_sampling") or {}
        measurement = acoustics.get("measurement") or {}
        if index == 0 and acoustics.get("status") == "success" and measurement.get("status") == "success":
            claims.append({
                "id": S.make_id("clm", f"{job_id}:0:acoustics"),
                "claim_type": "measured",
                "statement": (f"首条音轨测量：RMS {measurement.get('rms_dbfs')} dBFS，"
                              f"语音活动占比 {measurement.get('active_ratio')}，"
                              "仅声学测量，不代表音色、身份或情绪。"),
                "subject_ref": None,
                "time_window": {"start": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("start", 0.0)))},
                                "end": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("end", 0.0)))}},
                "evidence_ids": [],
                "method": {"id": "acoustic-window-v1", "version": "1",
                           "parameters": {"stream_selection": acoustics.get("stream_selection"),
                                          "sample_rate": acoustics.get("sample_rate"),
                                          "f0_min_hz": 60.0, "f0_max_hz": 500.0,
                                          "frame_hop_seconds": 0.02,
                                          "f0_hz": measurement.get("f0_hz"),
                                          "f0_variation_hz": measurement.get("f0_variation_hz"),
                                          "active_ratio": measurement.get("active_ratio"),
                                          "active_intervals": measurement.get("active_intervals")}},
                "model_run_id": _module_run_id(job_id, "acoustics"),
                "score": 1.0,
                "score_kind": "confidence",
                "measurement": {"value": measurement.get("rms_dbfs"),
                                "unit": "dBFS", "basis": "measured",
                                "analysis_window": {"start": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("start", 0.0)))},
                                                    "end": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("end", 0.0)))}}},
                "uncertainty_reason": "仅声学测量，不代表音色、身份或情绪；语音区间由能量阈值推出",
            })
        for event in keyframe.get("boundary_events") or []:
            kind = event.get("kind")
            if kind not in TRANSITION_KINDS:
                continue
            statements = {"cut": "发生硬切",
                          "fade_in": "为渐入（画面由黑/白过渡到正常）",
                          "fade_out": "为渐出（画面由正常过渡到黑/白）",
                          "dissolve": "为叠化",
                          "flash": "出现单帧闪光，未切断镜头",
                          "unknown_transition": "存在未建模的持续转场"}
            interval = event.get("interval") or {}
            if interval.get("start_frame") is not None and interval.get("end_frame") is not None:
                statement = (f"镜头边界检测判定 #{interval.get('start_frame')}–"
                             f"#{interval.get('end_frame')} 帧区间内{statements[kind]}"
                             f"（代表帧 #{event.get('anchor_frame')}，仅边界前后抽样帧，需人工复核）。")
            else:
                statement = (f"镜头边界检测在 #{event.get('anchor_frame')} 帧判定"
                             f"{statements[kind]}（仅边界前后抽样帧，需人工复核）。")
            metrics = {}
            for key, item in (event.get("metrics") or {}).items():
                if isinstance(item, bool):
                    metrics[str(key)] = item
                elif isinstance(item, (int, float)):
                    metrics[str(key)] = round(float(item), 4) if isinstance(item, float) else int(item)
                elif hasattr(item, "item"):
                    metrics[str(key)] = item.item()
                else:
                    metrics[str(key)] = str(item)
            transition_window, transition_window_is_exact = _event_time_window(candidate, event, window)
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:transition:{kind}:{event.get('anchor_frame')}"),
                "claim_type": "observable",
                "statement": statement,
                "subject_ref": None,
                "time_window": transition_window,
                "evidence_ids": [],
                "method": {"id": "shot-boundaries-v1", "version": "1",
                           "parameters": {"kind": kind,
                                          "anchor_frame": event.get("anchor_frame"),
                                          "previous_frame": event.get("previous_frame"),
                                          "representative_frame": event.get("representative_frame"),
                                          "split_point": event.get("split_point"),
                                          "interval_start_frame": interval.get("start_frame"),
                                          "interval_end_frame": interval.get("end_frame"),
                                          "interval_start_seconds": interval.get("start_seconds"),
                                          "interval_end_seconds": interval.get("end_seconds"),
                                          "confidence": event.get("confidence"),
                                          "needs_review": event.get("needs_review"),
                                          "boundary_radius": BOUNDARY_RADIUS,
                                          "metrics": metrics}},
                "model_run_id": _module_run_id(job_id, "boundary"),
                "score": float(event.get("confidence") or 0.0),
                "score_kind": "rule_score",
                "measurement": None,
                "uncertainty_reason": event.get("uncertainty")
                or (None if transition_window_is_exact
                    else "边界事件缺少可用的帧级区间；claim 使用镜头上下文窗口"),
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
            motion = keyframe.get("motion") or {}
            summary = motion.get("summary") or {}
            if (motion.get("status") == "success"
                    and summary.get("class") in MOTION_LABELS):
                tags.append("vision-motion-" + summary["class"])
            if keyframe.get("content_changes"):
                tags.append("vision-content-change")
            actions = candidate.get("actions_sampling") or {}
            for action in actions.get("candidates") or []:
                window_seconds = action.get("window_seconds") or {}
                try:
                    left = float(window_seconds.get("start"))
                    right = float(window_seconds.get("end"))
                except (TypeError, ValueError):
                    continue
                if right >= float(scene["start"]) and left <= float(scene["end"]):
                    tags.append("vision-action-" + str(action.get("action")))
            if index == 0 and (candidate.get("acoustics_sampling") or {}).get("status") == "success":
                tags.append("vision-acoustics")
            for event in keyframe.get("boundary_events") or []:
                if event.get("kind") in TRANSITION_LABELS:
                    tags.append("vision-transition-" + event["kind"])
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
