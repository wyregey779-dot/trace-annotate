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
import time
from importlib import metadata
from types import SimpleNamespace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import MetricThresholds
from ..extractors.actions import (ACTION_VOCAB, detect_translation_candidates,
                                  merge_candidates, provider_coverage)
from ..extractors.camera_motion import analyse_motion
from ..extractors.expressive_cues import acoustic_claim, measure_acoustic_window, face_cue_candidates, cues_to_claims
from ..extractors.persons import PersonExtractor, OFFICIAL_WEIGHTS_SHA256 as PERSON_WEIGHTS_SHA256
from . import optional_vision as OV
from . import model_worker as MW
from ..extractors.scene_change import detect_content_changes, minimum_samples
from ..extractors.clip_engine import CLIPExtractor
from ..extractors.faces import FaceExtractor
from ..extractors.ocr import OCRExtractor
from ..extractors.scene import SceneDetector, SceneDetectionError
from ..extractors.shot_attributes import classify_framing
from ..extractors.shot_boundaries import (classify_scene_cut, detect_shot_boundaries, frame_features,
                                          TRANSITION_KINDS, _params as boundary_parameters)
from ..extractors.vlm import CAPTION_TASK, VLMExtractor
from ..temporal.media import stream_sha256
from . import professional_schema as PS
from . import schema as S
from . import semantic_schema as SS
from . import vision_sampling as VS
from . import transition_discovery as TD
from .semantic_schema import SEMANTIC_VERSION
from .errors import Conflict, SchemaInvalid, WorkspaceError, revision_conflict
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
AUDIO_SAMPLE_RATE = 16000
AUDIO_DECODE_TIMEOUT_SECONDS = 600.0
AUDIO_ENCODE_TIMEOUT_SECONDS = 120.0
AUDIO_MAX_ANALYSIS_SECONDS = 120.0
AUDIO_ANALYSIS_TIMEOUT_SECONDS = 60.0
AUDIO_WORKSPACE_RESERVE_BYTES = 256 * 1024 * 1024
AUDIO_PCM_TOLERANCE_SAMPLES = 1024
EXECUTION_MANIFEST_VERSION = 1
ISOLATABLE_MODEL_ERRORS = frozenset({
    "INSUFFICIENT_MEMORY",
    "MEMORY_STATUS_UNAVAILABLE",
    "MODEL_RESOURCE_BUSY",
    "MODEL_TIMEOUT",
    "MODEL_LOG_LIMIT",
    "MODEL_MEMORY_FAILED",
    "MODEL_LOAD_FAILED",
    "MODEL_PROGRESS_FAILED",
    "MODEL_WORKER_FAILED",
})
_FILE_DESCRIPTOR_LOCK = threading.Lock()
_TOOL_VERSION_CACHE: Dict[Tuple[str, str, str], str] = {}
EXECUTION_SOURCE_FILES = (
    "__init__.py",
    "config.py",
    "core/hasher.py",
    "extractors/__init__.py",
    "extractors/base.py",
    "extractors/metrics.py",
    "temporal/__init__.py",
    "temporal/media.py",
    "temporal/preparation.py",
    "temporal/schema.py",
    "workspace/__init__.py",
    "workspace/artifact_refs.py",
    "workspace/errors.py",
    "workspace/http_media.py",
    "workspace/queue.py",
    "workspace/schema.py",
    "workspace/slots.py",
    "workspace/store.py",
    "workspace/vision_jobs.py",
    "workspace/vision_sampling.py",
    "workspace/transition_discovery.py",
    "workspace/optional_vision.py",
    "workspace/model_worker.py",
    "workspace/resource_budget.py",
    "workspace/frames.py",
    "workspace/semantic_schema.py",
    "workspace/professional_schema.py",
    "extractors/actions.py",
    "extractors/camera_motion.py",
    "extractors/expressive_cues.py",
    "extractors/shot_attributes.py",
    "extractors/scene_change.py",
    "extractors/shot_boundaries.py",
    "extractors/scene.py",
    "extractors/faces.py",
    "extractors/ocr.py",
    "extractors/persons.py",
    "extractors/face_landmarks.py",
    "extractors/clip_engine.py",
    "extractors/vlm.py",
    "extractors/model_loading.py",
)


def _path_identity(path: Path) -> str:
    """Bind a local deployment path without exporting the user's absolute path."""
    return S.sha256_text({"resolved_path": os.path.normcase(str(path.resolve()))})


def _file_descriptor(path: Optional[str], *, literal: Optional[str] = None) -> Dict[str, Any]:
    """Describe exact regular-file bytes; unavailable resources stay explicit."""
    if not path:
        return {"status": "unavailable", "literal": literal}
    candidate = Path(path)
    try:
        info = candidate.stat()
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
        signature = (getattr(info, "st_dev", None), getattr(info, "st_ino", None),
                     info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        digest = stream_sha256(candidate)
        after = candidate.stat()
        after_signature = (getattr(after, "st_dev", None), getattr(after, "st_ino", None),
                           after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if after_signature != signature:
            raise OSError("file changed while its identity was measured")
        descriptor = {"status": "available", "name": candidate.name,
                      "path_identity_sha256": _path_identity(candidate),
                      "sha256": digest, "size_bytes": str(info.st_size)}
        return descriptor
    except OSError as exc:
        return {"status": "unavailable", "name": candidate.name,
                "path_identity_sha256": _path_identity(candidate),
                "reason": type(exc).__name__}


def _tool_descriptor(name: str, explicit_path: Optional[str] = None) -> Dict[str, Any]:
    selected = explicit_path or shutil.which(name)
    descriptor = _file_descriptor(selected)
    if descriptor["status"] != "available":
        return descriptor
    cache_key = (name, descriptor["path_identity_sha256"], descriptor["sha256"])
    with _FILE_DESCRIPTOR_LOCK:
        cached_version = _TOOL_VERSION_CACHE.get(cache_key)
    if cached_version is not None:
        return {**descriptor, "version": cached_version}
    version = "unavailable"
    try:
        completed = subprocess.run([str(Path(selected)), "-version"], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", timeout=5,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if completed.returncode == 0 and completed.stdout:
            version = completed.stdout.splitlines()[0][:300]
    except (OSError, subprocess.TimeoutExpired):
        pass
    with _FILE_DESCRIPTOR_LOCK:
        _TOOL_VERSION_CACHE[cache_key] = version
    return {**descriptor, "version": version}


def _package_version_or_unavailable(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _ocr_deployment(model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Freeze explicit OCR model bytes without importing or loading the engine."""
    directory = Path(model_dir) if model_dir else None
    names = {
        "det": ("PP-OCRv6_det_small.onnx", "ch_PP-OCRv4_det_infer.onnx"),
        "cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
        "rec": ("PP-OCRv6_rec_small.onnx", "ch_PP-OCRv4_rec_infer.onnx"),
    }
    files: List[Dict[str, Any]] = []
    selected_runtime = None
    if directory is not None:
        for role, alternatives in names.items():
            selected = next((directory / item for item in alternatives
                             if (directory / item).is_file()), directory / alternatives[0])
            files.append({"role": role, **_file_descriptor(str(selected))})
    else:
        for distribution in ("rapidocr-onnxruntime", "rapidocr"):
            try:
                installed = metadata.distribution(distribution)
            except metadata.PackageNotFoundError:
                continue
            located = {role: next((Path(installed.locate_file(item))
                                   for item in installed.files or []
                                   if Path(str(item)).name in alternatives
                                   and Path(installed.locate_file(item)).is_file()), None)
                       for role, alternatives in names.items()}
            if all(located.values()):
                selected_runtime = distribution
                files = [{"role": role, **_file_descriptor(str(path))}
                         for role, path in located.items()]
                break
    return {"configured": directory is not None,
            "model_dir_identity_sha256": _path_identity(directory) if directory is not None else None,
            "files": files,
            "selected_runtime": selected_runtime,
            "runtime_versions": {
                "rapidocr": _package_version_or_unavailable("rapidocr"),
                "rapidocr_onnxruntime": _package_version_or_unavailable("rapidocr-onnxruntime"),
            }}


def _implementation_manifest(package_root: Optional[Path] = None,
                             relative_files: Sequence[str] = EXECUTION_SOURCE_FILES) -> Dict[str, Any]:
    root = package_root or Path(__file__).resolve().parents[1]
    files = []
    for relative in relative_files:
        path = root / PurePosixPath(relative)
        descriptor = _file_descriptor(str(path))
        files.append({"path": relative, "status": descriptor["status"],
                      "sha256": descriptor.get("sha256"),
                      "size_bytes": descriptor.get("size_bytes")})
    return {"files": files, "sha256": S.sha256_text(files)}


def _execution_deployment(*, face_model_path: Optional[str] = None,
                          ocr_model_dir: Optional[str] = None,
                          ffmpeg_path: Optional[str] = None,
                          ffprobe_path: Optional[str] = None,
                          package_root: Optional[Path] = None) -> Dict[str, Any]:
    face_selected = face_model_path or os.getenv("TRACE_ANNOTATE_FACE_MODEL")
    face = ({"status": "builtin", "literal": "opencv_haar_builtin"}
            if face_selected == "haar" else _file_descriptor(face_selected))
    deployment = {
        "version": EXECUTION_MANIFEST_VERSION,
        "implementation": _implementation_manifest(package_root),
        "ffmpeg": _tool_descriptor("ffmpeg", ffmpeg_path),
        "ffprobe": _tool_descriptor("ffprobe", ffprobe_path),
        "face": face,
        "ocr": _ocr_deployment(ocr_model_dir or os.getenv("TRACE_ANNOTATE_OCR_MODELS")),
        "runtime_versions": {
            "python": ".".join(str(item) for item in os.sys.version_info[:3]),
            "opencv_python": _package_version_or_unavailable("opencv-python"),
            "numpy": _package_version_or_unavailable("numpy"),
        },
    }
    return {**deployment, "deployment_sha256": S.sha256_text(deployment)}


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


def _job_identity(*, content_evidence_contract: Optional[str] = CONTENT_EVIDENCE_CONTRACT,
                  optional_providers: Optional[Dict[str, Any]] = None,
                  include_optional: bool = True, model_resources: Optional[Dict[str, Any]] = None,
                  include_model_resources: bool = True,
                  execution_deployment: Optional[Dict[str, Any]] = None,
                  include_execution_deployment: bool = True,
                  include_transition_discovery: bool = True,
                  transition_configuration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    if include_optional:
        config["optional_providers"] = OV.validate_config(optional_providers if optional_providers is not None else OV.provider_config())
    if include_model_resources:
        config["model_resources"] = MW.validate_configuration(
            model_resources if model_resources is not None else MW.configuration())
    if include_execution_deployment:
        deployment = execution_deployment if execution_deployment is not None else _execution_deployment()
        if (not isinstance(deployment, dict) or deployment.get("version") != EXECUTION_MANIFEST_VERSION
                or deployment.get("deployment_sha256") != S.sha256_text({
                    key: value for key, value in deployment.items() if key != "deployment_sha256"})):
            raise SchemaInvalid("SCHEMA_INVALID", "Execution deployment identity is invalid")
        config["execution_deployment"] = deployment
    if include_transition_discovery:
        config["transition_discovery"] = TD.validate_configuration(
            TD.configuration() if transition_configuration is None else transition_configuration)
    return {"pipeline": "vision-assist-v1", "processor_version": VERSION,
            "config_sha256": S.sha256_text(config), "config": config,
            "model_identity": "deployment frozen at submit; loaded bytes recorded per run",
            "capability_identity": "implementation+ffmpeg+ffprobe+local resources frozen by sha256"}


def _parameters(window: Optional[Dict[str, Any]] = None,
                subject_annotation_id: Optional[str] = None, *,
                optional_providers: Optional[Dict[str, Any]] = None,
                model_resources: Optional[Dict[str, Any]] = None,
                 execution_deployment: Optional[Dict[str, Any]] = None,
                 include_execution_deployment: bool = True,
                include_model_resources: bool = True,
                include_transition_discovery: bool = True,
                transition_configuration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                                        "time_domain": DOMAIN, "job_identity": _job_identity(
                                            optional_providers=optional_providers, model_resources=model_resources,
                                            execution_deployment=execution_deployment,
                                            include_execution_deployment=include_execution_deployment,
                                            include_model_resources=include_model_resources,
                                            include_transition_discovery=include_transition_discovery,
                                            transition_configuration=transition_configuration)}))


def _validate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if job["operation"] != OPERATION or job["processor_version"] != VERSION:
        raise WorkspaceError("UNSUPPORTED_PROCESSOR", "No installed vision processor matches this frozen job")
    parameters = job["parameters"]
    if (not isinstance(parameters, dict)
            or set(parameters) != {"window", "subject_annotation_id", "time_domain", "job_identity"}
            or parameters["time_domain"] != DOMAIN):
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid frozen vision parameters")
    identity = parameters["job_identity"]
    if not isinstance(identity, dict) or not isinstance(identity.get("config"), dict):
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid frozen vision parameters")
    optional = identity["config"].get("optional_providers")
    resources = identity["config"].get("model_resources")
    deployment = identity["config"].get("execution_deployment")
    if resources is not None:
        MW.validate_configuration(resources)
    if optional is not None:
        OV.validate_config(optional)
    resource_options = {"model_resources": resources, "include_model_resources": resources is not None,
                        "execution_deployment": deployment,
                        "include_execution_deployment": deployment is not None,
                        "include_transition_discovery": "transition_discovery" in identity["config"],
                        "transition_configuration": identity["config"].get("transition_discovery")}
    current = _parameters(parameters["window"], parameters["subject_annotation_id"],
                          optional_providers=optional, **resource_options)
    # Immutable completed candidates remain readable after optional deployment
    # or implementation changes. Execution separately requires the current
    # descriptor, so historical queued work cannot silently use new models.
    identities = [_job_identity(optional_providers=optional, **resource_options),
                  _job_identity(optional_providers=optional, content_evidence_contract=None, **resource_options)]
    if optional is None:
        identities.extend([_job_identity(include_optional=False, **resource_options),
                           _job_identity(include_optional=False, content_evidence_contract=None, **resource_options)])
    # Completed candidates from before execution-deployment freezing remain
    # readable/applyable.  The processor's strict current comparison below
    # rejects such queued work and requires an explicit resubmission.
    legacy_options = {key: value for key, value in resource_options.items()
                      if key not in {"execution_deployment", "include_execution_deployment"}}
    identities.extend([
        _job_identity(optional_providers=optional, include_execution_deployment=False,
                      **legacy_options),
        _job_identity(optional_providers=optional, content_evidence_contract=None,
                      include_execution_deployment=False, **legacy_options),
    ])
    if optional is None:
        identities.extend([
            _job_identity(include_optional=False, include_execution_deployment=False,
                          **legacy_options),
            _job_identity(include_optional=False, content_evidence_contract=None,
                          include_execution_deployment=False, **legacy_options),
        ])
    if not any(parameters == dict(current, job_identity=item) for item in identities):
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
    """One-to-one short box tracks; never link across a scene or a >5s gap.

    Person xyxy boxes take precedence. Face-only xywh geometry is retained as
    a labelled fallback, never mixed into a person's trajectory. IoU gives
    geometric continuity only, not a persistent identity or tracking accuracy.
    """
    tracks: List[Dict[str, Any]] = []
    per_scene: Dict[int, List[Dict[str, Any]]] = {}
    for keyframe in sorted(keyframes, key=lambda item: (item["scene_index"], S.rational_value(item["timestamp"]))):
        persons = keyframe.get("person") or {}
        face = keyframe.get("face") or {}
        detections = []
        kind = "person"
        if persons.get("status") == "success" and persons.get("persons_detected"):
            detections = [{"box": list(map(float, item["box"])),
                           "confidence": item.get("confidence")}
                          for item in persons["persons_detected"]]
        elif face.get("status") == "success":
            kind = "face"
            for item in face.get("faces_detected") or []:
                x, y, w, h = map(float, item["box"])
                detections.append({"box": [x, y, x + w, y + h],
                                   "confidence": item.get("confidence")})
        detections.sort(key=lambda item: tuple(item["box"]))
        boxes = [item["box"] for item in detections]
        seconds = float(S.rational_value(keyframe["timestamp"]))
        scene_index = int(keyframe["scene_index"])
        candidates = per_scene.setdefault(scene_index, [])
        # Assign strongest overlap first and consume both endpoints. A second
        # detection can never append to the same track in this frame.
        pairs = []
        for track_index, track in enumerate(candidates):
            gap = seconds - track["points"][-1]["seconds"]
            if not track.get("active", True) or track["kind"] != kind or not 0 < gap <= 5:
                continue
            for box_index, box in enumerate(boxes):
                overlap = _box_iou(track["points"][-1]["box"], box)
                if overlap >= .2:
                    pairs.append((overlap, track_index, box_index))
        # Near-tied candidates are an identity ambiguity, not permission to
        # force an ID through a crossing. Terminate the uncertain old track and
        # start a new visual track for the current detection.
        ambiguous_tracks, ambiguous_boxes = set(), set()
        for track_index in {item[1] for item in pairs}:
            scores = sorted((item[0] for item in pairs if item[1] == track_index), reverse=True)
            if len(scores) > 1 and scores[0] - scores[1] <= .05:
                ambiguous_tracks.add(track_index)
        for box_index in {item[2] for item in pairs}:
            scores = sorted((item[0] for item in pairs if item[2] == box_index), reverse=True)
            if len(scores) > 1 and scores[0] - scores[1] <= .05:
                ambiguous_boxes.add(box_index)
        # A tie contaminates its whole overlap component.  Keeping only the
        # locally tied endpoint can still swap the neighbouring identity.
        changed = True
        while changed:
            before = (len(ambiguous_tracks), len(ambiguous_boxes))
            ambiguous_boxes.update(box_index for _, track_index, box_index in pairs
                                   if track_index in ambiguous_tracks)
            ambiguous_tracks.update(track_index for _, track_index, box_index in pairs
                                    if box_index in ambiguous_boxes)
            changed = before != (len(ambiguous_tracks), len(ambiguous_boxes))
        for track_index in ambiguous_tracks:
            candidates[track_index]["active"] = False
            candidates[track_index]["termination_reason"] = "ambiguous_association"
        used_tracks, assigned = set(), {}
        for overlap, track_index, box_index in sorted(
                pairs, key=lambda item: (-item[0], candidates[item[1]]["subject_id"], item[2])):
            if track_index in ambiguous_tracks or box_index in ambiguous_boxes:
                continue
            if track_index not in used_tracks and box_index not in assigned:
                used_tracks.add(track_index)
                assigned[box_index] = (candidates[track_index], overlap)
        for box_index, box in enumerate(boxes):
            assignment = assigned.get(box_index)
            if assignment is None:
                track = {"subject_id": f"scene{scene_index}-subject{len(candidates) + 1}",
                         "scene_index": scene_index, "kind": kind, "points": [], "active": True}
                candidates.append(track)
                tracks.append(track)
                association = {"method": "new_track",
                               "reason": ("ambiguous_association" if box_index in ambiguous_boxes
                                          else "no_unambiguous_overlap")}
            else:
                track, overlap = assignment
                association = {"method": "iou", "score": round(float(overlap), 6)}
            track["points"].append({"box": [round(value, 2) for value in box],
                                    "frame_index": keyframe["frame_index"], "seconds": seconds,
                                    "evidence_sha256": keyframe.get("evidence_sha256"),
                                    "detection_confidence": detections[box_index].get("confidence"),
                                    "association": association})
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
        if track["kind"] != "person":
            result = {"status": "not_applicable", "candidates": [],
                      "reason": "face_box_is_not_a_body_motion_track"}
        else:
            result = detect_translation_candidates(points, subject_id=track["subject_id"], min_samples=3)
        track_results.append({"subject_id": track["subject_id"],
                              "scene_index": track["scene_index"],
                              "kind": track["kind"], "samples": len(points),
                              "status": result.get("status"),
                              "reason": result.get("reason"),
                              "metrics": result.get("metrics"),
                              "termination_reason": track.get("termination_reason"),
                              "candidate_count": len(result.get("candidates") or [])})
        for item in result.get("candidates") or []:
            item["provider"] = "subject-translation-v1"
            item["scene_index"] = track["scene_index"]
            candidates.append(item)
    for track in tracks:
        for point in track["points"]:
            point.pop("box", None)
    merged = merge_candidates(candidates)
    person_tracks = [track for track in tracks if track["kind"] == "person"]
    if not tracks:
        status, reason = "not_run", "no_subject_box_track_in_sampled_frames"
    elif not person_tracks:
        status, reason = "not_run", "no_person_box_track_in_sampled_frames"
    elif not any(item["kind"] == "person" and item["samples"] >= 3 for item in track_results):
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


def _audio_window(window: Optional[Dict[str, Any]], media_duration: Any) -> Tuple[float, float]:
    """Return a finite bounded half-open audio window in playback seconds."""
    if window is None:
        start, end = 0.0, S.rational_value(media_duration)
    else:
        start = S.rational_value(window["start"])
        end = S.rational_value(window["end"])
    if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start):
        raise ValueError("invalid_audio_window")
    return start, end - start


def _audio_child(command: List[str], output: Optional[Path], diagnostic: Path, *,
                 cancelled, timeout: float) -> Tuple[int, str]:
    """Run one owned ffmpeg child without retaining media bytes in pipe memory."""
    if cancelled():
        raise WorkspaceError("CANCELLED", "Acoustic analysis cancelled before decode")
    process = None
    output_handle = None
    diagnostic_handle = None
    try:
        output_handle = output.open("xb") if output is not None else None
        diagnostic_handle = diagnostic.open("xb")
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL,
            stdout=output_handle if output_handle is not None else subprocess.DEVNULL,
            stderr=diagnostic_handle,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if cancelled():
                raise WorkspaceError("CANCELLED", "Acoustic analysis cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("acoustic_ffmpeg_timeout")
            time.sleep(.05)
        returncode = int(process.returncode)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if output_handle is not None:
            output_handle.close()
        if diagnostic_handle is not None:
            diagnostic_handle.close()
    detail = diagnostic.read_text("utf-8", errors="replace")[-1000:]
    return returncode, detail


def _measure_audio(source: Path, ffmpeg: str, work: Path, *,
                   window: Optional[Dict[str, Any]] = None, media_duration: Any = None,
                   cancelled=lambda: False) -> Dict[str, Any]:
    """Measure one bounded playback window from the first real audio stream."""
    if media_duration is None and window is None:
        return {"status": "not_run", "reason": "audio_media_duration_unavailable",
                "measurement": None}
    try:
        start, requested_duration = _audio_window(window, media_duration)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return {"status": "failed", "reason": f"audio_window_invalid:{type(exc).__name__}",
                "measurement": None}
    import numpy as np

    audio_dir = work / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_path = audio_dir / "acoustic.f32le"
    decode_log = audio_dir / "decode.stderr"
    encode_log = audio_dir / "encode.stderr"
    audio_path = audio_dir / "acoustic.m4a"
    analysis_duration = min(requested_duration, AUDIO_MAX_ANALYSIS_SECONDS)
    expected_pcm_bytes = int(math.ceil(analysis_duration * AUDIO_SAMPLE_RATE)) * 4
    required_free_bytes = expected_pcm_bytes + AUDIO_WORKSPACE_RESERVE_BYTES
    try:
        available_bytes = shutil.disk_usage(audio_dir).free
    except OSError as exc:
        return {"status": "failed", "reason": "audio_storage_status_unavailable",
                "error": type(exc).__name__, "measurement": None}
    if required_free_bytes > available_bytes:
        return {"status": "failed", "reason": "insufficient_disk_for_audio_window",
                "required_bytes": required_free_bytes,
                "pcm_bytes": expected_pcm_bytes,
                "reserve_bytes": AUDIO_WORKSPACE_RESERVE_BYTES,
                "available_bytes": available_bytes,
                "measurement": None}
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", str(source)]
    if start:
        command.extend(["-ss", _frame_timestamp(start)])
    command.extend(["-t", _frame_timestamp(analysis_duration), "-map", "0:a:0", "-vn",
                    "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "f32le",
                    "-fs", str(expected_pcm_bytes), "pipe:1"])
    retain_audio = False
    try:
        try:
            code, detail = _audio_child(command, raw_path, decode_log,
                                        cancelled=cancelled, timeout=AUDIO_DECODE_TIMEOUT_SECONDS)
        except OSError as exc:
            return {"status": "not_run", "reason": f"audio_decoder_unavailable:{type(exc).__name__}",
                    "measurement": None}
        except TimeoutError:
            return {"status": "failed", "reason": "audio_decode_timeout", "measurement": None}
        no_stream = any(marker in detail.lower() for marker in
                        ("matches no streams", "does not contain any stream", "stream map '0:a:0'"))
        if code != 0:
            return {"status": "not_run" if no_stream else "failed",
                    "reason": "no_decodable_audio_track" if no_stream else "audio_decode_failed",
                    "detail": detail[-300:], "measurement": None}
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            return {"status": "failed", "reason": "audio_decode_empty", "measurement": None}
        if raw_path.stat().st_size % 4:
            return {"status": "failed", "reason": "audio_decode_invalid_pcm_size",
                    "measurement": None}
        maximum_pcm_bytes = expected_pcm_bytes + AUDIO_PCM_TOLERANCE_SAMPLES * 4
        if raw_path.stat().st_size > maximum_pcm_bytes:
            return {"status": "failed", "reason": "audio_decode_exceeded_budget",
                    "expected_pcm_bytes": expected_pcm_bytes,
                    "actual_pcm_bytes": raw_path.stat().st_size,
                    "tolerance_samples": AUDIO_PCM_TOLERANCE_SAMPLES,
                    "measurement": None}
        samples = np.memmap(raw_path, dtype="<f4", mode="r")
        analysis_deadline = time.monotonic() + AUDIO_ANALYSIS_TIMEOUT_SECONDS
        try:
            measurement = measure_acoustic_window(
                samples, AUDIO_SAMPLE_RATE, origin_seconds=start,
                cancelled=cancelled, deadline_monotonic=analysis_deadline)
        except InterruptedError as exc:
            raise WorkspaceError("CANCELLED", "Acoustic analysis cancelled") from exc
        except TimeoutError:
            return {"status": "failed", "reason": "audio_analysis_timeout",
                    "measurement": None}
        if measurement.get("status") != "success":
            return {"status": "not_run", "reason": measurement.get("reason"),
                    "measurement": measurement}
        actual_duration = int(samples.size) / float(AUDIO_SAMPLE_RATE)
        duration_tolerance = 1.0 / AUDIO_SAMPLE_RATE
        if requested_duration > AUDIO_MAX_ANALYSIS_SECONDS:
            coverage_kind = "bounded_prefix"
        elif actual_duration + duration_tolerance < requested_duration:
            coverage_kind = "decoded_source_shortfall"
        else:
            coverage_kind = "complete"
        coverage = {
            "kind": coverage_kind,
            "requested_seconds": requested_duration,
            "analysed_seconds": actual_duration,
            "ratio": min(1.0, actual_duration / requested_duration),
            "truncated": coverage_kind != "complete",
            "max_analysis_seconds": AUDIO_MAX_ANALYSIS_SECONDS,
        }
        measurement["coverage"] = coverage
        try:
            remaining_bytes = shutil.disk_usage(audio_dir).free
        except OSError as exc:
            return {"status": "failed", "reason": "audio_storage_status_unavailable",
                    "error": type(exc).__name__, "measurement": measurement}
        if remaining_bytes < AUDIO_WORKSPACE_RESERVE_BYTES:
            return {"status": "failed", "reason": "insufficient_disk_for_audio_evidence",
                    "reserve_bytes": AUDIO_WORKSPACE_RESERVE_BYTES,
                    "available_bytes": remaining_bytes,
                    "measurement": measurement}
        encode = [ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "f32le",
                  "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1", "-i", str(raw_path),
                  "-c:a", "aac", "-b:a", "32k", "-ac", "1", str(audio_path)]
        try:
            encode_code, encode_detail = _audio_child(
                encode, None, encode_log, cancelled=cancelled,
                timeout=AUDIO_ENCODE_TIMEOUT_SECONDS)
        except OSError as exc:
            return {"status": "failed", "reason": f"acoustic_evidence_encoder_failed:{type(exc).__name__}",
                    "measurement": measurement}
        except TimeoutError:
            return {"status": "failed", "reason": "acoustic_evidence_encode_timeout",
                    "measurement": measurement}
        if encode_code != 0 or not audio_path.is_file() or audio_path.stat().st_size == 0:
            return {"status": "failed", "reason": "acoustic_evidence_encode_failed",
                    "detail": encode_detail[-300:], "measurement": measurement}
        retain_audio = True
        return {"status": "success", "reason": None,
                "audio_relative_path": "audio/acoustic.m4a",
                "audio_sha256": stream_sha256(audio_path),
                "stream_selection": "0:a:0 (first audio stream; video stream ignored)",
                "sample_rate": AUDIO_SAMPLE_RATE, "samples": int(samples.size),
                "requested_window_seconds": {"start": start,
                                             "end": start + requested_duration},
                "window_seconds": {"start": start, "end": start + actual_duration},
                "coverage": coverage,
                "measurement": measurement}
    finally:
        if 'samples' in locals():
            mmap_handle = getattr(samples, "_mmap", None)
            del samples
            if mmap_handle is not None:
                mmap_handle.close()
        for temporary in (raw_path, decode_log, encode_log):
            temporary.unlink(missing_ok=True)
        if not retain_audio:
            audio_path.unlink(missing_ok=True)


def _shot_framing(frame: Any, face_result: Dict[str, Any],
                  person_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Use measured person xyxy geometry, with an explicit face-only fallback."""
    persons = person_result or {}
    if persons.get("status") == "success" and persons.get("persons_detected"):
        height, width = frame.shape[:2]
        result = classify_framing([item["box"] for item in persons["persons_detected"]],
                                  (width, height), subject_kind="person")
        result["status"] = "success"
        return result
    if face_result.get("status") != "success":
        return {"status": "not_run", "framing": None, "reason": "no_available_subject_boxes"}
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


def _run_person(frame: Any, extractor: Optional[PersonExtractor]) -> Dict[str, Any]:
    if extractor is None:
        return {"status": "not_run", "reason": "person_weights_unavailable", "persons_detected": []}
    try:
        return extractor.detect_people(frame)
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)[:300], "persons_detected": []}


def _expressive_result(measured: Dict[str, Any], keyframe: Dict[str, Any]) -> Dict[str, Any]:
    evaluated = [face_cue_candidates(face, frame_index=keyframe["frame_index"],
                                     evidence_id=keyframe["evidence_sha256"])
                 for face in measured.get("faces_detected", [])]
    cues = [cue for result in evaluated for cue in result.get("cues", [])]
    usable = any(result.get("status") == "success" for result in evaluated)
    status = "success" if usable else "failed" if measured["status"] == "failed" else "not_run"
    reason = (None if usable else measured.get("reason") or
              next((result.get("reason") for result in evaluated if result.get("reason")),
                   "no_faces_with_usable_landmarks"))
    return {"status": status, "reason": reason, "cues": cues, "face_results": evaluated,
            "scope": "sampled_frame_geometry_not_emotion_or_blink_duration"}


def _boundary_sample(item: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    """Release the decoded full frame immediately; retain only small features.

    Overlapping boundary windows can share these read-only measurements. The
    original decoded image and its evidence digest remain unchanged on disk.
    """
    return {"frame_index": item["frame_index"], "seconds": seconds,
            "source_pts": item["source_pts"], "proxy_pts": item["proxy_pts"],
            **frame_features(_load_frame(Path(item["path"])))}


def _attach_global_transitions(store, media_id, work, keyframes, scenes,
                               boundary_frames, window, ffmpeg, cancelled):
    """Project retained full-range proposals into existing scene candidates."""
    result = TD.scan_transitions(store, media_id, work / "transitions-global",
                                 window=window, ffmpeg=ffmpeg, cancelled=cancelled)
    result["measurement_relative_path"] = Path(result.pop("measurement_path")).relative_to(store.root).as_posix()
    selected, omitted, per_scene = [], [], {}
    existing = {str(frame["frame_index"]): frame for frame in boundary_frames}
    first_frames = {}
    for frame in keyframes:
        first_frames.setdefault(frame["scene_index"], frame)
    for event in result["events"]:
        anchor = store._frame_service.row(media_id, int(event["anchor_frame"]))
        seconds = float(S.rational_value(anchor["seconds"]))
        scene_index = next((i for i, scene in enumerate(scenes) if scene["start"] <= seconds < scene["end"]), None)
        if scene_index not in first_frames:
            omitted.append({"anchor_frame": event["anchor_frame"], "reason": "no_scene_keyframe"})
            continue
        first = first_frames[scene_index]
        interval = event.get("interval") or {}
        left = int(interval.get("start_frame", event["anchor_frame"]))
        right = int(interval.get("end_frame", event["anchor_frame"]))
        overlapping = [old for old in first.get("boundary_events", [])
            if old.get("kind") == event["kind"]
            and int((old.get("interval") or {}).get("start_frame", old["anchor_frame"])) <= right
            and int((old.get("interval") or {}).get("end_frame", old["anchor_frame"])) >= left]
        covered = any(int((old.get("interval") or {}).get("start_frame", old["anchor_frame"])) <= left
            and int((old.get("interval") or {}).get("end_frame", old["anchor_frame"])) >= right
            for old in overlapping)
        if covered:
            omitted.append({"anchor_frame": event["anchor_frame"], "reason": "covered_by_local_event"})
            continue
        if overlapping:
            event["metrics"]["overlapping_local_anchors"] = [old["anchor_frame"] for old in overlapping]
            event["uncertainty"] += "；与局部转场候选重叠，请核对完整区间后归并"
        if per_scene.get(scene_index, 0) >= 4:
            omitted.append({"anchor_frame": event["anchor_frame"], "reason": "scene_claim_projection_budget"})
            continue
        indices = {int(event["anchor_frame"]), left, right}
        if event.get("previous_frame") is not None:
            indices.add(int(event["previous_frame"]))
        if event.get("recovery_frame") is not None:
            indices.add(int(event["recovery_frame"]))
        if len(boundary_frames) + sum(len(item[2]) for item in selected) + len(indices) > BOUNDARY_MAX_FRAMES:
            omitted.append({"anchor_frame": event["anchor_frame"], "reason": "boundary_evidence_budget"})
            continue
        selected.append((scene_index, event, indices))
        per_scene[scene_index] = per_scene.get(scene_index, 0) + 1
    wanted = sorted({index for _, _, indices in selected for index in indices if str(index) not in existing})
    decoded = {}
    if wanted:
        plan = VS.plan_samples(store, media_id, window=window, uniform=0, include_edges=False,
            boundary_frames=wanted, budget=VS.SamplingBudget(max_frames=BOUNDARY_MAX_FRAMES, boundary_radius=0))
        images = VS.decode_samples(store, media_id, plan, work / "transitions-evidence",
                                   cancelled=cancelled, ffmpeg=ffmpeg)
        decoded = {item["frame_index"]: item for item in images["decoded"]}
    attached = {(frame["scene_index"], str(frame["frame_index"])) for frame in boundary_frames}
    for scene_index, event, indices in selected:
        for index in sorted(indices):
            identity = str(index)
            if (scene_index, identity) in attached:
                continue
            image = existing.get(identity)
            if image is None:
                sample = decoded[identity]
                image = {"frame_index": identity, "seconds": float(S.rational_value(sample["seconds"])),
                         "evidence_sha256": sample["sha256"]}
            boundary_frames.append({"scene_index": scene_index, "scene_indices": [scene_index],
                "frame_index": identity, "seconds": image["seconds"], "evidence_sha256": image["evidence_sha256"]})
            attached.add((scene_index, identity))
        first_frames[scene_index].setdefault("boundary_events", []).append(event)
    result.update(projected_events=len(selected), projection_omitted=omitted,
                  projection_limit_per_scene=4)
    return result


def _run_ocr(frame: Any, ocr_extractor: Optional[OCRExtractor]) -> Dict[str, Any]:
    if ocr_extractor is None:
        return {"status": "not_run", "reason": "ocr_runtime_unavailable"}
    try:
        result = ocr_extractor.extract_text(frame)
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    result.setdefault("status", "not_run" if ocr_extractor.engine is None else "success")
    if result["status"] == "not_run":
        result.setdefault("reason", "ocr_runtime_unavailable")
    return result


def _limit_global_claims(candidate):
    """Retain prior scene claims and fit new proposals to the semantic contract."""
    discovery = (candidate.get("boundary_sampling") or {}).get("global_discovery")
    if not discovery or discovery.get("status") != "success":
        return
    for index, scene in enumerate(candidate["scenes"]):
        frame = next((f for f in candidate["keyframes"] if f["scene_index"] == index), None)
        if frame is None:
            continue
        events = frame.get("boundary_events", [])
        global_events = [e for e in events if e.get("metrics", {}).get("analysis_scope") == "full_range"]
        while global_events:
            claims = _candidate_claims(candidate, index, scene, _seconds_ratio(scene["start"]),
                                       _seconds_ratio(scene["end"]), frame)["claims"]
            if len(claims) <= SS.MAX_CLAIMS:
                break
            removed = global_events.pop()
            events.remove(removed)
            discovery["projected_events"] -= 1
            discovery["projection_omitted"].append({"anchor_frame": removed["anchor_frame"],
                                                   "reason": "semantic_claim_budget"})


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


def _validate_loaded_deployment(extractors: Dict[str, Any],
                                deployment: Dict[str, Any]) -> None:
    """Bind loaded Face/OCR engines to the bytes frozen at submission."""
    face = extractors.get("face")
    face_identity = face.code_identity if face is not None else {}
    frozen_face = deployment["face"]
    if frozen_face["status"] == "available":
        if (face_identity.get("detector_kind") != "yunet"
                or face_identity.get("weights_sha256") != frozen_face.get("sha256")):
            raise WorkspaceError("VISION_RESUBMIT_REQUIRED",
                                 "Loaded face detector bytes differ from the frozen deployment")
    elif frozen_face["status"] == "builtin":
        if face_identity.get("detector_kind") != "haar":
            raise WorkspaceError("VISION_RESUBMIT_REQUIRED",
                                 "Loaded face detector differs from the frozen built-in detector")
    elif face_identity.get("detector_kind") is not None:
        raise WorkspaceError("VISION_RESUBMIT_REQUIRED",
                             "A face detector loaded although none was frozen")

    ocr = extractors.get("ocr")
    ocr_identity = ocr.code_identity if ocr is not None else {}
    frozen_ocr = {item["role"]: item.get("sha256")
                  for item in deployment["ocr"]["files"]
                  if item.get("status") == "available"}
    loaded_ocr = {item["role"]: item.get("sha256")
                  for item in ocr_identity.get("weights_files") or []}
    if loaded_ocr != frozen_ocr:
        raise WorkspaceError("VISION_RESUBMIT_REQUIRED",
                             "Loaded OCR resources differ from the frozen deployment")


def _run_isolated_model_phase(module: str, resources: Dict[str, Any],
                              images: List[Dict[str, Any]], work: Path,
                              relative: str, *, cancelled: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """Run one heavy module and retain an honest owned-process receipt.

    Resource and model-runtime failures are local to CLIP or VLM.  Integrity,
    cancellation, lease and protocol errors are intentionally not caught.
    """
    receipt = work / "models" / module / "execution" / "process-receipt.json"
    try:
        phase = MW.run_phase(module, resources, images, work, cancelled=cancelled)
    except WorkspaceError as exc:
        if exc.code not in ISOLATABLE_MODEL_ERRORS:
            raise
        if not receipt.is_file() or not 0 < receipt.stat().st_size <= 256 * 1024:
            raise WorkspaceError(
                "MODEL_WORKER_PROTOCOL",
                "Failed model phase did not retain a bounded process receipt") from exc
        try:
            execution = json.loads(receipt.read_text("utf-8"))
        except (OSError, ValueError) as receipt_error:
            raise WorkspaceError(
                "MODEL_WORKER_PROTOCOL",
                "Failed model process receipt cannot be decoded") from receipt_error
        if (not isinstance(execution, dict)
                or execution.get("status") != "failed"
                or execution.get("error_code") != exc.code):
            raise WorkspaceError(
                "MODEL_WORKER_PROTOCOL",
                "Failed model process receipt does not match the raised error") from exc
        failure = {"status": "failed", "reason": exc.code,
                   "message": exc.message[:240], "retryable": exc.retryable}
        if exc.details:
            failure["details"] = _jsonable(exc.details)
        phase = {
            "identity": MW.requested_identity(module, resources),
            "available": False,
            "results": [{"frame_index": image["frame_index"],
                         "path": image["path"],
                         "input_sha256": image["sha256"],
                         "result": dict(failure)}
                        for image in images],
            "execution": execution,
        }
    descriptor = None
    if receipt.is_file():
        descriptor = {
            "relative_path": relative + "/" + receipt.relative_to(work).as_posix(),
            "sha256": stream_sha256(receipt, cancelled=cancelled),
            "size_bytes": str(receipt.stat().st_size),
        }
    return phase, descriptor


class VisionJobs:
    """One optional drainer sharing durable leases; wrappers never write tasks.

    close() reports actual runner liveness. Queued jobs survive shutdown and
    resume on reopening. A successful job is a candidate, never a review.
    """

    def __init__(self, store: Any, *, queue: Optional[DurableQueue] = None,
                 closing: Optional[threading.Event] = None, poll_seconds: float = .1,
                 slot_wait_seconds: float = 900, face_model_path: Optional[str] = None,
                 clip_model_path: Optional[str] = None, vlm_model_path: Optional[str] = None,
                 person_model_path: Optional[str] = None, landmark_model_path: Optional[str] = None,
                 landmark_python: Optional[str] = None):
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
        self.person_model_path = person_model_path if person_model_path is not None else os.getenv("TRACE_ANNOTATE_PERSON_MODEL")
        self.landmark_model_path = landmark_model_path if landmark_model_path is not None else os.getenv("TRACE_ANNOTATE_LANDMARK_MODEL")
        self.landmark_python = landmark_python if landmark_python is not None else os.getenv("TRACE_ANNOTATE_LANDMARK_PYTHON")
        self._loaded_optional_config = None
        self._extractors = None
        self._extractor_error = None
        # CLIP/VLM are never constructed in the workspace process.  Keep this
        # invariant true even when tests or integrations replace the local
        # Face/OCR loader.
        self._isolated_heavy = True

    def _execution_config(self) -> Dict[str, Any]:
        return _execution_deployment(face_model_path=self.face_model_path)

    def _model_config(self):
        return MW.configuration(clip_model=self.clip_model_path, vlm_model=self.vlm_model_path)

    def _optional_config(self) -> Dict[str, Any]:
        return OV.provider_config(person_model=self.person_model_path or "",
                                  landmark_model=self.landmark_model_path or "",
                                  landmark_python=self.landmark_python or "")

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
            if getattr(instance, "released_worker", False):
                return "released" if instance.available else "unavailable"
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
            "person": adapter_state("person"),
            "landmarks": ("not_loaded" if self.landmark_model_path and self.landmark_python else "unavailable"),
        }
        return {"operation": OPERATION, "version": VERSION, "domain": DOMAIN,
                "capabilities": capabilities, "loaded": loaded,
                "extractor_error": self._extractor_error,
                "closing": self.is_closing()}

    def submit(self, task_id: str, *, command_id: str, expected_revision: int,
               expected_sha256: str, window: Optional[Dict[str, Any]] = None,
               subject_annotation_id: Optional[str] = None) -> Dict[str, Any]:
        S.require_id(command_id)
        parameters = _parameters(window, subject_annotation_id, optional_providers=self._optional_config(),
                                 model_resources=self._model_config(),
                                 execution_deployment=self._execution_config())
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
        parameters = _parameters(window, subject_annotation_id, optional_providers=self._optional_config(),
                                 model_resources=self._model_config(),
                                 execution_deployment=self._execution_config())
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
        optional = self._optional_config()
        if self._extractors is not None and self._loaded_optional_config == optional:
            return
        self._loaded_optional_config = optional
        # CLIP and VLM always run in owned child processes.  A local Face/OCR
        # constructor failure must not silently switch those modules back to
        # the legacy in-process path or omit their execution receipts.
        self._isolated_heavy = True
        try:
            face = FaceExtractor(model_path=self.face_model_path, thresholds=thresholds)
            ocr = OCRExtractor(thresholds=thresholds)
            person = PersonExtractor(model_path=optional["person"]["model_path"],
                                     expected_sha256=optional["person"]["weights_sha256"])
            self._extractors = {"face": face, "ocr": ocr, "clip": None, "vlm": None, "person": person}
        except Exception as exc:
            self._extractor_error = str(exc)
            self._extractors = {"face": None, "ocr": None, "clip": None, "vlm": None, "person": None}

    def _processor(self, job: Dict[str, Any], context: Any) -> Dict[str, Any]:
        parameters = _validate_job(job)
        resources = self._model_config()
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
                deployment = self._execution_config()
                if parameters["job_identity"] != _job_identity(
                        optional_providers=self._optional_config(), model_resources=resources,
                        execution_deployment=deployment):
                    raise WorkspaceError("VISION_RESUBMIT_REQUIRED",
                                         "Legacy or changed execution deployment; resubmit a new vision job")
                thresholds = MetricThresholds()
                self._load_extractors(thresholds)
                if parameters["job_identity"] != _job_identity(
                        optional_providers=self._optional_config(), model_resources=resources,
                        execution_deployment=deployment):
                    raise WorkspaceError("VISION_RESUBMIT_REQUIRED", "Optional provider configuration changed while loading; resubmit")
                extractors = self._extractors or {"face": None, "ocr": None, "clip": None}
                _validate_loaded_deployment(extractors, deployment)
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

                landmark_images = []
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
                        person_result = _run_person(frame, extractors.get("person"))
                        framing_result = _shot_framing(frame, face_result, person_result)
                        landmark_images.append({"path": str(item["path"]),
                                                "frame_index": item["frame_index"],
                                                "input_sha256": item["sha256"]})
                        ocr_result = _run_ocr(frame, extractors["ocr"])
                        clip_result = _run_clip(frame, extractors["clip"]) if not self._isolated_heavy else {}
                        vlm_result = _run_vlm(frame, extractors["vlm"]) if not self._isolated_heavy else {}
                        keyframes.append({
                            "scene_index": scene_index,
                            "frame_index": item["frame_index"],
                            "source_pts": item["source_pts"],
                            "proxy_pts": item["proxy_pts"],
                            "roles": item["roles"],
                            "timestamp": _seconds_ratio(seconds),
                            "evidence_sha256": item["sha256"],
                            "face": face_result,
                            "person": person_result,
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
                model_execution: Dict[str, Any] = {}
                if self._isolated_heavy and keyframes:
                    decoded_by_index = {item["frame_index"]: item for item in decoded["decoded"]}
                    images = [{"frame_index": item["frame_index"],
                               "path": str(decoded_by_index[item["frame_index"]]["path"]),
                               "sha256": item["evidence_sha256"]} for item in keyframes]
                    for module in ("clip", "vlm"):
                        phase, receipt_descriptor = _run_isolated_model_phase(
                            module, resources, images, work, relative, cancelled=cancelled)
                        model_execution[module] = {
                            "status": (phase.get("execution") or {}).get("status", "unknown"),
                            "execution": _jsonable(phase.get("execution")),
                            "worker": _jsonable({key: phase.get(key) for key in (
                                "actual_pid", "actual_parent_pid", "peak_working_set_bytes",
                                "thread_policy", "progress_publish_retries")}),
                            "receipt": receipt_descriptor,
                        }
                        for frame, result in zip(keyframes, phase["results"]):
                            frame[module] = result["result"]
                        # Only JSON identity survives. Native model allocations
                        # belong to the process already reaped by run_phase.
                        extractors[module] = SimpleNamespace(
                            code_identity=phase["identity"], model=None, released_worker=True,
                            available=phase["available"], task_prompt=phase["identity"].get("task_prompt", CAPTION_TASK))
                # Only filenames and small geometric results cross the worker
                # boundary; decoded images are not accumulated for MediaPipe.
                landmark_result = OV.run_landmarks(parameters["job_identity"]["config"]["optional_providers"],
                                                    landmark_images, work, cancelled=cancelled)
                landmark_frames = {item["frame_index"]: item for item in landmark_result.get("results", [])}
                for keyframe in keyframes:
                    measured = landmark_frames.get(keyframe["frame_index"], {
                        "status": landmark_result["status"], "reason": landmark_result.get("reason"),
                        "faces_detected": []})
                    keyframe["landmarks"] = {key: value for key, value in measured.items()
                                             if key not in {"path", "cue_results", "claims"}}
                    keyframe["expressive"] = _expressive_result(measured, keyframe)
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
                    acoustics_sampling = _measure_audio(
                        Path(source), ffmpeg, work, window=window,
                        media_duration=job["media_snapshot"]["duration"], cancelled=cancelled)
                except WorkspaceError as exc:
                    if exc.code == "CANCELLED":
                        raise
                    acoustics_sampling = {"status": "failed", "reason": exc.code,
                                          "message": str(exc)[:240], "measurement": None}
                except Exception as exc:
                    acoustics_sampling = {"status": "failed", "reason": type(exc).__name__,
                                          "message": str(exc)[:240], "measurement": None}
                boundary_frames: List[Dict[str, Any]] = []
                boundary_sampling: Dict[str, Any] = {
                    "status": "not_run",
                    "reason": "no_scene_boundary_in_window",
                    "global_discovery": {"status": "not_run", "reason": "scene_detection_unavailable_or_empty",
                                         "coverage": {"complete": False}},
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
                            sample = _boundary_sample(item, seconds)
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
                            if any(event.get("kind") != "cut" for event in events):
                                confirmation.update(needs_review=True,
                                    uncertainty="结构边界与转场候选重叠，请逐帧核对")
                            final_events[scene_index] = events + [confirmation]
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
                if scene_result["status"] == "success" and scenes:
                    try:
                        discovery = _attach_global_transitions(
                            self.store, job["media_snapshot"]["id"], work, keyframes,
                            scenes, boundary_frames, window, ffmpeg, cancelled)
                        boundary_sampling["global_discovery"] = discovery
                        if boundary_sampling["status"] == "not_run":
                            boundary_sampling.update(status="success", reason=None)
                    except WorkspaceError as exc:
                        if exc.code in {"CANCELLED", "LEASE_LOST"}:
                            raise
                        boundary_sampling["global_discovery"] = {"status": "failed", "error": exc.code,
                            "message": str(exc)[:240], "coverage": {"complete": False}}
                        boundary_sampling.update(local_status=boundary_sampling["status"], status="failed",
                                                 error=exc.code, message=str(exc)[:240])
                    except Exception as exc:
                        boundary_sampling["global_discovery"] = {"status": "failed", "error": type(exc).__name__,
                            "message": str(exc)[:240], "coverage": {"complete": False}}
                        boundary_sampling.update(local_status=boundary_sampling["status"], status="failed",
                                                 error=type(exc).__name__, message=str(exc)[:240])
                extractor_status = {
                    "scene": scene_result["status"],
                    "face": _aggregate_status(keyframes, "face"),
                    "ocr": _aggregate_status(keyframes, "ocr"),
                    "clip": _aggregate_status(keyframes, "clip"),
                    "vlm": _aggregate_status(keyframes, "vlm"),
                    "person": _aggregate_status(keyframes, "person"),
                    "landmarks": _aggregate_status(keyframes, "landmarks"),
                }
                motion_status = "success" if any(
                    frame.get("motion", {}).get("status") == "success" for frame in keyframes) else "not_run"
                module_status = dict(extractor_status)
                module_status.update({
                    "boundary": boundary_sampling.get("status", "not_run"),
                    "framing": _aggregate_status(keyframes, "framing"),
                    "motion": motion_status,
                    "transition": ("failed" if boundary_sampling.get("global_discovery", {}).get("status") == "failed"
                                   else "success" if any(frame.get("boundary_events") for frame in keyframes)
                                   else boundary_sampling.get("status", "not_run")),
                    "actions": actions_sampling.get("status", "not_run"),
                    "content": content_sampling.get("status", "not_run"),
                    "expressive": _aggregate_status(keyframes, "expressive"),
                    "acoustics": acoustics_sampling.get("status", "not_run"),
                })
                module_status["actions_reason"] = actions_sampling.get("reason")
                module_status["actions_supported"] = actions_sampling.get("supported")
                module_status["actions_unsupported"] = actions_sampling.get("unsupported")
                module_status["expressive_reason"] = _first_module_reason(keyframes, "expressive") or landmark_result.get("reason")
                module_status["content_reason"] = (content_sampling.get("message")
                                                   or content_sampling.get("error"))
                module_status["acoustics_reason"] = acoustics_sampling.get("reason")
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
                    "boundary_frames": boundary_frames, "boundary_sampling": boundary_sampling,
                    "content_sampling": content_sampling,
                    "content_frames": content_frames,
                    "actions_sampling": _jsonable(actions_sampling),
                    "acoustics_sampling": _jsonable(acoustics_sampling),
                    "extractor_status": extractor_status, "module_status": module_status,
                    "model_runs": [],
                    "model_execution": model_execution,
                    "job_identity": parameters["job_identity"], "evidence": evidence,
                    "semantic_listening": "not_performed", "source_unchanged": True,
                    "scope": "registered_playback_only; original_source_not_measured; candidate_not_applied"}
                _limit_global_claims(candidate)
                scene_counts = {str(index): len(frame.get("boundary_events", []))
                    for index in range(len(scenes))
                    for frame in keyframes if frame["scene_index"] == index and frame.get("boundary_events")}
                boundary_sampling.update(scene_events=scene_counts,
                    scenes_with_events=sorted(int(index) for index in scene_counts),
                    event_count=sum(scene_counts.values()))
                candidate["model_runs"] = _build_model_runs(
                    job, extractors, keyframes, scene_result,
                    boundary_sampling, boundary_frames, module_status,
                    digest, window, actions_sampling, acoustics_sampling,
                    content_sampling, content_frames, model_execution)
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
            stopped = not self._runners and not self._heartbeats and not (self._drainer is not None and self._drainer.is_alive())
            if stopped:
                self._extractors = None
                self._loaded_optional_config = None
            return {"stopped": stopped,
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
                or set(value) - expected - {"sampling", "content_frames", "model_execution"}
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
        requires_execution = "execution_deployment" in parameters["job_identity"]["config"]
        if requires_execution and "model_execution" not in value:
            raise WorkspaceError("RECOVERY_REQUIRED", "Current vision contract requires model execution receipts")
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
        optional = parameters["job_identity"]["config"].get("optional_providers")
        if optional is not None and any(_module_run_id(job["id"], name) not in run_ids
                                        for name in ("person", "landmarks", "expressive")):
            raise WorkspaceError("RECOVERY_REQUIRED", "Current vision contract requires optional provider status records")
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
        model_execution = value.get("model_execution", {})
        if (not isinstance(model_execution, dict)
                or any(module not in {"clip", "vlm"} for module in model_execution)):
            raise WorkspaceError("RECOVERY_REQUIRED", "Vision model execution descriptor is inconsistent")
        for module, execution in model_execution.items():
            if (not isinstance(execution, dict)
                    or set(execution) != {"status", "execution", "worker", "receipt"}
                    or execution["status"] not in {"completed", "failed", "not_run"}
                    or not isinstance(execution["execution"], dict)
                    or not isinstance(execution["worker"], dict)):
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision model execution receipt is invalid")
            receipt = execution["receipt"]
            if receipt is not None:
                if (not isinstance(receipt, dict)
                        or set(receipt) != {"relative_path", "sha256", "size_bytes"}
                        or receipt not in value["evidence"]):
                    raise WorkspaceError("RECOVERY_REQUIRED", "Vision process receipt is not registered evidence")
            inner_status = execution["execution"].get("status")
            if inner_status != execution["status"]:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision model execution status is inconsistent")
            if execution["status"] == "not_run" and receipt is not None:
                raise WorkspaceError("RECOVERY_REQUIRED", "A not-run model phase cannot have a process receipt")
            if execution["status"] in {"completed", "failed"} and receipt is None:
                raise WorkspaceError("RECOVERY_REQUIRED", "An executed model phase requires a process receipt")
            if (execution["status"] == "failed"
                    and not isinstance(execution["execution"].get("error_code"), str)):
                raise WorkspaceError("RECOVERY_REQUIRED", "A failed model phase requires a stable error code")
        if requires_execution and value["keyframes"] and set(model_execution) != {"clip", "vlm"}:
            raise WorkspaceError("RECOVERY_REQUIRED", "Current vision contract requires both model phase outcomes")
        for module in ("clip", "vlm"):
            run = next((item for item in model_runs
                        if item["id"] == _module_run_id(job["id"], module)), None)
            descriptor = model_execution.get(module)
            expected_execution = S.sha256_text(descriptor) if descriptor is not None else None
            if run is not None and run["parameters"].get("execution_sha256") != expected_execution:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision model run execution binding is inconsistent")
            expected_receipt = ((descriptor or {}).get("receipt") or {}).get("sha256")
            if run is not None and run["parameters"].get("process_receipt_sha256") != expected_receipt:
                raise WorkspaceError("RECOVERY_REQUIRED", "Vision model run receipt binding is inconsistent")
        discovery = (value.get("boundary_sampling") or {}).get("global_discovery")
        if "transition_discovery" in parameters["job_identity"]["config"]:
            if not isinstance(discovery, dict) or discovery.get("status") not in {"success", "failed", "not_run"}:
                raise WorkspaceError("RECOVERY_REQUIRED", "Current vision contract requires a full-range scan status")
            if discovery["status"] == "not_run" and value["extractor_status"].get("scene") == "success" and value["scenes"]:
                raise WorkspaceError("RECOVERY_REQUIRED", "Full-range scan was omitted despite available scenes")
            if discovery["status"] == "success" and (
                    discovery.get("contract") != TD.CONTRACT
                    or discovery.get("configuration") != parameters["job_identity"]["config"]["transition_discovery"]
                    or discovery.get("measurement_sha256") not in evidence_hashes):
                raise WorkspaceError("RECOVERY_REQUIRED", "Full-range transition evidence identity is inconsistent")
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
            S.require_id(command_id)
            S.require_hash(expected_sha256)
            if type(expected_revision) is not int or not 0 <= expected_revision <= S.MAX_SAFE_INTEGER:
                raise SchemaInvalid("SCHEMA_INVALID", "Expected revision must be a nonnegative safe integer")
            if not isinstance(request_body, bytes):
                raise SchemaInvalid("SCHEMA_INVALID", "Invalid command payload")
            result = self.result(job_id)
            receipt = self.store.conn.execute(
                "SELECT response FROM command_receipts WHERE command_id=?", (command_id,)).fetchone()
            if receipt:
                # Replay the payload that was actually committed, even when a
                # later mapper fixes projection rules or the task is edited.
                response = json.loads(receipt[0]) if receipt[0] else {}
                row = self.store.conn.execute(
                    "SELECT body FROM task_revisions WHERE task_id=? AND revision=?",
                    (result["task_id"], response.get("revision"))).fetchone()
                if row is None:
                    raise Conflict("IDEMPOTENCY_CONFLICT", "Command does not identify this candidate import")
                historical = json.loads(row[0])
                saved = {annotation["id"]: annotation for annotation in historical["annotations"]
                         if annotation.get("provenance", {}).get("job_id") == result["job_id"]}
                identifiers = [S.make_id("ann", result["job_id"] + ":" + str(index))
                               for index in range(len(result["scenes"]))]
                if any(identifier not in saved for identifier in identifiers):
                    raise Conflict("IDEMPOTENCY_CONFLICT", "Command does not identify this candidate import")
                annotations = [saved[identifier] for identifier in identifiers]
                return self.store.apply_task_command(command_id=command_id, task_id=result["task_id"],
                    expected_revision=expected_revision, expected_sha256=expected_sha256,
                    operation="import_vision_candidates", payload={"candidates": annotations,
                    "source_job_id": result["job_id"],
                    "model_runs": result.get("model_runs") or []}, request_body=request_body)["response"]
            current = self.store.get_task(result["task_id"])
            if current["revision"] != expected_revision:
                raise revision_conflict(expected_revision, current["revision"])
            if current["content_sha256"] != expected_sha256 or self.store.task_digest(current) != expected_sha256:
                raise Conflict("MEDIA_CHANGED", "Task content hash mismatch",
                               details={"expected_sha256": expected_sha256,
                                        "current_sha256": current["content_sha256"]})
            media = self.queue.get_job(job_id)["media_snapshot"]
            annotations = _candidate_annotations(result, duration=media["duration"])
            template = next(value for value in self.store.list_templates()
                            if (value["id"], value["version"]) ==
                            (current["template_id"], current["template_version"]))
            model_runs = result.get("model_runs") or []
            # Validate before publishing thumbnails: rolling back SQLite cannot
            # remove immutable objects written for a rejected projection.
            for annotation in annotations:
                PS.validate_annotation(annotation, media, template,
                                       model_run_ids={run["id"] for run in model_runs})
            _attach_candidate_evidence(self.store, result, annotations)
            return apply_vision_candidates(
                store=self.store, command_id=command_id, task_id=result["task_id"],
                expected_revision=expected_revision, expected_sha256=expected_sha256,
                candidates=annotations, source_job_id=result["job_id"],
                model_runs=model_runs, request_body=request_body)


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
    statuses = [(frame.get(extractor) or {}).get("status", "not_run") for frame in keyframes]
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
    # The frozen semantic contract has execution states only.  A completed
    # algorithm may legitimately refuse a judgement or report incomplete
    # coverage; the detailed outcome remains in the digest-bound output.
    if value in {"success", "partial", "insufficient_samples",
                 "insufficient_evidence", "not_applicable"}:
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
    reported_status = str(status)
    normalized = _run_status(reported_status)
    recorded_parameters = _jsonable(parameters)
    if reported_status != normalized:
        recorded_parameters = dict(recorded_parameters,
                                   reported_outcome_status=reported_status)
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
        "parameters": recorded_parameters,
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
                      content_frames: Optional[List[Dict[str, Any]]] = None,
                      model_execution: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Record one real run per executed module, including honest non-success.

    Weights are hashed only when a local file exists. A failed or not-run
    adapter still gets a record so the export can distinguish "did not run"
    from "ran and produced this". Rule-based detectors are recorded too: their
    identity is the frozen threshold set, not a fabricated model weight.
    """
    job_id = job["id"]
    thresholds = MetricThresholds()
    frame_input = _keyframe_input(keyframes)
    model_execution = model_execution or {}
    runs: List[Dict[str, Any]] = []

    def extractor_identity(module: str, fallback: str) -> Dict[str, Any]:
        instance = extractors.get(module)
        if instance is None:
            return {"provider": "unavailable", "model_name": fallback,
                    "model_version": "unavailable", "weights_path": None,
                    "license": "not declared"}
        return instance.code_identity

    face_identity = extractor_identity("face", "YuNet")
    # FaceExtractor records the digest of the exact bytes supplied to OpenCV.
    # Re-reading the source path after inference could record replacement bytes.
    face_weights = face_identity.get("weights_sha256")
    if face_weights is None and face_identity.get("detector_kind") != "yunet":
        face_weights = _weights_digest(face_identity.get("weights_path"))
    known_yunet = face_weights == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    runs.append(_model_run_record(
        job_id, "face", provider=face_identity.get("provider", "opencv"),
        model_name=face_identity.get("model_name", "YuNet"),
        model_version=face_identity.get("model_version", "unknown"),
        weights_sha256=face_weights,
        license="MIT" if known_yunet else face_identity.get("license", "not declared"),
        preprocessing={"input": "BGR frame as decoded",
                       "detector_kind": face_identity.get("detector_kind"),
                       "detector_input": "decoded_frame_width_height" if face_identity.get("detector_kind") == "yunet" else "native",
                       "detector_initial_input": [320, 320] if face_identity.get("detector_kind") == "yunet" else None,
                       "runtime_license": "Apache-2.0 (OpenCV)" if face_identity.get("provider") == "opencv" else None,
                       "weights_license_source": ("https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE"
                                                  if known_yunet else None)},
        parameters={"face_min_confidence": thresholds.face_min_confidence},
        input_value=frame_input, output_value=_frame_output(keyframes, "face"),
        status=module_status.get("face", "not_run"),
        failure_reason=_first_module_reason(keyframes, "face")
        or ("no sampled keyframes" if not keyframes else None),
        code_version=f"vision_jobs={VERSION}; opencv={_package_version('opencv-python')}",
        score_kind="confidence"))

    optional = job["parameters"]["job_identity"]["config"].get("optional_providers")
    if optional is not None:
        person = optional["person"]
        runs.append(_model_run_record(
            job_id, "person", provider="onnxruntime-cpu",
            model_name=("NanoDet-m-plus-1.5x_416_2022nov-person" if person["weights_sha256"] ==
                        PERSON_WEIGHTS_SHA256 else "unverified_local_onnx"),
            model_version=person["runtime_version"], weights_sha256=person["weights_sha256"],
            license=("Apache-2.0 (OpenCV Zoo)" if person["weights_sha256"] ==
                     PERSON_WEIGHTS_SHA256 else "unverified local weights"),
            preprocessing={"input": "BGR letterbox 416x416", "output_coordinates": "decoded_frame_xyxy"},
            parameters={"class": "COCO person 0", "score_threshold": .5, "nms_iou": .5,
                        "max_persons": 100, "score_calibration": "not_established"},
            input_value=frame_input, output_value=_frame_output(keyframes, "person"),
            status=module_status.get("person", "not_run"),
            failure_reason=_first_module_reason(keyframes, "person") or "no sampled frames",
            code_version="optional-local-providers-v1:" + optional["implementation_sha256"],
            score_kind="confidence"))
        landmark = optional["landmarks"]
        runs.append(_model_run_record(
            job_id, "landmarks", provider="mediapipe-tasks-cpu", model_name="Face Landmarker / Face Mesh V2",
            model_version=landmark["runtime"]["mediapipe_version"],
            weights_sha256=landmark["weights_sha256"],
            license=("Apache-2.0" if landmark["weights_sha256"] ==
                     "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff" else "unverified local task"),
            preprocessing={"input": "RGB decoded keyframes", "mode": "IMAGE", "max_faces": 4,
                           "worker_python_sha256": landmark["worker_python_sha256"]},
            parameters={"timeout_seconds": optional["worker_timeout_seconds"], "offline": True},
            input_value=frame_input, output_value=_frame_output(keyframes, "landmarks"),
            status=module_status.get("landmarks", "not_run"),
            failure_reason=module_status.get("expressive_reason") or "no sampled frames",
            code_version="optional-local-providers-v1:" + optional["implementation_sha256"], score_kind="none"))
        runs.append(_model_run_record(
            job_id, "expressive", provider="trace-annotate", model_name="expressive-cues-v1",
            model_version="1", weights_sha256=None, license="Project source",
            preprocessing={"input": "measured face landmarks with geometric quality filters"},
            parameters={"min_face_height": 48, "min_frame_height_ratio": .06,
                        "eyes_narrowed_ear": .18, "eyes_closed_ear": .10,
                        "emotion_inference": False, "brow_neutral_baseline_available": False},
            input_value=_frame_output(keyframes, "landmarks"), output_value=_frame_output(keyframes, "expressive"),
            status=module_status.get("expressive", "not_run"),
            failure_reason=module_status.get("expressive_reason") or "no sampled frames",
            code_version="optional-local-providers-v1:" + optional["implementation_sha256"], score_kind="rule_score"))

    ocr_identity = extractor_identity("ocr", "RapidOCR")
    runs.append(_model_run_record(
        job_id, "ocr", provider=ocr_identity.get("provider", "RapidOCR"),
        model_name=ocr_identity.get("model_name", "RapidOCR default det+cls+rec"),
        model_version=ocr_identity.get("model_version", "unavailable"),
        weights_sha256=ocr_identity.get("weights_sha256") or _weights_digest(ocr_identity.get("weights_path")),
        license=ocr_identity.get("weights_license") or "unverified local weights",
        preprocessing={"input": "BGR frame as decoded", "orientation": "as-shot",
                       "runtime_backend": ocr_identity.get("runtime_backend"),
                       "code_license": ocr_identity.get("code_license"),
                       "weights_license": ocr_identity.get("weights_license"),
                       "weights_hash_policy": ocr_identity.get("weights_hash_policy"),
                       "weights_files": [{key: item[key] for key in ("role", "sha256", "size_bytes")}
                                         for item in ocr_identity.get("weights_files", [])]},
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
        model_version=clip_identity.get("selected_revision") or clip_identity.get("pretrained", "unresolved"),
        weights_sha256=clip_identity.get("weights_sha256") or _weights_digest(clip_identity.get("weights_path")),
        license=clip_identity.get("weights_license") or clip_identity.get("license", "not declared"),
        preprocessing={"input": "BGR frame converted to RGB",
                       "transform": "open_clip pretrained eval transform",
                       "offline": clip_identity.get("offline"),
                       "downloads_allowed": clip_identity.get("downloads_allowed"),
                       "runtime_license": clip_identity.get("runtime_license"),
                       "weights_license_source": clip_identity.get("weights_license_source"),
                       "weights_identity_status": clip_identity.get("weights_identity_status")},
        parameters={"clip_binary_confidence": thresholds.clip_binary_confidence,
                    "clip_multiclass_confidence": thresholds.clip_multiclass_confidence,
                    "clip_entity_mismatch_cutoff": thresholds.clip_entity_mismatch_cutoff,
                    "force_quick_gelu": clip_identity.get("force_quick_gelu"),
                    "execution_sha256": (S.sha256_text(model_execution["clip"])
                                         if "clip" in model_execution else None),
                    "process_receipt_sha256": ((model_execution.get("clip", {}).get("receipt") or {})
                                               .get("sha256"))},
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
        model_version=vlm_identity.get("selected_revision") or "unresolved",
        weights_sha256=vlm_identity.get("weights_sha256") or _weights_digest(vlm_identity.get("weights_path")),
        license=vlm_identity.get("weights_license") or vlm_identity.get("license", "not declared"),
        preprocessing={"input": "BGR frame converted to RGB PIL image",
                       "trust_remote_code": vlm_identity.get("trust_remote_code"),
                       "offline": vlm_identity.get("offline"),
                       "downloads_allowed": vlm_identity.get("downloads_allowed"),
                       "runtime_license": vlm_identity.get("runtime_license"),
                       "weights_license_source": vlm_identity.get("weights_license_source"),
                       "weights_identity_status": vlm_identity.get("weights_identity_status"),
                       "model_code_sha256": vlm_identity.get("model_code_sha256"),
                       "model_code_license": vlm_identity.get("model_code_license")},
        parameters={"task_prompt": getattr(extractors.get("vlm"), "task_prompt", CAPTION_TASK), "max_new_tokens": 192,
                    "num_beams": 3, "do_sample": False,
                    "requested_revision": vlm_identity.get("revision"),
                    "execution_sha256": (S.sha256_text(model_execution["vlm"])
                                         if "vlm" in model_execution else None),
                    "process_receipt_sha256": ((model_execution.get("vlm", {}).get("receipt") or {})
                                               .get("sha256"))},
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
        preprocessing={"input": "one-to-one person xyxy tracks; explicit face xywh fallback; per scene; max gap 5s",
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
        job_id, "acoustics", provider="trace-annotate", model_name="acoustic-window-v2",
        model_version="2", weights_sha256=None, license="Project source",
        preprocessing={"stream": acoustics.get("stream_selection"),
                       "sample_rate": acoustics.get("sample_rate"),
                       "channels": 1,
                       "sample_format": "float32 analysis; mono AAC evidence",
                       "requested_window": acoustics.get("requested_window_seconds"),
                       "analysed_window": acoustics.get("window_seconds"),
                       "coverage": acoustics.get("coverage")},
        parameters={"f0_min_hz": 60.0, "f0_max_hz": 500.0,
                    "frame_hop_seconds": 0.02, "silence_dbfs": -45.0,
                    "max_analysis_seconds": AUDIO_MAX_ANALYSIS_SECONDS,
                    "analysis_timeout_seconds": AUDIO_ANALYSIS_TIMEOUT_SECONDS,
                    "workspace_reserve_bytes": AUDIO_WORKSPACE_RESERVE_BYTES},
        input_value={"file_sha256": digest, "module": "acoustics",
                     "analysis_window": (acoustics.get("measurement") or {}).get("window_seconds")},
        output_value={"module": "acoustics",
                      "measurement": _jsonable(acoustics.get("measurement")),
                      "audio_sha256": acoustics.get("audio_sha256")},
        status=module_status.get("acoustics", "not_run"),
        failure_reason=acoustics.get("reason"),
        code_version=f"vision_jobs={VERSION}; expressive_cues=2; acoustic_window=2; ffmpeg={_package_version('static-ffmpeg')}",
        score_kind="rule_score"))

    runs.append(_model_run_record(
        job_id, "boundary", provider="trace-annotate", model_name="shot-boundaries-v1",
        model_version="1", weights_sha256=None, license="Project source",
        preprocessing={"input": "boundary-radius samples and optional full-range grayscale scan",
                       "analysis_width": 96, "hist_bins": 32},
        parameters={**boundary_parameters(), "boundary_radius": BOUNDARY_RADIUS,
                    "black_level": 10, "white_level": 248},
        input_value={"global_discovery": (boundary_sampling or {}).get("global_discovery"), "boundary_frames": [
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
        code_version=f"vision_jobs={VERSION}; shot_boundaries=1; full_range={TD.CONTRACT}",
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
                for key in ("previous_frame", "interval_start_frame", "interval_end_frame", "anchor_frame", "recovery_frame"):
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
        for claim in annotation["attributes"]["semantic"]["claims"]:
            if claim["method"]["id"] != "subject-translation-v1":
                continue
            for sample in claim["method"]["parameters"].get("sample_frames", []):
                match = next((frame for frame in keyframes
                              if frame["scene_index"] == index
                              and str(frame["frame_index"]) == str(sample.get("frame_index"))
                              and frame.get("evidence_sha256") == sample.get("evidence_sha256")), None)
                if match is None:
                    raise WorkspaceError("VISION_EVIDENCE_UNAVAILABLE",
                                         "Action candidate has no matching decoded subject sample")
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
                           if claim["method"]["id"] in {"acoustic-window-v1", "acoustic-window-v2"}]
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
                for key in ("previous_frame", "interval_start_frame", "interval_end_frame", "anchor_frame", "recovery_frame"):
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
            elif claim["method"]["id"] in {"keyframe-sampling-local", "shot-framing-local", "person-box-local", "expressive-cues-v1"}:
                frame_index = str(claim["method"]["parameters"]["frame_index"])
                if frame_index in frame_artifacts:
                    claim["evidence_ids"] = [frame_artifacts[frame_index]]
            elif claim["method"]["id"] == "scene-change-v1":
                claim["evidence_ids"] = list(dict.fromkeys(
                    sample_artifacts[(sample["frame_index"], sample["evidence_sha256"])]
                    for sample in claim["method"]["parameters"].get("sample_frames", [])))
            elif claim["method"]["id"] == "subject-translation-v1":
                claim["evidence_ids"] = list(dict.fromkeys(
                    sample_artifacts[(str(sample["frame_index"]), sample["evidence_sha256"])]
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
    exact = event.get("time_window")
    if exact is not None:
        try:
            start = exact["start"]["seconds"]
            end = exact["end"]["seconds"]
            if S.rational(start) == start and S.rational(end) == end and S.rational_value(start) < S.rational_value(end):
                return exact, True
        except (KeyError, TypeError, ValueError, WorkspaceError):
            pass
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
        person = keyframe.get("person") or {}
        if person.get("status") == "success":
            count = len(person.get("persons_detected") or [])
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:persons"), "claim_type": "observable",
                "statement": f"关键帧 #{keyframe['frame_index']} 的人物检测器返回 {count} 个人物框候选。",
                "subject_ref": None, "time_window": window, "evidence_ids": [],
                "method": {"id": "person-box-local", "version": "1", "parameters": {
                    "frame_index": keyframe["frame_index"], "coordinate_space": "decoded_frame_xyxy",
                    "count": count, "score_calibration": "not_established"}},
                "model_run_id": _module_run_id(job_id, "person"), "score": None,
                "score_kind": "none", "measurement": None,
                "uncertainty_reason": "人物框可能误检雕像或屏幕人物；零检测不能证明画面无人，未校准检测准确率"})
        # Like framing, portable segment claims describe the representative
        # keyframe; every sampled frame's raw cues remain in the candidate.
        expressive = keyframe.get("expressive") or {}
        for number, cue in enumerate(expressive.get("cues") or []):
            basic = cues_to_claims([cue])[0]
            basic.update({"id": S.make_id("clm", f"{job_id}:{index}:expressive:{number}"),
                          "subject_ref": None, "time_window": window, "evidence_ids": [],
                          "measurement": None, "model_run_id": _module_run_id(job_id, "expressive"),
                          "method": {"id": "expressive-cues-v1", "version": "1", "parameters": {
                              "frame_index": keyframe["frame_index"], "kind": cue["kind"],
                              "metrics": cue.get("metrics"),
                              "scope": "representative_frame_geometry_not_emotion_or_blink_duration",
                              "landmark_model_run_id": _module_run_id(job_id, "landmarks")}}})
            claims.append(basic)
        framing = keyframe.get("framing") or {}
        if (framing.get("status") == "success"
                and framing.get("framing") in {"远景", "全景", "中景", "近景", "特写"}):
            measurements = framing.get("measurements") or []
            ratio = measurements[0].get("height_ratio") if measurements else None
            claims.append({
                "id": S.make_id("clm", f"{job_id}:{index}:framing"),
                "claim_type": "measured" if isinstance(ratio, (int, float)) else "observable",
                "statement": (f"按关键帧 #{keyframe['frame_index']} 的{'人物' if framing.get('subject_kind') == 'person' else '人脸'}框高度占比"
                              f"测得景别候选为{framing['framing']}（几何规则，需复核）。"),
                "subject_ref": None,
                "time_window": window,
                "evidence_ids": [],
                "method": {"id": "shot-framing-local", "version": "1",
                           "parameters": {"framing": framing["framing"],
                                          "subject_kind": framing.get("subject_kind", "face"),
                                          "subject_height_ratio": ratio,
                                          "frame_index": keyframe["frame_index"]}},
                "model_run_id": _module_run_id(job_id, "person" if framing.get("subject_kind") == "person" else "face"),
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
            if (not math.isfinite(left) or not math.isfinite(right)
                    or max(left, float(scene["start"])) >= min(right, float(scene["end"]))):
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
                                          "sample_frames": action.get("sample_frames") or [],
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
                              f"声能活动占比 {measurement.get('energy_active_ratio')}，"
                              "仅声学测量，不代表音色、身份或情绪。"),
                "subject_ref": None,
                "time_window": {"start": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("start", 0.0)))},
                                "end": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("end", 0.0)))}},
                "evidence_ids": [],
                "method": {"id": "acoustic-window-v2", "version": "2",
                           "parameters": {"stream_selection": acoustics.get("stream_selection"),
                                          "sample_rate": acoustics.get("sample_rate"),
                                          "f0_min_hz": 60.0, "f0_max_hz": 500.0,
                                          "frame_hop_seconds": 0.02,
                                          "f0_hz": measurement.get("f0_hz"),
                                          "f0_variation_hz": measurement.get("f0_variation_hz"),
                                          "activity_kind": measurement.get("activity_kind"),
                                          "active_ratio": measurement.get("active_ratio"),
                                          "active_intervals": measurement.get("active_intervals"),
                                          "energy_active_ratio": measurement.get("energy_active_ratio"),
                                          "energy_active_intervals": measurement.get("energy_active_intervals"),
                                          "coverage": acoustics.get("coverage")}},
                "model_run_id": _module_run_id(job_id, "acoustics"),
                "score": 1.0,
                "score_kind": "confidence",
                "measurement": {"value": measurement.get("rms_dbfs"),
                                "unit": "dBFS", "basis": "measured",
                                "analysis_window": {"start": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("start", 0.0)))},
                                                    "end": {"seconds": _seconds_ratio(float((measurement.get("window_seconds") or {}).get("end", 0.0)))}}},
                "uncertainty_reason": (
                    "仅声学测量，不代表语音、音色、身份或情绪；活动区间只表示能量高于自适应阈值；"
                    + ("请求窗口超过计算预算，仅测量并绑定已声明的前缀窗口"
                       if (acoustics.get("coverage") or {}).get("truncated")
                       else "已测量完整请求窗口")),
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
            analysis_description = ("全域低分辨率扫描候选，需逐帧复核"
                if event.get("metrics", {}).get("analysis_scope") == "full_range"
                else "边界前后抽样帧，需逐帧复核")
            if interval.get("start_frame") is not None and interval.get("end_frame") is not None:
                statement = (f"镜头边界检测判定 #{interval.get('start_frame')}–"
                             f"#{interval.get('end_frame')} 帧区间内{statements[kind]}"
                             f"（代表帧 #{event.get('anchor_frame')}，{analysis_description}）。")
            else:
                statement = (f"镜头边界检测在 #{event.get('anchor_frame')} 帧判定"
                             f"{statements[kind]}（{analysis_description}）。")
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
                                          "recovery_frame": event.get("recovery_frame"),
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


def _candidate_annotations(candidate: Dict[str, Any], *,
                           duration: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
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
        if duration is not None:
            start = max((S.rational(0), start), key=S.rational_value)
            start, end = _content_clip(start, end, duration)
            if not _content_overlap(start, end):
                raise SchemaInvalid("VISION_CANDIDATE_OUTSIDE_MEDIA",
                                    "Scene candidate has no interval inside the registered playback")
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
                if (math.isfinite(left) and math.isfinite(right)
                        and max(left, float(scene["start"])) < min(right, float(scene["end"]))):
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
