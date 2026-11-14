"""Offline MediaPipe Hand Landmarker observations, never grasp/action claims.

The adapter accepts only the exact official ``float16/1`` task bundle frozen
below.  It performs no network lookup and uses IMAGE mode so every result is
bound to one caller-supplied decoded frame identity.  The 21 landmarks and raw
handedness category are observations only: they do not establish a persistent
person identity, touch, grasp, support state, intent, or an action.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import ctypes
from ctypes import wintypes
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaseExtractor, safe_failure_reason


CODE_VERSION = "hand-landmarks-mediapipe-worker-v2"
PROVIDER = "mediapipe-hand-landmarker-float16-v1"
OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
OFFICIAL_MODEL_CARD = (
    "https://storage.googleapis.com/mediapipe-assets/"
    "Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf"
)
OFFICIAL_MODEL_SIZE = 7_819_105
OFFICIAL_WEIGHTS_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_LICENSE = "Apache-2.0"
MODEL_LICENSE_SOURCE = OFFICIAL_MODEL_CARD
RUNTIME_LICENSE_SOURCE = "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE"
LANDMARK_NAMES: Tuple[str, ...] = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_FRAME_PIXELS = 16 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_ABS_LANDMARK_COORDINATE = 1_000_000.0
MAX_WORKER_HEADER_BYTES = 16 * 1024
MAX_WORKER_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_WORKER_STDERR_BYTES = 1024 * 1024
WORKER_TIMEOUT_SECONDS = 30.0
WORKER_POLL_SECONDS = 0.025
WORKER_MODULE = "trace_annotate.extractors.hands_worker"
CANCELLATION_BOUNDARY = "owned_subprocess_polled_during_inference"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,95}$")
_SAFE_WORKER_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+!-]{0,199}$")


class _WorkerIsolationUnavailable(RuntimeError):
    """The platform could not establish the promised owned-process boundary."""


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJobApi:
    """Small injectable kernel32 adapter used by the owned-worker boundary."""

    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._create = kernel32.CreateJobObjectW
        self._create.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self._create.restype = wintypes.HANDLE
        self._set = kernel32.SetInformationJobObject
        self._set.argtypes = (wintypes.HANDLE, ctypes.c_int,
                              ctypes.c_void_p, wintypes.DWORD)
        self._set.restype = wintypes.BOOL
        self._assign = kernel32.AssignProcessToJobObject
        self._assign.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._assign.restype = wintypes.BOOL
        self._terminate = kernel32.TerminateJobObject
        self._terminate.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self._terminate.restype = wintypes.BOOL
        self._close = kernel32.CloseHandle
        self._close.argtypes = (wintypes.HANDLE,)
        self._close.restype = wintypes.BOOL

    def create(self) -> int:
        return int(self._create(None, None) or 0)

    def set_kill_on_close(self, handle: int) -> bool:
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        return bool(self._set(
            handle, self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits), ctypes.sizeof(limits)))

    def assign(self, handle: int, process_handle: int) -> bool:
        return bool(self._assign(handle, process_handle))

    def terminate(self, handle: int, exit_code: int) -> bool:
        return bool(self._terminate(handle, exit_code))

    def close(self, handle: int) -> bool:
        return bool(self._close(handle))

    @staticmethod
    def last_error() -> int:
        return int(ctypes.get_last_error())


class _WindowsKillJob:
    """One process-tree owner whose final handle has kill-on-close semantics."""

    def __init__(self, handle: int, api: _WindowsJobApi):
        self._handle = handle
        self._api = api
        self._lock = threading.Lock()

    @classmethod
    def attach(cls, process: subprocess.Popen) -> "_WindowsKillJob":
        api = _WindowsJobApi()
        handle = api.create()
        if not handle:
            raise _WorkerIsolationUnavailable(
                f"hand_landmark_worker_job_create_failed_{api.last_error()}")
        if not api.set_kill_on_close(handle):
            error = api.last_error()
            api.close(handle)
            raise _WorkerIsolationUnavailable(
                f"hand_landmark_worker_job_limit_failed_{error}")
        process_handle = int(getattr(process, "_handle", 0) or 0)
        if not process_handle or not api.assign(handle, process_handle):
            error = api.last_error()
            api.close(handle)
            raise _WorkerIsolationUnavailable(
                f"hand_landmark_worker_job_assign_failed_{error}")
        return cls(handle, api)

    def terminate_and_close(self, exit_code: int = 1) -> bool:
        """Terminate the assigned tree and release the kill-on-close handle once."""
        with self._lock:
            handle, self._handle = self._handle, 0
        if not handle:
            return True
        terminated = self._api.terminate(handle, exit_code)
        closed = self._api.close(handle)
        return bool(terminated and closed)

    def close(self) -> bool:
        """Release the final handle; kill-on-close applies if the tree still lives."""
        with self._lock:
            handle, self._handle = self._handle, 0
        return True if not handle else bool(self._api.close(handle))


def _is_reparse_point(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & flag)


def _read_exact_model(path: Path) -> Tuple[bytes, str]:
    """Read one stable regular file; never follow or race a replaced model."""
    if isinstance(path, Path):
        for component in (path.absolute(), *path.absolute().parents):
            try:
                component_info = component.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(component_info.st_mode) or _is_reparse_point(component_info):
                raise ValueError("hand_landmark_model_path_contains_reparse_point")
    before = path.lstat()
    if (stat.S_ISLNK(before.st_mode) or _is_reparse_point(before)
            or not stat.S_ISREG(before.st_mode)):
        raise ValueError("hand_landmark_model_must_be_a_regular_file")
    if before.st_size != OFFICIAL_MODEL_SIZE or before.st_size > MAX_MODEL_BYTES:
        raise ValueError("hand_landmark_weights_size_mismatch")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
            raise ValueError("hand_landmark_weights_changed_during_read")
        data = stream.read(MAX_MODEL_BYTES + 1)
        after_read = os.fstat(stream.fileno())
    after_path = path.lstat()
    if (len(data) != OFFICIAL_MODEL_SIZE or len(data) > MAX_MODEL_BYTES
            or _is_reparse_point(after_path)
            or (after_read.st_dev, after_read.st_ino, after_read.st_size, after_read.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise ValueError("hand_landmark_weights_changed_during_read")
    return data, hashlib.sha256(data).hexdigest()


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded ASCII identifier")
    return value


def _frame(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("frame_index must be a canonical non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_index must be a canonical non-negative integer") from exc
    if parsed < 0 or (not isinstance(value, int) and str(parsed) != value):
        raise ValueError("frame_index must be a canonical non-negative integer")
    return str(parsed)


def _pts(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"n", "d"}:
        raise ValueError("pts must be a reduced rational")
    try:
        point = Fraction(int(value["n"]), int(value["d"]))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("pts must be a reduced rational") from exc
    canonical = {"n": str(point.numerator), "d": str(point.denominator)}
    if value != canonical:
        raise ValueError("pts must be a reduced rational")
    return canonical


def _point(value: Any, *, what: str) -> List[float]:
    try:
        point = [float(value.x), float(value.y), float(value.z)]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{what} must contain finite x/y/z") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{what} must contain finite x/y/z")
    if any(abs(item) > MAX_ABS_LANDMARK_COORDINATE for item in point):
        raise ValueError(f"{what} exceeds the bounded coordinate range")
    return point


def _category(value: Any) -> Tuple[str, float, Optional[int]]:
    name = getattr(value, "category_name", None)
    score = getattr(value, "score", None)
    index = getattr(value, "index", None)
    if name not in {"Left", "Right"}:
        raise ValueError("handedness category must be Left or Right")
    if (not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
        raise ValueError("handedness score must be finite and between zero and one")
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError("handedness index must be a non-negative integer or null")
    return name, float(score), index


def _observation_id(identity: Dict[str, Any]) -> str:
    encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    return "handobs-" + hashlib.sha256(encoded).hexdigest()[:40]


def _hand_observations(result: Any, *, width: int, height: int, scene_index: int,
                       frame_index: str, pts: Dict[str, str], evidence_sha256: str,
                       pixel_sha256: str, model_run_id: str,
                       max_hands: int) -> List[Dict[str, Any]]:
    landmarks = getattr(result, "hand_landmarks", None)
    world = getattr(result, "hand_world_landmarks", None)
    handedness = getattr(result, "handedness", None)
    if not isinstance(landmarks, Sequence) or not isinstance(world, Sequence) or not isinstance(handedness, Sequence):
        raise ValueError("hand landmarker result is missing its parallel output arrays")
    if not (len(landmarks) == len(world) == len(handedness)):
        raise ValueError("hand landmarker parallel output lengths differ")
    if len(landmarks) > max_hands:
        raise ValueError("hand landmarker returned more hands than configured")
    observations: List[Dict[str, Any]] = []
    for hand_index, (image_points, world_points, categories) in enumerate(
            zip(landmarks, world, handedness)):
        if (not isinstance(image_points, Sequence) or len(image_points) != 21
                or not isinstance(world_points, Sequence) or len(world_points) != 21
                or not isinstance(categories, Sequence) or not categories):
            raise ValueError("each hand needs 21 image points, 21 world points and handedness")
        normalized = [_point(item, what="normalized hand landmark") for item in image_points]
        world_m = [_point(item, what="world hand landmark") for item in world_points]
        name, score, category_index = _category(categories[0])
        xy = np.asarray([item[:2] for item in normalized], dtype=float)
        minimum, maximum = xy.min(axis=0), xy.max(axis=0)
        if np.allclose(minimum, maximum):
            raise ValueError("hand landmark geometry is degenerate")
        pixel = [[item[0] * width, item[1] * height, item[2]] for item in normalized]
        if not all(math.isfinite(value) for point in pixel for value in point):
            raise ValueError("derived hand landmark pixels must be finite")
        identity = {
            "scene_index": scene_index, "frame_index": frame_index, "pts": pts,
            "evidence_sha256": evidence_sha256, "pixel_sha256": pixel_sha256,
            "model_run_id": model_run_id, "hand_index": hand_index,
        }
        observations.append({
            "contract": "hand-landmark-observation-v1",
            "observation_id": _observation_id(identity),
            "scene_index": scene_index,
            "frame_index": frame_index,
            "pts": pts,
            "evidence_sha256": evidence_sha256,
            "pixel_sha256": pixel_sha256,
            "model_run_id": model_run_id,
            "hand_index": hand_index,
            "subject_ref": None,
            "landmark_names": list(LANDMARK_NAMES),
            "landmarks_normalized": normalized,
            "landmarks_pixel": pixel,
            "world_landmarks_m": world_m,
            "box_xyxy": [
                float(max(0.0, minimum[0] * width)),
                float(max(0.0, minimum[1] * height)),
                float(min(float(width), maximum[0] * width)),
                float(min(float(height), maximum[1] * height)),
            ],
            "outside_frame": bool((minimum < 0).any() or (maximum > 1).any()),
            "handedness_raw": name,
            "handedness_score": score,
            "handedness_category_index": category_index,
            "handedness_score_kind": "mediapipe_category_score_uncalibrated",
            "handedness_mirrored_input_assumption_verified": False,
            "review_status": "unreviewed",
            "uncertainty": (
                "Hand landmarks and handedness are unreviewed model outputs. The input mirror "
                "state, persistent person/hand identity, occlusion, touch, grasp, support, intent "
                "and action are not established."
            ),
        })
        if (observations[-1]["box_xyxy"][2] <= observations[-1]["box_xyxy"][0]
                or observations[-1]["box_xyxy"][3] <= observations[-1]["box_xyxy"][1]):
            raise ValueError("hand landmark box falls outside the image")
    return observations


def _mediapipe_version() -> str:
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _terminate_process_tree(process: subprocess.Popen,
                            windows_job: Optional[_WindowsKillJob] = None) -> bool:
    """Stop only the owned worker tree and report whether tree ownership held."""
    if windows_job is not None:
        completed = windows_job.terminate_and_close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            completed = False
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                completed = False
        return completed
    if process.poll() is not None:
        return os.name != "nt"
    if os.name == "nt":
        # This path is only best-effort cleanup after Job Object attachment
        # itself failed.  Callers must surface isolation_unavailable and must
        # never treat it as the promised complete-tree boundary.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return os.name != "nt"


def _worker_output(value: Dict[str, Any]) -> Any:
    """Convert the bounded JSON worker payload into the local normalizer API."""
    if not isinstance(value, dict) or set(value) != {
            "kind", "version", "status", "reason", "error", "runtime_version",
            "pixel_sha256", "hands_raw"}:
        raise ValueError("hand_landmark_worker_response_shape_invalid")
    if (value["kind"] != "traceannotate.hand-landmarks-worker-result"
            or value["version"] != 1
            or value["status"] not in {"success", "failed", "not_run"}
            or not isinstance(value["runtime_version"], str)
            or _SAFE_WORKER_TOKEN_RE.fullmatch(value["runtime_version"]) is None):
        raise ValueError("hand_landmark_worker_response_identity_invalid")
    if value["status"] != "success":
        if (value["hands_raw"] != [] or value["pixel_sha256"] is not None
                or not isinstance(value["reason"], str) or not value["reason"]
                or _SAFE_WORKER_TOKEN_RE.fullmatch(value["reason"]) is None
                or value["error"] is not None
                and (not isinstance(value["error"], str)
                     or _SAFE_WORKER_TOKEN_RE.fullmatch(value["error"]) is None)):
            raise ValueError("hand_landmark_worker_failure_shape_invalid")
        return value
    if (value["reason"] is not None or value["error"] is not None
            or not isinstance(value["pixel_sha256"], str)
            or _HASH_RE.fullmatch(value["pixel_sha256"]) is None
            or not isinstance(value["hands_raw"], list)):
        raise ValueError("hand_landmark_worker_success_shape_invalid")
    landmarks, world, handedness = [], [], []
    for item in value["hands_raw"]:
        if not isinstance(item, dict) or set(item) != {"landmarks", "world_landmarks", "handedness"}:
            raise ValueError("hand_landmark_worker_hand_shape_invalid")
        if (not isinstance(item["landmarks"], list) or len(item["landmarks"]) != 21
                or not isinstance(item["world_landmarks"], list)
                or len(item["world_landmarks"]) != 21
                or not isinstance(item["handedness"], dict)
                or set(item["handedness"]) != {"category_name", "score", "index"}):
            raise ValueError("hand_landmark_worker_hand_shape_invalid")
        def points(raw):
            if not all(isinstance(point, list) and len(point) == 3 for point in raw):
                raise ValueError("hand_landmark_worker_point_shape_invalid")
            return [SimpleNamespace(x=point[0], y=point[1], z=point[2]) for point in raw]
        landmarks.append(points(item["landmarks"]))
        world.append(points(item["world_landmarks"]))
        handedness.append([SimpleNamespace(**item["handedness"])])
    return SimpleNamespace(hand_landmarks=landmarks,
                           hand_world_landmarks=world,
                           handedness=handedness)


class HandLandmarkExtractor(BaseExtractor):
    """Exact local MediaPipe task bundle executed in an owned subprocess."""

    def __init__(self, model_path: Optional[str] = None, *, max_hands: int = 4,
                 worker_command: Optional[Sequence[str]] = None,
                 worker_timeout: float = WORKER_TIMEOUT_SECONDS):
        if type(max_hands) is not int or not 1 <= max_hands <= 8:
            raise ValueError("max_hands must be an integer from 1 to 8")
        if (not isinstance(worker_timeout, (int, float)) or isinstance(worker_timeout, bool)
                or not math.isfinite(float(worker_timeout)) or not 0.05 <= float(worker_timeout) <= 600):
            raise ValueError("worker_timeout must be between 0.05 and 600 seconds")
        if worker_command is None:
            worker_command = [sys.executable, "-I", "-m", WORKER_MODULE]
        if (not isinstance(worker_command, (list, tuple)) or not worker_command
                or len(worker_command) > 16
                or any(not isinstance(item, str) or not item or len(item) > 4096
                       for item in worker_command)):
            raise ValueError("worker_command must be a bounded argv sequence")
        self.model_path = str(model_path) if model_path else None
        self.max_hands = max_hands
        self.worker_command = tuple(worker_command)
        self.worker_timeout = float(worker_timeout)
        self._weights_sha256: Optional[str] = None
        self._runtime_version = _mediapipe_version()
        self._closed = False
        self._process_lock = threading.Lock()
        self._active_processes: Dict[int, subprocess.Popen] = {}
        self._active_windows_jobs: Dict[int, _WindowsKillJob] = {}
        self._status, self._reason, self._error = (
            "not_run", "hand_landmark_weights_missing", None)
        path = Path(self.model_path) if self.model_path else None
        if path is None:
            return
        try:
            path.lstat()
            model_bytes, digest = _read_exact_model(path)
            del model_bytes
            self._weights_sha256 = digest
            if digest != OFFICIAL_WEIGHTS_SHA256:
                raise ValueError("hand_landmark_weights_changed")
            if self._runtime_version == "unavailable":
                self._status, self._reason = (
                    "not_run", "hand_landmark_runtime_unavailable")
            else:
                self._status, self._reason = "success", None
        except FileNotFoundError:
            return
        except Exception as exc:
            self._status = "failed"
            self._reason = safe_failure_reason(
                exc, "hand_landmark_model_load_failed")
            self._error = type(exc).__name__

    @property
    def name(self) -> str:
        return "hand_landmarks"

    @property
    def code_identity(self) -> Dict[str, Any]:
        verified = self._weights_sha256 == OFFICIAL_WEIGHTS_SHA256
        return {
            "provider": "mediapipe-tasks-cpu",
            "model_name": PROVIDER if verified else "unverified_local_task",
            "model_version": "float16/1" if verified else "unverified",
            "model_source": OFFICIAL_MODEL_URL if verified else None,
            "model_card": OFFICIAL_MODEL_CARD if verified else None,
            "weights_size": OFFICIAL_MODEL_SIZE if verified else None,
            "weights_sha256": self._weights_sha256,
            "license": MODEL_LICENSE if verified else "unverified",
            "license_source": MODEL_LICENSE_SOURCE if verified else None,
            "runtime_license_source": RUNTIME_LICENSE_SOURCE,
            "runtime_version": self._runtime_version,
            "code_version": CODE_VERSION,
            "running_mode": "IMAGE",
            "max_hands": self.max_hands,
            "offline": True,
            "downloads_allowed": False,
            "authority": "hand_keypoint_observations_only",
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.code_identity, sort_keys=True,
                             ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _invoke_worker(self, frame_bytes: bytes, header: Dict[str, Any],
                       cancelled: Optional[Callable[[], bool]]) -> Dict[str, Any]:
        header_bytes = json.dumps(
            header, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("ascii") + b"\n"
        if len(header_bytes) > MAX_WORKER_HEADER_BYTES:
            raise ValueError("hand_landmark_worker_header_too_large")
        creationflags = 0
        popen_options: Dict[str, Any] = {}
        if os.name == "nt":
            creationflags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(
            list(self.worker_command), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=creationflags, **popen_options)
        windows_job: Optional[_WindowsKillJob] = None
        if os.name == "nt":
            try:
                # Assign before request bytes are written.  If the host's
                # current Job forbids nesting, fail closed instead of silently
                # downgrading the complete-tree cancellation contract.
                windows_job = _WindowsKillJob.attach(process)
            except _WorkerIsolationUnavailable:
                _terminate_process_tree(process)
                raise
        with self._process_lock:
            if self._closed:
                reject_process = True
            else:
                reject_process = False
                self._active_processes[process.pid] = process
                if windows_job is not None:
                    self._active_windows_jobs[process.pid] = windows_job
        if reject_process:
            _terminate_process_tree(process, windows_job)
            raise InterruptedError("hand_landmark_analysis_cancelled")
        writer_errors: List[BaseException] = []
        reader_errors: List[BaseException] = []
        stdout_data = bytearray()
        stderr_data = bytearray()
        output_limit = threading.Event()

        def write_request() -> None:
            try:
                assert process.stdin is not None
                process.stdin.write(header_bytes)
                view = memoryview(frame_bytes)
                for offset in range(0, len(view), 1024 * 1024):
                    process.stdin.write(view[offset:offset + 1024 * 1024])
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                writer_errors.append(exc)
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass

        def read_bounded(stream: Any, target: bytearray, limit: int) -> None:
            try:
                while True:
                    chunk = stream.read(min(65536, limit + 1 - len(target)))
                    if not chunk:
                        return
                    target.extend(chunk)
                    if len(target) > limit:
                        output_limit.set()
                        return
            except OSError as exc:
                reader_errors.append(exc)
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        assert process.stdout is not None and process.stderr is not None
        writer = threading.Thread(target=write_request, daemon=True)
        stdout_reader = threading.Thread(
            target=read_bounded,
            args=(process.stdout, stdout_data, MAX_WORKER_RESPONSE_BYTES), daemon=True)
        stderr_reader = threading.Thread(
            target=read_bounded,
            args=(process.stderr, stderr_data, MAX_WORKER_STDERR_BYTES), daemon=True)
        workers = (writer, stdout_reader, stderr_reader)
        started_workers: List[threading.Thread] = []
        deadline = time.monotonic() + self.worker_timeout
        try:
            for worker in workers:
                worker.start()
                started_workers.append(worker)
            while process.poll() is None:
                with self._process_lock:
                    closing = self._closed
                if closing or (cancelled is not None and cancelled()):
                    if not _terminate_process_tree(process, windows_job):
                        raise _WorkerIsolationUnavailable(
                            "hand_landmark_worker_job_termination_failed")
                    raise InterruptedError("hand_landmark_analysis_cancelled")
                if time.monotonic() >= deadline:
                    if not _terminate_process_tree(process, windows_job):
                        raise _WorkerIsolationUnavailable(
                            "hand_landmark_worker_job_termination_failed")
                    raise TimeoutError("hand_landmark_worker_timeout")
                if output_limit.is_set():
                    if not _terminate_process_tree(process, windows_job):
                        raise _WorkerIsolationUnavailable(
                            "hand_landmark_worker_job_termination_failed")
                    raise ValueError("hand_landmark_worker_output_limit")
                time.sleep(WORKER_POLL_SECONDS)
        finally:
            try:
                if process.poll() is None:
                    _terminate_process_tree(process, windows_job)
                elif windows_job is not None:
                    windows_job.close()
                for worker in started_workers:
                    worker.join(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            pass
            finally:
                with self._process_lock:
                    self._active_processes.pop(process.pid, None)
                    self._active_windows_jobs.pop(process.pid, None)
        with self._process_lock:
            closed_during_inference = self._closed
        if closed_during_inference:
            raise InterruptedError("hand_landmark_analysis_cancelled")
        if (output_limit.is_set() or len(stdout_data) > MAX_WORKER_RESPONSE_BYTES
                or len(stderr_data) > MAX_WORKER_STDERR_BYTES):
            raise ValueError("hand_landmark_worker_output_limit")
        if any(worker.is_alive() for worker in started_workers):
            raise RuntimeError("hand_landmark_worker_pipe_shutdown_failed")
        if reader_errors:
            raise RuntimeError("hand_landmark_worker_pipe_read_failed")
        if process.returncode != 0:
            detail = hashlib.sha256(stderr_data).hexdigest() if stderr_data else "none"
            raise RuntimeError(
                f"hand_landmark_worker_exit_{process.returncode}_stderr_{detail}")
        if writer_errors:
            raise RuntimeError("hand_landmark_worker_input_rejected")
        try:
            value = json.loads(stdout_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("hand_landmark_worker_response_invalid") from exc
        canonical = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("ascii")
        if canonical != stdout_data:
            raise ValueError("hand_landmark_worker_response_not_canonical")
        return value

    def analyze_frame(self, image_bgr: np.ndarray, *, scene_index: int,
                      frame_index: Any, pts: Dict[str, str], evidence_sha256: str,
                      model_run_id: str,
                      cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        if cancelled is not None and not callable(cancelled):
            raise ValueError("cancelled must be callable")
        try:
            initially_cancelled = cancelled is not None and cancelled()
        except InterruptedError:
            raise
        except Exception as exc:
            return self._result(
                "failed", reason="hand_landmark_worker_failed",
                error=type(exc).__name__)
        if initially_cancelled:
            raise InterruptedError("hand_landmark_analysis_cancelled")
        if self._status != "success":
            return self._result(self._status, reason=self._reason, error=self._error)
        if type(scene_index) is not int or scene_index < 0:
            return self._result("failed", reason="invalid_scene_index")
        try:
            frame = _frame(frame_index)
            point = _pts(pts)
            run_id = _identifier(model_run_id, "model_run_id")
        except ValueError as exc:
            return self._result(
                "failed", reason="invalid_frame_identity", error=type(exc).__name__)
        if not isinstance(evidence_sha256, str) or _HASH_RE.fullmatch(evidence_sha256) is None:
            return self._result("failed", reason="invalid_evidence_sha256")
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3
                or min(image_bgr.shape[:2]) <= 0
                or int(image_bgr.shape[0]) * int(image_bgr.shape[1]) > MAX_FRAME_PIXELS
                or int(image_bgr.nbytes) > MAX_FRAME_BYTES):
            return self._result("failed", reason="invalid_image")
        try:
            contiguous = np.ascontiguousarray(image_bgr)
            frame_bytes = contiguous.tobytes()
            pixel_sha256 = hashlib.sha256(frame_bytes).hexdigest()
            height, width = image_bgr.shape[:2]
            response = self._invoke_worker(frame_bytes, {
                "kind": "traceannotate.hand-landmarks-worker-request",
                "version": 1, "model_path": self.model_path,
                "max_hands": self.max_hands,
                "scene_index": scene_index, "frame_index": frame,
                "pts": point, "evidence_sha256": evidence_sha256,
                "model_run_id": run_id, "width": int(width),
                "height": int(height), "channels": 3, "dtype": "uint8",
                "frame_bytes": len(frame_bytes), "pixel_sha256": pixel_sha256,
            }, cancelled)
            output = _worker_output(response)
            if isinstance(output, dict):
                self._runtime_version = output["runtime_version"]
                return self._result(
                    output["status"], reason=output["reason"], error=output["error"])
            if response["pixel_sha256"] != pixel_sha256:
                raise ValueError("hand_landmark_worker_pixel_identity_mismatch")
            self._runtime_version = response["runtime_version"]
            observations = _hand_observations(
                output, width=width, height=height, scene_index=scene_index,
                frame_index=frame, pts=point, evidence_sha256=evidence_sha256,
                pixel_sha256=pixel_sha256, model_run_id=run_id,
                max_hands=self.max_hands)
            if cancelled is not None and cancelled():
                raise InterruptedError("hand_landmark_analysis_cancelled")
            return self._result("success", hands=observations, frame_identity={
                "scene_index": scene_index, "frame_index": frame, "pts": point,
                "evidence_sha256": evidence_sha256, "pixel_sha256": pixel_sha256,
                "model_run_id": run_id, "shape": list(image_bgr.shape),
            })
        except InterruptedError:
            raise
        except TimeoutError:
            return self._result("failed", reason="hand_landmark_worker_timeout")
        except _WorkerIsolationUnavailable:
            return self._result(
                "failed", reason="hand_landmark_worker_isolation_unavailable")
        except Exception as exc:
            return self._result(
                "failed", reason="hand_landmark_worker_failed",
                error=type(exc).__name__)

    def _result(self, status: str, *, hands: Optional[List[Dict[str, Any]]] = None,
                frame_identity: Optional[Dict[str, Any]] = None,
                reason: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": status, "hands": hands or [], "provider": self.name,
            "model_identity": self.code_identity, "fingerprint": self.fingerprint,
            "review_status": "unreviewed",
            "authority": "hand_keypoint_observations_only",
            "unsupported_assertions": ["touch", "grasp", "held_by", "pick_up", "put_down",
                                       "open", "close", "hand_over"],
            "cancellation_boundary": CANCELLATION_BOUNDARY,
        }
        if frame_identity is not None:
            result["frame_identity"] = frame_identity
        if reason is not None:
            result["reason"] = reason
        if error is not None:
            result["error"] = str(error)[:200]
        return result

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        return self._result("not_run", reason="decoded_frame_identity_required")

    def close(self) -> None:
        with self._process_lock:
            self._closed = True
            processes = [
                (process, self._active_windows_jobs.get(pid))
                for pid, process in self._active_processes.items()
            ]
        complete_tree_cleanup = True
        for process, windows_job in processes:
            if not _terminate_process_tree(process, windows_job):
                complete_tree_cleanup = False
        if not complete_tree_cleanup:
            self._status, self._reason = (
                "failed", "hand_landmark_worker_isolation_unavailable")
        elif self._status == "success":
            self._status, self._reason = "not_run", "hand_landmark_runtime_closed"
