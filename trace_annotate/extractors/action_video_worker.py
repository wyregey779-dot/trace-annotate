"""Child-only worker for one exact VideoMAE/SSV2 auxiliary window."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Dict, List

import numpy as np

from .action_video import (
    ISOLATED_CONTRACT,
    MAX_ISOLATED_OUTPUT_BYTES,
    MAX_ISOLATED_REQUEST_BYTES,
    MAX_TOP_K,
    MAX_WINDOW_BYTES,
    PROVIDER,
    VideoActionAuxiliary,
    _window,
)


def _digest(path: Path, maximum: int) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError("isolated action input is missing or exceeds its byte budget") from exc
    reparse = bool(getattr(before, "st_file_attributes", 0) & 0x400)
    if (stat.S_ISLNK(before.st_mode) or reparse or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= maximum):
        raise ValueError("isolated action input is missing or exceeds its byte budget")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
            raise ValueError("isolated action input identity changed")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        after_open = os.fstat(stream.fileno())
    after_path = path.lstat()
    if ((after_open.st_dev, after_open.st_ino, after_open.st_size, after_open.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise ValueError("isolated action input identity changed")
    return value.hexdigest()


def _read_request(path: Path) -> bytes:
    """Read the small request once from a stable regular file handle."""
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError("isolated action request is missing") from exc
    reparse = bool(getattr(before, "st_file_attributes", 0) & 0x400)
    if (stat.S_ISLNK(before.st_mode) or reparse or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= MAX_ISOLATED_REQUEST_BYTES):
        raise ValueError("isolated action request exceeds its byte budget")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
            raise ValueError("isolated action request identity changed")
        raw = stream.read(MAX_ISOLATED_REQUEST_BYTES + 1)
        after_open = os.fstat(stream.fileno())
    after_path = path.lstat()
    if (not 0 < len(raw) <= MAX_ISOLATED_REQUEST_BYTES
            or (after_open.st_dev, after_open.st_ino, after_open.st_size, after_open.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise ValueError("isolated action request identity changed")
    return raw


def _peak_memory() -> Any:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged", "paged",
                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        value = Counters()
        value.size = ctypes.sizeof(value)
        function = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        if function(kernel.GetCurrentProcess(), ctypes.byref(value), value.size):
            return int(value.peak_working_set)
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _publish_progress(path: Path, phase: str) -> None:
    value = json.dumps({"phase": phase, "started_at": time.time()},
                       ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def execute(request_path: Path, output_path: Path, progress_path: Path) -> None:
    root = request_path.parent.resolve()
    if (output_path.parent.resolve() != root or progress_path.parent.resolve() != root
            or len({request_path.resolve(), output_path.resolve(), progress_path.resolve()}) != 3
            or os.path.lexists(output_path)):
        raise ValueError("isolated action output paths are outside the owned work directory")
    raw = _read_request(request_path)
    if not 0 < len(raw) <= MAX_ISOLATED_REQUEST_BYTES:
        raise ValueError("isolated action request exceeds its byte budget")
    request_sha256 = hashlib.sha256(raw).hexdigest()
    request = json.loads(raw.decode("utf-8"))
    expected = {"contract", "provider", "model_path", "top_k", "window", "frames",
                "authority", "participants_must_remain_null"}
    frames = request.get("frames") if isinstance(request, dict) else None
    model_path = request.get("model_path") if isinstance(request, dict) else None
    window = request.get("window") if isinstance(request, dict) else None
    if (not isinstance(request, dict) or set(request) != expected
            or request.get("contract") != ISOLATED_CONTRACT
            or request.get("provider") != PROVIDER
            or request.get("authority") != "auxiliary_window_semantics_only"
            or request.get("participants_must_remain_null") is not True
            or not isinstance(model_path, str) or not Path(model_path).is_absolute()
            or type(request.get("top_k")) is not int
            or not 1 <= request["top_k"] <= MAX_TOP_K
            or not isinstance(window, dict)
            or set(window) != {"start_frame", "end_frame", "window_pts",
                               "scene_index", "total_input_bytes"}
            or not isinstance(frames, list) or len(frames) != 16):
        raise ValueError("invalid isolated action request")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    samples: List[Dict[str, Any]] = []
    total_bytes = 0
    total_file_bytes = 0
    frame_keys = {"frame_index", "pts", "scene_index", "evidence_sha256",
                  "pixel_sha256", "shape", "file", "file_sha256"}
    for frame_number, frame in enumerate(frames):
        if (not isinstance(frame, dict) or set(frame) != frame_keys
                or not isinstance(frame.get("file"), str)):
            raise ValueError("invalid isolated action frame")
        relative = frame["file"]
        if (relative != f"frames/{frame_number:02d}.npy"
                or "\\" in relative or Path(relative).is_absolute()
                or any(part in {"", ".", ".."} for part in relative.split("/"))):
            raise ValueError("invalid isolated action frame path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("isolated action frame escapes its work directory") from exc
        if path.is_file():
            total_file_bytes += path.stat().st_size
        if total_file_bytes > MAX_WINDOW_BYTES + len(frames) * 4096:
            raise ValueError("isolated action frame files exceed their aggregate byte budget")
        if _digest(path, MAX_WINDOW_BYTES) != frame.get("file_sha256"):
            raise ValueError("isolated action frame file identity changed")
        image = np.load(path, allow_pickle=False)
        total_bytes += int(getattr(image, "nbytes", 0))
        if total_bytes > MAX_WINDOW_BYTES:
            raise ValueError("isolated action window exceeds its byte budget")
        if hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != frame.get("pixel_sha256"):
            raise ValueError("isolated action frame pixels changed")
        samples.append({
            "frame_index": frame.get("frame_index"),
            "pts": frame.get("pts"),
            "scene_index": frame.get("scene_index"),
            "evidence_sha256": frame.get("evidence_sha256"),
            "image": image,
        })
    _images, frozen_identity = _window(samples, cancelled=None)
    expected_frames = [
        {key: frame[key] for key in (
            "frame_index", "pts", "scene_index", "evidence_sha256",
            "pixel_sha256", "shape")}
        for frame in frames
    ]
    expected_window = {key: frozen_identity[key] for key in (
        "start_frame", "end_frame", "window_pts", "scene_index", "total_input_bytes")}
    if frozen_identity["frames"] != expected_frames or request["window"] != expected_window:
        raise ValueError("isolated action request identity is internally inconsistent")
    del _images
    _publish_progress(progress_path, "frame")
    provider = VideoActionAuxiliary(request["model_path"])
    result = provider.analyze_window(samples, top_k=request["top_k"], cancelled=None)
    if any(result.get(name) is not None
           for name in ("subject_ref", "object_ref", "recipient_ref")):
        raise ValueError("auxiliary action worker attempted to assign participants")
    if result.get("status") == "success":
        actual_window = result.get("window")
        if (not isinstance(actual_window, dict)
                or {key: actual_window.get(key) for key in request["window"]}
                != request["window"]
                or actual_window.get("frames") != expected_frames):
            raise ValueError("isolated action result does not match its frozen input window")
    value = {
        "contract": ISOLATED_CONTRACT,
        "provider": PROVIDER,
        "request_sha256": request_sha256,
        "result": result,
        "actual_pid": os.getpid(),
        "actual_parent_pid": os.getppid(),
        "peak_working_set_bytes": _peak_memory(),
    }
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_ISOLATED_OUTPUT_BYTES:
        raise ValueError("isolated action result exceeds its byte budget")
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, output_path)
    _publish_progress(progress_path, "complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    args = parser.parse_args()
    execute(args.request, args.output, args.progress)


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        raise SystemExit(75)
    except TimeoutError:
        raise SystemExit(76)
