"""Owned one-shot MediaPipe Hand Landmarker worker.

The parent sends one canonical JSON header followed by exact raw BGR bytes.
This process is intentionally disposable: a native runtime crash, deadlock or
allocator failure cannot take down the workspace process.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from trace_annotate.extractors import hands as H


REQUEST_FIELDS = {
    "kind", "version", "model_path", "max_hands", "scene_index",
    "frame_index", "pts", "evidence_sha256", "model_run_id", "width",
    "height", "channels", "dtype", "frame_bytes", "pixel_sha256",
}


def _response(status, *, reason=None, error=None, pixel_sha256=None, hands_raw=None):
    try:
        runtime_version = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = "unavailable"
    return {
        "kind": "traceannotate.hand-landmarks-worker-result",
        "version": 1,
        "status": status,
        "reason": reason,
        "error": error,
        "runtime_version": runtime_version,
        "pixel_sha256": pixel_sha256,
        "hands_raw": hands_raw or [],
    }


def _read_request():
    line = sys.stdin.buffer.readline(H.MAX_WORKER_HEADER_BYTES + 1)
    if (not line.endswith(b"\n") or len(line) > H.MAX_WORKER_HEADER_BYTES):
        raise ValueError("worker_request_header_invalid")
    header_bytes = line[:-1]
    request = json.loads(header_bytes)
    if (not isinstance(request, dict) or set(request) != REQUEST_FIELDS
            or request["kind"] != "traceannotate.hand-landmarks-worker-request"
            or request["version"] != 1):
        raise ValueError("worker_request_shape_invalid")
    canonical = json.dumps(
        request, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")
    if canonical != header_bytes:
        raise ValueError("worker_request_not_canonical")
    max_hands = request["max_hands"]
    width, height, channels = request["width"], request["height"], request["channels"]
    size = request["frame_bytes"]
    if (type(max_hands) is not int or not 1 <= max_hands <= 8
            or type(width) is not int or type(height) is not int
            or type(channels) is not int or channels != 3
            or min(width, height) <= 0 or width * height > H.MAX_FRAME_PIXELS
            or request["dtype"] != "uint8" or type(size) is not int
            or size != width * height * channels or size > H.MAX_FRAME_BYTES
            or not isinstance(request["model_path"], str)
            or not request["model_path"] or len(request["model_path"]) > 4096
            or "\x00" in request["model_path"]
            or type(request["scene_index"]) is not int
            or request["scene_index"] < 0
            or not isinstance(request["pixel_sha256"], str)
            or H._HASH_RE.fullmatch(request["pixel_sha256"]) is None
            or not isinstance(request["evidence_sha256"], str)
            or H._HASH_RE.fullmatch(request["evidence_sha256"]) is None):
        raise ValueError("worker_request_limits_invalid")
    H._frame(request["frame_index"])
    H._pts(request["pts"])
    H._identifier(request["model_run_id"], "model_run_id")
    raw = sys.stdin.buffer.read(size + 1)
    if len(raw) != size:
        raise ValueError("worker_frame_size_invalid")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != request["pixel_sha256"]:
        raise ValueError("worker_frame_hash_invalid")
    return request, raw, digest


def _point(value):
    return [float(value.x), float(value.y), float(value.z)]


def _run():
    request, raw, pixel_sha256 = _read_request()
    model_bytes, digest = H._read_exact_model(Path(request["model_path"]))
    if digest != H.OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("hand_landmark_weights_changed")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        return _response("not_run", reason="hand_landmark_runtime_unavailable")
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
        request["height"], request["width"], request["channels"])
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_buffer=model_bytes,
            delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=request["max_hands"],
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    runtime = vision.HandLandmarker.create_from_options(options)
    try:
        output = runtime.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb))
    finally:
        runtime.close()
    if not (len(output.hand_landmarks) == len(output.hand_world_landmarks)
            == len(output.handedness)):
        raise ValueError("hand_landmark_parallel_output_mismatch")
    if len(output.hand_landmarks) > request["max_hands"]:
        raise ValueError("hand_landmark_output_limit")
    hands_raw = []
    for landmarks, world, categories in zip(
            output.hand_landmarks, output.hand_world_landmarks, output.handedness):
        if len(landmarks) != 21 or len(world) != 21 or not categories:
            raise ValueError("hand_landmark_output_shape_invalid")
        category = categories[0]
        hands_raw.append({
            "landmarks": [_point(item) for item in landmarks],
            "world_landmarks": [_point(item) for item in world],
            "handedness": {
                "category_name": category.category_name,
                "score": float(category.score),
                "index": category.index,
            },
        })
    return _response(
        "success", pixel_sha256=pixel_sha256, hands_raw=hands_raw)


def main():
    try:
        value = _run()
    except ImportError:
        value = _response("not_run", reason="hand_landmark_runtime_unavailable")
    except BaseException as exc:
        value = _response(
            "failed", reason="hand_landmark_worker_failed",
            error=type(exc).__name__)
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")
    if len(data) > H.MAX_WORKER_RESPONSE_BYTES:
        raise SystemExit(70)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
