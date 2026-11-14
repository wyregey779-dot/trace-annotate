"""Local CPU MediaPipe landmarks; no model download or emotion inference.

The optional ``mediapipe`` runtime is imported only with an existing task file.
The official task bundles detection and the 478-point Face Mesh V2 network.
Boxes emitted here are xyxy (the expressive-cue contract), unlike YuNet xywh.
The module CLI provides an isolated-worker JSON protocol for optional runtimes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseExtractor

CODE_VERSION = "face-landmarks-mediapipe-v1"
PROTOCOL = "traceannotate.face-landmarks.v1"
OFFICIAL_WEIGHTS_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MAX_INPUT_BYTES = 64 * 1024 * 1024


def _sha256_file(path: Path, *, limit: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            if limit is not None and total > limit:
                raise ValueError("input_image_oversize")
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity() -> Dict[str, Any]:
    """Actual interpreter and MediaPipe metadata used by this worker."""
    python_path = Path(sys.executable).absolute()
    python_sha256 = _sha256_file(python_path)
    runtime = {"mediapipe_version": "unavailable", "metadata_sha256": None}
    try:
        distribution = importlib.metadata.distribution("mediapipe")
        metadata_path = next((Path(distribution.locate_file(item))
                              for item in distribution.files or []
                              if str(item).replace("\\", "/").endswith(".dist-info/METADATA")), None)
        metadata_sha256 = _sha256_file(metadata_path) if metadata_path is not None else None
        runtime = {"mediapipe_version": distribution.version,
                   "metadata_sha256": metadata_sha256}
    except importlib.metadata.PackageNotFoundError:
        pass
    return {"python_path": str(python_path), "python_sha256": python_sha256,
            "runtime": runtime}


def _verified_image(path_value: str, expected_sha256: str):
    """Decode the exact verified regular-file bytes and return its file identity."""
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid_input_sha256")
    path = Path(path_value).absolute()
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("input_image_not_regular_file")
    if before.st_size <= 0 or before.st_size > MAX_INPUT_BYTES:
        raise ValueError("input_image_size_invalid")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size)
                != (before.st_dev, before.st_ino, before.st_size)):
            raise ValueError("input_image_changed_before_read")
        data = stream.read(MAX_INPUT_BYTES + 1)
        after_read = os.fstat(stream.fileno())
    if len(data) != before.st_size or len(data) > MAX_INPUT_BYTES:
        raise ValueError("input_image_changed_during_read")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError("input_image_sha256_mismatch")
    if ((after_read.st_dev, after_read.st_ino, after_read.st_size, after_read.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)):
        raise ValueError("input_image_changed_during_read")
    try:
        import cv2
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except ImportError:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(data)) as source:
            image = np.array(source.convert("RGB"))[:, :, ::-1].copy()
    if image is None:
        raise ValueError("image_decode_failed")
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    return image, path, identity


def _verify_input_unchanged(path: Path, identity, expected_sha256: str) -> None:
    current = path.lstat()
    if (stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != identity
            or _sha256_file(path, limit=MAX_INPUT_BYTES) != expected_sha256):
        raise ValueError("input_image_changed_during_inference")


def mesh_to_face(points: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    """Map the documented MediaPipe topology to observable cue geometry.

    Pose tests are conservative geometry filters, not calibrated head pose or
    occlusion detection. The model does not provide a per-landmark visibility
    score; unknown occlusion must remain unknown.
    """
    mesh = np.asarray(points, dtype=float)
    if mesh.shape != (478, 3) or not np.isfinite(mesh).all():
        raise ValueError("expected 478 finite xyz landmarks")
    if width <= 0 or height <= 0:
        raise ValueError("invalid frame dimensions")
    xy = mesh[:, :2] * [width, height]
    left_eye, right_eye = xy[[33, 133]].mean(axis=0), xy[[362, 263]].mean(axis=0)
    axis = right_eye - left_eye
    eye_gap = float(np.linalg.norm(axis))
    if eye_gap <= 1e-6:
        raise ValueError("degenerate inter-eye geometry")
    horizontal = axis / eye_gap
    vertical = np.array([-horizontal[1], horizontal[0]])
    nose_offset = abs(float(np.dot(xy[1] - (left_eye + right_eye) / 2, horizontal))) / eye_gap
    roll = float(np.degrees(np.arctan2(axis[1], axis[0])))
    minimum, maximum = xy.min(axis=0), xy.max(axis=0)
    # Preserve unclipped extrema in quality evidence so edge truncation cannot
    # be hidden by clamping the display box.
    outside = bool((minimum < [0, 0]).any() or (maximum > [width, height]).any())
    mouth_mid = (xy[13] + xy[14]) / 2
    corner_mid = (xy[61] + xy[291]) / 2
    face_height = max(1.0, float(maximum[1] - minimum[1]))
    corner_lift = float(np.dot(mouth_mid - corner_mid, vertical)) / face_height
    return {
        "box": [float(max(0, minimum[0])), float(max(0, minimum[1])),
                float(min(width, maximum[0])), float(min(height, maximum[1]))],
        "box_format": "xyxy", "frame_size": [width, height],
        "confidence": None, "confidence_kind": "unavailable",
        "landmarks": {
            "mouth": xy[[61, 291, 13, 14]].tolist(),
            "mouth_corner_lift": round(corner_lift, 6),
            "brows": xy[[107, 336]].tolist(), "eye_gap": eye_gap,
            # A small anatomical gap alone does not establish a furrow. No
            # person-specific neutral reference is available in this adapter.
            "brow_baseline_available": False,
            # Each eye: outer/upper/upper/inner/lower/lower; EAR uses both lids.
            "eyes": [xy[[33, 160, 158, 133, 153, 144]].tolist(),
                     xy[[362, 385, 387, 263, 373, 380]].tolist()],
        },
        "quality": {"roll_degrees": round(roll, 3),
                    "nose_offset_eye_ratio": round(nose_offset, 6),
                    "side_facing": nose_offset > 0.45,
                    "excessive_roll": abs(roll) > 35,
                    "partial_outside_frame": outside,
                    "occlusion_assessment": "unavailable",
                    "method": "geometry_filter_not_calibrated_pose"},
        "landmark_count": 478,
        "uncertainty": "关键点预测与几何过滤，未校准表情准确率；未知遮挡无法自动排除",
    }


class FaceLandmarkExtractor(BaseExtractor):
    """Read-only task loading and explicit missing/failed/success states."""

    def __init__(self, model_path: Optional[str] = None, *, max_faces: int = 4,
                 expected_weights_sha256: Optional[str] = None):
        self.model_path = str(model_path) if model_path else None
        if isinstance(max_faces, bool) or not isinstance(max_faces, int) or not 1 <= max_faces <= 16:
            raise ValueError("max_faces must be an integer from 1 to 16")
        self.max_faces = max_faces
        self._mp = self._runtime = None
        self._weights_sha256 = None
        self._model_bytes = None
        self._load_status = "not_run"
        self._reason = "landmark_weights_missing"
        self._error = None
        path = Path(self.model_path) if self.model_path else None
        if path is None or not path.is_file():
            return
        try:
            self._model_bytes = path.read_bytes()
            self._weights_sha256 = hashlib.sha256(self._model_bytes).hexdigest()
            if expected_weights_sha256 and self._weights_sha256 != expected_weights_sha256:
                self._load_status, self._reason = "failed", "landmark_weights_changed"
                self._error = "task digest differs from frozen expected_weights_sha256"
                return
            self._mp, self._runtime = self._create_runtime()
            self._load_status, self._reason = "success", None
        except ImportError as exc:
            self._reason, self._error = "landmark_runtime_unavailable", str(exc)
        except Exception as exc:
            self._load_status = "failed"
            self._reason, self._error = "landmark_model_load_failed", str(exc)

    @property
    def name(self) -> str:
        return "face_landmarks"

    @property
    def code_identity(self) -> Dict[str, Any]:
        try:
            version = importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        official = self._weights_sha256 == OFFICIAL_WEIGHTS_SHA256
        return {"provider": "mediapipe-tasks-cpu", "model_name": "Face Landmarker / Face Mesh V2",
                "model_version": "float16/1" if official else "unverified_local_task", "runtime_version": version,
                "code_version": CODE_VERSION, "weights_path": self.model_path,
                "weights_sha256": self._weights_sha256, "license": "Apache-2.0" if official else "unverified",
                "running_mode": "IMAGE", "max_faces": self.max_faces,
                "offline": True, "downloads_allowed": False}

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.code_identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _create_runtime(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        # Buffer loading handles Unicode paths without a C-drive temporary copy.
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=self._model_bytes,
                                            delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, num_faces=self.max_faces,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False)
        return mp, vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        if self._load_status != "success":
            return self._result(self._load_status, reason=self._reason, error=self._error)
        if (not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or
                img_bgr.shape[2] != 3 or min(img_bgr.shape[:2]) == 0 or img_bgr.dtype != np.uint8):
            return self._result("failed", reason="invalid_image", error="expected nonempty uint8 HWC BGR image")
        try:
            rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
            result = self._runtime.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
            height, width = img_bgr.shape[:2]
            faces = [mesh_to_face(np.array([[p.x, p.y, p.z] for p in mesh]), width, height)
                     for mesh in result.face_landmarks]
            return self._result("success", faces=faces)
        except Exception as exc:
            return self._result("failed", reason="landmark_inference_failed", error=str(exc))

    def _result(self, status: str, *, faces=None, reason=None, error=None) -> Dict[str, Any]:
        result = {"status": status, "faces_detected": faces or [],
                  "provider": self.name, "model_identity": self.code_identity,
                  "fingerprint": self.fingerprint}
        if reason:
            result["reason"] = reason
        if error:
            result["error"] = error
        return result

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        if self._load_status != "success":
            return self._result(self._load_status, reason=self._reason, error=self._error)
        from trace_annotate.core.hasher import safe_load_image
        image = safe_load_image(staged_path)
        if image is None:
            return self._result("failed", reason="image_decode_failed", error="could not decode image")
        return self.detect_landmarks(image)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        if self._load_status == "success":
            self._load_status, self._reason = "not_run", "landmark_runtime_closed"
        self._model_bytes = None


def run_landmark_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Batch worker protocol; image paths and frame evidence stay caller-owned.

    Request: {schema, model_path, expected_worker_python_sha256,
    expected_runtime, max_faces?, images:[{path, frame_index, input_sha256,
    evidence_id?}]}. No network access, implicit model search, or remote code.
    """
    if request.get("schema") != PROTOCOL:
        raise ValueError("unsupported landmark request schema")
    images = request.get("images")
    if not isinstance(images, list) or not 1 <= len(images) <= 512:
        raise ValueError("images must contain 1 to 512 entries")
    if any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
           or not isinstance(item.get("frame_index"), str)
           or not isinstance(item.get("input_sha256"), str)
           or re.fullmatch("[0-9a-f]{64}", item["input_sha256"]) is None for item in images):
        raise ValueError("each image needs path, unique frame_index and input_sha256")
    if len({item["frame_index"] for item in images}) != len(images):
        raise ValueError("duplicate frame_index")
    worker_identity = _runtime_identity()
    if (request.get("expected_worker_python_sha256") != worker_identity["python_sha256"]
            or request.get("expected_runtime") != worker_identity["runtime"]):
        raise ValueError("landmark_worker_identity_changed")
    from .expressive_cues import face_cue_candidates, cues_to_claims
    extractor = FaceLandmarkExtractor(request.get("model_path"), max_faces=request.get("max_faces", 4),
                                      expected_weights_sha256=request.get("expected_weights_sha256"))
    results = []
    try:
        for item in images:
            image, path, input_identity = _verified_image(item["path"], item["input_sha256"])
            result = extractor.detect_landmarks(image)
            _verify_input_unchanged(path, input_identity, item["input_sha256"])
            result.update({"path": item["path"], "frame_index": item["frame_index"],
                           "input_sha256": item["input_sha256"]})
            evaluated = [face_cue_candidates(face, frame_index=item["frame_index"],
                                             evidence_id=item.get("evidence_id"))
                         for face in result["faces_detected"]]
            result["cue_results"] = evaluated
            result["claims"] = cues_to_claims([cue for face in evaluated for cue in face.get("cues", [])])
            results.append(result)
    finally:
        extractor.close()
    statuses = {item["status"] for item in results}
    status = "failed" if "failed" in statuses else "not_run" if "not_run" in statuses else "success"
    return {"schema": PROTOCOL, "status": status, "worker_identity": worker_identity,
            "results": results}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request_json).read_text(encoding="utf-8-sig"))
        result = run_landmark_request(request)
    except Exception as exc:
        result = {"schema": PROTOCOL, "status": "failed", "error": str(exc), "results": []}
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
