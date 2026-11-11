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
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseExtractor

CODE_VERSION = "face-landmarks-mediapipe-v1"
PROTOCOL = "traceannotate.face-landmarks.v1"
OFFICIAL_WEIGHTS_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


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

    Request: {schema, model_path, max_faces?, images:[{path, frame_index,
    evidence_id?}]}. No network access, implicit model search, or remote code.
    """
    if request.get("schema") != PROTOCOL:
        raise ValueError("unsupported landmark request schema")
    images = request.get("images")
    if not isinstance(images, list) or not 1 <= len(images) <= 512:
        raise ValueError("images must contain 1 to 512 entries")
    if any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
           or not isinstance(item.get("frame_index"), str) for item in images):
        raise ValueError("each image needs a string path and frame_index")
    from .expressive_cues import face_cue_candidates, cues_to_claims
    extractor = FaceLandmarkExtractor(request.get("model_path"), max_faces=request.get("max_faces", 4),
                                      expected_weights_sha256=request.get("expected_weights_sha256"))
    results = []
    try:
        for item in images:
            result = extractor.extract({}, item["path"])
            result.update({"path": item["path"], "frame_index": item["frame_index"]})
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
    return {"schema": PROTOCOL, "status": status, "results": results}


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
