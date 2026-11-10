"""
Face Detection Extractor with OpenCV 5.0 compatibility and explicit execution status.
Supports YuNet ONNX face detection loaded via memory buffers,
calculating face count, crowd density, and bounding boxes.
"""
from __future__ import annotations
import os
import shutil
import tempfile
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from trace_annotate.extractors.base import BaseExtractor, ExtractorStatus
from trace_annotate.core.hasher import safe_load_image
from trace_annotate.config import MetricThresholds


def _ascii_temp_root() -> str:
    """Return a temp root OpenCV's byte-oriented ONNX loader can address.

    A workspace may redirect TEMP/TMP into a Chinese directory. OpenCV's
    ``FaceDetectorYN.create`` takes a byte path, so every copy written there
    fails to load and the detector silently degrades to None. Pick the first
    candidate whose path is ASCII; LOCALAPPDATA/ProgramData are ASCII on the
    supported Windows hosts.
    """
    candidates = [
        tempfile.gettempdir(),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp"),
        os.environ.get("ProgramData", ""),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            candidate.encode("ascii")
        except UnicodeEncodeError:
            continue
        return candidate
    return tempfile.gettempdir()


class FaceExtractor(BaseExtractor):
    """Detects faces using YuNet ONNX or provides safe fallbacks with explicit status."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        thresholds: Optional[MetricThresholds] = None
    ):
        self.model_path = model_path or weights_path
        self.thresholds = thresholds or MetricThresholds()
        self.detector = None
        self.detector_kind = None
        self._input_size = None
        self._init_detector()

    @property
    def name(self) -> str:
        return "faces"

    @property
    def code_identity(self) -> Dict[str, Any]:
        """Detector and runtime identity for a model-run record.

        YuNet weights are an external file, so their hash is computed by the
        caller from ``model_path``; this property never invents a digest.
        """
        import cv2
        return {"provider": "opencv",
                "model_name": ("YuNet" if self.detector_kind == "yunet"
                               else "Haar cascade" if self.detector_kind == "haar"
                               else "unavailable"),
                "model_version": getattr(cv2, "__version__", "unknown"),
                "weights_path": self.model_path,
                "detector_kind": self.detector_kind,
                "license": "OpenCV Apache-2.0; detector weights license not bundled locally"}

    def _init_detector(self) -> None:
        """Initialize YuNet ONNX, or OpenCV's bundled Haar cascade.

        ``model_path="haar"`` selects the cascade that ships with OpenCV, so a
        local workspace can measure face boxes without downloading weights.
        The cascade returns no calibrated confidence; that is reported instead
        of inventing a score.
        """
        if self.model_path == "haar":
            if not HAS_CV2:
                return
            try:
                path = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
                cascade = cv2.CascadeClassifier(path)
                if cascade.empty():
                    return
                self.detector = cascade
                self.detector_kind = "haar"
            except Exception:
                self.detector = None
            return
        if not HAS_CV2 or not self.model_path or not os.path.exists(self.model_path):
            return

        try:
            # OpenCV's ONNX importer is byte-oriented and fails on a non-ASCII
            # path. The workspace itself may live under a Chinese directory, so
            # load an ASCII-named temporary copy of the bundled weights.
            model_path = self.model_path
            try:
                model_path.encode("ascii")
            except UnicodeEncodeError:
                cache_dir = os.path.join(_ascii_temp_root(), "traceannotate", "yunet")
                os.makedirs(cache_dir, exist_ok=True)
                model_path = os.path.join(cache_dir, os.path.basename(self.model_path))
                if not os.path.exists(model_path) or os.path.getsize(model_path) != os.path.getsize(self.model_path):
                    shutil.copyfile(self.model_path, model_path)
            with open(self.model_path, "rb") as f:
                model_buf = f.read()

            if hasattr(cv2, "FaceDetectorYN"):
                try:
                    self.detector = cv2.FaceDetectorYN.create(
                        model=model_path,
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.thresholds.face_min_confidence,
                        nms_threshold=0.3,
                        top_k=50
                    )
                    self.detector_kind = "yunet"
                except Exception:
                    self.detector = cv2.FaceDetectorYN.create(
                        "ONNX",
                        np.frombuffer(model_buf, dtype=np.uint8),
                        input_size=(320, 320),
                        score_threshold=self.thresholds.face_min_confidence,
                        nms_threshold=0.3,
                        top_k=50
                    )
                    self.detector_kind = "yunet"
        except Exception:
            self.detector = None

    def _summary(self, detected_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Face count/crowd summary that never turns a large face into a shot size."""
        count = len(detected_list)
        if count == 0:
            crowd, status = "未检出人脸", "未检出人脸"
        elif count == 1:
            crowd, status = "单人", "单人"
        elif count == 2:
            crowd, status = "双人", "双人"
        elif count <= 5:
            crowd, status = "少量3-5人", "小群体"
        else:
            crowd, status = "密集>5人", "人群密集"
        return {"status": ExtractorStatus.SUCCESS.value, "face_count": count,
                "face_status": status, "crowd_level": crowd,
                "faces_detected": detected_list,
                "detector": self.detector_kind}

    def detect_faces(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Detect faces on a given image array."""
        h, w = img_bgr.shape[:2]

        if self.detector is None:
            # Explicitly mark as not run, do not claim zero persons!
            return {
                "status": ExtractorStatus.NOT_RUN.value,
                "face_count": None,
                "face_status": None,
                "crowd_level": None,
                "faces_detected": []
            }

        if self.detector_kind == "haar":
            try:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                rects = self.detector.detectMultiScale(gray, scaleFactor=1.1,
                                                       minNeighbors=5, minSize=(30, 30))
            except Exception as e:
                return {"status": ExtractorStatus.FAILED.value, "error": str(e),
                        "face_count": None, "face_status": None,
                        "crowd_level": None, "faces_detected": []}
            detected_list = []
            for x, y, box_width, box_height in (rects if len(rects) else []):
                detected_list.append({
                    "box": [float(x), float(y), float(box_width), float(box_height)],
                    "confidence": None,
                    "confidence_kind": "unavailable",
                    "is_recognizable": (box_height / max(1, h))
                    >= self.thresholds.face_min_height_ratio,
                })
            return self._summary(detected_list)

        try:
            if self._input_size != (w, h):
                self.detector.setInputSize((w, h))
                self._input_size = (w, h)
            _, faces = self.detector.detect(img_bgr)
        except Exception as e:
            return {
                "status": ExtractorStatus.FAILED.value,
                "error": str(e),
                "face_count": None,
                "face_status": None,
                "crowd_level": None,
                "faces_detected": []
            }

        if faces is None or len(faces) == 0:
            return {
                "status": ExtractorStatus.SUCCESS.value,
                "face_count": 0,
                "face_status": "未检出人脸",
                "crowd_level": "未检出人脸",
                "faces_detected": []
            }

        detected_list = []
        recognizable_count = 0

        for f in faces:
            box_x, box_y, box_w, box_h = f[0:4]
            conf = float(f[-1])
            is_recog = (
                conf >= self.thresholds.face_min_confidence and
                (box_h / max(1, h)) >= self.thresholds.face_min_height_ratio
            )
            if is_recog:
                recognizable_count += 1

            detected_list.append({
                "box": [float(box_x), float(box_y), float(box_w), float(box_h)],
                "confidence": round(conf, 3),
                "is_recognizable": bool(is_recog)
            })

        count = len(detected_list)
        if count == 0:
            crowd = "未检出人脸"
            status = "未检出人脸"
        elif count == 1:
            crowd = "单人"
            status = "单人"
        elif count == 2:
            crowd = "双人"
            status = "双人"
        elif count <= 5:
            crowd = "少量3-5人"
            status = "小群体"
        else:
            crowd = "密集>5人"
            status = "人群密集"

        return {
            "status": ExtractorStatus.SUCCESS.value,
            "face_count": count,
            "face_status": status,
            "crowd_level": crowd,
            "faces_detected": detected_list,
            "detector": self.detector_kind
        }

    def detect_faces_in_video(self, video_path: str) -> Dict[str, Any]:
        """Convenience wrapper for video face detection."""
        if self.detector is None:
            return {
                "status": ExtractorStatus.NOT_RUN.value,
                "face_count": None,
                "face_status": None,
                "crowd_level": None,
                "faces_detected": []
            }
        return self.extract({}, video_path)

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        img = safe_load_image(staged_path)
        if img is None:
            return {
                "status": ExtractorStatus.FAILED.value,
                "face_count": None,
                "face_status": None,
                "crowd_level": None,
                "faces_detected": []
            }
        return self.detect_faces(img)


# Alias for backward compatibility
FaceDetector = FaceExtractor
