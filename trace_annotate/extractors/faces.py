"""
Face Detection Extractor with OpenCV 5.0 compatibility and explicit execution status.
Supports YuNet ONNX face detection loaded via memory buffers,
calculating face count, crowd density, and bounding boxes.
"""
from __future__ import annotations
import os
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
        self._init_detector()

    @property
    def name(self) -> str:
        return "faces"

    def _init_detector(self) -> None:
        """Initialize YuNet ONNX face detector with OpenCV 5.0 memory buffer fix."""
        if not HAS_CV2 or not self.model_path or not os.path.exists(self.model_path):
            return

        try:
            with open(self.model_path, "rb") as f:
                model_buf = f.read()

            if hasattr(cv2, "FaceDetectorYN"):
                try:
                    self.detector = cv2.FaceDetectorYN.create(
                        model=self.model_path,
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.thresholds.face_min_confidence,
                        nms_threshold=0.3,
                        top_k=50
                    )
                except Exception:
                    self.detector = cv2.FaceDetectorYN.create(
                        "ONNX",
                        np.frombuffer(model_buf, dtype=np.uint8),
                        input_size=(320, 320),
                        score_threshold=self.thresholds.face_min_confidence,
                        nms_threshold=0.3,
                        top_k=50
                    )
        except Exception:
            self.detector = None

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

        try:
            self.detector.setInputSize((w, h))
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
                "face_status": "无人物",
                "crowd_level": "无人",
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
                "is_recognizable": is_recog
            })

        count = len(detected_list)
        if count == 0:
            crowd = "无人"
            status = "无人物"
        elif count == 1:
            crowd = "单人"
            status = "单人特写" if recognizable_count > 0 else "中远景单人"
        elif count == 2:
            crowd = "双人"
            status = "双人互动"
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
            "faces_detected": detected_list
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
