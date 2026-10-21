"""
Visual Quality & Metric Extractor.
Extracts brightness, sharpness (Laplacian variance with CV2 5.0 fix),
aspect ratio, and bad-frame identification.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from trace_annotate.extractors.base import BaseExtractor
from trace_annotate.core.hasher import safe_load_image
from trace_annotate.config import MetricThresholds


class VisualMetricsExtractor(BaseExtractor):
    """Calculates image quality, brightness, sharpness, and bad-frame flags."""

    def __init__(self, thresholds: Optional[MetricThresholds] = None):
        self.thresholds = thresholds or MetricThresholds()

    @property
    def name(self) -> str:
        return "visual_metrics"

    def compute_metrics_from_array(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Compute metrics directly from a BGR image array.
        Includes battle-tested OpenCV 5.0 Laplacian fix (uint8 input -> CV_16S output).
        """
        h, w = img_bgr.shape[:2]
        aspect_ratio = round(w / max(1, h), 4)

        if HAS_CV2:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # OpenCV 5.0 bugfix: Use CV_16S output on uint8 input to avoid crashes
            lap = cv2.Laplacian(gray, cv2.CV_16S)
            sharpness = float(np.var(lap))
            brightness = float(np.mean(gray))
        else:
            gray = 0.299 * img_bgr[:, :, 2] + 0.587 * img_bgr[:, :, 1] + 0.114 * img_bgr[:, :, 0]
            brightness = float(np.mean(gray))
            padded = np.pad(gray, 1, mode="edge")
            lap = (
                padded[2:, 1:-1] + padded[:-2, 1:-1] +
                padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * gray
            )
            sharpness = float(np.var(lap))

        brightness = round(brightness, 2)
        sharpness = round(sharpness, 2)

        is_bad = (
            brightness < self.thresholds.dark_brightness_cutoff or
            brightness > self.thresholds.blown_brightness_cutoff or
            sharpness < self.thresholds.blur_sharpness_cutoff
        )

        return {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "brightness": brightness,
            "sharpness": sharpness,
            "is_bad_frame": is_bad
        }

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        """Extract visual metrics from staged asset."""
        img = safe_load_image(staged_path)
        if img is None:
            return {
                "width": 0,
                "height": 0,
                "aspect_ratio": 1.0,
                "brightness": 0.0,
                "sharpness": 0.0,
                "is_bad_frame": True
            }

        return self.compute_metrics_from_array(img)
