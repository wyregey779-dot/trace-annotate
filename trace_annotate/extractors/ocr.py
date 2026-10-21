"""
OCR Extractor with RapidOCR & fallback support.
Safely extracts text, storefront signboards, and UI text elements with array checks.
"""
from __future__ import annotations
import os
import re
from typing import Dict, Any, List, Optional
import numpy as np

from trace_annotate.extractors.base import BaseExtractor
from trace_annotate.core.hasher import safe_load_image
from trace_annotate.config import MetricThresholds


class OCRExtractor(BaseExtractor):
    """Extracts on-screen and sign text using RapidOCR."""

    def __init__(self, thresholds: Optional[MetricThresholds] = None):
        self.thresholds = thresholds or MetricThresholds()
        self.engine = None
        self._init_engine()

    @property
    def name(self) -> str:
        return "ocr"

    def _init_engine(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
        except ImportError:
            self.engine = None

    def extract_text(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Extract text elements from an image array."""
        h, w = img_bgr.shape[:2]

        if self.engine is None:
            return {
                "ocr_text": "",
                "ocr_readable": "",
                "ocr_clean": "",
                "ocr_boxes": []
            }

        try:
            result, _ = self.engine(img_bgr)
        except Exception:
            result = None

        if not result:
            return {
                "ocr_text": "",
                "ocr_readable": "",
                "ocr_clean": "",
                "ocr_boxes": []
            }

        all_texts = []
        readable_texts = []
        clean_texts = []
        boxes_out = []

        # Battle-tested fix: RapidOCR boxes can be numpy ndarray, never do `box or []`
        for item in result:
            if not item or len(item) < 3:
                continue
            box, text, score = item[0], str(item[1]).strip(), float(item[2])
            all_texts.append(text)

            # Box height ratio calculation
            if box is not None and len(box) >= 4:
                pts = np.array(box)
                ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                box_h = ymax - ymin
                height_ratio = box_h / max(1, h)
            else:
                height_ratio = 0.0

            is_readable = (
                score >= self.thresholds.ocr_min_confidence and
                height_ratio >= self.thresholds.ocr_min_height_ratio
            )
            if is_readable:
                readable_texts.append(text)

            cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
            if len(cleaned) >= 2:
                clean_texts.append(cleaned)

            boxes_out.append({
                "text": text,
                "score": round(score, 3),
                "is_readable": is_readable,
                "box": box.tolist() if isinstance(box, np.ndarray) else box
            })

        return {
            "ocr_text": " ".join(all_texts),
            "ocr_readable": " ".join(readable_texts),
            "ocr_clean": " ".join(clean_texts),
            "ocr_boxes": boxes_out
        }

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        img = safe_load_image(staged_path)
        if img is None:
            return {
                "ocr_text": "",
                "ocr_readable": "",
                "ocr_clean": "",
                "ocr_boxes": []
            }
        return self.extract_text(img)
