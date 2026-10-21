"""
Feature Extractors for Visual, Textual, and Scene Modalities.
"""
from trace_annotate.extractors.base import BaseExtractor
from trace_annotate.extractors.metrics import VisualMetricsExtractor
from trace_annotate.extractors.faces import FaceExtractor
from trace_annotate.extractors.ocr import OCRExtractor
from trace_annotate.extractors.clip_engine import CLIPExtractor
from trace_annotate.extractors.scene import SceneDetector

__all__ = [
    "BaseExtractor",
    "VisualMetricsExtractor",
    "FaceExtractor",
    "OCRExtractor",
    "CLIPExtractor",
    "SceneDetector",
]
