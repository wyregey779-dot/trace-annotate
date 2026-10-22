"""
Configuration management for TraceAnnotate.
Defines thresholds, hardware limits, cold-storage safety policies, and schema settings.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class StagingConfig:
    """Batch size, free-space threshold and copy verification settings."""
    group_size: int = 4                    # Number of files processed in a single staging chunk
    free_disk_guard_bytes: int = 2 * 1024**3 # 2GB safety guard on local disk
    read_only_source: bool = True          # Compatibility setting; does not change filesystem permissions
    verify_checksum: bool = True           # Verify size + head1mb md5 upon ingestion


@dataclass
class MetricThresholds:
    """Thresholds for visual metrics, near-duplicate clustering, and quality scoring."""
    dhash_hamming_threshold: int = 10     # Max hamming distance for similar cluster grouping
    scene_threshold: float = 0.35          # ffmpeg scene detection threshold
    scene_min_duration_s: float = 10.0     # Minimum duration in seconds to trigger scene splitting
    scene_cap_seconds: float = 180.0       # Max video duration processed for scene detection
    dark_brightness_cutoff: float = 15.0   # Bad frame: too dark
    blown_brightness_cutoff: float = 245.0 # Bad frame: overexposed
    blur_sharpness_cutoff: float = 8.0     # Bad frame: blurry (Laplacian variance)
    face_min_confidence: float = 0.70      # Minimum confidence for face detection
    face_min_height_ratio: float = 0.10    # Face bounding box height / image height for recognizable face
    ocr_min_height_ratio: float = 0.04     # Text box height / image height for readable text
    ocr_min_confidence: float = 0.85       # Minimum confidence for OCR readable text
    clip_binary_confidence: float = 0.85   # Softmax cutoff for binary zero-shot (indoor/outdoor, day/night)
    clip_multiclass_confidence: float = 0.70 # Softmax cutoff for multiclass zero-shot (weather, screen)
    clip_entity_mismatch_cutoff: float = 0.25 # Entity similarity below this triggers conflict review


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    source_dir: str = ""                   # Path to raw footage (e.g., external drive)
    staging_dir: str = "./staging"         # Local fast scratch directory for chunked processing
    output_dir: str = "./output"           # Directory for generated annotations, inventory, logs
    vocab_file: Optional[str] = None       # Path to controlled vocabulary JSON
    staging: StagingConfig = field(default_factory=StagingConfig)
    thresholds: MetricThresholds = field(default_factory=MetricThresholds)
    enable_clip: bool = True
    enable_ocr: bool = True
    enable_faces: bool = True
    enable_scene_detect: bool = True

    @classmethod
    def from_json(cls, path: str) -> PipelineConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        staging_data = data.pop("staging", {})
        thresh_data = data.pop("thresholds", {})
        return cls(
            staging=StagingConfig(**staging_data),
            thresholds=MetricThresholds(**thresh_data),
            **data
        )

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "source_dir": self.source_dir,
            "staging_dir": self.staging_dir,
            "output_dir": self.output_dir,
            "vocab_file": self.vocab_file,
            "enable_clip": self.enable_clip,
            "enable_ocr": self.enable_ocr,
            "enable_faces": self.enable_faces,
            "enable_scene_detect": self.enable_scene_detect,
            "staging": {
                "group_size": self.staging.group_size,
                "free_disk_guard_bytes": self.staging.free_disk_guard_bytes,
                "read_only_source": self.staging.read_only_source,
                "verify_checksum": self.staging.verify_checksum,
            },
            "thresholds": {
                "dhash_hamming_threshold": self.thresholds.dhash_hamming_threshold,
                "scene_threshold": self.thresholds.scene_threshold,
                "scene_min_duration_s": self.thresholds.scene_min_duration_s,
                "scene_cap_seconds": self.thresholds.scene_cap_seconds,
                "dark_brightness_cutoff": self.thresholds.dark_brightness_cutoff,
                "blown_brightness_cutoff": self.thresholds.blown_brightness_cutoff,
                "blur_sharpness_cutoff": self.thresholds.blur_sharpness_cutoff,
                "face_min_confidence": self.thresholds.face_min_confidence,
                "face_min_height_ratio": self.thresholds.face_min_height_ratio,
                "ocr_min_height_ratio": self.thresholds.ocr_min_height_ratio,
                "ocr_min_confidence": self.thresholds.ocr_min_confidence,
                "clip_binary_confidence": self.thresholds.clip_binary_confidence,
                "clip_multiclass_confidence": self.thresholds.clip_multiclass_confidence,
                "clip_entity_mismatch_cutoff": self.thresholds.clip_entity_mismatch_cutoff,
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
