"""
Video Scene Detection Extractor.
Performs scene boundary segmentation using calibrated thresholding (0.35)
to break raw footage into clean, usable micro-segments.
"""
from __future__ import annotations
import os
import subprocess
import json
from typing import Dict, Any, List, Optional

from trace_annotate.extractors.base import BaseExtractor
from trace_annotate.config import MetricThresholds


class SceneDetector(BaseExtractor):
    """Detects scene cut timestamps and segments video into shots."""

    def __init__(self, thresholds: Optional[MetricThresholds] = None):
        self.thresholds = thresholds or MetricThresholds()

    @property
    def name(self) -> str:
        return "scene"

    def detect_scenes_ffmpeg(
        self,
        video_path: str,
        threshold: Optional[float] = None,
        cap_seconds: Optional[float] = None
    ) -> List[Dict[str, float]]:
        """
        Run ffmpeg scene filter to detect cut timestamps.
        Falls back to rule-based segmentation if ffmpeg is unavailable.
        """
        th = threshold if threshold is not None else self.thresholds.scene_threshold
        cap_t = cap_seconds if cap_seconds is not None else self.thresholds.scene_cap_seconds

        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-t", str(cap_t),
            "-i", video_path,
            "-filter_complex", f"select='gt(scene,{th})',metadata=print:file=-",
            "-f", "null", "-"
        ]

        cuts = []
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            for line in p.stdout.splitlines():
                if "pts_time:" in line:
                    parts = line.split("pts_time:")
                    if len(parts) > 1:
                        try:
                            ts = float(parts[1].strip().split()[0])
                            cuts.append(round(ts, 2))
                        except ValueError:
                            pass
        except Exception:
            pass

        cuts = sorted(list(set([0.0] + cuts)))
        segments = []
        for i in range(len(cuts) - 1):
            s_start = cuts[i]
            s_end = cuts[i + 1]
            if s_end - s_start >= 1.0:
                segments.append({"start": s_start, "end": s_end, "duration": round(s_end - s_start, 2)})

        return segments

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        """Extract scene sub-segments if duration > min threshold."""
        duration = float(asset_item.get("duration", 0.0) or 0.0)
        if duration < self.thresholds.scene_min_duration_s:
            return {
                "scene_sub_segments": [{"start": 0.0, "end": duration, "duration": duration}]
            }

        segments = self.detect_scenes_ffmpeg(staged_path)
        if not segments:
            segments = [{"start": 0.0, "end": duration, "duration": duration}]

        return {"scene_sub_segments": segments}
