"""Local FFmpeg scene-cut assistance with explicit failure reporting.

Successful results partition the complete video timeline. Cuts are visual-change
candidates, not semantic event boundaries. Short shots are retained. The legacy
``scene_min_duration_s`` setting no longer bypasses video validation or hides
cuts in short clips. FFmpeg and ffprobe must both be available on PATH.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from trace_annotate.extractors.base import BaseExtractor, ExtractorStatus
from trace_annotate.config import MetricThresholds


class SceneDetectionError(RuntimeError):
    """The video could not be completely inspected within the configured limit."""


def _finite_number(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return result


def _run(command: List[str]) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=30,
                                encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SceneDetectionError(f"{command[0]} could not complete: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-2000:] or "no diagnostic output"
        raise SceneDetectionError(f"{command[0]} failed (exit {result.returncode}): {detail}")
    return result


def _video_duration(path: Path) -> float:
    result = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type,duration:format=duration",
        "-of", "json", str(path),
    ])
    try:
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])
        if not streams or streams[0].get("codec_type") != "video":
            raise ValueError("input has no video stream")
        # A longer audio track must not extend a known video-stream duration.
        raw = streams[0].get("duration")
        if raw in (None, "N/A"):
            raw = metadata.get("format", {}).get("duration")
        duration = float(raw)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("video duration is not finite and positive")
    except (TypeError, ValueError, AttributeError, IndexError, KeyError, OverflowError) as exc:
        raise SceneDetectionError(f"ffprobe did not report a usable video duration: {exc}") from exc
    return duration


class SceneDetector(BaseExtractor):
    """Suggest cuts after validating and decoding the whole local video."""

    def __init__(self, thresholds: Optional[MetricThresholds] = None):
        self.thresholds = thresholds or MetricThresholds()

    @property
    def name(self) -> str:
        return "scene"

    def detect_scenes_ffmpeg(
        self,
        video_path: str,
        threshold: Optional[float] = None,
        cap_seconds: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Return a complete partition, or raise instead of returning partial data.

        ``cap_seconds`` is a maximum accepted duration, not permission to omit
        the remainder of a clip. Increase it explicitly for longer inputs. Each
        external command also has a 30-second execution timeout. Timestamps are
        relative to the first decoded video frame; no nominal FPS is assumed.
        """
        th = _finite_number(self.thresholds.scene_threshold if threshold is None
                            else threshold, "threshold", positive=False)
        if not 0 <= th <= 1:
            raise ValueError("threshold must be between 0 and 1")
        cap = _finite_number(self.thresholds.scene_cap_seconds if cap_seconds is None
                             else cap_seconds, "cap_seconds")
        path = Path(video_path).resolve()
        if not path.is_file():
            raise SceneDetectionError(f"Video source is not an existing regular file: {path}")
        duration = _video_duration(path)
        if duration > cap:
            raise SceneDetectionError(
                f"Video duration {duration} seconds exceeds cap_seconds={cap}; "
                "increase the limit explicitly to inspect the complete video")

        result = _run([
            "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "error",
            "-xerror", "-err_detect", "explode", "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"setpts=PTS-STARTPTS,select='gt(scene,{th})',metadata=print:file=-",
            "-f", "null", "-",
        ])
        cuts = {0.0, duration}
        for line in result.stdout.splitlines():
            if "pts_time:" not in line:
                continue
            try:
                timestamp = float(line.split("pts_time:", 1)[1].split()[0])
                if not math.isfinite(timestamp) or not 0 <= timestamp <= duration:
                    raise ValueError("cut falls outside the video timeline")
            except (ValueError, IndexError) as exc:
                raise SceneDetectionError(f"Invalid FFmpeg scene timestamp: {line!r}") from exc
            cuts.add(timestamp)
        ordered = sorted(cuts)
        return [{"start": start, "end": end, "duration": end - start}
                for start, end in zip(ordered, ordered[1:])]

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        """Use decoded media metadata; caller inventory duration is not trusted.

        Failure has an empty candidate array and explicit failed status. Existing
        callers can still read ``scene_sub_segments``; they should additionally
        inspect ``status`` before interpreting an empty result.
        """
        try:
            segments = self.detect_scenes_ffmpeg(staged_path)
        except (SceneDetectionError, ValueError, OSError) as exc:
            return {"status": ExtractorStatus.FAILED.value,
                    "scene_sub_segments": [], "error": str(exc)}
        return {"status": ExtractorStatus.SUCCESS.value,
                "scene_sub_segments": segments,
                "analyzed_duration": segments[-1]["end"]}
