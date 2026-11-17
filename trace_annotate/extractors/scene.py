"""Local FFmpeg scene-cut assistance with explicit failure reporting.

Successful results partition the inspected video timeline. Cuts are visual-change
candidates, not semantic event boundaries. Short shots are retained. The legacy
``scene_min_duration_s`` setting no longer bypasses video validation or hides
cuts in short clips. FFmpeg and ffprobe must both be available on PATH.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

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


def _run(command: List[str], *, cancelled: Optional[Callable[[], bool]] = None,
         timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        if cancelled is None:
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, timeout=timeout,
                                    encoding="utf-8", errors="replace")
        else:
            if cancelled():
                raise InterruptedError("Scene detection was cancelled")
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True,
                                       encoding="utf-8", errors="replace")
            deadline = time.monotonic() + timeout
            try:
                while True:
                    if cancelled():
                        raise InterruptedError("Scene detection was cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        stdout, stderr = process.communicate(timeout=min(.25, remaining))
                        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                raise
    except InterruptedError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SceneDetectionError(f"{command[0]} could not complete: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-2000:] or "no diagnostic output"
        raise SceneDetectionError(f"{command[0]} failed (exit {result.returncode}): {detail}")
    return result


def _video_timeline(path: Path, cancelled: Optional[Callable[[], bool]] = None) -> tuple[float, float, float]:
    result = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type,start_time,duration:format=start_time,duration",
        "-of", "json", str(path),
    ], cancelled=cancelled)
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
        start = float(streams[0].get("start_time", 0))
        if not math.isfinite(start):
            raise ValueError("video start is not finite")
        raw_format_start = metadata.get("format", {}).get("start_time", 0)
        format_start = float(0 if raw_format_start in (None, "N/A") else raw_format_start)
        if not math.isfinite(format_start):
            raise ValueError("container start is not finite")
    except (TypeError, ValueError, AttributeError, IndexError, KeyError, OverflowError) as exc:
        raise SceneDetectionError(f"ffprobe did not report a usable video duration: {exc}") from exc
    return start, format_start, duration


def _video_duration(path: Path) -> float:
    return _video_timeline(path)[2]


def merge_adjacent_cuts(
    cuts: List[tuple[float, float]],
    *,
    window: float = 0.07,
    weak_ratio: float = 0.5,
) -> List[tuple[float, float]]:
    """Collapse duplicate detections while retaining genuinely strong short shots."""
    ordered = sorted((float(timestamp), float(score)) for timestamp, score in cuts)
    clusters: List[List[tuple[float, float]]] = []
    current: List[tuple[float, float]] = []
    for item in ordered:
        if current and item[0] - current[-1][0] > window:
            clusters.append(current)
            current = []
        current.append(item)
    if current:
        clusters.append(current)

    merged: List[tuple[float, float]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        strongest = max(cluster, key=lambda item: item[1])
        threshold = strongest[1] * weak_ratio
        for item in cluster:
            if item == strongest or item[1] >= threshold:
                merged.append(item)
    return merged


class SceneDetector(BaseExtractor):
    """Suggest cuts after validating and decoding the selected local range."""

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
        *,
        window: Optional[tuple[float, float]] = None,
        timeline_start: Optional[float] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[Dict[str, float]]:
        """Return a complete partition, or raise instead of returning partial data.

        ``cap_seconds`` is a maximum accepted duration, not permission to omit
        the remainder of a clip. Increase it explicitly for longer inputs. Each
        external command also has a 30-second execution timeout. Timestamps are
        relative to the first decoded video frame unless ``timeline_start`` is
        supplied. A selected window is scanned at its actual decoded timestamps,
        without assuming a constant frame rate.
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
        stream_start, format_start, duration = _video_timeline(path, cancelled)
        base = 0.0 if timeline_start is None else _finite_number(
            timeline_start, "timeline_start", positive=False)
        # A negative playback offset is invalid, while a positive video offset
        # is possible when an audio stream starts before the first video frame.
        if base < 0:
            raise ValueError("timeline_start must not be negative")
        end = base + duration
        if window is None:
            selected_start, selected_end = base, end
        else:
            if (not isinstance(window, tuple) or len(window) != 2):
                raise ValueError("window must be a finite (start, end) pair")
            requested_start = _finite_number(window[0], "window start", positive=False)
            requested_end = _finite_number(window[1], "window end", positive=False)
            if requested_start < 0 or requested_start >= requested_end:
                raise ValueError("window must be a positive half-open interval")
            selected_start = max(base, requested_start)
            selected_end = min(end, requested_end)
            if selected_start >= selected_end:
                raise SceneDetectionError("Requested window does not overlap the decoded video timeline")
        inspected_duration = selected_end - selected_start
        if inspected_duration > cap:
            raise SceneDetectionError(
                f"Selected video duration {inspected_duration} seconds exceeds cap_seconds={cap}; "
                "increase the limit explicitly to inspect the complete selected range")

        command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "error",
                   "-xerror", "-err_detect", "explode"]
        # Emit the first decoded frame as proof the inspected range has video.
        # It is a sentinel, not a scene cut.
        select = f"select='eq(n,0)+gt(scene,{th})',metadata=print:file=-"
        if timeline_start is None and window is None:
            command += ["-i", str(path)]
            filter_graph = f"setpts=PTS-STARTPTS,{select}"
        else:
            source_start = selected_start - base + stream_start
            source_end = selected_end - base + stream_start
            if window is not None:
                # Both options are input-side: decoding stops at the selected
                # end even when the scene filter emits no cut frames. Input
                # -ss is relative to the container start; trim uses stream PTS.
                input_seek = max(0.0, source_start - format_start)
                command += ["-ss", f"{input_seek:.9f}",
                            "-t", f"{inspected_duration:.9f}"]
            command += ["-copyts", "-i", str(path)]
            trim = (f"trim=start={source_start:.9f}:end={source_end:.9f},"
                    if window is not None else "")
            filter_graph = f"{trim}{select}"
        command += [
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", filter_graph,
            "-f", "null", "-",
        ]
        timeout = max(30.0, min(180.0, inspected_duration / 8.0))
        result = _run(command, cancelled=cancelled, timeout=timeout)
        interior: List[tuple[float, float]] = []
        observed_frame = False
        for line in result.stdout.splitlines():
            if "pts_time:" not in line:
                continue
            try:
                timestamp = float(line.split("pts_time:", 1)[1].split()[0])
                if timeline_start is not None or window is not None:
                    timestamp += base - stream_start
                if not math.isfinite(timestamp) or not selected_start <= timestamp <= selected_end:
                    raise ValueError("cut falls outside the video timeline")
                score = 0.0
                if "lavfi.scene_score=" in line:
                    score = float(line.split("lavfi.scene_score=", 1)[1].split()[0])
            except (ValueError, IndexError) as exc:
                raise SceneDetectionError(f"Invalid FFmpeg scene timestamp: {line!r}") from exc
            if not observed_frame:
                observed_frame = True
                continue
            interior.append((timestamp, score))
        if not observed_frame:
            raise SceneDetectionError("Selected range contains no decoded video frames")
        merged = merge_adjacent_cuts(interior)
        cuts = {selected_start, selected_end, *(timestamp for timestamp, _score in merged)}
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
