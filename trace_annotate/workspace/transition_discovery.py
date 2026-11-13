"""Budgeted, frame-map-bound transition proposals across the selected video range.

The scan retains one small grayscale stream and overlapping feature windows.
Full-size before/anchor/interval images are decoded separately only for retained
proposals. All outputs describe the admitted playback and require review.
"""
from __future__ import annotations

import copy
from collections import deque
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from ..extractors.shot_boundaries import detect_shot_boundaries, frame_features, _params as boundary_parameters
from ..temporal import preparation as P
from ..temporal.media import stream_sha256
from . import schema as S
from .errors import DependencyUnavailable, SchemaInvalid, WorkspaceError
from .frames import FrameService, _binding
from .store import _safe

CONTRACT = "full-range-transitions-v1"
WIDTH, HEIGHT = 96, 64
FRAME_BYTES = WIDTH * HEIGHT
MAX_BYTES = 128 * 1024 * 1024
MAX_EVENTS = 64
WINDOW_FRAMES, OVERLAP_FRAMES = 256, 64
TIMEOUT_SECONDS = 180
# Pair scores average spatial change. One changed gray pixel must be enough to
# propose a slow run; the subsequent mixture fit decides whether it is retained.
MINIMUM_STEP_SCORE = 1 / (255 * FRAME_BYTES)
KINDS = {"fade_in", "fade_out", "dissolve", "flash"}


def configuration():
    return {"contract": CONTRACT, "width": WIDTH, "height": HEIGHT,
            "pixel_format": "gray", "scale_flags": "area",
            "max_bytes": MAX_BYTES, "max_events": MAX_EVENTS,
            "window_frames": WINDOW_FRAMES, "overlap_frames": OVERLAP_FRAMES,
            "timeout_seconds": TIMEOUT_SECONDS,
            "kinds": sorted(KINDS), "detector_parameters": boundary_parameters(gradual_low=MINIMUM_STEP_SCORE)}


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise WorkspaceError("CANCELLED", "Full-range transition scan was cancelled")


def _bounds(event):
    interval = event.get("interval") or {}
    return (int(interval.get("start_frame", event["anchor_frame"])),
            int(interval.get("end_frame", event["anchor_frame"])))


def _retain(events, proposal):
    """Merge overlapping same-kind proposals produced by adjacent windows."""
    start, end = _bounds(proposal)
    for previous in events:
        left, right = _bounds(previous)
        if previous["kind"] != proposal["kind"] or end < left or start > right:
            continue
        if proposal["kind"] == "flash":
            return
        earlier = previous if left <= start else proposal
        later = previous if right >= end else proposal
        previous["interval"] = {
            "start_frame": earlier["interval"]["start_frame"],
            "start_seconds": earlier["interval"]["start_seconds"],
            "end_frame": later["interval"]["end_frame"],
            "end_seconds": later["interval"]["end_seconds"]}
        previous["previous_frame"] = earlier.get("previous_frame")
        previous["anchor_frame"] = previous["representative_frame"] = later["anchor_frame"]
        previous["split_point"] = previous["anchor_frame"]
        previous["uncertainty_frames"] = max(1, max(end, right) - min(start, left))
        previous["metrics"]["window_clipped"] |= proposal["metrics"]["window_clipped"]
        # Fit statistics still describe the explicitly identified first measured
        # interval, rather than implying a new fit over the merged interval.
        previous["metrics"]["merged_interval"] = True
        if previous["metrics"]["window_clipped"] and "窗口" not in previous["uncertainty"]:
            previous["uncertainty"] += "；跨分析窗口的完整区间边界待复核"
        return
    events.append(proposal)


def discover_events(samples: Iterable[Dict[str, Any]], *, max_events=MAX_EVENTS,
                    cancelled: Optional[Callable[[], bool]] = None):
    """Measure every adjacent pair using bounded overlapping feature windows.

    Events touching an internal window edge retain explicit interval uncertainty.
    Long intervals can merge across windows, while their exact limits remain a
    reviewer decision. Unknown sustained motion is counted without labelling it
    as a known transition.
    """
    if type(max_events) is not int or not 1 <= max_events <= MAX_EVENTS:
        raise SchemaInvalid("SCHEMA_INVALID", "Transition event budget is invalid")
    retained, buffer = [], deque()
    windows = unknown = observed = dropped = 0
    first_index = last_index = None

    def analyse(last=False):
        nonlocal windows, unknown, dropped
        if len(buffer) < 2:
            return
        block = list(buffer)
        measured = detect_shot_boundaries(block, gradual_low=MINIMUM_STEP_SCORE)
        windows += 1
        for value in measured["events"]:
            if value["kind"] == "unknown_transition":
                unknown += 1
            if value["kind"] not in KINDS:
                continue
            event = copy.deepcopy(value)
            start, end = _bounds(event)
            clipped = ((int(block[0]["frame_index"]) != first_index and start == int(block[0]["frame_index"]))
                       or (not last and end == int(block[-1]["frame_index"])))
            event["needs_review"] = True
            event["uncertainty"] = "低分辨率全域扫描候选，请核对逐帧证据" + (
                "；分析窗口边缘可能截断转场区间" if clipped else "")
            event["metrics"].update(analysis_scope="full_range", window_clipped=clipped,
                                    fit_interval={"start_frame": str(start), "end_frame": str(end)},
                                    merged_interval=False)
            # At most MAX_EVENTS are retained, including while scanning noisy
            # long material. Duplicate/overlapping windows do not consume slots.
            _retain(retained, event)
            if len(retained) > max_events:
                retained.pop()
                dropped += 1

    for item in samples:
        _check(cancelled)
        index = int(item["frame_index"])
        if last_index is not None and index != last_index + 1:
            # Do not interpret a sampling gap as one physical video frame.
            analyse(last=True)
            buffer.clear()
            first_index = None
        if first_index is None:
            first_index = index
        last_index = index
        sample = {key: item.get(key) for key in ("frame_index", "seconds", "source_pts", "proxy_pts")}
        sample.update({key: item[key] for key in ("gray", "luma", "hist")}
                      if "gray" in item else frame_features(item["frame"]))
        buffer.append(sample)
        observed += 1
        if len(buffer) == WINDOW_FRAMES:
            analyse()
            for _ in range(WINDOW_FRAMES - OVERLAP_FRAMES):
                buffer.popleft()
    _check(cancelled)
    analyse(last=True)
    retained.sort(key=lambda event: int(event["anchor_frame"]))
    return {"events": retained, "events_truncated": bool(dropped), "dropped_events": dropped,
            "measured_frames": observed, "windows": windows,
            "unknown_window_events": unknown, "contract": CONTRACT}


def _range(service, media_id, total, window):
    if window is None:
        return 0, total
    if (not isinstance(window, dict) or set(window) != {"start", "end"}
            or any(S.rational(window[key]) != window[key] for key in window)
            or S.rational_value(window["start"]) < 0
            or S.rational_value(window["start"]) >= S.rational_value(window["end"])):
        raise SchemaInvalid("SCHEMA_INVALID", "Transition window must be a reduced half-open interval")
    def index_at(value):
        low, high = 0, total
        while low < high:
            mid = (low + high) // 2
            if S.rational_value(service.row(media_id, mid)["seconds"]) < S.rational_value(value):
                low = mid + 1
            else:
                high = mid
        return low
    return index_at(window["start"]), index_at(window["end"])


def scan_transitions(store, media_id, output: Path, *, window=None, cancelled=None,
                     ffmpeg=None, max_bytes=MAX_BYTES, timeout=TIMEOUT_SECONDS):
    """Decode sequential gray frames, then bind proposals to verified map rows."""
    if type(max_bytes) is not int or not FRAME_BYTES <= max_bytes <= MAX_BYTES:
        raise SchemaInvalid("SCHEMA_INVALID", "Transition byte budget is invalid")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise SchemaInvalid("SCHEMA_INVALID", "Transition timeout must be finite and positive")
    started = time.monotonic()
    output = Path(output)
    _safe(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "frames.gray"
    def control():
        _check(cancelled)
        if time.monotonic() - started >= timeout:
            raise WorkspaceError("FRAME_DECODE_TIMEOUT", "Full-range transition scan exceeded its time budget")
        if raw.exists() and raw.stat().st_size > max_bytes:
            raise WorkspaceError("FRAME_TOO_LARGE", "Transition scan exceeded its byte budget")
        return False
    control()
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "FFmpeg is required for transition discovery")
    media, playback, playback_hash = _binding(store, media_id, "playback_path")
    if stream_sha256(playback, cancelled=control) != playback_hash:
        raise WorkspaceError("RECOVERY_REQUIRED", "Transition playback bytes changed")
    service = getattr(store, "_frame_service", None)
    if service is None:
        service = store._frame_service = FrameService(store)
    page = service.page(media_id, 0, 1)
    total, map_hash = int(page["total"]), page["frame_map_sha256"]
    start, requested_end = _range(service, media_id, total, window)
    if window is not None and S.rational_value(window["end"]) > S.rational_value(media["duration"]):
        raise SchemaInvalid("SCHEMA_INVALID", "Transition window exceeds playback duration")
    end = min(requested_end, start + max_bytes // FRAME_BYTES)
    rows = []
    for offset in range(start, end, 1000):
        control()
        part = service.page(media_id, offset, min(1000, end - offset))
        if part["frame_map_sha256"] != map_hash:
            raise WorkspaceError("RECOVERY_REQUIRED", "Transition frame map identity changed")
        rows.extend(part["frames"])
    count = end - start
    if count:
        command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", "-xerror",
                   "-protocol_whitelist", "file,pipe", "-err_detect", "explode",
                   "-threads", "1", "-filter_threads", "1", "-i", str(playback), "-map", "0:v:0",
                   "-vf", f"trim=start_frame={start}:end_frame={end},scale={WIDTH}:{HEIGHT}:flags=area,format=gray",
                   "-fps_mode", "passthrough", "-an", "-sn", "-dn", "-frames:v", str(count),
                   "-pix_fmt", "gray", "-f", "rawvideo", str(raw)]
        try:
            P._run(command, output, "scan", control)
        except (P.MediaPreparationError, OSError) as exc:
            raise WorkspaceError("FRAME_DECODE_FAILED", f"Transition decoder failed: {str(exc)[-500:]}") from exc
        control()
        if not raw.is_file() or raw.stat().st_size != count * FRAME_BYTES:
            raise WorkspaceError("FRAME_DECODE_FAILED", "Transition decoded count differs from the admitted map")
    def measured():
        if not count:
            return
        with raw.open("rb") as incoming:
            for row in rows:
                control()
                frame = np.frombuffer(incoming.read(FRAME_BYTES), dtype=np.uint8).reshape(HEIGHT, WIDTH)
                yield {**row, "seconds": float(S.rational_value(row["seconds"])), "frame": frame}
    found = discover_events(measured(), cancelled=control)
    for event in found["events"]:
        if event["kind"] == "flash":
            anchor = int(event["anchor_frame"])
            row = service.row(media_id, anchor)
            following = service.row(media_id, anchor + 1)
            event["recovery_frame"] = following["frame_index"]
            event["interval"] = {"start_frame": str(anchor), "end_frame": str(anchor),
                "start_seconds": float(S.rational_value(row["seconds"])),
                "end_seconds": float(S.rational_value(following["seconds"]))}
        interval = event.get("interval")
        if interval is not None:
            initial_row = service.row(media_id, int(interval["start_frame"]))
            final_index = int(interval["end_frame"])
            endpoint = (service.row(media_id, final_index + 1)["seconds"]
                        if final_index + 1 < total else media["video_end"])
            if window is not None and S.rational_value(endpoint) > S.rational_value(window["end"]):
                endpoint = window["end"]
            # Public annotation windows are half-open, including the complete
            # final measured transition frame and excluding the next frame.
            interval["end_seconds"] = float(S.rational_value(endpoint))
            event["time_window"] = {"start": {"seconds": initial_row["seconds"]},
                                    "end": {"seconds": endpoint}}
            if final_index == end - 1 and end < total:
                event["metrics"]["window_clipped"] = True
                event["uncertainty"] += "；所选范围或字节预算在转场中结束，完整终点待复核"
    if stream_sha256(playback, cancelled=control) != playback_hash or service.page(media_id, 0, 1)["frame_map_sha256"] != map_hash:
        raise WorkspaceError("RECOVERY_REQUIRED", "Transition input changed during scan")
    result = {**found, "status": "success", "media_id": media_id,
              "playback_sha256": playback_hash, "frame_map_sha256": map_hash,
              "decoded_frames": count, "raw_sha256": stream_sha256(raw, cancelled=control) if count else None,
              "coverage": {"complete": end == requested_end, "requested_frames": requested_end - start,
                           "first_frame": str(start) if count else None,
                           "last_frame": str(end - 1) if count else None,
                           "reason": None if end == requested_end else "raw_frame_byte_budget"},
              "configuration": configuration(), "time_domain": "playback_seconds"}
    measurement = output / "discovery.json"
    measurement.write_bytes(S.canonical_json(result))
    return {**result, "measurement_path": str(measurement),
            "measurement_sha256": stream_sha256(measurement, cancelled=control)}
