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

from ..extractors.shot_boundaries import (detect_shot_boundaries, frame_features, pair_metrics,
    _dissolve_fit, _params as boundary_parameters)
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
LONG_MAX_FRAMES = 2048
LONG_MAX_STATIC_FRAMES = 16
LONG_MIN_FRAMES = WINDOW_FRAMES + 1
LONG_ENDPOINT_AFFINE_RESIDUAL = .02
LONG_ENDPOINT_MONOTONE_RESIDUAL = .02
LONG_MAX_STEP_SCORE = boundary_parameters()["cut_threshold"] * 2.5
LONG_MAX_SPLIT_ATTEMPTS = 4
LONG_MIN_PLATFORM_FRAMES = 3
MAX_COMPONENT_FITS = 16
MAX_LONG_REJECTION_RECORDS = 16
KINDS = {"fade_in", "fade_out", "dissolve", "flash"}

# Frozen configuration emitted by commit 72462d0. It remains readable after a
# detector update; execution admission separately requires the current identity.
_LEGACY_CONFIGURATION = {"contract": "full-range-transitions-v1", "width": 96, "height": 64,
    "pixel_format": "gray", "scale_flags": "area", "max_bytes": 134217728, "max_events": 64,
    "window_frames": 256, "overlap_frames": 64, "timeout_seconds": 180,
    "kinds": ["dissolve", "fade_in", "fade_out", "flash"],
    "detector_parameters": {"analysis_width": 96, "hist_bins": 32, "cut_threshold": .3,
        "strong_cut": .6, "gradual_low": 6.382761437908496e-7, "flash_ratio": .5,
        "min_gradual_frames": 3, "dissolve_residual": .2, "dissolve_far": .15,
        "dissolve_relative_residual": .2, "dissolve_min_interior": 3,
        "dissolve_alpha_margin": .1, "dissolve_reversal_tolerance": .03, "neighbour_radius": 6}}


def configuration():
    return {"contract": CONTRACT, "width": WIDTH, "height": HEIGHT,
            "pixel_format": "gray", "scale_flags": "area",
            "max_bytes": MAX_BYTES, "max_events": MAX_EVENTS,
            "window_frames": WINDOW_FRAMES, "overlap_frames": OVERLAP_FRAMES,
            "timeout_seconds": TIMEOUT_SECONDS,
            "long_dissolve": {"max_frames": LONG_MAX_FRAMES,
                "max_static_frames": LONG_MAX_STATIC_FRAMES, "min_frames": LONG_MIN_FRAMES,
                "endpoint_affine_residual": LONG_ENDPOINT_AFFINE_RESIDUAL,
                "endpoint_monotone_residual": LONG_ENDPOINT_MONOTONE_RESIDUAL,
                "max_step_score": LONG_MAX_STEP_SCORE,
                "max_split_attempts": LONG_MAX_SPLIT_ATTEMPTS,
                "min_platform_frames": LONG_MIN_PLATFORM_FRAMES,
                "max_rejection_records": MAX_LONG_REJECTION_RECORDS,
                "filter_scope": "all_global_dissolves",
                "image_precision": "rounded_uint8", "max_component_fits": MAX_COMPONENT_FITS},
            "kinds": sorted(KINDS), "detector_parameters": boundary_parameters(gradual_low=MINIMUM_STEP_SCORE)}


def validate_configuration(value):
    """Accept only exact current or frozen legacy detector configurations."""
    try:
        encoded = S.canonical_json(value)
    except (TypeError, ValueError, WorkspaceError) as exc:
        raise SchemaInvalid("SCHEMA_INVALID", "Transition configuration is unsupported") from exc
    if encoded not in (S.canonical_json(configuration()), S.canonical_json(_LEGACY_CONFIGURATION)):
        raise SchemaInvalid("SCHEMA_INVALID", "Transition configuration is unsupported")
    return copy.deepcopy(value)


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
        component_keys = ("fit_interval", "endpoint_distance", "relative_residual",
                          "interior_mixture_frames", "measurement_method")
        components = previous["metrics"].get("component_fits", []) + [
            {key: event["metrics"][key] for key in component_keys if key in event["metrics"]}
            for event in (previous, proposal)]
        unique = []
        for component in components:
            if component and component not in unique:
                unique.append(component)
        # A newly measured long run can cover every frame in the merged interval.
        # In that case its metrics supersede the earlier component-window fit.
        fitted = proposal["metrics"].get("fit_interval", {})
        prior_truncated = previous["metrics"].get("component_fits_truncated", False)
        if (int(fitted.get("start_frame", end + 1)) <= min(start, left)
                and int(fitted.get("end_frame", start - 1)) >= max(end, right)):
            previous["metrics"] = copy.deepcopy(proposal["metrics"])
            previous["uncertainty"] = proposal["uncertainty"]
        previous["metrics"]["component_fits"] = unique[:MAX_COMPONENT_FITS]
        previous["metrics"]["component_fits_truncated"] = (
            prior_truncated
            or len(unique) > MAX_COMPONENT_FITS)
        previous["metrics"]["window_clipped"] |= proposal["metrics"]["window_clipped"]
        # Fit statistics still describe the explicitly identified first measured
        # interval, rather than implying a new fit over the merged interval.
        previous["metrics"]["merged_interval"] = True
        if previous["metrics"]["window_clipped"] and "窗口" not in previous["uncertainty"]:
            previous["uncertainty"] += "；跨分析窗口的完整区间边界待复核"
        return
    events.append(proposal)


def _photometric_metrics(first, last):
    """Measure brightness/contrast and monotone-luminance endpoint explanations."""
    first = np.asarray(first, dtype=np.float64)
    last = np.asarray(last, dtype=np.float64)
    centered = first - first.mean()
    energy = float(np.sum(centered * centered))
    gain = max(0.0, float(np.sum(centered * (last - last.mean())) / energy)) if energy else 0.0
    affine_residual = float(np.abs(last - (last.mean() + gain * centered)).mean() / 255)
    # Exposure/gamma changes can satisfy a two-endpoint mixture approximation.
    # Test an observed nondecreasing luminance mapping as a competing explanation.
    bins = np.rint(first).clip(0, 255).astype(np.uint8)
    counts = np.bincount(bins.ravel(), minlength=256)
    sums = np.bincount(bins.ravel(), weights=last.ravel(), minlength=256)
    occupied = counts > 0
    lookup = np.zeros(256)
    lookup[occupied] = np.maximum.accumulate(sums[occupied] / counts[occupied])
    monotone_residual = float(np.abs(last - lookup[bins]).mean() / 255)
    diagnostics = {"endpoint_affine_residual": round(affine_residual, 6),
                   "endpoint_monotone_residual": round(monotone_residual, 6)}
    if affine_residual < LONG_ENDPOINT_AFFINE_RESIDUAL:
        return diagnostics, "brightness_or_contrast"
    if monotone_residual < LONG_ENDPOINT_MONOTONE_RESIDUAL:
        return diagnostics, "monotone_luminance_mapping"
    return diagnostics, None


def _long_dissolve_event(run, *, clipped, previous_frame):
    """Fit every retained uint8 frame with the global photometric exclusions."""
    diagnostics, reason = _photometric_metrics(run[0]["gray"], run[-1]["gray"])
    if reason:
        return None, reason, diagnostics
    fit = _dissolve_fit(run, 0, len(run) - 1)
    if fit is None:
        return None, "mixture_fit_rejected", diagnostics
    start, end = run[0], run[-1]
    fit.update(analysis_scope="full_range", window_clipped=clipped,
        fit_interval={"start_frame": start["frame_index"], "end_frame": end["frame_index"]},
        merged_interval=False, measurement_method="full_long_run_mixture",
        **diagnostics,
        observed_mixture_frames=len(run), image_precision="rounded_uint8")
    return {"kind": "dissolve", "anchor_frame": end["frame_index"],
        "previous_frame": previous_frame,
        "representative_frame": end["frame_index"], "split_point": end["frame_index"],
        "interval": {"start_frame": start["frame_index"], "end_frame": end["frame_index"],
                     "start_seconds": start["seconds"], "end_seconds": end["seconds"]},
        "confidence": .85, "needs_review": True, "uncertainty_frames": len(run) - 1,
        "uncertainty": "长区间混合候选，请逐帧核对运动、遮挡与完整边界" + (
            "；分析范围或帧数预算可能截断转场" if clipped else ""), "metrics": fit}, None, diagnostics


def _split_platforms(run):
    """Return at most four longest observed static plateaus with viable sides."""
    selected = []
    start = 0
    for index in range(1, len(run) + 1):
        if index < len(run) and np.array_equal(run[index]["gray"], run[start]["gray"]):
            continue
        end = index - 1
        length = index - start
        if (length >= LONG_MIN_PLATFORM_FRAMES and start + 1 >= LONG_MIN_FRAMES
                and len(run) - end >= LONG_MIN_FRAMES):
            selected.append((start, end))
            selected.sort(key=lambda pair: (-(pair[1] - pair[0] + 1), pair[0]))
            del selected[LONG_MAX_SPLIT_ATTEMPTS:]
        start = index
    return selected


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
    long_run, long_previous = [], None
    long_last_change = -1
    long_left_clipped = False
    long_segment_open, long_static_steps = True, 0
    long_stats = {"evaluated_runs": 0, "accepted_runs": 0, "rejected_runs": {},
                  "budget_windows": 0, "frame_gaps": 0, "max_buffer_frames": 0,
                  "max_buffer_pixel_bytes": 0, "split_attempts": 0,
                  "split_accepted_pairs": 0, "split_rejected_sides": {},
                  "short_dissolve_rejections": {},
                  "rejected_fit_measurements": [], "rejected_fit_measurements_truncated": False}

    def record_rejection(part, reason, diagnostics, scope="long_run"):
        records = long_stats["rejected_fit_measurements"]
        if len(records) >= MAX_LONG_REJECTION_RECORDS:
            long_stats["rejected_fit_measurements_truncated"] = True
            return
        records.append({"start_frame": part[0]["frame_index"], "end_frame": part[-1]["frame_index"],
                        "reason": reason, "source_scope": scope, **diagnostics})

    def retain(event):
        nonlocal dropped
        _retain(retained, event)
        if len(retained) > max_events:
            retained.pop()
            dropped += 1

    def finish_long(clipped_right=False):
        nonlocal long_run, long_last_change, long_left_clipped
        active = long_run if clipped_right else long_run[:long_last_change + 1]
        if len(active) >= LONG_MIN_FRAMES:
            _check(cancelled)
            long_stats["evaluated_runs"] += 1
            previous_frame = None if long_left_clipped else active[0].get("previous_frame")
            event, reason, diagnostics = _long_dissolve_event(active,
                clipped=long_left_clipped or clipped_right, previous_frame=previous_frame)
            _check(cancelled)
            if event is not None:
                long_stats["accepted_runs"] += 1
                retain(event)
            else:
                rejected = long_stats["rejected_runs"]
                rejected[reason] = rejected.get(reason, 0) + 1
                record_rejection(active, reason, diagnostics)
                # Quantization needs the normal 16-frame gap allowance. If it
                # joined two different mixtures, try a finite static-boundary
                # fallback only after their whole-run hypothesis was rejected.
                for left, right in _split_platforms(active):
                    long_stats["split_attempts"] += 1
                    sides = []
                    for part, clipped, before in (
                        (active[:left + 1], long_left_clipped, previous_frame),
                        (active[right:], clipped_right, active[right].get("previous_frame"))):
                        _check(cancelled)
                        measured, failure, details = _long_dissolve_event(part, clipped=clipped, previous_frame=before)
                        _check(cancelled)
                        sides.append(measured)
                        if failure:
                            failures = long_stats["split_rejected_sides"]
                            failures[failure] = failures.get(failure, 0) + 1
                            record_rejection(part, failure, details)
                    if not all(side is not None for side in sides):
                        continue
                    platform = {"start_frame": active[left]["frame_index"],
                                "end_frame": active[right]["frame_index"], "frames": right - left + 1}
                    for measured in sides:
                        measured["metrics"].update(measurement_method="split_long_run_mixture",
                            split_platform=platform, parent_run_fit_rejection=reason,
                            parent_run_interval={"start_frame": active[0]["frame_index"],
                                                 "end_frame": active[-1]["frame_index"]})
                        measured["uncertainty"] += "；相邻混合按已观察静帧平台分段，边界待复核"
                        retain(measured)
                    long_stats["accepted_runs"] += 2
                    long_stats["split_accepted_pairs"] += 1
                    break
        long_run, long_last_change, long_left_clipped = [], -1, False

    def observe_long(sample):
        nonlocal long_run, long_last_change, long_left_clipped, long_previous
        nonlocal long_segment_open, long_static_steps
        compact = {key: sample.get(key) for key in ("frame_index", "seconds", "source_pts", "proxy_pts")}
        compact["frame_index"] = str(sample["frame_index"])
        compact["previous_frame"] = long_previous[0]["frame_index"] if long_previous is not None else None
        compact["gray"] = np.rint(sample["gray"]).clip(0, 255).astype(np.uint8)
        if long_previous is not None:
            previous, features = long_previous
            score = pair_metrics(features, sample)["score"]
            if score > LONG_MAX_STEP_SCORE:
                finish_long()
                long_segment_open, long_static_steps = False, 0
            elif score > 0:
                long_static_steps = 0
                if not long_run:
                    long_run = [previous]
                    long_left_clipped = long_segment_open
                long_run.append(compact)
                long_last_change = len(long_run) - 1
            else:
                long_static_steps += 1
                if long_static_steps > LONG_MAX_STATIC_FRAMES:
                    long_segment_open = False
                if long_run:
                    long_run.append(compact)
                    if len(long_run) - 1 - long_last_change > LONG_MAX_STATIC_FRAMES:
                        finish_long()
            long_stats["max_buffer_frames"] = max(long_stats["max_buffer_frames"], len(long_run))
            long_stats["max_buffer_pixel_bytes"] = max(long_stats["max_buffer_pixel_bytes"],
                                                        len(long_run) * compact["gray"].nbytes)
            if len(long_run) >= LONG_MAX_FRAMES:
                tail = long_run[-OVERLAP_FRAMES:]
                changed = long_last_change - (len(long_run) - len(tail))
                finish_long(clipped_right=True)
                long_run, long_last_change, long_left_clipped = tail, changed, True
                long_stats["budget_windows"] += 1
        long_previous = compact, sample

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
            if event["kind"] == "dissolve":
                left = start - int(block[0]["frame_index"])
                right = end - int(block[0]["frame_index"])
                part = block[left:right + 1]
                _check(cancelled)
                diagnostics, reason = _photometric_metrics(part[0]["gray"], part[-1]["gray"])
                _check(cancelled)
                if reason:
                    rejected = long_stats["short_dissolve_rejections"]
                    rejected[reason] = rejected.get(reason, 0) + 1
                    record_rejection(part, reason, diagnostics, scope="short_global_window")
                    continue
                event["metrics"].update(diagnostics)
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
            retain(event)

    for item in samples:
        _check(cancelled)
        index = int(item["frame_index"])
        if last_index is not None and index != last_index + 1:
            # Do not interpret a sampling gap as one physical video frame.
            analyse(last=True)
            buffer.clear()
            first_index = None
            finish_long(clipped_right=True)
            long_previous = None
            long_segment_open, long_static_steps = True, 0
            long_stats["frame_gaps"] += 1
        if first_index is None:
            first_index = index
        last_index = index
        sample = {key: item.get(key) for key in ("frame_index", "seconds", "source_pts", "proxy_pts")}
        sample.update({key: item[key] for key in ("gray", "luma", "hist")}
                      if "gray" in item else frame_features(item["frame"]))
        observe_long(sample)
        buffer.append(sample)
        observed += 1
        if len(buffer) == WINDOW_FRAMES:
            analyse()
            for _ in range(WINDOW_FRAMES - OVERLAP_FRAMES):
                buffer.popleft()
    _check(cancelled)
    analyse(last=True)
    # A short final plateau is also consistent with quantization inside a very
    # slow dissolve. A hold longer than the allowed gap closes the run earlier.
    finish_long(clipped_right=bool(long_run))
    retained.sort(key=lambda event: int(event["anchor_frame"]))
    return {"events": retained, "events_truncated": bool(dropped), "dropped_events": dropped,
            "measured_frames": observed, "windows": windows,
            "unknown_window_events": unknown, "contract": CONTRACT,
            "long_run_analysis": long_stats}


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
