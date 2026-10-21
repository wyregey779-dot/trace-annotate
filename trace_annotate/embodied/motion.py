"""
Motion Analysis & Boundary Snapping Engine for Video Annotation.
Extracts localized motion energy, detects idle pauses (>2s),
and snaps manual boundary clicks to the exact frame where hand motion begins or ends.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_motion_energy(
    video_path: str,
    sample_fps: float = 4.0,
    max_duration_s: Optional[float] = None,
    focus_lower_half: bool = True
) -> List[Dict[str, float]]:
    """
    Extract temporal motion energy profile from video.
    Focuses on the lower half/center of the frame where hand-object interaction occurs.
    
    Returns:
        List of dicts: [{'timestamp': ts, 'energy': float, 'is_moving': bool}, ...]
    """
    if not HAS_CV2 or not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(orig_fps / sample_fps)))

    prev_gray = None
    motion_series: List[Dict[str, float]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_s = frame_idx / orig_fps
        if max_duration_s and current_time_s > max_duration_s:
            break

        if frame_idx % frame_interval == 0:
            # Resize for fast processing
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Focus on lower half for egocentric hand action
            if focus_lower_half:
                gray = gray[int(120 * 0.35) :, :]

            if prev_gray is not None:
                # Frame differencing motion metric
                diff = cv2.absdiff(gray, prev_gray)
                energy = float(np.mean(diff))
            else:
                energy = 0.0

            prev_gray = gray
            motion_series.append({
                "timestamp": round(current_time_s, 2),
                "energy": round(energy, 2),
                "is_moving": energy > 4.5  # Standard hand movement cutoff
            })

        frame_idx += 1

    cap.release()
    return motion_series


def propose_idle_pauses(
    motion_series: List[Dict[str, float]],
    min_pause_s: float = 2.0
) -> List[Dict[str, float]]:
    """
    Scan motion profile and propose idle candidates where motion is sustained below threshold.
    Rule: pauses < 2s do not split working, but pauses >= 2s are proposed as idle.
    """
    if not motion_series:
        return []

    idle_candidates = []
    in_idle = False
    start_ts = 0.0

    for item in motion_series:
        ts = item["timestamp"]
        moving = item["is_moving"]

        if not moving:
            if not in_idle:
                in_idle = True
                start_ts = ts
        else:
            if in_idle:
                in_idle = False
                dur = ts - start_ts
                if dur >= min_pause_s:
                    idle_candidates.append({
                        "start": round(start_ts, 2),
                        "end": round(ts, 2),
                        "duration": round(dur, 2)
                    })

    # Trailing idle
    if in_idle:
        last_ts = motion_series[-1]["timestamp"]
        dur = last_ts - start_ts
        if dur >= min_pause_s:
            idle_candidates.append({
                "start": round(start_ts, 2),
                "end": round(last_ts, 2),
                "duration": round(dur, 2)
            })

    return idle_candidates


def snap_boundary_to_motion(
    motion_series: List[Dict[str, float]],
    target_ts: float,
    window_s: float = 0.6,
    snap_to_onset: bool = True
) -> float:
    """
    Snap an approximate manual click to the nearest motion acceleration peak or onset frame.
    
    Args:
        motion_series: precomputed motion timeline
        target_ts: user clicked timestamp
        window_s: search radius in seconds (+/- window_s)
        snap_to_onset: True to snap to motion start (working start), False for motion end (working end)
    """
    if not motion_series:
        return target_ts

    candidates = [
        m for m in motion_series
        if abs(m["timestamp"] - target_ts) <= window_s
    ]
    if not candidates:
        return target_ts

    if snap_to_onset:
        # Find first transition from static to moving or highest energy delta
        for i in range(len(candidates) - 1):
            if not candidates[i]["is_moving"] and candidates[i + 1]["is_moving"]:
                return candidates[i + 1]["timestamp"]
        # Fallback: maximum energy gradient
        diffs = [candidates[i + 1]["energy"] - candidates[i]["energy"] for i in range(len(candidates) - 1)]
        if diffs:
            best_idx = int(np.argmax(diffs)) + 1
            return candidates[best_idx]["timestamp"]
    else:
        # Find transition from moving to static
        for i in range(len(candidates) - 1):
            if candidates[i]["is_moving"] and not candidates[i + 1]["is_moving"]:
                return candidates[i]["timestamp"]

    return target_ts
