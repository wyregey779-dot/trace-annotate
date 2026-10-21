"""
Idle Complement Interval Calculator.
Computes the complement intervals of working segments along the video timeline.
"""
from __future__ import annotations
from typing import List, Dict, Any


def compute_idle_complement(
    working_segments: List[Dict[str, Any]],
    total_duration: float
) -> List[Dict[str, float]]:
    """
    Compute idle intervals: timeline complement of working segments.
    
    Args:
        working_segments: list of dicts with 'start' and 'end' keys
        total_duration: total duration of the clip in seconds
        
    Returns:
        List of idle interval dicts: [{'start': ..., 'end': ...}, ...]
    """
    if not working_segments:
        if total_duration > 0:
            return [{"start": 0.0, "end": round(total_duration, 2)}]
        return []

    sorted_w = sorted(working_segments, key=lambda x: x["start"])

    merged = []
    for w in sorted_w:
        s, e = float(w["start"]), float(w["end"])
        if not merged:
            merged.append([s, e])
        else:
            prev_s, prev_e = merged[-1]
            if s <= prev_e:
                merged[-1][1] = max(prev_e, e)
            else:
                merged.append([s, e])

    idle_segs: List[Dict[str, float]] = []
    cursor = 0.0

    for s, e in merged:
        if s > cursor:
            idle_segs.append({"start": round(cursor, 2), "end": round(s, 2)})
        cursor = max(cursor, e)

    if cursor < total_duration:
        idle_segs.append({"start": round(cursor, 2), "end": round(total_duration, 2)})

    return idle_segs
