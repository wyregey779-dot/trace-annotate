"""Compute a complement only from valid, non-overlapping working intervals."""
from __future__ import annotations
from typing import Any, Dict, List
from ._intervals import positive_duration, validated_intervals


def compute_idle_complement(
    working_segments: List[Dict[str, Any]], total_duration: float
) -> List[Dict[str, float]]:
    """Return the exact complement in [0, duration], without rounding.

    The duration must come from media or an explicit caller assertion; the last
    working end is not evidence of the full video duration. Invalid input raises
    ValueError. Mid-segments remain in working and do not create outer idle.
    """
    duration = positive_duration(total_duration)
    working = validated_intervals(working_segments, "working", duration)
    idle: List[Dict[str, float]] = []
    cursor = 0.0
    for start, end, _ in working:
        if start > cursor:
            idle.append({"start": cursor, "end": start})
        cursor = end
    if cursor < duration:
        idle.append({"start": cursor, "end": duration})
    return idle
