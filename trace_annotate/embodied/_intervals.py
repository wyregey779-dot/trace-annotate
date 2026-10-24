"""Shared interval validation; never clip, merge, or repair invalid input."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple


def finite_number(value: Any, label: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not numeric")
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"INVALID_TIME: {label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"INVALID_TIME: {label} must be a finite number")
    return result


def positive_duration(value: Any) -> float:
    try:
        result = finite_number(value, "duration")
    except ValueError as exc:
        raise ValueError("DURATION_REQUIRED: duration must be a finite positive number") from exc
    if result <= 0:
        raise ValueError("DURATION_REQUIRED: duration must be a finite positive number")
    return result


def interval(segment: Dict[str, Any], label: str, duration: Optional[float] = None) -> Tuple[float, float]:
    if not isinstance(segment, dict):
        raise ValueError(f"INVALID_TIME: {label} segment must be a JSON object")
    start = finite_number(segment.get("start"), f"{label} start timestamp")
    end = finite_number(segment.get("end"), f"{label} end timestamp")
    if start < 0:
        raise ValueError(f"INVALID_TIME: {label} start timestamp cannot be negative ({start})")
    if start >= end:
        raise ValueError(f"INVALID_TIME: {label} start ({start}) must be strictly less than end ({end})")
    if duration is not None and end > duration:
        raise ValueError(f"INVALID_TIME: {label} end ({end}) exceeds duration ({duration})")
    return start, end


def overlap_errors(intervals: List[Tuple[float, float, int]], label: str) -> List[str]:
    """Track the furthest end, so a nested interval cannot hide later overlaps."""
    errors: List[str] = []
    furthest = None
    for current in sorted(intervals):
        if furthest is not None and current[0] < furthest[1]:
            errors.append(
                f"INVALID_TIME: {label} segments overlap in time: "
                f"[{furthest[2]}] ({furthest[0]}..{furthest[1]}) and "
                f"[{current[2]}] ({current[0]}..{current[1]})"
            )
        if furthest is None or current[1] > furthest[1]:
            furthest = current
    return errors


def validated_intervals(segments: Any, label: str, duration: Optional[float] = None) -> List[Tuple[float, float, int]]:
    if not isinstance(segments, list):
        raise ValueError(f"INVALID_TIME: {label} must be an array")
    result = [(*interval(segment, f"{label}[{index}]", duration), index)
              for index, segment in enumerate(segments)]
    errors = overlap_errors(result, label)
    if errors:
        raise ValueError("; ".join(errors))
    return sorted(result)

