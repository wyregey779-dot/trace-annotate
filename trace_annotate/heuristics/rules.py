"""
Deterministic Rules & Tag Synthesis.
Infers indoor/outdoor, day/night, and environmental tags from path patterns and visual metrics.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Tuple


NIGHT_KEYWORDS = re.compile(r"夜|晚上|夜景|灯光秀|night", re.IGNORECASE)
DAY_KEYWORDS = re.compile(r"白天|日景|上午|下午|day", re.IGNORECASE)
INDOOR_KEYWORDS = re.compile(r"室内|商场|展厅|车厢|大堂|内部|inside|indoor", re.IGNORECASE)
OUTDOOR_KEYWORDS = re.compile(r"室外|广场|街区|山顶|江边|公园|街道|outside|outdoor", re.IGNORECASE)


def infer_environment_rules(
    rel_path: str,
    clip_io: Optional[str] = None,
    clip_tod: Optional[str] = None
) -> Tuple[str, str, List[str]]:
    """
    Infer indoor/outdoor and time-of-day combining path patterns with CLIP predictions.
    
    Returns:
        indoor_outdoor: '室内' | '室外' | '未定'
        time_of_day: '白天' | '夜' | '未定'
        tags: list of derived semantic tags
    """
    tags = []

    # Time of Day
    if NIGHT_KEYWORDS.search(rel_path):
        tod = "夜"
        tags.append("夜景")
    elif DAY_KEYWORDS.search(rel_path):
        tod = "白天"
    else:
        tod = clip_tod or "白天"

    # Indoor / Outdoor
    if INDOOR_KEYWORDS.search(rel_path):
        io = "室内"
        tags.append("室内")
    elif OUTDOOR_KEYWORDS.search(rel_path):
        io = "室外"
        tags.append("室外")
    else:
        io = clip_io or "室外"

    return io, tod, tags
