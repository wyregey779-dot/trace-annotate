"""
Heuristic F1: Statue & Cultural Artifact Face Disambiguation.
Decouples real human faces from statues, bronze sculptures, historical figurines,
and museum artifacts to prevent false positive person annotations in raw tourist footage.
"""
from __future__ import annotations
import re
from typing import Dict, Any, Tuple


STATUE_KEYWORDS = re.compile(r"雕像|塑像|铜像|石雕|造像|毛主席像|人物像|蜡像")
ARTIFACT_KEYWORDS = re.compile(r"文物|展厅|博物院|展览|马王堆|木俑|辛追|陶俑|古墓|陈列")


def evaluate_f1_face(
    asset_rel_path: str,
    raw_face_count: int,
    raw_face_status: str,
    primary_entity: str = "",
    clip_context: str = ""
) -> Tuple[str, str, bool, str]:
    """
    Apply Composite Heuristic F1 to separate statue/artifact faces from real people.
    
    Returns:
        face_subject_kind: 'real_person' | 'statue_face' | 'artifact_face' | 'none'
        adjusted_face_status: string (e.g. '无人物' for statues with faces)
        face_review_required: bool (whether human auditor must inspect)
        reason: explanation string
    """
    if raw_face_count == 0:
        return "none", "无人物", False, "No faces detected"

    full_context = f"{asset_rel_path} {primary_entity} {clip_context}"

    if STATUE_KEYWORDS.search(full_context):
        return "statue_face", "无人物", True, "Statue context detected: face belongs to sculpture"

    if ARTIFACT_KEYWORDS.search(full_context):
        return "artifact_face", "无人物", True, "Artifact/museum context: face belongs to relic/figurine"

    return "real_person", raw_face_status, False, "Standard human face detection"
