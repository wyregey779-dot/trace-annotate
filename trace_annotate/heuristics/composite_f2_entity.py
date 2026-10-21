"""
Heuristic F2: Primary Entity Tiering & Consensus Conflict Detection.
Separates primary entities into 'confirmed' vs 'candidate' tiers.
Flags severe conflicts between CLIP zero-shot predictions and folder path semantics.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple


def evaluate_f2_entity_tier(
    primary_entity: Optional[str],
    has_human_override: bool,
    path_depth_matched: bool,
    clip_top_entity: Optional[str] = None,
    clip_top_score: float = 0.0,
    path_entity_score: float = 0.0
) -> Tuple[str, str, bool, Optional[str]]:
    """
    Apply Composite Heuristic F2.
    
    Returns:
        entity_status: 'confirmed' | 'candidate' | 'none'
        fill_tier: 'human_confirmed' | 'auto_path_confirmed' | 'candidate_only' | 'pending_review'
        has_conflict: bool
        conflict_note: Optional[str]
    """
    if not primary_entity or primary_entity == "未定":
        return "none", "pending_review", False, None

    # Priority 1: Human override is permanently confirmed
    if has_human_override:
        return "confirmed", "human_confirmed", False, None

    # Check for conflict between path entity and CLIP prediction
    has_conflict = False
    conflict_note = None

    if clip_top_entity and clip_top_entity != primary_entity:
        diff = clip_top_score - path_entity_score
        if diff >= 0.05 and clip_top_score >= 0.40:
            has_conflict = True
            conflict_note = (
                f"CLIP suggests '{clip_top_entity}' ({clip_top_score:.3f}) vs "
                f"Path entity '{primary_entity}' ({path_entity_score:.3f})"
            )

    # Priority 2: Deep path hierarchy match
    if path_depth_matched and not has_conflict:
        return "confirmed", "auto_path_confirmed", False, None

    # Priority 3: Low-confidence or conflict proposal -> candidate tier
    return "candidate", "candidate_only", has_conflict, conflict_note
