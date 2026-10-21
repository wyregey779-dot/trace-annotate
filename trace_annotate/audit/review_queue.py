"""
Prioritized Review Queue Generator.
Sorts pending visual inspections into P1 (unresolved/conflict) -> P2 (collision/blacklist) ->
P3 (high-value video keyframe selection) -> P4 (general quality audit).
"""
from __future__ import annotations
import json
from typing import List, Dict, Any


def build_review_queue(
    annotations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Construct a prioritized review queue from annotation records.
    """
    p1_items = []
    p2_items = []
    p3_items = []
    p4_items = []

    for r in annotations:
        rel = r.get("rel_path", "")
        status = r.get("entity_status", "none")
        entity = r.get("primary_entity", "未定")
        face_rev = r.get("face_review", False)
        tech_q = r.get("tech_quality", {})
        conflict = tech_q.get("clip_entity_conflict") or tech_q.get("conflict_note")
        blacklist = r.get("usage_blacklist", False)
        kind = r.get("kind", "image")

        queue_entry = {
            "asset_id": r.get("asset_id"),
            "rel_path": rel,
            "primary_entity": entity,
            "entity_status": status,
            "priority": "P4",
            "reasons": []
        }

        # Check P1: Unresolved entity or conflict
        if status == "none" or entity in {"未定", "", None}:
            queue_entry["reasons"].append("Unresolved primary entity")
            queue_entry["priority"] = "P1"
            p1_items.append(queue_entry)
            continue
        elif conflict:
            queue_entry["reasons"].append(f"Entity conflict: {conflict}")
            queue_entry["priority"] = "P1"
            p1_items.append(queue_entry)
            continue

        # Check P2: Blacklist or face disambiguation review
        if blacklist:
            queue_entry["reasons"].append("Usage blacklist check")
            queue_entry["priority"] = "P2"
            p2_items.append(queue_entry)
            continue
        elif face_rev:
            queue_entry["reasons"].append("Statue/artifact face disambiguation review")
            queue_entry["priority"] = "P2"
            p2_items.append(queue_entry)
            continue

        # Check P3: Video assets with candidate status needing best frame selection
        if kind == "video" and status == "candidate":
            queue_entry["reasons"].append("Video candidate needing frame audit")
            queue_entry["priority"] = "P3"
            p3_items.append(queue_entry)
            continue

        # Check P4: General low-priority reviews
        if r.get("is_bad_frame"):
            queue_entry["reasons"].append("Bad frame flag inspection")
            p4_items.append(queue_entry)
        elif status == "candidate":
            queue_entry["reasons"].append("Candidate status confirmation")
            p4_items.append(queue_entry)

    return p1_items + p2_items + p3_items + p4_items
