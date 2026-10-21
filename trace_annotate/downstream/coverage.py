"""
Narrative Script to Asset Pool Coverage Engine.
Maps script lines/sentences to annotated primary entities and shot windows.
Identifies footage gaps, thin pools, and usable editing durations.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple


def analyze_narrative_coverage(
    script_sentences: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    min_assets_per_sentence: int = 3
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Map narrative script sentences against annotated asset catalog.
    
    Args:
        script_sentences: list of dicts with 'index', 'text', and 'target_entity'
        annotations: list of annotation records
        min_assets_per_sentence: threshold below which pool is marked 'thin'
        
    Returns:
        coverage_rows: detailed breakdown per sentence
        summary_stats: aggregated pool health metrics
    """
    # Group annotations by primary entity
    entity_to_assets: Dict[str, List[Dict[str, Any]]] = {}
    for r in annotations:
        if r.get("usage_blacklist"):
            continue
        ent = r.get("primary_entity")
        if ent and ent != "未定":
            entity_to_assets.setdefault(ent, []).append(r)

    coverage_rows = []
    gap_count = 0
    thin_count = 0
    rich_count = 0

    for s in script_sentences:
        idx = s.get("index", 0)
        text = s.get("text", "")
        target = s.get("target_entity", "")

        matched_assets = entity_to_assets.get(target, [])
        asset_count = len(matched_assets)
        total_dur = sum([float(a.get("duration", 0.0) or 0.0) for a in matched_assets])

        if asset_count == 0:
            status = "缺口 (0素材)"
            gap_count += 1
        elif asset_count < min_assets_per_sentence:
            status = "池薄 (<3素材)"
            thin_count += 1
        else:
            status = "充裕"
            rich_count += 1

        coverage_rows.append({
            "index": idx,
            "text": text,
            "target_entity": target,
            "matched_assets": asset_count,
            "total_duration_s": round(total_dur, 2),
            "status": status,
            "sample_assets": [a.get("rel_path") for a in matched_assets[:3]]
        })

    summary = {
        "total_sentences": len(script_sentences),
        "gap_sentences": gap_count,
        "thin_sentences": thin_count,
        "rich_sentences": rich_count,
        "coverage_rate": round((len(script_sentences) - gap_count) / max(1, len(script_sentences)), 4)
    }

    return coverage_rows, summary
