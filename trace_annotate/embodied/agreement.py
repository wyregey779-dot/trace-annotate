"""
Inter-Annotator Temporal Agreement & Cohen's Kappa.
Calculates second-by-second timeline IoU agreement between two annotators
and computes Cohen's Kappa score with trial benchmark checks (>= 0.80).
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple


def build_temporal_mask(
    working_segments: List[Dict[str, Any]],
    total_duration: float,
    sample_rate_hz: float = 2.0
) -> List[bool]:
    """Build a boolean timeline mask sampled at sample_rate_hz."""
    n_samples = max(1, int(total_duration * sample_rate_hz))
    mask = [False] * n_samples

    for w in working_segments:
        start_idx = max(0, int(float(w["start"]) * sample_rate_hz))
        end_idx = min(n_samples, int(float(w["end"]) * sample_rate_hz))
        for k in range(start_idx, end_idx):
            mask[k] = True

    return mask


def calculate_temporal_kappa(
    annotator_a_records: Dict[str, Dict[str, Any]],
    annotator_b_records: Dict[str, Dict[str, Any]],
    sample_rate_hz: float = 2.0
) -> Dict[str, Any]:
    """
    Compute inter-annotator temporal Cohen's Kappa.
    
    Args:
        annotator_a_records: dict of clip_id -> annotation_doc (or dict with 'working' segments)
        annotator_b_records: dict of clip_id -> annotation_doc
        sample_rate_hz: sampling frequency along timeline (default 2 Hz / half-second precision)
        
    Returns:
        Dict with observed agreement p0, expected agreement pe, kappa, and pass thresholds.
    """
    common_clips = set(annotator_a_records.keys()) & set(annotator_b_records.keys())
    if not common_clips:
        return {
            "common_clips": 0,
            "observed_agreement_p0": 0.0,
            "kappa": 0.0,
            "meets_trial_bar": False,
            "meets_baseline_bar": False
        }

    total_samples = 0
    agree_working = 0
    agree_idle = 0
    total_w_a = 0
    total_w_b = 0

    for clip in common_clips:
        wa = annotator_a_records[clip].get("working", [])
        wb = annotator_b_records[clip].get("working", [])

        max_end_a = max([float(w["end"]) for w in wa] or [1.0])
        max_end_b = max([float(w["end"]) for w in wb] or [1.0])
        duration = max(max_end_a, max_end_b)

        mask_a = build_temporal_mask(wa, duration, sample_rate_hz)
        mask_b = build_temporal_mask(wb, duration, sample_rate_hz)

        for a_val, b_val in zip(mask_a, mask_b):
            total_samples += 1
            if a_val:
                total_w_a += 1
            if b_val:
                total_w_b += 1

            if a_val == b_val:
                if a_val:
                    agree_working += 1
                else:
                    agree_idle += 1

    if total_samples == 0:
        return {"common_clips": len(common_clips), "observed_agreement_p0": 0.0, "kappa": 0.0}

    p0 = (agree_working + agree_idle) / total_samples

    p_w_a = total_w_a / total_samples
    p_w_b = total_w_b / total_samples
    p_i_a = 1.0 - p_w_a
    p_i_b = 1.0 - p_w_b

    pe = (p_w_a * p_w_b) + (p_i_a * p_i_b)

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (p0 - pe) / (1.0 - pe)

    kappa = round(kappa, 4)
    p0 = round(p0, 4)

    return {
        "common_clips": len(common_clips),
        "total_samples": total_samples,
        "observed_agreement_p0": p0,
        "expected_chance_pe": round(pe, 4),
        "kappa": kappa,
        "meets_trial_bar": kappa >= 0.80,
        "meets_baseline_bar": kappa >= 0.70
    }
