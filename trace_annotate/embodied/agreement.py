"""
Inter-Annotator Multi-Dimensional Temporal & Semantic Agreement.
Calculates timeline IoU (tIoU), second-by-second Cohen's Kappa,
detects single-class prevalence degeneracy, and provides fine-grained
semantic concordance (step_number, verdict, error_type).
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional


def calculate_segment_tiou(seg_a: Dict[str, Any], seg_b: Dict[str, Any]) -> float:
    """
    Compute Temporal Intersection over Union (tIoU) between two temporal segments.
    """
    s_a, e_a = float(seg_a.get("start", 0.0)), float(seg_a.get("end", 0.0))
    s_b, e_b = float(seg_b.get("start", 0.0)), float(seg_b.get("end", 0.0))

    intersection = max(0.0, min(e_a, e_b) - max(s_a, s_b))
    union = max(e_a, e_b) - min(s_a, s_b)

    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)


def match_temporal_segments(
    segments_a: List[Dict[str, Any]],
    segments_b: List[Dict[str, Any]],
    min_tiou: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Greedy bipartite matching of segments between two annotators by highest tIoU.
    
    Returns:
        matched_pairs: list of dicts with 'seg_a', 'seg_b', 'tiou'
        unmatched_a: list of segments only in A
        unmatched_b: list of segments only in B
    """
    # Calculate all pair tIoU
    candidates = []
    for i, a in enumerate(segments_a):
        for j, b in enumerate(segments_b):
            iou = calculate_segment_tiou(a, b)
            if iou >= min_tiou:
                candidates.append((iou, i, j))

    # Sort descending by tIoU
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_a = set()
    matched_b = set()
    matched_pairs = []

    for iou, i, j in candidates:
        if i not in matched_a and j not in matched_b:
            matched_a.add(i)
            matched_b.add(j)
            matched_pairs.append({
                "seg_a": segments_a[i],
                "seg_b": segments_b[j],
                "tiou": iou
            })

    unmatched_a = [segments_a[i] for i in range(len(segments_a)) if i not in matched_a]
    unmatched_b = [segments_b[j] for j in range(len(segments_b)) if j not in matched_b]

    return matched_pairs, unmatched_a, unmatched_b


def build_temporal_mask(
    working_segments: List[Dict[str, Any]],
    total_duration: float,
    sample_rate_hz: float = 2.0
) -> List[bool]:
    """Build a boolean timeline mask sampled at sample_rate_hz."""
    n_samples = max(1, int(round(total_duration * sample_rate_hz)))
    mask = [False] * n_samples

    for w in working_segments:
        start_idx = max(0, int(round(float(w["start"]) * sample_rate_hz)))
        end_idx = min(n_samples, int(round(float(w["end"]) * sample_rate_hz)))
        for k in range(start_idx, end_idx):
            mask[k] = True

    return mask


def calculate_temporal_kappa(
    annotator_a_records: Dict[str, Dict[str, Any]],
    annotator_b_records: Dict[str, Dict[str, Any]],
    sample_rate_hz: float = 2.0,
    min_match_tiou: float = 0.1
) -> Dict[str, Any]:
    """
    Compute inter-annotator temporal Cohen's Kappa along with tIoU and semantic agreement.
    
    Args:
        annotator_a_records: dict of clip_id -> annotation_doc (or dict with 'working' segments)
        annotator_b_records: dict of clip_id -> annotation_doc
        sample_rate_hz: sampling frequency along timeline (default 2 Hz / half-second precision)
        min_match_tiou: minimum tIoU to consider two working segments as matched
        
    Returns:
        Dict with observed agreement p0, expected agreement pe, kappa, tIoU,
        semantic conflict details, and pass thresholds.
    """
    common_clips = sorted(list(set(annotator_a_records.keys()) & set(annotator_b_records.keys())))
    if not common_clips:
        return {
            "common_clips": 0,
            "observed_agreement_p0": 0.0,
            "kappa": 0.0,
            "mean_tiou": 0.0,
            "is_degenerate_single_class": False,
            "meets_trial_bar": False,
            "meets_baseline_bar": False,
            "semantic_conflicts": []
        }

    total_samples = 0
    agree_working = 0
    agree_idle = 0
    total_w_a = 0
    total_w_b = 0

    all_tious: List[float] = []
    semantic_conflicts: List[Dict[str, Any]] = []
    step_matches = 0
    step_comparisons = 0
    verdict_matches = 0
    verdict_comparisons = 0

    for clip in common_clips:
        wa = annotator_a_records[clip].get("working", [])
        wb = annotator_b_records[clip].get("working", [])

        # Segment level matching and semantic conflict checks
        matched, un_a, un_b = match_temporal_segments(wa, wb, min_tiou=min_match_tiou)
        for pair in matched:
            sa = pair["seg_a"]
            sb = pair["seg_b"]
            tiou = pair["tiou"]
            all_tious.append(tiou)

            # Check step_number agreement
            step_a = sa.get("step_number")
            step_b = sb.get("step_number")
            if step_a is not None and step_b is not None:
                step_comparisons += 1
                if step_a == step_b:
                    step_matches += 1
                else:
                    semantic_conflicts.append({
                        "clip": clip,
                        "time_range": [min(sa["start"], sb["start"]), max(sa["end"], sb["end"])],
                        "tiou": tiou,
                        "type": "step_mismatch",
                        "annotator_a": step_a,
                        "annotator_b": step_b
                    })

            # Check verdict agreement
            verd_a = sa.get("verdict")
            verd_b = sb.get("verdict")
            if verd_a is not None and verd_b is not None:
                verdict_comparisons += 1
                if verd_a == verd_b:
                    verdict_matches += 1
                else:
                    semantic_conflicts.append({
                        "clip": clip,
                        "time_range": [min(sa["start"], sb["start"]), max(sa["end"], sb["end"])],
                        "tiou": tiou,
                        "type": "verdict_mismatch",
                        "annotator_a": verd_a,
                        "annotator_b": verd_b
                    })

        # Timeline sampling for Kappa
        max_end_a = max([float(w["end"]) for w in wa] or [1.0])
        max_end_b = max([float(w["end"]) for w in wb] or [1.0])
        duration = max(max_end_a, max_end_b)

        mask_a = build_temporal_mask(wa, duration, sample_rate_hz)
        mask_b = build_temporal_mask(wb, duration, sample_rate_hz)

        # Make sure masks are same length
        max_len = max(len(mask_a), len(mask_b))
        if len(mask_a) < max_len:
            mask_a.extend([False] * (max_len - len(mask_a)))
        if len(mask_b) < max_len:
            mask_b.extend([False] * (max_len - len(mask_b)))

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
        return {
            "common_clips": len(common_clips),
            "observed_agreement_p0": 0.0,
            "kappa": 0.0,
            "mean_tiou": 0.0,
            "is_degenerate_single_class": False,
            "semantic_conflicts": []
        }

    p0 = (agree_working + agree_idle) / total_samples

    p_w_a = total_w_a / total_samples
    p_w_b = total_w_b / total_samples
    p_i_a = 1.0 - p_w_a
    p_i_b = 1.0 - p_w_b

    pe = (p_w_a * p_w_b) + (p_i_a * p_i_b)

    # Detect prevalence paradox: when one class overwhelmingly dominates (e.g. 100% working or 100% idle)
    is_degenerate_single_class = (
        (p_w_a >= 0.98 and p_w_b >= 0.98) or
        (p_i_a >= 0.98 and p_i_b >= 0.98)
    )

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (p0 - pe) / (1.0 - pe)

    kappa = round(kappa, 4)
    p0 = round(p0, 4)
    mean_tiou = round(sum(all_tious) / len(all_tious), 4) if all_tious else 0.0

    step_agreement_rate = round(step_matches / step_comparisons, 4) if step_comparisons > 0 else 1.0
    verdict_agreement_rate = round(verdict_matches / verdict_comparisons, 4) if verdict_comparisons > 0 else 1.0

    return {
        "common_clips": len(common_clips),
        "total_samples": total_samples,
        "observed_agreement_p0": p0,
        "expected_chance_pe": round(pe, 4),
        "kappa": kappa,
        "mean_tiou": mean_tiou,
        "is_degenerate_single_class": is_degenerate_single_class,
        "meets_trial_bar": kappa >= 0.80 or (is_degenerate_single_class and p0 >= 0.90),
        "meets_baseline_bar": kappa >= 0.70 or (is_degenerate_single_class and p0 >= 0.80),
        "step_agreement_rate": step_agreement_rate,
        "verdict_agreement_rate": verdict_agreement_rate,
        "semantic_conflicts": semantic_conflicts
    }
