"""Temporal diagnostics and a separate, fail-closed agreement gate.

Binary working/idle Cohen kappa cannot establish semantic correctness, annotator
independence, or agreement with the video. These limitations are always reported.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

from ._intervals import finite_number, interval, positive_duration, validated_intervals
from .schema import ClipStatus, ErrorType, Verdict
from .validator import validate_embodied_doc

_MAX_SAMPLES = 2_000_000
_MAX_PAIR_COMPARISONS = 1_000_000


def calculate_segment_tiou(seg_a: Dict[str, Any], seg_b: Dict[str, Any]) -> float:
    sa, ea = interval(seg_a, "segment_a")
    sb, eb = interval(seg_b, "segment_b")
    intersection = max(0.0, min(ea, eb) - max(sa, sb))
    union = (ea - sa) + (eb - sb) - intersection
    return round(intersection / union, 4)


def match_temporal_segments(segments_a: List[Dict[str, Any]],
                            segments_b: List[Dict[str, Any]],
                            min_tiou: float = 0.1
                            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic greedy one-to-one matching by tIoU; not an optimal matcher."""
    threshold = finite_number(min_tiou, "min_tiou")
    if not 0 < threshold <= 1:
        raise ValueError("min_tiou must be greater than zero and at most one")
    validated_intervals(segments_a, "annotator_a working")
    validated_intervals(segments_b, "annotator_b working")
    if len(segments_a) * len(segments_b) > _MAX_PAIR_COMPARISONS:
        raise ValueError("MATCH_LIMIT: too many segment pairs; split this batch")
    candidates = []
    for i, segment_a in enumerate(segments_a):
        for j, segment_b in enumerate(segments_b):
            tiou = calculate_segment_tiou(segment_a, segment_b)
            if tiou >= threshold:
                candidates.append((tiou, i, j))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    used_a, used_b = set(), set()
    matched = []
    for tiou, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append({"seg_a": segments_a[i], "seg_b": segments_b[j], "tiou": tiou})
    return (matched, [segment for i, segment in enumerate(segments_a) if i not in used_a],
            [segment for j, segment in enumerate(segments_b) if j not in used_b])


def build_temporal_mask(working_segments: List[Dict[str, Any]], total_duration: float,
                        sample_rate_hz: float = 2.0) -> List[bool]:
    """Sample half-open intervals at bin centers, including the final partial bin.

    Explicit duration preserves trailing idle. Neither negative/out-of-bounds
    values nor overlapping segments are silently clamped or merged.
    """
    duration = positive_duration(total_duration)
    rate = positive_duration(sample_rate_hz)
    count_float = duration * rate
    if not math.isfinite(count_float) or count_float > _MAX_SAMPLES:
        raise ValueError("SAMPLE_LIMIT: duration * sample_rate_hz exceeds 2000000")
    count = math.ceil(count_float)
    segments = validated_intervals(working_segments, "working", duration)
    result: List[bool] = []
    cursor = 0
    for index in range(count):
        left, right = index / rate, min((index + 1) / rate, duration)
        center = (left + right) / 2
        while cursor < len(segments) and segments[cursor][1] <= center:
            cursor += 1
        result.append(cursor < len(segments) and segments[cursor][0] <= center < segments[cursor][1])
    return result


def _records(records: Any, side: str):
    mapped: Dict[str, Dict[str, Any]] = {}
    duplicates, errors = [], []
    if isinstance(records, dict):
        entries = list(records.items())
    elif isinstance(records, list):
        entries = [(record.get("clip") if isinstance(record, dict) else None, record) for record in records]
    else:
        return {}, [], [f"{side}: annotation records must be a mapping or array"]
    for clip, record in entries:
        if not isinstance(clip, str) or not clip.strip() or not isinstance(record, dict):
            errors.append(f"{side}: each record needs a non-empty clip ID and JSON object")
            continue
        if clip in mapped:
            duplicates.append(clip)
            continue  # Preserve the first record and make the entire gate fail.
        if record.get("clip") is not None and record["clip"] != clip:
            errors.append(f"{side}:{clip}: mapping ID disagrees with record clip")
        mapped[clip] = record
    return mapped, sorted(set(duplicates)), errors


def calculate_temporal_kappa(annotator_a_records: Any, annotator_b_records: Any,
                             sample_rate_hz: float = 2.0,
                             min_match_tiou: float = 0.1) -> Dict[str, Any]:
    """Report temporal scores independently of coverage/semantic acceptance.

    Missing duration permits only a diagnostic horizon inferred from endpoints,
    explicitly marked invalid for a production gate. Single-class kappa is null,
    never a manufactured 1.0. Missing clips, duplicate IDs, unmatched segments,
    invalid labels or intervals, uncertain labels and semantic conflicts block
    acceptance even if the diagnostic temporal score is high.
    """
    rate = positive_duration(sample_rate_hz)
    threshold = finite_number(min_match_tiou, "min_match_tiou")
    if not 0 < threshold <= 1:
        raise ValueError("min_match_tiou must be greater than zero and at most one")
    a, duplicates_a, errors_a = _records(annotator_a_records, "annotator_a")
    b, duplicates_b, errors_b = _records(annotator_b_records, "annotator_b")
    common = sorted(set(a) & set(b))
    missing_a, missing_b = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    coverage_complete = bool(a) and bool(b) and not (missing_a or missing_b or duplicates_a or duplicates_b or errors_a or errors_b)
    validation_errors = errors_a + errors_b
    if not a or not b:
        validation_errors.append("EMPTY_BATCH: both annotators must provide at least one clip")
    semantic_conflicts: List[Dict[str, Any]] = []
    semantic_missing: List[Dict[str, Any]] = []
    unmatched_segments: List[Dict[str, Any]] = []
    boundary_findings: List[Dict[str, Any]] = []
    horizons: List[Dict[str, Any]] = []
    total_samples = total_w_a = total_w_b = agree = 0
    all_tious: List[float] = []
    comparisons = {"step": 0, "verdict": 0, "error_type": 0}
    matches = {"step": 0, "verdict": 0, "error_type": 0}
    semantic_complete = bool(common)

    def conflict(clip, kind, value_a, value_b, pair=None):
        finding = {"clip": clip, "type": kind, "annotator_a": value_a, "annotator_b": value_b}
        if pair is not None:
            sa, sb = pair["seg_a"], pair["seg_b"]
            finding.update(time_range=[min(float(sa["start"]), float(sb["start"])),
                                       max(float(sa["end"]), float(sb["end"]))], tiou=pair["tiou"])
        semantic_conflicts.append(finding)

    for clip in common:
        ra, rb = a[clip], b[clip]
        wa, wb = ra.get("working"), rb.get("working")
        durations = []
        for side, record in (("annotator_a", ra), ("annotator_b", rb)):
            doc = dict(record)
            doc.setdefault("clip", clip)
            validation_errors.extend(f"{side}:{error}" for error in validate_embodied_doc(
                doc, require_duration=True, require_description=True, require_idle_coverage=True))
            try:
                durations.append(positive_duration(record.get("duration")))
            except ValueError:
                durations.append(None)
        if durations[0] is not None and durations[1] is not None and durations[0] != durations[1]:
            validation_errors.append(f"DURATION_MISMATCH: {clip}: annotators must use the same media duration")
        status_a, status_b = ra.get("clip_status"), rb.get("clip_status")
        if status_a is None or status_b is None:
            semantic_complete = False
            semantic_missing.append({"clip": clip, "field": "clip_status"})
        elif status_a != status_b:
            conflict(clip, "clip_status_mismatch", status_a, status_b)
        if ra.get("text_description") != rb.get("text_description"):
            conflict(clip, "description_mismatch", ra.get("text_description"), rb.get("text_description"))
        for field in ("video_sha256", "source_sha256"):
            if ra.get(field) is not None or rb.get(field) is not None:
                if not ra.get(field) or ra.get(field) != rb.get(field):
                    conflict(clip, f"{field}_mismatch", ra.get(field), rb.get(field))
        try:
            # Validate time input separately so invalid/missing semantics do not
            # erase useful temporal diagnostics.
            intervals_a = validated_intervals(wa, f"{clip} annotator_a working", durations[0])
            intervals_b = validated_intervals(wb, f"{clip} annotator_b working", durations[1])
            available = [value for value in durations if value is not None]
            inferred = not (durations[0] is not None and durations[1] is not None)
            duration = max(available + [end for _, end, _ in intervals_a + intervals_b] + [1.0]) if inferred else max(available)
            horizons.append({"clip": clip, "duration": duration, "source": "inferred_diagnostic_only" if inferred else "supplied"})
            matched, un_a, un_b = match_temporal_segments(wa, wb, threshold)
            mask_a, mask_b = build_temporal_mask(wa, duration, rate), build_temporal_mask(wb, duration, rate)
            if total_samples + len(mask_a) > 10_000_000:
                raise ValueError("SAMPLE_LIMIT: batch exceeds 10000000 samples")
        except (ValueError, TypeError, OverflowError) as exc:
            validation_errors.append(f"{clip}: {exc}")
            semantic_complete = False
            continue
        for side, segments in (("annotator_a", un_a), ("annotator_b", un_b)):
            for segment in segments:
                unmatched_segments.append({"clip": clip, "annotator": side, "segment": segment})
        if un_a or un_b:
            semantic_complete = False
        total_boundary_offset = 0.0
        for pair in matched:
            sa, sb = pair["seg_a"], pair["seg_b"]
            all_tious.append(pair["tiou"])
            offsets = {field: abs(float(sa[field]) - float(sb[field])) for field in ("start", "end")}
            total_boundary_offset += sum(offsets.values())
            if any(value > 1.0 for value in offsets.values()):
                boundary_findings.append({"clip": clip, "type": "boundary_offset_review",
                                          "offsets_seconds": offsets})
            for field, short in (("step_number", "step"), ("verdict", "verdict")):
                va, vb = sa.get(field), sb.get(field)
                valid_a = isinstance(va, int) and not isinstance(va, bool) and va > 0 if field == "step_number" else isinstance(va, str) and va in {v.value for v in Verdict}
                valid_b = isinstance(vb, int) and not isinstance(vb, bool) and vb > 0 if field == "step_number" else isinstance(vb, str) and vb in {v.value for v in Verdict}
                if not valid_a or not valid_b:
                    semantic_complete = False
                    semantic_missing.append({"clip": clip, "field": field, "time_range": [sa["start"], sa["end"]]})
                    continue
                comparisons[short] += 1
                if va == vb:
                    matches[short] += 1
                else:
                    conflict(clip, f"{short}_mismatch", va, vb, pair)
            if sa.get("verdict") == Verdict.WRONG.value and sb.get("verdict") == Verdict.WRONG.value:
                ea, eb = sa.get("error_type"), sb.get("error_type")
                if not isinstance(ea, str) or not isinstance(eb, str) or ea not in {e.value for e in ErrorType} or eb not in {e.value for e in ErrorType}:
                    semantic_complete = False
                    semantic_missing.append({"clip": clip, "field": "error_type"})
                else:
                    comparisons["error_type"] += 1
                    if ea == eb:
                        matches["error_type"] += 1
                    else:
                        conflict(clip, "error_type_mismatch", ea, eb, pair)
            if sa.get("verdict") == Verdict.UNCERTAIN.value or sb.get("verdict") == Verdict.UNCERTAIN.value:
                semantic_complete = False
            if sa.get("mid_segments", []) != sb.get("mid_segments", []):
                conflict(clip, "mid_segments_mismatch", sa.get("mid_segments", []), sb.get("mid_segments", []), pair)
            if sa.get("correction_events", []) != sb.get("correction_events", []):
                conflict(clip, "correction_events_review", sa.get("correction_events", []), sb.get("correction_events", []), pair)
        if total_boundary_offset > 3.0:
            boundary_findings.append({"clip": clip, "type": "total_boundary_offset_review",
                                      "offset_seconds": total_boundary_offset})
        total_samples += len(mask_a)
        total_w_a += sum(mask_a)
        total_w_b += sum(mask_b)
        agree += sum(left == right for left, right in zip(mask_a, mask_b))

    p0 = agree / total_samples if total_samples else None
    prevalence_a = total_w_a / total_samples if total_samples else None
    prevalence_b = total_w_b / total_samples if total_samples else None
    pe = prevalence_a * prevalence_b + (1 - prevalence_a) * (1 - prevalence_b) if total_samples else None
    degenerate = bool(total_samples and ((prevalence_a >= .98 and prevalence_b >= .98) or
                                        (prevalence_a <= .02 and prevalence_b <= .02)))
    kappa = (p0 - pe) / (1 - pe) if pe is not None and pe < 1.0 else None
    kappa_status = "NO_VALID_SAMPLES" if not total_samples else "UNDEFINED_SINGLE_CLASS" if kappa is None else "PREVALENCE_REVIEW_REQUIRED" if degenerate else "VALID"
    temporal_trial = kappa is not None and kappa >= .80
    temporal_baseline = kappa is not None and kappa > .70
    eligible = bool(coverage_complete and not validation_errors and semantic_complete
                    and not semantic_conflicts and not unmatched_segments and not boundary_findings
                    and not degenerate and kappa is not None)
    return {
        "common_clips": len(common), "total_samples": total_samples,
        "observed_agreement_p0": round(p0, 4) if p0 is not None else None,
        "expected_chance_pe": round(pe, 4) if pe is not None else None,
        "kappa": round(kappa, 4) if kappa is not None else None,
        "kappa_unrounded": kappa,
        "kappa_status": kappa_status,
        "mean_tiou": round(sum(all_tious) / len(all_tious), 4) if all_tious else None,
        "is_degenerate_single_class": degenerate,
        "temporal_meets_trial_bar": temporal_trial, "temporal_meets_baseline_bar": temporal_baseline,
        "meets_trial_bar": eligible and temporal_trial, "meets_baseline_bar": eligible and temporal_baseline,
        "gate_eligible": eligible, "coverage_complete": coverage_complete,
        "missing_clips_a": missing_a, "missing_clips_b": missing_b,
        "duplicate_clips_a": duplicates_a, "duplicate_clips_b": duplicates_b,
        "validation_errors": validation_errors, "semantic_complete": semantic_complete,
        "semantic_missing": semantic_missing, "semantic_conflicts": semantic_conflicts,
        "unmatched_segments": unmatched_segments, "boundary_findings": boundary_findings,
        "timeline_horizons": horizons,
        "step_agreement_rate": round(matches["step"] / comparisons["step"], 4) if comparisons["step"] else None,
        "verdict_agreement_rate": round(matches["verdict"] / comparisons["verdict"], 4) if comparisons["verdict"] else None,
        "error_type_agreement_rate": round(matches["error_type"] / comparisons["error_type"], 4) if comparisons["error_type"] else None,
        "sampling_policy": "half_open_bin_centers_final_partial_bin",
        "review_scope": "format, temporal and label agreement only; video truth and annotator independence are not established",
    }
