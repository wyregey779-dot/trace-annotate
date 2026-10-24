"""Structural and temporal QC, not a claim that a label matches the video."""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ._intervals import interval, overlap_errors, positive_duration
from .schema import ClipStatus, ErrorType, Verdict

ALLOWED_ERROR_TYPES = {e.value for e in ErrorType}
ALLOWED_VERDICTS = {v.value for v in Verdict}
ALLOWED_CLIP_STATUSES = {s.value for s in ClipStatus}
_TIME = re.compile(r"\bt\s*=\s*(\d+(?::\d{1,2})?(?:\.\d+)?)(?:\s*(?:s|秒))?(?=\s|[,，。;；]|$)", re.I)
_STEP = re.compile(r"第\s*([1-9]\d*)\s*步")


def _evidence_times(text: str) -> Tuple[List[float], bool]:
    values: List[float] = []
    valid = len(_TIME.findall(text)) == len(re.findall(r"\bt\s*=", text, re.I))
    for match in _TIME.finditer(text):
        token = match.group(1)
        if ":" in token:
            minutes, seconds = token.split(":")
            if float(seconds) >= 60:
                valid = False
            values.append(float(minutes) * 60 + float(seconds))
        else:
            values.append(float(token))
    return values, valid and all(math.isfinite(value) for value in values)


def _check_evidence(segment: Dict[str, Any], prefix: str,
                    bounds: Optional[Tuple[float, float]]) -> List[str]:
    evidence = segment.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return [f"{prefix}: verdict='错' requires non-empty template-based evidence"]
    kind = segment.get("error_type")
    times, valid_time = _evidence_times(evidence)
    errors = []
    if not valid_time or (bounds is not None and any(t < bounds[0] or t > bounds[1] for t in times)):
        errors.append(f"EVIDENCE: {prefix}: evidence timestamp must be valid and within its working segment")
    steps = _STEP.findall(evidence)
    if kind == ErrorType.MISSING_STEP.value:
        valid = bool(steps) and any(word in evidence for word in ("未出现", "未做", "遗漏", "缺失", "没做"))
    elif kind == ErrorType.OUT_OF_ORDER.value:
        valid = len(set(steps)) >= 2 and any(word in evidence for word in ("之前", "先于", "顺序", "倒序"))
    elif kind == ErrorType.DUPLICATE_EXTRA.value:
        valid = bool(times or steps) and any(word in evidence for word in ("重复", "多做", "两遍", "多出", "规定外"))
    elif kind == ErrorType.WRONG_OBJECT.value:
        valid = bool(times) and "用" in evidence and any(word in evidence for word in ("应为", "应该", "规定", "应使用"))
    elif kind == ErrorType.WRONG_METHOD.value:
        valid = bool(times) and any(word in evidence for word in ("方式", "手法", "动作")) and any(word in evidence for word in ("应", "规定"))
    elif kind == ErrorType.ACCIDENT.value:
        valid = bool(times) and any(word in evidence for word in ("掉落", "打翻", "滑落", "摔落"))
    elif kind == ErrorType.OTHER.value:
        explanation = _TIME.sub("", _STEP.sub("", evidence)).strip(" ，,。:：;；")
        valid = bool(times or steps) and len(explanation) >= 4
    else:
        valid = True  # Invalid error class is reported separately.
    if not valid:
        errors.append(f"EVIDENCE: {prefix}: evidence does not satisfy the '{kind}' locator and explanation template")
    return errors


def validate_embodied_doc(doc: Dict[str, Any], *, require_duration: bool = False,
                          require_idle_coverage: bool = False,
                          require_description: bool = False) -> List[str]:
    """Return violations. Strict options are used by the production pipeline.

    Legacy time/step diagnostics may omit duration, description, and idle. When
    present they are always checked. Empty findings establish format consistency
    only; actual actions, boundaries, and evidence truth need video review.
    """
    if not isinstance(doc, dict):
        return ["Annotation record must be a JSON object"]
    errors: List[str] = []
    clip = doc.get("clip", "unknown_clip")
    if not isinstance(doc.get("clip"), str) or not doc["clip"].strip():
        errors.append("clip must be a non-empty string")
    description = doc.get("text_description")
    if require_description or description is not None:
        if not isinstance(description, str) or not description.strip():
            errors.append(f"DESCRIPTION_REQUIRED: {clip}: text_description must be a non-empty string")
    duration = None
    if doc.get("duration") is not None:
        try:
            duration = positive_duration(doc["duration"])
        except ValueError as exc:
            errors.append(f"{exc}: {clip}")
    elif require_duration or require_idle_coverage:
        errors.append(f"DURATION_REQUIRED: {clip}: explicit media duration is required")

    working = doc.get("working")
    if not isinstance(working, list):
        errors.append(f"{clip}: working must be an array")
        working = []
    idle = doc.get("idle", [])
    if not isinstance(idle, list):
        errors.append(f"{clip}: idle must be an array")
        idle = []
    status = doc.get("clip_status", ClipStatus.NORMAL.value)
    if not isinstance(status, str) or status not in ALLOWED_CLIP_STATUSES:
        errors.append(f"{clip}: Invalid clip_status '{status}'")

    partial = status == ClipStatus.DESC_MISMATCH_PART.value
    permits_working = status == ClipStatus.NORMAL.value or partial
    if partial:
        errors.append(
            f"REVIEW_REQUIRED: {clip}: partial-clip policy conflicts between specification sections 1 and 8; "
            "section 1 permits matching working segments, pending human adjudication"
        )
    if status != ClipStatus.NORMAL.value:
        if status != ClipStatus.EMPTY.value or doc.get("report") is not None:
            report = doc.get("report")
            if not isinstance(report, str) or not report.strip():
                errors.append(f"{clip}: clip_status='{status}' requires non-empty report with Quick ID and description")
            else:
                locator = re.search(r"(?:quick\s*id|ID)\s*[:：#=]?\s*([^\s，,：:;；]+)", report, re.I)
                remainder = report[locator.end():].strip(" ，,。:：;；") if locator else ""
                if locator is None or len(remainder) < 2:
                    errors.append(f"REPORT: {clip}: report must identify a Quick ID and describe the phenomenon")
        if not permits_working:
            if working:
                errors.append(f"{clip}: clip_status='{status}' but working segments array is not empty ({len(working)} segments)")
            if idle:
                errors.append(f"IDLE_CONFLICT: {clip}: excluded clips must have empty idle segments")
    elif not working:
        errors.append(f"{clip}: normal clip requires at least one working segment")

    valid_working: List[Tuple[float, float, int]] = []
    for index, segment in enumerate(working):
        prefix = f"{clip} working[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{prefix}: segment must be a JSON object")
            continue
        bounds = None
        try:
            bounds = interval(segment, prefix, duration)
            valid_working.append((*bounds, index))
        except ValueError as exc:
            errors.append(str(exc))
        step = segment.get("step_number")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            errors.append(f"{prefix}: step_number must be a positive integer (got {type(step).__name__}: {step})")
        if "label" in segment and not isinstance(segment["label"], str):
            errors.append(f"{prefix}: optional label must be a string")
        verdict = segment.get("verdict")
        if not isinstance(verdict, str) or verdict not in ALLOWED_VERDICTS:
            errors.append(f"{prefix}: Invalid verdict '{verdict}' (expected one of {sorted(ALLOWED_VERDICTS)})")
        if verdict == Verdict.WRONG.value:
            kind = segment.get("error_type")
            if not isinstance(kind, str) or kind not in ALLOWED_ERROR_TYPES:
                errors.append(f"{prefix}: verdict='错' requires error_type in 7 standard classes (got '{kind}')")
            errors.extend(_check_evidence(segment, prefix, bounds))
        elif verdict == Verdict.CORRECT.value:
            if segment.get("error_type"):
                errors.append(f"{prefix}: verdict='对' must not have error_type")
            if segment.get("evidence"):
                errors.append(f"{prefix}: verdict='对' must not have error evidence; use correction_events")
        elif verdict == Verdict.UNCERTAIN.value:
            errors.append(f"REVIEW_REQUIRED: {prefix}: uncertain verdict requires review of the video")

        mid = segment.get("mid_segments", [])
        if not isinstance(mid, list):
            errors.append(f"MID_SEGMENT: {prefix}: mid_segments must be an array")
            mid = []
        valid_mid = []
        for mid_index, sub in enumerate(mid):
            label = f"{prefix} mid_segments[{mid_index}]"
            try:
                start, end = interval(sub, label, duration)
                valid_mid.append((start, end, mid_index))
                if bounds is not None and (start < bounds[0] or end > bounds[1]):
                    errors.append(f"MID_SEGMENT: {label}: must be contained within its working segment")
            except ValueError as exc:
                errors.append(f"MID_SEGMENT: {exc}")
            if isinstance(sub, dict) and sub.get("reason") not in ("idle", "wrist_anomaly"):
                errors.append(f"MID_SEGMENT: {label}: reason must be idle or wrist_anomaly")
        errors.extend(f"MID_SEGMENT: {error}" for error in overlap_errors(valid_mid, prefix))
        corrections = segment.get("correction_events", [])
        if not isinstance(corrections, list):
            errors.append(f"CORRECTION: {prefix}: correction_events must be an array")
        else:
            for correction in corrections:
                times, valid = _evidence_times(correction) if isinstance(correction, str) else ([], False)
                if (not times or not valid or not isinstance(correction, str)
                        or not any(word in correction for word in ("纠正", "改正", "mispick"))
                        or (bounds is not None and any(t < bounds[0] or t > bounds[1] for t in times))):
                    errors.append(f"CORRECTION: {prefix}: correction event requires a valid in-segment timestamp and correction explanation")
    errors.extend(overlap_errors(valid_working, f"{clip} working"))

    valid_idle = []
    for index, segment in enumerate(idle):
        try:
            valid_idle.append((*interval(segment, f"{clip} idle[{index}]", duration), index))
        except ValueError as exc:
            errors.append(str(exc))
    errors.extend(overlap_errors(valid_idle, f"{clip} idle"))
    # Mid-segments deliberately do not participate in outer idle checks.
    ordered_working, ordered_idle = sorted(valid_working), sorted(valid_idle)
    cursor = 0
    for start, end, idle_index in ordered_idle:
        while cursor < len(ordered_working) and ordered_working[cursor][1] <= start:
            cursor += 1
        if cursor < len(ordered_working) and ordered_working[cursor][0] < end:
            errors.append(f"IDLE_CONFLICT: {clip}: idle[{idle_index}] overlaps working[{ordered_working[cursor][2]}]")
    if require_idle_coverage and duration is not None and permits_working:
        cursor_time = 0.0
        for start, end, _ in sorted(valid_working + valid_idle):
            if start > cursor_time:
                errors.append(f"IDLE_COVERAGE: {clip}: uncovered timeline {cursor_time}..{start}")
            cursor_time = max(cursor_time, end)
        if cursor_time < duration:
            errors.append(f"IDLE_COVERAGE: {clip}: uncovered timeline {cursor_time}..{duration}")
    return errors


def validate_embodied_export(file_or_dir: str) -> Dict[str, Any]:
    """Legacy field diagnostics, retaining per-line errors for JSONL inputs.

    Decode through the same strict parser as the production pipeline. This API
    deliberately retains optional duration/description/coverage compatibility;
    use embodied-validate or the quality pipeline for the strict admission gate.
    """
    from .quality_io import _decode_json
    targets = []
    if os.path.isdir(file_or_dir):
        for root, _, files in os.walk(file_or_dir):
            targets.extend(os.path.join(root, name) for name in sorted(files)
                           if name.lower().endswith((".json", ".jsonl")))
    else:
        targets.append(file_or_dir)
    total_records = 0
    violations: List[str] = []
    for target in sorted(targets):
        try:
            with open(target, "r", encoding="utf-8-sig") as stream:
                entries = ((str(n), line) for n, line in enumerate(stream, 1) if line.strip()) if target.lower().endswith(".jsonl") else [("document", stream.read())]
                before = total_records
                for location, content in entries:
                    try:
                        data = _decode_json(content)
                    except (ValueError, RecursionError) as exc:
                        violations.append(f"{target}:{location}: JSON decode error: {exc}")
                        continue
                    for item in data if isinstance(data, list) else [data]:
                        total_records += 1
                        violations.extend(f"{target}:{location}: {error}" for error in validate_embodied_doc(item))
                if total_records == before and not any(v.startswith(f"{target}:") for v in violations):
                    violations.append(f"EMPTY_BATCH: {target}: no annotation records")
        except (OSError, UnicodeError) as exc:
            violations.append(f"READ_ERROR: {target}: {exc}")
    if not targets:
        violations.append("EMPTY_BATCH: no JSON or JSONL annotation files found")
    return {"files_checked": len(targets), "records_checked": total_records,
            "violations_count": len(violations), "passed": not violations, "violations": violations,
            "scope": "Legacy supplied-field diagnostics; complete duration, description and idle coverage are not required. Not a production admission gate or video-content verification."}
