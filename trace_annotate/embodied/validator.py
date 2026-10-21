"""
Embodied Annotation Validator & Compliance Checker.
Ensures strict compliance with Annotation Specification v2.1:
  - Diagnostic consistency (non-normal clips must have empty working and required report).
  - Verdict consistency (wrong segments must specify error_type in 7 classes with non-empty evidence).
  - Step numbering integrity.
  - Temporal consistency (non-negative timestamps, start < end, non-overlapping working segments).
"""
from __future__ import annotations
import os
import json
import math
from typing import List, Dict, Any
from trace_annotate.embodied.schema import ErrorType, Verdict, ClipStatus


ALLOWED_ERROR_TYPES = {e.value for e in ErrorType}
ALLOWED_VERDICTS = {v.value for v in Verdict}
ALLOWED_CLIP_STATUSES = {s.value for s in ClipStatus}


def validate_embodied_doc(doc: Dict[str, Any]) -> List[str]:
    """
    Validate a single embodied annotation dictionary.
    Returns a list of violation error strings (empty if compliant).
    """
    errors = []
    if not isinstance(doc, dict):
        return ["Annotation record must be a JSON object"]
    clip_id = doc.get("clip", "unknown_clip")
    if not isinstance(doc.get("clip"), str) or not doc["clip"].strip():
        errors.append("clip must be a non-empty string")
    working = doc.get("working")
    if not isinstance(working, list):
        errors.append(f"{clip_id}: working must be an array")
        return errors
    duration = None
    if doc.get("duration") is not None:
        try:
            raw_duration = doc["duration"]
            if isinstance(raw_duration, bool):
                raise ValueError("Boolean duration is not numeric")
            duration = float(raw_duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Duration must be finite and positive")
        except (ValueError, TypeError):
            duration = None
            errors.append(f"{clip_id}: duration must be a finite positive number")

    status = doc.get("clip_status", ClipStatus.NORMAL.value)
    if not isinstance(status, str) or status not in ALLOWED_CLIP_STATUSES:
        errors.append(f"{clip_id}: Invalid clip_status '{status}'")

    if status != ClipStatus.NORMAL.value:
        if working:
            errors.append(f"{clip_id}: clip_status='{status}' but working segments array is not empty ({len(working)} segments)")
        report = doc.get("report")
        if not isinstance(report, str) or not report.strip():
            errors.append(f"{clip_id}: clip_status='{status}' requires non-empty report with Quick ID and description")
        return errors

    valid_time_segs = []

    for idx, w in enumerate(working):
        prefix = f"{clip_id} working[{idx}]"
        if not isinstance(w, dict):
            errors.append(f"{prefix}: segment must be a JSON object")
            continue
        start = w.get("start")
        end = w.get("end")

        if start is None or end is None:
            errors.append(f"{prefix}: Missing start or end timestamp")
        else:
            try:
                if isinstance(start, bool) or isinstance(end, bool):
                    raise ValueError("Boolean timestamps are not numeric timestamps")
                s_val = float(start)
                e_val = float(end)
                if not math.isfinite(s_val) or not math.isfinite(e_val):
                    errors.append(f"{prefix}: timestamps must be finite numbers")
                    continue
                if s_val < 0:
                    errors.append(f"{prefix}: start timestamp cannot be negative ({s_val})")
                if s_val >= e_val:
                    errors.append(f"{prefix}: start ({start}) must be strictly less than end ({end})")
                else:
                    valid_time_segs.append((s_val, e_val, idx))
            except (ValueError, TypeError):
                errors.append(f"{prefix}: Invalid numeric timestamp format")

        step = w.get("step_number")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            errors.append(f"{prefix}: step_number must be a positive integer (got {type(step).__name__}: {step})")

        verdict = w.get("verdict")
        if not isinstance(verdict, str) or verdict not in ALLOWED_VERDICTS:
            errors.append(f"{prefix}: Invalid verdict '{verdict}' (expected one of {sorted(ALLOWED_VERDICTS)})")

        if verdict == Verdict.WRONG.value:
            err_type = w.get("error_type")
            if not isinstance(err_type, str) or err_type not in ALLOWED_ERROR_TYPES:
                errors.append(
                    f"{prefix}: verdict='错' requires error_type in 7 standard classes (got '{err_type}')"
                )
            evidence = w.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                errors.append(f"{prefix}: verdict='错' requires non-empty template-based evidence")
        elif verdict == Verdict.CORRECT.value:
            if w.get("error_type"):
                errors.append(f"{prefix}: verdict='对' must not have error_type (found '{w.get('error_type')}')")

    # Check for overlapping working segments
    valid_time_segs.sort(key=lambda x: x[0])
    for k in range(1, len(valid_time_segs)):
        prev_s, prev_e, prev_idx = valid_time_segs[k - 1]
        curr_s, curr_e, curr_idx = valid_time_segs[k]
        if curr_s < prev_e:
            errors.append(
                f"{clip_id}: working segments overlap in time: working[{prev_idx}] ({prev_s}..{prev_e}) and working[{curr_idx}] ({curr_s}..{curr_e})"
            )

    if duration is not None and any(end > duration for _, end, _ in valid_time_segs):
        errors.append(f"{clip_id}: duration must be at least the maximum working end timestamp")

    return errors


def validate_embodied_export(file_or_dir: str) -> Dict[str, Any]:
    """
    Validate a single JSON/JSONL export file or a folder of exports.
    """
    targets = []
    if os.path.isdir(file_or_dir):
        for root, _, files in os.walk(file_or_dir):
            for f in files:
                if f.endswith((".json", ".jsonl")):
                    targets.append(os.path.join(root, f))
    else:
        targets.append(file_or_dir)

    total_records = 0
    all_violations: List[str] = []

    for t in targets:
        with open(t, "r", encoding="utf-8") as f:
            if t.lower().endswith(".jsonl"):
                entries = ((str(line_no), line) for line_no, line in enumerate(f, 1) if line.strip())
            else:
                # JSON is a complete document and commonly contains indentation;
                # only JSONL should be parsed one physical line at a time.
                entries = [("document", f.read())]
            for location, content in entries:
                try:
                    data = json.loads(content)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        total_records += 1
                        if not isinstance(item, dict):
                            all_violations.append(f"{t}:{location}: Annotation record must be a JSON object")
                        else:
                            all_violations.extend(validate_embodied_doc(item))
                except json.JSONDecodeError as e:
                    all_violations.append(f"{t}:{location}: JSON decode error: {e}")

    return {
        "files_checked": len(targets),
        "records_checked": total_records,
        "violations_count": len(all_violations),
        "passed": len(all_violations) == 0,
        "violations": all_violations
    }
