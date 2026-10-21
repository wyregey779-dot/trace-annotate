"""
Embodied Annotation Validator & Compliance Checker.
Ensures strict compliance with Annotation Specification v2.1:
  - Diagnostic consistency (non-normal clips must have empty working and required report).
  - Verdict consistency (wrong segments must specify error_type in 7 classes with non-empty evidence).
  - Step numbering integrity.
"""
from __future__ import annotations
import os
import json
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
    clip_id = doc.get("clip", "unknown_clip")

    status = doc.get("clip_status", ClipStatus.NORMAL.value)
    if status not in ALLOWED_CLIP_STATUSES:
        errors.append(f"{clip_id}: Invalid clip_status '{status}'")

    if status != ClipStatus.NORMAL.value:
        working = doc.get("working", [])
        if working:
            errors.append(f"{clip_id}: clip_status='{status}' but working segments array is not empty ({len(working)} segments)")
        report = (doc.get("report") or "").strip()
        if not report:
            errors.append(f"{clip_id}: clip_status='{status}' requires non-empty report with Quick ID and description")
        return errors

    working = doc.get("working", [])
    for idx, w in enumerate(working):
        prefix = f"{clip_id} working[{idx}]"
        start = w.get("start")
        end = w.get("end")

        if start is None or end is None:
            errors.append(f"{prefix}: Missing start or end timestamp")
        elif float(start) >= float(end):
            errors.append(f"{prefix}: start ({start}) must be strictly less than end ({end})")

        step = w.get("step_number")
        if not isinstance(step, int):
            errors.append(f"{prefix}: step_number must be an integer (got {type(step).__name__}: {step})")

        verdict = w.get("verdict")
        if verdict not in ALLOWED_VERDICTS:
            errors.append(f"{prefix}: Invalid verdict '{verdict}' (expected one of {sorted(ALLOWED_VERDICTS)})")

        if verdict == Verdict.WRONG.value:
            err_type = w.get("error_type")
            if err_type not in ALLOWED_ERROR_TYPES:
                errors.append(
                    f"{prefix}: verdict='错' requires error_type in 7 standard classes (got '{err_type}')"
                )
            evidence = (w.get("evidence") or "").strip()
            if not evidence:
                errors.append(f"{prefix}: verdict='错' requires non-empty template-based evidence")
        elif verdict == Verdict.CORRECT.value:
            if w.get("error_type"):
                errors.append(f"{prefix}: verdict='对' must not have error_type (found '{w.get('error_type')}')")

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
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, list):
                        for item in data:
                            total_records += 1
                            all_violations.extend(validate_embodied_doc(item))
                    else:
                        total_records += 1
                        all_violations.extend(validate_embodied_doc(data))
                except json.JSONDecodeError as e:
                    all_violations.append(f"{t}:{line_no}: JSON decode error: {e}")

    return {
        "files_checked": len(targets),
        "records_checked": total_records,
        "violations_count": len(all_violations),
        "passed": len(all_violations) == 0,
        "violations": all_violations
    }
