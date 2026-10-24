"""One quality pipeline shared by the packaged CLI and legacy launch scripts.

Exit codes: 0 = declared machine-check scope passed, 1 = a quality/review gate
failed, 2 = invalid invocation, malformed input or an I/O failure. These tools do
not watch the video or establish that two files came from independent annotators.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agreement import calculate_temporal_kappa, build_temporal_mask
from .complement import compute_idle_complement
from .differ import diff_expected_steps
from .schema import ClipStatus
from .validator import validate_embodied_doc
from .quality_io import (
    AnnotationBatch, QualityInputError, QualityViolation, annotation_json,
    collect_annotations, load_annotations, load_docs_from_file,
    replace_annotation_explicitly, require_new_output, write_new_text,
)


SCOPE = ("Machine checks of supplied annotation fields, declared step coverage and "
         "sampled working/idle agreement. No video-content verification, ground-truth "
         "certification or independent-human-annotator verification.")
PROCEDURE_SCOPE = ("Differences from the declared procedure require adjudication. "
                   "They may faithfully describe an execution error in the video; "
                   "they do not establish that the annotation is incorrect.")


def positive_number(value) -> float:
    if isinstance(value, bool):
        raise ValueError("duration must be a finite positive number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("duration must be a finite positive number") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError("duration must be a finite positive number")
    return result


def _check_options(nsteps=None, fps=2.0, min_kappa=0.80):
    if nsteps is not None and (isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 1):
        raise QualityInputError("nsteps must be a positive integer from the independent procedure")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise QualityInputError("fps/sample rate must be finite and positive")
    if isinstance(min_kappa, bool) or not isinstance(min_kappa, (int, float)) or not math.isfinite(min_kappa) or not 0.70 <= min_kappa <= 1:
        raise QualityInputError("min-kappa must be between 0.70 and 1.0")


def complete_document(doc, duration_arg=None, *, require_description=False):
    """Prepare a separate repaired document, without changing the caller's data.

    Idle may fill gaps, but cannot hide malformed working/idle/mid/evidence data.
    A duration must be explicitly declared or supplied, never inferred from the
    final working endpoint (which would silently discard trailing idle).
    """
    if not isinstance(doc, dict):
        raise QualityViolation("Annotation must be a JSON object")
    prepared = copy.deepcopy(doc)
    duration = positive_number(duration_arg if duration_arg is not None else prepared.get("duration"))
    prepared["duration"] = duration
    errors = validate_embodied_doc(prepared, require_duration=True,
                                   require_description=require_description,
                                   require_idle_coverage=False)
    blockers = [error for error in errors if not error.startswith("IDLE_COVERAGE")]
    if blockers:
        raise QualityViolation("; ".join(blockers))
    status = prepared.get("clip_status", ClipStatus.NORMAL.value)
    if status != ClipStatus.NORMAL.value:
        # Diagnostic branches that require no timeline must stay empty. Partial
        # work remains REVIEW_REQUIRED and is rejected above, not auto-approved.
        return prepared, len(prepared.get("working", [])), len(prepared.get("idle", [])), duration
    prepared["idle"] = compute_idle_complement(prepared["working"], duration)
    errors = validate_embodied_doc(prepared, require_duration=True,
                                   require_description=require_description,
                                   require_idle_coverage=True)
    if errors:
        raise QualityViolation("; ".join(errors))
    return prepared, len(prepared["working"]), len(prepared["idle"]), duration


def analyze_step_diff(working_segments, nsteps=None):
    """Legacy field aliases over the canonical step differ; no separate algorithm."""
    result = diff_expected_steps(working_segments, expected_step_count=nsteps)
    compliant = bool(result.get("is_perfect_sequence")) and bool(result.get("coverage_known"))
    return {**result, "is_compliant": compliant,
            "status": "PASS_SCOPED" if compliant else "REVIEW_REQUIRED",
            "interpretation": PROCEDURE_SCOPE, "video_execution_verified": False,
            "missing_steps": result["missing"], "out_of_order_steps": result["out_of_order"],
            "duplicate_steps": result["duplicate"], "expected_nsteps": nsteps,
            "total_working_segments": len(working_segments) if isinstance(working_segments, list) else 0}


def calculate_cohens_kappa(annotator_a, annotator_b, fps=2.0):
    """Legacy metric aliases over the canonical temporal/semantic calculation."""
    _check_options(fps=fps)
    result = calculate_temporal_kappa(annotator_a, annotator_b, sample_rate_hz=fps)
    return {**result, "overall_kappa": result.get("kappa"),
            "overall_p0": result.get("observed_agreement_p0"),
            "overall_pe": result.get("expected_chance_pe"),
            "common_clips_count": result.get("common_clips"),
            "total_seconds_evaluated": result.get("total_samples", 0) / fps,
            "measurement_scope": "sampled binary working/idle timeline; not semantic or visual truth",
            "annotator_independence": "NOT_VERIFIED"}


def agreement_passed(metrics, minimum=0.80):
    value = metrics.get("kappa_unrounded", metrics.get("kappa"))
    bar = "meets_trial_bar" if minimum >= 0.80 else "meets_baseline_bar"
    return (metrics.get("gate_eligible") is True and metrics.get(bar) is True and metrics.get("coverage_complete") is True
            and metrics.get("semantic_complete") is True
            and not metrics.get("validation_errors") and not metrics.get("semantic_conflicts")
            and not metrics.get("unmatched_segments")
            and not metrics.get("is_degenerate_single_class")
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


def _evaluate_records(batch: AnnotationBatch, nsteps, auto_idle):
    results, duplicate_ids = [], set()
    seen = set()
    for record in batch.records:
        original = record.doc
        doc = copy.deepcopy(original)
        errors, review_findings = [], []
        correction = {"requested": auto_idle, "applied": False}
        clip = doc.get("clip")
        if isinstance(clip, str):
            if clip in seen:
                duplicate_ids.add(clip)
            seen.add(clip)
        if auto_idle:
            try:
                doc, _, _, _ = complete_document(doc, require_description=True)
                correction["applied"] = doc.get("idle") != original.get("idle")
            except (ValueError, TypeError, KeyError) as error:
                correction["blocked_reason"] = str(error)
        violations = validate_embodied_doc(doc, require_duration=True,
                                          require_description=True, require_idle_coverage=True)
        errors.extend(error for error in violations if not error.startswith("REVIEW_REQUIRED:"))
        review_findings.extend(error for error in violations if error.startswith("REVIEW_REQUIRED:"))
        expected = nsteps if nsteps is not None else doc.get("expected_step_count")
        if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int) or expected < 1):
            errors.append("EXPECTED_STEPS_INVALID: expected_step_count must be a positive integer")
            expected = None
        status = doc.get("clip_status", ClipStatus.NORMAL.value)
        active = isinstance(status, str) and status in {
            ClipStatus.NORMAL.value, ClipStatus.DESC_MISMATCH_PART.value}
        diff = {"is_compliant": True, "coverage_known": None, "status": "NOT_APPLICABLE"}
        procedure_findings = []
        if active:
            try:
                diff = analyze_step_diff(doc.get("working"), expected)
                if not diff.get("coverage_known"):
                    procedure_findings.append("STEP_COVERAGE_UNKNOWN: supply --nsteps or declared expected_step_count; the observed maximum is not an independent procedure")
                elif not diff.get("is_compliant"):
                    procedure_findings.append("STEP_DIFF_REVIEW_REQUIRED: missing, extra, invalid, repeated or out-of-order steps; reconcile the procedure and video before adjudication")
            except (ValueError, TypeError, KeyError) as error:
                diff = {"is_compliant": False, "coverage_known": False, "status": "REVIEW_REQUIRED", "error": str(error)}
                procedure_findings.append(f"STEP_DIFF_REVIEW_REQUIRED: {error}")
        errors = list(dict.fromkeys(errors))
        review_findings = list(dict.fromkeys(review_findings + procedure_findings))
        procedure = {"status": "NOT_APPLICABLE" if not active else "REVIEW_REQUIRED" if procedure_findings else "PASS_SCOPED",
                     "passed": not procedure_findings, "findings": procedure_findings,
                     "interpretation": PROCEDURE_SCOPE, "video_execution_verified": False}
        status = "FAILED" if errors else "REVIEW_REQUIRED" if review_findings else "PASS_SCOPED"
        results.append({"file": record.path, "location": record.location, "source_sha256": record.sha256,
                        "doc": doc, "errors": errors, "diff": diff, "correction": correction,
                        "structure": {"status": "FAILED" if errors else "PASS_SCOPED", "passed": not errors,
                                      "errors": list(errors)}, "procedure": procedure,
                        "review_findings": review_findings, "status": status,
                        "passed": status == "PASS_SCOPED"})
    if duplicate_ids:
        for result in results:
            if isinstance(result["doc"].get("clip"), str) and result["doc"]["clip"] in duplicate_ids:
                result["errors"].append("DUPLICATE_CLIP_ID: repeated clip IDs cannot be silently merged")
                result["structure"] = {"status": "FAILED", "passed": False, "errors": list(result["errors"])}
                result["status"] = "FAILED"
                result["passed"] = False
    return results


def evaluate_quality(target_path, compare_path=None, auto_idle=False, nsteps=None,
                     fps=2.0, min_kappa=0.80):
    _check_options(nsteps, fps, min_kappa)
    batch = collect_annotations(target_path)
    results = _evaluate_records(batch, nsteps, auto_idle)
    comparison = None
    compare_results = []
    input_errors = list(batch.errors)
    comparison_files = []
    if compare_path:
        other = collect_annotations(compare_path)
        input_errors.extend(other.errors)
        comparison_files = other.files
        compare_results = _evaluate_records(other, nsteps, False)
        comparison = calculate_cohens_kappa([r["doc"] for r in results], other.docs, fps)
        comparison["gate_passed"] = agreement_passed(comparison, min_kappa)
        comparison["minimum_kappa"] = min_kappa
        comparison["scope"] = SCOPE
    passed = (bool(results) and not input_errors and all(result["passed"] for result in results)
              and all(result["passed"] for result in compare_results)
              and (comparison is None or comparison["gate_passed"]))
    status = ("INPUT_ERROR" if input_errors else "FAILED" if any(not r["structure"]["passed"] for r in results + compare_results)
              else "PASS_SCOPED" if passed else "REVIEW_REQUIRED")
    return {"schema": "trace-annotate.embodied-quality/1", "run_id": uuid.uuid4().hex,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "passed": passed, "exit_code": 2 if input_errors else 0 if passed else 1,
            "scope": SCOPE, "files": batch.files, "comparison_files": comparison_files,
            "input_errors": input_errors, "results": results, "comparison_results": compare_results,
            "agreement": comparison, "auto_idle": auto_idle,
            "original_annotations_modified": False, "video_content_verified": False,
            "annotator_independence": "NOT_VERIFIED"}


def render_quality_html(report):
    """Static HTML: untrusted values are text, never markup, attributes or scripts."""
    def text(value):
        return escape(str(value), quote=True)
    rows = []
    for result in report.get("results", []) + report.get("comparison_results", []):
        doc = result["doc"]
        messages = "\n".join(result["errors"] + result.get("review_findings", [])) or "Declared machine-check scope passed"
        structure = result.get("structure", {}).get("status", "FAILED" if result["errors"] else "PASS_SCOPED")
        procedure = result.get("procedure", {}).get("status", result.get("diff", {}).get("status", "NOT_EVALUATED"))
        rows.append("<tr><td>" + text(doc.get("clip", "(missing clip)")) + "</td><td>"
                    + text(doc.get("text_description", "")) + "</td><td>" + text(result["file"])
                    + " " + text(result["location"]) + "</td><td>" + text(structure) + "</td><td>" + text(procedure)
                    + "</td><td>" + text(result.get("status", "PASS_SCOPED" if result["passed"] else "FAILED"))
                    + "</td><td><pre>" + text(messages) + "</pre></td></tr>")
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    return ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
            "<title>TraceAnnotate annotation quality report</title>"
            "<style>body{font:16px system-ui,sans-serif;margin:2rem;line-height:1.5}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:.5rem;text-align:left;vertical-align:top}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}td{overflow-wrap:anywhere}</style></head><body>"
            "<h1>TraceAnnotate 标注文件机器质检</h1><p>状态：<strong>" + text(report["status"])
            + "</strong></p><p>Run: " + text(report["run_id"]) + " · " + text(report["recorded_at"])
            + "</p><p>" + text(report["scope"]) + "</p>"
            "<p>仅比较所提交文件；没有核实独立人工标注身份，也没有观看视频证明动作或判定正确。"
            "模拟/扰动对照只能作为软件测试。时间 Kappa 不等于语义准确率或真实黄金标准。</p>"
            "<p>结构检查与步骤对账分别报告。步骤差异可能真实记录视频中的执行错误，必须仲裁，不能据此断言标注错误。</p>"
            "<h2>输入错误</h2><pre>" + text("\n".join(report.get("input_errors", [])) or "None")
            + "</pre><h2>片段门禁</h2><table><thead><tr><th>Clip</th><th>任务描述</th><th>输入位置</th>"
            "<th>结构检查</th><th>步骤对账</th><th>结果</th><th>原因</th></tr></thead><tbody>" + "".join(rows)
            + "</tbody></table><h2>全部机器检查结果与输入摘要</h2><pre>" + text(payload)
            + "</pre></body></html>\n")


def generate_html_report(all_results, kappa_result=None, out_html_path="qc_report.html"):
    """Compatibility report writer. A report path is always newly created."""
    if isinstance(all_results, dict) and "run_id" in all_results:
        report = all_results
    else:
        normalized = []
        for item in all_results:
            item = dict(item)
            item.setdefault("file", "(API input)")
            item.setdefault("location", "record")
            item["passed"] = not item.get("errors") and bool(item.get("diff", {}).get("is_compliant"))
            normalized.append(item)
        passed = bool(normalized) and all(r["passed"] for r in normalized)
        if kappa_result is not None:
            passed = passed and agreement_passed(kappa_result)
        report = {"run_id": uuid.uuid4().hex, "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "status": "PASS_SCOPED" if passed else "FAILED", "passed": passed,
                  "scope": SCOPE, "results": normalized, "agreement": kappa_result, "input_errors": []}
    inputs = [info["path"] for key in ("files", "comparison_files") for info in report.get(key, [])]
    return write_new_text(out_html_path, render_quality_html(report), inputs)


def run_quality_pipeline(target_path, compare_path=None, auto_idle=False, nsteps=None,
                         html_out="qc_report.html", fps=2.0, min_kappa=0.80, corrected_out=None):
    # Existing reports fail before evaluation. Never announce an old report as new.
    require_new_output(html_out)
    if corrected_out:
        if not auto_idle:
            raise QualityInputError("--corrected-output requires --auto-idle")
        require_new_output(corrected_out)
        if Path(corrected_out).absolute() == Path(html_out).absolute():
            raise QualityInputError("Report and corrected output must be different paths")
    report = evaluate_quality(target_path, compare_path, auto_idle, nsteps, fps, min_kappa)
    if corrected_out:
        # A failed source can still produce a failure report, but never a clean
        # looking corrected export that hides another failed gate.
        if report["passed"]:
            inputs = [info["path"] for key in ("files", "comparison_files") for info in report[key]]
            write_new_text(corrected_out, annotation_json([r["doc"] for r in report["results"]],
                           jsonl=Path(corrected_out).suffix.lower() == ".jsonl"), inputs)
            report["corrected_output"] = str(Path(corrected_out).absolute())
        else:
            report["corrected_output"] = None
    generate_html_report(report, out_html_path=html_out)
    report["report_path"] = str(Path(html_out).absolute())
    return report


def run_pipeline(target_path, compare_path=None, auto_idle=False, nsteps=None, html_out="qc_report.html", **kwargs):
    return run_quality_pipeline(target_path, compare_path, auto_idle, nsteps, html_out, **kwargs)["passed"]


def _summary(report):
    return {key: report[key] for key in ("run_id", "status", "passed", "exit_code", "scope", "input_errors", "report_path")}


def pipeline_command(args):
    report = run_quality_pipeline(args.target, getattr(args, "compare", None), getattr(args, "auto_idle", False),
                                  getattr(args, "nsteps", None), getattr(args, "report", "qc_report.html"),
                                  getattr(args, "fps", 2.0), getattr(args, "min_kappa", .80),
                                  getattr(args, "corrected_output", None))
    print(json.dumps(_summary(report), ensure_ascii=False, indent=2))
    return report["exit_code"]


def validate_command(args):
    batch = collect_annotations(args.target)
    violations = list(batch.errors)
    for record in batch.records:
        violations.extend(f"{record.path} {record.location}: {error}" for error in validate_embodied_doc(
            record.doc, require_duration=True, require_description=True, require_idle_coverage=True))
    clips = [record.doc.get("clip") for record in batch.records]
    if len([clip for clip in clips if isinstance(clip, str)]) != len(set(clip for clip in clips if isinstance(clip, str))):
        violations.append("DUPLICATE_CLIP_ID: duplicate records cannot be silently merged")
    print(json.dumps({"passed": bool(batch.records) and not violations, "records_checked": len(batch.records),
                      "files_checked": len(batch.files), "violations": violations, "scope": SCOPE}, ensure_ascii=False, indent=2))
    return 2 if batch.errors else 1 if violations else 0


def diff_command(args):
    nsteps = getattr(args, "nsteps", None)
    _check_options(nsteps)
    batch = collect_annotations(getattr(args, "file", None) or args.json_file)
    if batch.errors:
        raise QualityInputError("; ".join(batch.errors))
    reports = [{"clip": record.doc.get("clip"), **analyze_step_diff(record.doc.get("working"),
                nsteps if nsteps is not None else record.doc.get("expected_step_count"))} for record in batch.records]
    print(json.dumps(reports[0] if len(reports) == 1 else reports, ensure_ascii=False, indent=2))
    return 0 if all(report["is_compliant"] for report in reports) else 1


def kappa_command(args):
    fps = getattr(args, "fps", 2.0)
    minimum = getattr(args, "min_kappa", .80)
    _check_options(fps=fps, min_kappa=minimum)
    a, b = collect_annotations(args.annotator_a), collect_annotations(args.annotator_b)
    if a.errors or b.errors:
        raise QualityInputError("; ".join(a.errors + b.errors))
    result = calculate_cohens_kappa(a.docs, b.docs, fps)
    result["gate_passed"] = agreement_passed(result, minimum)
    result["scope"] = SCOPE
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["gate_passed"] else 1


def idle_command(args):
    source = getattr(args, "file", None) or args.json_file
    inplace = getattr(args, "inplace", False)
    output = getattr(args, "output", None)
    if inplace and output:
        raise QualityInputError("Use either --inplace or --output, not both")
    if output:
        require_new_output(output, [source])
    batch = collect_annotations(source)
    if batch.errors:
        raise QualityInputError("; ".join(batch.errors))
    if Path(source).is_dir():
        raise QualityInputError("Idle repair accepts one file; use the pipeline for batch inspection")
    original = Path(source).read_bytes()
    if hashlib.sha256(original).hexdigest() != batch.files[0]["sha256"]:
        raise QualityInputError("Input changed after parsing")
    documents = [complete_document(record.doc, getattr(args, "duration", None))[0] for record in batch.records]
    # Keep single-object output compatibility; array inputs remain arrays.
    source_is_object = original.decode("utf-8-sig").lstrip().startswith("{") and len(documents) == 1
    text = annotation_json(documents, single=source_is_object,
                           jsonl=Path(output or source).suffix.lower() == ".jsonl")
    if inplace:
        backup = replace_annotation_explicitly(source, original, text)
        print(f"[OK] Validated idle repair saved atomically. Original backup: {backup}")
    elif output:
        write_new_text(output, text, [source])
        print(f"[OK] New idle-repaired copy: {Path(output).absolute()}")
    else:
        print(text, end="")
    return 0


def add_pipeline_arguments(parser):
    parser.add_argument("target", help="JSON/JSONL file or top-level batch directory")
    parser.add_argument("--compare", "-c")
    parser.add_argument("--auto-idle", action="store_true", help="Repair only valid working timelines in memory; never infer duration")
    parser.add_argument("--nsteps", "-n", type=int, help="Independent expected step count; no maximum-observed-step assumption")
    parser.add_argument("--report", "-r", default="qc_report.html", help="New HTML report path; existing files are refused")
    parser.add_argument("--corrected-output", help="Optional new repaired JSON/JSONL copy, written only after all gates pass")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-kappa", type=float, default=.80)


def guarded(command, args):
    try:
        return command(args)
    except QualityViolation as error:
        print(f"[FAILED] {error}", file=sys.stderr)
        return 1
    except (QualityInputError, OSError, ValueError, TypeError) as error:
        print(f"[INPUT_ERROR] {error}", file=sys.stderr)
        return 2


def legacy_main(kind, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if kind == "pipeline":
        add_pipeline_arguments(parser)
        command = pipeline_command
    elif kind == "validate":
        parser.add_argument("target")
        parser.add_argument("--verbose", "-v", action="store_true")
        command = validate_command
    elif kind == "diff":
        parser.add_argument("file")
        parser.add_argument("--nsteps", "-n", type=int)
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        command = diff_command
    elif kind == "kappa":
        parser.add_argument("annotator_a")
        parser.add_argument("annotator_b")
        parser.add_argument("--fps", type=float, default=2.0)
        parser.add_argument("--min-kappa", type=float, default=.80)
        parser.add_argument("--verbose", "-v", action="store_true")
        command = kappa_command
    elif kind == "idle":
        parser.add_argument("file")
        parser.add_argument("--duration", "-d", type=float)
        parser.add_argument("--output", "-o")
        parser.add_argument("--inplace", action="store_true", help="Explicit atomic replacement with a new backup of original bytes")
        command = idle_command
    else:
        raise ValueError(f"Unknown quality command {kind}")
    return guarded(command, parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(legacy_main("pipeline"))
