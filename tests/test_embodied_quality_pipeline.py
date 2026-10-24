"""Quality admission must propagate every gate without mutating source data."""
import copy
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from trace_annotate.embodied.quality_io import (
    QualityInputError, QualityViolation, collect_annotations, load_annotations,
    load_docs_from_file,
)
from trace_annotate.embodied.quality_pipeline import (
    agreement_passed, complete_document, evaluate_quality, guarded, idle_command, legacy_main,
    run_quality_pipeline,
)
from trace_annotate.cli import cmd_embodied_export
from trace_annotate.embodied.exporters.srt import export_to_srt, format_srt_time


def annotation(clip="sample.mp4"):
    return {"clip": clip, "duration": 10.0, "clip_status": "正常",
            "text_description": "按规定安装面板", "expected_step_count": 1,
            "working": [{"start": 2.0, "end": 6.0, "step_number": 1, "verdict": "对"}],
            "idle": [{"start": 0.0, "end": 2.0}, {"start": 6.0, "end": 10.0}]}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("suffix,value", [(".json", annotation()), (".json", [annotation()]),
                                          (".jsonl", [annotation()])])
def test_pretty_object_or_array_is_read_without_loss(tmp_path, suffix, value):
    path = tmp_path / ("input" + suffix)
    save(path, value)
    assert load_docs_from_file(str(path)) == [annotation()]


def test_jsonl_and_directory_include_all_records_and_file_hashes(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "a.jsonl").write_text(json.dumps(annotation("a")) + "\n" + json.dumps(annotation("b")), encoding="utf-8")
    save(folder / "b.json", annotation("c"))
    batch = collect_annotations(str(folder))
    assert not batch.errors
    assert [doc["clip"] for doc in batch.docs] == ["a", "b", "c"]
    assert len(batch.files) == 2
    assert all(len(info["sha256"]) == 64 for info in batch.files)


@pytest.mark.parametrize("payload", ["", "[]", "42", "null", '[{} , 42]', '{"clip":"a","clip":"b"}', '{"duration":NaN}'])
def test_empty_scalar_duplicate_fields_and_nonfinite_inputs_fail_closed(tmp_path, payload):
    source = tmp_path / "bad.json"
    source.write_text(payload, encoding="utf-8")
    report = evaluate_quality(str(source))
    assert report["exit_code"] == 2
    assert not report["passed"]
    assert report["input_errors"]


def test_bad_jsonl_line_is_explicit_and_cannot_leave_a_passed_subset(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(annotation()) + "\nbroken\n" + json.dumps(annotation("b")), encoding="utf-8")
    result = evaluate_quality(str(source))
    assert result["exit_code"] == 2
    assert any(":2:" in error for error in result["input_errors"])
    assert result["results"] == []


def test_jsonl_error_uses_physical_line_after_leading_blank_lines(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text("\n\n" + json.dumps(annotation()) + "\nbroken\n", encoding="utf-8")
    result = evaluate_quality(str(source))
    assert any(":4:" in error for error in result["input_errors"])


def test_invalid_unhashable_fields_with_duplicate_valid_ids_do_not_crash(tmp_path):
    malformed = annotation()
    malformed.update(clip=["invalid"], clip_status=["invalid"])
    result = evaluate_quality(save(tmp_path / "input.json", [annotation(), annotation(), malformed]))
    assert result["exit_code"] == 1
    assert len(result["results"]) == 3


def test_custom_comparison_threshold_uses_unrounded_kappa():
    metrics = {"gate_eligible": True, "meets_trial_bar": True, "coverage_complete": True,
               "semantic_complete": True, "kappa": .9, "kappa_unrounded": .899999}
    assert agreement_passed(metrics, .9) is False


def test_empty_directory_is_input_failure_not_zero_of_zero_pass(tmp_path):
    result = evaluate_quality(str(tmp_path))
    assert result["exit_code"] == 2 and result["passed"] is False


def test_deep_json_is_input_error_without_unhandled_recursion(tmp_path):
    source = tmp_path / "deep.json"
    source.write_text('[' * 1500 + '0' + ']' * 1500, encoding="utf-8")
    result = evaluate_quality(str(source))
    assert result["exit_code"] == 2 and result["input_errors"]


def test_nan_error_retains_the_field_path(tmp_path):
    source = tmp_path / "nan.json"
    source.write_text('{"clip":"a", "duration":NaN}', encoding="utf-8")
    result = evaluate_quality(str(source))
    assert any("$.duration" in error for error in result["input_errors"])


def test_complete_annotation_pass_is_scoped_and_source_hash_is_real(tmp_path):
    source = tmp_path / "input.json"
    save(source, annotation())
    before = source.read_bytes()
    result = run_quality_pipeline(str(source), html_out=str(tmp_path / "fresh.html"))
    assert result["exit_code"] == 0 and result["status"] == "PASS_SCOPED"
    assert result["files"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["video_content_verified"] is False
    assert result["annotator_independence"] == "NOT_VERIFIED"
    assert source.read_bytes() == before


def test_step_failure_cannot_pass_pipeline_just_because_fields_are_valid(tmp_path):
    doc = annotation()
    doc["expected_step_count"] = 2
    result = evaluate_quality(save(tmp_path / "input.json", doc))
    assert result["exit_code"] == 1 and not result["passed"]
    assert result["results"][0]["diff"]["missing_steps"] == [2]
    assert result["status"] == "REVIEW_REQUIRED"
    record = result["results"][0]
    assert record["structure"]["passed"] is True and record["errors"] == []
    assert record["procedure"]["status"] == "REVIEW_REQUIRED"
    assert record["procedure"]["video_execution_verified"] is False


def test_expected_coverage_must_not_be_inferred_from_last_observed_step(tmp_path):
    doc = annotation()
    del doc["expected_step_count"]
    result = evaluate_quality(save(tmp_path / "input.json", doc))
    assert result["exit_code"] == 1
    assert any("STEP_COVERAGE_UNKNOWN" in finding for finding in result["results"][0]["procedure"]["findings"])
    assert result["results"][0]["structure"]["passed"] is True


def test_a_faithful_wrong_execution_annotation_is_not_a_structure_error(tmp_path):
    doc = annotation()
    doc["expected_step_count"] = 2
    doc["working"][0].update(verdict="错", error_type="遗漏步骤", evidence="第2步全程未出现")
    result = run_quality_pipeline(save(tmp_path / "input.json", doc), html_out=str(tmp_path / "review.html"))
    record = result["results"][0]
    assert record["doc"] == doc
    assert record["errors"] == [] and record["structure"]["status"] == "PASS_SCOPED"
    assert record["procedure"]["status"] == result["status"] == "REVIEW_REQUIRED"
    assert result["exit_code"] == 1 and not result["passed"]
    html = (tmp_path / "review.html").read_text(encoding="utf-8")
    assert "结构检查" in html and "步骤对账" in html and "不能据此断言标注错误" in html


def test_valid_wrong_object_label_can_pass_machine_scope_without_claiming_video_truth(tmp_path):
    doc = annotation()
    doc["working"][0].update(verdict="错", error_type="用错物体", evidence="t=3s 用了改锥应为扳手")
    result = evaluate_quality(save(tmp_path / "input.json", doc))
    assert result["passed"] and result["status"] == "PASS_SCOPED"
    assert not result["video_content_verified"]


def test_hard_structure_failure_has_priority_over_another_record_pending_review(tmp_path):
    review = annotation("review.mp4")
    review["expected_step_count"] = 2
    invalid = annotation("invalid.mp4")
    invalid["working"][0]["start"] = -1
    result = evaluate_quality(save(tmp_path / "input.json", [review, invalid]))
    assert result["status"] == "FAILED" and result["exit_code"] == 1
    assert result["results"][0]["status"] == "REVIEW_REQUIRED"
    assert result["results"][1]["structure"]["status"] == "FAILED"


@pytest.mark.parametrize("field", ["duration", "unrelated_metadata"])
def test_json_exponent_overflow_is_rejected_at_the_input_boundary(tmp_path, field):
    source = tmp_path / "overflow.json"
    source.write_text('{"' + field + '":1e999}', encoding="utf-8")
    result = evaluate_quality(str(source))
    assert result["status"] == "INPUT_ERROR" and result["exit_code"] == 2
    assert any("$." + field in error for error in result["input_errors"])


def test_auto_idle_repairs_only_separate_copy_then_checks_full_coverage(tmp_path):
    doc = annotation()
    doc["idle"] = []
    path = tmp_path / "input.json"
    save(path, doc)
    before = path.read_bytes()
    result = run_quality_pipeline(str(path), auto_idle=True, html_out=str(tmp_path / "report.html"),
                                  corrected_out=str(tmp_path / "corrected.json"))
    assert result["passed"]
    assert result["results"][0]["correction"]["applied"]
    assert path.read_bytes() == before
    assert json.loads((tmp_path / "corrected.json").read_text(encoding="utf-8"))[0]["idle"] == annotation()["idle"]


@pytest.mark.parametrize("alter", [
    lambda doc: doc["working"].append({"start": 3, "end": 7, "step_number": 2, "verdict": "对"}),
    lambda doc: doc.update(duration=None),
    lambda doc: doc["working"][0].update(start=7, end=2),
    lambda doc: doc.update(idle=[{"start": 0, "end": 5}]),
    lambda doc: doc["working"][0].update(mid_segments=[{"start": 1, "end": 3, "reason": "idle"}]),
    lambda doc: doc["working"][0].update(verdict="错", error_type="意外事故", evidence="随便写的"),
])
def test_auto_idle_cannot_mask_a_preexisting_invalid_annotation(tmp_path, alter):
    doc = annotation()
    alter(doc)
    source = tmp_path / "input.json"
    save(source, doc)
    before = source.read_bytes()
    result = run_quality_pipeline(str(source), auto_idle=True, html_out=str(tmp_path / "report.html"),
                                  corrected_out=str(tmp_path / "corrected.json"))
    assert result["exit_code"] == 1
    assert not (tmp_path / "corrected.json").exists()
    assert source.read_bytes() == before


def test_partial_work_remains_review_required_and_is_not_rewritten(tmp_path):
    doc = annotation()
    doc.update(clip_status="描述不符-仅部分工作", report="Quick ID sample：只完成安装的一部分")
    result = evaluate_quality(save(tmp_path / "input.json", doc), auto_idle=True)
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["results"][0]["doc"]["working"] == doc["working"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "semantic", "degenerate"])
def test_comparison_failure_propagates_despite_high_temporal_agreement(tmp_path, change):
    a, b = [annotation()], [annotation()]
    if change == "missing":
        a.append(annotation("other.mp4"))
    elif change == "duplicate":
        b.append(annotation())
    elif change == "semantic":
        b[0]["working"][0].update(verdict="错", error_type="其他", evidence="t=3s 操作与规定不符合")
    else:
        for docs in (a, b):
            docs[0]["working"][0].update(start=0, end=10)
            docs[0]["idle"] = []
    result = evaluate_quality(save(tmp_path / "a.json", a), save(tmp_path / "b.json", b))
    assert result["exit_code"] == 1
    assert result["agreement"]["gate_passed"] is False


def test_identical_complete_comparison_passes_without_claiming_independent_people(tmp_path):
    path_a = save(tmp_path / "a.json", annotation())
    path_b = save(tmp_path / "b.json", annotation())
    result = evaluate_quality(path_a, path_b)
    assert result["passed"] and result["agreement"]["gate_passed"]
    assert result["agreement"]["kappa"] == 1
    assert result["agreement"]["annotator_independence"] == "NOT_VERIFIED"


def test_duplicate_clip_mapping_loader_refuses_overwrite(tmp_path):
    source = save(tmp_path / "input.json", [annotation(), annotation()])
    with pytest.raises(QualityInputError, match="duplicate clip"):
        load_annotations(source)


def test_report_escapes_every_untrusted_text_context(tmp_path):
    doc = annotation('\"><img src=x onerror="alert(1)">')
    doc["text_description"] = "</pre><script>alert(2)</script> __ERROR_ROWS__"
    report_path = tmp_path / "report.html"
    run_quality_pipeline(save(tmp_path / "input.json", doc), html_out=str(report_path))
    html = report_path.read_text(encoding="utf-8")
    assert "<script>" not in html and "<img src=" not in html
    assert "&lt;script&gt;" in html and "&lt;img src=" in html
    assert "__ERROR_ROWS__" in html
    assert "没有核实独立人工标注身份" in html


def test_existing_report_is_preserved_and_cannot_be_claimed_as_new(tmp_path):
    source = save(tmp_path / "input.json", annotation())
    output = tmp_path / "report.html"
    output.write_bytes(b"OLD SUCCESS REPORT")
    with pytest.raises(QualityInputError, match="already exists"):
        run_quality_pipeline(source, html_out=str(output))
    assert output.read_bytes() == b"OLD SUCCESS REPORT"


def test_report_cannot_replace_input_and_output_alias_is_refused(tmp_path):
    source = tmp_path / "input.json"
    save(source, annotation())
    original = source.read_bytes()
    with pytest.raises(QualityInputError):
        run_quality_pipeline(str(source), html_out=str(source))
    assert source.read_bytes() == original
    with pytest.raises(QualityInputError, match="different paths"):
        run_quality_pipeline(str(source), auto_idle=True, html_out=str(tmp_path / "same"),
                             corrected_out=str(tmp_path / "same"))


def test_standalone_idle_default_is_stdout_and_does_not_mutate_input(tmp_path, capsys):
    doc = annotation()
    doc["idle"] = []
    source = tmp_path / "input.json"
    save(source, doc)
    before = source.read_bytes()
    idle_command(Namespace(file=str(source), duration=None, inplace=False, output=None))
    assert json.loads(capsys.readouterr().out)["idle"] == annotation()["idle"]
    assert source.read_bytes() == before


def test_explicit_inplace_is_validated_atomic_and_preserves_original_backup(tmp_path, capsys):
    doc = annotation()
    doc["idle"] = []
    source = tmp_path / "input.json"
    save(source, doc)
    before = source.read_bytes()
    idle_command(Namespace(file=str(source), duration=12.0, inplace=True, output=None))
    backups = list(tmp_path.glob("*.before-idle-*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert json.loads(source.read_text(encoding="utf-8"))["idle"][-1]["end"] == 12.0


def test_invalid_explicit_inplace_leaves_original_and_no_backup(tmp_path):
    doc = annotation()
    doc["working"][0]["end"] = 1
    source = tmp_path / "input.json"
    save(source, doc)
    before = source.read_bytes()
    code = guarded(idle_command, Namespace(file=str(source), duration=None, inplace=True, output=None))
    assert code == 1 and source.read_bytes() == before
    assert not list(tmp_path.glob("*.bak"))


@pytest.mark.parametrize("kind,extra", [("validate", []), ("diff", []), ("kappa", ["COMPARE"])])
def test_legacy_entrypoints_return_nonzero_on_quality_failure(tmp_path, kind, extra):
    doc = annotation()
    doc["working"][0]["step_number"] = False
    source = save(tmp_path / "input.json", doc)
    compare = save(tmp_path / "other.json", annotation())
    args = [source] + ([compare] if extra else [])
    if kind == "diff":
        args += ["--nsteps", "1"]
    assert legacy_main(kind, args) == 1


@pytest.mark.parametrize("command", ["embodied-pipeline", "embodied-qc"])
def test_packaged_cli_commands_propagate_gate_failure_and_emit_fresh_report(tmp_path, command):
    doc = annotation()
    doc["expected_step_count"] = 2
    source = save(tmp_path / "input.json", doc)
    output = str(tmp_path / "fresh.html")
    option = "--report" if command == "embodied-pipeline" else "--output"
    result = subprocess.run([sys.executable, "-B", "-c", "from trace_annotate.cli import main; main()",
                             command, source, option, output], capture_output=True, text=True,
                            encoding="utf-8", cwd=str(Path(__file__).parents[1]))
    assert result.returncode == 1, result.stdout + result.stderr
    assert Path(output).is_file()
    assert json.loads(result.stdout)["passed"] is False


def test_malformed_input_exit_two_still_has_a_fresh_failure_report(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text("not json", encoding="utf-8")
    report = tmp_path / "fresh.html"
    assert legacy_main("pipeline", [str(source), "--report", str(report)]) == 2
    assert "INPUT_ERROR" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("format", ["srt", "cvat", "action"])
def test_cli_export_never_overwrites_an_input_or_existing_destination(tmp_path, format):
    source = tmp_path / "input.json"
    save(source, annotation())
    original = source.read_bytes()
    with pytest.raises(QualityInputError):
        cmd_embodied_export(Namespace(file=str(source), output=str(source), format=format))
    assert source.read_bytes() == original
    output = tmp_path / ("existing" if format == "action" else "existing." + format)
    if format == "action":
        output.mkdir()
        marker = output / "mapping.txt"
    else:
        marker = output
    marker.write_bytes(b"existing user data")
    with pytest.raises(QualityInputError):
        cmd_embodied_export(Namespace(file=str(source), output=str(output), format=format))
    assert marker.read_bytes() == b"existing user data"


def test_cli_export_rejects_duplicate_json_fields_and_multiple_clip_subtitles(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(annotation())[:-1] + ', "duration":-1, "duration":10}', encoding="utf-8")
    output = tmp_path / "output.srt"
    with pytest.raises(QualityInputError):
        cmd_embodied_export(Namespace(file=str(source), output=str(output), format="srt"))
    save(source, [annotation(), annotation("other.mp4")])
    with pytest.raises(QualityInputError, match="exactly one clip"):
        cmd_embodied_export(Namespace(file=str(source), output=str(output), format="srt"))
    assert not output.exists()


@pytest.mark.parametrize("seconds,timestamp", [(1.9999, "00:00:02,000"), (59.9999, "00:01:00,000"),
                                               (3599.9999, "01:00:00,000"), (.0005, "00:00:00,001")])
def test_embodied_srt_uses_shared_millisecond_rounding_with_carry(seconds, timestamp):
    assert format_srt_time(seconds) == timestamp


def test_embodied_srt_refuses_precision_collapse_and_evidence_cue_injection():
    doc = annotation()
    doc["working"][0].update(start=2.00001, end=2.00002)
    with pytest.raises(ValueError, match="collapses"):
        export_to_srt(doc)
    doc = annotation()
    doc["working"][0].update(verdict="错", error_type="用错物体", evidence="t=3s 用了改锥应为扳手\n\n99\n00:00:00,000 --> 00:00:09,000\n伪造字幕")
    with pytest.raises(ValueError, match="blank subtitle lines"):
        export_to_srt(doc)
