"""Behavioral regressions for embodied QC; fixtures are not video ground truth."""
import copy
import json

import pytest

from trace_annotate.embodied.agreement import (
    build_temporal_mask, calculate_segment_tiou, calculate_temporal_kappa,
)
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import ProcedureSpec, diff_expected_steps
from trace_annotate.embodied.validator import validate_embodied_doc, validate_embodied_export


def segment(start=1, end=3, step=1, **extra):
    result = {"start": start, "end": end, "step_number": step, "verdict": "对"}
    result.update(extra)
    return result


def document(**extra):
    result = {"clip": "clip.mp4", "duration": 10, "text_description": "取工具并安装",
              "clip_status": "正常", "working": [segment()],
              "idle": [{"start": 0, "end": 1}, {"start": 3, "end": 10}]}
    result.update(extra)
    return result


def strict(doc):
    return validate_embodied_doc(doc, require_duration=True,
                                require_description=True, require_idle_coverage=True)


@pytest.mark.parametrize("start,end", [
    (-1, 3), (1, 1), (2, 1), (1, 11), (True, 3), (1, False),
    (float("nan"), 3), (1, float("inf")), (None, 3),
])
def test_invalid_times_are_rejected_in_all_working_consumers(start, end):
    working = [segment(start, end)]
    assert strict(document(working=working))
    with pytest.raises(ValueError):
        compute_idle_complement(working, 10)
    with pytest.raises(ValueError):
        build_temporal_mask(working, 10)
    # The step-only differ has no media-duration parameter; the validator owns
    # that boundary. All intrinsic interval failures must still be reported.
    if end != 11:
        assert diff_expected_steps(working, 1)["temporal_errors"]
    agreement = calculate_temporal_kappa([document(working=working)], [document()])
    assert agreement["meets_trial_bar"] is False
    assert agreement["validation_errors"]


@pytest.mark.parametrize("duration", [None, False, 0, -2, float("nan"), float("inf")])
def test_idle_requires_explicit_finite_positive_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        compute_idle_complement([segment()], duration)


def test_nested_overlaps_cannot_hide_later_intervals():
    working = [segment(0, 10), segment(1, 2, 2), segment(3, 4, 3)]
    errors = validate_embodied_doc(document(working=working, idle=[]))
    assert len([error for error in errors if "overlap in time" in error]) == 2
    with pytest.raises(ValueError, match="overlap"):
        compute_idle_complement(working, 10)


def test_idle_preserves_subcentisecond_boundaries_and_does_not_subtract_mid():
    working = [segment(.001, .004, mid_segments=[{"start": .002, "end": .003, "reason": "idle"}]),
               segment(.005, .009, 2)]
    idle = compute_idle_complement(working, .01)
    assert idle == [{"start": 0., "end": .001}, {"start": .004, "end": .005},
                    {"start": .009, "end": .01}]
    assert all(item["start"] < item["end"] for item in idle)
    assert compute_idle_complement([], .001) == [{"start": 0., "end": .001}]


@pytest.mark.parametrize("idle,code", [
    ([{"start": 0, "end": 2}], "IDLE_CONFLICT"),
    ([{"start": -1, "end": 1}], "INVALID_TIME"),
    ([{"start": 3, "end": 11}], "duration"),
    ([{"start": 0, "end": 1}, {"start": .5, "end": 1}], "overlap"),
    ([{"start": 0, "end": 0}], "strictly less"),
    ("not an array", "idle must be an array"),
])
def test_supplied_invalid_idle_is_not_a_repairable_coverage_gap(idle, code):
    errors = validate_embodied_doc(document(idle=idle), require_duration=True)
    assert any(code in error for error in errors)
    assert any(not error.startswith("IDLE_COVERAGE:") for error in errors)


def test_missing_idle_is_only_repairable_when_explicitly_requested():
    doc = document()
    del doc["idle"]
    assert validate_embodied_doc(doc, require_duration=True, require_description=True) == []
    assert strict(doc) and all(error.startswith("IDLE_COVERAGE:") for error in strict(doc))
    doc["idle"] = compute_idle_complement(doc["working"], doc["duration"])
    assert strict(doc) == []


@pytest.mark.parametrize("sub", [
    {"start": 2, "end": 1, "reason": "idle"},
    {"start": 0, "end": 2, "reason": "idle"},
    {"start": 2, "end": 4, "reason": "wrist_anomaly"},
    {"start": 1.5, "end": 2, "reason": "unrelated"},
    {"start": 1.5, "end": float("nan"), "reason": "idle"},
    None,
])
def test_mid_segments_have_real_parent_bounds_and_known_reason(sub):
    assert any("MID_SEGMENT" in error for error in strict(document(working=[segment(mid_segments=[sub])])))


def test_valid_mid_segments_are_not_outer_idle():
    doc = document(working=[segment(mid_segments=[{"start": 1.1, "end": 2, "reason": "idle"}])])
    assert strict(doc) == []


@pytest.mark.parametrize("kind,evidence", [
    ("遗漏步骤", "第2步全程未出现"),
    ("顺序错误", "第2步出现在第1步之前"),
    ("重复/多做", "第1步重复做了两遍"),
    ("用错物体", "t=2s 用了改锥应为扳手"),
    ("操作方式错误", "t=2s 拧紧方式不对，规定应为顺时针"),
    ("意外事故", "t=2s 扳手掉落"),
    ("其他", "t=2s 两种错误同时发生，需人工检查"),
])
def test_evidence_is_validated_by_error_class_not_one_universal_seconds_rule(kind, evidence):
    doc = document(working=[segment(verdict="错", error_type=kind, evidence=evidence)])
    assert strict(doc) == []


@pytest.mark.parametrize("kind,evidence", [
    ("遗漏步骤", "x"), ("遗漏步骤", "t=-1 第2步全程未出现"),
    ("顺序错误", "第1步出错"), ("重复/多做", "重复"),
    ("用错物体", "用了改锥应为扳手"), ("用错物体", "t=99s 用了改锥应为扳手"),
    ("用错物体", "t=2x 用了改锥应为扳手"),
    ("操作方式错误", "t=2s x"), ("意外事故", "t=2s 看不清楚"),
    ("其他", "t=2s x"),
])
def test_placeholder_or_invalid_evidence_cannot_pass(kind, evidence):
    assert any("evidence" in error for error in strict(document(working=[
        segment(verdict="错", error_type=kind, evidence=evidence)])))


def test_partial_clip_preserves_matching_work_but_requires_adjudication():
    doc = document(clip_status="描述不符-仅部分工作", report="Quick ID: C001 只完成了安装的一部分")
    errors = strict(doc)
    assert len(errors) == 1 and errors[0].startswith("REVIEW_REQUIRED:")
    assert doc["working"] == [segment()]


def test_excluded_clip_cannot_contain_idle_or_work():
    doc = document(clip_status="佩戴问题", report="Quick ID: C001 全程黑屏")
    errors = strict(doc)
    assert any("working segments array is not empty" in error for error in errors)
    assert any("IDLE_CONFLICT" in error for error in errors)
    assert strict(document(clip_status="佩戴问题", working=[], idle=[], report="Quick ID: C001 全程黑屏")) == []


@pytest.mark.parametrize("input_kind", ["array", "jsonl", "directory"])
def test_export_empty_inputs_fail(tmp_path, input_kind):
    if input_kind == "directory":
        target = tmp_path
    else:
        target = tmp_path / ("empty.json" if input_kind == "array" else "empty.jsonl")
        target.write_text("[]" if input_kind == "array" else "\n\n", encoding="utf-8")
    result = validate_embodied_export(str(target))
    assert not result["passed"] and result["records_checked"] == 0
    assert any("EMPTY_BATCH" in error for error in result["violations"])


def test_export_io_errors_are_reported_instead_of_passing(tmp_path):
    result = validate_embodied_export(str(tmp_path / "missing.json"))
    assert not result["passed"] and any("READ_ERROR" in error for error in result["violations"])


@pytest.mark.parametrize("extra", [', "duration": -1, "duration": 10', ', "metadata": 1e999'])
def test_legacy_export_uses_the_strict_json_decoder(tmp_path, extra):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(document())[:-1] + extra + "}", encoding="utf-8")
    result = validate_embodied_export(str(source))
    assert not result["passed"] and result["violations"]
    assert "Not a production admission gate" in result["scope"]


@pytest.mark.parametrize("value", [True, False, 0, -1, "1", None, 4])
def test_steps_reject_bool_noninteger_and_extra_beyond_independent_n(value):
    result = diff_expected_steps([segment(step=value)], 3)
    assert result["invalid_steps"] and not result["is_perfect_sequence"]


def test_unknown_n_is_not_inferred_as_workflow_completeness():
    result = diff_expected_steps([{"step_number": 1}, {"step_number": 999999999}])
    assert result["observed_sequence_valid"]
    assert result["coverage_known"] is False and result["is_perfect_sequence"] is False
    assert result["missing"] == []


def test_chronological_order_overrides_number_sorted_json():
    result = diff_expected_steps([segment(4, 5, 1), segment(1, 2, 2)], 2)
    assert result["sequence"] == [2, 1]
    assert result["out_of_order"][0]["index"] == 0
    assert result["out_of_order"][0]["previous_index"] == 1
    assert result["temporal_errors"] and not result["is_perfect_sequence"]


def test_order_indices_refer_to_original_records_despite_invalid_steps():
    result = diff_expected_steps([{"step_number": None}, {"step_number": 3},
                                  {"step_number": 1}, {"step_number": 2}], 3)
    assert result["out_of_order"][0]["index"] == 2


def test_custom_procedure_order_and_transition_are_enforced():
    spec = ProcedureSpec(steps=[20, 10], allowed_transitions={20: [10]})
    assert diff_expected_steps([{"step_number": 20}, {"step_number": 10}], procedure=spec)["is_perfect_sequence"]
    assert not diff_expected_steps([{"step_number": 10}, {"step_number": 20}], procedure=spec)["is_perfect_sequence"]


@pytest.mark.parametrize("config", [0, -1, True, 1.5])
def test_expected_count_is_not_coerced(config):
    with pytest.raises(ValueError, match="positive integer"):
        diff_expected_steps([], config)


def test_expected_count_resource_limit_fails_explicitly_instead_of_allocating():
    with pytest.raises(ValueError, match="STEP_LIMIT"):
        diff_expected_steps([], 999999999)


@pytest.mark.parametrize("kwargs", [
    {"steps": [True]}, {"steps": [1, 1]}, {"steps": [1], "required_steps": [2]},
    {"steps": [1], "allowed_transitions": {1: [2]}},
    {"steps": [1], "allowed_transitions": []}, {"steps": 1},
])
def test_invalid_procedure_configs_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        ProcedureSpec(**kwargs)


def test_complete_labeled_temporal_agreement_can_pass_format_gate():
    result = calculate_temporal_kappa([document()], [document()])
    assert result["kappa"] == 1 and result["gate_eligible"] and result["meets_trial_bar"]
    assert result["coverage_complete"] and result["semantic_complete"]
    assert result["validation_errors"] == [] and result["semantic_conflicts"] == []
    assert result["total_samples"] == 20  # Includes seven seconds of trailing idle.
    assert "video truth" in result["review_scope"]


def test_missing_and_duplicate_clips_block_high_common_subset_kappa():
    a = [document(), document(clip="other.mp4")]
    result = calculate_temporal_kappa(a, [document()])
    assert result["kappa"] == 1 and result["missing_clips_b"] == ["other.mp4"]
    assert not result["coverage_complete"] and not result["meets_trial_bar"]
    result = calculate_temporal_kappa([document(), document()], [document()])
    assert result["duplicate_clips_a"] == ["clip.mp4"]
    assert not result["meets_trial_bar"]


def test_same_times_different_verdict_cannot_pass():
    wrong = document(working=[segment(verdict="错", error_type="用错物体", evidence="t=2s 用了改锥应为扳手")])
    result = calculate_temporal_kappa([document()], [wrong])
    assert result["kappa"] == 1 and not result["meets_trial_bar"]
    assert [finding["type"] for finding in result["semantic_conflicts"]] == ["verdict_mismatch"]


def test_same_wrong_verdict_different_error_class_is_a_conflict():
    wrong_a = document(working=[segment(verdict="错", error_type="用错物体", evidence="t=2s 用了改锥应为扳手")])
    wrong_b = document(working=[segment(verdict="错", error_type="意外事故", evidence="t=2s 扳手掉落")])
    result = calculate_temporal_kappa([wrong_a], [wrong_b])
    assert result["error_type_agreement_rate"] == 0
    assert result["semantic_conflicts"][0]["type"] == "error_type_mismatch"
    assert not result["meets_trial_bar"]


def test_unmatched_segments_cannot_hide_behind_high_temporal_score():
    a = document(working=[segment(1, 3), segment(5, 5.001, 2)], idle=[])
    result = calculate_temporal_kappa([a], [document()])
    assert result["kappa"] == 1 and result["unmatched_segments"]
    assert not result["meets_trial_bar"]


@pytest.mark.parametrize("status", ["all_working", "all_idle"])
def test_single_class_kappa_is_undefined_not_one(status):
    doc = document(working=[segment(0, 10)], idle=[]) if status == "all_working" else document(
        clip_status="这里没有", working=[], idle=[])
    result = calculate_temporal_kappa([doc], [doc])
    assert result["observed_agreement_p0"] == 1
    assert result["kappa"] is None and result["kappa_status"] == "UNDEFINED_SINGLE_CLASS"
    assert result["is_degenerate_single_class"] and not result["meets_trial_bar"]
    json.dumps(result, allow_nan=False)


def test_mismatched_duration_and_missing_semantics_remain_diagnostics_only():
    result = calculate_temporal_kappa([document()], [document(duration=11)])
    assert any("DURATION_MISMATCH" in error for error in result["validation_errors"])
    assert not result["meets_trial_bar"]
    result = calculate_temporal_kappa({"clip": {"working": [{"start": 1, "end": 3}]}},
                                      {"clip": {"working": [{"start": 1, "end": 3}]}})
    assert result["kappa"] == 1 and result["temporal_meets_trial_bar"]
    assert not result["semantic_complete"] and not result["meets_trial_bar"]


def test_sampling_is_explicit_half_open_and_retains_last_partial_bin():
    assert build_temporal_mask([segment(0, .5)], 1.1, 2) == [True, False, False]
    assert calculate_segment_tiou(segment(0, 1), segment(2, 3)) == 0


@pytest.mark.parametrize("records", [[], {}, [None], [{"clip": "x", "working": []}], "bad"])
def test_empty_or_malformed_agreement_never_claims_acceptance(records):
    result = calculate_temporal_kappa(records, records)
    assert not result["meets_trial_bar"] and result["validation_errors"]


def test_unhashable_semantic_values_fail_without_crashing():
    doc = document(working=[segment(verdict=[])])
    result = calculate_temporal_kappa([doc], [doc])
    assert not result["meets_trial_bar"] and not result["semantic_complete"]


def test_mask_allocation_limit_is_explicit():
    with pytest.raises(ValueError, match="SAMPLE_LIMIT"):
        build_temporal_mask([], 10000000, 2)


def test_large_boundary_disagreement_is_not_hidden_by_high_kappa():
    a = document(duration=100, working=[segment(10, 80)],
                 idle=[{"start": 0, "end": 10}, {"start": 80, "end": 100}])
    b = document(duration=100, working=[segment(12, 78)],
                 idle=[{"start": 0, "end": 12}, {"start": 78, "end": 100}])
    result = calculate_temporal_kappa([a], [b])
    assert result["kappa"] > .8 and result["temporal_meets_trial_bar"]
    assert {finding["type"] for finding in result["boundary_findings"]} == {
        "boundary_offset_review", "total_boundary_offset_review"}
    assert not result["gate_eligible"] and not result["meets_trial_bar"]


def test_equal_outer_segments_do_not_hide_different_mid_reasons():
    a = document(working=[segment(mid_segments=[{"start": 1.5, "end": 2, "reason": "idle"}])])
    b = copy.deepcopy(a)
    b["working"][0]["mid_segments"][0]["reason"] = "wrist_anomaly"
    result = calculate_temporal_kappa([a], [b])
    assert result["kappa"] == 1
    assert result["semantic_conflicts"][0]["type"] == "mid_segments_mismatch"
    assert not result["meets_trial_bar"]


def test_correct_labels_use_valid_correction_events_instead_of_error_fields():
    doc = document(working=[segment(correction_events=["t=00:02 曾用错工具，已自行纠正"])])
    assert strict(doc) == []
    doc["working"][0]["evidence"] = "t=2s 用了错物"
    assert any("must not have error evidence" in error for error in strict(doc))
    doc["working"][0].pop("evidence")
    doc["working"][0]["correction_events"] = ["t=99s 已纠正"]
    assert any("CORRECTION" in error for error in strict(doc))


def test_reports_need_a_locator_and_an_actual_explanation():
    for report in ("黑屏", "Quick ID: C001"):
        doc = document(clip_status="佩戴问题", working=[], idle=[], report=report)
        assert any("REPORT" in error for error in strict(doc))
