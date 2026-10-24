import copy
import json

from trace_annotate.embodied.quality_pipeline import evaluate_quality, render_quality_html
from trace_annotate.embodied.quality_report import quality_statistics, summary_markup


def test_report_statistics_come_from_actual_input_and_do_not_double_count_comparison(tmp_path):
    doc = {"clip": "synthetic.mp4", "duration": 12, "text_description": "Synthetic format test",
           "clip_status": "正常", "expected_step_count": 2,
           "working": [{"start": 1, "end": 3, "step_number": 1, "verdict": "对"},
                       {"start": 5, "end": 9, "step_number": 2, "verdict": "对"}],
           "idle": [{"start": 0, "end": 1}, {"start": 3, "end": 5}, {"start": 9, "end": 12}]}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(doc), encoding="utf-8")
    b.write_text(json.dumps(doc), encoding="utf-8")
    report = evaluate_quality(str(a), str(b))
    stats = quality_statistics(report)
    assert stats["videos"] == 1
    assert stats["working_seconds"] == stats["idle_seconds"] == 6
    assert stats["structural_errors"] == stats["review_items"] == 0
    html = render_quality_html(report)
    assert 'name="color-scheme" content="dark"' in html
    assert "6.0 <small>s</small>" in html and "Working 50.0%, Idle 50.0%" in html
    assert 'scope="col"' in html and "没有观看视频" in html


def test_invalid_or_missing_records_never_display_a_clean_zero_time_dashboard(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"clip":"bad","duration":10,"working":[{"start":-1,"end":2}]}', encoding="utf-8")
    report = evaluate_quality(str(source))
    stats = quality_statistics(report)
    assert not stats["time_totals_complete"]
    assert stats["working_seconds"] is None and stats["idle_seconds"] is None
    assert stats["structural_errors"] > 0
    html = render_quality_html(report)
    assert "时间汇总未生成" in html and 'role="img"' not in html
    source.write_text("broken", encoding="utf-8")
    missing = quality_statistics(evaluate_quality(str(source)))
    assert missing["videos"] == 0 and missing["structural_errors"] > 0


def test_pending_review_remains_visible_even_when_time_totals_and_structure_are_valid(tmp_path):
    source = tmp_path / "review.json"
    source.write_text(json.dumps({"clip":"review","duration":10,"text_description":"task","clip_status":"正常",
        "expected_step_count":2,"working":[{"start":2,"end":4,"step_number":1,"verdict":"对"}],
        "idle":[{"start":0,"end":2},{"start":4,"end":10}]}), encoding="utf-8")
    report = evaluate_quality(str(source))
    before = copy.deepcopy(report)
    stats = quality_statistics(report)
    assert stats["structural_errors"] == 0 and stats["review_items"] == 1
    assert stats["time_totals_complete"]
    html = render_quality_html(report)
    assert "status-review" in html and "待复核项目 1" in html
    assert report == before


def test_batch_total_overflow_is_unknown_and_large_finite_ratios_stay_valid():
    record = {"structure": {"passed": True}, "errors": [], "doc": {
        "working": [{"start": 0, "end": 1e308}], "idle": [{"start": 0, "end": 1e308}]}}
    report = {"status": "PASS_SCOPED", "results": [record]}
    html = summary_markup(report)
    assert "Working 50.0%, Idle 50.0%" in html
    assert "nan" not in html.lower() and "inf" not in html.lower()
    report["results"].append(copy.deepcopy(record))
    stats = quality_statistics(report)
    assert stats["working_seconds"] is None and not stats["time_totals_complete"]
    assert 'role="img"' not in summary_markup(report)


def test_comparison_failure_reasons_and_sampling_are_visible_without_expanding_json(tmp_path):
    doc = {"clip": "sample.mp4", "duration": 10, "text_description": "Synthetic comparison",
           "clip_status": "正常", "expected_step_count": 1,
           "working": [{"start": 2, "end": 6, "step_number": 1, "verdict": "对"}],
           "idle": [{"start": 0, "end": 2}, {"start": 6, "end": 10}]}
    other = copy.deepcopy(doc)
    other["working"][0].update(start=0, end=3)
    other["idle"] = [{"start": 3, "end": 10}]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(doc), encoding="utf-8")
    b.write_text(json.dumps(other), encoding="utf-8")
    report = evaluate_quality(str(a), str(b), fps=4)
    assert report["status"] == "REVIEW_REQUIRED"
    assert all(item["structure"]["passed"] for item in report["results"] + report["comparison_results"])
    visible_top = render_quality_html(report).split("<details>", 1)[0]
    assert "对照比较门禁" in visible_top and "低于要求 0.8" in visible_top
    assert "时间边界差异" in visible_top and "4.0 Hz" in visible_top
    other["clip"] = '<script>alert("x")</script>'
    b.write_text(json.dumps(other), encoding="utf-8")
    missing_html = render_quality_html(evaluate_quality(str(a), str(b))).split("<details>", 1)[0]
    assert "比较记录覆盖不完整" in missing_html and "&lt;script&gt;" in missing_html
    assert '<script>alert' not in missing_html


def test_optional_segment_labels_are_checked_and_compared_without_hiding_temporal_agreement(tmp_path):
    doc = {"clip": "sample.mp4", "duration": 10, "text_description": "Synthetic comparison",
           "clip_status": "正常", "expected_step_count": 1,
           "working": [{"start": 2, "end": 6, "step_number": 1, "verdict": "对", "label": "position"}],
           "idle": [{"start": 0, "end": 2}, {"start": 6, "end": 10}]}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(doc), encoding="utf-8")
    doc["working"][0]["label"] = "fasten"
    b.write_text(json.dumps(doc), encoding="utf-8")
    report = evaluate_quality(str(a), str(b))
    assert report["agreement"]["kappa"] == 1 and not report["agreement"]["gate_passed"]
    assert report["agreement"]["semantic_conflicts"][0]["type"] == "label_mismatch"
    visible_top = render_quality_html(report).split("<details>", 1)[0]
    assert "标签冲突" in visible_top and "position" in visible_top and "fasten" in visible_top
    doc["working"][0]["label"] = 42
    b.write_text(json.dumps(doc), encoding="utf-8")
    assert any("optional label must be a string" in error for error in evaluate_quality(str(b))["results"][0]["errors"])
