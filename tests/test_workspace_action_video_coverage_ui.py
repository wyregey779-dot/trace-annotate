"""Keep optional action-window coverage visible without implying reviewed action facts."""

import os
from pathlib import Path

import pytest


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def test_action_video_v2_coverage_is_partial_and_legacy_ui_is_unchanged():
    playwright = pytest.importorskip("playwright.sync_api")
    source = UI.read_text(encoding="utf-8")
    script = source[source.index("function renderVisionCoverage(){"):
                    source.index("function renderVisionRetry(){")]
    bootstrap = """
const $=id=>document.getElementById(id);
const state={visionJob:{id:'job-1',state:'succeeded',requested_window:{start:{n:'0',d:'1'},end:{n:'6',d:'1'}}},visionResult:null};
const showTime=value=>String(Number(value.n)/Number(value.d));
""" + script + "\nwindow.fixture={state,renderVisionCoverage,renderActionVideoCoverage};"
    mixed = {
        "contract": "videomae-window-sampling-v2", "status": "failed",
        "reason": "<img src=x onerror=alert(1)>", "eligible_scenes": 1,
        "processed_windows": 1,
        "windows": [
            {"scene_index": 0, "status": "success", "reason": None,
             "result": {"status": "success"},
             "input": {"window_pts": {"start": {"n": "0", "d": "1"}, "end": {"n": "2", "d": "1"}}}},
            {"scene_index": 0, "status": "not_run", "reason": "video_action_phase_budget_exhausted",
             "result": None,
             "input": {"window_pts": {"start": {"n": "3", "d": "1"}, "end": {"n": "4", "d": "1"}}}},
        ],
        "coverage": [
            {"scene_index": 0, "scene_window": {"start": {"n": "0", "d": "1"}, "end": {"n": "4", "d": "1"}},
             "decoded_samples": 24, "selected_windows": 2, "processed_windows": 1,
             "successful_windows": 1,
             "uncovered_intervals": [{"window_pts": {"start": {"n": "2", "d": "1"}, "end": {"n": "4", "d": "1"}}}],
             "omitted_window_envelopes": [{"window_pts": {"start": {"n": "2", "d": "1"}, "end": {"n": "3", "d": "1"}}, "reason": "decoded_sample_gap_exceeds_limit"}]},
            {"scene_index": 1, "scene_window": {"start": {"n": "4", "d": "1"}, "end": {"n": "6", "d": "1"}},
             "decoded_samples": 3, "selected_windows": 0, "processed_windows": 0,
             "successful_windows": 0,
             "uncovered_intervals": [{"window_pts": {"start": {"n": "4", "d": "1"}, "end": {"n": "6", "d": "1"}}}],
             "omitted_window_envelopes": []},
        ],
    }
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content("<!doctype html><meta charset='utf-8'><div id='vision-coverage'></div><section id='vision-evidence'><p id='vision-evidence-scope'></p></section>")
            page.add_script_tag(content=bootstrap)
            page.evaluate("fixture.state.visionResult={keyframes:[],module_status:{},sampling:{},action_video_sampling:{contract:'videomae-window-sampling-v1'}};fixture.renderVisionCoverage()")
            assert page.locator("#vision-action-coverage").count() == 0
            legacy_text = page.locator("#vision-coverage").inner_text()
            page.evaluate("sampling=>{fixture.state.visionResult.action_video_sampling=sampling;fixture.renderVisionCoverage()}", mixed)
            summary = page.locator("#vision-action-coverage p").first.inner_text()
            assert "局部覆盖" in summary
            assert "成功 1 / 选中 2 窗口" in summary
            assert "已处理 1 窗口" in summary
            assert "仍未审核" in page.locator("#vision-action-coverage").inner_text()
            assert page.locator("#vision-action-coverage img").count() == 0
            page.locator("#vision-action-coverage summary").click()
            assert page.locator("#vision-action-coverage li").count() == 2
            assert "相邻解码样本间隔过大" in page.locator("#vision-action-coverage li").first.inner_text()
            assert "分析时间预算用尽" in page.locator("#vision-action-coverage li").first.inner_text()
            assert "decoded_sample_gap_exceeds_limit" in page.locator("#vision-action-coverage li").first.get_attribute("title")
            assert "未形成可选窗口" in page.locator("#vision-action-coverage li").last.inner_text()
            page.evaluate("fixture.renderVisionCoverage()")
            assert page.locator("#vision-action-coverage details").evaluate("node=>node.open") is True
            page.evaluate("sampling=>fixture.renderActionVideoCoverage(sampling,{historicalJobId:'job-saved'})", mixed)
            assert "来源作业 job-saved" in page.locator("#vision-action-coverage-saved").inner_text()
            assert "成功 1 / 选中 2 窗口" in page.locator("#vision-action-coverage").inner_text()
            page.evaluate("document.getElementById('vision-action-coverage-saved').remove()")
            assert page.locator("#vision-action-coverage").count() == 1
            v3 = dict(mixed, contract="videomae-window-sampling-v3", status="not_run",
                      reason="video_action_weights_unavailable", processed_windows=0,
                      windows=[], decode={"status": "not_run", "reason": "video_action_weights_unavailable",
                                          "planned_indices": [], "samples": []})
            page.evaluate("sampling=>{fixture.state.visionResult.action_video_sampling=sampling;fixture.renderVisionCoverage()}", v3)
            v3_text = page.locator("#vision-action-coverage").inner_text()
            assert "精确取帧未运行 · 实得 0 / 计划 0 帧" in v3_text
            assert "本地动作模型权重不可用" in v3_text
            assert "仍未审核" in v3_text
            assert page.locator("#vision-action-coverage img").count() == 0
            evidence_dir = os.getenv("TRACE_ANNOTATE_UI_EVIDENCE_DIR")
            if evidence_dir:
                target = Path(evidence_dir)
                target.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(target / "action-v2-mixed-coverage.png"))
            page.evaluate("fixture.state.visionResult.action_video_sampling={contract:'videomae-window-sampling-v1'};fixture.renderVisionCoverage()")
            assert page.locator("#vision-action-coverage").count() == 0
            assert page.locator("#vision-coverage").inner_text() == legacy_text
            assert not errors, errors
        finally:
            browser.close()
