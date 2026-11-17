"""Exercise the shipped evidence viewer in Chromium with a decoded PNG.

The fixture supplies already verified frame rows; the test checks the viewer's
candidate identity fence, bounded overlay and honest loss display. It does not
claim model accuracy or human review.
"""
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
import os
from pathlib import Path
import re

import pytest
from PIL import Image


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def test_visual_evidence_viewer_bounds_and_rejects_mismatched_pts():
    playwright = pytest.importorskip("playwright.sync_api", reason="Chromium is needed for image and SVG evidence")
    source = UI.read_text(encoding="utf-8")
    panel = re.search(r'<section id="vision-evidence".*?</section>', source, re.DOTALL).group()
    script = source[source.index("const VISION_EVIDENCE_PAGE_SIZE="):source.index("function bind(){")]
    candidate_script = source[source.index("function renderVisionCandidates("):source.index("async function refreshVisionStatus(")]
    png = BytesIO()
    Image.new("RGB", (100, 100), (20, 30, 40)).save(png, format="PNG")
    encoded = b64encode(png.getvalue()).decode("ascii")
    bootstrap = f"""
const $=id=>document.getElementById(id);
const state={{task:null,media:null,visionJob:null,visionResult:null,visionEvidencePage:0,visionEvidenceToken:0,visionEvidenceBlob:null}};
const showTime=r=>String(Number(r.n)/Number(r.d));
async function selectAnnotation(id){{state.selected=id;}}
async function json(path){{throw new Error('No geometry fixture for '+path);}}
const pixels=Uint8Array.from(atob('{encoded}'),char=>char.charCodeAt(0));
async function verifiedFramePng(index){{
 const url=URL.createObjectURL(new Blob([pixels],{{type:'image/png'}}));
 return {{row:{{frame_index:String(index),source_pts:String(index),proxy_pts:String(index),seconds:{{n:String(index),d:'1'}}}},url,image_sha256:'preview-png-hash'}};
}}
{script}
{candidate_script}
window.fixture={{state,renderVisionEvidence,loadVisionEvidenceFrame,clearVisionEvidence,openVisionEvidenceForCandidate,renderVisionCandidates}};
"""
    frames = []
    for index in range(13):
        frames.append({
            "frame_index": str(index), "source_pts": str(index), "proxy_pts": str(index),
            "timestamp": {"n": str(index), "d": "1"}, "evidence_sha256": "e" * 64,
            "object": {"status": "success", "reason": None, "objects_detected": [
                {"class_label": "<img onerror=alert(1)>", "box": [1, 1, 20, 20],
                 "score": 0.5} for _ in range(30)]},
            "hands": {"status": "success", "reason": None,
                      "frame_identity": {"shape": [100, 100, 3], "frame_index": str(index),
                                         "evidence_sha256": "e" * 64},
                      "hands": [{"handedness_raw": "Left", "landmarks_pixel": [[2, 3, 0]] * 21}
                                for _ in range(5)]},
        })
    result = {
        "job_id": "job-1", "task_id": "task-1", "media_id": "media-1",
        "input_revision": 0, "input_sha256": "a" * 64, "file_sha256": "b" * 64,
        "scope": "registered_playback_only; candidate_not_applied",
        "sampling": {"planned": 20, "returned": 13, "truncated": True, "frame_map_sha256": "map-1"},
        "keyframes": frames,
        "action_video_sampling": {"windows": [{"input": {"frames": [{"frame_index": "0"}]},
                                               "status": "success", "result": {"decision": "unknown",
                                                                                 "reason": "outside_supported_mapping"}}]},
    }
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content("<!doctype html><meta charset='utf-8'>" + panel + "<table id='vision-candidates'><tbody></tbody></table>")
            page.add_script_tag(content=bootstrap)
            page.evaluate("result=>{const s=fixture.state;s.task={id:'task-1',annotations:[]};s.media={id:'media-1',playback:{sha256:'b'.repeat(64)},frame_map_sha256:'map-1',display_dimensions:{playback_width:100,playback_height:100}};s.visionJob={id:'job-1',input_sha256:'a'.repeat(64),result_sha256:'c'.repeat(64)};s.visionResult=result;fixture.renderVisionEvidence();}", result)
            playwright.expect(page.locator("#vision-evidence-image")).to_be_visible()
            assert page.locator("#vision-evidence-frame option").count() == 12
            assert page.locator("#vision-evidence-overlay rect").count() == 24
            assert page.locator("#vision-evidence-overlay circle").count() == 84
            assert page.locator("#vision-evidence-notes img").count() == 0
            assert "未采样后段" in page.locator("#vision-evidence-loss").inner_text()
            assert "outside_supported_mapping" in page.locator("#vision-evidence-notes").inner_text()
            assert "preview-png-hash" in page.locator("#vision-evidence-identity").inner_text()
            assert "c" * 64 in page.locator("#vision-evidence-identity").inner_text()
            page.evaluate("()=>{fixture.state.visionEvidencePage=1;fixture.renderVisionEvidence();}")
            playwright.expect(page.locator("#vision-evidence-frame option")).to_have_count(1)
            playwright.expect(page.locator("#vision-evidence-image")).to_be_visible()
            page.evaluate("()=>{fixture.state.visionResult.keyframes[12].source_pts='wrong';fixture.loadVisionEvidenceFrame();}")
            playwright.expect(page.locator("#vision-evidence-status")).to_contain_text("PTS 不一致")
            playwright.expect(page.locator("#vision-evidence-image")).to_be_hidden()
            annotation = {"id": "ann-1", "label": "saved candidate", "type": "whole_clip",
                          "review_status": "unreviewed",
                          "provenance": {"origin": "model_candidate", "job_id": "job-1"}}
            page.evaluate("({annotation,result})=>{const s=fixture.state;fixture.clearVisionEvidence();s.visionJob=null;s.visionResult=null;s.task.annotations=[annotation];s.selected=null;window.json=async path=>path.endsWith('/vision-status')?{job_id:'job-1',task_id:'task-1',state:'succeeded',has_candidate:true,input_revision:0,input_sha256:'a'.repeat(64),result_sha256:'d'.repeat(64)}:result;fixture.renderVisionCandidates([annotation]);}", {"annotation": annotation, "result": result})
            page.locator("#vision-candidates .segment-select").click()
            playwright.expect(page.locator("#vision-evidence-image")).to_be_visible()
            assert page.evaluate("fixture.state.visionEvidenceReviewJob?.id") == "job-1"
            assert "d" * 64 in page.locator("#vision-evidence-identity").inner_text()
            race = page.evaluate("async()=>{const s=fixture.state;let release;window.json=()=>new Promise(resolve=>{release=resolve});const ann=s.task.annotations[0];const pending=fixture.openVisionEvidenceForCandidate(ann);s.task={id:'task-2',annotations:[]};fixture.clearVisionEvidence();release({job_id:'job-1',task_id:'task-1',state:'succeeded',has_candidate:true});await pending;return {review:s.visionEvidenceReviewJob,panelHidden:document.getElementById('vision-evidence').hidden};}")
            assert race == {"review": None, "panelHidden": True}
            assert not errors, errors
        finally:
            browser.close()
