"""The browser shows bounded geometry only after its job and frame identities match."""
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import os
from pathlib import Path
import re

import pytest
from PIL import Image


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def test_read_only_hand_object_panel_identity_limits_and_stale_response():
    playwright = pytest.importorskip("playwright.sync_api")
    source = UI.read_text(encoding="utf-8")
    panel = re.search(r'<section id="vision-evidence".*?</section>', source, re.DOTALL).group()
    script = source[source.index("const VISION_EVIDENCE_PAGE_SIZE="):source.index("function bind(){")]
    png = BytesIO()
    Image.new("RGB", (100, 100), (20, 30, 40)).save(png, format="PNG")
    encoded = b64encode(png.getvalue()).decode("ascii")
    bootstrap = f"""
const $=id=>document.getElementById(id);
const state={{task:null,media:null,visionJob:null,visionResult:null,visionEvidencePage:0,visionEvidenceToken:0,visionEvidenceBlob:null,visionHandObjectToken:0,visionHandObjectKey:null,visionHandObject:null,visionEvidenceDisplayedFrame:null}};
const showTime=r=>String(Number(r.n)/Number(r.d));
const pixels=Uint8Array.from(atob('{encoded}'),char=>char.charCodeAt(0));
async function verifiedFramePng(index){{
 const url=URL.createObjectURL(new Blob([pixels],{{type:'image/png'}}));
 return {{row:{{frame_index:String(index),source_pts:String(index),proxy_pts:String(index),seconds:{{n:String(index),d:'1'}}}},url,image_sha256:'preview-hash'}};
}}
async function json(path){{
 if(!path.endsWith('/hand-object-geometry'))throw new Error('unexpected request: '+path);
 return window.geometryReply;
}}
{script}
window.fixture={{state,renderVisionEvidence,clearVisionEvidence}};
"""
    frame = {
        "scene_index": 0, "frame_index": "0", "source_pts": "0", "proxy_pts": "0",
        "timestamp": {"n": "0", "d": "1"}, "evidence_sha256": "e" * 64,
        "object": {"status": "success", "objects_detected": []},
        "hands": {"status": "success", "hands": [],
                  "frame_identity": {"shape": [100, 100, 3], "frame_index": "0",
                                     "evidence_sha256": "e" * 64}},
    }
    later = {**frame, "frame_index": "1", "source_pts": "1", "proxy_pts": "1",
             "timestamp": {"n": "1", "d": "1"}, "evidence_sha256": "f" * 64,
             "hands": {**frame["hands"],
                       "frame_identity": {"shape": [100, 100, 3], "frame_index": "1",
                                          "evidence_sha256": "f" * 64}}}
    result = {
        "job_id": "job-1", "task_id": "task-1", "media_id": "media-1",
        "input_revision": 0, "input_sha256": "a" * 64,
        "file_sha256": "b" * 64, "scope": "sampled_frames_only",
        "sampling": {"planned": 2, "returned": 2, "truncated": False,
                     "frame_map_sha256": "map-1"},
        "keyframes": [frame, later],
    }
    payload = {
        "kind": "traceannotate.hand-object-geometry", "version": 1,
        "source": {"job_id": "job-1", "task_id": "task-1", "input_revision": 0,
                   "vision_result_sha256": "c" * 64, "playback_sha256": "b" * 64},
        "geometry": {"status": "success", "reason": "geometry_only",
                     "pairs": [{"id": "<img onerror=alert(1)>", "scene_index": 0,
                                "frame_index": "0", "pts": {"n": "0", "d": "1"},
                                "evidence_sha256": "e" * 64,
                                "hand_observation_id": "hand-1",
                                "object_observation_id": "object-1",
                                "landmark_to_object_box_pixels": 4.5,
                                "hand_box_to_object_box_pixels": 2.0,
                                "box_iou": 0.0}],
                     "coverage": {"total_keyframes": 1,
                                  "hand_successful_keyframes": 1,
                                  "object_successful_keyframes": 1,
                                  "omitted_pairs": 0,
                                  "object_detection_omissions": 0,
                                  "truncated": False}},
    }
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content("<!doctype html><meta charset='utf-8'>" + panel)
            page.add_script_tag(content=bootstrap)
            page.evaluate("({result,payload})=>{const s=fixture.state;s.task={id:'task-1',annotations:[]};s.media={id:'media-1',playback:{sha256:'b'.repeat(64)},frame_map_sha256:'map-1',display_dimensions:{playback_width:100,playback_height:100}};s.visionJob={id:'job-1',input_sha256:'a'.repeat(64),result_sha256:'c'.repeat(64)};s.visionResult=result;window.geometryReply=payload;fixture.renderVisionEvidence();}",
                          {"result": result, "payload": payload})
            playwright.expect(page.locator("#vision-hand-object-status")).to_contain_text("本帧有 1 条")
            assert page.locator("#vision-hand-object-pairs li").count() == 1
            assert page.locator("#vision-hand-object-pairs img").count() == 0
            assert "4.5 px" in page.locator("#vision-hand-object-pairs").inner_text()
            assert "触碰、持有或动作" in page.locator("#vision-hand-object-status").inner_text()
            page.evaluate("()=>{fixture.state.visionJob.result_sha256=null;fixture.renderVisionEvidence();}")
            playwright.expect(page.locator("#vision-hand-object-status")).to_contain_text("已暂停显示同帧几何")
            assert page.locator("#vision-hand-object-pairs li").count() == 0
            assert page.locator("#vision-hand-object-coverage").inner_text() == ""
            page.evaluate("()=>{fixture.state.visionJob.result_sha256='c'.repeat(64);fixture.renderVisionEvidence();}")
            playwright.expect(page.locator("#vision-hand-object-status")).to_contain_text("本帧有 1 条")
            page.evaluate("payload=>{window.geometryReply=payload;fixture.state.visionHandObjectKey=null;fixture.renderVisionEvidence();}",
                          {**payload, "source": {**payload["source"], "vision_result_sha256": "d" * 64}})
            playwright.expect(page.locator("#vision-hand-object-status")).to_contain_text("身份不一致")
            assert page.locator("#vision-hand-object-pairs li").count() == 0
            page.evaluate("payload=>{window.geometryReply=payload;fixture.state.visionHandObjectKey=null;fixture.renderVisionEvidence();}",
                          {**payload, "geometry": {**payload["geometry"], "status": "not_run",
                                                     "reason": "hand_landmarks_unavailable", "pairs": []}})
            playwright.expect(page.locator("#vision-hand-object-status")).to_contain_text("未运行同帧几何")
            assert page.locator("#vision-hand-object-pairs li").count() == 0
            limited = {**payload, "geometry": {**payload["geometry"], "pairs": [],
                                                "coverage": {**payload["geometry"]["coverage"],
                                                             "omitted_pairs": 2,
                                                             "truncated": True,
                                                             "object_truncated_keyframes": 1,
                                                             "object_detection_omissions_unknown_keyframes": 2,
                                                             "sampled_module_coverage_complete": False}}}
            page.evaluate("payload=>{window.geometryReply=payload;fixture.state.visionHandObjectKey=null;fixture.renderVisionEvidence();}", limited)
            page.locator("#vision-evidence-frame").select_option("1")
            playwright.expect(page.locator("#vision-hand-object-status")).to_contain_text("可能有被上限省略的观测")
            playwright.expect(page.locator("#vision-hand-object-coverage")).to_contain_text("2 帧遗漏数量未知")
            assert "物体结果截断 1 帧" in page.locator("#vision-hand-object-coverage").inner_text()
            race = page.evaluate("async payload=>{const s=fixture.state;window.geometryReply=new Promise(resolve=>window.releaseGeometry=resolve);s.visionHandObjectKey=null;fixture.renderVisionEvidence();s.task={id:'task-2',annotations:[]};fixture.clearVisionEvidence();window.releaseGeometry(payload);await Promise.resolve();await new Promise(resolve=>setTimeout(resolve,0));return {panelHidden:document.getElementById('vision-evidence').hidden,pairs:document.querySelectorAll('#vision-hand-object-pairs li').length,geometry:s.visionHandObject};}", payload)
            assert race == {"panelHidden": True, "pairs": 0, "geometry": None}
            assert not errors, errors
        finally:
            browser.close()
