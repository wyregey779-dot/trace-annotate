"""Exercise the shipped candidate-decision controls in a real browser."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def test_candidate_decisions_require_actor_evidence_and_explicit_command():
    playwright = pytest.importorskip("playwright.sync_api")
    source = UI.read_text(encoding="utf-8")
    script = source[source.index("function renderVisionCandidates("):source.index("function resetVision()")]
    browser_html = """<!doctype html><meta charset='utf-8'>
      <select id='vision-actor-type'><option value=''></option><option value='assistant'>助手</option></select>
      <input id='vision-actor-id'><textarea id='vision-diff-summary'></textarea>
      <table id='vision-candidates'><tbody></tbody></table>"""
    bootstrap = """
      const $=id=>document.getElementById(id);
      const state={task:null,selected:null,dirty:false,busy:false,pending:null,conflict:false,
        loadingTask:false,visionCandidateDigests:{},visionGeneration:0,
        visionEvidenceReviewAnnotation:null,visionEvidenceReviewJob:null};
      const calls=[];const notices=[];
      const clone=value=>structuredClone(value);
      const showTime=value=>String(Number(value.n)/Number(value.d));
      const notice=(message)=>notices.push(message);
      async function selectAnnotation(id){state.selected=id;}
      async function openVisionEvidenceForCandidate(ann){state.visionEvidenceReviewAnnotation=ann.id;}
      async function confirmAction(){return true;}
      function buildChanges(){return {label:'corrected',review_status:'reviewed'};}
      async function saveCommand(operation,payload){
        calls.push({operation,payload});
        state.task.revision++;
        state.task.content_sha256='f'.repeat(64);
        if(payload.decision==='rejected')state.task.annotations=state.task.annotations.filter(a=>a.id!==payload.annotation_id);
        else state.task.annotations=state.task.annotations.map(a=>a.id===payload.annotation_id?
          {...a,label:payload.changes.label||a.label,provenance:{origin:'manual',actor_id:payload.actor_id,job_id:null},review_status:'unreviewed'}:a);
        return {revision:state.task.revision};
      }
      async function json(){return {task_id:state.task.id,revision:state.task.revision,
        content_sha256:state.task.content_sha256,candidates:[],candidate_sha256:{}};}
      function clearVisionEvidence(){state.visionEvidenceReviewAnnotation=null;}
      function renderVisionEvidence(){}
    """ + script + "window.fixture={state,calls,notices,renderVisionCandidates,decideVisionCandidate};"
    ann = {"id": "ann-candidate", "label": "candidate", "type": "segment",
           "start": {"seconds": {"n": "0", "d": "1"}},
           "end": {"seconds": {"n": "1", "d": "1"}},
           "provenance": {"origin": "model_candidate", "job_id": "job-1"},
           "review_status": "unreviewed"}
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content(browser_html)
            page.add_script_tag(content=bootstrap)
            page.evaluate("ann=>{const s=fixture.state;s.task={id:'task-1',revision:1,content_sha256:'a'.repeat(64),annotations:[ann]};s.visionCandidateDigests[ann.id]='b'.repeat(64);fixture.renderVisionCandidates([ann]);}", ann)
            page.get_by_role("button", name="接受").click()
            assert page.evaluate("fixture.calls.length") == 0
            assert "实际操作主体" in page.evaluate("fixture.notices.at(-1)")
            page.locator("#vision-actor-type").select_option("assistant")
            page.locator("#vision-actor-id").fill("codex-assistant")
            page.get_by_role("button", name="接受").click()
            assert page.evaluate("fixture.calls.length") == 0
            assert "逐帧证据" in page.evaluate("fixture.notices.at(-1)")
            page.get_by_role("button", name="candidate").click()
            page.get_by_role("button", name="接受").click()
            page.wait_for_function("fixture.calls.length===1")
            accepted = page.evaluate("fixture.calls[0]")
            assert accepted["operation"] == "decide_candidate"
            assert accepted["payload"] == {"annotation_id": ann["id"], "decision": "accepted",
                "actor_id": "codex-assistant", "actor_type": "assistant",
                "expected_candidate_sha256": "b" * 64, "changes": {}, "diff_summary": ""}
            assert page.evaluate("fixture.state.task.annotations[0].review_status") == "unreviewed"
            page.evaluate("ann=>{const s=fixture.state;s.task.annotations=[ann];s.visionCandidateDigests[ann.id]='c'.repeat(64);fixture.renderVisionCandidates([ann]);}", ann)
            page.get_by_role("button", name="candidate").click()
            page.get_by_role("button", name="拒绝").click()
            page.wait_for_function("fixture.calls.length===2")
            rejected = page.evaluate("fixture.calls[1]")
            assert rejected["operation"] == "decide_candidate"
            assert rejected["payload"]["decision"] == "rejected"
            assert rejected["payload"]["expected_candidate_sha256"] == "c" * 64
            assert page.evaluate("fixture.state.task.annotations.length") == 0
            page.evaluate("()=>fixture.decideVisionCandidate('ann-candidate','accepted')")
            assert page.evaluate("fixture.calls.length") == 2
            assert "候选或摘要已变化" in page.evaluate("fixture.notices.at(-1)")
            assert not errors, errors
        finally:
            browser.close()
