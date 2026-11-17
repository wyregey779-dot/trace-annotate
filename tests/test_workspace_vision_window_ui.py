"""Execute the shipped analysis-window form logic and inspect its submitted contract."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from trace_annotate.workspace.ui import EDITOR_HTML


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def _implementation(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _window_submit_binding(source):
    bind = source[source.index("function bind(){"):]
    match = re.search(
        r"\$\(['\"]vision-window-form['\"]\)\.addEventListener\(['\"]submit['\"],\s*runVision\);",
        bind,
    )
    assert match is not None, "The shipped form must bind submit to runVision"
    return match.group(0)


def test_window_form_submits_exact_bounded_interval_and_reports_sparse_coverage():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the shipped browser logic")
    source = UI.read_text(encoding="utf-8")
    assert '<form id="vision-window-form"' in EDITOR_HTML
    assert 'id="vision-window-start"' in EDITOR_HTML
    assert 'id="vision-window-end"' in EDITOR_HTML
    snippets = [
        _implementation(source, "function gcd(", "function clone("),
        _implementation(source, "function setVisionWindowMedia(", "function renderVisionRetry("),
        _implementation(source, "async function runVision(", "async function cancelVision("),
    ]
    script = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const snippets=JSON.parse(process.argv[1]);
const fields=new Map();
const element=id=>{
 if(!fields.has(id))fields.set(id,{value:'',textContent:'',hidden:false,disabled:false,
  attributes:{},setAttribute(name,value){this.attributes[name]=value;},focus(){this.focused=true;}});
 return fields.get(id);
};
const calls=[],responses=[];
const media={duration:{n:'240',d:'1'},video_start:{n:'0',d:'1'},video_end:{n:'240',d:'1'}};
const state={task:{id:'task-A',revision:4,content_sha256:'a'.repeat(64)},media,
 sequence:1,busy:false,dirty:false,loadingTask:false,visionTimer:null,visionResult:null,visionJob:null};
const context={state,$:element,BigInt,console,JSON,crypto:{randomUUID:()=> '1'.repeat(32)},
 commandId:()=> 'vis-test',renderSaveState:()=>{},clearVisionEvidence:()=>{},renderVisionJob:()=>{},pollVision:()=>{},
 notice:(...args)=>calls.push(args),describeError:error=>String(error),
 json:async(path,options)=>{responses.push({path,body:JSON.parse(options.body)});return{jobs:[{id:'job-A',state:'queued'}]};}};
vm.createContext(context);
for(const snippet of snippets)vm.runInContext(snippet,context);
(async()=>{
 context.setVisionWindowMedia(media);
 assert.equal(element('vision-window-start').value,'0');
 assert.equal(element('vision-window-end').value,'30');
 element('vision-window-start').value='1/3';
 element('vision-window-end').value='100/3';
 await context.runVision({preventDefault(){}});
 assert.equal(responses.length,1);
 assert.deepEqual(JSON.parse(JSON.stringify(responses[0].body.window)),
  {start:{n:'1',d:'3'},end:{n:'100',d:'3'}});
 assert.equal(state.visionJob.requested_window.start.n,'1');
 element('vision-window-start').value='0';element('vision-window-end').value='121';
 await context.runVision({preventDefault(){}});
 assert.equal(responses.length,1,'overlong window must not reach server');
 assert.match(element('vision-window-error').textContent,/120/);
 assert.equal(element('vision-window-end').attributes['aria-invalid'],'true');
 element('vision-window-start').value='200';element('vision-window-end').value='241';
 await context.runVision({preventDefault(){}});
 assert.equal(responses.length,1,'beyond-media window must not reach server');
 element('vision-window-start').value='10';element('vision-window-end').value='10';
 await context.runVision({preventDefault(){}});
 assert.equal(responses.length,1,'empty window must not reach server');
 state.visionResult={window:{start:{n:'1',d:'3'},end:{n:'100',d:'3'}},
  keyframes:[{timestamp:{n:'1',d:'3'}},{timestamp:{n:'99',d:'3'}}],
  sampling:{returned:2,planned:96,truncated:true},module_status:{scene:'success',person:'failed',face:'not_run'}};
 context.renderVisionCoverage();
 assert.match(element('vision-coverage').textContent,/采样 2\/96，后段被截断/);
 assert.match(element('vision-coverage').textContent,/person=failed/);
 assert.match(element('vision-coverage').textContent,/face=not_run/);
 assert.match(element('vision-coverage').textContent,/不证明逐帧覆盖/);
 state.task={...state.task,id:'task-B'};
 state.media={duration:{n:'60',d:'1'},video_start:{n:'21',d:'2'},video_end:{n:'51',d:'2'}};
 context.setVisionWindowMedia(state.media);
 assert.equal(element('vision-window-start').value,'10.5');
 assert.equal(element('vision-window-end').value,'25.5');
 assert.match(element('vision-window-help').textContent,/10.5–25.5/);
 element('vision-window-start').value='10';element('vision-window-end').value='20';
 await context.runVision({preventDefault(){}});
 assert.equal(responses.length,1,'before-first-frame window must not reach server');
 element('vision-window-start').value='11';element('vision-window-end').value='26';
 await context.runVision({preventDefault(){}});
 assert.equal(responses.length,1,'past-picture-end window must not reach server');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", script, json.dumps(snippets)], capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_chrome_click_submits_window_and_rejects_empty_input():
    source = UI.read_text(encoding="utf-8")
    binding = _window_submit_binding(source)
    with pytest.raises(AssertionError, match="must bind submit"):
        _window_submit_binding(source.replace(binding, "", 1))
    playwright = pytest.importorskip("playwright.sync_api")

    form_start = EDITOR_HTML.index('<form id="vision-window-form"')
    form_end = EDITOR_HTML.index("</form>", form_start) + len("</form>")
    form = EDITOR_HTML[form_start:form_end]
    assert "novalidate" in form
    snippets = [
        _implementation(source, "function gcd(", "function clone("),
        _implementation(source, "function setVisionWindowMedia(", "function renderVisionRetry("),
        _implementation(source, "async function runVision(", "async function cancelVision("),
    ]
    bootstrap = """
const $=id=>document.getElementById(id);
const media={duration:{n:'240',d:'1'},video_start:{n:'0',d:'1'},video_end:{n:'240',d:'1'}};
const state={task:{id:'task-A',revision:4,content_sha256:'a'.repeat(64)},media,
 sequence:1,busy:false,dirty:false,loadingTask:false,visionTimer:null,visionResult:null,visionJob:null};
const posts=[];
const commandId=()=> 'vis-browser-test';
const renderSaveState=()=>{},clearVisionEvidence=()=>{},renderVisionJob=()=>{},pollVision=()=>{};
const notice=()=>{},describeError=error=>String(error);
async function json(path,options){posts.push({path,method:options.method,body:JSON.parse(options.body)});
 return {jobs:[{id:'job-A',state:'queued'}]};}
"""
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content("<html><body>" + form + "</body></html>")
            page.add_script_tag(content=bootstrap + "\n" + "\n".join(snippets)
                                + "\n" + binding + "\nwindow.visionWindowPosts=posts;")
            page.evaluate("setVisionWindowMedia(media)")
            page.locator("#vision-run").evaluate("button => button.disabled = false")
            page.locator("#vision-window-start").fill("1/3")
            page.locator("#vision-window-end").fill("100/3")
            page.locator("#vision-run").click()
            page.wait_for_function("window.visionWindowPosts.length === 1")
            post = page.evaluate("window.visionWindowPosts[0]")
            assert post == {
                "path": "/tasks/task-A/vision-jobs", "method": "POST",
                "body": {"command_id": "vis-browser-test", "expected_revision": 4,
                         "expected_sha256": "a" * 64,
                         "window": {"start": {"n": "1", "d": "3"},
                                    "end": {"n": "100", "d": "3"}},
                         "subject_annotation_id": None},
            }

            page.locator("#vision-window-start").fill("")
            page.locator("#vision-run").click()
            assert page.evaluate("window.visionWindowPosts.length") == 1
            assert page.locator("#vision-window-error").is_visible()
            assert page.locator("#vision-window-start").get_attribute("aria-invalid") == "true"
            assert page.evaluate("document.activeElement.id") == "vision-window-start"
            assert not errors, errors
        finally:
            browser.close()
