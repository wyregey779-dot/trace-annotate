"""Execute shipped professional-export controls, including the reference opt-in."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from urllib.parse import parse_qs, urlsplit

import pytest


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def _implementation(begin, end):
    source = UI.read_text(encoding="utf-8")
    offset = source.index(begin)
    return source[offset:source.index(end, offset)]


def _node_check(program):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for shipped professional-export logic")
    bootstrap = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
const implementation=JSON.parse(process.argv[1]),calls=[],downloads=[],notices=[],confirms=[];
const button={disabled:false};
const response=(mode='embedded',revision='7',sha='saved-sha')=>({
 headers:new Headers({'X-TraceAnnotate-Evidence-Mode':mode,'X-TraceAnnotate-Task-Revision':revision,'X-TraceAnnotate-Content-SHA256':sha}),
 blob:async()=>new Blob(['{"evidence_mode":"'+mode+'"}'],{type:'application/json'})});
const failure=code=>Object.assign(new Error(code),{code});
const context={state:{task:{id:'task-A',revision:7,content_sha256:'saved-sha'},sequence:4,busy:false,pending:null,loadingTask:false,dirty:false},
 $:()=>button,Headers,Blob,
 URL:{createObjectURL:()=> 'blob:download',revokeObjectURL:()=>{}},setTimeout:()=>1,
 document:{createElement:()=>({click(){downloads.push(this.download);},remove(){}}),body:{append(){}}},
 notice:(...args)=>notices.push(args),describeError:e=>e.code||e.message,
 confirmAction:async(message,options)=>{confirms.push({message,options});return true;},
 request:async path=>{calls.push(path);return response();}};
vm.createContext(context);vm.runInContext(implementation,context);
(async()=>{
'''
    completed = subprocess.run(
        [node, "-e", bootstrap + program + "\n})().catch(e=>{console.error(e);process.exitCode=1;});",
         json.dumps(_implementation("async function exportProfessionalTask(", "async function exportTemporalTask("))],
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_small_professional_export_keeps_the_embedded_download_path():
    _node_check(r'''
await context.exportProfessionalTask();
assert.deepEqual(calls,['/tasks/task-A/export?format=professional']);
assert.equal(confirms.length,0);assert.equal(downloads[0],'TraceAnnotate-task-A-r7.professional.json');
assert.equal(context.state.busy,false);assert.equal(context.state.pending,null);assert.equal(button.disabled,false);
''')


def test_reference_export_requires_specific_limit_and_explicit_confirmation():
    _node_check(r'''
context.request=async path=>{calls.push(path);if(calls.length===1)throw failure('PROFESSIONAL_LIMIT_EXCEEDED');return response('references');};
await context.exportProfessionalTask();
assert.equal(calls.length,2);assert.ok(calls[1].endsWith('&evidence=references'));
assert.equal(confirms.length,1);assert.equal(confirms[0].options.accept,'导出引用JSON');assert.equal(confirms[0].options.cancel,'返回');
assert.match(confirms[0].message,/证据/);assert.match(confirms[0].message,/项目|备份/);
assert.match(downloads[0],/references/);assert.match(notices.at(-1)[0],/引用/);
assert.equal(context.state.busy,false);assert.equal(context.state.pending,null);
''')


@pytest.mark.parametrize("code", ["INTERNAL_ERROR", "SOURCE_MISSING", "HTTP_413"])
def test_other_export_errors_never_fall_back(code):
    _node_check(r'''
context.request=async path=>{calls.push(path);throw failure(ERROR_CODE);};
await context.exportProfessionalTask();assert.equal(calls.length,1);assert.equal(confirms.length,0);
assert.equal(downloads.length,0);assert.equal(notices.at(-1)[1],'error');
'''.replace("ERROR_CODE", json.dumps(code)))


def test_reference_confirmation_cancel_makes_no_second_request():
    _node_check(r'''
context.request=async path=>{calls.push(path);throw failure('PROFESSIONAL_LIMIT_EXCEEDED');};
context.confirmAction=async()=>false;await context.exportProfessionalTask();
assert.equal(calls.length,1);assert.equal(downloads.length,0);assert.equal(button.disabled,false);
''')


@pytest.mark.parametrize("change", ["task", "sequence", "revision", "hash"])
def test_stale_reference_confirmation_cannot_export_another_context(change):
    updates = {"task": "context.state.task.id='task-B';", "sequence": "context.state.sequence++;",
               "revision": "context.state.task.revision++;", "hash": "context.state.task.content_sha256='new-sha';"}
    _node_check(r'''
context.request=async path=>{calls.push(path);throw failure('PROFESSIONAL_LIMIT_EXCEEDED');};
context.confirmAction=async()=>{UPDATE;return true;};
await context.exportProfessionalTask();assert.equal(calls.length,1);assert.equal(downloads.length,0);
'''.replace("UPDATE", updates[change]))


def test_task_switch_before_response_or_dirty_confirmation_cannot_download():
    _node_check(r'''
let resolve;context.request=path=>{calls.push(path);return new Promise(done=>{resolve=done;});};
const pending=context.exportProfessionalTask();context.state.sequence++;context.state.task.id='task-B';resolve(response());await pending;
assert.equal(downloads.length,0);
context.state.dirty=true;context.confirmAction=async()=>{context.state.sequence++;return true;};
const before=calls.length;await context.exportProfessionalTask();assert.equal(calls.length,before);
''')


@pytest.mark.parametrize("mode,revision,sha", [("references", "7", "saved-sha"), ("embedded", "8", "saved-sha"), ("embedded", "7", "changed")])
def test_export_response_must_match_frozen_mode_and_revision(mode, revision, sha):
    _node_check(r'''
context.request=async path=>{calls.push(path);return response(MODE,REVISION,SHA);};
await context.exportProfessionalTask();assert.equal(downloads.length,0);assert.equal(notices.at(-1)[1],'error');
'''.replace("MODE", json.dumps(mode)).replace("REVISION", json.dumps(revision)).replace("SHA", json.dumps(sha)))


def _browser_harness():
    source = UI.read_text(encoding="utf-8")
    button = re.search(r'<button id="export-professional-btn"[^>]*>.*?</button>', source).group()
    dialog = re.search(r'<dialog id="confirm-dialog".*?</dialog>', source).group()
    script = "\n".join([
        "const token='fixture-token',API='/api/v1/projects/current';const $=id=>document.getElementById(id);",
        "const state={task:{id:'task-A',revision:7,content_sha256:'saved-sha'},sequence:4,busy:false,pending:null,loadingTask:false,dirty:false};",
        _implementation("function notice(", "async function json("),
        _implementation("function confirmAction(", "async function mayLeave("),
        _implementation("async function exportProfessionalTask(", "async function exportTemporalTask("),
        "$('export-professional-btn').addEventListener('click',exportProfessionalTask);",
    ])
    return '<!doctype html><meta charset="utf-8"><title>Shipped export-control regression</title>' + button + '<p id="status" role="status"></p>' + dialog + '<script>' + script + '</script>'


def test_headless_browser_clicks_confirmation_download_and_context_race(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api", reason="Optional browser-test dependency is required for real export controls")
    observed = {"mode": "small", "requests": [], "headers": [], "entered": threading.Event(), "release": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                body = _browser_harness().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/api/v1/projects/current/tasks/task-A/export":
                observed["requests"].append(self.path)
                observed["headers"].append(self.headers.get("X-TraceAnnotate-Token"))
                query = parse_qs(parsed.query)
                evidence_mode = query.get("evidence", ["embedded"])[0]
                failure = observed["mode"] in {"limit", "other"} and evidence_mode == "embedded"
                if observed["mode"] == "delay":
                    observed["entered"].set()
                    observed["release"].wait(10)
                if failure:
                    self.send_response(400)
                    code = "PROFESSIONAL_LIMIT_EXCEEDED" if observed["mode"] == "limit" else "SOURCE_MISSING"
                    body = json.dumps({"error": {"code": code, "message": "Export fixture rejection"}}).encode()
                else:
                    self.send_response(200)
                    body = json.dumps({"evidence_mode": evidence_mode, "task": {"id": "task-A", "revision": 7}, "evidence": [{"sha256": "fixture-object", "embedded": evidence_mode == "embedded"}]}).encode()
                    for key, value in {"X-TraceAnnotate-Evidence-Mode": evidence_mode, "X-TraceAnnotate-Task-Revision": "7", "X-TraceAnnotate-Content-SHA256": "saved-sha", "X-TraceAnnotate-Export-Losses": "Fixture evidence bytes external" if evidence_mode == "references" else "Fixture current task"}.items():
                        self.send_header(key, value)
                self.send_header("Content-Type", "application/json")
            else:
                self.send_response(404)
                body = b"Not found"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runtime:
            channel = os.environ.get("TRACE_ANNOTATE_BROWSER_CHANNEL") or ("chrome" if os.name == "nt" else None)
            browser = runtime.chromium.launch(headless=True, channel=channel, downloads_path=str(tmp_path / "downloads"))
            context = browser.new_context(accept_downloads=True)
            page_errors, downloads, network = [], [], []

            def page_for(mode):
                observed["mode"] = mode
                observed["requests"].clear()
                page = context.new_page()
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.on("download", lambda download: downloads.append(download.suggested_filename))
                page.on("response", lambda response: network.append({"url": response.url, "status": response.status, "headers": response.headers}) if "/export?" in response.url else None)
                page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
                return page

            page = page_for("small")
            with page.expect_download() as download:
                page.locator("#export-professional-btn").click()
            assert json.loads(Path(download.value.path()).read_text())["evidence_mode"] == "embedded"
            download.value.save_as(tmp_path / "embedded.professional.json")
            assert len(observed["requests"]) == 1
            page.close()

            page = page_for("limit")
            page.locator("#export-professional-btn").click()
            playwright.expect(page.locator("#confirm-dialog")).to_be_visible()
            assert "备份" in page.locator("#confirm-message").inner_text()
            with page.expect_download() as download:
                page.get_by_role("button", name="导出引用JSON", exact=True).click()
            downloaded = json.loads(Path(download.value.path()).read_text())
            download.value.save_as(tmp_path / "references.professional.json")
            assert downloaded["evidence_mode"] == "references"
            assert downloaded["evidence"][0]["embedded"] is False
            assert "references" in download.value.suggested_filename
            assert len(observed["requests"]) == 2
            playwright.expect(page.locator("#status")).to_contain_text("引用")
            page.close()

            for close_with_escape in (False, True):
                page = page_for("limit")
                before = len(downloads)
                page.locator("#export-professional-btn").click()
                playwright.expect(page.locator("#confirm-dialog")).to_be_visible()
                if close_with_escape:
                    page.keyboard.press("Escape")
                else:
                    page.get_by_role("button", name="返回", exact=True).click()
                playwright.expect(page.locator("#export-professional-btn")).to_be_enabled()
                assert len(observed["requests"]) == 1
                assert len(downloads) == before
                page.close()

            page = page_for("other")
            page.locator("#export-professional-btn").click()
            playwright.expect(page.locator("#status")).to_contain_text("SOURCE_MISSING")
            assert len(observed["requests"]) == 1
            playwright.expect(page.locator("#confirm-dialog")).not_to_be_visible()
            page.close()

            page = page_for("limit")
            before = len(downloads)
            page.locator("#export-professional-btn").click()
            playwright.expect(page.locator("#confirm-dialog")).to_be_visible()
            page.evaluate("state.task.id='task-B';state.sequence++")
            page.get_by_role("button", name="导出引用JSON", exact=True).click()
            playwright.expect(page.locator("#export-professional-btn")).to_be_enabled()
            assert len(observed["requests"]) == 1
            assert len(downloads) == before
            page.close()

            page = page_for("delay")
            before = len(downloads)
            page.locator("#export-professional-btn").click()
            assert observed["entered"].wait(5)
            page.evaluate("state.task.id='task-B';state.sequence++")
            observed["release"].set()
            playwright.expect(page.locator("#export-professional-btn")).to_be_enabled()
            assert len(downloads) == before
            assert all(value == "fixture-token" for value in observed["headers"])
            assert page_errors == []
            (tmp_path / "browser-observations.json").write_text(json.dumps({
                "schema": "traceannotate.professional-export-browser-regression.v1",
                "status": "PASS_SCOPED", "headless": True, "browser_channel": channel,
                "scope": "Shipped export button, native confirmation, request helper and export implementation exercised against a loopback fixture API. This does not establish production-store schema or end-to-end video quality.",
                "ui_source_sha256": hashlib.sha256(UI.read_bytes()).hexdigest(),
                "checks": ["embedded normal download", "explicit references confirmation and metadata", "cancel button without second request", "Escape without second request", "unrelated error without fallback", "task change while confirmation open", "task change while response pending"],
                "network": network, "downloads": downloads, "page_errors": page_errors,
                "downloaded_files": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ("embedded.professional.json", "references.professional.json")},
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            context.close()
            browser.close()
    finally:
        observed["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
