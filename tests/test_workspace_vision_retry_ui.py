"""Shipped retry controls in real Chromium, plus SQLite/HTTP retry fencing.

The browser server supplies controlled job states; this verifies controls,
network receipts and context races, not model quality or manual review.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import threading

import pytest

from trace_annotate.workspace.queue import DurableQueue
from tests.test_workspace_queue import store, queue_clip, task, enqueue, code
from tests.test_workspace_server import api, http_video, request, command, assert_error


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_guarded_retry_replays_receipt_after_edit_without_reentering_job(store, terminal):
    queue = DurableQueue(store)
    job = enqueue(queue, store)
    if terminal == "failed":
        claimed = queue.claim("retry-worker")
        queue.fail(job["id"], claimed["lease_generation"], "retry-worker", "INSUFFICIENT_MEMORY")
    else:
        queue.cancel(job["id"])
    frozen = task(store)
    options = {"command_id": "guarded-retry", "expected_revision": frozen["revision"],
               "expected_sha256": frozen["content_sha256"]}
    receipt = queue.control_job(job["id"], "retry_" + terminal, **options)
    claimed = queue.claim("retried-worker")
    queue.fail(job["id"], claimed["lease_generation"], "retried-worker")
    path, edit = command(store, "edit-after-retry")
    annotation = edit["payload"]["annotation"]
    annotation["type"] = "whole_clip"
    annotation.pop("start")
    annotation.pop("end")
    store.apply_task_command(task_id=frozen["id"], request_body=json.dumps(edit).encode(), **edit)
    assert queue.control_job(job["id"], "retry_" + terminal, **options) == receipt
    assert queue.get_job(job["id"])["state"] == "failed"
    code("REVISION_CONFLICT", lambda: queue.control_job(job["id"], "retry_failed",
         **{**options, "command_id": "new-stale-retry"}))
    code("COMMAND_UNKNOWN", lambda: store.get_command_receipt("new-stale-retry"))
    assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_guarded_retry_hash_and_schema_rejection_do_not_change_attempt(store):
    queue = DurableQueue(store)
    job = enqueue(queue, store)
    queue.cancel(job["id"])
    before = queue.get_job(job["id"])
    code("MEDIA_CHANGED", lambda: queue.control_job(job["id"], "retry_cancelled",
         command_id="wrong-hash", expected_revision=0, expected_sha256="0" * 64))
    for options in ({"expected_revision": 0}, {"expected_sha256": "a" * 64},
                    {"expected_revision": True, "expected_sha256": "a" * 64}):
        code("SCHEMA_INVALID", lambda: queue.control_job(job["id"], "retry_cancelled", **options))
    code("SCHEMA_INVALID", lambda: queue.control_job(job["id"], "cancel",
         expected_revision=0, expected_sha256=task(store)["content_sha256"]))
    assert queue.get_job(job["id"]) == before


def test_http_guarded_retry_refuses_edit_between_get_and_post(api):
    server, store = api
    queue = server.queue
    current = task(store)
    job = enqueue(queue, store)
    queue.cancel(job["id"])
    prefix = "/api/v1/projects/current"
    assert request(server, "GET", prefix + "/tasks/" + current["id"])[2] == current
    route = prefix + "/jobs/" + job["id"] + "/control"
    committed_body = {"command_id": "confirmed-http-retry", "action": "retry_cancelled",
                      "expected_revision": current["revision"], "expected_sha256": current["content_sha256"]}
    committed = request(server, "POST", route, committed_body)
    assert committed[0] == 200 and committed[2] == {"job_id": job["id"], "action": "retry_cancelled", "state": "queued"}
    queue.cancel(job["id"])
    path, edit = command(store, "intervening-edit")
    assert request(server, "POST", path, edit)[0] == 200
    assert request(server, "POST", route, committed_body)[2] == committed[2]
    assert queue.get_job(job["id"])["state"] == "cancelled"
    body = {"command_id": "stale-http-retry", "action": "retry_cancelled",
            "expected_revision": current["revision"], "expected_sha256": current["content_sha256"]}
    assert_error(request(server, "POST", route, body), 409, "REVISION_CONFLICT")
    assert_error(request(server, "GET", prefix + "/commands/stale-http-retry"), 404, "COMMAND_UNKNOWN")
    assert queue.get_job(job["id"])["state"] == "cancelled"
    assert queue.get_job(job["id"])["attempt"] == job["attempt"]
    for patch in ({"expected_revision": None, "expected_sha256": None},
                  {"expected_revision": 0}, {"expected_sha256": "0" * 64}):
        malformed = {"command_id": "bad-http-retry", "action": "retry_cancelled", **patch}
        assert_error(request(server, "POST", route, malformed), 422, "SCHEMA_INVALID")
    assert_error(request(server, "POST", route, {**body, "action": "cancel"}), 422, "SCHEMA_INVALID")
    batch = queue.list_batches()[0]
    assert_error(request(server, "POST", prefix + "/batches/" + batch["id"] + "/control", body), 422, "SCHEMA_INVALID")


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def _implementation(source, begin, end):
    offset = source.index(begin)
    return source[offset:source.index(end, offset)]


def _harness():
    source = UI.read_text(encoding="utf-8")
    buttons = "".join(re.search(r'<button id="' + name + r'"[^>]*>.*?</button>', source).group()
                      for name in ("vision-run", "vision-apply", "vision-cancel", "vision-retry"))
    script = "\n".join([
        "const token='fixture-token',API='/api/v1/projects/current';const $=id=>document.getElementById(id);",
        "const fixtureTask={id:'task-A',revision:7,content_sha256:'" + "a" * 64 + "',annotations:[]};",
        "const state={snapshot:{project_id:'fixture-project'},task:fixtureTask,sequence:4,busy:false,pending:null,loadingTask:false,dirty:false,visionGeneration:0,visionJob:null,visionResult:null,visionTimer:null,visionAbort:null};",
        _implementation(source, "function notice(", "function confirmAction("),
        _implementation(source, "async function loadVisionOptions(", "function renderVisionCandidates("),
        _implementation(source, "function visionInputIsStale(", "async function runVision("),
        "function renderSaveState(){renderVisionRetry();}",
        "function installJob(value){state.visionJob={id:'job-fixture',task_id:'task-A',input_revision:7,input_sha256:fixtureTask.content_sha256,terminal:true,...value};renderVisionJob();renderSaveState();}",
        "window.fixture={installJob,switchTask(){state.visionGeneration++;if(state.visionAbort)state.visionAbort.abort();state.visionAbort=null;state.visionJob=null;state.visionResult=null;state.task={id:'task-B',revision:2,content_sha256:'b'.repeat(64),annotations:[]};state.busy=false;state.visionRetryOwner=null;renderVisionJob();renderSaveState();},view(){return JSON.parse(JSON.stringify(state));}};",
        "$('vision-retry').addEventListener('click',retryVision);installJob({state:'failed',error_code:'INSUFFICIENT_MEMORY'});",
    ])
    return '<!doctype html><meta charset="utf-8"><title>Shipped vision retry-control regression</title>' + buttons + '<p id="status" role="status"></p><p id="vision-options"></p><div id="vision-job"></div><script>' + script + '</script>'


def test_real_chrome_retry_click_receipts_stale_input_and_task_races(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api", reason="Real retry control requires optional Playwright")
    observed = {"mode": "success", "state": "failed", "posts": [], "receipts": {},
                "status_count": 0, "adapter": "not_loaded", "entered": threading.Event(), "release": threading.Event()}
    network = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value, content_type="application/json"):
            body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/":
                network.append({"method": "GET", "path": self.path, "scenario": observed["mode"]})
            if self.path == "/":
                self.send(200, _harness(), "text/html; charset=utf-8")
            elif self.path.endswith("/vision-options"):
                self.send(200, {"capabilities": {"scene": True, **{key: observed["adapter"] for key in
                                  ("person", "landmarks", "face", "ocr", "clip", "vlm")}}})
            elif "/commands/" in self.path:
                receipt = observed["receipts"].get(self.path.rsplit("/", 1)[1])
                self.send(200 if receipt else 404, {"state": "committed", "response": receipt} if receipt else
                          {"error": {"code": "COMMAND_UNKNOWN", "message": "No receipt"}})
            elif self.path.endswith("/tasks/task-A"):
                self.send(200, {"id": "task-A", "revision": 8 if observed["mode"] == "stale" else 7,
                               "content_sha256": "a" * 64, "annotations": []})
            elif self.path.endswith("/vision-status"):
                observed["status_count"] += 1
                if observed["state"] == "queued" and observed["status_count"] > 2:
                    observed["state"] = "succeeded"
                self.send(200, {"job_id": "job-fixture", "task_id": "task-A", "input_revision": 7,
                               "input_sha256": "a" * 64, "state": observed["state"]})
            elif self.path.endswith("/vision-result"):
                self.send(200, {"scenes": [], "keyframes": [], "model_runs": [], "module_status": {},
                               "annotations": [{"review_status": "unreviewed"}]})
            else:
                self.send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed["posts"].append({"path": self.path, "body": body, "type": self.headers.get("Content-Type"),
                                      "token": self.headers.get("X-TraceAnnotate-Token")})
            network.append({"method": "POST", "scenario": observed["mode"], **observed["posts"][-1]})
            if observed["mode"] in ("post-stale", "post-rejected"):
                self.send(409, {"error": {"code": "REVISION_CONFLICT" if observed["mode"] == "post-stale" else "INVALID_JOB_STATE", "message": "Changed",
                                         "details": {"commit_status": "not_committed"}}})
                return
            if observed["mode"] == "uncommitted" and len(observed["posts"]) == 1:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            reply = {"job_id": "job-fixture", "action": body["action"], "state": "queued"}
            observed["receipts"][body["command_id"]] = reply
            observed["state"] = "queued"
            if observed["mode"] == "lost":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if observed["mode"] in ("switch", "gate"):
                observed["entered"].set()
                observed["release"].wait(10)
            self.send(200, reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runtime:
            channel = os.environ.get("TRACE_ANNOTATE_BROWSER_CHANNEL") or ("chrome" if os.name == "nt" else None)
            browser = runtime.chromium.launch(headless=True, channel=channel)
            context = browser.new_context()
            page_errors = []

            def page_for(mode="success", state="failed"):
                observed.update(mode=mode, state=state, posts=[], receipts={}, status_count=0)
                observed["entered"].clear()
                observed["release"].clear()
                page = context.new_page()
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                if state != "failed":
                    page.evaluate("value=>fixture.installJob({state:value})", state)
                return page

            page = page_for("gate")
            playwright.expect(page.locator("#vision-job")).to_contain_text("可用内存不足")
            page.get_by_role("button", name="重试任务", exact=True).click()
            assert observed["entered"].wait(5)
            playwright.expect(page.locator("#vision-retry")).to_be_disabled()
            assert page.evaluate("fixture.view().visionJob.state") == "failed"  # No fake queued before receipt.
            page.locator("#vision-retry").click(force=True)
            observed["release"].set()
            playwright.expect(page.locator("#vision-job")).to_contain_text("已完成", timeout=8000)
            assert len(observed["posts"]) == 1
            post = observed["posts"][0]
            assert post["body"]["action"] == "retry_failed" and post["body"]["command_id"].startswith("vretry-")
            assert post["body"]["expected_revision"] == 7 and post["body"]["expected_sha256"] == "a" * 64
            assert post["type"] == "application/json" and post["token"] == "fixture-token"
            view = page.evaluate("fixture.view()")
            assert view["task"]["annotations"] == [] and view["visionResult"]["annotations"][0]["review_status"] == "unreviewed"
            page.close()

            page = page_for(state="cancelled")
            page.get_by_role("button", name="重试任务", exact=True).click()
            playwright.expect(page.locator("#vision-job")).to_contain_text("已完成", timeout=8000)
            assert len(observed["posts"]) == 1 and observed["posts"][0]["body"]["action"] == "retry_cancelled"
            page.close()

            for mode in ("stale", "post-stale"):
                page = page_for(mode)
                page.get_by_role("button", name="重试任务", exact=True).click()
                playwright.expect(page.locator("#status")).to_contain_text("旧任务未重试")
                assert len(observed["posts"]) == (1 if mode == "post-stale" else 0)
                assert page.evaluate("fixture.view().visionJob.state") == "failed"
                page.close()

            page = page_for("post-rejected")
            page.get_by_role("button", name="重试任务", exact=True).click()
            playwright.expect(page.locator("#status")).to_contain_text("重试未执行")
            assert "尚未确认" not in page.locator("#status").inner_text()
            view = page.evaluate("fixture.view()")
            assert view["visionJob"]["state"] == "failed" and view["visionJob"]["retryControl"] is None
            assert len(observed["posts"]) == 1 and not observed["receipts"]
            page.close()

            page = page_for("lost")
            page.get_by_role("button", name="重试任务", exact=True).click()
            playwright.expect(page.locator("#status")).to_contain_text("尚未确认")
            first_id = observed["posts"][0]["body"]["command_id"]
            assert page.evaluate("fixture.view().visionJob.retryControl.command_id") == first_id
            page.get_by_role("button", name="重试任务", exact=True).click()
            playwright.expect(page.locator("#vision-job")).to_contain_text("已完成", timeout=8000)
            assert len(observed["posts"]) == 1
            page.close()

            page = page_for("uncommitted")
            page.get_by_role("button", name="重试任务", exact=True).click()
            playwright.expect(page.locator("#status")).to_contain_text("尚未确认")
            first_body = observed["posts"][0]["body"]
            page.get_by_role("button", name="重试任务", exact=True).click()
            playwright.expect(page.locator("#vision-job")).to_contain_text("已完成", timeout=8000)
            assert len(observed["posts"]) == 2 and observed["posts"][1]["body"] == first_body
            assert len(observed["receipts"]) == 1
            page.close()

            page = page_for("switch")
            page.get_by_role("button", name="重试任务", exact=True).click()
            assert observed["entered"].wait(5)
            page.evaluate("fixture.switchTask()")
            observed["release"].set()
            page.wait_for_timeout(250)
            view = page.evaluate("fixture.view()")
            assert view["task"]["id"] == "task-B" and view["visionJob"] is None and view["busy"] is False
            playwright.expect(page.locator("#vision-retry")).to_be_hidden()
            page.close()

            page = page_for()
            for actual, label in (("not_loaded", "未加载"), ("released", "已验证，可按需运行"),
                                  ("unavailable", "不可用"), ("loaded", "可用")):
                observed["adapter"] = actual
                page.evaluate("loadVisionOptions()")
                playwright.expect(page.locator("#vision-options")).to_contain_text("CLIP " + label)
                assert "released" not in page.locator("#vision-options").inner_text()
            for error, text in {"MEMORY_STATUS_UNAVAILABLE": "无法读取可用内存", "MODEL_RESOURCE_BUSY": "其他模型任务",
                                "MODEL_TIMEOUT": "模型运行超时", "MODEL_MEMORY_FAILED": "内存不足停止",
                                "MODEL_WORKER_FAILED": "模型进程已停止", "MODEL_LOAD_FAILED": "本地模型加载失败", "MODEL_WORKER_PROTOCOL": "模型结果无法验证"}.items():
                page.evaluate("value=>fixture.installJob({state:'failed',error_code:value})", error)
                playwright.expect(page.locator("#vision-job")).to_contain_text(text)
            page.evaluate("fixture.installJob({state:'failed',input_revision:6})")
            playwright.expect(page.locator("#vision-retry")).to_be_disabled()
            page.evaluate("fixture.installJob({state:'failed',error_code:'VISION_RESUBMIT_REQUIRED'})")
            playwright.expect(page.locator("#vision-retry")).to_be_hidden()
            playwright.expect(page.locator("#vision-job")).to_contain_text("重新运行分析")
            assert not observed["posts"]
            page.close()
            assert page_errors == []
            (tmp_path / "retry-browser-observations.json").write_text(json.dumps({
                "scope": "shipped controls; controlled HTTP job states; not model or semantic acceptance",
                "browser": "Chrome headless" if channel == "chrome" else "Chromium headless",
                "browser_version": browser.version,
                "checks": ["failed retry", "cancelled retry", "no queued before receipt", "disabled repeat click",
                           "GET stale input", "POST stale race", "definite rejection differs from unknown receipt", "lost committed receipt", "lost uncommitted same command",
                           "task-switch isolation", "resource error messages", "outdated configuration requires new analysis",
                           "per-adapter unloaded/released/unavailable labels"],
                "page_errors": page_errors, "network": network}, ensure_ascii=False, indent=2), encoding="utf-8")
            browser.close()
    finally:
        observed["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(5)
