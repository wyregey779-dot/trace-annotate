"""Real Chromium coverage for persistent per-video analysis history.

The media is synthetic, with missing optional-model resources. The shipped
browser, HTTP routes, queue and SQLite results all run without substitutes.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os

import pytest

from tests.test_workspace_server import command, running
from tests.test_workspace_vision_jobs import clip
from trace_annotate.workspace.store import WorkspaceStore
from trace_annotate.workspace.vision_jobs import VisionJobs


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    channel = os.getenv("TRACE_ANNOTATE_BROWSER_CHANNEL") or (
        "chrome" if os.name == "nt" else None
    )
    with playwright.sync_playwright() as runtime:
        instance = runtime.chromium.launch(headless=True, channel=channel)
        try:
            yield instance, playwright.expect
        finally:
            instance.close()


def _submit(jobs, task_id, command_id, *, end=None):
    task = jobs.store.get_task(task_id)
    window = None
    if end is not None:
        duration = Fraction(end, 100)
        window = {
            "start": {"n": "0", "d": "1"},
            "end": {"n": str(duration.numerator), "d": str(duration.denominator)},
        }
    return jobs.submit(
        task_id,
        command_id=command_id,
        expected_revision=task["revision"],
        expected_sha256=task["content_sha256"],
        window=window,
    )["jobs"][0]


def _revise_task(store, task_id):
    route, edit = command(store, "history-revision")
    assert route.endswith(f"/{task_id}/commands")
    store.apply_task_command(task_id=task_id, request_body=json.dumps(edit).encode(), **edit)


@pytest.fixture
def project(tmp_path, clip, monkeypatch):
    # Maintain the same media identity in both tasks. Task ID, not input video,
    # must isolate history. Disable only the automatic vision worker for the
    # server so the test controls its active job's terminal transition.
    monkeypatch.setattr(VisionJobs, "start", lambda self: self)
    monkeypatch.setenv("TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD", "0")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path / "missing-clip"))
    monkeypatch.setenv("TRACE_ANNOTATE_VLM_CACHE", str(tmp_path / "missing-vlm"))
    monkeypatch.setenv("TRACE_ANNOTATE_OCR_MODELS", str(tmp_path / "missing-ocr"))
    store = WorkspaceStore(tmp_path / "workspace")
    store.create()
    try:
        task_a = store.import_source(clip, display_name="history-a.mp4")["task_id"]
        jobs = VisionJobs(store, clip_model_path="missing-clip", vlm_model_path="missing-vlm",
                          person_model_path="", landmark_model_path="", hand_model_path="",
                          action_video_model_path="")
        try:
            old = _submit(jobs, task_a, "history-original")
            assert jobs.run_pending()[0]["state"] == "succeeded"
            original = jobs.result(old["id"])
            assert original["input_revision"] == 0 and original["scenes"]
            _revise_task(store, task_a)
            for index in range(20):
                cancelled = _submit(jobs, task_a, f"history-cancel-{index}", end=50 + index)
                jobs.queue.cancel(cancelled["id"])

            task_b = store.import_source(clip, display_name="history-b.mp4")["task_id"]
            other = _submit(jobs, task_b, "other-task-history")
            assert jobs.run_pending()[0]["state"] == "succeeded"
            current = _submit(jobs, task_a, "history-running", end=80)
            claimed = jobs.queue.claim("history-worker")
            assert claimed["id"] == current["id"]
            yield store, task_a, task_b, original, other, claimed
        finally:
            jobs.close()
    finally:
        store.close()


def _task(page, task_id, expect):
    button = page.locator(f'button[data-task-id="{task_id}"]')
    button.click()
    expect(button).to_have_attribute("aria-current", "true")


def _open(page):
    history = page.locator("#vision-history")
    if history.get_attribute("open") is None:
        history.locator("summary").click()


def _row(page, job_id):
    return page.locator(f'[data-vision-job-id="{job_id}"]')


def _job_url(job_id, endpoint):
    return f"**/jobs/{job_id}/{endpoint}"


def test_reopened_browser_can_page_and_view_an_existing_read_only_result(project, browser):
    store, task_a, _, original, _, active = project
    chrome, expect = browser
    initial_task = store.get_task(task_a)
    original_id = original["job_id"]
    page_errors = []
    vision_writes = []

    def capture(request):
        if request.method == "POST" and request.url.endswith(("/vision-jobs", "/apply-vision", "/control")):
            vision_writes.append(request.url)

    with running(store) as server:
        for reopened in (False, True):
            context = chrome.new_context()
            try:
                page = context.new_page()
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.on("request", capture)
                page.goto(server.editor_url)
                _task(page, task_a, expect)
                if reopened:
                    page.locator("#vision-refresh").click()
                    expect(page.locator("#vision-history")).to_have_attribute("open", "")
                else:
                    summary = page.locator("#vision-history-toggle")
                    summary.focus()
                    summary.press("Space")
                    expect(page.locator("#vision-history")).to_have_attribute("open", "")
                    assert page.locator("#player").evaluate("video => video.paused")
                rows = page.locator("#vision-history [data-vision-job-id]")
                expect(rows).to_have_count(20)
                expect(_row(page, active["id"])).to_be_visible()
                expect(_row(page, original_id)).to_have_count(0)
                expect(page.locator("#vision-job")).to_be_hidden()

                page.locator("#vision-history-next").click()
                expect(rows).to_have_count(2)
                expect(_row(page, original_id)).to_be_visible()
                with page.expect_response(_job_url(original_id, "vision-result")) as response:
                    _row(page, original_id).click()
                assert response.value.ok
                assert response.value.json() == original
                expect(page.locator("#vision-job")).to_contain_text(original_id[-8:])
                expect(page.locator("#vision-job")).to_contain_text("已完成")
                expect(page.locator("#vision-job")).to_contain_text("仅供查看")
                expect(page.locator("#vision-apply")).to_be_visible()
                expect(page.locator("#vision-apply")).to_be_disabled()

                page.locator("#vision-history-prev").click()
                expect(rows).to_have_count(20)
                expect(page.locator("#vision-job")).to_contain_text(original_id[-8:])
            finally:
                context.close()

        assert not page_errors, page_errors
        assert not vision_writes, vision_writes
        assert store.get_task(task_a) == initial_task
        with store.transaction():
            assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 23


def test_late_history_and_running_status_cannot_overwrite_the_user_selection(project, browser):
    store, task_a, task_b, original, other, active = project
    chrome, expect = browser
    original_id = original["job_id"]

    with running(store) as server:
        page = chrome.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(server.editor_url)
            _task(page, task_a, expect)
            history = f"**/tasks/{task_a}/vision-history*"
            delayed_history = []
            delayed_result = []
            delayed_active = []

            def hold_history(route):
                delayed_history.append((route, route.fetch()))

            def hold_active(route):
                delayed_active.append((route, route.fetch()))

            def hold_result(route):
                delayed_result.append((route, route.fetch()))

            page.route(history, hold_history)
            _open(page)
            expect(page.locator("#vision-history-next")).to_be_disabled()
            _task(page, task_b, expect)
            _open(page)
            expect(_row(page, other["id"])).to_be_visible()
            assert delayed_history, "Task A response was held before switching to task B"
            for route, response in delayed_history:
                route.fulfill(response=response)
            page.unroute(history, hold_history)
            expect(page.locator("#vision-history [data-vision-job-id]")).to_have_count(1)
            expect(_row(page, other["id"])).to_be_visible()
            expect(_row(page, active["id"])).to_have_count(0)

            # Hold a valid completed result after its status arrived. The new
            # task's completed result must remain the active in-memory result
            # even if the first task's older response arrives afterward.
            _task(page, task_a, expect)
            _open(page)
            page.locator("#vision-history-next").click()
            result_endpoint = _job_url(original_id, "vision-result")
            page.route(result_endpoint, hold_result)
            with page.expect_request(result_endpoint):
                _row(page, original_id).click()
            _task(page, task_b, expect)
            _open(page)
            with page.expect_response(_job_url(other["id"], "vision-result")) as current_result:
                _row(page, other["id"]).click()
            assert current_result.value.ok
            expect(page.locator("#vision-job")).to_contain_text(other["id"][-8:])
            expect(page.locator("#vision-apply")).to_be_visible()
            expect(page.locator("#vision-apply")).to_be_enabled()
            assert delayed_result
            for route, response in delayed_result:
                assert response.ok and response.json()["job_id"] == original_id
                route.fulfill(response=response)
            page.unroute(result_endpoint, hold_result)
            expect(page.locator("#vision-job")).to_contain_text(other["id"][-8:])
            assert page.evaluate("state.visionResult?.job_id") == other["id"]

            _task(page, task_a, expect)
            _open(page)
            expect(_row(page, active["id"])).to_be_visible()
            status = _job_url(active["id"], "vision-status")
            page.route(status, hold_active)
            _row(page, active["id"]).click()
            expect(page.locator("#vision-job")).to_contain_text(active["id"][-8:])
            page.locator("#vision-history-next").click()
            expect(_row(page, original_id)).to_be_visible()
            _row(page, original_id).click()
            expect(page.locator("#vision-job")).to_contain_text("已完成")
            assert delayed_active, "The active job's response was held before selecting an older job"
            for route, response in delayed_active:
                assert response.json()["state"] == "running"
                route.fulfill(response=response)
            page.unroute(status, hold_active)

            server.queue.fail(active["id"], active["lease_generation"], "history-worker")
            page.locator("#vision-history-prev").click()
            expect(_row(page, active["id"])).to_be_visible()
            expect(page.locator("#vision-job")).to_contain_text(original_id[-8:])
            expect(page.locator("#vision-apply")).to_be_disabled()
            _row(page, active["id"]).click()
            expect(page.locator("#vision-job")).to_contain_text(active["id"][-8:])
            expect(page.locator("#vision-job")).to_contain_text("失败")
            expect(page.locator("#vision-apply")).to_be_hidden()
            assert not errors, errors
        finally:
            page.close()
