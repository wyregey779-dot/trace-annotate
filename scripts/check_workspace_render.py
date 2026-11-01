"""Exercise annotation-video controls in an independent headless Chromium.

Uses synthetic media and real app upload/edit/preview/render/download actions.
This is an automated development regression, never native user-Chrome,
semantic annotation, listening, final screen recording or release acceptance.
Output must be NEW; every configurable temporary path stays below it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback
import zipfile

SOURCE = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preserve_browser_installation():
    """Pin read-only browser dependencies before isolating writable XDG caches.

    Playwright resolves Linux's default registry through XDG_CACHE_HOME at
    driver startup. Moving that cache must not move previously installed
    browsers. Explicit registries (including hermetic "0") remain explicit.
    """
    for key in ("PLAYWRIGHT_BROWSERS_PATH", "npm_config_playwright_browsers_path",
                "npm_package_config_playwright_browsers_path"):
        if key in os.environ:
            configured = os.environ[key]
            if configured:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = configured
                return {"source": key, "path": configured, "read_only_dependency": True}
            break  # An explicit empty value uses the platform default.
    if sys.platform == "linux":
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    elif sys.platform == "darwin":
        cache = Path.home() / "Library/Caches"
    elif sys.platform == "win32":
        cache = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    else:
        raise RuntimeError("Unsupported Playwright browser-cache platform: " + sys.platform)
    location = str(cache / "ms-playwright")
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = location
    return {"source": "original_platform_cache", "path": location, "read_only_dependency": True}


def load_harness(installed_wheel):
    if installed_wheel is not None and not sys.flags.isolated:
        raise RuntimeError("Installed acceptance requires the target venv Python with -I")
    spec = importlib.util.spec_from_file_location("traceannotate_render_acceptance_support", SOURCE / "scripts/check_workspace_editor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if installed_wheel is not None:
        module.INSTALLED_IDENTITY["render_font_resources"] = verify_font_resources(module.PACKAGE, installed_wheel)
    return module


def verify_font_resources(package, wheel):
    """The editor support verifies code; also bind font and licence bytes."""
    prefix = "trace_annotate/workspace/fonts/"
    expected = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            item = Path(name)
            if ".." in item.parts or name in expected:
                raise RuntimeError("Invalid or duplicated font resource in wheel")
            local = package.joinpath(*item.parts[1:])
            data = archive.read(name)
            if not local.is_file() or local.read_bytes() != data:
                raise RuntimeError("Installed render font or licence differs from wheel: " + name)
            expected[name] = hashlib.sha256(data).hexdigest()
    actual = {"trace_annotate/" + path.relative_to(package).as_posix()
              for path in (package / "workspace/fonts").rglob("*") if path.is_file()}
    if not expected or set(expected) != actual:
        raise RuntimeError("Installed render font resource inventory differs from wheel")
    return expected


def run_tool(output, name, command, timeout=90):
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    (output / (name + ".stdout")).write_bytes(result.stdout)
    (output / (name + ".stderr")).write_bytes(result.stderr)
    write_json(output / (name + ".command.json"), {"argv": command, "exit_code": result.returncode})
    if result.returncode:
        raise RuntimeError(name + " failed; see recorded tool evidence")
    return result.stdout


def make_source(output):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe are required; media regression cannot be skipped")
    folder = output / "media"
    folder.mkdir()
    path = folder / "synthetic-stereo.mov"
    run_tool(folder, "encode", [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n",
        "-f", "lavfi", "-i", "testsrc2=duration=2:size=320x180:rate=10",
        "-f", "lavfi", "-i", "aevalsrc=0.5*sin(2*PI*997*t)|0.25*sin(2*PI*499*t):s=48000:d=2:c=stereo",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s24le", str(path)])
    probe = json.loads(run_tool(folder, "probe", [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"], video["nb_frames"]) == (320, 180, "20")
    assert audio["channels"] == 2 and audio["sample_rate"] == "48000"
    return {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size,
            "kind": "synthetic_two_second_stereo", "actual_video_frames": 20, "probe": probe}


def verify_origins(package):
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == "trace_annotate" or name.startswith("trace_annotate."):
            location = getattr(module, "__file__", None)
            if location is not None:
                path = Path(location).resolve()
                if package not in path.parents:
                    raise AssertionError("Product module escaped verified package: " + name)
                result[name] = str(path)
    return result


def verify_video(output, path, plan, face):
    """Independently decode ALL frames and compare both endpoint panels."""
    from PIL import Image, ImageChops, ImageStat
    from trace_annotate.workspace import render_layout as L
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    probe = json.loads(run_tool(output, "downloaded-probe", [ffprobe, "-v", "error", "-show_streams", "-show_frames",
        "-show_entries", "stream=index,codec_type,width,height,time_base,channels:frame=media_type,pts", "-of", "json", str(path)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    frames = [f for f in probe["frames"] if f["media_type"] == "video"]
    assert (video["width"], video["height"]) == tuple(plan["geometry"]["canvas"])
    assert len(frames) == int(plan["frame_count"]) == 20
    base = Fraction(video["time_base"])
    times = [int(f["pts"]) * base for f in frames]
    assert times == [Fraction(n, 10) for n in range(20)]
    assert next(s for s in probe["streams"] if s["codec_type"] == "audio")["channels"] == 2
    raw = run_tool(output, "downloaded-full-decode", [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-xerror",
        "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    width, height = plan["geometry"]["canvas"]
    stride = width * height * 3
    assert len(raw) == stride * 20
    x, y, w, h = plan["geometry"]["panel_rect"]
    checks = []
    for index in (0, 19):
        image = Image.frombytes("RGB", (width, height), raw[index*stride:(index+1)*stride])
        actual = image.crop((x, y, x+w, y+h))
        expected = L.render_panel(plan, L.panel_for_frame(plan, index), font=face)
        difference = ImageChops.difference(actual, expected)
        mae = sum(ImageStat.Stat(difference).mean) / 3
        assert mae < 15, (index, mae)
        assert actual.convert("L").getextrema()[1] > 220
        actual.save(output / ("downloaded-panel-frame-" + str(index) + ".png"))
        expected.save(output / ("expected-panel-frame-" + str(index) + ".png"))
        checks.append({"frame_index": index, "panel_mean_absolute_error": mae, "full_panel_compared": True})
    return {"decoded_frames": len(raw)//stride, "all_pts_exact": True, "stereo_audio_present": True, "endpoint_panels": checks}


def run_checks(output, source, harness, channel, summary):
    from playwright.sync_api import sync_playwright, expect
    from PIL import Image, ImageChops
    from trace_annotate.workspace import professional_schema as PS, rendering as R
    from trace_annotate.temporal import preparation as P
    store = harness.WorkspaceStore(output / "workspace")
    server = thread = playwright = browser = context = page = None
    original_render = R.render_video
    release = threading.Event()
    summary.update(checks=[], page_errors=[], external_requests=[], cleanup_errors=[], responses=[], downloads=[])

    def check(name, **evidence):
        summary["checks"].append({"name": name, "status": "PASS", **evidence})
        print("[CHECK] " + name + ": PASS", flush=True)
        write_json(output / "result.json", summary)

    def observe(target):
        target.on("pageerror", lambda error: summary["page_errors"].append(str(error)))
        target.on("request", lambda request: summary["external_requests"].append(request.url)
            if request.url.startswith(("http:", "https:")) and not request.url.startswith(server.origin + "/") else None)
        target.on("response", lambda response: summary["responses"].append({"path": response.url.split(server.origin, 1)[-1],
            "status": response.status, "method": response.request.method}))

    def save_segment(label, start, end, text):
        page.locator("#new-segment").click()
        page.locator("#segment-label").fill(label)
        page.locator("#segment-start").fill(str(start))
        page.locator("#segment-end").fill(str(end))
        page.locator("#segment-text").fill(text)
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")) as saved:
            page.locator("#save-segment").click()
        assert saved.value.status == 200
        expect(page.locator("#save-segment")).to_be_enabled(timeout=15000)

    def open_panel():
        if page.locator("#professional-render").get_attribute("open") is None:
            page.locator("#professional-render > summary").click()
        expect(page.locator("#render-preview")).to_be_enabled(timeout=15000)

    def submit():
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/render-jobs")) as accepted:
            page.locator("#render-submit").click()
        assert accepted.value.status == 202
        batch = accepted.value.json()
        assert len(batch["jobs"]) == 1
        body = accepted.value.request.post_data_json
        assert body["expected_revision"] == store.get_task(task_id)["revision"]
        assert body["expected_sha256"] == store.get_task(task_id)["content_sha256"]
        return batch["jobs"][0]["id"], body

    try:
        store.create()
        server = harness.make_server(store)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        playwright = sync_playwright().start()
        launch = {"headless": True, "downloads_path": str(output / "download-staging"), "args": ["--disk-cache-dir=" + str(output / "browser-cache")]}
        if channel:
            launch["channel"] = channel
        browser = playwright.chromium.launch(**launch)
        summary["browser"] = {"engine": "chromium", "version": browser.version, "channel": channel, "headless": True,
                              "playwright_version": importlib.metadata.version("playwright")}
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1000})
        context.on("page", observe)
        page = context.new_page()
        page.goto(server.editor_url)
        expect(page.locator("#empty-workspace")).to_be_visible()
        page.locator("#file-input").set_input_files(source["path"])
        expect(page.locator("#import-status")).to_contain_text("已入库 1", timeout=90000)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(1, timeout=90000)
        intake = store.list_intakes()[0]
        assert intake["state"] == "succeeded"
        task_id = intake["task_id"]
        page.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(page.locator("#new-segment")).to_be_enabled(timeout=30000)
        media = store.get_media(store.get_task(task_id)["media_id"])
        assert media["source"]["sha256"] == source["sha256"]
        save_segment("开场画面", 0, 1, "完整正文：保留蓝绿色画面和节奏，不截字。")
        save_segment("结尾画面", 1, 2, "完整结尾：最后一帧仍显示此段标注。")
        frozen = store.get_task(task_id)
        assert frozen["revision"] == 2 and len(frozen["annotations"]) == 2
        assert all(a["review_status"] == "unreviewed" for a in frozen["annotations"])
        check("Actual upload and two manually entered synthetic annotations preserve source identity and unreviewed status", task_id=task_id)
        open_panel()
        expect(page.locator("#render-font-info")).to_contain_text("SIL Open Font License 1.1")
        page.locator("#render-panel-extent").fill("128")
        page.locator("#render-body-font").fill("14")
        page.locator("#render-title-font").fill("16")
        page.locator("#render-padding").fill("8")
        page.locator("#render-line-spacing").fill("2")
        for field in ("interval", "track", "tags", "review_status"):
            page.locator("#render-field-" + field).uncheck()
        for field in ("label", "text"):
            page.locator("#render-field-" + field).check()
        previews = []
        for position in ("bottom", "top", "left", "right"):
            page.locator("#render-position").select_option(position)
            expect(page.locator("#render-preview-image")).to_be_hidden()
            with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/render-preview")) as preview:
                page.locator("#render-preview").click()
            response = preview.value
            assert response.status == 200, response.text()
            assert response.headers["x-traceannotate-task-revision"] == "2"
            assert response.headers["x-traceannotate-content-sha256"] == frozen["content_sha256"]
            expect(page.locator("#render-preview-image")).to_be_visible(timeout=15000)
            expect(page.locator("#render-preview-help")).to_contain_text("帧 #0")
            chosen = response.request.post_data_json
            assert chosen["frame_index"] == "0" and chosen["recipe"]["position"] == position
            assert chosen["recipe"]["body_font_px"] == 14 and chosen["recipe"]["title_font_px"] == 16
            assert chosen["recipe"]["fields"] == ["label", "text"]
            png = output / ("preview-" + position + ".png")
            png.write_bytes(response.body())
            plan = server.render_jobs.plan(task_id, expected_revision=2, expected_sha256=frozen["content_sha256"], recipe=chosen["recipe"])
            with Image.open(png) as image:
                assert image.size == tuple(plan["geometry"]["canvas"])
                x, y, w, h = plan["geometry"]["panel_rect"]
                panel = image.crop((x, y, x+w, y+h)).convert("RGB")
                red, green, blue = panel.split()
                assert ImageChops.difference(red, green).getbbox() is None
                assert ImageChops.difference(red, blue).getbbox() is None
                assert panel.getextrema()[0] == (0, 255)
            previews.append({"position": position, "path": png.name, "sha256": digest(png), "geometry": plan["geometry"]})
        assert store.get_task(task_id) == frozen and server.queue.list_batches() == []
        check("Four position selectors produce actual black-and-white PNG previews with custom fonts and no saved mutations", previews=previews)

        # Invalid dense text is refused visibly, never silently shrunk or cut.
        page.locator("#render-position").select_option("bottom")
        page.locator("#render-panel-extent").fill("64")
        page.locator("#render-body-font").fill("96")
        page.locator("#render-title-font").fill("128")
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/render-preview")) as overflow:
            page.locator("#render-preview").click()
        assert overflow.value.status == 422 and overflow.value.json()["error"]["code"] == "RENDER_TEXT_OVERFLOW"
        expect(page.locator("#render-preview-help")).to_contain_text("未截字或自动缩小")
        expect(page.locator("#render-preview-image")).to_be_hidden()
        assert store.get_task(task_id) == frozen and server.queue.list_batches() == []
        check("Text overflow is shown explicitly with no truncation, automatic shrinking or queue mutation")
        page.locator("#render-panel-extent").fill("128")
        page.locator("#render-body-font").fill("14")
        page.locator("#render-title-font").fill("16")
        job_id, request_body = submit()
        expect(page.locator("#render-jobs button", has_text="下载视频")).to_be_visible(timeout=120000)
        expect(page.locator("#render-jobs")).to_contain_text("当前已保存修订的工件")
        result = server.render_jobs.result(job_id)
        assert result["input_binding"] == result["current_binding"]
        assert result["input_binding"]["revision"] == 2
        assert result["manifest"]["renderer"]["validation"]["full_decode_exit"] == 0
        assert result["manifest"]["renderer"]["validation"]["all_frame_pts_exact"] is True
        manifest_path = store.root / result["manifest"]["files"]["render-plan.json"]["relative_path"]
        plan = json.loads(manifest_path.read_bytes())
        for annotation in frozen["annotations"]:
            for field in ("label", "text"):
                pieces = [entry["text"] for panel in plan["panels"].values() for entry in panel["entries"]
                          if entry["annotation_id"] == annotation["id"] and entry["source_field"] == field]
                assert "".join(pieces) == annotation[field]
        write_json(output / "render-result.json", result)
        write_json(output / "render-plan.json", plan)
        for kind, label, filename in (("video", "下载视频", "downloaded.annotated.mp4"), ("json", "下载配对JSON", "downloaded.professional.json")):
            with page.expect_download() as downloaded:
                page.locator("#render-jobs button", has_text=label).click()
            target = output / filename
            downloaded.value.save_as(target)
            assert target.stat().st_size > 0
            summary["downloads"].append({"kind": kind, "path": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
                                         "suggested_filename": downloaded.value.suggested_filename})
        document = PS.decode_document((output / "downloaded.professional.json").read_bytes())
        assert document["task"] == frozen and document["history"]["scope"] == "not_included"
        assert digest(output / "downloaded.annotated.mp4") == result["manifest"]["files"]["annotation.mp4"]["sha256"]
        verification = verify_video(output, output / "downloaded.annotated.mp4", plan, server.render_jobs.trusted_font())
        check("Actual rendered MP4 and paired professional JSON downloads preserve full text, exact frozen input and all frames including the end", job_id=job_id,
              verification=verification, frozen_revision=2, request_body=request_body)

        # A later real editor save must not relabel the earlier download as new.
        page.locator("#segment-text").fill("稍后修改：旧视频不包含当前修订。")
        expect(page.locator("#render-preview")).to_be_disabled()
        expect(page.locator("#render-submit")).to_be_disabled()
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")) as later:
            page.locator("#save-segment").click()
        assert later.value.status == 200
        expect(page.locator("#render-jobs")).to_contain_text("旧修订工件；不含当前修改", timeout=15000)
        page.locator("#render-jobs button", has_text="下载配对JSON").click()
        expect(page.locator("#confirm-dialog")).to_be_visible()
        expect(page.locator("#confirm-dialog")).to_contain_text("固定于输入 r2")
        expect(page.locator("#confirm-dialog")).to_contain_text("不包含当前修改")
        page.locator("#confirm-cancel").click()
        assert PS.decode_document(server.render_jobs.frozen_document(job_id)["data"])["task"] == frozen
        check("Unsaved draft blocks preview/render and later saved edits require explicit old-revision download confirmation")

        # Deliberate test-only barrier before encoding gives deterministic UI
        # cancellation, without faking a terminal state or subprocess evidence.
        reached = threading.Event()
        first = True
        def before_encoder(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                reached.set()
                if not release.wait(30):
                    raise RuntimeError("Regression pre-encoder cancellation barrier timed out")
            return original_render(*args, **kwargs)
        R.render_video = before_encoder
        page.locator("#render-position").select_option("top")
        cancelled_id, _ = submit()
        assert reached.wait(15)
        expect(page.locator("#render-jobs button", has_text="取消渲染")).to_be_visible(timeout=15000)
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/jobs/{cancelled_id}/control")) as cancelled:
            page.locator("#render-jobs button", has_text="取消渲染").click()
        assert cancelled.value.status == 200 and cancelled.value.request.post_data_json["action"] == "cancel"
        release.set()
        expect(page.locator("#render-jobs button", has_text="重新导出")).to_be_visible(timeout=30000)
        assert server.queue.get_job(cancelled_id)["state"] == "cancelled"
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/jobs/{cancelled_id}/control")) as resumed:
            page.locator("#render-jobs button", has_text="重新导出").click()
        assert resumed.value.status == 200 and resumed.value.request.post_data_json["action"] == "retry_cancelled"
        expect(page.locator("#render-jobs button", has_text="下载视频")).to_have_count(2, timeout=120000)
        assert server.queue.get_job(cancelled_id)["state"] == "succeeded"
        assert server.queue.get_job(cancelled_id)["attempt"] == 2
        render_ids = {job["id"] for batch in server.queue.list_batches() for job in batch["jobs"] if job["operation"] == "render-annotations"}
        assert render_ids == {job_id, cancelled_id}
        summary["cancellation_tested"] = "controlled_pre_encoder_real_ui_cancel_and_resume"
        check("Real cancel and resume buttons reuse one frozen job after a controlled pre-encoder cancellation", job_id=cancelled_id,
              owned_live_encoder_cancellation_tested=False, attempt=2)
        page.reload()
        page.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        open_panel()
        expect(page.locator("#render-jobs button", has_text="下载视频")).to_have_count(2, timeout=15000)
        expect(page.locator("#render-jobs")).to_contain_text("旧修订工件")
        page.screenshot(path=str(output / "render-downloads-after-reload.png"), full_page=True)
        check("Refresh preserves both verified render jobs and their exact revision labels")
        if summary["page_errors"] or summary["external_requests"]:
            raise AssertionError("Unexpected browser script errors or external network requests")
        summary["module_origins"] = verify_origins(harness.PACKAGE)
    except BaseException:
        if page is not None:
            try:
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                (output / "failure-dom.html").write_text(page.content().replace(server.token, "REDACTED_EPHEMERAL_TOKEN"), encoding="utf-8")
            except Exception as error:
                summary["failure_capture_error"] = str(error)
        raise
    finally:
        release.set()
        R.render_video = original_render
        cleanups = []
        if context is not None:
            cleanups.append(("browser-context", context.close))
        if browser is not None:
            cleanups.append(("browser", browser.close))
        if playwright is not None:
            cleanups.append(("playwright", playwright.stop))
        if server is not None:
            if thread is not None and thread.is_alive():
                cleanups.append(("server-shutdown", server.shutdown))
            cleanups.append(("server-close", server.server_close))
        if thread is not None:
            cleanups.append(("server-thread", lambda: thread.join(timeout=10)))
        cleanups.append(("workspace-store", store.close))
        for name, action in cleanups:
            try:
                action()
            except Exception as error:
                summary["cleanup_errors"].append({"resource": name, "error": str(error)})
        if server is not None and hasattr(server, "shutdown_report"):
            summary["shutdown_report"] = server.shutdown_report
        if thread is not None and thread.is_alive():
            summary["cleanup_errors"].append({"resource": "server-thread", "error": "Still alive after shutdown"})
        summary["owned_media_children_after"] = P.live_media_processes()
        if summary["owned_media_children_after"]:
            summary["cleanup_errors"].append({"resource": "media-processes", "error": "Owned media children remain"})
        if summary["cleanup_errors"]:
            raise RuntimeError("Regression resource cleanup failed; see result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--installed-wheel", type=Path, help="Require -I and exact installed package bytes from this wheel")
    parser.add_argument("--channel", help="Optional installed Chromium channel; remains an independent headless process")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("Output must be a new directory; existing evidence preserved: " + str(error))
    summary = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(), "argv": [sys.executable] + sys.argv,
        "python": sys.version, "installed_identity": None, "headless": True, "native_file_picker_tested": False,
        "cancellation_tested": False, "scope": "Independent automated headless Chromium: synthetic media import and edit, four real single-frame previews, custom font sizes and overflow refusal, actual render and paired downloads, all-frame decoding and endpoint panels, frozen-revision warning, controlled pre-encoder cancel/resume and reload. Not native user Chrome, live encoder cancellation, semantic annotation accuracy, human listening, final screen recording or release readiness."}
    write_json(output / "result.json", summary)
    previous = {key: os.environ.get(key) for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME", "PLAYWRIGHT_BROWSERS_PATH")}
    bytecode = sys.dont_write_bytecode
    try:
        for name in ("temp", "cache", "browser-cache", "download-staging"):
            (output / name).mkdir()
        for key in ("TEMP", "TMP", "TMPDIR"):
            os.environ[key] = str(output / "temp")
        summary["browser_installation"] = preserve_browser_installation()
        os.environ["XDG_CACHE_HOME"] = str(output / "cache")
        sys.dont_write_bytecode = True
        harness = load_harness(args.installed_wheel)
        files = [Path(__file__).resolve(), SOURCE / "scripts/check_workspace_editor.py"] + sorted(
            p for p in harness.PACKAGE.rglob("*") if p.is_file() and p.suffix in {".py", ".html", ".js", ".otf", ".txt", ".json"})
        identities = {str(p): digest(p) for p in files}
        summary.update(source_sha256=identities, installed_identity=harness.INSTALLED_IDENTITY)
        source = make_source(output)
        summary["fixture"] = source
        write_json(output / "result.json", summary)
        run_checks(output, source, harness, args.channel, summary)
        if any(digest(path) != expected for path, expected in identities.items()):
            raise AssertionError("Implementation changed during regression; rerun against frozen bytes")
        if digest(source["path"]) != source["sha256"]:
            raise AssertionError("Synthetic source bytes changed during regression")
        if harness.INSTALLED_IDENTITY and digest(harness.INSTALLED_IDENTITY["wheel"]) != harness.INSTALLED_IDENTITY["wheel_sha256"]:
            raise AssertionError("Installed wheel changed during regression")
        summary["status"] = "PASS"
    except BaseException as error:
        summary.update(status="INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "FAIL", error=str(error) or type(error).__name__)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("Error: " + summary["error"] + "; evidence: " + str(output), file=sys.stderr)
    finally:
        sys.dont_write_bytecode = bytecode
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "result.json", summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
