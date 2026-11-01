"""Click the real analysis panel against source or an isolated installed wheel.

Requires a NEW output directory. Uses generated two-second stereo media, real
admission/FFmpeg analysis and an actual Chromium JSON download. It does not
certify human listening, semantic labels, native Chrome or cancellation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

SOURCE = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


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
    # The existing harness checks -I, site-packages membership, exact package
    # resources and extra/missing modules. Only test support loads by file.
    if installed_wheel is not None and not sys.flags.isolated:
        raise RuntimeError("Installed acceptance requires the target venv Python with -I")
    support = SOURCE / "scripts" / "check_workspace_editor.py"
    spec = importlib.util.spec_from_file_location("traceannotate_analysis_acceptance_support", support)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(output):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe are required; actual analysis cannot be skipped")
    folder = output / "media"
    folder.mkdir()
    source = folder / "synthetic-stereo.mov"
    command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
        "testsrc2=duration=2:size=160x90:rate=10", "-f", "lavfi", "-i",
        "aevalsrc=0.5*sin(2*PI*997*t)|0.25*sin(2*PI*499*t):s=48000:d=2:c=stereo",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s24le", str(source)]
    process = subprocess.run(command, capture_output=True, timeout=60)
    (folder / "encode.stdout").write_bytes(process.stdout)
    (folder / "encode.stderr").write_bytes(process.stderr)
    write_json(folder / "encode-command.json", {"argv": command, "exit_code": process.returncode})
    if process.returncode:
        raise RuntimeError("Synthetic stereo encoding failed; see media/encode.stderr")
    probe_command = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)]
    probe = subprocess.run(probe_command, capture_output=True, timeout=30)
    (folder / "probe.stdout").write_bytes(probe.stdout)
    (folder / "probe.stderr").write_bytes(probe.stderr)
    write_json(folder / "probe-command.json", {"argv": probe_command, "exit_code": probe.returncode})
    if probe.returncode:
        raise RuntimeError("Independent synthetic media probe failed")
    metadata = json.loads(probe.stdout)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    assert video["width"] == 160 and video["height"] == 90 and video["nb_frames"] == "20"
    assert audio["sample_rate"] == "48000" and audio["channels"] == 2 and audio["channel_layout"] == "stereo"
    return {"path": str(source), "sha256": digest(source), "size_bytes": source.stat().st_size,
            "kind": "synthetic_two_second_stereo", "actual_video_frames": 20,
            "sample_rate": 48000, "channels": 2, "probe": metadata}


def verify_module_origins(package):
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name == "trace_annotate" or name.startswith("trace_annotate."):
            location = getattr(module, "__file__", None)
            if location is None:
                continue
            path = Path(location).resolve()
            if package not in path.parents:
                raise AssertionError("Product module escaped the verified package: " + name)
            origins[name] = str(path)
    return origins


def run_checks(output, source, harness, channel, summary):
    from playwright.sync_api import sync_playwright, expect
    from trace_annotate.workspace import professional_schema as PS
    from trace_annotate.temporal import preparation as P

    store = harness.WorkspaceStore(output / "workspace")
    server = thread = playwright = browser = context = page = None
    summary.update(checks=[], page_errors=[], external_requests=[], cleanup_errors=[], responses=[], downloads=[])

    def check(name, **evidence):
        summary["checks"].append({"name": name, "status": "PASS", **evidence})
        print("[CHECK] " + name + ": PASS", flush=True)
        write_json(output / "result.json", summary)

    def observe(target):
        target.on("pageerror", lambda error: summary["page_errors"].append(str(error)))
        target.on("request", lambda request: summary["external_requests"].append(request.url)
                  if request.url.startswith(("http:", "https:")) and not request.url.startswith(server.origin + "/") else None)
        target.on("response", lambda response: summary["responses"].append({
            "path": response.url.split(server.origin, 1)[-1], "status": response.status,
            "method": response.request.method}))

    def open_analysis():
        panel = page.locator("#professional-analysis")
        if panel.get_attribute("open") is None:
            page.locator("#professional-analysis > summary").click()
        expect(page.locator("#analysis-start")).to_be_visible()

    try:
        store.create()
        server = harness.make_server(store)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        playwright = sync_playwright().start()
        launch = {"headless": True, "downloads_path": str(output / "download-staging"),
                  "args": ["--disk-cache-dir=" + str(output / "browser-cache")]}
        if channel:
            launch["channel"] = channel
        browser = playwright.chromium.launch(**launch)
        summary["browser"] = {"engine": "chromium", "version": browser.version,
                              "channel": channel, "headless": True,
                              "playwright_version": importlib.metadata.version("playwright")}
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1000})
        context.on("page", observe)
        page = context.new_page()
        page.goto(server.editor_url)
        expect(page.locator("#empty-workspace")).to_be_visible()
        expect(page.locator("#file-input")).to_be_enabled()
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
        assert media["source"]["sha256"] != media["playback"]["sha256"]
        check("Real UI upload admits the exact stereo source with a distinct playback identity", task_id=task_id, media=media)

        # A synthetic manual label establishes that measuring audio does not
        # automatically review annotations or replace existing editor content.
        page.locator("#new-segment").click()
        page.locator("#segment-label").fill("Synthetic analysis regression")
        page.locator("#segment-start").fill("0")
        page.locator("#segment-end").fill("1")
        page.locator("#segment-text").fill("Synthetic fixture only; not semantic or listening evidence.")
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith(f"/tasks/{task_id}/commands")) as saved:
            page.locator("#save-segment").click()
        assert saved.value.status == 200
        expect(page.locator("#save-segment")).to_be_enabled(timeout=15000)
        before = store.get_task(task_id)
        assert before["revision"] == 1 and len(before["annotations"]) == 1
        assert before["annotations"][0]["review_status"] == "unreviewed"
        open_analysis()
        expect(page.locator("#analysis-status")).to_contain_text("已保存 0 份报告", timeout=15000)
        expect(page.locator("#professional-analysis")).to_contain_text("原片报告与代理报告分别标明来源")
        check("Original annotation controls and source-versus-proxy explanation remain available")

        with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith(f"/tasks/{task_id}/analysis-jobs")) as started:
            page.locator("#analysis-start").click()
        assert started.value.status == 202
        batch = started.value.json()
        assert len(batch["jobs"]) == 1
        job_id = batch["jobs"][0]["id"]
        request_body = started.value.request.post_data_json
        assert request_body["expected_revision"] == before["revision"]
        assert request_body["expected_sha256"] == before["content_sha256"]
        expect(page.locator("#analysis-jobs button", has_text="保存结果到标注")).to_be_visible(timeout=90000)
        expect(page.locator("#analysis-jobs")).to_contain_text("分析完成，结果为候选")
        assert store.get_task(task_id) == before
        candidate = server.analysis_jobs.result(job_id)
        assert candidate["time_domain"] == "playback_seconds"
        assert candidate["file_sha256"] == media["playback"]["sha256"]
        assert candidate["semantic_listening"] == "not_performed"
        assert "original_source_not_measured" in candidate["scope"]
        assert len(candidate["reports"]) == 5
        kinds = [report["kind"] for report in candidate["reports"]]
        assert kinds.count("technical_declared") == 3 and kinds.count("frame_timing") == 1 and kinds.count("loudness") == 1
        for report in candidate["reports"]:
            PS.validate_analysis(report, media)
            assert report["file_sha256"] == media["playback"]["sha256"] and report["time_domain"] == "playback_seconds"
        loudness = next(report for report in candidate["reports"] if report["kind"] == "loudness")
        measurements = {v["name"]: v for v in loudness["values"]}
        assert measurements["sample_peak_dbfs"]["status"] == "available"
        assert measurements["integrated_lufs"]["status"] == "available"
        assert measurements["channels"]["value"] == 2
        write_json(output / "candidate.json", candidate)
        page.screenshot(path=str(output / "candidate-not-applied.png"), full_page=True)
        check("Visible start click completes actual playback measurement as five unsaved candidates", job_id=job_id,
              frozen_input_revision=candidate["input_revision"], frozen_input_sha256=candidate["input_sha256"])

        before_revisions = store.conn.execute("SELECT COUNT(*) FROM task_revisions WHERE task_id=?", (task_id,)).fetchone()[0]
        page.locator("#analysis-jobs button", has_text="保存结果到标注").click()
        expect(page.locator("#confirm-dialog")).to_be_visible()
        expect(page.locator("#confirm-dialog")).to_contain_text("不会给标注勾选人工审核")
        expect(page.locator("#confirm-dialog")).to_contain_text("不代表听过旁白或确认音色")
        with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith(f"/jobs/{job_id}/apply-analysis")) as applied:
            page.locator("#confirm-accept").click()
        assert applied.value.status == 200
        receipt = applied.value.json()
        assert receipt["previous_revision"] == before["revision"] and receipt["revision"] == before["revision"] + 1
        assert receipt["report_count"] == 5
        expect(page.locator("#analysis-status")).to_contain_text("已保存 5 份报告", timeout=15000)
        expect(page.locator("#analysis-jobs")).to_contain_text("报告已保存")
        after = store.get_task(task_id)
        assert after["annotations"] == before["annotations"]
        assert after["extensions"]["professional"]["analysis_reports"] == candidate["reports"]
        assert store.conn.execute("SELECT COUNT(*) FROM task_revisions WHERE task_id=?", (task_id,)).fetchone()[0] == before_revisions + 1
        assert store.get_command_receipt(receipt["command_id"])["response"] == receipt
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        expect(page.locator("#analysis-reports > details")).to_have_count(5)
        for report_summary in page.locator("#analysis-reports > details > summary").all():
            expect(report_summary).to_contain_text("播放代理")
        write_json(output / "saved-receipt.json", receipt)
        check("Explicit confirmation saves all five reports in one revision without granting manual review", receipt=receipt)

        with page.expect_download() as downloaded:
            page.locator("#export-professional-btn").click()
        target = output / "downloaded.professional.json"
        downloaded.value.save_as(target)
        document = PS.decode_document(target.read_bytes())
        assert document["task"] == after
        assert document["analyses"] == candidate["reports"]
        assert document["media"]["source"]["sha256"] == source["sha256"]
        assert all(item["status"] == "fresh" for item in document["analysis_assessment"])
        summary["downloads"].append({"path": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
                                     "suggested_filename": downloaded.value.suggested_filename})
        check("Actual browser professional JSON download validates all reports and exact saved task")

        page.reload()
        page.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(page.locator("#new-segment")).to_be_enabled(timeout=15000)
        open_analysis()
        expect(page.locator("#analysis-status")).to_contain_text("已保存 5 份报告", timeout=15000)
        expect(page.locator("#analysis-reports > details")).to_have_count(5)
        assert store.get_task(task_id) == after
        assert store.get_command_receipt(receipt["command_id"])["state"] == "committed"
        analysis_jobs = {job["id"] for batch in server.queue.list_batches() for job in batch["jobs"]
                         if job["operation"] == "analyze-media"}
        assert len(analysis_jobs) == 1
        page.screenshot(path=str(output / "saved-after-reload.png"), full_page=True)
        check("Page reload preserves reports and the original annotation without creating another job or revision")

        for selector, input_selector in (("#open-general-tool", "#videoInput"), ("#open-embodied-tool", "#fileInput")):
            expect(page.locator(selector)).to_be_visible()
            with page.expect_popup() as opened:
                page.locator(selector).click()
            classic = opened.value
            expect(classic.locator(input_selector)).to_be_attached(timeout=15000)
            assert classic.url.startswith(server.origin + "/tools/")
            classic.close()
        check("Both existing classic tool links open their actual local editor pages")
        if summary["page_errors"] or summary["external_requests"]:
            raise AssertionError("Unexpected browser script errors or external network requests")
        summary["module_origins"] = verify_module_origins(harness.PACKAGE)
    except BaseException:
        if page is not None:
            try:
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                html = page.content().replace(server.token, "REDACTED_EPHEMERAL_TOKEN")
                (output / "failure-dom.html").write_text(html, encoding="utf-8")
            except Exception as error:
                summary["failure_capture_error"] = str(error)
        raise
    finally:
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
            raise RuntimeError("Acceptance resource cleanup failed; see result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--installed-wheel", type=Path, help="Require -I and exact installed package bytes from this wheel")
    parser.add_argument("--channel", help="Optional installed Chromium channel; remains headless")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("Output must be a new directory; existing evidence preserved: " + str(error))
    summary = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [sys.executable] + sys.argv, "python": sys.version, "installed_identity": None,
        "headless": True, "native_file_picker_tested": False, "cancellation_tested": False,
        "scope": "Real headless Chromium clicks, synthetic stereo admission and playback analysis, explicit atomic report save, professional JSON download and page reload. Existing classic links are opened, not exhaustively retested. Not native user Chrome, source-file job analysis, human listening/timbre, semantic annotation accuracy, cancellation coverage or release readiness."}
    write_json(output / "result.json", summary)
    # Browser profiles, downloads and FFmpeg/analysis outputs stay inside this
    # new evidence root. Existing browser executables are read-only dependencies.
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
            p for p in harness.PACKAGE.rglob("*") if p.is_file() and p.suffix in {".py", ".html", ".js"})
        identities = {str(p): digest(p) for p in files}
        summary.update(source_sha256=identities, installed_identity=harness.INSTALLED_IDENTITY)
        source = make_source(output)
        summary["fixture"] = source
        write_json(output / "result.json", summary)
        run_checks(output, source, harness, args.channel, summary)
        if any(digest(path) != expected for path, expected in identities.items()):
            raise AssertionError("Implementation changed during acceptance; rerun against frozen bytes")
        if digest(source["path"]) != source["sha256"]:
            raise AssertionError("Synthetic source bytes changed during acceptance")
        if harness.INSTALLED_IDENTITY and digest(harness.INSTALLED_IDENTITY["wheel"]) != harness.INSTALLED_IDENTITY["wheel_sha256"]:
            raise AssertionError("Installed wheel changed during acceptance")
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
