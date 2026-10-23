"""Exercise compatibility preparation and annotation in an isolated browser.

Creates only synthetic ProRes/AAC and variable-rate sources in a new output
directory. Each is prepared, served on loopback, played, annotated, downloaded,
restored from a draft and reopened in a fresh browser context. The test labels
are operation fixtures; this does not evaluate automatic semantic recognition.
Requires FFmpeg/ffprobe and the browser-test extra with an installed browser.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from trace_annotate.temporal.formats import load_document
from trace_annotate.temporal.preparation import prepare_media
from trace_annotate.temporal.project_server import make_project_server


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(folder, case):
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg must be installed and available on PATH")
    source = folder / ("input.mov" if case == "prores-aac" else "input.mp4")
    command = [executable, "-hide_banner", "-v", "error", "-nostdin", "-n",
               "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=60:duration=2"]
    if case == "prores-aac":
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-r", "15", "-c:v", "prores_ks", "-profile:v", "2",
                    "-pix_fmt", "yuv422p10le", "-c:a", "aac"]
    else:
        command += ["-vf", "select=if(lt(t\\,1)\\,not(mod(n\\,6))\\,not(mod(n\\,2)))",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p"]
    command += [str(source)]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    write_json(folder / "generation.json", {"command": command, "exit_code": result.returncode,
                                            "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError("Synthetic media creation failed; see generation.json")
    return source


def exercise(browser, folder, source, expect):
    before = digest(source)
    manifest = prepare_media(source, folder / "project")
    project = folder / "project"
    original = load_document(project / "annotations.json")
    report = {"case": folder.name, "status": "RUNNING", "source_sha256": before,
              "playback_sha256": digest(project / "playback.mp4"), "manifest": manifest,
              "checks": [], "downloads": [], "page_errors": [], "external_requests": [],
              "responses": [], "headless": True, "user_browser_connected": False,
              "native_file_picker_tested": False}
    server = make_project_server(project)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1080})
    page = context.new_page()

    def observe(target):
        target.on("dialog", lambda dialog: dialog.accept())
        target.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        target.on("request", lambda request: report["external_requests"].append(request.url)
                  if request.url.startswith(("http:", "https:"))
                  and not request.url.startswith(f"http://127.0.0.1:{server.server_port}/") else None)
        target.on("response", lambda response: report["responses"].append({
            "resource": response.url.rsplit("/", 1)[-1], "status": response.status,
            "content_range": response.headers.get("content-range")}))

    def ready(target):
        target.wait_for_function("!document.querySelector('#editorFields').disabled && "
                                 "document.querySelector('video').readyState >= 2", timeout=45000)

    def download(target, name, expected=None):
        with target.expect_download() as pending:
            target.locator("#exportJSON").click()
        path = folder / (name + ".json")
        pending.value.save_as(path)
        document = load_document(path)
        report["downloads"].append({"path": str(path), "sha256": digest(path)})
        if expected is not None and document != expected:
            raise AssertionError(name + ": downloaded native JSON changed")
        return document, path

    observe(page)
    try:
        page.goto(server.editor_url)
        ready(page)
        download(page, "01-auto-loaded", original)
        page.locator("video").evaluate(
            "async v => { v.muted=true; v.playbackRate=1; v.currentTime=0; await v.play(); }")
        page.wait_for_function("document.querySelector('video').ended",
                               timeout=int(original["duration"] * 1000 + 20000))
        played = page.locator("video").evaluate(
            "v => ({duration:v.duration, ended:v.ended, error:v.error?"
            "{code:v.error.code,message:v.error.message}:null, width:v.videoWidth,height:v.videoHeight,"
            "total:v.getVideoPlaybackQuality().totalVideoFrames,"
            "dropped:v.getVideoPlaybackQuality().droppedVideoFrames})")
        if played["error"] or not played["ended"] or abs(played["duration"] - original["duration"]) > 0.05:
            raise AssertionError("Prepared copy did not play to its declared endpoint")
        report["played_to_end"] = played
        views = []
        for name, fraction in (("start", 0), ("middle", 0.5), ("tail", 0.99)):
            requested = original["duration"] * fraction
            page.locator("#seek").evaluate(
                "(element,t) => {element.value=String(t);element.dispatchEvent(new Event('input',{bubbles:true}));}", requested)
            page.wait_for_function("!document.querySelector('video').seeking && "
                                   "document.querySelector('video').readyState >= 2")
            page.locator("video").screenshot(path=str(folder / (name + ".png")))
            views.append({"requested_seconds": requested,
                          "player_seconds": page.locator("video").evaluate("v => v.currentTime"),
                          "frame_feedback": page.locator("#frameInfo").inner_text()})
        report["seek_views"] = views
        report["checks"].append("Automatic project loading, 1x playback to ended, start/middle/tail seconds-based seeks")
        page.locator("#newSegment").click()
        page.locator("#startInput").fill("0.1")
        page.locator("#endInput").fill("1")
        page.locator("#labelInput").fill("Compatibility acceptance fixture")
        page.locator("#textInput").fill("Operation test only; not a semantic video annotation.")
        page.locator("#trackInput").fill("Acceptance")
        page.locator("#tagsInput").fill('["synthetic-fixture"]')
        page.locator("#saveSegment").click()
        expect(page.locator("#formError")).to_be_empty()
        document, native = download(page, "02-created-segment")
        if len(document["segments"]) != 1 or document["segments"][0]["review_status"] != "unreviewed":
            raise AssertionError("Created segment or review status is incorrect")
        for key in ("metadata", "duration", "media_size", "media_sha256", "clip"):
            if document[key] != original[key]:
                raise AssertionError("Segment creation changed project field: " + key)
        page.wait_for_function("id => Object.values(localStorage).some(v => v.includes(id))",
                               arg=document["segments"][0]["id"])
        page.reload()
        ready(page)
        expect(page.locator("#restoreDraft")).to_be_visible(timeout=15000)
        page.locator("#restoreDraft").click()
        download(page, "03-restored-draft", document)
        marker = "Unsaved form restoration acceptance fixture"
        page.locator("#textInput").fill(marker)
        page.wait_for_function("marker => Object.values(localStorage).some(v => v.includes(marker))", arg=marker)
        page.reload()
        ready(page)
        expect(page.locator("#restoreDraft")).to_be_visible(timeout=15000)
        page.locator("#restoreDraft").click()
        expect(page.locator("#textInput")).to_have_value(marker)
        page.locator("#discardForm").click()
        download(page, "04-restored-form-discarded", document)
        page.locator("#jsonInput").set_input_files(str(native))
        expect(page.locator("#status")).to_contain_text("导入 JSON")
        download(page, "05-reimported-native", document)
        fresh = browser.new_context(accept_downloads=True)
        try:
            reopened = fresh.new_page()
            observe(reopened)
            reopened.goto(server.editor_url)
            ready(reopened)
            reopened.locator("#jsonInput").set_input_files(str(native))
            expect(reopened.locator("#segmentList li")).to_have_count(1)
            download(reopened, "06-clean-context-reopen", document)
        finally:
            fresh.close()
        page.screenshot(path=str(folder / "workspace-final.png"), full_page=True)
        report["checks"].append("Created unreviewed segment; exact metadata/media binding; downloaded JSON; committed and unsaved-form draft restoration; real file-input reimport; clean-context reopen")
        if report["page_errors"] or report["external_requests"]:
            raise AssertionError("Unexpected page errors or external network requests")
        if not any(item["resource"] == "playback.mp4" and item["status"] == 206 for item in report["responses"]):
            raise AssertionError("Browser did not exercise HTTP Range playback")
        if digest(source) != before or digest(project / "playback.mp4") != report["playback_sha256"]:
            raise AssertionError("Source or playback copy changed during acceptance")
        report["source_unchanged"] = True
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        try:
            page.screenshot(path=str(folder / "failure.png"), full_page=True)
        except Exception:
            pass
        raise
    finally:
        write_json(folder / "report.json", report)
        context.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--channel", help="Optional installed browser channel, e.g. chrome; default: bundled Chromium")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error("Output must be a new directory; existing files were not changed: " + str(exc))
    source_files = [Path(__file__), SOURCE / "trace_annotate/temporal/preparation.py",
                    SOURCE / "trace_annotate/temporal/project_server.py", SOURCE / "trace_annotate/temporal/ui/annotator.html"]
    hashes = {str(path.relative_to(SOURCE)): digest(path) for path in source_files}
    summary = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable] + sys.argv, "python": sys.version,
               "channel": args.channel or "bundled-chromium", "source_sha256": hashes, "cases": [],
               "scope": "Synthetic compatibility workflow only; not semantic recognition or frame-accurate seeking certification."}
    try:
        from playwright.sync_api import sync_playwright, expect
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, **({"channel": args.channel} if args.channel else {}))
            summary["browser"] = browser.version
            try:
                for case in ("prores-aac", "true-vfr"):
                    folder = output / case
                    folder.mkdir()
                    source = generate(folder, case)
                    record = exercise(browser, folder, source, expect)
                    summary["cases"].append({"case": case, "status": record["status"], "report": str(folder / "report.json")})
                    print(case + ": PASS", flush=True)
            finally:
                browser.close()
        if any(digest(path) != hashes[str(path.relative_to(SOURCE))] for path in source_files):
            raise AssertionError("Implementation changed while acceptance was running; rerun against a fixed revision")
        summary["status"] = "PASS"
    except Exception as exc:
        summary.update(status="FAIL", error=str(exc))
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("Error: " + str(exc) + "; diagnostics: " + str(output), file=sys.stderr)
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
