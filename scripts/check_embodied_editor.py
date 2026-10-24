"""Record actual embodied-editor operations using a synthetic video fixture.

Creates a new evidence directory, launches isolated headless Chromium, fills
the real controls, downloads JSON, and runs the production QC command. Labels
are scripted operation fixtures, not independent human ground truth. No user
browser, profile, source video, or existing output is opened or replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_command(output, name, command, *, timeout=120, expected_exit=0):
    record = {"command": list(map(str, command)), "cwd": str(SOURCE),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        result = subprocess.run(command, cwd=SOURCE, capture_output=True, timeout=timeout)
        stdout, stderr = result.stdout, result.stderr
        record.update(exit_code=result.returncode, timed_out=False)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        record.update(exit_code=None, timed_out=True, timeout_seconds=timeout)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    for kind, value in (("stdout", stdout), ("stderr", stderr)):
        path = output / (name + "." + kind + ".log")
        with path.open("xb") as stream:
            stream.write(value)
        record[kind] = {"path": str(path), "sha256": digest(path)}
    write_json(output / (name + ".command.json"), record)
    if record["timed_out"] or record["exit_code"] != expected_exit:
        raise RuntimeError(name + " failed: " + ("timed out" if record["timed_out"] else stderr.decode("utf-8", "replace")))
    return stdout


def demo_font(size):
    from PIL import ImageFont
    for candidate in ("C:/Windows/Fonts/consola.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_synthetic_video(output):
    """Draw a controlled vector fixture, never pretend it is filmed hand work."""
    from PIL import Image, ImageDraw, ImageFont
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required on PATH")
    frames = output / "synthetic-frames"
    frames.mkdir()
    font, small = demo_font(22), demo_font(16)
    for frame in range(300):
        time = frame / 25
        image = Image.new("RGB", (1280, 720), "#10202f")
        draw = ImageDraw.Draw(image)
        for x in range(40, 1280, 60):
            draw.line((x, 0, x, 720), fill="#1a3043")
        for y in range(40, 720, 60):
            draw.line((0, y, 1280, y), fill="#1a3043")
        draw.text((45, 30), "SYNTHETIC OPERATION FIXTURE", font=font, fill="#d6eaff")
        draw.text((45, 64), "12 seconds | controlled shapes | not video ground truth", font=small, fill="#8faac5")
        draw.rounded_rectangle((360, 190, 950, 550), 18, fill="#243c50", outline="#52738e", width=3)
        progress = min(1, max(0, (time - 1) / 2))
        x = int(90 + 410 * progress)
        draw.rounded_rectangle((x, 285, x + 230, 420), 12, fill="#52d3db", outline="#9eeef0", width=3)
        for cx, cy in ((530, 310), (700, 310), (530, 395), (700, 395)):
            active = 5 <= time <= 7
            draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill="#ffc36b" if active else "#283e51", outline="#cad9df", width=2)
        if 9 <= time <= 11:
            draw.rounded_rectangle((478, 263, 752, 441), 15, outline="#7ee7a9", width=6)
        draw.text((45, 630), "01 POSITION     02 FASTEN     03 INSPECT", font=font, fill="#d8e6f5")
        draw.text((1000, 650), f"{time:05.2f} s", font=font, fill="#92c6dd")
        image.save(frames / f"{frame:04d}.png")
    media = output / "synthetic-component-demo.mp4"
    run_command(output, "generate-media", [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-framerate", "25", "-i", str(frames / "%04d.png"), "-an", "-c:v", "libx264",
                "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(media)])
    return media


def record_workflow(output, media):
    from playwright.sync_api import sync_playwright, expect
    ui = SOURCE / "trace_annotate/embodied/ui/annotator.html"
    before = digest(ui)
    report = {"status": "RUNNING", "scope": "Scripted synthetic-media operations; no independent human/video semantic certification",
              "headless": True, "user_browser_connected": False, "native_file_picker_tested": False,
              "ui_sha256": before, "input": {"path": str(media), "sha256": digest(media)},
              "operations": [], "checks": [], "page_errors": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    def operation(action, **data):
        report["operations"].append({"action": action, "at_utc": datetime.now(timezone.utc).isoformat(), **data})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        report["browser"] = browser.version
        context = browser.new_context(accept_downloads=True, viewport={"width": 1920, "height": 1080},
                                      record_video_dir=str(output / "recording"),
                                      record_video_size={"width": 1920, "height": 1080})
        page = context.new_page()
        recording = page.video
        accept = {"value": True}
        page.on("dialog", lambda dialog: dialog.accept() if accept["value"] else dialog.dismiss())
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        try:
            page.goto(ui.as_uri())
            operation("open_editor", url=ui.as_uri())
            page.wait_for_timeout(800)
            expect(page.locator("#exportJSON")).to_be_disabled()
            page.locator("#fileInput").set_input_files(media)
            operation("select_synthetic_video", input_method="actual file input through set_input_files")
            expect(page.locator("#exportJSON")).to_be_enabled(timeout=20000)
            page.wait_for_function("document.querySelector('video').readyState >= 2")
            assert page.locator("video").evaluate("v=>v.duration") == 12
            page.locator("#taskDescription").fill("合成操作演示：定位组件、紧固、检查。仅用于软件验收，不代表独立人工标注真值。")
            page.locator("#expectedSteps").fill("3")
            operation("fill_task_and_procedure", expected_step_count=3)
            def seek(seconds):
                page.locator("#seekSeconds").fill(str(seconds))
                page.locator("#seekTo").click()
                page.wait_for_function("t => !document.querySelector('video').seeking && Math.abs(document.querySelector('video').currentTime-t)<0.001", arg=seconds)
                operation("seek_seconds", requested=seconds, actual=page.locator("video").evaluate("v=>v.currentTime"))
                page.wait_for_timeout(400)
            for step, start, end, label in ((1, 1, 3, "position component"), (2, 5, 7, "fasten"), (3, 9, 11, "inspect")):
                seek(start)
                page.locator("#markStart").click()
                seek(end)
                page.locator("#markEnd").click()
                expect(page.locator("#selVerdict")).to_have_value("无法判断")
                page.locator("#segmentLabel").fill(label)
                page.locator("#segmentLabel").press("Tab")
                page.locator("#inpStep").fill(str(step))
                page.locator("#inpStep").press("Tab")
                page.locator("#selVerdict").select_option("对")
                operation("create_segment_and_explicitly_confirm", start=start, end=end, label=label, step_number=step, verdict="对")
                page.wait_for_timeout(1200)
            page.locator("#computeIdle").click()
            expect(page.locator("#workingDuration")).to_have_text("6.000s")
            expect(page.locator("#idleDuration")).to_have_text("6.000s")
            operation("compute_idle", expected_working_seconds=6, expected_idle_seconds=6)
            with page.expect_download() as pending:
                page.locator("#exportJSON").click()
            native = output / "actual-ui-download.json"
            pending.value.save_as(native)
            document = json.loads(native.read_text(encoding="utf-8"))
            assert [(s["start"], s["end"], s["step_number"], s["label"], s["verdict"]) for s in document["working"]] == [(1, 3, 1, "position component", "对"), (5, 7, 2, "fasten", "对"), (9, 11, 3, "inspect", "对")]
            assert document["idle"] == [{"start": 0, "end": 1}, {"start": 3, "end": 5}, {"start": 7, "end": 9}, {"start": 11, "end": 12}]
            assert document["expected_step_count"] == 3 and document["text_description"].strip()
            report["download"] = {"path": str(native), "sha256": digest(native)}
            operation("download_json", **report["download"])
            seek(9.5)
            page.locator("#segList button").last.click()
            page.screenshot(path=str(output / "editor-1920x1080.png"))
            page.wait_for_timeout(1600)
            report["checks"].append("Actual UI selected media, sought boundaries, created three labels, explicitly confirmed verdicts, computed 6s working/6s idle and downloaded exact JSON")
            # Error/cancel checks keep the original annotation in this live page.
            seek(5)
            page.locator("#markStart").click()
            seek(4)
            page.locator("#markEnd").click()
            expect(page.locator("#annotationStatus")).to_contain_text("终点必须晚于起点")
            seek(2)
            page.locator("#markStart").click()
            seek(4)
            page.locator("#markEnd").click()
            expect(page.locator("#annotationStatus")).to_contain_text("不能重叠")
            operation("reject_reverse_and_overlap")
            accept["value"] = False
            page.locator("#fileInput").set_input_files(media)
            expect(page.locator("#annotationStatus")).to_contain_text("已取消切换")
            operation("cancel_video_replacement")
            accept["value"] = True
            bad = output / "invalid.mp4"
            bad.write_bytes(b"deliberately invalid synthetic video")
            page.locator("#fileInput").set_input_files(bad)
            expect(page.locator("#annotationStatus")).to_contain_text("视频加载失败", timeout=20000)
            with page.expect_download() as pending:
                page.locator("#exportJSON").click()
            preserved = output / "after-errors.json"
            pending.value.save_as(preserved)
            assert json.loads(preserved.read_text(encoding="utf-8")) == document
            operation("invalid_media_rejected_and_document_preserved", output_sha256=digest(preserved))
            # A missing description must remain saveable as a draft, then fail QC.
            description = page.locator("#taskDescription").input_value()
            page.locator("#taskDescription").fill("")
            with page.expect_download() as pending:
                page.locator("#exportJSON").click()
            draft = output / "missing-description-draft.json"
            pending.value.save_as(draft)
            expect(page.locator("#annotationStatus")).to_contain_text("待复核草稿")
            assert json.loads(draft.read_text(encoding="utf-8"))["working"] == document["working"]
            page.locator("#taskDescription").fill(description)
            operation("save_incomplete_draft_without_losing_segments")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            page.screenshot(path=str(output / "editor-mobile.png"), full_page=True)
            report["checks"].append("Reverse/overlapping ranges, cancelled replacement and bad-media replacement preserve saved work; blank-description draft remains exportable; 390px layout has no horizontal overflow")
            # Exercise zero and full working coverage through real controls.
            page.set_viewport_size({"width": 1920, "height": 1080})
            for _ in range(3):
                page.locator("#deleteSegment").click()
            expect(page.locator("#idleDuration")).to_have_text("未计算")
            page.locator("#computeIdle").click()
            expect(page.locator("#idleDuration")).to_have_text("12.000s")
            seek(0)
            page.locator("#markStart").click()
            seek(12)
            page.locator("#markEnd").click()
            expect(page.locator("#idleDuration")).to_have_text("未计算")
            page.locator("#computeIdle").click()
            expect(page.locator("#workingDuration")).to_have_text("12.000s")
            expect(page.locator("#idleDuration")).to_have_text("0.000s")
            operation("zero_and_full_working_coverage", full_working_idle_seconds=0)
            report["checks"].append("Computed zero idle is displayed as 0.000s; edits invalidate previous idle totals")
            # This is the actual CLI and actual report, kept in the recording.
            qc = output / "actual-qc-report.html"
            run_command(output, "actual-qc", [sys.executable, "-B", "-m", "trace_annotate.cli", "embodied-pipeline", str(native), "--report", str(qc)])
            page.goto(qc.as_uri())
            expect(page.locator(".metric .working")).to_contain_text("6.0")
            expect(page.locator(".metric .idle")).to_contain_text("6.0")
            expect(page.locator(".metric").last).to_contain_text("0")
            expect(page.locator(".status").first).to_have_text("PASS_SCOPED")
            page.screenshot(path=str(output / "qc-report-1920x1080.png"))
            page.wait_for_timeout(2400)
            operation("open_actual_cli_qc_report", report_sha256=digest(qc))
            assert not report["page_errors"], report["page_errors"]
            assert digest(ui) == before
            report["status"] = "PASS_SCOPED"
        except BaseException as error:
            report.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            raise
        finally:
            cleanup_errors = []
            for name, cleanup in (("context", context.close), ("browser", browser.close)):
                try:
                    cleanup()
                except Exception as error:
                    cleanup_errors.append(name + ": " + str(error))
            try:
                report["recording"] = str(recording.path())
            except Exception as error:
                cleanup_errors.append("recording: " + str(error))
            if cleanup_errors:
                report.update(status="FAIL", cleanup_errors=cleanup_errors)
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "browser-results.json", report)
            if cleanup_errors and "error" not in report:
                raise RuntimeError("Browser cleanup failed: " + "; ".join(cleanup_errors))
    return report


def make_demo_clip(output, recording):
    from PIL import Image, ImageDraw, ImageFont
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe are required")
    probe = json.loads(run_command(output, "recording-probe", [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", recording]))
    duration = float(probe["format"]["duration"])
    speed = duration / 12
    image = Image.new("RGBA", (1920, 54), (7, 13, 23, 245))
    font = demo_font(22)
    ImageDraw.Draw(image).text((24, 13), f"REAL BROWSER RECORDING | {speed:.3f}x speed | SYNTHETIC OPERATION FIXTURE | no semantic accuracy claim", fill="#c6e9f0", font=font)
    caption = output / "recording-speed-caption.png"
    image.save(caption)
    target = output / "traceannotate-real-browser-12s.mp4"
    run_command(output, "encode-recording", [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", recording,
                "-i", str(caption), "-filter_complex", f"[0:v]setpts=(PTS-STARTPTS)/{speed:.12f},fps=25[base];[base][1:v]overlay=0:H-h:format=auto",
                "-t", "12", "-an", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)])
    final = json.loads(run_command(output, "demo-probe", [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(target)]))
    video = next(stream for stream in final["streams"] if stream["codec_type"] == "video")
    assert video["width"] == 1920 and video["height"] == 1080
    assert 10 <= float(final["format"]["duration"]) <= 15
    result = {"path": str(target), "sha256": digest(target), "recording_speed": speed,
              "source_duration_seconds": duration, "duration_seconds": float(final["format"]["duration"]),
              "scope": "Accelerated unaltered UI recording with an explicit speed caption; includes synthetic/error-case operations"}
    write_json(output / "demo-clip.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory; existing paths refused")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("Output must be new: " + str(error))
    try:
        media = create_synthetic_video(output)
        browser = record_workflow(output, media)
        # Run the real command, retaining its expected nonzero result separately.
        command = [sys.executable, "-B", "-m", "trace_annotate.cli", "embodied-pipeline", str(output / "missing-description-draft.json"), "--report", str(output / "draft-failure-report.html")]
        run_command(output, "draft-qc", command, expected_exit=1)
        clip = make_demo_clip(output, browser["recording"])
        inventory = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                     for path in sorted(output.rglob("*")) if path.is_file()]
        write_json(output / "evidence-inventory.json", inventory)
        print(json.dumps({"status": "PASS_SCOPED", "output": str(output), "ui_sha256": browser["ui_sha256"], "demo": clip}, ensure_ascii=False, indent=2))
    except BaseException:
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
