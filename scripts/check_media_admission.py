"""Exercise decoded-video admission and preservation using real local media.

FFmpeg creates video, audio-only, and unsupported-video-with-audio fixtures.
No user video or browser profile is used. Metadata/audio playback alone must
not enable video annotation. This is not a complete-file corruption scanner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from trace_annotate.temporal.formats import load_document


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixtures(output):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg must be installed on PATH")
    commands = []
    specifications = [
        ("valid.webm", ["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=2",
                        "-an", "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8"]),
        ("audio.m4a", ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac"]),
        ("prores-audio.mov", ["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=2",
                              "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                              "-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le",
                              "-c:a", "aac", "-shortest"]),
        ("offset.mp4", ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=5:duration=5",
                        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-output_ts_offset", "7"]),
    ]
    for name, options in specifications:
        command = [executable, "-nostdin", "-v", "error", *options, str(output / name)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=120)
        commands.append(dict(command=command, exit_code=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
        write(output / "fixture-commands.json", commands)
        if result.returncode:
            raise RuntimeError("FFmpeg fixture generation failed: " + name)
    shutil.copyfile(output / "audio.m4a", output / "audio-renamed.mp4")
    (output / "empty.mp4").write_bytes(b"")
    (output / "not-video.mp4").write_bytes(b"This file is deliberately not a video.")
    names = ["audio.m4a", "audio-renamed.mp4", "prores-audio.mov", "offset.mp4", "empty.mp4", "not-video.mp4"]
    write(output / "fixtures.json", [{"name": path.name, "size": path.stat().st_size,
                                      "sha256": sha(path)} for path in [output / "valid.webm", *[output / n for n in names]]])
    return [output / n for n in names]


def run(output, channel, ui):
    from playwright.sync_api import sync_playwright, expect
    invalid = fixtures(output)
    report = {"status": "RUNNING", "ui_sha256": sha(ui), "headless": True,
              "native_os_picker_tested": False, "user_browser_profile_used": False,
              "channel": channel or "bundled-chromium", "checks": [], "page_errors": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"channel": channel} if channel else {}))
        report["browser"] = browser.version
        context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 1000})
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        try:
            page.goto(ui.as_uri())
            page.locator("#videoInput").set_input_files(output / "audio.m4a")
            expect(page.locator("#mediaLoad")).to_be_hidden(timeout=40000)
            expect(page.locator("#status")).to_contain_text("加载失败")
            expect(page.locator("#saveSegment")).to_be_disabled()
            expect(page.locator("#exportJSON")).to_be_disabled()
            report["checks"].append({"case": "blank-audio-rejected", "status": page.locator("#status").inner_text()})
            page.locator("#videoInput").set_input_files(output / "valid.webm")
            expect(page.locator("#saveSegment")).to_be_enabled(timeout=40000)
            page.wait_for_function("document.querySelector('video').readyState >= 2")
            page.locator("#startInput").fill("0.25")
            page.locator("#endInput").fill("1.5")
            page.locator("#labelInput").fill("合成测试图样")
            page.locator("#textInput").fill("保存过的原始说明")
            page.locator("#saveSegment").click()

            def export(name):
                with page.expect_download() as pending:
                    page.locator("#exportJSON").click()
                target = output / name
                pending.value.save_as(target)
                return load_document(target)

            original = export("original.json")
            original_source = page.locator("video").evaluate("v=>v.currentSrc")
            marker = "未保存修改仍需保留"
            for i, bad in enumerate(invalid):
                page.locator("#textInput").fill(marker)
                page.locator("#videoInput").set_input_files(bad)
                expect(page.locator("#mediaLoad")).to_be_hidden(timeout=40000)
                expect(page.locator("#status")).to_contain_text("加载失败")
                status = page.locator("#status").inner_text()
                expect(page.locator("#textInput")).to_have_value(marker)
                expect(page.locator("#saveSegment")).to_be_enabled()
                expect(page.locator("#segmentList li")).to_have_count(1)
                assert page.locator("video").evaluate("v=>v.currentSrc") == original_source
                page.screenshot(path=str(output / (bad.name + ".png")), full_page=True)
                page.locator("#discardForm").click()
                actual = export("preserved-%02d.json" % i)
                assert actual == original, bad.name
                report["checks"].append({"case": bad.name, "status": status,
                                         "saved_document_equal": True, "unsaved_form_preserved": True,
                                         "original_video_preserved": True})
            page.locator("#togglePlay").click()
            page.wait_for_function("document.querySelector('video').ended", timeout=10000)
            # Controlled fault injection covers a late duration update separately
            # from the real nonzero-timeline file rejected before replacement.
            page.locator("#textInput").fill(marker)
            page.locator("video").evaluate("v=>{Object.defineProperty(v,'duration',{configurable:true,value:v.duration+1});v.dispatchEvent(new Event('durationchange'));}")
            expect(page.locator("#saveSegment")).to_be_disabled()
            expect(page.locator("#status")).to_contain_text("时长发生变化")
            expect(page.locator("#textInput")).to_have_value(marker)
            assert export("late-duration-saved-backup.json") == original
            expect(page.locator("#status")).to_contain_text("仅备份已保存片段")
            expect(page.locator("#textInput")).to_have_value(marker)
            report["checks"].append({"case": "late-duration-change-fault-injection", "saved_document_equal": True,
                                     "unsaved_form_preserved": True, "editor_disabled": True})
            assert not report["page_errors"]
            assert sha(ui) == report["ui_sha256"]
            report["status"] = "PASS"
        except BaseException as error:
            report.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            raise
        finally:
            write(output / "report.json", report)
            context.close()
            browser.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--channel", help="Installed browser channel, e.g. chrome")
    parser.add_argument("--ui", type=Path, default=SOURCE / "trace_annotate/temporal/ui/annotator.html")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("Output must be a new directory: " + str(error))
    try:
        result = run(output, args.channel, args.ui.resolve())
        print(json.dumps({"status": result["status"], "checks": len(result["checks"]), "output": str(output)}))
    except Exception:
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
