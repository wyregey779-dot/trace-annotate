"""Run the general editor's real browser workflow against synthetic media only.

Use a new output directory. Each run creates its own headless browser context;
it never attaches to a user browser or profile. The bundled Chromium installed
by Playwright is the default; --channel chrome selects an installed Chrome.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from trace_annotate.temporal.formats import load_document, import_subtitles
from trace_annotate.temporal.schema import validate_document


def run_editor(output, channel=None):
    try:
        from playwright.sync_api import sync_playwright, expect
    except ImportError as exc:
        raise RuntimeError(
            'Playwright is missing. Run: python -m pip install ".[dev,browser-test]"; '
            'then: python -m playwright install chromium'
        ) from exc
    ROOT = output
    findings = []
    errors = []
    ui = SOURCE / "trace_annotate/temporal/ui/annotator.html"
    media = ROOT / "general-demo.webm"
    dialogs = {"accept": True}
    with sync_playwright() as playwright:
        launch_options = {"headless": True}
        if channel:
            launch_options["channel"] = channel
        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept() if dialogs["accept"] else dialog.dismiss())
        page.goto(ui.as_uri())
        try:
            expect(page.locator("#exportJSON")).to_be_disabled()
            page.locator("#videoInput").set_input_files(media)
            expect(page.locator("#saveSegment")).to_be_enabled(timeout=15000)
            assert page.locator("video").evaluate("v => v.duration") == 12
            findings.append("Loaded actual synthetic 12-second WebM through the standard file input")

            def fill(start, end, label, text, track="事件", tags='["人物", "室内"]'):
                page.locator("#startInput").fill(str(start))
                page.locator("#endInput").fill(str(end))
                page.locator("#labelInput").fill(label)
                page.locator("#textInput").fill(text)
                page.locator("#trackInput").fill(track)
                page.locator("#tagsInput").fill(tags)

            def save_json(name):
                with page.expect_download() as download:
                    page.locator("#exportJSON").click()
                output = ROOT / name
                download.value.save_as(output)
                return load_document(output)

            fill(5, 2, "反向区间", "应拒绝")
            page.locator("#saveSegment").click()
            expect(page.locator("#formError")).not_to_be_empty()
            assert page.locator("#segmentList li").count() == 0
            fill(1, 5, "人物进入画面", "人工描述 😀")
            page.locator("#saveSegment").click()
            page.locator("#newSegment").click()
            fill(3, 7, "镜头移动", "与第一个事件重叠")
            page.locator("#saveSegment").click()
            assert page.locator("#segmentList li").count() == 2
            assert page.locator(".lane").count() == 2
            first = save_json("ui-overlap.json")
            assert all(s["review_status"] == "unreviewed" for s in first["segments"])
            findings.append("Rejected reversed range; created two same-track overlapping events in separate visual lanes")

            page.locator("#endInput").fill("8")
            page.locator("#reviewInput").check()
            page.locator("#saveSegment").click()
            page.locator("#deleteSegment").click()
            assert page.locator("#segmentList li").count() == 1
            page.locator("#undo").click()
            assert page.locator("#segmentList li").count() == 2
            modified = save_json("ui-edited.json")
            assert modified["segments"][1]["id"] == first["segments"][1]["id"]
            assert modified["segments"][1]["end"] == 8
            assert modified["segments"][1]["review_status"] == "reviewed"
            findings.append("Edited times/review state, deleted and undid deletion while retaining stable IDs")

            fixture = json.loads((SOURCE / "examples/annotations/general-video.json").read_text(encoding="utf-8"))
            fixture["clip"] = media.name
            fixture.update(schema_version=2, media_size=media.stat().st_size,
                           media_sha256=hashlib.sha256(media.read_bytes()).hexdigest())
            fixture_path = ROOT / "browser-fixture.json"
            fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
            page.locator("#jsonInput").set_input_files(fixture_path)
            expect(page.locator("#segmentCount")).to_contain_text("4 个片段")
            assert save_json("ui-restored-fixture.json") == fixture
            findings.append("Imported and downloaded native JSON without loss of IDs, metadata, tags, text, tracks or times")

            with page.expect_download() as download:
                page.locator("#exportCSV").click()
            csv_output = ROOT / "ui-table.csv"
            download.value.save_as(csv_output)
            rows = list(csv.DictReader(io.StringIO(csv_output.read_text(encoding="utf-8-sig"))))
            assert len(rows) == 4 and json.loads(rows[0]["tags"]) == fixture["segments"][0]["tags"]
            page.locator("#exportSRT").click()
            expect(page.locator("#status")).to_contain_text("选择")
            page.locator("#subtitleTrack").select_option("字幕")
            for fmt in ("srt", "vtt"):
                with page.expect_download() as download:
                    page.locator("#export" + fmt.upper()).click()
                subtitle_output = ROOT / ("ui-subtitles." + fmt)
                download.value.save_as(subtitle_output)
                imported = import_subtitles(subtitle_output.read_text(encoding="utf-8"), fmt, media.name, 12)
                assert [(s["start"], s["end"], s["text"]) for s in imported["segments"]] == [
                    (s["start"], s["end"], s["text"]) for s in fixture["segments"] if s["track"] == "字幕"]
            findings.append("Downloaded CSV/SRT/WebVTT; actual Python parser read both browser subtitle downloads with expected times and text")

            mismatch = copy.deepcopy(fixture)
            mismatch["media_sha256"] = "0" * 64
            mismatch_path = ROOT / "mismatched.json"
            mismatch_path.write_text(json.dumps(mismatch, ensure_ascii=False), encoding="utf-8")
            page.locator("#jsonInput").set_input_files(mismatch_path)
            expect(page.locator("#status")).to_contain_text("不一致")
            assert save_json("ui-after-rejected-import.json") == fixture
            duplicate_path = ROOT / "duplicate-key.json"
            duplicate_path.write_text('{"kind":"temporal","kind":"embodied"}', encoding="utf-8")
            page.locator("#jsonInput").set_input_files(duplicate_path)
            expect(page.locator("#status")).to_contain_text("重复")
            assert page.locator("#segmentList li").count() == 4
            findings.append("Rejected media mismatch and duplicate-key JSON without changing current annotations")

            # Compare the browser's structural gate with the production Python gate.
            cases = [copy.deepcopy(fixture)]
            for key, value in [("duration", True), ("duration", "12"), ("duration", 0), ("duration", 9007199254740992),
                               ("clip", "a:b.mp4"), ("clip", "bad\tname.mp4"), ("media_size", -1),
                               ("media_sha256", "F" * 64), ("media_sha256", "0" * 63),
                               ("metadata", {"bad": "\ud800"}), ("metadata", {"big": 9007199254740992})]:
                doc = copy.deepcopy(fixture)
                doc[key] = value
                cases.append(doc)
            for key, value in [("start", True), ("end", 50), ("label", ""), ("text", "\ud800"),
                               ("track", []), ("review_status", "correct"), ("tags", [""])]:
                doc = copy.deepcopy(fixture)
                doc["segments"][0][key] = value
                cases.append(doc)
            browser_results = page.evaluate("cases => cases.map(doc => validateDocument(doc).length === 0)", cases)
            assert browser_results == [not validate_document(doc) for doc in cases]
            findings.append(f"Browser and Python validation agree on {len(cases)} valid/invalid Unicode, numeric and schema fixtures")
            for start in (0.5005, 0.5015, 59.9999, 3599.9999):
                rounding_doc = copy.deepcopy(fixture)
                rounding_doc["duration"] = 4001
                rounding_doc["segments"] = [dict(fixture["segments"][0], start=start, end=start + 1)]
                from trace_annotate.temporal.formats import export_subtitles
                for fmt in ("srt", "vtt"):
                    actual = page.evaluate("arg => toSubtitles(arg.doc, '事件', arg.fmt)", {"doc": rounding_doc, "fmt": fmt})
                    assert actual == export_subtitles(rounding_doc, fmt, track="事件")
            findings.append("Browser subtitle bytes match Python for 0.5005/0.5015-second ties and minute/hour carry boundaries")

            # Drafts include the document; browser reload has no access to media until re-selected.
            page.wait_for_function("() => document.querySelector('#draftStatus').textContent.includes('已更新')")
            page.reload()
            page.locator("#videoInput").set_input_files(media)
            expect(page.locator("#restoreDraft")).to_be_visible(timeout=15000)
            page.locator("#restoreDraft").click()
            expect(page.locator("#segmentCount")).to_contain_text("4 个片段")
            assert save_json("ui-after-reload.json") == fixture
            findings.append("Reloaded the actual page, re-selected media, explicitly restored its draft and verified complete JSON")

            # Untrusted label/description render as literal text, never executable markup.
            page.locator("#newSegment").click()
            fill(9, 11, '<img src=x onerror="window.injectionRan=true">', '<b>literal</b> & text', "特殊文字", '[]')
            page.locator("#saveSegment").click()
            assert page.locator("#segmentList img").count() == 0
            assert not page.evaluate("() => window.injectionRan === true")
            page.locator("#subtitleTrack").select_option("特殊文字")
            with page.expect_download() as download:
                page.locator("#exportVTT").click()
            escaped_vtt = ROOT / "ui-escaped.vtt"
            download.value.save_as(escaped_vtt)
            assert import_subtitles(escaped_vtt.read_text(encoding="utf-8"), "vtt", media.name, 12)["segments"][0]["text"] == '<b>literal</b> & text'
            findings.append("Rendered malicious-looking labels as text and verified literal-markup WebVTT escaping and readback")

            page.locator("#jsonInput").set_input_files(fixture_path)
            expect(page.locator("#segmentCount")).to_contain_text("4 个片段")
            page.locator("#togglePlay").click()
            page.wait_for_function("() => document.querySelector('video').currentTime > 0.5 && document.querySelector('video').readyState >= 3")
            page.locator("#togglePlay").click()
            page.locator("#segmentList button").first.click()
            page.wait_for_function("() => !document.querySelector('video').seeking && document.querySelector('video').readyState >= 2")
            findings.append("Played and decoded actual WebM frames, paused, and selected a segment to seek")
            other_media = ROOT / "other-video.webm"
            other_media.write_bytes(media.read_bytes() + b"traceannotate-other-content")
            dialogs["accept"] = False
            page.locator("#videoInput").set_input_files(other_media)
            expect(page.locator("#status")).to_contain_text("已取消切换")
            dialogs["accept"] = True
            assert save_json("ui-after-cancelled-media.json") == fixture
            bad_media = ROOT / "invalid.webm"
            bad_media.write_bytes(b"deliberately invalid synthetic media")
            page.locator("#videoInput").set_input_files(bad_media)
            expect(page.locator("#status")).to_contain_text("失败", timeout=15000)
            assert save_json("ui-after-invalid-media.json") == fixture
            findings.append("Cancelled a video switch and rejected an undecodable synthetic file while preserving current work")

            # Dense time intervals must not acquire wider pointer targets from
            # button padding or borders and select their neighbours instead.
            dense = copy.deepcopy(fixture)
            dense["segments"] = []
            short_start = 12 - 0.63 - 0.000001
            boundaries = [round(short_start * i / 76, 6) for i in range(77)] + [11.999999, 12]
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
                dense["segments"].append({
                    "id": f"dense-{index:03d}", "start": start, "end": end,
                    "label": f"Dense shot {index:03d}", "text": "Pointer and keyboard selection fixture",
                    "tags": [], "track": "Dense shots", "review_status": "unreviewed",
                })
            for index, (start, end) in enumerate(((1, 3), (2, 4), (4, 5)), 1):
                dense["segments"].append(dict(dense["segments"][0], id=f"overlap-{index}",
                                               start=start, end=end, label=f"Overlap {index}", track="Overlap"))
            dense_path = ROOT / "dense-timeline-fixture.json"
            dense_path.write_text(json.dumps(dense, ensure_ascii=False, indent=2), encoding="utf-8")
            page.set_viewport_size({"width": 1069, "height": 674})
            page.locator("#jsonInput").set_input_files(dense_path)
            expect(page.locator("#segmentCount")).to_contain_text("81 个片段")
            assert page.locator(".lane").count() == 3
            overlap_layout = page.evaluate("""() => {
                const lane = id => document.querySelector('.clip[data-segment-id="' + id + '"]').parentElement;
                return {overlappingSeparate: lane('overlap-1') !== lane('overlap-2'),
                        adjacentReusesLane: lane('overlap-1') === lane('overlap-3')};
            }""")
            assert overlap_layout == {"overlappingSeparate": True, "adjacentReusesLane": True}
            dense_clicks = []
            for segment in dense["segments"][:77]:
                target = page.locator('.clip[data-segment-id="' + segment["id"] + '"]')
                target.scroll_into_view_if_needed()
                geometry = target.evaluate("""el => {
                    const rect = el.getBoundingClientRect();
                    const lane = el.parentElement.getBoundingClientRect();
                    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    return {width: rect.width, laneWidth: lane.width - 2,
                            centerHit: hit?.closest('button')?.dataset.segmentId};
                }""")
                assert geometry["centerHit"] == segment["id"], (segment["id"], geometry)
                expected_width = (segment["end"] - segment["start"]) / 12 * geometry["laneWidth"]
                assert abs(geometry["width"] - expected_width) < 0.05, (segment["id"], geometry)
                target.click()
                expect(page.locator("#segmentId")).to_have_text("ID：" + segment["id"])
                expect(page.locator("#labelInput")).to_have_value(segment["label"])
                assert float(page.locator("#startInput").input_value()) == segment["start"]
                dense_clicks.append({"id": segment["id"], **geometry})
            first_clip = page.locator('.clip[data-segment-id="dense-001"]')
            second_clip = page.locator('.clip[data-segment-id="dense-002"]')
            first_clip.focus()
            page.keyboard.press("Enter")
            expect(first_clip).to_be_focused()
            expect(page.locator("#segmentId")).to_have_text("ID：dense-001")
            page.keyboard.press("Tab")
            expect(second_clip).to_be_focused()
            assert second_clip.evaluate("el => el.matches(':focus-visible')")
            page.keyboard.press("Space")
            expect(second_clip).to_be_focused()
            expect(page.locator("#segmentId")).to_have_text("ID：dense-002")
            tiny_list_item = page.locator('.segment-item[data-segment-id="dense-078"]')
            tiny_list_item.scroll_into_view_if_needed()
            tiny_list_item.focus()
            list_position = """el => {
                const list = el.closest('ol'), item = el.getBoundingClientRect(), box = list.getBoundingClientRect();
                return {scrollTop: list.scrollTop, visible: item.top >= box.top && item.bottom <= box.bottom};
            }"""
            list_before = tiny_list_item.evaluate(list_position)
            assert list_before["visible"]
            page.keyboard.press("Enter")
            expect(tiny_list_item).to_be_focused()
            expect(page.locator("#segmentId")).to_have_text("ID：dense-078")
            list_after = tiny_list_item.evaluate(list_position)
            assert list_after["visible"] and abs(list_after["scrollTop"] - list_before["scrollTop"]) <= 1
            page.screenshot(path=str(ROOT / "dense-timeline.png"), full_page=True)
            page.emulate_media(forced_colors="active")
            expect(first_clip).to_have_css("outline-style", "solid")
            expect(first_clip).to_have_css("outline-width", "1px")
            expect(first_clip).to_have_css("outline-offset", "-1px")
            first_clip.focus()
            expect(first_clip).to_have_css("outline-width", "3px")
            first_clip.press("Enter")
            expect(first_clip).to_be_focused()
            expect(page.locator("#segmentId")).to_have_text("ID：dense-001")
            page.screenshot(path=str(ROOT / "dense-timeline-forced-colors.png"), full_page=True)
            page.emulate_media(forced_colors="none")
            assert save_json("ui-dense-timeline.json") == dense
            (ROOT / "dense-timeline-results.json").write_text(json.dumps({
                "status": "PASS", "viewport": {"width": 1069, "height": 674},
                "pointer_checks": dense_clicks, "keyboard_enter_tab_space": True,
                "tiny_segment_list_fallback": True, "overlap_lanes": 2,
                "overlap_layout": overlap_layout, "focused_list_before": list_before, "focused_list_after": list_after,
                "forced_colors_edges_and_focus": True, "download_equal": True,
            }, indent=2), encoding="utf-8")
            findings.append("Dense 78-shot track: 77 true interval widths/center hits and actual pointer selections; 0.63s interval; keyboard Enter/Tab/Space focus retained; subpixel segment selectable from list; same-track overlap stays in separate lanes")
            page.locator("#jsonInput").set_input_files(fixture_path)
            expect(page.locator("#segmentCount")).to_contain_text("4 个片段")
            page.set_viewport_size({"width": 1440, "height": 1100})
            page.screenshot(path=str(ROOT / "general-workspace.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth + 1")
            page.screenshot(path=str(ROOT / "general-mobile.png"), full_page=True)
            findings.append("Captured desktop/mobile screenshots and verified no horizontal document overflow at 390 CSS pixels")
            page.locator("#textInput").fill("故障前尚未保存的表单文字")
            page.locator("video").evaluate("v => v.dispatchEvent(new Event('error'))")
            expect(page.locator("#saveSegment")).to_be_disabled()
            assert save_json("ui-decode-failure-backup.json") == fixture
            assert page.locator("#textInput").input_value() == "故障前尚未保存的表单文字"
            expect(page.locator("#status")).to_contain_text("仅备份已保存片段")
            findings.append("Injected a decoder error with a dirty form; explicitly backed up saved segments while retaining unsaved text")
            assert not errors, errors
            report = {"status": "PASS", "browser": browser.version, "headless": True,
                      "user_browser_connected": False, "native_file_picker_tested": False,
                      "media": str(media), "checks": findings, "page_errors": errors}
            (ROOT / "browser-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        except Exception as exc:
            report = {"status": "FAIL", "error": str(exc), "browser": browser.version,
                      "headless": True, "user_browser_connected": False,
                      "native_file_picker_tested": False, "checks": findings,
                      "page_errors": errors}
            (ROOT / "browser-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                page.screenshot(path=str(ROOT / "failure.png"), full_page=True)
            except Exception:
                pass  # The page may already be closed; the traceback still survives.
            raise
        finally:
            context.close()
            browser.close()


def create_synthetic_media(output):
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "FFmpeg is missing from PATH. On Ubuntu run: sudo apt-get install ffmpeg. "
            "On Windows install FFmpeg and add its bin directory to PATH."
        )
    command = [executable, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
               "testsrc2=size=640x360:rate=12:duration=12", "-an", "-c:v", "libvpx",
               "-b:v", "500k", "-deadline", "realtime", "-cpu-used", "8",
               "-pix_fmt", "yuv420p", str(output / "general-demo.webm")]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    (output / "synthetic-media.ffmpeg.log").write_text(
        "Command: " + json.dumps(command) + "\nExit: " + str(result.returncode)
        + "\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError("FFmpeg could not create the synthetic video; see synthetic-media.ffmpeg.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for synthetic media, downloads, screenshots and JSON; existing paths are refused")
    parser.add_argument("--channel", help="Optional installed browser channel (for example chrome); default: bundled Chromium")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error("Output must be a new directory; no existing files were changed: " + str(exc))
    try:
        create_synthetic_media(output)
        run_editor(output, args.channel)
    except Exception as exc:
        report_path = output / "browser-results.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        except (ValueError, OSError):
            report = {}
        report.update({"status": "FAIL", "error": str(exc), "headless": True,
                       "user_browser_connected": False, "native_file_picker_tested": False,
                       "channel": args.channel or "bundled-chromium",
                       "setup": ['python -m pip install ".[dev,browser-test]"',
                                 "python -m playwright install chromium",
                                 "sudo apt-get install ffmpeg (Ubuntu)"]})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("Error: " + str(exc) + "; diagnostics: " + str(output), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
