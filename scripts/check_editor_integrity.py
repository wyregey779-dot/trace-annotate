"""Exercise lossless edits and review invalidation in an actual browser.

Only synthetic media and a new output directory are used. The default browser
is Playwright's bundled Chromium; --channel chrome uses an installed Chrome in
an isolated headless context, never a user profile. Missing prerequisites are
failures, not skips. JSON assertions inspect actual browser downloads.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

from check_general_editor import create_synthetic_media

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from trace_annotate.temporal.formats import import_subtitles, load_document


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def differences(actual, expected, path="document"):
    """Describe exact JSON differences, including invisible newline changes."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(set(actual) | set(expected)):
            if key not in actual:
                result.append({"path": path + "." + key, "missing_actual": True, "expected": expected[key]})
            elif key not in expected:
                result.append({"path": path + "." + key, "actual": actual[key], "unexpected_actual": True})
            else:
                result.extend(differences(actual[key], expected[key], path + "." + key))
        return result
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [{"path": path, "actual_length": len(actual), "expected_length": len(expected)}]
        return [difference for index, pair in enumerate(zip(actual, expected))
                for difference in differences(pair[0], pair[1], path + "[" + str(index) + "]")]
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        return [{"path": path, "actual": actual, "expected": expected}]
    return []


def fixture_document(media):
    return {
        "kind": "temporal", "schema_version": 2, "clip": media.name,
        "duration": 12, "media_size": media.stat().st_size,
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "metadata": {
            "document_id": "integrity-document-stable-001",
            "origin": "synthetic-browser-regression",
            "nested": {"text": "元信息\r\n保持😀", "values": [1, True, None]},
        },
        "segments": [
            {"id": "stable-primary-id", "start": 1, "end": 4,
             "label": "第一行\n第二行", "text": "a\r\nb",
             "tags": ["第一项\n第二行", "emoji😀", "不应改动"],
             "track": "A\nB", "review_status": "reviewed"},
            {"id": "stable-control-id", "start": 6, "end": 9,
             "label": "对照片段", "text": "不会与前一条轨道合并",
             "tags": ["对照"], "track": "AB", "review_status": "reviewed"},
        ],
    }


class BrowserCase:
    def __init__(self, page, output, fixture, expect, record, dialog_policy):
        self.page, self.output, self.fixture = page, output, fixture
        self.expect, self.record = expect, record
        self.dialog_policy = dialog_policy
        self.download_index = 0

    def equal(self, actual, expected, description):
        changes = differences(actual, expected)
        check = {"description": description, "status": "FAIL" if changes else "PASS"}
        if changes:
            check["differences"] = changes
        self.record["assertions"].append(check)
        if changes:
            raise AssertionError(description + ": " + json.dumps(changes, ensure_ascii=False))

    def select(self):
        self.page.locator("#segmentList button").first.click()
        self.expect(self.page.locator("#deleteSegment")).to_be_enabled()

    def save(self):
        self.page.locator("#saveSegment").click()
        self.expect(self.page.locator("#formError")).to_be_empty()

    def export(self, name):
        self.download_index += 1
        path = self.output / (str(self.download_index).zfill(2) + "-" + name + ".json")
        with self.page.expect_download() as download:
            self.page.locator("#exportJSON").click()
        download.value.save_as(path)
        self.record["downloads"].append(str(path))
        return load_document(path)

    def reviewed(self, expected, description):
        self.equal(self.page.locator("#reviewInput").is_checked(), expected, description)

    def expected(self, **changes):
        document = copy.deepcopy(self.fixture)
        document["segments"][0].update(changes)
        return document


def case_time_only(case):
    case.equal(case.export("import-roundtrip"), case.fixture,
               "Import/download preserves native JSON before any form editing")
    case.select()
    case.page.locator("#endInput").fill("5")
    case.reviewed(False, "Changing only end invalidates the previous review immediately")
    case.save()
    case.equal(case.export("end-only"), case.expected(end=5, review_status="unreviewed"),
               "End-only save preserves exact label, CRLF text, track, tags, metadata and IDs")


def case_label_only(case):
    case.select()
    case.page.locator("#labelInput").fill("新类别 😀")
    case.reviewed(False, "Changing only label invalidates the previous review")
    case.save()
    case.equal(case.export("label-only"), case.expected(label="新类别 😀", review_status="unreviewed"),
               "Label-only save preserves every untouched field byte-for-byte as JSON strings")


def case_explicit_review(case):
    case.select()
    text = "主动修订\n新说明😀"
    case.page.locator("#textInput").fill(text)
    case.reviewed(False, "Editing description clears review before a new explicit review")
    case.page.locator("#reviewInput").check()
    case.save()
    case.equal(case.export("explicit-review"), case.expected(text=text),
               "Explicit review after the final content edit persists as reviewed")
    case.page.locator("#startInput").fill("1.5")
    case.reviewed(False, "A later start edit invalidates that explicit review")
    case.save()
    case.equal(case.export("changed-after-review"),
               case.expected(start=1.5, text=text, review_status="unreviewed"),
               "Saving after another content edit requires a new review")


def case_unchanged(case):
    case.select()
    case.reviewed(True, "Selecting an imported reviewed segment does not invalidate it")
    case.save()
    case.equal(case.export("unchanged-save"), case.fixture,
               "Selecting and saving unchanged content preserves the complete document and review")


def case_history(case):
    case.select()
    case.page.locator("#endInput").fill("5.25")
    case.save()
    expected = case.expected(end=5.25, review_status="unreviewed")
    case.equal(case.export("before-undo"), expected, "Time change persists without losing original text")
    case.page.locator("#undo").click()
    case.equal(case.export("after-undo"), case.fixture,
               "Undo restores original CRLF text, metadata, stable document/segment IDs and review")
    case.reviewed(True, "Undo restores the original reviewed editor state")
    case.page.locator("#redo").click()
    case.equal(case.export("after-redo"), expected,
               "Redo restores the time edit and invalidated review while keeping original text and IDs")
    case.reviewed(False, "Redo restores the edited unreviewed editor state")


def case_dirty_draft(case):
    case.select()
    case.page.locator("#endInput").fill("5.5")
    case.reviewed(False, "An unsaved time edit invalidates review before a reload")
    # This exceeds the documented draft debounce. Reload/pagehide must also
    # flush pending edits; no internal functions or storage objects are patched.
    case.page.wait_for_timeout(700)
    case.page.reload()
    media = case.output.parent / "general-demo.webm"
    case.page.locator("#videoInput").set_input_files(media)
    case.expect(case.page.locator("#restoreDraft")).to_be_visible(timeout=15000)
    case.page.locator("#restoreDraft").click()
    case.expect(case.page.locator("#endInput")).to_have_value("5.5")
    case.reviewed(False, "Restoring an unsaved edited form does not restore a stale review")
    case.save()
    case.equal(case.export("restored-dirty-draft"),
               case.expected(end=5.5, review_status="unreviewed"),
               "Reload/reselect/restore/save retains untouched original CRLF, multiline label/track and metadata")


def case_intentional_newline(case):
    case.select()
    text = "用户主动输入\n第二行😀\n第三行"
    case.page.locator("#textInput").fill(text)
    case.save()
    case.equal(case.export("intentional-newlines"), case.expected(text=text, review_status="unreviewed"),
               "An explicitly edited description saves the new multiline value without normalizing other fields")


def case_legacy_draft(case):
    primary = case.fixture["segments"][0]
    envelopes = {}
    for variant, end in (("edited", "5.75"), ("unchanged", "4")):
        # This is the exact display shape stored by the historical single-line
        # label/track inputs and browser-normalized textarea. It is a fixture,
        # not a call to any current draft-writing application implementation.
        envelope = {
            "draft_format": 1, "updated_at": "2026-10-22T00:00:00Z",
            "document": copy.deepcopy(case.fixture), "selected_id": primary["id"],
            "editor": {
                "startInput": "1", "endInput": end,
                "labelInput": primary["label"].replace("\r", "").replace("\n", ""),
                "textInput": primary["text"].replace("\r\n", "\n").replace("\r", "\n"),
                "trackInput": primary["track"].replace("\r", "").replace("\n", ""),
                "tagsInput": json.dumps(primary["tags"], ensure_ascii=False, separators=(",", ":")),
                "reviewInput": True,
            },
        }
        envelope["document"]["schema_version"] = 1
        del envelope["document"]["media_sha256"]
        envelopes[variant] = envelope
        write_json(case.output / ("legacy-" + variant + "-fixture.json"), envelope)
    media = case.output.parent / "general-demo.webm"
    storage_key = "traceannotate.temporal.v1:" + media.name + ":" + str(media.stat().st_size)
    content_key = "traceannotate.temporal.v2:sha256:" + hashlib.sha256(media.read_bytes()).hexdigest()
    payload = json.dumps({"fixtures": envelopes, "storage_key": storage_key,
                          "content_key": content_key}, ensure_ascii=False)
    case.page.add_init_script(
        "(() => { const seed = " + payload + "; "
        "const name = new URL(location.href).searchParams.get('integrity_legacy'); "
        "if (Object.hasOwn(seed.fixtures, name)) { "
        "localStorage.removeItem(seed.content_key); "
        "localStorage.setItem(seed.storage_key, JSON.stringify(seed.fixtures[name])); } })();"
    )
    case.record["fixture_injection"] = {
        "purpose": "Seed documented historical format-1 persisted drafts before application initialization",
        "scope": "Historical draft fixture and removal of the newer synthetic-media draft key before initialization",
        "application_functions_replaced": False,
    }
    for variant in ("edited", "unchanged"):
        case.page.goto((SOURCE / "trace_annotate/temporal/ui/annotator.html").as_uri()
                       + "?integrity_legacy=" + variant)
        case.page.locator("#videoInput").set_input_files(media)
        case.expect(case.page.locator("#restoreDraft")).to_be_visible(timeout=15000)
        case.page.locator("#restoreDraft").click()
        case.expect(case.page.locator("#endInput")).to_have_value("5.75" if variant == "edited" else "4")
        case.reviewed(False, "Legacy " + variant + " draft migration invalidates an unbound review")
        case.save()
        expected = (case.expected(end=5.75, review_status="unreviewed")
                    if variant == "edited" else case.fixture)
        expected = copy.deepcopy(expected)
        for item in expected["segments"]:
            item["review_status"] = "unreviewed"
        case.equal(case.export("legacy-" + variant), expected,
                   "Legacy " + variant + " draft preserves native CRLF and multiline label/track while applying only intended changes")


def case_export_preflight(case):
    document = copy.deepcopy(case.fixture)
    document["segments"][0].update(track="字幕", review_status="unreviewed")
    third = copy.deepcopy(document["segments"][1])
    third.update(id="stable-third-id", start=9, end=11, track="字幕", label="第二条字幕", text="末尾字幕😀")
    document["segments"].append(third)
    fixture_path = case.output / "exchange-input.json"
    write_json(fixture_path, document)
    case.page.locator("#jsonInput").set_input_files(fixture_path)
    case.expect(case.page.locator("#segmentList li")).to_have_count(3)
    case.page.locator("#subtitleTrack").select_option("字幕")
    for fmt in ("csv", "srt", "vtt"):
        event_count = len(case.record["download_events"])
        dialog_count = len(case.record["dialogs"])
        case.dialog_policy["accept"] = False
        case.page.locator("#export" + fmt.upper()).click()
        # A cancelled synchronous confirm must not schedule a browser download.
        # Keep an explicit observation interval and capture every download event.
        case.page.wait_for_timeout(500)
        case.equal(len(case.record["dialogs"]), dialog_count + 1,
                   fmt.upper() + " opens one actual preflight dialog")
        dialog = case.record["dialogs"][-1]
        message = dialog["message"]
        count = 3 if fmt == "csv" else 2
        tokens = [fmt.upper(), "导出 " + str(count) + " 个片段", "待复核 1 个", "metadata", "JSON"]
        if fmt == "csv":
            tokens.extend(["不包含", "媒体时长", "文件大小", "防公式"])
        else:
            tokens.extend(["所选轨道", '"字幕"', "片段 ID", "标签集合", "复核状态", "毫秒", "重叠"])
        case.equal({token: token in message for token in tokens}, {token: True for token in tokens},
                   fmt.upper() + " preflight explains selected counts, pending reviews, track scope and format losses")
        case.equal(dialog["action"], "dismiss", fmt.upper() + " preflight was actually cancelled")
        case.equal(len(case.record["download_events"]), event_count,
                   fmt.upper() + " cancellation produces no download during the observation interval")
        case.equal(case.export(fmt + "-after-cancel"), document,
                   fmt.upper() + " cancelled export does not change the native JSON")

        case.dialog_policy["accept"] = True
        export_path = case.output / ("accepted." + fmt)
        with case.page.expect_download() as download:
            case.page.locator("#export" + fmt.upper()).click()
        download.value.save_as(export_path)
        case.record["downloads"].append(str(export_path))
        case.equal(case.record["dialogs"][-1]["action"], "accept",
                   fmt.upper() + " download follows explicit preflight acceptance")
        if fmt == "csv":
            # newline='' retains embedded CRLF inside quoted CSV fields.
            with export_path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            restored = []
            for row in rows:
                restored.append({"id": row["id"], "start": float(row["start"]), "end": float(row["end"]),
                                 "label": row["label"], "text": row["text"], "tags": json.loads(row["tags"]),
                                 "track": row["track"], "review_status": row["review_status"]})
            case.equal(restored, document["segments"], "Accepted CSV preserves the documented segment-column projection")
        else:
            restored = import_subtitles(export_path.read_text(encoding="utf-8"), fmt,
                                        document["clip"], document["duration"], track="字幕")
            expected = [{"start": item["start"], "end": item["end"],
                         "text": item["text"].replace("\r\n", "\n").replace("\r", "\n")}
                        for item in document["segments"] if item["track"] == "字幕"]
            case.equal([{key: item[key] for key in ("start", "end", "text")}
                        for item in restored["segments"]], expected,
                       "Accepted " + fmt.upper() + " reads back the selected track with expected timings and normalized subtitle text")
        case.equal(case.export(fmt + "-after-accept"), document,
                   fmt.upper() + " accepted exchange export also leaves native JSON unchanged")


def case_nontext_controls(case):
    case.select()
    seek = case.page.locator("#seek")
    # Selecting a segment requests its start time after render() has displayed
    # the previous player position. Home on an already-zero, not-yet-updated
    # range is a native no-op: it emits no input event, then the selection's
    # outstanding seek completes at 1s. Wait for the actual selection operation
    # (player AND range) before issuing a second keyboard operation.
    wait_media_position(case, case.fixture["segments"][0]["start"],
                        "Selected segment seek completed before keyboard navigation")
    seek.focus()
    seek.press("Home")
    case.expect(seek).to_have_value("0")
    wait_media_position(case, 0, "Home moved the actual player and range to zero")
    case.page.locator("#markStart").click()
    case.reviewed(False, "Recording a changed start through the mark button invalidates review")
    case.save()
    case.equal(case.export("mark-start"), case.expected(start=0, review_status="unreviewed"),
               "Mark-start saves the actual player position and preserves untouched native strings")
    case.page.locator("#undo").click()
    case.equal(case.export("mark-start-undone"), case.fixture, "Undoing mark-start restores the original reviewed segment")
    seek.focus()
    seek.press("End")
    case.expect(seek).to_have_value("12")
    wait_media_position(case, 12, "End moved the actual player and range to the endpoint")
    case.page.locator("#markEnd").click()
    case.reviewed(False, "Recording a changed end through the mark button invalidates review")
    case.save()
    case.equal(case.export("mark-end"), case.expected(end=12, review_status="unreviewed"),
               "Mark-end saves the actual player position and preserves untouched native strings")
    case.page.locator("#undo").click()
    case.reviewed(True, "Undoing mark-end restores the original reviewed editor state")
    case.page.locator("#trackNames").select_option("AB")
    case.reviewed(False, "Choosing an existing different track invalidates review")
    case.save()
    case.equal(case.export("track-select"), case.expected(track="AB", review_status="unreviewed"),
               "Existing-track selection changes only the intended track and review status")


def wait_media_position(case, seconds, description):
    """Wait for a completed operation, preserving exact endpoint assertions.

    This does not synthesize events or set player/range values. Failure retains
    the real playback state so a decoder failure differs from a stale control.
    """
    try:
        case.page.wait_for_function("""seconds => {
            const video=document.querySelector('video');
            const range=document.querySelector('#seek');
            return video.currentTime === seconds && !video.seeking &&
                video.readyState >= 2 && Number(range.value) === seconds;
        }""", arg=seconds)
    except Exception:
        case.record.setdefault("media_position_failures", []).append({
            "description": description, "expected_seconds": seconds,
            "actual": case.page.evaluate("""() => {
                const video=document.querySelector('video');
                return {currentTime:video.currentTime,seeking:video.seeking,
                    readyState:video.readyState,range:document.querySelector('#seek').value,
                    error:video.error?{code:video.error.code,message:video.error.message}:null};
            }""")})
        raise
    case.equal(case.page.locator("video").evaluate("video=>video.currentTime"), seconds,
               description)


def case_selection_seek_synchronization(case):
    # Run the genuine DOM click handler and inspect its synchronous result in
    # the same browser task, before queued seeking/timeupdate events can hide a
    # stale range. This is a dispatch-order regression, not a trusted-input or
    # completed-decoding claim. No application/browser functions are replaced.
    observed = case.page.locator("#segmentList button").first.evaluate("""button => {
        const video=document.querySelector('video'),range=document.querySelector('#seek');
        const before={currentTime:video.currentTime,range:Number(range.value)};
        button.click();
        return {before,after:{currentTime:video.currentTime,range:Number(range.value),
            seeking:video.seeking,readyState:video.readyState}};
    }""")
    write_json(case.output / "synchronous-selection-observation.json", observed)
    case.record["dispatch_scope"] = "Native DOM button.click handler in one browser task; no application function or media API replacement"
    case.equal(observed["before"], {"currentTime": 0, "range": 0},
               "Newly imported fixture starts with the player and range at zero")
    expected = case.fixture["segments"][0]["start"]
    case.equal({key: observed["after"][key] for key in ("currentTime", "range")},
               {"currentTime": expected, "range": expected},
               "Selection synchronizes the requested player and range positions before queued media events")
    wait_media_position(case, expected,
                        "The separately observed real seek completes after synchronous range feedback")
    case.equal(case.export("after-synchronous-selection"), case.fixture,
               "Selection feedback and completed seeking preserve exact annotations and reviewed state")


CASES = [
    ("01-time-only", case_time_only),
    ("02-label-only", case_label_only),
    ("03-explicit-review", case_explicit_review),
    ("04-unchanged-save", case_unchanged),
    ("05-undo-redo", case_history),
    ("06-dirty-draft", case_dirty_draft),
    ("07-intentional-newlines", case_intentional_newline),
    ("08-legacy-format1-draft", case_legacy_draft),
    ("09-export-preflight", case_export_preflight),
    ("10-mark-and-track-controls", case_nontext_controls),
    ("11-selection-seek-synchronization", case_selection_seek_synchronization),
]


def run_cases(output, channel, report):
    try:
        from playwright.sync_api import sync_playwright, expect
    except ImportError as exc:
        raise RuntimeError(
            'Playwright is missing. Run: python -m pip install ".[dev,browser-test]"; '
            'then: python -m playwright install chromium'
        ) from exc
    media = output / "general-demo.webm"
    fixture = fixture_document(media)
    report["media"] = {"path": str(media), "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                       "synthetic": True, "expected_duration_seconds": 12}
    with sync_playwright() as playwright:
        options = {"headless": True}
        if channel:
            options["channel"] = channel
        browser = playwright.chromium.launch(**options)
        report["browser_version"] = browser.version
        try:
            for name, operation in CASES:
                case_output = output / name
                case_output.mkdir()
                record = {"name": name, "status": "RUNNING", "assertions": [], "downloads": [],
                          "page_errors": [], "dialogs": [], "download_events": []}
                report["cases"].append(record)
                context = browser.new_context(accept_downloads=True,
                                              viewport={"width": 1440, "height": 1100})
                page = context.new_page()
                page.on("pageerror", lambda error, result=record: result["page_errors"].append(str(error)))
                page.on("download", lambda download, result=record:
                        result["download_events"].append(download.suggested_filename))
                dialog_policy = {"accept": True}

                def accept_dialog(dialog, result=record, policy=dialog_policy):
                    result["dialogs"].append({"type": dialog.type, "message": dialog.message,
                                              "action": "accept" if policy["accept"] else "dismiss"})
                    if policy["accept"]:
                        dialog.accept()
                    else:
                        dialog.dismiss()

                page.on("dialog", accept_dialog)
                case = BrowserCase(page, case_output, fixture, expect, record, dialog_policy)
                try:
                    record["ui_sha256"] = hashlib.sha256(
                        (SOURCE / "trace_annotate/temporal/ui/annotator.html").read_bytes()).hexdigest()
                    case.equal(record["ui_sha256"], report["ui_sha256"],
                               "The UI source is unchanged since this test run started")
                    page.goto((SOURCE / "trace_annotate/temporal/ui/annotator.html").as_uri())
                    page.locator("#videoInput").set_input_files(media)
                    expect(page.locator("#saveSegment")).to_be_enabled(timeout=15000)
                    case.equal(page.locator("video").evaluate("video => video.duration"), 12,
                               "Actual browser reads the synthetic video duration")
                    fixture_path = case_output / "input-native.json"
                    write_json(fixture_path, fixture)
                    page.locator("#jsonInput").set_input_files(fixture_path)
                    expect(page.locator("#segmentList li")).to_have_count(2)
                    operation(case)
                    case.equal(record["page_errors"], [], "No uncaught browser script errors")
                    record["status"] = "PASS"
                except Exception as exc:
                    record.update({"status": "FAIL", "error": str(exc)})
                    (case_output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
                finally:
                    try:
                        snapshot = {key: (page.locator("#" + key).is_checked() if key == "reviewInput"
                                          else page.locator("#" + key).input_value())
                                    for key in ("startInput", "endInput", "labelInput", "textInput",
                                                "trackInput", "tagsInput", "reviewInput")}
                        snapshot["status"] = page.locator("#status").inner_text()
                        write_json(case_output / "form-snapshot.json", snapshot)
                        screenshot = case_output / ("failure.png" if record["status"] == "FAIL" else "workspace.png")
                        page.screenshot(path=str(screenshot), full_page=True)
                        record["screenshot"] = str(screenshot)
                    except Exception as exc:
                        record["diagnostic_capture_error"] = str(exc)
                        record["status"] = "FAIL"
                    write_json(case_output / "case-results.json", record)
                    context.close()
                report["completed_cases"] = len(report["cases"])
                write_json(output / "browser-results.json", report)
        finally:
            browser.close()
    report["passed_cases"] = sum(case["status"] == "PASS" for case in report["cases"])
    report["failed_cases"] = sum(case["status"] == "FAIL" for case in report["cases"])
    if report["failed_cases"]:
        raise RuntimeError(str(report["failed_cases"]) + " browser integrity cases failed; see case reports")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for reports, synthetic media, downloads and screenshots; existing paths refused")
    parser.add_argument("--channel", help="Optional installed browser channel; default: bundled Chromium")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error("Output must be a new directory; no existing files were changed: " + str(exc))
    ui = SOURCE / "trace_annotate/temporal/ui/annotator.html"
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "channel": args.channel or "bundled-chromium", "headless": True,
              "user_browser_connected": False, "native_file_picker_tested": False,
              "input_method": "standard file input using Playwright set_input_files",
              "ui_path": str(ui), "command": [sys.executable] + sys.argv,
              "python_version": sys.version,
              "planned_cases": len(CASES), "completed_cases": 0, "cases": []}
    exit_code = 0
    try:
        report["ui_sha256"] = hashlib.sha256(ui.read_bytes()).hexdigest()
        create_synthetic_media(output)
        run_cases(output, args.channel, report)
        report["final_ui_sha256"] = hashlib.sha256(ui.read_bytes()).hexdigest()
        if report["final_ui_sha256"] != report["ui_sha256"]:
            raise RuntimeError("UI source changed during this run; repeat in a new output directory")
        report["status"] = "PASS"
    except Exception as exc:
        report.update({"status": "FAIL", "error": str(exc)})
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 1
    report.update({"finished_at": datetime.now(timezone.utc).isoformat(), "exit_code": exit_code})
    write_json(output / "browser-results.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
