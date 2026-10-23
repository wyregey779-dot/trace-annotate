"""Verify content-bound media imports and drafts in an isolated real browser.

FFmpeg creates one synthetic WebM. Equal-size trailing payloads make two files
with the same basename, dimensions, duration and byte count but different SHA256
digests. Original user media and browser profiles are never opened. Reports use
actual downloaded JSON and hashlib as an independent full-file oracle.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

from check_general_editor import create_synthetic_media
from check_editor_integrity import differences, write_json

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from trace_annotate.temporal.formats import load_document
from trace_annotate.temporal.schema import validate_document


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fixture_document(media):
    return {
        "kind": "temporal", "schema_version": 2, "clip": media.name,
        "duration": 12, "media_size": media.stat().st_size,
        "media_sha256": digest(media),
        "metadata": {"document_id": "media-identity-fixture", "nested": ["原始元信息😀", True, None]},
        "segments": [{"id": "stable-identity-segment", "start": 1, "end": 4,
                      "label": "经过复核的片段", "text": "原始说明\r\n第二行😀",
                      "tags": ["原始标签"], "track": "主轨", "review_status": "reviewed"}],
    }


class Case:
    def __init__(self, page, expect, output, record, media):
        self.page, self.expect, self.output, self.record, self.media = page, expect, output, record, media
        self.accept_dialogs = True
        self.export_number = 0
        page.on("pageerror", lambda error: record["page_errors"].append(str(error)))
        page.on("download", lambda download: record["download_events"].append(download.suggested_filename))

        def respond(dialog):
            record["dialogs"].append({"type": dialog.type, "message": dialog.message,
                                       "action": "accept" if self.accept_dialogs else "dismiss"})
            dialog.accept() if self.accept_dialogs else dialog.dismiss()

        page.on("dialog", respond)

    def equal(self, actual, expected, description):
        changes = differences(actual, expected)
        self.record["assertions"].append({"description": description,
                                           "status": "FAIL" if changes else "PASS", "differences": changes})
        if changes:
            raise AssertionError(description + ": " + json.dumps(changes, ensure_ascii=False))

    def open(self, media):
        self.page.goto((SOURCE / "trace_annotate/temporal/ui/annotator.html").as_uri())
        self.choose(media)

    def choose(self, media):
        self.page.locator("#videoInput").set_input_files(media)
        self.expect(self.page.locator("#saveSegment")).to_be_enabled(timeout=45000)
        self.expect(self.page.locator("#cancelMediaLoad")).not_to_be_visible(timeout=45000)
        self.page.wait_for_function("() => !document.querySelector('video').seeking && document.querySelector('video').readyState >= 2")

    def import_doc(self, document, name="input.json"):
        target = self.output / name
        write_json(target, document)
        self.page.locator("#jsonInput").set_input_files(target)

    def export(self, name):
        self.export_number += 1
        target = self.output / (str(self.export_number).zfill(2) + "-" + name + ".json")
        with self.page.expect_download() as pending:
            self.page.locator("#exportJSON").click()
        pending.value.save_as(target)
        self.record["downloads"].append(str(target))
        return load_document(target)

    def import_fixture(self):
        document = fixture_document(self.media["a"])
        self.import_doc(document)
        self.expect(self.page.locator("#segmentList li")).to_have_count(1)
        return document


def case_hash_roundtrip(case):
    case.open(case.media["a"])
    blank = case.export("blank-content-bound")
    case.equal({key: blank[key] for key in ("schema_version", "media_size", "media_sha256")},
               {"schema_version": 2, "media_size": case.media["a"].stat().st_size,
                "media_sha256": digest(case.media["a"])},
               "New document records the full-file Python SHA256 oracle and exact size")
    document = case.import_fixture()
    case.equal(case.export("native-roundtrip"), document,
               "Content-bound native JSON retains reviewed state, IDs, CRLF and metadata exactly")
    case.equal(case.page.locator("video").evaluate("video => video.duration"), 12,
               "Trailing fixture payload does not change actual decoded WebM duration")


def case_equal_weak_identity(case):
    case.open(case.media["a"])
    first = case.import_fixture()
    case.choose(case.media["b"])
    blank_b = case.export("different-content-blank")
    case.equal(blank_b["segments"], [], "Equal-name and equal-size different content starts with no old annotations")
    case.equal(blank_b["media_sha256"], digest(case.media["b"]), "Replacement records the different complete content digest")
    case.equal(case.page.locator("video").evaluate("video => video.duration"), 12,
               "Both unequal-content fixture videos have the same actual duration")
    case.import_doc(first, "wrong-content.json")
    case.expect(case.page.locator("#status")).to_contain_text("不一致")
    case.equal(case.export("after-wrong-content-rejection"), blank_b,
               "Same filename, size and duration cannot bypass a mismatched content digest")
    case.equal(case.page.locator("#restoreDraft").is_visible(), False,
               "A content-addressed draft is not offered for another file with the same weak identity")


def case_rename_import(case):
    case.open(case.media["renamed"])
    blank = case.export("renamed-blank")
    document = fixture_document(case.media["a"])
    case.accept_dialogs = False
    case.import_doc(document, "rename-cancel.json")
    case.expect(case.page.locator("#segmentList li")).to_have_count(0)
    case.equal(case.export("after-rename-cancel"), blank,
               "Declining a content-identical filename retarget leaves the current document intact")
    case.accept_dialogs = True
    case.import_doc(document, "rename-accept.json")
    case.expect(case.page.locator("#segmentList li")).to_have_count(1)
    expected = copy.deepcopy(document)
    expected["clip"] = case.media["renamed"].name
    case.equal(case.export("after-rename-accept"), expected,
               "Accepting a content-identical rename changes only the clip basename and preserves valid reviews")
    case.equal(len(case.record["dialogs"]), 2, "Each rename attempt requires an actual explicit browser confirmation")


def case_rename_draft(case):
    case.open(case.media["a"])
    document = case.import_fixture()
    case.page.locator("#segmentList button").first.click()
    case.page.locator("#endInput").fill("4.5")
    case.expect(case.page.locator("#reviewInput")).not_to_be_checked()
    case.page.wait_for_timeout(700)
    case.page.reload()
    case.choose(case.media["renamed"])
    case.expect(case.page.locator("#restoreDraft")).to_be_visible(timeout=15000)
    case.page.locator("#restoreDraft").click()
    case.expect(case.page.locator("#endInput")).to_have_value("4.5")
    case.expect(case.page.locator("#reviewInput")).not_to_be_checked()
    case.page.locator("#saveSegment").click()
    expected = copy.deepcopy(document)
    expected["clip"] = case.media["renamed"].name
    expected["segments"][0].update(end=4.5, review_status="unreviewed")
    case.equal(case.export("renamed-dirty-draft"), expected,
               "Identical content under a new basename finds its actual draft and preserves the unsaved intended edit")
    keys = case.page.evaluate("() => Object.keys(localStorage)")
    expected_key = "traceannotate.temporal.v2:sha256:" + digest(case.media["a"])
    case.equal(expected_key in keys, True, "Persisted recovery key uses the complete digest instead of basename and size")


def case_legacy_import(case):
    case.open(case.media["a"])
    blank = case.export("before-legacy")
    legacy = fixture_document(case.media["a"])
    legacy["schema_version"] = 1
    del legacy["media_sha256"]
    case.accept_dialogs = False
    case.import_doc(legacy, "legacy-cancel.json")
    case.expect(case.page.locator("#segmentList li")).to_have_count(0)
    case.equal(case.export("after-legacy-cancel"), blank,
               "Declining migration of an unbound legacy annotation leaves the current content-bound project intact")
    case.accept_dialogs = True
    case.import_doc(legacy, "legacy-accept.json")
    case.expect(case.page.locator("#segmentList li")).to_have_count(1)
    expected = fixture_document(case.media["a"])
    expected["segments"][0]["review_status"] = "unreviewed"
    case.equal(case.export("after-legacy-accept"), expected,
               "Accepted v1 import gains actual full-file identity and invalidates every unbound review")
    case.equal(len(case.record["dialogs"]), 2, "Both legacy migration attempts ask for explicit confirmation")


def case_schema_parity(case):
    case.open(case.media["a"])
    valid = fixture_document(case.media["a"])
    cases = [valid]
    for value in (None, "", "0" * 63, "0" * 65, "A" * 64, "g" * 64, 12, True):
        candidate = copy.deepcopy(valid)
        candidate["media_sha256"] = value
        cases.append(candidate)
    candidate = copy.deepcopy(valid)
    del candidate["media_sha256"]
    cases.append(candidate)
    candidate = copy.deepcopy(valid)
    del candidate["media_size"]
    cases.append(candidate)
    for version in (1, 0, 3, True, "2"):
        candidate = copy.deepcopy(valid)
        candidate["schema_version"] = version
        cases.append(candidate)
    browser = case.page.evaluate("documents => documents.map(document => validateDocument(document).length === 0)", cases)
    python = [not validate_document(document) for document in cases]
    case.equal(browser, python, "Browser and Python agree on strict v1/v2, required identity fields and invalid digest values")
    write_json(case.output / "schema-fixtures.json", cases)
    write_json(case.output / "schema-results.json", {"browser": browser, "python": python})


def case_legacy_draft(case):
    document = fixture_document(case.media["a"])
    legacy = copy.deepcopy(document)
    legacy["schema_version"] = 1
    del legacy["media_sha256"]
    primary = legacy["segments"][0]
    envelope = {"draft_format": 1, "updated_at": "2026-10-22T00:00:00Z", "document": legacy,
                "selected_id": primary["id"], "editor": {
                    "startInput": "1", "endInput": "4.75", "labelInput": primary["label"],
                    "textInput": primary["text"].replace("\r\n", "\n"), "tagsInput": '["原始标签"]',
                    "trackInput": "主轨", "reviewInput": True}}
    key = "traceannotate.temporal.v1:" + case.media["a"].name + ":" + str(case.media["a"].stat().st_size)
    seed = json.dumps({"key": key, "envelope": envelope}, ensure_ascii=False)
    case.page.add_init_script("(() => { const seed=" + seed
                              + "; localStorage.setItem(seed.key, JSON.stringify(seed.envelope)); })();")
    case.record["fixture_injection"] = {
        "purpose": "A documented historical weak-identity draft seeded before page startup",
        "application_functions_replaced": False, "scope": "One isolated synthetic-media localStorage key"}
    case.open(case.media["a"])
    case.expect(case.page.locator("#restoreDraft")).to_be_visible(timeout=15000)
    blank = case.export("legacy-draft-before-confirm")
    case.accept_dialogs = False
    case.page.locator("#restoreDraft").click()
    case.equal(case.export("legacy-draft-after-cancel"), blank,
               "Declining a weak-identity draft migration preserves the content-bound blank project")
    case.accept_dialogs = True
    case.page.locator("#restoreDraft").click()
    case.expect(case.page.locator("#endInput")).to_have_value("4.75")
    case.expect(case.page.locator("#reviewInput")).not_to_be_checked()
    case.page.locator("#saveSegment").click()
    expected = copy.deepcopy(document)
    expected["segments"][0].update(end=4.75, review_status="unreviewed")
    case.equal(case.export("legacy-draft-after-accept"), expected,
               "Explicit legacy recovery binds the actual digest, preserves the unsaved time edit and invalidates the old review")


def case_cancel_hash(case):
    case.open(case.media["a"])
    original = case.import_fixture()
    case.page.locator("#segmentList button").first.click()
    text = "取消载入必须保留的未保存文字😀"
    case.page.locator("#textInput").fill(text)
    for attempt in range(2):
        case.page.locator("#videoInput").set_input_files(case.media["large"])
        case.expect(case.page.locator("#cancelMediaLoad")).to_be_visible(timeout=15000)
        case.page.wait_for_function("() => { const p = document.querySelector('#hashProgress'); return p && Number(p.value) > 0 && Number(p.value) < Number(p.max); }", timeout=15000)
        progress = case.page.locator("#hashProgress").evaluate("p => ({value:p.value,max:p.max})")
        case.record.setdefault("observed_hash_progress", []).append(progress)
        case.page.locator("#cancelMediaLoad").click()
        case.expect(case.page.locator("#cancelMediaLoad")).not_to_be_visible()
        case.equal(case.page.locator("#textInput").input_value(), text,
                   "Cancellation " + str(attempt + 1) + " retains the current unsaved form")
        case.page.wait_for_timeout(500)
        case.equal(case.page.locator("#textInput").input_value(), text,
                   "A cancelled background read cannot replace the dirty form later")
    case.page.locator("#saveSegment").click()
    expected = copy.deepcopy(original)
    expected["segments"][0].update(text=text, review_status="unreviewed")
    case.equal(case.export("after-repeated-cancel"), expected,
               "Repeated real in-progress cancellations preserve the current media, document and intended edit")
    case.page.locator("#videoInput").set_input_files(case.media["large"])
    case.expect(case.page.locator("#cancelMediaLoad")).to_be_visible(timeout=15000)
    case.page.wait_for_function("() => { const p=document.querySelector('#hashProgress'); return Number(p.value)>0 && Number(p.value)<Number(p.max); }")
    case.choose(case.media["b"])
    latest = case.export("after-replacement-race")
    case.equal(latest["media_sha256"], digest(case.media["b"]),
               "A newer actual file selection wins over a previous in-progress content hash")
    case.equal(latest["segments"], [], "The newest different-content file does not acquire old annotations")
    case.page.wait_for_timeout(1000)
    case.equal(case.export("after-race-settled"), latest,
               "Superseded asynchronous reads cannot overwrite the newest media after settling")


def case_bad_current_draft(case):
    document = fixture_document(case.media["a"])
    editor = {"startInput": "1", "endInput": "4", "labelInput": document["segments"][0]["label"],
              "textInput": "原始说明\n第二行😀", "tagsInput": '["原始标签"]',
              "trackInput": "主轨", "reviewInput": True}
    legacy = copy.deepcopy(document)
    legacy["schema_version"] = 1
    del legacy["media_sha256"]
    legacy_envelope = {"draft_format": 1, "document": legacy, "selected_id": "stable-identity-segment", "editor": editor}
    wrong_digest = copy.deepcopy(document)
    wrong_digest["media_sha256"] = digest(case.media["b"])
    fixtures = {
        "unknown-format": {"draft_format": 999, "document": document, "editor": editor},
        "wrong-digest": {"draft_format": 1, "document": wrong_digest,
                         "selected_id": "stable-identity-segment", "editor": editor},
    }
    content_key = "traceannotate.temporal.v2:sha256:" + digest(case.media["a"])
    legacy_key = "traceannotate.temporal.v1:" + case.media["a"].name + ":" + str(case.media["a"].stat().st_size)
    payload = json.dumps({"fixtures": fixtures, "content_key": content_key,
                          "legacy_key": legacy_key, "legacy": legacy_envelope}, ensure_ascii=False)
    case.page.add_init_script(
        "(() => { const seed=" + payload + "; const name=new URL(location.href).searchParams.get('bad_current'); "
        "if(Object.hasOwn(seed.fixtures,name)){localStorage.setItem(seed.content_key,JSON.stringify(seed.fixtures[name]));"
        "localStorage.setItem(seed.legacy_key,JSON.stringify(seed.legacy));} })();")
    case.record["fixture_injection"] = {
        "purpose": "A corrupt content-addressed draft and an otherwise valid weak-identity legacy draft",
        "application_functions_replaced": False, "scope": "Two isolated synthetic-media draft keys"}
    for variant in fixtures:
        case.page.goto((SOURCE / "trace_annotate/temporal/ui/annotator.html").as_uri() + "?bad_current=" + variant)
        case.choose(case.media["a"])
        case.expect(case.page.locator("#restoreDraft")).to_be_visible(timeout=15000)
        blank = case.export(variant + "-before")
        case.page.locator("#restoreDraft").click()
        case.expect(case.page.locator("#status")).to_contain_text("恢复失败")
        case.equal(case.export(variant + "-after"), blank,
                   variant + " current draft fails without silently importing a weaker legacy draft")
        stored = case.page.evaluate("key => JSON.parse(localStorage.getItem(key))", content_key)
        case.equal(stored, fixtures[variant], variant + " recovery leaves the original failed draft unchanged")
    case.equal(case.record["dialogs"], [], "Invalid content-bound drafts fail before any legacy migration confirmation")


def case_frame_callback_order(case):
    case.open(case.media["a"])
    original = case.import_fixture()
    case.record["fault_injection"] = {
        "purpose": "Deterministic ordering regression for a real HEVC callback-before-seeked defect",
        "browser_interfaces_replaced": ["video.requestVideoFrameCallback", "video.cancelVideoFrameCallback",
                                         "video.seeking getter"],
        "application_functions_replaced": False,
        "injected_pts_are_real_media_measurements": False,
        "scope": "One isolated synthetic-media page; restores all original browser interfaces afterward",
    }
    observations = case.page.evaluate("""() => {
        const video=document.querySelector('video');
        const info=document.querySelector('#frameInfo');
        const nativeRequest=video.requestVideoFrameCallback;
        const nativeCancel=video.cancelVideoFrameCallback;
        const ownSeeking=Object.getOwnPropertyDescriptor(video,'seeking');
        stopFrameObservation();
        let seeking=false, nextId=1000000;
        const callbacks=new Map();
        video.requestVideoFrameCallback=callback => {const id=nextId++;callbacks.set(id,callback);return id;};
        video.cancelVideoFrameCallback=id => callbacks.delete(id);
        Object.defineProperty(video,'seeking',{configurable:true,get:()=>seeking});
        function deliver(pts){
            const first=callbacks.entries().next().value;
            if(!first)throw new Error('Application did not request a frame callback');
            callbacks.delete(first[0]);first[1](performance.now(),{mediaTime:pts});
        }
        function event(type){video.dispatchEvent(new Event(type));}
        const observations={injected_first_pts:0.733333,injected_second_pts:1.433333,
                            player_current_time:video.currentTime};
        try{
            observeFrames();
            seeking=true;event('seeking');
            deliver(observations.injected_first_pts);
            observations.callback_before_seeked_pending=info.textContent;
            seeking=false;event('seeked');
            observations.callback_before_seeked_final=info.textContent;
            seeking=true;event('seeking');
            observations.new_seek_clears_previous=info.textContent;
            seeking=false;event('seeked');
            observations.seeked_before_callback_pending=info.textContent;
            deliver(observations.injected_second_pts);
            observations.seeked_before_callback_final=info.textContent;
            return observations;
        }finally{
            stopFrameObservation();
            video.requestVideoFrameCallback=nativeRequest;
            video.cancelVideoFrameCallback=nativeCancel;
            if(ownSeeking)Object.defineProperty(video,'seeking',ownSeeking);else delete video.seeking;
            observeFrames();
        }
    }""")
    write_json(case.output / "fault-injected-frame-observations.json", observations)
    case.equal("PTS" in observations["callback_before_seeked_pending"], False,
               "A callback during seeking does not prematurely claim positioning has finished")
    case.equal("最近呈现帧 PTS 0.733333" in observations["callback_before_seeked_final"], True,
               "Callback-before-seeked retains the injected callback PTS until seeked, not the player time")
    case.equal("0.733333" in observations["new_seek_clears_previous"], False,
               "A new seek clears the previous seek epoch's presented timestamp")
    case.equal("PTS" in observations["seeked_before_callback_pending"], False,
               "Seeked without a callback never reuses the previous epoch's timestamp")
    case.equal("最近呈现帧 PTS 1.433333" in observations["seeked_before_callback_final"], True,
               "Seeked-before-callback displays the later injected callback PTS")
    for name in ("callback_before_seeked_final", "seeked_before_callback_final"):
        case.equal("不保证逐帧定位" in observations[name] and "秒级位置" in observations[name], True,
                   name + " explicitly retains the second-based positioning limit")
    case.equal(case.export("after-interface-fault-injection"), original,
               "Frame callback interface fault injection never changes saved annotations or review state")


CASES = [("01-full-hash-roundtrip", case_hash_roundtrip),
         ("02-same-weak-identity", case_equal_weak_identity),
         ("03-content-identical-rename", case_rename_import),
         ("04-renamed-dirty-draft", case_rename_draft),
         ("05-legacy-import-migration", case_legacy_import),
         ("06-schema-parity", case_schema_parity),
         ("07-legacy-draft-migration", case_legacy_draft),
         ("08-cancel-hashing", case_cancel_hash),
         ("09-corrupt-current-draft", case_bad_current_draft),
         ("10-frame-callback-event-order", case_frame_callback_order)]


def prepare_media(output):
    create_synthetic_media(output)
    original = output / "general-demo.webm"
    media = {}
    # The changed trailing bytes are beyond the first two MiB. A header-only or
    # sampled digest would wrongly consider these two equal-size files equal.
    for key, suffix in (("a", b"A" * (3 * 1024 * 1024)), ("b", b"A" * (3 * 1024 * 1024 - 1) + b"B")):
        folder = output / ("media-" + key)
        folder.mkdir()
        path = folder / "same-name.webm"
        path.write_bytes(original.read_bytes() + suffix)
        media[key] = path
    media["renamed"] = output / "renamed-content.webm"
    shutil.copyfile(media["a"], media["renamed"])
    media["large"] = output / "large-cancellation.webm"
    with media["large"].open("xb") as stream:
        stream.write(original.read_bytes())
        block = b"\0" * (1024 * 1024)
        for _ in range(192):
            stream.write(block)
    assert media["a"].name == media["b"].name
    assert media["a"].stat().st_size == media["b"].stat().st_size
    assert digest(media["a"]) != digest(media["b"])
    write_json(output / "media-manifest.json", {key: {"path": str(path), "bytes": path.stat().st_size,
                                                     "sha256": digest(path), "synthetic": True}
                                                 for key, path in media.items()})
    return media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--channel", help="Optional installed channel, e.g. chrome; default bundled Chromium")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error("Output must be a new directory; existing files unchanged: " + str(exc))
    ui = SOURCE / "trace_annotate/temporal/ui/annotator.html"
    helper = ui.with_name("media-hash.js")
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable] + sys.argv, "channel": args.channel or "bundled-chromium",
              "headless": True, "user_browser_connected": False, "native_file_picker_tested": False,
              "input_method": "standard file input via Playwright set_input_files", "cases": [],
              "ui_sha256": digest(ui), "helper_path": str(helper)}
    exit_code = 0
    try:
        from playwright.sync_api import sync_playwright, expect
        media = prepare_media(output)
        report["helper_sha256"] = digest(helper)
        with sync_playwright() as playwright:
            launch = {"headless": True}
            if args.channel:
                launch["channel"] = args.channel
            browser = playwright.chromium.launch(**launch)
            report["browser_version"] = browser.version
            try:
                for name, operation in CASES:
                    folder = output / name
                    folder.mkdir()
                    record = {"name": name, "status": "RUNNING", "assertions": [], "downloads": [],
                              "dialogs": [], "page_errors": [], "download_events": []}
                    report["cases"].append(record)
                    context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1100})
                    page = context.new_page()
                    case = Case(page, expect, folder, record, media)
                    try:
                        operation(case)
                        case.equal(record["page_errors"], [], "No uncaught browser script errors")
                        record["status"] = "PASS"
                    except Exception as exc:
                        record.update(status="FAIL", error=str(exc))
                        (folder / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
                    finally:
                        try:
                            page.screenshot(path=str(folder / "workspace.png"), full_page=True)
                            record["last_status"] = page.locator("#status").inner_text()
                        except Exception as exc:
                            record["capture_error"] = str(exc)
                        write_json(folder / "case-results.json", record)
                        context.close()
                    write_json(output / "browser-results.json", report)
            finally:
                browser.close()
        if digest(ui) != report["ui_sha256"] or digest(helper) != report["helper_sha256"]:
            raise RuntimeError("Browser sources changed during testing; repeat in a new output directory")
        report["passed_cases"] = sum(item["status"] == "PASS" for item in report["cases"])
        report["failed_cases"] = len(CASES) - report["passed_cases"]
        if report["failed_cases"]:
            raise RuntimeError(str(report["failed_cases"]) + " media identity browser cases failed")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 1
    report.update(finished_at=datetime.now(timezone.utc).isoformat(), exit_code=exit_code)
    write_json(output / "browser-results.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
