"""Real-browser check for the workspace vision flow.

Creates two local synthetic clips, runs the actual editor server, drives the
vision job from the UI, applies candidates, edits and saves one, reloads the
page, exports professional JSON and checks the exported annotation text.
It also switches tasks while a job is active and requires that the late result
cannot attach to the newly selected task.

Only local data is touched. An existing output directory is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from trace_annotate.workspace.server import make_server
from trace_annotate.workspace.store import WorkspaceStore


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_clip(ffmpeg, target, first, second):
    command = [ffmpeg, "-hide_banner", "-v", "error", "-y",
               "-f", "lavfi", "-i", f"color=c={first}:s=320x180:r=25:d=3",
               "-f", "lavfi", "-i", f"color=c={second}:s=320x180:r=25:d=3",
               "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", str(target)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--channel", default=None, help="Installed browser channel, e.g. chrome")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.ffmpeg:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing output directory")
    args.output.mkdir(parents=True)
    a = args.output / "vision-a.mp4"
    b = args.output / "vision-b.mp4"
    make_clip(args.ffmpeg, a, "blue", "red")
    make_clip(args.ffmpeg, b, "green", "yellow")
    summary = {"clips": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                         for path in (a, b)}, "checks": [], "page_errors": []}

    def check(name):
        summary["checks"].append(name)
        print(name + ": PASS", flush=True)
        write_json(args.output / "summary.json", summary)

    store = WorkspaceStore(args.output / "workspace")
    store.create()
    store.import_source(a)
    store.import_source(b)
    server = make_server(store)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    from playwright.sync_api import sync_playwright, expect
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=args.headless,
                                            **({"channel": args.channel} if args.channel else {}))
        context = browser.new_context(accept_downloads=True, viewport={"width": 1366, "height": 768})
        page = context.new_page()
        page.on("pageerror", lambda error: summary["page_errors"].append(str(error)))

        def task_id(name):
            return next(item["task_id"] for item in store.list_intakes()
                        if item["display_name"] == name)

        first, second = task_id(a.name), task_id(b.name)

        def select(target):
            target.locator(f'#task-list button[data-task-id="{target_id}"]').click()
            expect(target.locator(f'#task-list button[data-task-id="{target_id}"]')).to_have_attribute(
                "aria-current", "true")
            expect(target.locator("#new-segment")).to_be_enabled(timeout=30000)

        page.goto(server.editor_url)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(2, timeout=60000)
        target_id = first
        select(page)
        expect(page.locator("#vision-run")).to_be_enabled(timeout=30000)
        page.locator("#vision-run").click()
        expect(page.locator("#vision-job")).to_contain_text("已完成", timeout=120000)
        expect(page.locator("#vision-apply")).to_be_visible(timeout=30000)
        summary["job_panel"] = page.locator("#vision-job").inner_text()
        check("Vision job progress reaches a terminal state in the UI")

        page.locator("#vision-apply").click()
        expect(page.locator("#vision-candidates tbody tr").first).to_be_visible(timeout=60000)
        rows = page.locator("#vision-candidates tbody tr").count()
        if rows < 1:
            raise AssertionError("No applied candidate became visible")
        summary["candidate_rows"] = rows
        persisted = store.get_task(target_id)
        if not persisted["annotations"]:
            raise AssertionError("Apply did not persist annotations")
        check("Applied candidates are listed from refreshVisionStatus")

        target_id = second
        select(page)
        expect(page.locator("#vision-job")).to_be_hidden(timeout=30000)
        if page.locator("#vision-candidates").is_visible():
            raise AssertionError("Candidate table leaked into the next task")
        check("Switching tasks clears the prior vision job and candidates")

        target_id = first
        select(page)
        expect(page.locator("#vision-candidates tbody tr").first).to_be_visible(timeout=30000)
        page.locator("#vision-candidates tbody tr").first.locator("button.segment-select").click()
        expect(page.locator("#segment-text")).to_be_editable(timeout=30000)
        marker = "vision-ui-edit-" + str(time.monotonic_ns())
        page.locator("#segment-text").fill(marker)
        expect(page.locator("#save-state")).to_contain_text("草稿未保存", timeout=10000)
        before = store.get_task(target_id)["revision"]
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#save-segment").click()
        if pending.value.status != 200:
            raise AssertionError("UI save did not return 200")
        expect(page.locator("#save-state")).to_contain_text("已保存", timeout=30000)
        if store.get_task(target_id)["revision"] != before + 1:
            raise AssertionError("UI save did not advance the revision")
        page.reload()
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(2, timeout=60000)
        select(page)
        expect(page.locator("#vision-candidates tbody tr").first).to_be_visible(timeout=30000)
        page.locator("#vision-candidates tbody tr").first.locator("button.segment-select").click()
        expect(page.locator("#segment-text")).to_have_value(marker, timeout=30000)
        check("Candidate edit survives save and page reload")

        rows = page.locator("#vision-candidates tbody tr").count()
        if rows < 2:
            raise AssertionError("Both accept and reject need two pending candidates")
        page.locator("#vision-candidates tbody tr").first.get_by_role("button", name="接受").click()
        expect(page.locator("#vision-candidates tbody tr")).to_have_count(rows - 1, timeout=60000)
        accepted = [item for item in store.get_task(target_id)["annotations"]
                    if item.get("review_status") == "reviewed"
                    and item.get("provenance", {}).get("origin") == "manual"]
        if not accepted:
            raise AssertionError("Accept did not create a manual, reviewed annotation")
        check("Explicit accept upgrades the candidate to a manual reviewed annotation")
        before = len(store.get_task(target_id)["annotations"])
        page.locator("#vision-candidates tbody tr").first.get_by_role("button", name="拒绝").click()
        expect(page.locator("#confirm-dialog")).to_be_visible(timeout=10000)
        page.locator("#confirm-accept").click()
        expect(page.locator("#vision-candidates tbody tr")).to_have_count(rows - 2, timeout=60000)
        if len(store.get_task(target_id)["annotations"]) != before - 1:
            raise AssertionError("Reject did not delete exactly one annotation")
        check("Explicit reject removes the candidate through a new revision")

        page.locator("#new-segment").click()
        expect(page.locator("#segment-label")).to_be_editable(timeout=30000)
        page.locator("#segment-label").fill("feature-edit")
        page.locator("#segment-start").fill("0.5")
        page.locator("#segment-end").fill("4.5")
        page.locator("#segment-text").fill("feature text")
        page.locator("#segment-track").fill("feature-track")
        page.locator("#segment-tags").fill("alpha")
        page.locator("#segment-transition").select_option("cut")
        page.locator("#gradient-start").fill("1")
        page.locator("#gradient-end").fill("2")
        before = store.get_task(target_id)["revision"]
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#save-segment").click()
        if pending.value.status != 200:
            raise AssertionError("Feature segment save did not return 200")
        expect(page.locator("#save-state")).to_contain_text("已保存", timeout=30000)
        task = store.get_task(target_id)
        if task["revision"] != before + 1:
            raise AssertionError("Feature segment save did not advance the revision")
        feature = next(item for item in task["annotations"] if item["label"] == "feature-edit")
        for tag in ("transition:cut", "gradient-start:1", "gradient-end:2", "alpha"):
            if tag not in feature["tags"]:
                raise AssertionError(f"Feature tags lack {tag}: {feature['tags']}")
        check("Transition and gradient fields persist as reserved tags")

        page.locator("#segments button.segment-select").filter(has_text="feature-edit").first.click()
        expect(page.locator("#segment-text")).to_have_value("feature text", timeout=30000)
        page.locator("#compare-refresh").click()
        expect(page.locator('#boundary-compare .compare-cell[data-available="true"]')).to_have_count(4, timeout=60000)
        page.wait_for_function("""() => ['start-before','start-after','end-before','end-after']
            .every(slot => document.querySelector('[data-slot="' + slot + '"] img').naturalWidth > 0)""",
            timeout=30000)
        natural = page.evaluate("""() => ['start-before','start-after','end-before','end-after']
            .map(slot => document.querySelector('[data-slot="' + slot + '"] img').naturalWidth)""")
        if any(width <= 0 for width in natural):
            raise AssertionError(f"Boundary comparison image did not decode: {natural}")
        if "SHA256" not in page.locator("#compare-status").inner_text():
            raise AssertionError("Boundary comparison did not report hash verification")
        summary["boundary_compare"] = {"slots": 4, "natural_widths": natural}
        check("Boundary comparison loads and hash-verifies four adjacent frames")

        for _ in range(3):
            page.locator("#compare-refresh").click()
        expect(page.locator('#boundary-compare .compare-cell[data-available="true"]')).to_have_count(4, timeout=30000)
        if "SHA256" not in page.locator("#compare-status").inner_text():
            raise AssertionError("Repeated boundary refresh lost the verified status")
        check("Repeated boundary refresh coalesces without losing the four frames")

        page.locator("#compare-refresh").click()
        page.locator(f'#task-list button[data-task-id="{second}"]').click()
        expect(page.locator(f'#task-list button[data-task-id="{second}"]')).to_have_attribute("aria-current", "true")
        page.wait_for_function(
            """() => document.querySelectorAll('#boundary-compare .compare-cell[data-available="true"]').length === 0""",
            timeout=10000)
        expect(page.locator("#compare-status")).to_contain_text("选择片段", timeout=10000)
        page.locator(f'#task-list button[data-task-id="{first}"]').click()
        expect(page.locator(f'#task-list button[data-task-id="{first}"]')).to_have_attribute("aria-current", "true")
        page.locator("#segments button.segment-select").filter(has_text="feature-edit").first.click()
        expect(page.locator("#segment-text")).to_have_value("feature text", timeout=30000)
        check("Fast task switch discards an in-flight boundary load")

        seek_before = page.locator("#seek").input_value()
        page.locator("#segment-text").click()
        page.keyboard.press("End")
        page.keyboard.press("Space")
        page.keyboard.press("i")
        if page.locator("#segment-text").input_value() != "feature text i":
            raise AssertionError("Space/I were not delivered to the focused text field")
        if page.locator("#seek").input_value() != seek_before:
            raise AssertionError("A text-field keystroke changed the frame seek field")
        page.locator("#segment-text").fill("feature text")
        check("Editing shortcuts do not fire while a text field has focus")

        page.locator("#split-at").fill("2.5")
        before = store.get_task(target_id)["revision"]
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#split-segment").click()
        if pending.value.status != 200:
            raise AssertionError("Split did not return 200")
        task = store.get_task(target_id)
        if task["revision"] != before + 1:
            raise AssertionError("Split did not advance the revision")
        split = [item for item in task["annotations"]
                 if item.get("label") == "feature-edit" and item.get("track") == "feature-track"]
        if len(split) != 2:
            raise AssertionError(f"Split did not produce two feature segments: {len(split)}")
        if not any(item["end"]["seconds"] == {"n": "5", "d": "2"} for item in split):
            raise AssertionError("Split left segment lacks the exact 2.5s end")
        if not any(item["start"]["seconds"] == {"n": "5", "d": "2"} for item in split):
            raise AssertionError("Split right segment lacks the exact 2.5s start")
        check("Split uses one atomic replace_annotations command with exact rationals")

        before_count = len(task["annotations"])
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#merge-next").click()
        if pending.value.status != 200:
            raise AssertionError("Merge did not return 200")
        task = store.get_task(target_id)
        merged = [item for item in task["annotations"]
                  if item.get("label") == "feature-edit" and item.get("track") == "feature-track"]
        if len(merged) != 1 or len(task["annotations"]) != before_count - 1:
            raise AssertionError("Merge did not collapse the two feature segments")
        if merged[0]["start"]["seconds"] != {"n": "1", "d": "2"} or merged[0]["end"]["seconds"] != {"n": "9", "d": "2"}:
            raise AssertionError("Merged feature segment has the wrong range")
        check("Merge next refuses gaps and collapses the adjacent same-label segment")

        settled_revision = store.get_task(target_id)["revision"]
        expect(page.locator("#save-state")).to_contain_text("已保存", timeout=10000)
        expect(page.locator("#revision-label")).to_contain_text(f"r{settled_revision}", timeout=10000)
        task = store.get_task(target_id)
        base_text = next(item for item in task["annotations"] if item["id"] == feature["id"])["text"]
        stale_head = task["revision"]
        background_id = "ann-" + "4" * 64
        background = {
            "id": background_id, "type": "segment", "label": "background-note",
            "text": "written outside the open page", "track": "main",
            "tags": [], "start": {"seconds": {"n": "2", "d": "1"}},
            "end": {"seconds": {"n": "3", "d": "1"}},
            "attributes": {}, "evidence_ids": [],
            "provenance": {"origin": "manual", "actor_id": "local-user", "job_id": None},
            "review_status": "unreviewed",
        }
        stale_command = "cmd-" + "4" * 64
        stale_body = {"command_id": stale_command, "expected_revision": stale_head,
                      "expected_sha256": task["content_sha256"],
                      "operation": "add_annotation", "payload": {"annotation": background}}
        store.apply_task_command(
            command_id=stale_command, task_id=target_id, expected_revision=stale_head,
            expected_sha256=task["content_sha256"], operation="add_annotation",
            payload={"annotation": background},
            request_body=json.dumps(stale_body, ensure_ascii=False, separators=(",", ":")).encode())
        if store.get_task(target_id)["revision"] != stale_head + 1:
            raise AssertionError("Background fixture did not advance the revision")

        page.locator("#segment-text").fill("stale edit")
        expect(page.locator("#save-state")).to_contain_text("草稿未保存")
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#save-segment").click()
        if pending.value.status != 409:
            raise AssertionError(f"Stale save did not return 409: {pending.value.status}")
        expect(page.locator("#save-state")).to_contain_text("修订冲突")
        if page.locator("#segment-text").input_value() != "stale edit":
            raise AssertionError("A stale save discarded the unsaved draft")
        feature_after_conflict = next(item for item in store.get_task(target_id)["annotations"]
                                      if item["id"] == feature["id"])
        if feature_after_conflict["text"] != base_text:
            raise AssertionError("A stale save changed stored text")
        check("Stale revision is rejected while the unsaved draft stays in the editor")

        page.reload()
        select(page)
        page.locator("#segments button.segment-select").filter(has_text="feature-edit").first.click()
        expect(page.locator("#segment-text")).to_have_value(base_text, timeout=30000)
        check("Reload after a revision conflict shows the persisted revision")

        page.context.set_offline(True)
        page.locator("#segment-text").fill("offline edit")
        page.locator("#save-segment").click()
        expect(page.locator("#pending-panel")).to_be_visible(timeout=10000)
        if page.locator("#segment-text").input_value() != "offline edit":
            raise AssertionError("A disconnected save discarded the draft")
        if store.get_task(target_id)["revision"] != stale_head + 1:
            raise AssertionError("A disconnected save reached the store")
        page.context.set_offline(False)
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#pending-retry").click()
        if pending.value.status != 200:
            raise AssertionError(f"Pending retry did not return 200: {pending.value.status}")
        expect(page.locator("#save-state")).to_contain_text("已保存")
        feature_after_retry = next(item for item in store.get_task(target_id)["annotations"]
                                   if item["id"] == feature["id"])
        if feature_after_retry["text"] != "offline edit":
            raise AssertionError("Pending retry did not commit the retained draft")
        check("Disconnected save keeps the draft and retry commits the same command")

        page.locator("#new-segment").click()
        expect(page.locator("#segment-fields")).to_be_enabled(timeout=10000)
        page.locator("#segment-label").fill("中文片段")
        page.locator("#segment-text").click()
        cdp = page.context.new_cdp_session(page)
        cdp.send("Input.imeSetComposition", {
            "text": "中文输入", "selectionStart": 4, "selectionEnd": 4})
        page.wait_for_timeout(200)
        cdp.send("Input.insertText", {"text": "中文输入"})
        if page.locator("#segment-text").input_value() != "中文输入":
            raise AssertionError("IME composition did not commit the Chinese text")
        ime_revision = store.get_task(target_id)["revision"]
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#save-segment").click()
        if pending.value.status != 200:
            raise AssertionError(f"Chinese IME save did not return 200: {pending.value.status}")
        if store.get_task(target_id)["revision"] != ime_revision + 1:
            raise AssertionError("Chinese IME save did not advance the revision")
        ime = next(item for item in store.get_task(target_id)["annotations"]
                   if item.get("label") == "中文片段")
        if ime["text"] != "中文输入":
            raise AssertionError("Chinese IME text was not persisted")
        check("Chinese IME composition commits and persists")

        task = store.get_task(target_id)
        now = {"n": "1", "d": "2"}
        later = {"n": "9", "d": "2"}
        window = {"start": {"seconds": now}, "end": {"seconds": later}}
        claim_id = "clm-" + "1" * 64
        semantic_id = "ann-" + "2" * 64
        fixture = {
            "id": semantic_id, "type": "segment", "label": "semantic-fixture",
            "text": "semantic fixture for UI measurement editing", "track": "semantic-track",
            "tags": [], "start": {"seconds": now}, "end": {"seconds": later},
            "attributes": {"semantic": {"version": "visual-semantics-v2", "claims": [{
                "id": claim_id, "claim_type": "measured", "statement": "fixture measurement",
                "subject_ref": None, "time_window": window, "evidence_ids": [],
                "method": {"id": "manual-fixture", "version": "1", "parameters": {}},
                "model_run_id": None, "score": None, "score_kind": "none",
                "uncertainty_reason": "fixture only",
                "measurement": {"value": 0.25, "unit": "ratio", "basis": "declared",
                                "analysis_window": window},
            }]}},
            "evidence_ids": [],
            "provenance": {"origin": "manual", "actor_id": "local-user", "job_id": None},
            "review_status": "unreviewed",
        }
        command_id = "cmd-" + "3" * 64
        body = {"command_id": command_id, "expected_revision": task["revision"],
                "expected_sha256": task["content_sha256"], "operation": "add_annotation",
                "payload": {"annotation": fixture}}
        request_body = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        receipt = store.apply_task_command(command_id=command_id, task_id=target_id,
                                           expected_revision=task["revision"],
                                           expected_sha256=task["content_sha256"],
                                           operation="add_annotation",
                                           payload={"annotation": fixture}, request_body=request_body)
        if not receipt.get("applied"):
            raise AssertionError(f"Semantic fixture was not applied: {receipt}")

        page.reload()
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(2, timeout=60000)
        select(page)
        page.locator("#segments button.segment-select").filter(has_text="semantic-fixture").first.click()
        expect(page.locator("#measurement-box")).to_be_visible(timeout=30000)
        expect(page.locator("#measurement-value")).to_have_value("0.25")
        page.locator("#measurement-value").fill("0.75")
        page.locator("#measurement-unit").fill("ratio")
        page.locator("#measurement-basis").select_option("declared")
        before = store.get_task(target_id)["revision"]
        with page.expect_response(lambda response: response.request.method == "POST"
                                  and response.url.endswith(f"/tasks/{target_id}/commands")) as pending:
            page.locator("#measurement-save").dblclick()
        if pending.value.status != 200:
            raise AssertionError("Semantic measurement update did not return 200")
        task = store.get_task(target_id)
        if task["revision"] != before + 1:
            raise AssertionError("Semantic measurement update did not advance the revision")
        claim = next(item for item in task["annotations"]
                     if item["id"] == semantic_id)["attributes"]["semantic"]["claims"][0]
        if claim["measurement"]["value"] != 0.75:
            raise AssertionError("Semantic measurement value was not updated")
        if claim["measurement"]["analysis_window"] != window:
            raise AssertionError("Semantic measurement analysis_window changed")
        check("Semantic measurement editor updates value only and preserves its window")

        page.locator("#export-text-format").select_option("csv")
        with page.expect_download() as download:
            page.locator("#export-text-btn").click()
        csv_path = args.output / "export.csv"
        download.value.save_as(csv_path)
        csv_bytes = csv_path.read_bytes()
        if not csv_bytes.startswith(b"\xef\xbb\xbf"):
            raise AssertionError("CSV export lacks the UTF-8 BOM")
        csv_text = csv_bytes.decode("utf-8-sig")
        if "feature-edit" not in csv_text or "semantic-fixture" not in csv_text or "中文输入" not in csv_text:
            raise AssertionError("CSV export lacks UI-created annotations")
        summary["text_exports"] = {"csv": {"sha256": digest(csv_path), "bytes": csv_path.stat().st_size}}

        page.locator("#export-text-format").select_option("srt")
        expect(page.locator("#export-track")).to_be_visible(timeout=10000)
        page.locator("#export-track").select_option("feature-track")
        with page.expect_download() as download:
            page.locator("#export-text-btn").click()
        srt_path = args.output / "export.srt"
        download.value.save_as(srt_path)
        srt_text = srt_path.read_text(encoding="utf-8-sig")
        if "00:00:00,500 --> 00:00:04,500" not in srt_text or "offline edit" not in srt_text:
            raise AssertionError(f"SRT export content unexpected: {srt_text[:200]}")
        summary["text_exports"]["srt"] = {"sha256": digest(srt_path), "bytes": srt_path.stat().st_size}

        page.locator("#export-text-format").select_option("vtt")
        page.locator("#export-track").select_option("feature-track")
        with page.expect_download() as download:
            page.locator("#export-text-btn").click()
        vtt_path = args.output / "export.vtt"
        download.value.save_as(vtt_path)
        vtt_text = vtt_path.read_text(encoding="utf-8-sig")
        if not vtt_text.startswith("WEBVTT\n\n") or "00:00:00.500 --> 00:00:04.500" not in vtt_text:
            raise AssertionError(f"WebVTT export content unexpected: {vtt_text[:200]}")
        summary["text_exports"]["vtt"] = {"sha256": digest(vtt_path), "bytes": vtt_path.stat().st_size}
        check("UI exports CSV, SRT and WebVTT with track selection")

        with page.expect_download() as download:
            page.locator("#export-professional-btn").click()
        path = args.output / "professional-export.json"
        download.value.save_as(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        exported = [item.get("text") for item in document["task"]["annotations"]]
        if marker not in exported:
            raise AssertionError("Exported professional JSON lacks the UI edit")
        summary["export"] = {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}
        check("Professional export matches the UI-edited annotation")

        if summary["page_errors"]:
            raise AssertionError("Browser page errors: " + str(summary["page_errors"][:3]))
        context.close()
        browser.close()
        playwright.stop()
    finally:
        server.shutdown()
        server.server_close()
        server.vision_jobs.close(timeout=10)
        store.close()
    summary["ok"] = True
    write_json(args.output / "summary.json", summary)
    print("VISION UI ACCEPTANCE PASS", flush=True)


if __name__ == "__main__":
    main()
