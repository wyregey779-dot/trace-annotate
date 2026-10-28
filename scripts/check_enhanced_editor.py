"""Exercise progressive enhancements in an isolated Chromium browser:
1. Multi-track visual timeline with sub-lanes and interactive clips.
2. Manual review toggle with automatic invalidation on input modification.
3. Undo and Redo operations via replace_annotations command.
4. Lossless Temporal JSON export and import endpoints.
5. Emergency draft rescue snapshot export.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from trace_annotate.workspace.server import make_server
from trace_annotate.workspace.store import WorkspaceStore
from playwright.sync_api import sync_playwright, expect


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for chunk in iter(lambda: s.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checks(output_dir):
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    fixtures = Path(r"D:\099\TraceAnnotate-implementation\runs\test_ui_run_01\media")
    media_file = fixtures / "cfr.mp4"
    if not media_file.is_file():
        raise RuntimeError("cfr.mp4 fixture missing from test_ui_run_01/media")

    store = WorkspaceStore(output / "workspace")
    store.create()
    server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    summary = {"checks": [], "page_errors": []}

    def check(name):
        summary["checks"].append(name)
        print(f"[CHECK] {name}: PASS", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 1060})
        page = context.new_page()
        page.on("pageerror", lambda err: summary["page_errors"].append(str(err)))

        page.goto(server.editor_url)
        expect(page.locator("#file-input")).to_be_enabled()
        page.locator("#file-input").set_input_files(str(media_file))
        expect(page.locator("#import-status")).to_contain_text("本次导入结束", timeout=30000)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(1)

        task_id = store.list_intakes()[0]["task_id"]
        page.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(page.locator("#new-segment")).to_be_enabled(timeout=15000)

        # -------------------------------------------------------------
        # Feature 1: Manual Review Checkbox & Auto-invalidation on Edit
        # -------------------------------------------------------------
        page.locator("#segment-label").fill("Clip A: Reviewed")
        page.locator("#segment-start").fill("0.1")
        page.locator("#segment-end").fill("0.5")
        page.locator("#segment-track").fill("action-track")
        page.locator("#segment-text").fill("First reviewed clip")
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        page.locator("#segment-reviewed").check()
        expect(page.locator("#review-notice")).to_contain_text("已勾选人工复核")

        # Save with review
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#save-segment").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)

        task = store.get_task(task_id)
        assert len(task["annotations"]) == 1
        assert task["annotations"][0]["review_status"] == "reviewed"
        expect(page.locator("#segments .review-badge.reviewed")).to_contain_text("已复核")
        check("Segment saved with manual review checkbox persists review_status: reviewed and shows green badge")

        # Select clip, verify checkbox is checked
        clip1_id = task["annotations"][0]["id"]
        page.locator(f'#segments tbody button[data-annotation-id="{clip1_id}"]').click()
        expect(page.locator("#segment-reviewed")).to_be_checked()

        # Modify label -> check that review checkbox automatically unchecks and warns
        page.locator("#segment-label").fill("Clip A: Modified")
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        expect(page.locator("#review-notice")).to_contain_text("内容已修改，旧复核已自动取消")
        check("Editing segment label automatically invalidates and unchecks manual review")

        # Save modified segment without re-checking review
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#save-segment").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)

        task = store.get_task(task_id)
        assert task["annotations"][0]["review_status"] == "unreviewed"
        expect(page.locator("#segments .review-badge")).to_contain_text("未审核")
        check("Saving modified segment persists unreviewed status")

        # -------------------------------------------------------------
        # Feature 2: Multi-Track Timeline & Sub-lane Layout
        # -------------------------------------------------------------
        # Add a second clip on the same track with overlap to test sub-lane allocation
        page.locator("#new-segment").click()
        page.locator("#segment-label").fill("Clip B: Overlap")
        page.locator("#segment-start").fill("0.3")
        page.locator("#segment-end").fill("0.8")
        page.locator("#segment-track").fill("action-track")
        page.locator("#segment-reviewed").check()
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#save-segment").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)

        # Add a third clip on a different track
        page.locator("#new-segment").click()
        page.locator("#segment-label").fill("Clip C: Dialogue")
        page.locator("#segment-start").fill("0.2")
        page.locator("#segment-end").fill("0.9")
        page.locator("#segment-track").fill("speech-track")
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#save-segment").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)

        # Verify multi-track rendering
        expect(page.locator("#timeline-tracks .track-section")).to_have_count(2)
        action_track = page.locator("#timeline-tracks .track-section").first
        expect(action_track.locator(".track-lane")).to_have_count(2)
        expect(action_track.locator(".track-clip")).to_have_count(2)
        # Verify clip with reviewed status has .reviewed class and checkmark
        reviewed_clip = page.locator('.track-clip.reviewed')
        expect(reviewed_clip).to_contain_text("✓ Clip B: Overlap")
        check("Multi-track timeline correctly allocates overlapping sub-lanes and renders review markers")

        # Click track clip to select and seek
        reviewed_clip.click()
        expect(page.locator("#segment-label")).to_have_value("Clip B: Overlap")
        expect(page.locator("#segment-start")).to_have_value("0.3")
        check("Clicking timeline track clip selects segment and loads into editor form")

        # -------------------------------------------------------------
        # Feature 3: Undo and Redo Operation
        # -------------------------------------------------------------
        expect(page.locator("#undo-btn")).to_be_enabled()
        assert len(store.get_task(task_id)["annotations"]) == 3

        # Click Undo -> should revert adding Clip C
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#undo-btn").click()
        expect(page.locator("#redo-btn")).to_be_enabled(timeout=10000)
        assert len(store.get_task(task_id)["annotations"]) == 2
        expect(page.locator("#timeline-tracks .track-section")).to_have_count(1)
        check("Undo button reverts last annotation change via replace_annotations")

        # Click Redo -> should restore Clip C
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#redo-btn").click()
        expect(page.locator("#undo-btn")).to_be_enabled(timeout=10000)
        assert len(store.get_task(task_id)["annotations"]) == 3
        expect(page.locator("#timeline-tracks .track-section")).to_have_count(2)
        check("Redo button restores reverted annotation change")

        # Test Keyboard shortcut Ctrl+Z (Undo) and Ctrl+Y (Redo)
        page.locator("body").focus()
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.keyboard.press("Control+z")
        expect(page.locator("#redo-btn")).to_be_enabled(timeout=10000)
        assert len(store.get_task(task_id)["annotations"]) == 2
        check("Ctrl+Z keyboard shortcut triggers Undo")

        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.keyboard.press("Control+y")
        expect(page.locator("#undo-btn")).to_be_enabled(timeout=10000)
        assert len(store.get_task(task_id)["annotations"]) == 3
        check("Ctrl+Y keyboard shortcut triggers Redo")

        # -------------------------------------------------------------
        # Feature 4: Lossless Temporal JSON Export
        # -------------------------------------------------------------
        with page.expect_download() as dl_info:
            page.locator("#export-temporal-btn").click()
        dl_path = output / "exported_temporal.json"
        dl_info.value.save_as(dl_path)
        exported_doc = json.loads(dl_path.read_text(encoding="utf-8"))
        assert exported_doc["schema_version"] == 2
        assert len(exported_doc["segments"]) == 3
        assert exported_doc["segments"][1]["label"] == "Clip B: Overlap"
        assert exported_doc["segments"][1]["review_status"] == "reviewed"
        check("Export Temporal JSON produces valid temporal schema v2 preserving review statuses")

        # -------------------------------------------------------------
        # Feature 5: Emergency Draft Rescue
        # -------------------------------------------------------------
        page.locator("#segment-label").fill("Uncommitted Emergency Note")
        with page.expect_download() as dl_info:
            page.locator("#rescue-draft-btn").click()
        rescue_path = output / "draft_rescue.json"
        dl_info.value.save_as(rescue_path)
        rescue_data = json.loads(rescue_path.read_text(encoding="utf-8"))
        assert rescue_data["task_id"] == task_id
        assert rescue_data["current_form"]["label"] == "Uncommitted Emergency Note"
        check("Emergency Draft Rescue exports uncommitted in-memory form and task snapshot")

        # -------------------------------------------------------------
        # Feature 6: Temporal JSON Import
        # -------------------------------------------------------------
        test_import_doc = {
            "kind": "temporal",
            "schema_version": 2,
            "clip": "cfr.mp4",
            "duration": exported_doc["duration"],
            "media_size": exported_doc["media_size"],
            "media_sha256": exported_doc["media_sha256"],
            "segments": [
                {
                    "id": "imported-seg-1",
                    "start": 0.05,
                    "end": 0.45,
                    "label": "Imported Shot 1",
                    "text": "Commentary on shot 1",
                    "track": "camera",
                    "tags": ["wide", "exterior"],
                    "review_status": "reviewed"
                },
                {
                    "id": "imported-seg-2",
                    "start": 0.5,
                    "end": 0.95,
                    "label": "Imported Shot 2",
                    "text": "Commentary on shot 2",
                    "track": "camera",
                    "tags": ["close-up"],
                    "review_status": "unreviewed"
                }
            ],
            "metadata": {"editor": "classic-import-test"}
        }
        import_path = output / "to_import.json"
        import_path.write_text(json.dumps(test_import_doc, ensure_ascii=False, indent=2), encoding="utf-8")

        # Set input files on hidden #import-temporal-file
        page.locator("#import-temporal-file").set_input_files(str(import_path))
        expect(page.locator("#confirm-dialog")).to_be_visible()
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/import-temporal")):
            page.locator("#confirm-accept").click()

        # Wait for task reload and notice
        expect(page.locator("#status.success")).to_contain_text("成功导入", timeout=15000)
        task_after = store.get_task(task_id)
        assert len(task_after["annotations"]) == 2
        assert task_after["annotations"][0]["label"] == "Imported Shot 1"
        assert task_after["annotations"][0]["review_status"] == "reviewed"
        assert task_after["annotations"][0]["id"] == "imported-seg-1"
        assert task_after["annotations"][1]["label"] == "Imported Shot 2"
        check("Temporal JSON import endpoint seamlessly imports and replaces task annotations preserving legacy IDs")

        if summary["page_errors"]:
            raise AssertionError(f"Browser had unexpected page errors: {summary['page_errors']}")

        browser.close()

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    store.close()
    print("\nALL PROGRESSIVE ENHANCEMENT CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    output_target = Path(r"D:\099\TraceAnnotate-implementation\runs\test_enhanced_ui_run")
    run_checks(output_target)
