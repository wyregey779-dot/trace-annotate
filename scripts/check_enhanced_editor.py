"""Run enhancement browser regressions against source or an isolated installed wheel.

Requires a NEW --output directory. Generates only synthetic media, preserves
failure evidence, and never removes prior output. Installed mode requires -I
and reuses the workspace acceptance resource/fixture identity checks.
This is headless workflow evidence, not human listening or semantic accuracy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import traceback

SOURCE = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_harness(installed_wheel):
    # Load test support by file, never the product from the checkout in installed
    # mode. The shared harness verifies site-packages and every wheel resource.
    if installed_wheel is not None and not sys.flags.isolated:
        raise RuntimeError("Installed acceptance requires the target venv Python with -I")
    support = SOURCE / "scripts" / "check_workspace_editor.py"
    spec = importlib.util.spec_from_file_location("traceannotate_workspace_acceptance", support)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(output, media_file, harness, channel, summary):
    from playwright.sync_api import sync_playwright, expect
    store = harness.WorkspaceStore(output / "workspace")
    server = thread = playwright = browser = context = page = None
    summary.update(checks=[], page_errors=[], external_requests=[], cleanup_errors=[])

    def check(name):
        summary["checks"].append(name)
        print("[CHECK] " + name + ": PASS", flush=True)
        write_json(output / "summary.json", summary)

    try:
        store.create()
        server = harness.make_server(store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        playwright = sync_playwright().start()
        launch = {"headless": True}
        if channel:
            launch["channel"] = channel
        browser = playwright.chromium.launch(**launch)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 1060})
        page = context.new_page()
        page.on("pageerror", lambda err: summary["page_errors"].append(str(err)))
        page.on("request", lambda request: summary["external_requests"].append(request.url)
                if request.url.startswith(("http:", "https:")) and not request.url.startswith(server.origin + "/") else None)

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

        # A programmatic button change must invalidate review, too. Returning
        # to the old time does not constitute a new independent review action.
        page.locator("#use-video-end").click()
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        page.locator("#segment-end").fill("0.5")
        expect(page.locator("#segment-reviewed")).not_to_be_checked()
        check("Video-end button invalidates review and restoring a boundary does not revive it")

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
        check("Undo button conditionally reverts the last annotation change")

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
        # Feature 4: Temporal segment projection (not a full workspace backup)
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
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/drafts")) as draft_saved:
            page.locator("#segment-label").fill("Uncommitted Emergency Note")
        assert draft_saved.value.status == 200, draft_saved.value.text()
        with page.expect_download() as dl_info:
            page.locator("#rescue-draft-btn").click()
        rescue_path = output / "draft_rescue.json"
        dl_info.value.save_as(rescue_path)
        rescue_data = json.loads(rescue_path.read_text(encoding="utf-8"))
        assert rescue_data["task_id"] == task_id
        assert rescue_data["current_form"]["label"] == "Uncommitted Emergency Note"
        check("Emergency Draft Rescue exports uncommitted in-memory form and task snapshot")
        unchanged_task = store.get_task(task_id)
        recovered_draft = context.new_page()
        recovered_draft.goto(server.editor_url)
        expect(recovered_draft.locator("#restore-saved-draft")).to_be_enabled()
        recovered_draft.locator("#restore-saved-draft").click()
        expect(recovered_draft.locator("#confirm-dialog")).to_be_visible()
        recovered_draft.locator("#confirm-accept").click()
        expect(recovered_draft.locator("#segment-label")).to_have_value("Uncommitted Emergency Note")
        assert store.get_task(task_id) == unchanged_task
        recovered_draft.close()
        check("Acknowledged project draft restores in a fresh tab without applying an annotation command")

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
        expect(page.locator("#status.success")).to_contain_text("导入已确认保存", timeout=15000)
        task_after = store.get_task(task_id)
        assert len(task_after["annotations"]) == 2
        assert task_after["annotations"][0]["label"] == "Imported Shot 1"
        assert task_after["annotations"][0]["review_status"] == "unreviewed"
        assert task_after["annotations"][0]["id"] == "imported-seg-1"
        assert task_after["annotations"][1]["label"] == "Imported Shot 2"
        check("Temporal import preserves legacy IDs and marks historical reviews unreviewed for current work")

        # Hold actual File.text() completion, then try to select another task.
        # The import must retain its original task/base before any await.
        second_id = store.import_source(media_file)["task_id"]
        untouched_second = store.get_task(second_id)
        page.locator("#refresh-workspace").click()
        expect(page.locator("#save-segment")).to_be_enabled()
        page.evaluate("""() => {
            const original = File.prototype.text;
            window.traceReadEntered = false;
            File.prototype.text = function() {
                window.traceReadEntered = true;
                return new Promise(resolve => {window.traceReleaseRead = async () => resolve(await original.call(this));});
            };
            window.traceOriginalFileText = original;
        }""")
        page.locator("#import-temporal-file").set_input_files(str(import_path))
        expect(page.locator("#confirm-dialog")).to_be_visible()
        page.locator("#confirm-accept").click()
        page.wait_for_function("() => window.traceReadEntered === true")
        page.locator(f'#task-list button[data-task-id="{second_id}"]').click()
        expect(page.locator(f'#task-list button[data-task-id="{task_id}"]')).to_have_attribute("aria-current", "true")
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/import-temporal")):
            page.evaluate("() => {File.prototype.text = window.traceOriginalFileText; window.traceReleaseRead();}")
        expect(page.locator("#save-segment")).to_be_enabled()
        assert store.get_task(second_id) == untouched_second
        check("File-read barrier and attempted task switch cannot redirect an import to another task")

        # Reproduce the destructive old sequence using actual tabs and a
        # refresh; a fresh task revision must not reauthorize old snapshots.
        page.locator('#segments tbody button[data-annotation-id="imported-seg-1"]').click()
        page.locator("#segment-label").fill("A saved change before B")
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            page.locator("#save-segment").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)
        other = context.new_page()
        other.on("pageerror", lambda error: summary["page_errors"].append(str(error)))
        other.goto(server.editor_url)
        other.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(other.locator("#new-segment")).to_be_enabled(timeout=15000)
        other.locator("#new-segment").click()
        other.locator("#segment-label").fill("B independently saved annotation")
        other.locator("#segment-start").fill("1.1")
        other.locator("#segment-end").fill("1.8")
        with other.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            other.locator("#save-segment").click()
        expect(other.locator("#save-segment")).to_be_enabled(timeout=10000)
        before_undo = store.get_task(task_id)
        assert len(before_undo["annotations"]) == 3
        page.locator("#refresh-workspace").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")) as rejected:
            page.locator("#undo-btn").click()
        assert rejected.value.status == 409
        rejection = rejected.value.json()
        assert rejection["error"]["code"] == "HISTORY_DIVERGED"
        expect(page.locator("#status.success")).to_have_count(0)
        assert store.get_task(task_id) == before_undo
        write_json(output / "cross-tab-history-conflict.json", {
            "response_status": rejected.value.status, "response": rejection,
            "before": before_undo, "after": store.get_task(task_id)})
        page.screenshot(path=str(output / "cross-tab-history-conflict.png"), full_page=True)
        other.close()
        check("Stale cross-tab undo after refresh is rejected without removing B's saved annotation")

        # Force an unknown save outcome and failed receipt lookup. Rescue must
        # remain actionable outside the disabled editing fieldset.
        page.locator("#refresh-workspace").click()
        expect(page.locator("#save-segment")).to_be_enabled(timeout=10000)
        page.locator('#segments tbody button[data-annotation-id="imported-seg-1"]').click()
        page.locator("#segment-label").fill("Unconfirmed offline rescue form")
        page.route("**/commands/*", lambda route: route.abort("failed"))
        page.route("**/tasks/" + task_id + "/commands", lambda route: route.abort("failed"), times=1)
        page.locator("#save-segment").click()
        expect(page.locator("#pending-retry")).to_be_visible(timeout=15000)
        expect(page.locator("#save-segment")).to_be_disabled()
        expect(page.locator("#rescue-draft-btn")).to_be_enabled()
        with page.expect_download() as rescued:
            page.locator("#rescue-draft-btn").click()
        pending_rescue = output / "pending-draft-rescue.json"
        rescued.value.save_as(pending_rescue)
        pending_data = json.loads(pending_rescue.read_text(encoding="utf-8"))
        assert pending_data["task_id"] == task_id
        assert pending_data["current_form"]["label"] == "Unconfirmed offline rescue form"
        assert pending_data["pending_command"]
        page.screenshot(path=str(output / "pending-save-rescue.png"), full_page=True)
        check("Unknown-save rescue remains enabled and downloads the original pending command and form")

        # Restore the actual downloaded rescue into a fresh tab. The blocked
        # command never committed; restore may query it but must not submit it.
        unchanged = store.get_task(task_id)
        restored = context.new_page()
        restored.goto(server.editor_url)
        expect(restored.locator("#save-segment")).to_be_enabled(timeout=15000)
        restored.locator("#restore-draft-file").set_input_files(str(pending_rescue))
        expect(restored.locator("#confirm-dialog")).to_be_visible()
        restored.locator("#confirm-accept").click()
        expect(restored.locator("#segment-label")).to_have_value("Unconfirmed offline rescue form")
        expect(restored.locator("#pending-retry")).to_be_visible()
        assert store.get_task(task_id) == unchanged
        # Explicit receipt recovery is the only step that may resend the exact
        # original request; the rescue import above did not automatically do it.
        with restored.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/tasks/{task_id}/commands")):
            restored.locator("#pending-retry").click()
        expect(restored.locator("#save-segment")).to_be_enabled(timeout=15000)
        assert store.get_task(task_id)["annotations"][0]["label"] == "Unconfirmed offline rescue form"
        assert store.get_command_receipt(pending_data["pending_command"]["command"]["command_id"])["state"] == "committed"
        restored.screenshot(path=str(output / "restored-rescue-explicit-save.png"), full_page=True)
        restored.close()
        check("Downloaded pending rescue restores in a fresh tab without writing until explicit original-command recovery")

        if summary["page_errors"]:
            raise AssertionError(f"Browser had unexpected page errors: {summary['page_errors']}")

        if summary["external_requests"]:
            raise AssertionError("Unexpected external browser requests")
    except BaseException:
        if page is not None:
            try:
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                redacted = page.content().replace(server.token, "REDACTED_EPHEMERAL_TOKEN")
                (output / "failure-dom.html").write_text(redacted, encoding="utf-8")
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
            cleanups.append(("server-thread", lambda: thread.join(timeout=5)))
        cleanups.append(("workspace-store", store.close))
        for name, action in cleanups:
            try:
                action()
            except Exception as error:
                summary["cleanup_errors"].append({"resource": name, "error": str(error)})
        if thread is not None and thread.is_alive():
            summary["cleanup_errors"].append({"resource": "server-thread", "error": "Still alive after shutdown"})
    if summary["cleanup_errors"]:
        raise RuntimeError("Acceptance resource cleanup failed; see summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channel", help="Optional browser channel; still headless")
    parser.add_argument("--installed-wheel", type=Path,
                        help="Require -I and site-packages bytes identical to this wheel")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("Output must be a new directory; existing evidence preserved: " + str(error))
    summary = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "argv": [sys.executable] + sys.argv, "python": sys.version,
               "headless": True, "native_file_picker_tested": False,
               "installed_identity": None,
               "scope": "Synthetic enhancement workflow and persistence checks. Not native user Chrome, "
                        "human listening, semantic annotation accuracy or release readiness."}
    write_json(output / "summary.json", summary)
    try:
        harness = load_harness(args.installed_wheel)
        files = [Path(__file__).resolve(), SOURCE / "scripts" / "check_workspace_editor.py"] + sorted(
            path for path in harness.PACKAGE.rglob("*")
            if path.is_file() and path.suffix in {".py", ".html", ".js"})
        identities = {str(path): digest(path) for path in files}
        summary.update(source_sha256=identities, installed_identity=harness.INSTALLED_IDENTITY)
        write_json(output / "summary.json", summary)
        sources = harness.make_sources(output)
        summary["fixtures"] = sources
        write_json(output / "summary.json", summary)
        run_checks(output, Path(sources["cfr"]["path"]), harness, args.channel, summary)
        if any(digest(path) != expected for path, expected in identities.items()):
            raise AssertionError("Implementation changed during acceptance; rerun against fixed bytes")
        if any(digest(item["path"]) != item["sha256"] for item in sources.values()):
            raise AssertionError("A source fixture changed during acceptance")
        identity = harness.INSTALLED_IDENTITY
        if identity and digest(identity["wheel"]) != identity["wheel_sha256"]:
            raise AssertionError("Installed acceptance wheel changed during acceptance")
        summary["status"] = "PASS"
    except BaseException as error:
        description = str(error) or type(error).__name__
        summary.update(status="INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "FAIL", error=description)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("Error: " + description + "; evidence: " + str(output), file=sys.stderr)
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
