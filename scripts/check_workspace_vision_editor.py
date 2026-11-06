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
