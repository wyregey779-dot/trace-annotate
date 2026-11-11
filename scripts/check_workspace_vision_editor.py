"""Real-browser check for the workspace vision flow.

Creates two local synthetic clips, runs the actual editor server, drives the
vision job from the UI, applies candidates, edits and saves one, reloads the
page, exports professional JSON and checks the exported annotation text.
It checks task isolation after the vision job completes, and switches tasks
while a boundary preview is loading to check that stale frames are discarded.

Only local data is touched. An existing output directory is refused.
"""
import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import site
import subprocess
import sys
import sysconfig
import threading
import time
import traceback
from types import SimpleNamespace
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
EXPECTED_CHECKS = 20


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, body):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def phase(args, summary, name):
    summary["phase"] = name
    summary["phase_started_utc"] = utc_now()
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "progress.json", {"phase": name, "updated_utc": utc_now(),
                                               "checks_completed": len(summary["checks"])})
    print("PHASE " + name, flush=True)


def verify_package_resources(package, wheel):
    """Bind all shipped package resources, including models, fonts and licences."""
    expected = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith("trace_annotate/") or name.endswith("/"):
                continue
            item = Path(name)
            if ".." in item.parts or "\\" in name or name in expected:
                raise RuntimeError("Invalid or duplicated package member in wheel: " + name)
            local = package.joinpath(*item.parts[1:])
            contents = archive.read(name)
            if package not in local.resolve().parents or not local.is_file() or local.read_bytes() != contents:
                raise RuntimeError("Installed resource differs from wheel: " + name)
            expected[name] = hashlib.sha256(contents).hexdigest()
    actual = {"trace_annotate/" + path.relative_to(package).as_posix()
              for path in package.rglob("*") if path.is_file()
              and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}}
    if not expected or set(expected) != actual:
        raise RuntimeError("Installed package resource inventory differs from wheel")
    return expected


def load_runtime(installed_wheel):
    if installed_wheel is None:
        sys.path.insert(0, str(SOURCE))
    elif not sys.flags.isolated:
        raise RuntimeError("Installed acceptance requires the target venv Python with -I")
    import trace_annotate
    package = Path(trace_annotate.__file__).resolve().parent
    identity = {"mode": "source", "package": str(package)}
    if installed_wheel is not None:
        roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
        if (sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE
                or package == SOURCE / "trace_annotate" or SOURCE in package.parents
                or not any(root in package.parents for root in roots)):
            raise RuntimeError("Installed acceptance must load a separate venv site-packages, without user-site or checkout imports")
        configuration = Path(sys.prefix) / "pyvenv.cfg"
        if not configuration.is_file() or "include-system-site-packages = false" not in configuration.read_text(encoding='utf-8').lower():
            raise RuntimeError("Installed acceptance requires a venv without system-site-packages")
        installed_wheel = installed_wheel.expanduser().resolve(strict=True)
        identity = {"mode": "installed-wheel", "wheel": str(installed_wheel),
                    "wheel_sha256": digest(installed_wheel), "package": str(package),
                    "resources": verify_package_resources(package, installed_wheel),
                    "isolated": bool(sys.flags.isolated), "user_site": site.ENABLE_USER_SITE,
                    "prefix": sys.prefix, "base_prefix": sys.base_prefix}
    from trace_annotate.workspace.server import make_server
    from trace_annotate.workspace.store import WorkspaceStore
    for name, module in list(sys.modules.items()):
        if name == "trace_annotate" or name.startswith("trace_annotate."):
            origin = getattr(module, "__file__", None)
            if origin and package not in Path(origin).resolve().parents:
                raise RuntimeError("Mixed package imports are forbidden: " + name)
    return SimpleNamespace(make_server=make_server, WorkspaceStore=WorkspaceStore,
                           package=package, identity=identity)


def make_clip(ffmpeg, target, first, second):
    command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n",
               "-f", "lavfi", "-i", f"color=c={first}:s=320x180:r=25:d=3",
               "-f", "lavfi", "-i", f"color=c={second}:s=320x180:r=25:d=3",
               "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", str(target)]
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    except subprocess.TimeoutExpired as error:
        target.with_suffix(".ffmpeg.stderr.log").write_bytes(error.stderr or b"")
        write_json(target.with_suffix(".ffmpeg.command.json"), {"argv": command, "timed_out": True})
        raise
    target.with_suffix(".ffmpeg.stderr.log").write_bytes(result.stderr)
    write_json(target.with_suffix(".ffmpeg.command.json"), {"argv": command, "exit_code": result.returncode})
    result.check_returncode()


def run_checks(args, summary, runtime):
    a = args.output / "vision-a.mp4"
    b = args.output / "vision-b.mp4"
    phase(args, summary, "fixtures.encode")
    make_clip(args.ffmpeg, a, "blue", "red")
    make_clip(args.ffmpeg, b, "green", "yellow")
    summary["clips"] = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size} for path in (a, b)}

    def check(name):
        summary["checks"].append(name)
        print(name + ": PASS", flush=True)
        phase(args, summary, "checks." + str(len(summary["checks"]) + 1))

    phase(args, summary, "workspace.open")
    store = runtime.WorkspaceStore(args.output / "workspace")
    server = playwright = browser = context = page = None
    try:
        phase(args, summary, "workspace.create")
        store.create()
        phase(args, summary, "workspace.import.a")
        store.import_source(a)
        phase(args, summary, "workspace.import.b")
        store.import_source(b)
        phase(args, summary, "server.start")
        server = runtime.make_server(store)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        summary["server_url"] = server.editor_url
        phase(args, summary, "playwright.import")
        from playwright.sync_api import sync_playwright, expect
        summary["playwright_version"] = importlib.metadata.version("playwright")
        phase(args, summary, "playwright.start")
        playwright = sync_playwright().start()
        phase(args, summary, "browser.launch")
        browser = playwright.chromium.launch(headless=args.headless,
                                            timeout=30000,
                                            **({"channel": args.channel} if args.channel else {}))
        phase(args, summary, "browser.context")
        context = browser.new_context(accept_downloads=True, viewport={"width": 1366, "height": 768},
                                      device_scale_factor=args.device_scale_factor)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.set_default_timeout(15000)
        page.set_default_navigation_timeout(30000)
        page.on("pageerror", lambda error: summary["page_errors"].append(str(error)))
        page.on("console", lambda message: summary["console_errors"].append(message.text)
                if message.type == "error" else None)
        page.on("requestfailed", lambda request: summary["failed_requests"].append(
            {"url": request.url, "failure": request.failure}))
        summary["browser"] = {"headless": args.headless, "channel": args.channel,
                              "version": browser.version, "viewport": "1366x768",
                              "device_scale_factor": args.device_scale_factor,
                              "chrome_menu_zoom_tested": False}

        def task_id(name):
            return next(item["task_id"] for item in store.list_intakes()
                        if item["display_name"] == name)

        first, second = task_id(a.name), task_id(b.name)

        def select(target):
            target.locator(f'#task-list button[data-task-id="{target_id}"]').click()
            expect(target.locator(f'#task-list button[data-task-id="{target_id}"]')).to_have_attribute(
                "aria-current", "true")
            expect(target.locator("#new-segment")).to_be_enabled(timeout=30000)

        phase(args, summary, "page.goto")
        page.goto(server.editor_url)
        phase(args, summary, "page.wait_tasks")
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(2, timeout=60000)
        target_id = first
        select(page)
        expect(page.locator("#vision-run")).to_be_enabled(timeout=30000)
        phase(args, summary, "checks.1.vision_job")
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
        binding = accepted[0].get("review_binding") or {}
        if (binding.get("actor_id") != "local-user"
                or not str(binding.get("command_id", "")).startswith("cmd-")
                or not isinstance(binding.get("task_revision"), int)
                or not binding.get("content_sha256")
                or not binding.get("reviewed_at")):
            raise AssertionError("Accepted review is not bound to the acting user, content and command")
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
        if args.visible_evidence:
            page.screenshot(path=str(args.visible_evidence))
            summary["visible_evidence"] = {"path": args.visible_evidence.name,
                                           "sha256": digest(args.visible_evidence),
                                           "bytes": args.visible_evidence.stat().st_size}
        if args.ready_marker:
            args.ready_marker.write_text(json.dumps({"viewport": "1366x768",
                "device_scale_factor": args.device_scale_factor,
                "headless": args.headless}) + "\n", encoding="utf-8")
        if args.hold_seconds:
            phase(args, summary, "browser.hold")
            time.sleep(args.hold_seconds)
    except BaseException as error:
        record_failure(args.output, summary, error)
        if page is not None:
            try:
                page.screenshot(path=str(args.output / "failure.png"), timeout=5000)
                (args.output / "failure.html").write_text(page.content(), encoding="utf-8")
            except Exception as diagnostic_error:
                summary["diagnostic_error"] = str(diagnostic_error)
        raise
    finally:
        cleanup_errors = []
        actions = []
        if context is not None:
            actions.extend([("browser.trace", lambda: context.tracing.stop(path=str(args.output / "trace.zip"))),
                            ("browser.context.close", context.close)])
        if browser is not None:
            actions.append(("browser.close", browser.close))
        if playwright is not None:
            actions.append(("playwright.stop", playwright.stop))
        if server is not None:
            actions.extend([("server.shutdown", server.shutdown), ("server.close", server.server_close),
                            ("vision_jobs.close", lambda: server.vision_jobs.close(timeout=10))])
        actions.append(("workspace.close", store.close))
        for name, action in actions:
            phase(args, summary, name)
            try:
                action()
            except Exception as error:
                cleanup_errors.append({"phase": name, "error": str(error)})
        if cleanup_errors:
            summary["cleanup_errors"] = cleanup_errors
            if summary["status"] == "RUNNING":
                raise RuntimeError("Cleanup failed; see cleanup_errors")


def record_failure(output, summary, error):
    summary.update(ok=False, status="INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "FAIL",
                   error=str(error) or type(error).__name__, failed_phase=summary.get("phase"))
    (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "summary.json", summary)


def worker(args):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    with (args.output / "thread-dumps.log").open("w", encoding="utf-8") as dumps:
        faulthandler.enable(file=dumps, all_threads=True)
        faulthandler.dump_traceback_later(min(30, args.stall_timeout_seconds), repeat=True, file=dumps)
        try:
            # All writable process caches and Chrome's temporary profile stay in this run.
            for name in ("temp", "cache"):
                (args.output / name).mkdir()
            for key in ("TEMP", "TMP", "TMPDIR"):
                os.environ[key] = str(args.output / "temp")
            # Changing writable XDG caches must not hide preinstalled browsers.
            if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and sys.platform == "linux":
                original_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(original_cache / "ms-playwright")
            os.environ["XDG_CACHE_HOME"] = str(args.output / "cache")
            sys.dont_write_bytecode = True
            phase(args, summary, "runtime.verify")
            runtime = load_runtime(args.installed_wheel)
            summary["installed_identity"] = runtime.identity
            paths = [Path(__file__).resolve()] + [p for p in runtime.package.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}]
            identities = {str(path): digest(path) for path in paths}
            summary["source_sha256"] = identities
            summary["optional_dependencies"] = {}
            for name in ("opencv-python", "pillow-heif", "rapidocr-onnxruntime", "torch", "open-clip-torch", "transformers"):
                try:
                    summary["optional_dependencies"][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    summary["optional_dependencies"][name] = None
            if not args.ffmpeg or not shutil.which("ffprobe"):
                raise RuntimeError("FFmpeg and ffprobe are required; browser checks cannot be skipped")
            run_checks(args, summary, runtime)
            if len(summary["checks"]) != EXPECTED_CHECKS:
                raise AssertionError(f"Expected {EXPECTED_CHECKS} checks, observed {len(summary['checks'])}")
            if any(digest(path) != expected for path, expected in identities.items()):
                raise AssertionError("Implementation changed during acceptance; rerun against frozen bytes")
            if args.installed_wheel is not None:
                if digest(runtime.identity["wheel"]) != runtime.identity["wheel_sha256"]:
                    raise AssertionError("Installed wheel changed during acceptance")
                verify_package_resources(runtime.package, Path(runtime.identity["wheel"]))
            summary.update(ok=True, status="PASS")
        except BaseException as error:
            if "error" not in summary:
                record_failure(args.output, summary, error)
            print("VISION UI ACCEPTANCE " + summary["status"] + ": " + summary["error"], file=sys.stderr, flush=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
            summary["finished_utc"] = utc_now()
            write_json(args.output / "summary.json", summary)
    if summary["ok"]:
        print("VISION UI ACCEPTANCE PASS", flush=True)
    return 0 if summary["ok"] else 1


def stop_process_tree(process):
    """Terminate only this harness's owned worker and its browser/encoder children."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe")
        subprocess.run([taskkill, "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=20, check=False)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def supervise(command, args, summary):
    """A separate process bounds synchronous Playwright startup and cleanup too."""
    started = last_progress = time.monotonic()
    observed = None
    timeout_error = None
    with (args.output / "worker.stdout.log").open("wb") as stdout, (args.output / "worker.stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       start_new_session=os.name != "nt")
        except OSError as error:
            record_failure(args.output, summary, error)
            summary["finished_utc"] = utc_now()
            write_json(args.output / "summary.json", summary)
            return 1
        write_json(args.output / "worker.json", {"pid": process.pid, "argv": command})
        try:
            while process.poll() is None:
                progress = args.output / "progress.json"
                if progress.is_file():
                    current = progress.read_bytes()
                    if current != observed:
                        observed = current
                        last_progress = time.monotonic()
                        progress_record = json.loads(current)
                        print(f"PHASE {progress_record['phase']} ({progress_record['checks_completed']}/{EXPECTED_CHECKS})", flush=True)
                elapsed = time.monotonic() - started
                stalled = time.monotonic() - last_progress
                if elapsed > args.timeout_seconds or stalled > args.stall_timeout_seconds:
                    timeout_error = f"Worker timeout: elapsed={elapsed:.1f}s, phase_idle={stalled:.1f}s"
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            timeout_error = "Worker interrupted by operator"
        finally:
            if process.poll() is None:
                stop_process_tree(process)
    result = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    result["worker_exit_code"] = process.returncode
    if timeout_error or process.returncode != 0 or result.get("status") != "PASS" or len(result["checks"]) != EXPECTED_CHECKS:
        result.update(ok=False, status="INCOMPLETE" if timeout_error else "FAIL")
        result.setdefault("failed_phase", result.get("phase", "worker.starting"))
        result["error"] = timeout_error or result.get("error", "Worker exited without completing the 20 checks")
        result["finished_utc"] = utc_now()
        (args.output / "supervisor-failure.log").write_text(result["error"] + "\n", encoding="utf-8")
    write_json(args.output / "summary.json", result)
    print("VISION UI ACCEPTANCE " + result["status"] + "; evidence: " + str(args.output), flush=True)
    return 0 if result["ok"] else 1


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-wheel", type=Path, help="Require separate venv Python -I and exact wheel resource bytes")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--channel", default=None, help="Installed browser channel, e.g. chrome")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device-scale-factor", type=float, default=1.0,
                        help="Device pixel ratio only; does not test Chrome menu zoom")
    parser.add_argument("--visible-evidence", type=Path, default=None)
    parser.add_argument("--ready-marker", type=Path, default=None)
    parser.add_argument("--hold-seconds", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=900, help="Hard timeout including startup and cleanup")
    parser.add_argument("--stall-timeout-seconds", type=float, default=180, help="Maximum time without phase/check progress")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 < args.device_scale_factor <= 4) or not (0 <= args.hold_seconds <= 180):
        parser.error("Device scale must be in (0, 4]; hold must be in [0, 180]")
    if not (0 < args.stall_timeout_seconds <= args.timeout_seconds):
        parser.error("Timeouts must satisfy 0 < stall-timeout <= timeout")
    if args.headless and (args.visible_evidence or args.ready_marker or args.hold_seconds):
        parser.error("Visible evidence, ready marker and hold require a non-headless run")
    args.output = args.output.expanduser().resolve()
    for name in ("visible_evidence", "ready_marker"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser().resolve()
            if args.output not in path.parents:
                parser.error("Evidence paths must stay inside --output")
            setattr(args, name, path)
    if args._worker:
        return worker(args)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("Output must be a new directory; existing evidence preserved: " + str(error))
    summary = {"status": "RUNNING", "ok": False, "started_utc": utc_now(), "checks": [],
               "expected_checks": EXPECTED_CHECKS, "page_errors": [], "console_errors": [], "failed_requests": [],
               "installed_identity": None, "python": sys.version, "argv": [sys.executable] + sys.argv,
               "driver_sha256": digest(__file__), "phase": "worker.starting",
               "scope": "20 automated UI checks on synthetic clips; CDP composition is not native Windows IME. Device scale is not Chrome menu zoom. Not semantic accuracy, human review or release readiness.",
               "coverage_limits": ["Task switching during a still-running vision job is not exercised",
                                   "The merge check exercises adjacent segments; a gap-rejection negative case is not exercised"],
               "browser_requested": {"headless": args.headless, "channel": args.channel,
                                     "device_scale_factor": args.device_scale_factor},
               "timeouts": {"total_seconds": args.timeout_seconds, "stall_seconds": args.stall_timeout_seconds}}
    write_json(args.output / "summary.json", summary)
    command = [sys.executable] + (["-I"] if sys.flags.isolated else []) + ["-B", str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    return supervise(command, args, summary)


if __name__ == "__main__":
    raise SystemExit(main())
