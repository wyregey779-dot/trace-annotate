"""Drive the workspace editor in a real, headed Chrome window (P11).

This is deliberately not a headless layout test. It starts the candidate's own
server, opens a real Chrome window, and observes:

* the layout at 1366x768 and 1070x720, including whether the primary playback
  control is fully inside the viewport and not covered by anything;
* a real 200 % device scale factor (``--force-device-scale-factor=2``), which
  is operating-system scaling rather than a CSS transform, and the resulting
  device pixel ratio the page actually reports;
* Chinese input through Chrome's own IME pipeline (``Input.imeSetComposition``
  followed by ``Input.insertText``), not synthetic composition events;
* Space/I/O typed into a text field, which must not play, mark or save;
* the native file chooser being opened and then cancelled, which must leave no
  intake behind;
* unmuted playback of a source that really carries an AAC track, with decoded
  audio bytes reported by the media element (this is decoded-audio evidence,
  not a listening test);
* seek, 2x rate and segment looping on the media element;
* editing, saving, reloading and reading the change back from the database;
* cancelling queued batch items and retrying a failed intake.

Everything it writes stays inside ``--output``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import traceback
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--installed-wheel", type=Path)
_bootstrap_args, _ = _bootstrap.parse_known_args()
INSTALLED_WHEEL = _bootstrap_args.installed_wheel
INSTALLED_IDENTITY = None
if INSTALLED_WHEEL is None:
    sys.path.insert(0, str(SOURCE))
else:
    if not sys.flags.isolated:
        raise RuntimeError("Installed acceptance requires the target venv Python with -I")
    import trace_annotate
    package = Path(trace_annotate.__file__).resolve().parent
    roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    source_package = (SOURCE / "trace_annotate").resolve()
    if source_package in [package, *package.parents] or not any(root in package.parents for root in roots):
        raise RuntimeError("Installed acceptance must load trace_annotate from target site-packages")
    INSTALLED_WHEEL = INSTALLED_WHEEL.expanduser().resolve(strict=True)
    expected = {}
    with zipfile.ZipFile(INSTALLED_WHEEL) as archive:
        for name in archive.namelist():
            item = Path(name)
            if name.startswith("trace_annotate/") and item.suffix in {".py", ".html", ".js"}:
                if ".." in item.parts or name in expected:
                    raise RuntimeError("Invalid or duplicated package member in wheel")
                local = package.joinpath(*item.parts[1:])
                contents = archive.read(name)
                if not local.is_file() or local.read_bytes() != contents:
                    raise RuntimeError("Installed resource differs from wheel: " + name)
                expected[name] = hashlib.sha256(contents).hexdigest()
    actual = {"trace_annotate/" + path.relative_to(package).as_posix()
              for path in package.rglob("*") if path.is_file() and path.suffix in {".py", ".html", ".js"}}
    if not expected or set(expected) != actual:
        raise RuntimeError("Installed Python/HTML/JS resource inventory differs from wheel")
    INSTALLED_IDENTITY = {"wheel": str(INSTALLED_WHEEL),
                          "wheel_sha256": hashlib.sha256(INSTALLED_WHEEL.read_bytes()).hexdigest(),
                          "package": str(package), "resources": expected}

from trace_annotate.workspace.server import make_server  # noqa: E402
from trace_annotate.workspace.store import WorkspaceStore  # noqa: E402
import trace_annotate  # noqa: E402

PACKAGE = Path(trace_annotate.__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, body) -> None:
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def ffmpeg() -> str:
    tool = shutil.which("ffmpeg")
    if not tool or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and ffprobe are required for the native playback checks")
    return tool


def make_sources(output: Path) -> dict:
    folder = output / "media"
    folder.mkdir()
    tool = ffmpeg()
    commands = []

    def run(argv):
        result = subprocess.run(argv, capture_output=True, timeout=300)
        commands.append({"argv": argv, "exit_code": result.returncode,
                         "stdout_sha256": hashlib.sha256(result.stdout).hexdigest()})
        write_json(folder / "commands.json", commands)
        if result.returncode:
            raise RuntimeError("fixture command failed: " + " ".join(argv))
        return result

    voiced = folder / "voiced.mp4"
    run([tool, "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=duration=8:size=640x360:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-q:v", "4",
         "-c:a", "aac", "-b:a", "96k", "-pix_fmt", "yuv420p", "-shortest", str(voiced)])
    batch = folder / "batch.mp4"
    run([tool, "-v", "error", "-y", "-f", "lavfi", "-i",
         "smptebars=duration=60:size=854x480:rate=10", "-c:v", "mpeg4", "-q:v", "6",
         "-pix_fmt", "yuv420p", str(batch)])
    plain = folder / "plain.mp4"
    run([tool, "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=4:size=320x180:rate=10", "-c:v", "mpeg4", "-q:v", "6",
         "-pix_fmt", "yuv420p", str(plain)])
    invalid = folder / "invalid.mp4"
    invalid.write_bytes(b"TraceAnnotate deliberately undecodable native fixture\n")
    probe = json.loads(run([shutil.which("ffprobe"), "-v", "error", "-show_streams",
                            "-of", "json", str(voiced)]).stdout)
    audio = [s for s in probe["streams"] if s.get("codec_type") == "audio"]
    if not audio or audio[0].get("codec_name") != "aac":
        raise RuntimeError("the playback fixture does not carry an AAC track")
    return {"voiced": {"path": str(voiced), "sha256": digest(voiced)},
            "batch": {"path": str(batch), "sha256": digest(batch)},
            "plain": {"path": str(plain), "sha256": digest(plain)},
            "invalid": {"path": str(invalid), "sha256": digest(invalid)},
            "audio_stream": {"codec": audio[0]["codec_name"],
                             "sample_rate": audio[0].get("sample_rate")}}


def run(output: Path, channel, summary: dict) -> None:
    from playwright.sync_api import expect, sync_playwright

    store = WorkspaceStore(output / "workspace")
    store.create()
    server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    sources = make_sources(output)
    summary["sources"] = sources
    summary["installed_identity"] = INSTALLED_IDENTITY
    summary["module"] = {"file": str(PACKAGE), "import_origin": str(Path(trace_annotate.__file__))}

    def wait_task_count(count: int, budget: float = 180.0) -> None:
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            if len(store.list_tasks()) >= count:
                return
            time.sleep(0.05)
        raise AssertionError(f"expected {count} tasks, saw {len(store.list_tasks())}")

    def wait_intake(predicate, budget: float = 180.0):
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            for row in store.list_intakes():
                if predicate(row):
                    return row
            time.sleep(0.05)
        raise AssertionError("intake condition not reached")

    def seek_to(seconds: float) -> None:
        page.locator("#seek").evaluate(
            "(element,t) => {element.value=String(t);"
            "element.dispatchEvent(new Event('input',{bubbles:true}));}", seconds)
        page.wait_for_function("() => {const v=document.querySelector('#player');"
                               "return !v.seeking && Math.abs(v.currentTime-%f) < 0.35;}" % seconds,
                               timeout=20000)

    def record(name: str, body: dict) -> None:
        summary["observations"].append({"scenario": name, **body})
        write_json(output / "result.json", summary)

    playwright = sync_playwright().start()
    try:
        launch = {"headless": False, "args": ["--autoplay-policy=no-user-gesture-required",
                                              "--window-size=1400,900"]}
        if channel:
            launch["channel"] = channel
        browser = playwright.chromium.launch(**launch)
        summary["browser"] = {"version": browser.version, "channel": channel,
                              "headless": False}
        context = browser.new_context(accept_downloads=True, viewport={"width": 1366, "height": 768})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(server.editor_url)
        expect(page.locator("#file-input")).to_be_enabled()

        # -- native file picker: open the dialog, then cancel it -------------
        chooser_events = []
        page.on("filechooser", lambda fc: chooser_events.append(fc))
        before_intakes = len(store.list_intakes())
        page.locator("#import-label").click()
        deadline = time.monotonic() + 15
        while not chooser_events and time.monotonic() < deadline:
            page.wait_for_timeout(50)
        if not chooser_events:
            raise AssertionError("the native file chooser never opened")
        chooser_events[0].set_files([])
        page.wait_for_timeout(1500)
        after_cancel = len(store.list_intakes())
        status_text = page.locator("#import-status").inner_text()
        assert after_cancel == before_intakes, (after_cancel, before_intakes)
        record("native_file_picker_cancel", {
            "dialog_opened": True, "cancelled": True,
            "intakes_before": before_intakes, "intakes_after": after_cancel,
            "import_status": status_text,
            "tasks": len(store.list_tasks())})

        # -- import the voiced source --------------------------------------
        page.locator("#file-input").set_input_files([sources["voiced"]["path"]])
        wait_task_count(1)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(1, timeout=60000)
        task_id = store.list_tasks()[0]["task_id"]
        page.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(page.locator("#play-toggle")).to_be_enabled(timeout=30000)
        media = store.get_media(store.get_task(task_id)["media_id"])

        # -- unmuted playback with decoded audio ---------------------------
        page.locator("#play-toggle").click()
        page.wait_for_function("() => {const v=document.querySelector('#player');"
                               "return !v.paused && v.currentTime > 0.4;}", timeout=30000)
        playback = page.evaluate("""() => {
            const v = document.querySelector('#player');
            return {paused: v.paused, muted: v.muted, volume: v.volume,
                    currentTime: v.currentTime, playbackRate: v.playbackRate,
                    audio_bytes: v.webkitAudioDecodedByteCount ?? null,
                    video_bytes: v.webkitVideoDecodedByteCount ?? null,
                    has_audio_track: typeof v.audioTracks === 'object' && v.audioTracks !== null
                        ? v.audioTracks.length : null};
        }""")
        assert playback["paused"] is False and playback["muted"] is False, playback
        assert playback["volume"] > 0, playback
        assert playback["audio_bytes"] is None or playback["audio_bytes"] > 0, playback
        record("audible_playback", {"media_id": media["id"], "audio_stream": sources["audio_stream"],
                                    **playback, "note": "decoded-audio evidence, not a listening test"})
        page.locator("#play-toggle").click()
        page.wait_for_function("() => document.querySelector('#player').paused", timeout=10000)

        # -- seek ----------------------------------------------------------
        seek_to(3.5)
        seeked = page.evaluate("() => document.querySelector('#player').currentTime")
        record("seek", {"requested": 3.5, "currentTime": seeked,
                        "seeking": page.evaluate("() => document.querySelector('#player').seeking")})

        # -- 2x rate -------------------------------------------------------
        page.locator("#playback-rate").select_option("2")
        page.locator("#play-toggle").click()
        start = page.evaluate("() => document.querySelector('#player').currentTime")
        page.wait_for_timeout(1500)
        end = page.evaluate("() => document.querySelector('#player').currentTime")
        page.locator("#play-toggle").click()
        rate = page.evaluate("() => document.querySelector('#player').playbackRate")
        assert rate == 2, rate
        assert end - start > 2.0, (start, end)
        record("speed", {"playbackRate": rate, "media_seconds_advanced": round(end - start, 3),
                         "wall_seconds": 1.5})

        # -- segment loop --------------------------------------------------
        seek_to(0.2)
        page.locator("#mark-start").click()
        seek_to(1.2)
        page.locator("#mark-end").click()
        page.locator("#segment-label").fill("循环片段")
        page.locator("#segment-text").fill("真实浏览器片段循环检查")
        page.locator("#save-segment").click()
        expect(page.locator("#segments tbody tr")).to_have_count(1, timeout=30000)
        page.locator("#segments tbody button[data-annotation-id]").first.click()
        expect(page.locator("#play-selection")).to_be_enabled(timeout=10000)
        page.locator("#loop-selection").check()
        page.locator("#play-selection").click()
        page.wait_for_timeout(600)
        samples = []
        for _ in range(14):
            samples.append(page.evaluate("() => document.querySelector('#player').currentTime"))
            page.wait_for_timeout(200)
        page.locator("#play-toggle").click()
        inside = [value for value in samples if 0.05 <= value <= 1.45]
        assert len(inside) >= len(samples) - 2, samples
        assert max(samples) < 1.6, samples
        record("segment_loop", {"samples": [round(value, 3) for value in samples],
                                "start": 0.2, "end": 1.2, "loop_enabled": True})

        # -- Chinese IME through the browser's input pipeline ---------------
        page.locator("#new-segment").click()
        page.locator("#segment-label").click()
        cdp = context.new_cdp_session(page)
        page.evaluate("""() => {window.__ime = {start: 0, update: 0, end: 0};
            const el = document.querySelector('#segment-label');
            el.addEventListener('compositionstart', () => window.__ime.start++);
            el.addEventListener('compositionupdate', () => window.__ime.update++);
            el.addEventListener('compositionend', () => window.__ime.end++);}""")
        cdp.send("Input.imeSetComposition", {"text": "zhong", "selectionStart": 5, "selectionEnd": 5})
        page.wait_for_timeout(150)
        cdp.send("Input.imeSetComposition", {"text": "中文标签", "selectionStart": 4, "selectionEnd": 4})
        page.wait_for_timeout(150)
        cdp.send("Input.insertText", {"text": "中文标签"})
        page.wait_for_timeout(200)
        ime = page.evaluate("""() => ({events: window.__ime,
            value: document.querySelector('#segment-label').value,
            paused: document.querySelector('#player').paused})""")
        assert ime["events"]["start"] >= 1, ime
        assert ime["events"]["end"] >= 1, ime
        assert ime["value"] == "中文标签", ime
        assert ime["paused"] is True, ime
        record("chinese_ime", ime)

        # -- Space / I / O inside a text field -----------------------------
        before = {name: page.locator("#segment-" + name).input_value()
                  for name in ("start", "end")}
        revision_before = store.get_task(task_id)["revision"]
        page.locator("#segment-text").click()
        for key in (" ", "i", "o"):
            page.keyboard.press(key)
        page.wait_for_timeout(300)
        typed = page.evaluate("""() => ({paused: document.querySelector('#player').paused,
            start: document.querySelector('#segment-start').value,
            end: document.querySelector('#segment-end').value,
            text: document.querySelector('#segment-text').value})""")
        assert typed["paused"] is True, typed
        assert typed["start"] == before["start"] and typed["end"] == before["end"], typed
        assert store.get_task(task_id)["revision"] == revision_before
        page.locator("#segment-text").fill("")
        page.locator("#segment-text").evaluate("el => el.blur()")
        page.keyboard.press(" ")
        page.wait_for_function("() => !document.querySelector('#player').paused", timeout=10000)
        page.keyboard.press(" ")
        page.wait_for_function("() => document.querySelector('#player').paused", timeout=10000)
        record("space_i_o_in_input", {**typed, "outside_input_space_plays": True,
                                      "revision": revision_before})
        # The IME text is an unsaved draft; discard it explicitly so the next
        # task switch is not blocked by the unsaved-changes dialog.
        page.locator("#discard-draft").click()
        page.wait_for_timeout(200)
        if page.locator("#confirm-dialog").is_visible():
            page.locator("#confirm-accept").click()
            page.wait_for_timeout(200)
        assert not page.locator("#confirm-dialog").is_visible()

        # -- layout at 1366x768 and 1070x720 -------------------------------
        for width, height in ((1366, 768), (1070, 720)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("() => window.scrollTo(0, 0)")
            page.locator("#play-toggle").scroll_into_view_if_needed()
            metrics = page.evaluate("""() => {
                const el = document.querySelector('#play-toggle');
                const r = el.getBoundingClientRect();
                const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
                return {x: r.x, y: r.y, width: r.width, height: r.height,
                        innerWidth: innerWidth, innerHeight: innerHeight,
                        scrollWidth: document.documentElement.scrollWidth,
                        devicePixelRatio: devicePixelRatio,
                        unobscured: hit === el || el.contains(hit),
                        hit: hit ? (hit.tagName + '#' + (hit.id || '') + '.' + (hit.className || '')) : null};
            }""")
            page.screenshot(path=str(output / f"layout-{width}x{height}.png"))
            assert metrics["x"] >= 0 and metrics["y"] >= 0, metrics
            assert metrics["x"] + metrics["width"] <= metrics["innerWidth"] + 0.5, metrics
            assert metrics["y"] + metrics["height"] <= metrics["innerHeight"] + 0.5, metrics
            assert metrics["unobscured"], metrics
            assert metrics["scrollWidth"] <= metrics["innerWidth"] + 1, metrics
            record(f"layout_{width}x{height}", metrics)
        page.set_viewport_size({"width": 1366, "height": 768})

        # -- edit, save, reload, read back --------------------------------
        page.locator("#segments tbody button[data-annotation-id]").first.click()
        page.locator("#segment-text").fill("真实浏览器编辑后重新打开")
        page.locator("#save-segment").click()
        expect(page.locator("#save-state")).to_contain_text("已保存", timeout=30000)
        page.reload()
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(1, timeout=60000)
        page.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(page.locator("#segments tbody tr")).to_have_count(1, timeout=30000)
        page.locator("#segments tbody button[data-annotation-id]").first.click()
        expect(page.locator("#segment-text")).to_have_value("真实浏览器编辑后重新打开", timeout=20000)
        reopened = store.get_task(task_id)
        record("edit_save_reopen", {"task_id": task_id, "revision": reopened["revision"],
                                    "text": "真实浏览器编辑后重新打开",
                                    "annotations": [item["id"] for item in reopened["annotations"]]})

        # -- batch cancel ------------------------------------------------
        page.locator("#file-input").set_input_files([sources["batch"]["path"], sources["plain"]["path"]])
        first = wait_intake(lambda row: row["state"] in ("queued", "preparing")
                            and row["display_name"] == "batch.mp4", budget=120)
        second = wait_intake(lambda row: row["state"] in ("queued", "preparing")
                             and row["display_name"] == "plain.mp4", budget=120)
        expect(page.locator("#cancel-batch-queued")).to_be_enabled(timeout=60000)
        page.locator("#cancel-batch-queued").click()
        settled = {}
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and len(settled) < 2:
            for ident in (first["id"], second["id"]):
                row = [item for item in store.list_intakes() if item["id"] == ident][0]
                if row["state"] in ("cancelled", "succeeded", "failed"):
                    settled[ident] = row
            time.sleep(0.05)
        assert len(settled) == 2, settled
        assert all(row["state"] == "cancelled" for row in settled.values()), settled
        assert all(row["task_id"] is None for row in settled.values()), settled
        record("batch_cancel", {"cancelled": [
            {"id": row["id"], "name": row["display_name"], "state": row["state"],
             "error_code": row["error_code"], "task_id": row["task_id"]}
            for row in settled.values()], "tasks": len(store.list_tasks())})

        # -- retry a failed intake ----------------------------------------
        expect(page.locator("#file-input")).to_be_enabled(timeout=120000)
        page.locator("#file-input").set_input_files([sources["invalid"]["path"]])
        failed = wait_intake(lambda row: row["state"] == "failed", budget=180)
        attempt_before = failed["attempt"]
        # Polling stops once nothing is queued, so ask for a fresh list before
        # clicking the retry control a user would see.
        page.locator("#refresh-workspace").click()
        page.locator(".intake-details summary").click()
        retry = page.locator(f'#intake-list li[data-intake-id="{failed["id"]}"] button[data-action="retry"]')
        expect(retry).to_be_visible(timeout=30000)
        retry.click()
        deadline = time.monotonic() + 180
        retried = None
        while time.monotonic() < deadline:
            row = [item for item in store.list_intakes() if item["id"] == failed["id"]][0]
            if row["state"] == "failed" and row["attempt"] > attempt_before:
                retried = row
                break
            time.sleep(0.05)
        assert retried is not None, "the retry did not produce a second attempt"
        assert retried["task_id"] is None
        record("batch_cancel_retry", {
            "retried": {"id": retried["id"], "state": retried["state"],
                        "attempt": retried["attempt"], "error_code": retried["error_code"],
                        "task_id": retried["task_id"]},
            "tasks": len(store.list_tasks())})

        # -- real 200 % device scaling ------------------------------------
        scaled = playwright.chromium.launch(
            headless=False,
            args=["--force-device-scale-factor=2", "--window-size=1400,900"],
            **({"channel": channel} if channel else {}))
        try:
            context2 = scaled.new_context(no_viewport=True)
            page2 = context2.new_page()
            page2.goto(server.editor_url)
            expect(page2.locator("#file-input")).to_be_enabled()
            page2.locator(f'#task-list button[data-task-id="{task_id}"]').click()
            expect(page2.locator("#play-toggle")).to_be_enabled(timeout=30000)
            zoom = page2.evaluate("""() => {
                const el = document.querySelector('#play-toggle');
                const r = el.getBoundingClientRect();
                const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
                return {devicePixelRatio: devicePixelRatio, innerWidth: innerWidth,
                        innerHeight: innerHeight,
                        screenWidth: screen.width, screenHeight: screen.height,
                        scrollWidth: document.documentElement.scrollWidth,
                        unobscured: hit === el || el.contains(hit),
                        x: r.x, y: r.y, width: r.width, height: r.height};
            }""")
            assert zoom["devicePixelRatio"] == 2, zoom
            assert zoom["scrollWidth"] <= zoom["innerWidth"] + 1, zoom
            assert zoom["unobscured"], zoom
            page2.screenshot(path=str(output / "layout-200pct-device-scale.png"))
            record("real_200pct_zoom", {**zoom, "method": "--force-device-scale-factor=2"})
            context2.close()
        finally:
            scaled.close()

        summary["page_errors"] = errors
        assert not errors, errors
        context.close()
        browser.close()
    finally:
        playwright.stop()
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channel", default="chrome",
                        help="Installed browser channel; 'chrome' is required for native acceptance")
    parser.add_argument("--installed-wheel", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to reuse an existing output directory")
    output.mkdir(parents=True)
    summary = {"document_type": "NATIVE_BROWSER_RESULT", "result_version": 1,
               "output": str(output), "overall": "RUNNING",
               "expected_cases": ["NATIVE_LAYOUT_INPUT", "NATIVE_PLAY_EDIT_BATCH"],
               "observations": [], "started_at": time.time()}
    write_json(output / "result.json", summary)
    try:
        run(output, args.channel, summary)
    except Exception as exc:  # noqa: BLE001
        summary["overall"] = "FAIL"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        summary["finished_at"] = time.time()
        write_json(output / "result.json", summary)
        raise SystemExit(1)
    summary["overall"] = "PASS"
    summary["finished_at"] = time.time()
    write_json(output / "result.json", summary)
    print(json.dumps({"overall": "PASS", "observations": len(summary["observations"]),
                      "browser": summary["browser"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
