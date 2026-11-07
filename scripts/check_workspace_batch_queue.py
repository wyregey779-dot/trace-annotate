"""Real-browser check for the batch import queue and recovery display.

Creates three local clips with different frame rates and audio shapes, starts the
actual workspace server, uploads them through the UI file input, and requires:
- a percentage that moves through intermediate values to 100%;
- all three intakes admitted with no failures;
- the persisted batch display still shows 100% after a page reload.

Only local data is used. An existing output directory is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

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


def make_clip(ffmpeg, target, color, size, rate, seconds, audio=False):
    command = [ffmpeg, "-hide_banner", "-v", "error", "-y",
               "-f", "lavfi", "-i", f"color=c={color}:s={size}:r={rate}:d={seconds}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
        command += ["-c:a", "aac", "-b:a", "64k"]
    command += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", str(target)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    return {"path": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
            "width": int(size.split("x")[0]), "height": int(size.split("x")[1]),
            "fps": rate, "seconds": seconds, "audio": bool(audio)}


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

    clips = [
        make_clip(args.ffmpeg, args.output / "queue-a.mp4", "blue", "32x32", 5, 0.6),
        make_clip(args.ffmpeg, args.output / "queue-b.mp4", "red", "64x36", 10, 1.0, audio=True),
        make_clip(args.ffmpeg, args.output / "queue-c.mp4", "green", "48x32", 25, 0.8),
    ]
    summary = {"clips": clips, "checks": [], "progress_sequence": [], "page_errors": []}

    def check(name):
        summary["checks"].append(name)
        print(name + ": PASS", flush=True)
        write_json(args.output / "summary.json", summary)

    store = WorkspaceStore(args.output / "workspace")
    store.create()
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
        page.goto(server.editor_url)
        expect(page.locator("#task-count")).to_have_text("0", timeout=30000)
        check("Empty workspace exposes the batch import entry")

        page.evaluate("""() => {
          window.__batchProgress = [];
          const el = document.getElementById('import-status');
          const record = () => {
            const value = Number(el.dataset.progressPercent);
            if (Number.isFinite(value)) window.__batchProgress.push(value);
          };
          record();
          new MutationObserver(record).observe(el, {childList: true, subtree: true,
                                                    attributes: true, characterData: true});
        }""")
        page.locator("#file-input").set_input_files([str(args.output / clip["path"]) for clip in clips])
        expect(page.locator("#import-status")).to_contain_text("进度 100%", timeout=180000)
        expect(page.locator("#import-status")).to_contain_text("已入库 3", timeout=30000)
        expect(page.locator("#import-status")).to_contain_text("待处理 0", timeout=30000)
        expect(page.locator("#import-status")).to_contain_text("失败或未知 0", timeout=30000)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(3, timeout=60000)
        summary["progress_sequence"] = page.evaluate("window.__batchProgress")
        summary["batch_status"] = page.locator("#import-status").inner_text()
        check("Three mixed-frame-rate clips reach admitted with 100% progress")

        intermediate = [value for value in summary["progress_sequence"] if 0 < value < 100]
        if not intermediate:
            raise AssertionError("No intermediate percentage was observed")
        check("Progress reports intermediate percentages before completion")

        page.reload()
        expect(page.locator("#import-status")).to_contain_text("进度 100%", timeout=60000)
        expect(page.locator("#import-status")).to_contain_text("已入库 3", timeout=30000)
        summary["progress_after_reload"] = page.locator("#import-status").inner_text()
        check("Persisted batch display survives a page reload at 100%")

        summary["intakes"] = [{"id": row["id"], "display_name": row["display_name"],
                               "state": row["state"], "task_id": row.get("task_id")}
                              for row in store.list_intakes()]
        if any(row["state"] != "succeeded" for row in summary["intakes"]):
            raise AssertionError("An intake did not finish successfully")
        if summary["page_errors"]:
            raise AssertionError("Page errors were observed: " + str(summary["page_errors"]))
        check("Intake rows are all succeeded and the page has no errors")

        browser.close()
        playwright.stop()
    finally:
        server.shutdown()
        server.server_close()
        store.close()

    summary["ok"] = len(summary["checks"]) == 5
    write_json(args.output / "summary.json", summary)
    if not summary["ok"]:
        raise SystemExit("Batch queue acceptance did not complete")
    print("BATCH QUEUE ACCEPTANCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
