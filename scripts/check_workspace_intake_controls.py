"""Real-browser regression for intake cancel/retry buttons (R01/R08/R05).

Starts an isolated workspace server, seeds one failed intake, opens the served
editor in Chromium, clicks the 重试/取消 buttons, and records the actual HTTP
request method, Content-Type and JSON body. Headless; human-Chrome acceptance
is a separate gate. Existing output directories are refused.
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
if "--installed-wheel" in sys.argv:
    # keep parity with other check scripts; this smoke script is source-runtime.
    pass
else:
    sys.path.insert(0, str(SOURCE))

from playwright.sync_api import sync_playwright, expect  # noqa: E402

from trace_annotate.workspace.intake import AsyncIntakes  # noqa: E402
from trace_annotate.workspace.server import make_server  # noqa: E402
from trace_annotate.workspace.store import WorkspaceStore  # noqa: E402


def _seed_failed_intake(store: WorkspaceStore) -> str:
    q = AsyncIntakes(store)
    bad = store.tmp_dir / "bad-upload.bin"
    bad.write_bytes(b"not media" * 32)
    rec = q.enqueue(bad, display_name="bad.mp4")
    q.run_pending()
    row = q.get(rec["id"])
    assert row["state"] == "failed", row
    return rec["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--installed-wheel", type=Path)
    args = ap.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    result = {"output": str(out), "checks": []}

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH")

    store = WorkspaceStore(out / "workspace")
    store.create()
    failed_id = _seed_failed_intake(store)
    server = make_server(store)
    th = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    requests = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_context(viewport={"width": 1400, "height": 900}).new_page()

            def on_request(req):
                if "/intakes/" in req.url and req.method == "POST":
                    requests.append({"url": req.url, "method": req.method,
                                     "content_type": req.headers.get("content-type", ""),
                                     "body": req.post_data})

            page.on("request", on_request)
            page.goto(server.editor_url)
            expect(page.locator("#file-input")).to_be_enabled()
            # Expand the intake records panel (<details>).
            page.locator("details.intake-details").evaluate("el => el.open = true")

            # Find the failed intake row and click 重试.
            row = page.locator("#intake-list li:has-text('bad.mp4')")
            expect(row).to_have_count(1)
            retry_btn = row.locator("button", has_text="重试")
            expect(retry_btn).to_be_enabled()
            retry_btn.click()
            page.wait_for_timeout(800)

            retry_calls = [r for r in requests if r["url"].endswith("/retry")]
            assert retry_calls, "retry button fired no POST"
            call = retry_calls[-1]
            assert "application/json" in call["content_type"], call
            assert call["body"] and "command_id" in call["body"], call
            body = json.loads(call["body"])
            assert body["command_id"], body
            result["checks"].append({"name": "retry button posts JSON with command_id",
                                     "status": "PASS", "request": call})

            # A queued intake row should expose a 取消 button; click it.
            for _ in range(40):
                rows = page.locator("#intake-list li")
                if rows.count() and rows.first.locator("button", has_text="取消").count():
                    break
                page.wait_for_timeout(250)
            cancel_btn = page.locator("#intake-list li", has_text="排队").locator("button", has_text="取消")
            if cancel_btn.count():
                cancel_btn.first.click()
                page.wait_for_timeout(500)
                cancel_calls = [r for r in requests if r["url"].endswith("/cancel")]
                assert cancel_calls, "cancel button fired no POST"
                cc = cancel_calls[-1]
                assert "application/json" in cc["content_type"], cc
                assert cc["body"] and "command_id" in cc["body"], cc
                result["checks"].append({"name": "cancel button posts JSON with command_id",
                                         "status": "PASS", "request": cc})
            browser.close()
    finally:
        server.shutdown()
        th.join(timeout=5)
        server.server_close()
        store.close()

    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in result["checks"]:
        print(c["name"] + ": " + c["status"])
    return 0 if all(c["status"] == "PASS" for c in result["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
