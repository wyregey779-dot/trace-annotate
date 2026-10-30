"""P09 browser controls: real clicks, real requests, real database transitions.

Four cases are irreplaceable by Python API tests and are each executed here:

``UI_CANCEL_QUEUED``
    The row is genuinely ``queued`` in SQLite and its cancel button is reachable;
    the click must produce a legal response, a persisted receipt, a terminal
    ``cancelled`` row and **no** task, and a reload must agree.
``UI_CANCEL_PREPARING``
    A real ffmpeg child of this process must be alive and associated with the
    intake (its argv names the intake's attempt directory) before the click;
    afterwards the row must be ``cancelled``, the child gone, no task created, a
    late commit rejected and a reload still consistent.
``UI_RETRY_TRANSIENT_FAILURE``
    A valid upload fails once because of a controllable transient condition
    (its staged bytes are briefly corrupted); the retry button must produce a
    real successful prepare with exactly one task that the editor can open.
``UI_CANCEL_TERMINAL_REPLAY``
    Replaying the same ``command_id`` against a terminal row must return the
    first response unchanged and mutate nothing.

Two controlled faults prove the script can actually fail:

``--fault missing-control`` removes the button after load, so the reachability
assertion must fail. ``--fault no-effect`` fakes a 200 response without touching
the database, so the post-condition assertion must fail. In fault mode the
script exits 0 only when the case *did* fail.

Every case records the click, the HTTP request and response, the database before
and after, the state trace, the child-process exit and the receipt. The script
exits 0 only when the complete expected case set ran and every case passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
INSTALLED_WHEEL = None
if "--installed-wheel" in sys.argv:
    INSTALLED_WHEEL = Path(sys.argv[sys.argv.index("--installed-wheel") + 1]).resolve()
else:
    sys.path.insert(0, str(SOURCE))

import importlib.util  # noqa: E402

from playwright.sync_api import sync_playwright  # noqa: E402

REQUIRED_CASES = ["UI_CANCEL_QUEUED", "UI_CANCEL_PREPARING", "UI_RETRY_TRANSIENT_FAILURE",
                  "UI_CANCEL_TERMINAL_REPLAY"]
POLL_SECONDS = 0.2
CANCEL_STOP_BUDGET = 60.0      # seconds for the media child to exit after a cancel click
TERMINAL_BUDGET = 60.0         # seconds for a row to reach a terminal state
PREPARE_START_BUDGET = 120.0   # seconds to observe a real prepare child
UI_SYNC_BUDGET = 20.0          # seconds for the intake list poller to catch up


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def verify_installed_wheel(wheel: Path) -> dict:
    """Prove the imported product came from this wheel, not from the source tree."""
    spec = importlib.util.find_spec("trace_annotate")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    report = {"wheel": str(wheel), "wheel_sha256": sha256_file(wheel),
              "module_origin": str(origin), "checked": 0, "mismatched": [], "missing": []}
    if origin is None or "site-packages" not in str(origin).replace("\\", "/"):
        report["error"] = "trace_annotate was not imported from site-packages"
        return report
    root = origin.parent
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("trace_annotate/") or not member.endswith(".py"):
                continue
            target = root / member[len("trace_annotate/"):]
            report["checked"] += 1
            if not target.is_file():
                report["missing"].append(member)
                continue
            if hashlib.sha256(archive.read(member)).hexdigest() != sha256_file(target):
                report["mismatched"].append(member)
    return report


class CaseFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaseFailure(message)


def wait_for(predicate, *, budget: float, interval: float = POLL_SECONDS,
             message: str = "condition not reached"):
    deadline = time.monotonic() + budget
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise CaseFailure(f"{message} within {budget:.0f}s (last={last!r})")


def make_video(path: Path, *, seconds: float, pattern: str, size: str = "320x240") -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not on PATH; the browser control gate cannot run")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"{pattern}=duration={seconds}:size={size}:rate=10",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


class Harness:
    def __init__(self, out: Path, *, fault: str | None):
        self.out = out
        self.fault = fault
        self.case_records: list[dict] = []
        self.shots = out / "screenshots"
        self.shots.mkdir(parents=True, exist_ok=True)
        self.requests: list[dict] = []
        self.responses: list[dict] = []
        self.page_errors: list[str] = []
        self.first_cancel_response: dict | None = None
        self.first_cancel_command: str | None = None
        self.slot = None
        self.seed = None
        self.live = None
        self.strip_controls = False

    # -- evidence helpers ------------------------------------------------
    def shot(self, name: str) -> str:
        path = self.shots / f"{name}.png"
        self.page.screenshot(path=str(path), full_page=True)
        return str(path.relative_to(self.out)).replace("\\", "/")

    def db_row(self, intake_id: str) -> dict | None:
        return self.intakes.get(intake_id)

    def task_count_for(self, media_id: str | None) -> int:
        if not media_id:
            return 0
        with self.store.transaction():
            row = self.store.conn.execute("SELECT COUNT(*) FROM tasks WHERE media_id=?",
                                          (media_id,)).fetchone()
        return int(row[0])

    def receipt(self, command_id: str) -> dict | None:
        with self.store.transaction():
            row = self.store.conn.execute(
                "SELECT command_id,task_id,request_sha256,state,response,created_at "
                "FROM command_receipts WHERE command_id=?", (command_id,)).fetchone()
        if row is None:
            return None
        return {"command_id": row[0], "task_id": row[1], "request_sha256": row[2],
                "state": row[3], "response": row[4], "created_at": row[5]}

    def state_of(self, intake_id: str) -> str | None:
        row = self.db_row(intake_id)
        return None if row is None else row["state"]

    # -- page helpers ----------------------------------------------------
    def row(self, intake_id: str):
        return self.page.locator(f'#intake-list li[data-intake-id="{intake_id}"]')

    def button(self, intake_id: str, action: str):
        return self.page.locator(
            f'#intake-list li[data-intake-id="{intake_id}"] button[data-action="{action}"]')

    def open_panel(self) -> None:
        self.page.locator("details.intake-details").evaluate("el => el.open = true")

    def reload(self) -> None:
        self.page.reload(wait_until="load")
        self.page.wait_for_selector("#file-input", state="attached")
        self.open_panel()

    def wait_row_state(self, intake_id: str, state: str) -> None:
        wait_for(lambda: self.row(intake_id).count() == 1
                 and self.row(intake_id).get_attribute("data-intake-state") == state,
                 budget=UI_SYNC_BUDGET,
                 message=f"row for {intake_id} never showed state {state}")

    def last_command(self) -> dict:
        log = self.page.evaluate("() => window.__traceAnnotateIntakeCommands()")
        require(bool(log), "the editor recorded no intake command")
        return log[-1]

    def click_and_capture(self, button) -> tuple[dict, dict]:
        before = len(self.requests)
        with self.page.expect_response(
                lambda r: r.request.method == "POST" and "/intakes/" in r.url
                and (r.url.endswith("/cancel") or r.url.endswith("/retry"))) as info:
            button.click()
        response = info.value
        body = response.json()
        request = {"url": response.request.url, "method": response.request.method,
                   "content_type": response.request.headers.get("content-type", ""),
                   "body": response.request.post_data, "at": time.time()}
        reply = {"url": response.url, "status": response.status, "body": body, "at": time.time()}
        require(len(self.requests) > before, "the click produced no observable request")
        return request, reply

    def replay_cancel(self, intake_id: str, command_id: str) -> dict:
        return self.page.evaluate(
            """async ([id, cid]) => {
                 const r = await request('/intakes/' + encodeURIComponent(id) + '/cancel', {
                   method: 'POST',
                   body: JSON.stringify({command_id: cid}),
                   headers: {'Content-Type': 'application/json'}
                 });
                 return {status: r.status, body: await r.json()};
               }""", [intake_id, command_id])

    # -- cases -----------------------------------------------------------
    def cancel_queued(self, intake_id: str) -> dict:
        detail = {"intake_id": intake_id, "trace": []}
        row = wait_for(lambda: (self.db_row(intake_id) or {}).get("state") == "queued"
                       and self.db_row(intake_id), budget=30,
                       message="intake never reached queued in the database")
        detail["db_before"] = {"state": row["state"], "task_id": row["task_id"],
                               "attempt": row["attempt"], "generation": row["generation"],
                               "worker_id": row["worker_id"], "lease_expires_at": row["lease_expires_at"]}
        detail["trace"].append({"db": "queued", "at": time.time()})

        self.reload()
        if self.strip_controls:
            # Controlled fault: the control never appears, so the reachability
            # assertion below must fail. A script that skips missing buttons
            # would still "pass" here, which is exactly what we are testing.
            self.page.evaluate(
                "() => document.querySelectorAll("
                "\"#intake-list button[data-action='cancel']\").forEach(b => b.remove())")
        require(self.row(intake_id).count() == 1, "the target intake row is not rendered")
        require(self.row(intake_id).get_attribute("data-intake-state") == "queued",
                "the rendered row is not in the queued state")
        button = self.button(intake_id, "cancel")
        require(button.count() == 1, "the cancel button is missing on the queued row")
        require(button.is_enabled(), "the cancel button is disabled")
        detail["screenshot_before"] = self.shot("UI_CANCEL_QUEUED-before")

        request, reply = self.click_and_capture(button)
        detail["request"] = request
        detail["response"] = reply
        require(reply["status"] == 200, f"cancel returned HTTP {reply['status']}")
        require(reply["body"].get("state") == "cancelled",
                f"cancel response state is {reply['body'].get('state')!r}")

        command = self.last_command()
        require(command["action"] == "cancel", f"editor recorded action {command['action']!r}")
        require(command["intakeId"] == intake_id, "editor recorded a different intake id")
        command_id = command["commandId"]
        require(isinstance(command_id, str) and command_id, "no command id was recorded")
        detail["command_id"] = command_id
        self.first_cancel_command = command_id
        self.first_cancel_response = reply["body"]

        final = wait_for(lambda: self.state_of(intake_id) == "cancelled" and self.db_row(intake_id),
                         budget=TERMINAL_BUDGET, message="intake never became cancelled")
        detail["db_after"] = {"state": final["state"], "task_id": final["task_id"],
                              "media_id": final["media_id"], "attempt": final["attempt"],
                              "worker_id": final["worker_id"]}
        require(final["task_id"] is None, "a cancelled queued intake still has a task id")
        require(final["media_id"] is None, "a cancelled queued intake still has a media id")
        require(self.task_count_for(final["media_id"]) == 0, "a task was created for a cancelled intake")
        receipt = self.receipt(command_id)
        require(receipt is not None, "no persisted command receipt for the cancel")
        detail["receipt"] = receipt
        require(receipt["state"] == "committed", f"receipt state is {receipt['state']!r}")

        self.reload()
        self.wait_row_state(intake_id, "cancelled")
        require(self.button(intake_id, "cancel").count() == 0,
                "a terminal row still offers a cancel control")
        detail["screenshot_after"] = self.shot("UI_CANCEL_QUEUED-after")
        detail["trace"].append({"db": "cancelled", "at": time.time()})
        return detail

    def cancel_preparing(self, intake_id: str, slow_media: Path) -> dict:
        detail = {"intake_id": intake_id, "trace": [], "media": str(slow_media)}
        self.slot.release()  # let the drainer claim and start a real prepare
        claimed = wait_for(lambda: (self.db_row(intake_id) or {}).get("state") == "preparing"
                           and self.db_row(intake_id), budget=PREPARE_START_BUDGET,
                           message="prepare never started")
        detail["db_claimed"] = {"state": claimed["state"], "generation": claimed["generation"],
                                "attempt": claimed["attempt"], "worker_id": claimed["worker_id"],
                                "lease_expires_at": claimed["lease_expires_at"]}
        detail["trace"].append({"db": "preparing", "at": time.time()})

        def children_for_intake():
            return [item for item in self.live() if intake_id in " ".join(item.get("argv") or [])]

        child = wait_for(lambda: children_for_intake() or None, budget=PREPARE_START_BUDGET,
                         message="no live media child was associated with the intake")
        child = child[0]
        detail["child"] = child
        require(pid_alive(child["pid"]), f"media child {child['pid']} is not alive")
        require(child["parent_pid"] == os.getpid(),
                f"media child parent {child['parent_pid']} is not this process {os.getpid()}")
        require(self.state_of(intake_id) == "preparing", "the intake left preparing before the click")

        # No page reload here: the intake poller already reflects the state, and
        # a reload would burn most of the prepare window before the click.
        self.wait_row_state(intake_id, "preparing")
        button = self.button(intake_id, "cancel")
        require(button.count() == 1, "the cancel button is missing on the preparing row")
        require(button.is_enabled(), "the cancel button is disabled on the preparing row")
        detail["screenshot_before"] = self.shot("UI_CANCEL_PREPARING-before")

        request, reply = self.click_and_capture(button)
        detail["request"] = request
        detail["response"] = reply
        require(reply["status"] == 200, f"cancel returned HTTP {reply['status']}")
        require(reply["body"].get("state") in ("cancel_requested", "cancelled"),
                f"cancel response state is {reply['body'].get('state')!r} "
                f"(the prepare finished before the click was delivered)")
        command = self.last_command()
        require(command["action"] == "cancel" and command["intakeId"] == intake_id,
                "the editor did not record this cancel click")
        detail["command_id"] = command["commandId"]
        detail["trace"].append({"db": reply["body"].get("state"), "at": time.time()})

        final = wait_for(lambda: self.state_of(intake_id) == "cancelled" and self.db_row(intake_id),
                         budget=TERMINAL_BUDGET, message="cancelled preparing intake never settled")
        detail["db_after"] = {"state": final["state"], "task_id": final["task_id"],
                              "media_id": final["media_id"], "attempt": final["attempt"]}
        require(final["task_id"] is None, "the cancelled preparing intake created a task")
        require(self.task_count_for(final["media_id"]) == 0, "a task row exists for a cancelled intake")

        wait_for(lambda: not pid_alive(child["pid"]), budget=CANCEL_STOP_BUDGET,
                 message=f"media child {child['pid']} did not exit after the cancel")
        detail["child_exited"] = {"pid": child["pid"], "alive": pid_alive(child["pid"]),
                                  "at": time.time()}
        # A late prepare result must not be able to commit after the cancel.
        time.sleep(3.0)
        require(self.state_of(intake_id) == "cancelled", "a late result overwrote the cancellation")
        require(self.db_row(intake_id)["task_id"] is None, "a late result created a task")
        require(self.task_count_for(self.db_row(intake_id)["media_id"]) == 0,
                "a late result created a media row")

        self.reload()
        self.wait_row_state(intake_id, "cancelled")
        detail["screenshot_after"] = self.shot("UI_CANCEL_PREPARING-after")
        return detail

    def retry_transient(self, key: str, source: Path) -> dict:
        detail = {"trace": []}
        # Hold the decode slot so the corrupt bytes are in place before any claim.
        self.slot.acquire()
        copy = self.store.tmp_dir / f"upload-{key}.tmp"
        shutil.copy2(source, copy)
        receipt = self.seed.enqueue(copy, display_name="retry.mp4")
        intake_id = receipt["id"]
        detail["intake_id"] = intake_id
        staged = self.store.root / receipt["staged_path"]
        original = staged.read_bytes()
        staged.write_bytes(original + b"\x00transient-corruption")
        detail["transient_fault"] = {"kind": "staged_bytes_corrupted",
                                     "sha256_corrupt": sha256_file(staged)}
        self.slot.release()

        failed = wait_for(lambda: self.state_of(intake_id) == "failed" and self.db_row(intake_id),
                          budget=TERMINAL_BUDGET, message="the transient failure never happened")
        detail["db_failed"] = {"state": failed["state"], "error_code": failed["error_code"],
                               "attempt": failed["attempt"]}
        require(failed["error_code"] == "MEDIA_CHANGED",
                f"unexpected failure code {failed['error_code']!r}")

        # Remove the transient condition, then retry through the real button.
        staged.write_bytes(original)
        detail["transient_fault"]["restored_sha256"] = sha256_file(staged)
        self.reload()
        self.wait_row_state(intake_id, "failed")
        button = self.button(intake_id, "retry")
        require(button.count() == 1, "the retry button is missing on the failed row")
        require(button.is_enabled(), "the retry button is disabled")
        detail["screenshot_before"] = self.shot("UI_RETRY_TRANSIENT_FAILURE-before")

        request, reply = self.click_and_capture(button)
        detail["request"] = request
        detail["response"] = reply
        require(reply["status"] == 200, f"retry returned HTTP {reply['status']}")
        require(reply["body"].get("state") == "queued",
                f"retry response state is {reply['body'].get('state')!r}")
        command = self.last_command()
        require(command["action"] == "retry" and command["intakeId"] == intake_id,
                "the editor did not record this retry click")
        detail["command_id"] = command["commandId"]

        done = wait_for(lambda: self.state_of(intake_id) == "succeeded" and self.db_row(intake_id),
                        budget=TERMINAL_BUDGET, message="the retry never succeeded")
        detail["db_after"] = {"state": done["state"], "task_id": done["task_id"],
                              "media_id": done["media_id"], "attempt": done["attempt"]}
        require(done["task_id"], "a successful retry has no task id")
        require(done["attempt"] == 2, f"retry ran as attempt {done['attempt']}, expected 2")
        require(self.task_count_for(done["media_id"]) == 1,
                "the retried intake produced more than one task")

        # The editor must be able to open and read the produced task.
        self.reload()
        self.wait_row_state(intake_id, "succeeded")
        task_button = self.page.locator(f'button.task-button[data-task-id="{done["task_id"]}"]')
        wait_for(lambda: task_button.count() == 1, budget=UI_SYNC_BUDGET,
                 message="the retried task is not listed in the editor")
        task_button.click()
        wait_for(lambda: "retry.mp4" in (self.page.locator("#media-name").inner_text() or ""),
                 budget=UI_SYNC_BUDGET, message="the editor did not open the retried task")
        detail["editor_media_name"] = self.page.locator("#media-name").inner_text()
        detail["screenshot_after"] = self.shot("UI_RETRY_TRANSIENT_FAILURE-after")
        return detail

    def cancel_terminal_replay(self, intake_id: str) -> dict:
        detail = {"intake_id": intake_id}
        command_id = self.first_cancel_command
        require(bool(command_id), "no earlier cancel command id to replay")
        require(self.state_of(intake_id) == "cancelled", "the target intake is not terminal")
        before_row = self.db_row(intake_id)
        before_receipt = self.receipt(command_id)
        require(before_receipt is not None, "the original cancel has no persisted receipt")
        detail["command_id"] = command_id
        detail["db_before"] = {"state": before_row["state"], "task_id": before_row["task_id"]}
        detail["receipt_before"] = before_receipt

        self.reload()
        self.wait_row_state(intake_id, "cancelled")
        require(self.button(intake_id, "cancel").count() == 0,
                "a terminal row still offers a cancel control")

        reply = self.replay_cancel(intake_id, command_id)
        detail["replay_response"] = reply
        require(reply["status"] == 200, f"replay returned HTTP {reply['status']}")
        require(reply["body"] == self.first_cancel_response,
                "the replay did not return the first response unchanged")

        after_row = self.db_row(intake_id)
        after_receipt = self.receipt(command_id)
        detail["db_after"] = {"state": after_row["state"], "task_id": after_row["task_id"]}
        detail["receipt_after"] = after_receipt
        require(after_receipt == before_receipt, "the replay changed the persisted receipt")
        require(after_row["state"] == "cancelled", "the replay changed the terminal state")
        require(after_row["task_id"] is None, "the replay created a task")
        detail["screenshot_after"] = self.shot("UI_CANCEL_TERMINAL_REPLAY-after")
        return detail

    # -- run -------------------------------------------------------------
    def run_case(self, name: str, function) -> dict:
        record = {"case_id": name, "status": "FAIL", "started_at": time.time()}
        try:
            record["detail"] = function()
            record["status"] = "PASS"
        except Exception as exc:  # noqa: BLE001 - recorded as a failed case
            record["error"] = f"{type(exc).__name__}: {exc}"
            try:
                record["screenshot"] = self.shot(f"FAIL-{name}")
            except Exception:
                pass
        record["finished_at"] = time.time()
        self.case_records.append(record)
        suffix = "" if record["status"] == "PASS" else f" ({record.get('error')})"
        print(f"{name}: {record['status']}{suffix}", flush=True)
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description="P09 real-browser intake controls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-wheel", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--headed", dest="headless", action="store_false")
    parser.add_argument("--fault", choices=["missing-control", "no-effect"], default=None)
    parser.add_argument("--slow-media-seconds", type=float, default=90.0)
    parser.add_argument("--slow-media-size", default="854x480")
    args = parser.parse_args()

    out = args.output.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    (out / "media").mkdir()

    identity = {"mode": "source" if INSTALLED_WHEEL is None else "installed-wheel"}
    if INSTALLED_WHEEL is not None:
        identity.update(verify_installed_wheel(INSTALLED_WHEEL))
        if identity.get("error") or identity["mismatched"] or identity["missing"]:
            (out / "result.json").write_text(json.dumps(
                {"output": str(out), "identity": identity, "cases": [],
                 "overall": "FAIL", "error": "installed wheel identity check failed"},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print("installed wheel identity check failed", identity)
            return 2

    from trace_annotate.temporal.preparation import live_media_processes
    from trace_annotate.workspace.intake import AsyncIntakes
    from trace_annotate.workspace.server import make_server
    from trace_annotate.workspace.slots import get_prepare_slot
    from trace_annotate.workspace.store import WorkspaceStore

    harness = Harness(out, fault=args.fault)
    harness.live = live_media_processes
    store = WorkspaceStore(out / "workspace")
    store.create()
    harness.store = store

    slot = get_prepare_slot()
    harness.slot = slot
    slot.acquire()  # keep the drainer from claiming until a case asks for it

    queued_media = make_video(out / "media" / "queued.mp4", seconds=1.0, pattern="testsrc")
    slow_media = make_video(out / "media" / "slow.mp4", seconds=args.slow_media_seconds,
                            pattern="smptebars", size=args.slow_media_size)
    retry_media = make_video(out / "media" / "retry.mp4", seconds=1.0, pattern="rgbtestsrc")

    seed = AsyncIntakes(store)
    harness.seed = seed
    for key, source in (("queued", queued_media), ("slow", slow_media)):
        copy = store.tmp_dir / f"upload-{key}.tmp"
        shutil.copy2(source, copy)
    queued = seed.enqueue(store.tmp_dir / "upload-queued.tmp", display_name="queued.mp4")
    slow = seed.enqueue(store.tmp_dir / "upload-slow.tmp", display_name="slow.mp4")

    server = make_server(store)
    harness.intakes = server.async_intakes
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=args.headless, channel="chrome")
            context = browser.new_context(viewport={"width": 1440, "height": 900},
                                          accept_downloads=False)
            page = context.new_page()
            harness.page = page
            page.on("pageerror", lambda exc: harness.page_errors.append(str(exc)))
            page.on("request", lambda req: harness.requests.append(
                {"url": req.url, "method": req.method,
                 "content_type": req.headers.get("content-type", ""), "body": req.post_data,
                 "at": time.time()}))
            page.on("response", lambda res: harness.responses.append(
                {"url": res.url, "status": res.status, "at": time.time()}))

            page.goto(server.editor_url, wait_until="load")
            page.wait_for_selector("#file-input", state="attached")
            harness.open_panel()

            if args.fault == "missing-control":
                harness.strip_controls = True
                harness.run_case("UI_MISSING_CONTROL_NEGATIVE",
                                 lambda: harness.cancel_queued(queued["id"]))
            elif args.fault == "no-effect":
                page.route("**/intakes/*/cancel", lambda route: route.fulfill(
                    status=200, content_type="application/json",
                    body=json.dumps({"id": queued["id"], "state": "cancelled"})))
                harness.run_case("UI_CANCEL_NO_EFFECT_NEGATIVE",
                                 lambda: harness.cancel_queued(queued["id"]))
            else:
                harness.run_case("UI_CANCEL_QUEUED", lambda: harness.cancel_queued(queued["id"]))
                harness.run_case("UI_CANCEL_PREPARING",
                                 lambda: harness.cancel_preparing(slow["id"], slow_media))
                harness.run_case("UI_RETRY_TRANSIENT_FAILURE",
                                 lambda: harness.retry_transient("retry", retry_media))
                harness.run_case("UI_CANCEL_TERMINAL_REPLAY",
                                 lambda: harness.cancel_terminal_replay(queued["id"]))

            context.close()
            browser.close()
    finally:
        try:
            slot.release()
        except RuntimeError:
            pass
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        store.close()

    expected = (["UI_MISSING_CONTROL_NEGATIVE"] if args.fault == "missing-control"
                else ["UI_CANCEL_NO_EFFECT_NEGATIVE"] if args.fault == "no-effect"
                else REQUIRED_CASES)
    seen = [record["case_id"] for record in harness.case_records]
    complete = seen == expected
    passed = bool(harness.case_records) and all(r["status"] == "PASS" for r in harness.case_records)

    if args.fault:
        exit_code = 0 if (complete and not passed) else 1
        overall = "NEGATIVE_CONTROL_OK" if exit_code == 0 else "NEGATIVE_CONTROL_BLIND"
    else:
        exit_code = 0 if (complete and passed) else 1
        overall = "PASS" if exit_code == 0 else "FAIL"

    result = {
        "document_type": "BROWSER_CONTROL_RESULT",
        "result_version": 1,
        "output": str(out),
        "identity": identity,
        "overall": overall,
        "expected_cases": expected,
        "cases": harness.case_records,
        "case_set_complete": complete,
        "all_passed": passed,
        "page_errors": harness.page_errors,
        "http_requests": [r for r in harness.requests if "/intakes/" in r["url"]],
        "http_responses": [r for r in harness.responses if "/intakes/" in r["url"]],
        "workspace": str(out / "workspace"),
        "screenshots": sorted(p.name for p in harness.shots.glob("*.png")),
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"overall: {overall} (cases={len(seen)} expected={len(expected)})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
