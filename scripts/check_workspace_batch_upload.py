"""P07/P08 real-browser batch upload gate.

Eight cases that cannot be proven from Python alone are executed here in real
Chrome against a real workspace, real SQLite and real ffmpeg children:

``BATCH_ACCEPT_BEFORE_PREPARE_DONE``
    A second file is accepted (202) while the first intake is genuinely
    ``preparing`` in the database, and the accepted row is visible in the
    editor without waiting for any preparation.
``UPLOAD_LOST_RESPONSE_RECONCILE``
    The upload response is destroyed after the server committed it. The editor
    must report "unknown" first, then adopt the intake through the original
    command id, and must never upload the same file again under a new id.
``UPLOAD_REFRESH_AND_RESELECT``
    The batch survives a reload, a reload during a partial upload settles as
    "unknown" instead of a duplicate, identical content reselected later is
    reused, and same-name/different-content reselection is not treated as the
    same item.
``BATCH_MIXED_IDENTITIES``
    One selection mixing a valid clip, a corrupt file, an exact duplicate and a
    same-name/different-content pair ends with the right rows and no duplicate
    tasks.
``BATCH_STOP_UPLOAD_AND_QUEUED``
    Stop aborts the in-flight body, marks the not-yet-sent files as not
    submitted, reconciles the aborted one by its original id, and the batch's
    queued rows can be cancelled together.
``BATCH_STOP_PREPARING_ISOLATION``
    Cancelling this batch stops its own preparing child, leaves a seeded
    foreign intake queued and leaves the editing task untouched.
``EDIT_DRAFT_SURVIVES_IMPORT``
    Chinese field text, a review flag, the selection, the player position, the
    undo history and the synced draft all survive an import.
``STATE_FAILURE_NO_DUPLICATE``
    A 503 on the snapshot keeps the last good state and flags the status as
    unavailable; a 503 on the intake list does not turn into a re-upload.

Two controlled faults prove the script can fail:

``--fault missing-control`` removes the batch controls after load, so the
reachability assertion must fail. ``--fault no-effect`` replaces the stop and
cancel actions with no-ops, so the post-condition assertions must fail. In fault
mode the script exits 0 only when the case did fail.

Every case records the real request/response pair, the database before and
after, the live ffmpeg child, the batch trace and a screenshot. The script exits
0 only when the complete expected case set ran and every case passed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
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

from playwright.sync_api import sync_playwright  # noqa: E402

REQUIRED_CASES = [
    "BATCH_ACCEPT_BEFORE_PREPARE_DONE",
    "UPLOAD_LOST_RESPONSE_RECONCILE",
    "UPLOAD_REFRESH_AND_RESELECT",
    "BATCH_MIXED_IDENTITIES",
    "BATCH_STOP_UPLOAD_AND_QUEUED",
    "BATCH_STOP_PREPARING_ISOLATION",
    "EDIT_DRAFT_SURVIVES_IMPORT",
    "STATE_FAILURE_NO_DUPLICATE",
]
POLL_SECONDS = 0.1
UI_SYNC_BUDGET = 30.0         # seconds for the editor's own polling to catch up
PREPARE_START_BUDGET = 180.0  # seconds to observe a real prepare child
TERMINAL_BUDGET = 240.0       # seconds for an intake to reach a terminal state

IMPORT_ROUTE = re.compile(r"/imports\?async=1$")
COMMAND_ROUTE = re.compile(r"/commands/[^/]+$")
SNAPSHOT_ROUTE = re.compile(r"/snapshot$")
INTAKE_LIST_ROUTE = re.compile(r"/intakes$")
CANCEL_ROUTE = re.compile(r"/intakes/[^/]+/cancel$")


# --------------------------------------------------------------------------
# helpers
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


def make_video(path: Path, *, seconds: float, pattern: str, size: str = "320x240",
               rate: int = 10) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not on PATH; the browser batch gate cannot run")
    # mpeg4 keeps the encoder's memory footprint small enough to run while the
    # rest of the suite is active.
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"{pattern}=duration={seconds}:size={size}:rate={rate}",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=600)
    return path


def same_size_different_bytes(source: Path, destination: Path) -> Path:
    """Copy a clip and flip a byte in the middle, keeping the exact size."""
    raw = bytearray(source.read_bytes())
    middle = len(raw) // 2
    raw[middle] = raw[middle] ^ 0xFF
    destination.write_bytes(bytes(raw))
    return destination


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
        self.store = None
        self.page = None
        self.live = lambda: []
        self.strip_controls = False
        self.no_effect = False

    # -- evidence helpers ------------------------------------------------
    def shot(self, name: str) -> str:
        path = self.shots / f"{name}.png"
        self.page.screenshot(path=str(path), full_page=True)
        return str(path.relative_to(self.out)).replace("\\", "/")

    def db_intake(self, intake_id: str) -> dict | None:
        with self.store.transaction():
            cursor = self.store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            names = [column[0] for column in cursor.description]
        return dict(zip(names, row))

    def db_intakes(self) -> list[dict]:
        with self.store.transaction():
            cursor = self.store.conn.execute("SELECT * FROM intakes ORDER BY rowid")
            names = [column[0] for column in cursor.description]
            return [dict(zip(names, row)) for row in cursor]

    def intakes_for_name(self, name: str) -> list[dict]:
        return [row for row in self.db_intakes() if row["display_name"] == name]

    def task_count(self) -> int:
        with self.store.transaction():
            return int(self.store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    def receipt(self, command_id: str) -> dict | None:
        with self.store.transaction():
            row = self.store.conn.execute(
                "SELECT command_id,task_id,request_sha256,state,response,created_at "
                "FROM command_receipts WHERE command_id=?", (command_id,)).fetchone()
        if row is None:
            return None
        return {"command_id": row[0], "task_id": row[1], "request_sha256": row[2],
                "state": row[3],
                "response": json.loads(row[4]) if row[4] else None,
                "created_at": row[5]}

    def live_children_for(self, intake_id: str) -> list[dict]:
        return [record for record in self.live()
                if any(intake_id in str(part) for part in record.get("argv") or [])]

    # -- page helpers ----------------------------------------------------
    def batch(self) -> dict | None:
        return self.page.evaluate("() => window.__traceAnnotateBatch()")

    def items(self) -> list[dict]:
        payload = self.batch()
        return [] if not payload else payload["items"]

    def item(self, *, name: str | None = None, command_id: str | None = None) -> dict | None:
        for entry in self.items():
            if name is not None and entry["name"] == name:
                return entry
            if command_id is not None and entry["commandId"] == command_id:
                return entry
        return None

    def items_named(self, name: str) -> list[dict]:
        return [entry for entry in self.items() if entry["name"] == name]

    def wait_phase(self, name: str, phases, *, budget: float = UI_SYNC_BUDGET,
                   message: str | None = None) -> dict:
        wanted = {phases} if isinstance(phases, str) else set(phases)
        return wait_for(lambda: (self.item(name=name) or {}).get("phase") in wanted
                        and self.item(name=name),
                        budget=budget,
                        message=message or f"batch item {name} never reached {sorted(wanted)}")

    def open_panel(self) -> None:
        self.page.locator("details.intake-details").evaluate("el => el.open = true")

    def open_task(self, task_id: str, display_name: str) -> None:
        """Click a task and wait until the editor really shows it.

        Waiting for ``#editor:not([hidden])`` is not enough: the editor is
        already visible for the auto-selected task, so a form write would race
        the load and be overwritten by ``fillForm``.
        """
        self.page.locator(f'button.task-button[data-task-id="{task_id}"]').click()
        wait_for(lambda: self.page.text_content("#media-name") == display_name
                 and self.page.text_content("#save-state") not in {"正在加载…", "正在保存…"},
                 budget=UI_SYNC_BUDGET,
                 message=f"the editor never finished opening {display_name}")
        wait_for(lambda: self.page.evaluate(
            "() => document.getElementById('player').readyState >= 1"),
            budget=UI_SYNC_BUDGET, message="the player never became ready")

    def reset_page(self, *, keep_batch: bool = False) -> None:
        if not keep_batch:
            self.page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
        self.page.reload(wait_until="load")
        self.page.wait_for_selector("#file-input", state="attached")
        self.open_panel()

    def select_files(self, paths) -> None:
        self.page.set_input_files("#file-input", [str(path) for path in paths])

    def posts_to(self, needle: str) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST" and needle in r["url"]]

    def install_batch_trace(self) -> None:
        self.page.evaluate(
            """() => {
                 window.__batchTrace = [];
                 if (window.__batchTraceTimer) clearInterval(window.__batchTraceTimer);
                 window.__batchTraceTimer = setInterval(() => {
                   const b = window.__traceAnnotateBatch();
                   if (!b) return;
                   window.__batchTrace.push({at: Date.now(),
                     items: b.items.map(i => [i.commandId, i.phase])});
                 }, 25);
               }""")

    def batch_trace(self) -> list[dict]:
        return self.page.evaluate("() => window.__batchTrace || []")

    def phases_seen(self, command_id: str) -> list[str]:
        seen = []
        for frame in self.batch_trace():
            for cid, phase in frame["items"]:
                if cid == command_id and (not seen or seen[-1] != phase):
                    seen.append(phase)
        return seen

    # -- cases -----------------------------------------------------------
    def case_accept_before_prepare(self, slow: Path, small: Path) -> dict:
        detail = {"slow": str(slow), "small": str(small), "trace": []}
        self.reset_page()
        posts_before = len(self.posts_to("/imports?async=1"))
        self.select_files([slow])
        first = self.wait_phase(slow.name, "accepted", budget=PREPARE_START_BUDGET,
                                message="the first upload was never accepted")
        detail["first"] = {k: first[k] for k in ("commandId", "phase", "intakeId")}
        require(first["intakeId"], "the accepted item has no intake id")
        intake_id = first["intakeId"]
        preparing = wait_for(lambda: (self.db_intake(intake_id) or {}).get("state") == "preparing"
                             and self.db_intake(intake_id),
                             budget=PREPARE_START_BUDGET,
                             message="the first intake never started preparing")
        detail["db_preparing"] = {k: preparing[k] for k in
                                  ("state", "attempt", "generation", "worker_id", "lease_expires_at")}
        detail["trace"].append({"db": "preparing", "at": time.time()})

        started = time.monotonic()
        with self.page.expect_response(
                lambda r: IMPORT_ROUTE.search(r.url) and r.request.method == "POST") as info:
            self.select_files([small])
        response = info.value
        elapsed = time.monotonic() - started
        status = response.status
        detail["second_response"] = {"status": status, "at": time.time()}
        require(status == 202, f"second upload returned HTTP {status}")
        second = self.wait_phase(small.name, {"accepted", "admitted"},
                                 budget=UI_SYNC_BUDGET,
                                 message="the second upload was never accepted")
        second_id = second["intakeId"]
        require(second_id, "the accepted second upload has no intake id")
        second_row = self.db_intake(second_id)
        detail["second_row"] = {k: second_row[k] for k in
                                ("id", "state", "task_id", "source_sha256", "source_size")}
        require(second_row["state"] in {"queued", "preparing"},
                f"the second intake is already {second_row['state']!r}")
        require(second_row["task_id"] is None, "the second upload already has a task")
        # The decisive observation: the first file is still decoding.
        still = self.db_intake(intake_id)
        require(still["state"] == "preparing",
                f"the first intake was {still['state']!r} when the second was accepted")
        require(elapsed < 30.0, f"the second upload waited {elapsed:.1f}s for the first prepare")
        detail["accepted_while_preparing"] = {"elapsed_seconds": round(elapsed, 3),
                                              "first_state": still["state"]}
        wait_for(lambda: self.page.locator(
            f'#intake-list li[data-intake-id="{second_id}"]').count() == 1,
            budget=UI_SYNC_BUDGET,
            message="the accepted intake was never rendered in the editor")
        detail["visible_row"] = True
        detail["screenshot"] = self.shot("BATCH_ACCEPT_BEFORE_PREPARE_DONE")

        wait_for(lambda: (self.db_intake(intake_id) or {}).get("state") == "succeeded",
                 budget=TERMINAL_BUDGET, message="the first intake never succeeded")
        wait_for(lambda: (self.db_intake(second_id) or {}).get("state") == "succeeded",
                 budget=TERMINAL_BUDGET, message="the second intake never succeeded")
        first_task = self.db_intake(intake_id)["task_id"]
        second_task = self.db_intake(second_id)["task_id"]
        require(first_task and second_task and first_task != second_task,
                "the two uploads did not produce two distinct tasks")
        detail["tasks"] = [first_task, second_task]
        detail["uploads"] = len(self.posts_to("/imports?async=1")) - posts_before
        require(detail["uploads"] == 2, f"{detail['uploads']} upload requests for 2 files")
        return detail

    def case_lost_response(self, small: Path) -> dict:
        detail = {"file": str(small), "trace": []}
        self.reset_page()
        intakes_before = len(self.db_intakes())
        self.install_batch_trace()
        attempts = {"count": 0}

        def lose_the_response(route):
            # The server really commits the upload; the client never sees the
            # reply. This is the only honest way to test reconciliation.
            try:
                route.fetch()
            except Exception:  # noqa: BLE001 - the abort below is the point
                pass
            route.abort()

        def flaky_command_lookup(route):
            attempts["count"] += 1
            if attempts["count"] <= 2:
                route.fulfill(status=503, content_type="application/json",
                              body=json.dumps({"error": {"code": "UNAVAILABLE",
                                                         "message": "temporarily unavailable"}}))
                return
            route.continue_()

        self.page.route(IMPORT_ROUTE, lose_the_response)
        self.page.route(COMMAND_ROUTE, flaky_command_lookup)
        posts_before = len(self.posts_to("/imports?async=1"))
        try:
            self.select_files([small])
            accepted = wait_for(
                lambda: (self.item(name=small.name) or {}).get("phase") in {"accepted", "admitted"}
                and self.item(name=small.name),
                budget=UI_SYNC_BUDGET,
                message="the lost upload was never reconciled into an accepted intake")
        finally:
            self.page.unroute(IMPORT_ROUTE)
            self.page.unroute(COMMAND_ROUTE)

        command_id = accepted["commandId"]
        phases = self.phases_seen(command_id)
        detail["command_id"] = command_id
        detail["phases_seen"] = phases
        require("unknown" in phases,
                f"the editor never reported an unknown outcome (saw {phases})")
        require(phases[-1] in {"accepted", "admitted"},
                f"the editor settled on {phases[-1]!r} instead of an accepted intake")
        require(attempts["count"] >= 3,
                f"only {attempts['count']} receipt lookups; the bounded retry never ran")

        receipt = self.receipt(command_id)
        require(receipt is not None, "the server stored no receipt for the lost response")
        detail["receipt"] = receipt
        require(receipt["state"] == "committed", f"receipt state is {receipt['state']!r}")
        require(accepted["intakeId"] == receipt["response"]["id"],
                "the editor adopted a different intake than the receipt names")
        uploads = len(self.posts_to("/imports?async=1")) - posts_before
        detail["uploads"] = uploads
        require(uploads == 1, f"the file was uploaded {uploads} times after a lost response")
        require(len(self.db_intakes()) - intakes_before == 1,
                f"{len(self.db_intakes()) - intakes_before} new intake rows for one lost upload")
        lookups = [r for r in self.requests if "/commands/" in r["url"]]
        detail["receipt_lookups"] = len(lookups)
        require(lookups, "the editor never queried the original command id")
        detail["screenshot"] = self.shot("UPLOAD_LOST_RESPONSE_RECONCILE")
        return detail

    def case_refresh_and_reselect(self, small: Path, other: Path, twin: Path) -> dict:
        detail = {"trace": []}
        # --- reload after a completed enqueue -----------------------------
        self.reset_page()
        self.select_files([small])
        accepted = self.wait_phase(small.name, {"accepted", "admitted"},
                                   budget=PREPARE_START_BUDGET)
        command_id = accepted["commandId"]
        self.reset_page(keep_batch=True)
        restored = wait_for(lambda: self.item(command_id=command_id), budget=UI_SYNC_BUDGET,
                            message="the batch item did not survive a reload")
        detail["after_reload"] = {k: restored[k] for k in ("commandId", "phase", "intakeId")}
        require(restored["intakeId"] == accepted["intakeId"],
                "the reloaded item points at a different intake")

        # --- reload during a partial upload -------------------------------
        self.reset_page()
        self.install_batch_trace()
        held: dict = {}

        def hold_the_body(route):
            held["route"] = route  # never resolved: the body stays in flight

        self.page.route(IMPORT_ROUTE, hold_the_body)
        posts_before = len(self.posts_to("/imports?async=1"))
        self.select_files([other])
        uploading = wait_for(lambda: (self.item(name=other.name) or {}).get("phase") == "uploading"
                             and self.item(name=other.name),
                             budget=UI_SYNC_BUDGET,
                             message="the upload never entered the in-flight phase")
        detail["partial_command"] = uploading["commandId"]
        detail["screenshot_partial"] = self.shot("UPLOAD_REFRESH_AND_RESELECT-partial")
        self.reset_page(keep_batch=True)
        try:
            held["route"].abort()
        except Exception:  # noqa: BLE001 - the navigation may already have failed it
            pass
        finally:
            self.page.unroute(IMPORT_ROUTE)
        settled = wait_for(
            lambda: (self.item(command_id=uploading["commandId"]) or {}).get("phase")
            in {"unknown", "accepted", "admitted"}
            and self.item(command_id=uploading["commandId"]),
            budget=UI_SYNC_BUDGET,
            message="the interrupted upload never settled")
        detail["after_partial_reload"] = {k: settled[k] for k in ("phase", "intakeId", "note")}
        require(settled["phase"] != "failed",
                "an interrupted upload was reported as a failure instead of an unknown outcome")
        rows = self.intakes_for_name(other.name)
        require(len(rows) <= 1, f"{len(rows)} intake rows for one interrupted upload")
        require(settled["phase"] == "unknown" or settled["intakeId"],
                "the interrupted upload settled with neither an intake nor an unknown outcome")
        uploads = len(self.posts_to("/imports?async=1")) - posts_before
        detail["partial_uploads"] = uploads
        require(uploads == 1, f"the interrupted upload was sent {uploads} times")

        # --- matching content reselection ---------------------------------
        self.reset_page()
        intakes_before = len(self.db_intakes())
        self.select_files([small])
        first = self.wait_phase(small.name, {"accepted", "admitted"},
                                budget=PREPARE_START_BUDGET)
        posts_before = len(self.posts_to("/imports?async=1"))
        self.select_files([small])
        reused = wait_for(lambda: len(self.items_named(small.name)) == 2
                          and self.items_named(small.name)[-1]["phase"] == "reused"
                          and self.items_named(small.name)[-1],
                          budget=UI_SYNC_BUDGET,
                          message="reselected identical content was not reused")
        detail["reused"] = {k: reused[k] for k in ("phase", "intakeId", "sha256", "note")}
        require(reused["intakeId"] == first["intakeId"],
                "the reused item points at a different intake")
        require(len(self.posts_to("/imports?async=1")) == posts_before,
                "identical content was uploaded a second time")
        require(len(self.db_intakes()) - intakes_before == 1,
                "identical content created a second intake row")

        # --- different content, same name and size ------------------------
        self.select_files([twin])
        distinct = wait_for(lambda: len(self.items_named(small.name)) == 3
                            and self.items_named(small.name)[-1]["phase"]
                            in {"accepted", "admitted", "failed"}
                            and self.items_named(small.name)[-1],
                            budget=UI_SYNC_BUDGET,
                            message="same-name different-content reselection never settled")
        detail["distinct"] = {k: distinct[k] for k in ("phase", "sha256", "intakeId", "note")}
        require(distinct["phase"] != "reused",
                "different content was wrongly treated as the same item")
        require(distinct["sha256"] and distinct["sha256"] != reused["sha256"],
                "the two selections produced the same content hash")
        require(distinct["intakeId"] and distinct["intakeId"] != first["intakeId"],
                "the different-content file did not get its own intake")
        rows = [row for row in self.db_intakes() if row["id"] in
                {first["intakeId"], distinct["intakeId"]}]
        detail["intakes_for_name"] = [row["id"] for row in rows]
        require(len(rows) == 2, f"{len(rows)} intakes for two different clips with one name")
        require(len(self.db_intakes()) - intakes_before == 2,
                f"{len(self.db_intakes()) - intakes_before} new intakes for one reused "
                "and one distinct selection")
        detail["screenshot"] = self.shot("UPLOAD_REFRESH_AND_RESELECT")
        return detail

    def case_mixed_identities(self, valid: Path, corrupt: Path, dupe: Path,
                              pair_a: Path, pair_b: Path) -> dict:
        detail = {"files": [valid.name, corrupt.name, dupe.name, pair_a.name, pair_b.name]}
        self.reset_page()
        intakes_before = len(self.db_intakes())
        tasks_before = self.task_count()
        self.select_files([valid, corrupt, dupe, pair_a, pair_b])
        wait_for(lambda: len(self.items()) == 5
                 and all(entry["phase"] in {"admitted", "failed", "reused"}
                         for entry in self.items()),
                 budget=TERMINAL_BUDGET,
                 message="the mixed batch never settled")
        snapshot = self.items()
        detail["items"] = [{k: entry[k] for k in ("name", "phase", "error", "note", "commandId")}
                           for entry in snapshot]
        by_name: dict[str, list[dict]] = {}
        for entry in snapshot:
            by_name.setdefault(entry["name"], []).append(entry)

        good = by_name[valid.name][0]
        require(good["phase"] == "admitted", f"the valid clip ended as {good['phase']!r}")
        bad = by_name[corrupt.name][0]
        require(bad["phase"] == "failed", f"the corrupt clip ended as {bad['phase']!r}")
        require(bad["error"], "the corrupt clip failed without a stable error code")
        duplicate = by_name[dupe.name][0]
        require(duplicate["phase"] == "reused", f"the duplicate ended as {duplicate['phase']!r}")
        require(duplicate["intakeId"] == good["intakeId"],
                "the duplicate points at a different intake")
        twins = by_name[pair_a.name]
        require(len(twins) == 2, f"{len(twins)} items for the same-name pair")
        require(all(entry["phase"] == "admitted" for entry in twins),
                f"same-name different-content items ended as {[e['phase'] for e in twins]}")
        require(twins[0]["intakeId"] != twins[1]["intakeId"],
                "same-name different-content files shared one intake")

        rows = self.db_intakes()
        detail["new_intakes"] = len(rows) - intakes_before
        require(detail["new_intakes"] == 4,
                f"{detail['new_intakes']} intakes for 4 distinct identities")
        detail["new_tasks"] = self.task_count() - tasks_before
        require(detail["new_tasks"] == 3, f"{detail['new_tasks']} tasks for 3 admitted clips")
        corrupt_row = self.intakes_for_name(corrupt.name)[0]
        detail["corrupt_row"] = {"state": corrupt_row["state"],
                                 "error_code": corrupt_row["error_code"],
                                 "task_id": corrupt_row["task_id"]}
        require(corrupt_row["state"] == "failed" and corrupt_row["task_id"] is None,
                "the corrupt upload left a task behind")
        detail["screenshot"] = self.shot("BATCH_MIXED_IDENTITIES")
        return detail

    def case_stop_upload_and_queued(self, slow: Path, small: Path, other: Path) -> dict:
        detail = {"trace": []}
        self.reset_page()
        held: dict = {}

        def hold_the_body(route):
            held["route"] = route

        self.page.route(IMPORT_ROUTE, hold_the_body)
        posts_before = len(self.posts_to("/imports?async=1"))
        self.select_files([slow, small, other])
        wait_for(lambda: len(self.items()) == 3 and self.items()[0]["phase"] == "uploading",
                 budget=UI_SYNC_BUDGET, message="the first upload never started")
        stop = self.page.locator("#stop-import")
        if self.strip_controls:
            self.page.evaluate(
                "() => document.querySelectorAll('#stop-import').forEach(b => b.remove())")
        require(stop.count() == 1, "the stop control is missing")
        require(stop.is_enabled(), "the stop control is disabled while an upload is running")
        detail["screenshot_before"] = self.shot("BATCH_STOP_UPLOAD_AND_QUEUED-before")
        if self.no_effect:
            self.page.evaluate("() => { window.stopBatchUploads = () => {}; }")
        stop.click()
        try:
            held["route"].abort()
        except Exception:  # noqa: BLE001 - the client abort may win the race
            pass
        finally:
            self.page.unroute(IMPORT_ROUTE)
        wait_for(lambda: all(entry["phase"] in {"unknown", "skipped", "accepted", "failed"}
                             for entry in self.items()),
                 budget=UI_SYNC_BUDGET, message="the stopped batch never settled")
        snapshot = self.batch()
        detail["items"] = [{k: entry[k] for k in ("name", "phase", "note")}
                           for entry in snapshot["items"]]
        first, second, third = snapshot["items"]
        require(first["phase"] in {"unknown", "accepted"},
                f"the aborted upload ended as {first['phase']!r}")
        require(second["phase"] == "skipped" and third["phase"] == "skipped",
                f"files after the stop were not skipped ({second['phase']}, {third['phase']})")
        require(snapshot["stopped"] is True, "the batch did not record that it was stopped")
        detail["started_uploads"] = len(self.posts_to("/imports?async=1")) - posts_before
        require(detail["started_uploads"] == 1,
                f"{detail['started_uploads']} uploads started after stop was clicked")
        detail["reconciled_by"] = first["commandId"]
        detail["screenshot_after_stop"] = self.shot("BATCH_STOP_UPLOAD_AND_QUEUED-stopped")

        # --- cancel the batch's queued rows ------------------------------
        self.reset_page()
        self.select_files([slow, small])
        accepted = wait_for(lambda: len(self.items()) == 2
                            and all(entry["phase"] in {"accepted", "admitted"}
                                    for entry in self.items())
                            and self.items(),
                            budget=PREPARE_START_BUDGET,
                            message="the second batch was never accepted")
        intake_ids = [entry["intakeId"] for entry in accepted]
        require(all(intake_ids), "an accepted item has no intake id")
        cancel = self.page.locator("#cancel-batch-queued")
        if self.strip_controls:
            self.page.evaluate(
                "() => document.querySelectorAll('#cancel-batch-queued').forEach(b => b.remove())")
        require(cancel.count() == 1, "the batch cancel control is missing")
        require(cancel.is_enabled(), "the batch cancel control is disabled with queued work")
        with self.page.expect_response(lambda r: CANCEL_ROUTE.search(r.url)) as info:
            cancel.click()
        detail["cancel_response"] = {"status": info.value.status, "url": info.value.url}
        for intake_id in intake_ids:
            wait_for(lambda: (self.db_intake(intake_id) or {}).get("state")
                     in {"cancelled", "succeeded"},
                     budget=TERMINAL_BUDGET,
                     message=f"intake {intake_id} never settled after the batch cancel")
        final = [self.db_intake(intake_id) for intake_id in intake_ids]
        detail["final_rows"] = [{"id": row["id"], "state": row["state"], "task_id": row["task_id"]}
                                for row in final]
        detail["screenshot"] = self.shot("BATCH_STOP_UPLOAD_AND_QUEUED")
        return detail

    def case_stop_preparing_isolation(self, slow: Path, other: Path, foreign_id: str,
                                      editing_task: dict) -> dict:
        detail = {"foreign_intake": foreign_id, "editing_task": editing_task["id"], "trace": []}
        self.reset_page()
        self.open_task(editing_task["id"], editing_task["display_name"])
        self.page.fill("#segment-label", "导入期间保留的中文标签")
        self.page.fill("#segment-text", "这段中文描述在批量导入期间必须原样保留。")
        form_before = {
            "label": self.page.input_value("#segment-label"),
            "text": self.page.input_value("#segment-text"),
            "start": self.page.input_value("#segment-start"),
            "end": self.page.input_value("#segment-end"),
        }
        detail["form_before"] = form_before

        self.select_files([slow, other])
        first = wait_for(lambda: len(self.items()) == 2
                         and all(entry["phase"] in {"accepted", "admitted"}
                                 for entry in self.items())
                         and self.items(),
                         budget=PREPARE_START_BUDGET,
                         message="the batch was never accepted")
        slow_id = first[0]["intakeId"]
        other_id = first[1]["intakeId"]
        preparing = wait_for(lambda: (self.db_intake(slow_id) or {}).get("state") == "preparing"
                             and self.db_intake(slow_id),
                             budget=PREPARE_START_BUDGET,
                             message="the batch never started preparing")
        detail["preparing"] = {k: preparing[k] for k in ("id", "state", "attempt", "generation")}
        child = wait_for(lambda: next((record for record in self.live_children_for(slow_id)
                                       if pid_alive(record["pid"])), None),
                         budget=PREPARE_START_BUDGET,
                         message="no live media child for the preparing intake")
        detail["child"] = {k: child[k] for k in ("pid", "parent_pid", "tool", "created_at")}
        require(child["parent_pid"] == os.getpid(),
                "the media child is not owned by this process")
        require(pid_alive(child["pid"]), "the recorded media child was not alive")
        detail["trace"].append({"child": child["pid"], "at": time.time()})
        detail["screenshot_before"] = self.shot("BATCH_STOP_PREPARING_ISOLATION-before")

        cancel = self.page.locator("#cancel-batch-queued")
        require(cancel.count() == 1, "the batch cancel control is missing")
        require(cancel.is_enabled(), "the batch cancel control is disabled")
        if self.no_effect:
            self.page.evaluate("() => { window.cancelOwnedBatchIntakes = async () => 0; }")
        cancel.click()

        wait_for(lambda: (self.db_intake(slow_id) or {}).get("state") == "cancelled",
                 budget=TERMINAL_BUDGET, message="the preparing intake was never cancelled")
        wait_for(lambda: not pid_alive(child["pid"]), budget=TERMINAL_BUDGET,
                 message="the media child survived the cancel")
        detail["trace"].append({"child_exit": child["pid"], "at": time.time()})
        other_row = self.db_intake(other_id)
        detail["other_row"] = {"id": other_row["id"], "state": other_row["state"],
                               "task_id": other_row["task_id"]}
        foreign = self.db_intake(foreign_id)
        detail["foreign_row"] = {"id": foreign["id"], "state": foreign["state"],
                                 "task_id": foreign["task_id"]}
        require(foreign["state"] != "cancelled",
                "cancelling this batch also cancelled a foreign intake")
        require(foreign["state"] in {"queued", "preparing", "succeeded"},
                f"the seeded foreign intake is in an impossible state {foreign['state']!r}")
        current = self.store.get_task(editing_task["id"])
        detail["editing_after"] = {"revision": current["revision"],
                                   "annotations": len(current["annotations"])}
        require(current["revision"] == editing_task["revision"],
                "cancelling the batch changed the editing task revision")
        form_after = {
            "label": self.page.input_value("#segment-label"),
            "text": self.page.input_value("#segment-text"),
            "start": self.page.input_value("#segment-start"),
            "end": self.page.input_value("#segment-end"),
        }
        detail["form_after"] = form_after
        require(form_after == form_before, "cancelling the batch changed the unsaved form")
        detail["screenshot"] = self.shot("BATCH_STOP_PREPARING_ISOLATION")
        return detail

    def case_draft_survives_import(self, editing_task: dict, clip: Path) -> dict:
        detail = {"editing_task": editing_task["id"]}
        self.reset_page()
        self.open_task(editing_task["id"], editing_task["display_name"])
        # Create real history first, so undo has something to undo later.
        self.page.fill("#segment-label", "导入前已保存片段")
        self.page.fill("#segment-start", "0")
        self.page.fill("#segment-end", "0.5")
        self.page.fill("#segment-text", "这一条在导入前保存，用于验证撤销历史仍在。")
        with self.page.expect_response(
                lambda r: "/commands" in r.url and r.request.method == "POST") as saved:
            self.page.click("#save-segment")
        require(saved.value.status == 200, f"the pre-import save returned {saved.value.status}")
        annotations_before = wait_for(
            lambda: len(self.store.get_task(editing_task["id"])["annotations"]) == 1,
            budget=UI_SYNC_BUDGET, message="the pre-import save never committed")
        annotations_before = 1
        # The HTTP response is not the end of the save: the editor still has to
        # reload the task and record history before undo becomes available.
        wait_for(lambda: self.page.locator("#undo-btn").is_enabled(),
                 budget=UI_SYNC_BUDGET,
                 message="the pre-import save produced no undo entry")
        wait_for(lambda: self.page.text_content("#save-state") not in {"正在保存…", "待确认保存"},
                 budget=UI_SYNC_BUDGET, message="the editor never finished the save")

        # A genuinely unsaved Chinese draft on top of it.
        self.page.fill("#segment-label", "未保存的中文标签：装配步骤三")
        self.page.fill("#segment-text", "这段描述从未保存，导入任何文件都不能清空它。")
        self.page.check("#segment-reviewed")
        self.page.evaluate("() => { document.getElementById('player').currentTime = 1.25; }")
        form_before = {
            "label": self.page.input_value("#segment-label"),
            "text": self.page.input_value("#segment-text"),
            "start": self.page.input_value("#segment-start"),
            "end": self.page.input_value("#segment-end"),
            "reviewed": self.page.is_checked("#segment-reviewed"),
            "selected": self.page.evaluate(
                "() => document.querySelector('#segments tbody tr.selected')?.dataset.annotationId || null"),
            "dirty": self.page.evaluate(
                "() => Boolean(document.getElementById('save-state').classList.contains('dirty'))"),
            "player_time": self.page.evaluate(
                "() => document.getElementById('player').currentTime"),
        }
        detail["form_before"] = form_before
        require(form_before["dirty"] is True, "the unsaved draft was not marked dirty")
        require(form_before["selected"], "no segment was selected before the import")
        detail["screenshot_before"] = self.shot("EDIT_DRAFT_SURVIVES_IMPORT-before")

        self.select_files([clip])
        self.wait_phase(clip.name, {"accepted", "admitted"}, budget=PREPARE_START_BUDGET)
        time.sleep(1.5)  # let the editor's own poller run at least once

        form_after = {
            "label": self.page.input_value("#segment-label"),
            "text": self.page.input_value("#segment-text"),
            "start": self.page.input_value("#segment-start"),
            "end": self.page.input_value("#segment-end"),
            "reviewed": self.page.is_checked("#segment-reviewed"),
            "selected": self.page.evaluate(
                "() => document.querySelector('#segments tbody tr.selected')?.dataset.annotationId || null"),
            "dirty": self.page.evaluate(
                "() => Boolean(document.getElementById('save-state').classList.contains('dirty'))"),
            "player_time": self.page.evaluate(
                "() => document.getElementById('player').currentTime"),
        }
        detail["form_after"] = form_after
        require(form_after["label"] == form_before["label"], "the import changed the label draft")
        require(form_after["text"] == form_before["text"], "the import changed the description draft")
        require(form_after["start"] == form_before["start"], "the import changed the start time")
        require(form_after["end"] == form_before["end"], "the import changed the end time")
        require(form_after["reviewed"] == form_before["reviewed"], "the import reset the review flag")
        require(form_after["selected"] == form_before["selected"], "the import changed the selection")
        require(form_after["dirty"] is True, "the import reported the draft as saved")
        require(abs(form_after["player_time"] - form_before["player_time"]) < 0.75,
                f"the import moved the player from {form_before['player_time']} "
                f"to {form_after['player_time']}")
        current = self.store.get_task(editing_task["id"])
        detail["annotations_after"] = len(current["annotations"])
        require(len(current["annotations"]) == annotations_before,
                "the import changed the task's annotations")

        # The draft must still reach the project draft store.
        drafts = wait_for(lambda: [d for d in self.page.evaluate(
            """async () => (await (await fetch('/api/v1/projects/current/drafts',
                 {headers: {'X-TraceAnnotate-Token': token}})).json()).drafts"""
        ) if d["task_id"] == editing_task["id"]],
            budget=UI_SYNC_BUDGET,
            message="the unsaved draft was never synced to the project")
        detail["drafts"] = [{"draft_id": d["draft_id"], "revision": d["draft_revision"]}
                            for d in drafts]

        # Review invalidation must still work after the import.
        self.page.fill("#segment-label", "未保存的中文标签：装配步骤三（已修改）")
        require(not self.page.is_checked("#segment-reviewed"),
                "editing after the import did not invalidate the review flag")
        detail["review_invalidated_after_edit"] = True

        # And the undo history still applies.
        self.page.click("#undo-btn")
        if self.page.locator("#confirm-dialog").get_attribute("open") is not None:
            self.page.click("#confirm-accept")
        wait_for(lambda: len(self.store.get_task(editing_task["id"])["annotations"]) == 0,
                 budget=UI_SYNC_BUDGET,
                 message="undo no longer applied after the import")
        detail["undo_applied"] = True
        detail["screenshot"] = self.shot("EDIT_DRAFT_SURVIVES_IMPORT")
        return detail

    def case_state_failure(self, editing_task: dict, clip: Path) -> dict:
        detail = {"editing_task": editing_task["id"]}
        self.reset_page()
        self.select_files([clip])
        accepted = self.wait_phase(clip.name, {"accepted", "admitted"},
                                   budget=PREPARE_START_BUDGET)
        wait_for(lambda: (self.db_intake(accepted["intakeId"]) or {}).get("state") == "succeeded",
                 budget=TERMINAL_BUDGET, message="the reference clip was never admitted")
        intakes_before, tasks_before = len(self.db_intakes()), self.task_count()
        self.open_task(editing_task["id"], editing_task["display_name"])
        title_before = self.page.text_content("#media-name")

        # --- snapshot 503 -------------------------------------------------
        posts_before = len(self.posts_to("/imports?async=1"))
        self.page.route(SNAPSHOT_ROUTE, lambda route: route.fulfill(
            status=503, content_type="application/json",
            body=json.dumps({"error": {"code": "UNAVAILABLE", "message": "snapshot unavailable"}})))
        try:
            self.page.click("#refresh-workspace")
            wait_for(lambda: self.page.locator(
                '#intake-list li[data-status-unavailable]').count() == 1,
                budget=UI_SYNC_BUDGET,
                message="a failing snapshot was not reported as an unavailable status")
            detail["snapshot_marker"] = True
            detail["snapshot_message"] = self.page.text_content(
                "#intake-list li[data-status-unavailable]")
            detail["screenshot_snapshot"] = self.shot("STATE_FAILURE_NO_DUPLICATE-snapshot")
            require(self.page.text_content("#media-name") == title_before,
                    "a failing snapshot wiped the open task")
            require(self.page.locator(f'button.task-button[data-task-id="{editing_task["id"]}"]').count() == 1,
                    "a failing snapshot wiped the task list")
        finally:
            self.page.unroute(SNAPSHOT_ROUTE)
        require(len(self.posts_to("/imports?async=1")) == posts_before,
                "a failing snapshot triggered an upload")

        # --- intake list 503 while a Chinese draft is unsaved -------------
        self.page.fill("#segment-label", "状态不可用时保留的草稿")
        self.page.route(INTAKE_LIST_ROUTE, lambda route: route.fulfill(
            status=503, content_type="application/json",
            body=json.dumps({"error": {"code": "UNAVAILABLE", "message": "intakes unavailable"}})))
        try:
            self.page.evaluate("() => refreshIntakes()")
            wait_for(lambda: self.page.locator(
                "#intake-list li[data-status-unavailable]").count() == 1,
                budget=UI_SYNC_BUDGET,
                message="a failing intake list was not reported as unavailable")
            detail["status_unavailable"] = True
            require(self.page.input_value("#segment-label") == "状态不可用时保留的草稿",
                    "a failing intake list wiped the unsaved draft")
        finally:
            self.page.unroute(INTAKE_LIST_ROUTE)

        # --- already admitted, no duplicate -------------------------------
        self.reset_page(keep_batch=True)
        self.page.route(INTAKE_LIST_ROUTE, lambda route: route.fulfill(
            status=503, content_type="application/json",
            body=json.dumps({"error": {"code": "UNAVAILABLE", "message": "intakes unavailable"}})))
        try:
            self.page.reload(wait_until="load")
            self.page.wait_for_selector("#file-input", state="attached")
            time.sleep(2.5)
            detail["reloaded_items"] = [{k: entry[k] for k in ("name", "phase", "commandId")}
                                        for entry in self.items()]
            require(len(self.db_intakes()) == intakes_before,
                    f"{len(self.db_intakes()) - intakes_before} duplicate intakes after reload")
            require(self.task_count() == tasks_before,
                    f"{self.task_count() - tasks_before} duplicate tasks after reload")
            require(len(self.posts_to("/imports?async=1")) == posts_before,
                    "reloading with an unavailable status re-uploaded a file")
        finally:
            self.page.unroute(INTAKE_LIST_ROUTE)
        detail["no_duplicate"] = True
        detail["screenshot"] = self.shot("STATE_FAILURE_NO_DUPLICATE")
        return detail

    # -- runner ----------------------------------------------------------
    def run_case(self, case_id: str, action) -> None:
        started = time.time()
        record = {"case_id": case_id, "status": "PASS", "started_at": started}
        try:
            record["detail"] = action()
        except CaseFailure as exc:
            record["status"] = "FAIL"
            record["failure"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            record["status"] = "FAIL"
            record["failure"] = f"{type(exc).__name__}: {exc}"
        if record["status"] != "PASS":
            try:
                record["batch_at_failure"] = self.batch()
            except Exception:  # noqa: BLE001 - diagnostics are best effort
                pass
            try:
                record["page_state_at_failure"] = self.page.evaluate(
                    """() => ({
                         busy: state.busy, pending: Boolean(state.pending),
                         loading: state.loadingTask, dirty: state.dirty,
                         selected: state.selected,
                         undo: state.task ? getUndoStack(state.task.id).length : -1,
                         redo: state.task ? getRedoStack(state.task.id).length : -1,
                         undoDisabled: document.getElementById('undo-btn').disabled,
                         mediaName: document.getElementById('media-name').textContent
                       })""")
            except Exception:  # noqa: BLE001 - diagnostics are best effort
                pass
        record["finished_at"] = time.time()
        record["duration_seconds"] = round(record["finished_at"] - started, 2)
        self.case_records.append(record)
        print(f"[{record['status']}] {case_id} ({record['duration_seconds']}s)"
              + ("" if record["status"] == "PASS" else f" -> {record['failure']}"),
              flush=True)


def seed_foreign_intake(store, source: Path, *, name: str) -> str:
    """Create an intake that the browser batch never owned."""
    from trace_annotate.workspace.intake import AsyncIntakes
    staged = store.tmp_dir / f"seed-{name}"
    shutil.copy2(source, staged)
    return AsyncIntakes(store).enqueue(staged, display_name=name)["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="P07/P08 real-browser batch upload gate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-wheel", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--headed", dest="headless", action="store_false")
    parser.add_argument("--fault", choices=["missing-control", "no-effect"], default=None)
    parser.add_argument("--slow-media-seconds", type=float, default=180.0)
    parser.add_argument("--slow-media-size", default="640x360")
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
    from trace_annotate.workspace.server import make_server
    from trace_annotate.workspace.slots import get_prepare_slot
    from trace_annotate.workspace.store import WorkspaceStore

    harness = Harness(out, fault=args.fault)
    harness.live = live_media_processes
    store = WorkspaceStore(out / "workspace")
    store.create()
    harness.store = store

    slot = get_prepare_slot(store.root)
    slot.acquire()  # hold the decoder until a case deliberately releases it

    media = out / "media"
    slow = make_video(media / "slow.mp4", seconds=args.slow_media_seconds,
                      pattern="smptebars", size=args.slow_media_size)
    slow2 = make_video(media / "slow2.mp4", seconds=args.slow_media_seconds * 3,
                       pattern="testsrc2", size=args.slow_media_size)
    small = make_video(media / "small.mp4", seconds=1.0, pattern="testsrc")
    other = make_video(media / "other.mp4", seconds=2.0, pattern="rgbtestsrc")
    corrupt = media / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a media container" * 96)
    dupe = media / "dupe.mp4"
    shutil.copy2(small, dupe)
    # Same name as ``small`` but different bytes and the same size: the editor
    # must not treat a same-name reselection as the same item.
    (media / "twin").mkdir()
    twin = same_size_different_bytes(small, media / "twin" / "small.mp4")
    # Same name, same size, different content, both selected at once.
    (media / "pair-a").mkdir()
    (media / "pair-b").mkdir()
    pair_a = make_video(media / "pair-a" / "clip.mp4", seconds=2.0, pattern="smptehdbars")
    pair_b = make_video(media / "pair-b" / "clip.mp4", seconds=3.0, pattern="testsrc2")
    require(pair_a.name == pair_b.name, "the pair fixture must share one display name")
    require(sha256_file(pair_a) != sha256_file(pair_b),
            "the same-name fixture pair must differ in content")
    require(twin.stat().st_size == small.stat().st_size,
            "the reselection twin must keep the same size")
    require(sha256_file(twin) != sha256_file(small),
            "the reselection twin must differ in content")

    foreign_source = make_video(media / "foreign.mp4", seconds=1.0, pattern="testsrc2")
    editing_source = make_video(media / "editing.mp4", seconds=2.0, pattern="testsrc2")
    foreign_id = seed_foreign_intake(store, foreign_source, name="foreign.mp4")
    editing = store.import_source(editing_source, display_name="editing.mp4")
    editing_task = store.get_task(editing["task_id"])
    editing_task["display_name"] = "editing.mp4"

    server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()

    def with_slot_released(action):
        slot.release()
        try:
            return action()
        finally:
            slot.acquire()

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
                 "content_type": req.headers.get("content-type", ""),
                 "at": time.time()}))
            page.on("response", lambda res: harness.responses.append(
                {"url": res.url, "status": res.status, "at": time.time()}))

            page.goto(server.editor_url, wait_until="load")
            page.wait_for_selector("#file-input", state="attached")
            harness.open_panel()

            if args.fault == "missing-control":
                harness.strip_controls = True
                harness.run_case("BATCH_STOP_MISSING_CONTROL_NEGATIVE",
                                 lambda: harness.case_stop_upload_and_queued(slow, small, other))
            elif args.fault == "no-effect":
                harness.no_effect = True
                harness.run_case("BATCH_STOP_NO_EFFECT_NEGATIVE",
                                 lambda: harness.case_stop_upload_and_queued(slow, small, other))
            else:
                harness.run_case("BATCH_ACCEPT_BEFORE_PREPARE_DONE",
                                 lambda: with_slot_released(
                                     lambda: harness.case_accept_before_prepare(slow, small)))
                harness.run_case("UPLOAD_LOST_RESPONSE_RECONCILE",
                                 lambda: harness.case_lost_response(small))
                harness.run_case("UPLOAD_REFRESH_AND_RESELECT",
                                 lambda: harness.case_refresh_and_reselect(small, other, twin))
                harness.run_case("BATCH_MIXED_IDENTITIES",
                                 lambda: with_slot_released(
                                     lambda: harness.case_mixed_identities(
                                         small, corrupt, dupe, pair_a, pair_b)))
                harness.run_case("BATCH_STOP_UPLOAD_AND_QUEUED",
                                 lambda: harness.case_stop_upload_and_queued(slow, small, other))
                harness.run_case("BATCH_STOP_PREPARING_ISOLATION",
                                 lambda: with_slot_released(
                                     lambda: harness.case_stop_preparing_isolation(
                                         slow2, other, foreign_id, editing_task)))
                harness.run_case("EDIT_DRAFT_SURVIVES_IMPORT",
                                 lambda: harness.case_draft_survives_import(editing_task, other))
                harness.run_case("STATE_FAILURE_NO_DUPLICATE",
                                 lambda: with_slot_released(
                                     lambda: harness.case_state_failure(editing_task, other)))

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

    expected = (["BATCH_STOP_MISSING_CONTROL_NEGATIVE"] if args.fault == "missing-control"
                else ["BATCH_STOP_NO_EFFECT_NEGATIVE"] if args.fault == "no-effect"
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
        "document_type": "BROWSER_BATCH_RESULT",
        "result_version": 1,
        "output": str(out),
        "identity": identity,
        "overall": overall,
        "expected_cases": expected,
        "cases": harness.case_records,
        "case_set_complete": complete,
        "all_passed": passed,
        "page_errors": harness.page_errors,
        "workspace": str(out / "workspace"),
        "screenshots": sorted(p.name for p in harness.shots.glob("*.png")),
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"overall: {overall} (cases={len(seen)} expected={len(expected)})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
