"""P02: a successful admission has one durable, replayable completion record.

The invariants under test:

* the 202 receipt of an accepted upload is immutable: it keeps describing the
  queued state even after the intake succeeded;
* the completion record is derived from the intake (``adm`` namespace) and is
  the same record after a restart, so a repeated confirmation cannot invent a
  second task;
* the completion namespace is disjoint from user command receipts, and the
  database itself refuses a second completion for one intake.
"""
from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.server import TOKEN_HEADER, make_server
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools

PREFIX = "/api/v1/projects/current"


def _ffmpeg() -> str:
    return require_media_tools()[0]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("p02-done") / "clip.mp4"
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=10",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, timeout=300)
    return out


@pytest.fixture()
def workspace(tmp_path):
    store = WorkspaceStore(tmp_path / "ws")
    store.create()
    yield store
    try:
        store.close()
    except Exception:
        pass


@pytest.fixture()
def server(workspace):
    httpd = make_server(workspace)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield httpd, workspace
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _upload(httpd, path: Path, *, command_id: str):
    raw = path.read_bytes()
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=120)
    conn.request("POST", PREFIX + "/imports?async=1", body=raw,
                 headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                          "Content-Type": "application/octet-stream",
                          "Content-Length": str(len(raw)),
                          "X-TraceAnnotate-Filename": path.name,
                          "X-TraceAnnotate-Command-Id": command_id})
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, payload


def _get(httpd, path):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
    conn.request("GET", path, headers={TOKEN_HEADER: httpd.token})
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, payload


def _wait_state(store, intake_id, state, *, budget=120.0):
    import time
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        with store.transaction():
            row = store.conn.execute("SELECT state FROM intakes WHERE id=?", (intake_id,)).fetchone()
        if row and row[0] == state:
            return
        time.sleep(0.05)
    raise AssertionError(f"intake never reached {state}")


def test_the_enqueue_receipt_stays_the_queued_answer(server, clip):
    httpd, store = server
    status, accepted = _upload(httpd, clip, command_id="upl-immutable")
    assert status == 202, accepted
    intake_id = accepted["id"]
    _wait_state(store, intake_id, "succeeded")

    status, receipt = _get(httpd, f"{PREFIX}/commands/upl-immutable")
    assert status == 200, receipt
    assert receipt["state"] == "committed"
    assert receipt["response"] == accepted, "the accepted receipt changed after completion"
    assert receipt["response"]["state"] == "queued"
    assert receipt["response"]["task_id"] is None
    assert receipt["response"]["completion"] is None

    status, row = _get(httpd, f"{PREFIX}/intakes/{intake_id}")
    assert status == 200 and row["state"] == "succeeded"
    assert row["completion"]["task_id"] == row["task_id"]


def test_the_completion_identity_is_derived_and_stable(workspace, clip):
    intakes = AsyncIntakes(workspace)
    upload = workspace.tmp_dir / "stable.mp4"
    shutil.copy2(clip, upload)
    accepted = intakes.enqueue(upload, display_name="stable.mp4")
    intakes.run_pending()
    row = intakes.get(accepted["id"])
    assert row["state"] == "succeeded"
    record = row["completion"]
    assert record["id"] == S.make_id("adm", f"{row['id']}:{row['generation']}"), record
    assert record["intake_id"] == row["id"]
    assert record["task_id"] == row["task_id"] and record["media_id"] == row["media_id"]
    assert record["source_sha256"] == row["source_sha256"]
    assert record["source_size"] == str(row["source_size"])
    task = workspace.get_task(row["task_id"])
    assert record["task_content_sha256"] == task["content_sha256"]
    assert workspace.get_completion(row["id"]) == record
    assert workspace.get_completion(row["id"]) == record


def test_a_restart_returns_the_same_completion(workspace, clip):
    intakes = AsyncIntakes(workspace)
    upload = workspace.tmp_dir / "restart.mp4"
    shutil.copy2(clip, upload)
    accepted = intakes.enqueue(upload, display_name="restart.mp4")
    intakes.run_pending()
    record = intakes.get(accepted["id"])["completion"]
    root = workspace.root
    workspace.close()

    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        assert reopened.get_completion(accepted["id"]) == record
        again = AsyncIntakes(reopened)
        assert again.get(accepted["id"])["completion"] == record
        # Confirming the same success again is a no-op, not a second task.
        assert again.finish(accepted["id"], record["generation"], "any-worker",
                            {"media_id": record["media_id"], "task_id": record["task_id"]})["task_id"] \
            == record["task_id"]
        with reopened.transaction():
            assert reopened.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
            assert reopened.conn.execute("SELECT COUNT(*) FROM admission_completions").fetchone()[0] == 1
    finally:
        reopened.close()


def test_the_completion_namespace_is_separate_from_user_commands(server, clip):
    httpd, store = server
    status, accepted = _upload(httpd, clip, command_id="upl-namespace")
    assert status == 202, accepted
    _wait_state(store, accepted["id"], "succeeded")
    record = store.get_completion(accepted["id"])
    assert record is not None and record["id"].startswith("adm")
    # No command receipt exists under the completion id.
    status, payload = _get(httpd, f"{PREFIX}/commands/{record['id']}")
    assert status == 404 and payload["error"]["code"] == "COMMAND_UNKNOWN", payload
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM command_receipts WHERE command_id=?",
                                  (record["id"],)).fetchone()[0] == 0
    # A user command may pick that literal id; it lands in its own namespace.
    task = store.get_task(record["task_id"])
    store.apply_task_command(
        command_id=record["id"], task_id=task["id"], expected_revision=0,
        expected_sha256=task["content_sha256"], operation="add_annotation",
        payload={"annotation": {
            "id": "ann-ns", "type": "segment", "label": "working", "text": "ns",
            "tags": [], "track": "main", "attributes": {},
            "provenance": {"origin": "manual", "actor_id": "user", "job_id": None},
            "evidence_ids": [],
            "start": {"seconds": S.rational(0.0)}, "end": {"seconds": S.rational(1.0)},
        }}, request_body=b"{}")
    assert store.get_completion(accepted["id"]) == record
    status, receipt = _get(httpd, f"{PREFIX}/commands/{record['id']}")
    assert status == 200 and receipt["response"]["revision"] == 1


def test_the_database_refuses_a_second_completion_for_one_intake(workspace, clip):
    intakes = AsyncIntakes(workspace)
    upload = workspace.tmp_dir / "unique.mp4"
    shutil.copy2(clip, upload)
    accepted = intakes.enqueue(upload, display_name="unique.mp4")
    intakes.run_pending()
    record = workspace.get_completion(accepted["id"])
    assert record is not None
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        with workspace.transaction() as conn:
            conn.execute(
                "INSERT INTO admission_completions(id,intake_id,generation,task_id,media_id,"
                "source_sha256,source_size,task_content_sha256,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                ("adm-second", record["intake_id"], record["generation"], record["task_id"],
                 record["media_id"], record["source_sha256"], record["source_size"],
                 record["task_content_sha256"], record["created_at"]))
    with workspace.transaction():
        assert workspace.conn.execute(
            "SELECT COUNT(*) FROM admission_completions WHERE intake_id=?",
            (record["intake_id"],)).fetchone()[0] == 1
