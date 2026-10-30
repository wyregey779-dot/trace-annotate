"""Batch upload semantics over the real HTTP boundary (P07/P08).

These tests exercise the properties the browser batch engine depends on, using
the real server, real SQLite rows and real ffmpeg preparation:

* a second upload is accepted (202) while the first file is still preparing,
  and the accepted row is immediately visible in the intake list;
* an accepted item is reconciled through its original command id when the
  response was lost, and a replay of that command never creates a second row;
* an upload interrupted before its last byte leaves no intake and no receipt,
  so the same command id can still be used for a clean retry;
* a corrupt file fails on its own without blocking the rest of the batch;
* cancelling one batch's queued work leaves another batch and the editing task
  untouched.
"""
from __future__ import annotations

import http.client
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.workspace.server import make_server, TOKEN_HEADER
from trace_annotate.workspace.store import WorkspaceStore

PREFIX = "/api/v1/projects/current"


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")
    return ffmpeg


def _make_video(path: Path, *, seconds: float, size: str = "320x240", rate: int = 10) -> Path:
    # mpeg4 keeps the encoder's memory footprint small so the fixture still
    # works on a machine that is already running the full regression.
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size={size}:rate={rate}",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=300)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("batch-clips")
    return {
        # A long, cheap clip: preparation really takes several seconds, which is
        # the window the second upload must be accepted inside.
        "slow": _make_video(root / "slow.mp4", seconds=180, size="640x360", rate=10),
        "small": _make_video(root / "small.mp4", seconds=1, size="160x120", rate=10),
        "other": _make_video(root / "other.mp4", seconds=2, size="192x144", rate=10),
    }


@pytest.fixture()
def server(tmp_path):
    store = WorkspaceStore(tmp_path / "ws")
    store.create()
    httpd = make_server(store)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield httpd, store
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()
        store.close()


def _post_upload(httpd, path: Path, *, command_id=None, sha256=None, name=None,
                 timeout=120, declared_length=None):
    raw = path.read_bytes()
    headers = {TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
               "Content-Type": "application/octet-stream",
               "Content-Length": str(declared_length if declared_length is not None else len(raw)),
               "X-TraceAnnotate-Filename": name or path.name}
    if command_id:
        headers["X-TraceAnnotate-Command-Id"] = command_id
    if sha256:
        headers["X-TraceAnnotate-SHA256"] = sha256
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=timeout)
    conn.request("POST", PREFIX + "/imports?async=1", body=raw, headers=headers)
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, payload


def _get(httpd, path):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
    conn.request("GET", PREFIX + path, headers={TOKEN_HEADER: httpd.token})
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, payload


def _post_json(httpd, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
    raw = json.dumps(body or {}).encode()
    conn.request("POST", PREFIX + path, body=raw,
                 headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                          "Content-Type": "application/json", "Content-Length": str(len(raw))})
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, payload


def _wait(predicate, *, budget=60.0, interval=0.1, message="condition not reached"):
    deadline = time.monotonic() + budget
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"{message} within {budget:.0f}s (last={last!r})")


def _intake(store, intake_id):
    with store.transaction():
        row = store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)).fetchone()
        if row is None:
            return None
        names = [column[0] for column in store.conn.execute(
            "SELECT * FROM intakes WHERE id=?", (intake_id,)).description]
    return dict(zip(names, row))


def _task_count(store):
    with store.transaction():
        return int(store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])


def _receipt(store, command_id):
    with store.transaction():
        row = store.conn.execute(
            "SELECT state,response FROM command_receipts WHERE command_id=?", (command_id,)).fetchone()
    return None if row is None else {"state": row[0], "response": json.loads(row[1])}


# ---------------------------------------------------------------------------
# BATCH_ACCEPT_BEFORE_PREPARE_DONE
# ---------------------------------------------------------------------------
def test_second_upload_is_accepted_while_the_first_is_still_preparing(server, clips):
    httpd, store = server
    status, first = _post_upload(httpd, clips["slow"], command_id="upl-first")
    assert status == 202, first
    first_id = first["id"]
    _wait(lambda: (_intake(store, first_id) or {}).get("state") == "preparing",
          budget=120, message="the first intake never started preparing")
    started = time.monotonic()
    status, second = _post_upload(httpd, clips["small"], command_id="upl-second")
    elapsed = time.monotonic() - started
    assert status == 202, second
    assert second["state"] == "queued" and second["task_id"] is None
    # The first file must still be preparing: acceptance cannot be a side effect
    # of the first prepare having already finished.
    assert _intake(store, first_id)["state"] == "preparing"
    assert elapsed < 30.0, f"second upload waited {elapsed:.1f}s for the first prepare"
    # ... and the accepted row is immediately visible, without any prepare.
    status, listing = _get(httpd, "/intakes")
    assert status == 200
    ids = [row["id"] for row in listing["intakes"]]
    assert first_id in ids and second["id"] in ids
    second_row = _intake(store, second["id"])
    assert second_row["state"] == "queued"
    assert second_row["worker_id"] is None and second_row["lease_expires_at"] is None
    # Both eventually commit, each with its own task.
    _wait(lambda: (_intake(store, first_id) or {}).get("state") == "succeeded",
          budget=180, message="the first intake never succeeded")
    _wait(lambda: (_intake(store, second["id"]) or {}).get("state") == "succeeded",
          budget=180, message="the second intake never succeeded")
    assert _intake(store, first_id)["task_id"] != _intake(store, second["id"])["task_id"]
    assert _task_count(store) == 2


# ---------------------------------------------------------------------------
# UPLOAD_LOST_RESPONSE_RECONCILE
# ---------------------------------------------------------------------------
def test_lost_response_is_reconciled_by_the_original_command_id(server, clips):
    httpd, store = server
    command_id = "upl-lost-1"
    status, accepted = _post_upload(httpd, clips["small"], command_id=command_id)
    assert status == 202, accepted
    intake_id = accepted["id"]
    # The client never saw the response: it only has the command id.
    status, receipt = _get(httpd, f"/commands/{command_id}")
    assert status == 200, receipt
    assert receipt["state"] == "committed"
    assert receipt["response"]["id"] == intake_id
    assert receipt["response"]["source_sha256"] == accepted["source_sha256"]
    # A blind re-send under the same id must return the original intake, not a
    # second one, and must not create a second task.
    status, replay = _post_upload(httpd, clips["small"], command_id=command_id)
    assert status == 202, replay
    assert replay["id"] == intake_id
    assert len([r for r in _get(httpd, "/intakes")[1]["intakes"]]) == 1
    _wait(lambda: (_intake(store, intake_id) or {}).get("state") == "succeeded", budget=120)
    assert _task_count(store) == 1


def test_replaying_a_command_id_with_different_bytes_is_refused(server, clips):
    httpd, store = server
    command_id = "upl-conflict-1"
    status, first = _post_upload(httpd, clips["small"], command_id=command_id)
    assert status == 202, first
    status, conflict = _post_upload(httpd, clips["other"], command_id=command_id)
    assert status == 409, conflict
    assert conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(_get(httpd, "/intakes")[1]["intakes"]) == 1


def test_an_unknown_command_id_reports_a_missing_receipt(server):
    httpd, _store = server
    status, payload = _get(httpd, "/commands/upl-never-sent")
    assert status == 404
    assert payload["error"]["code"] == "COMMAND_UNKNOWN"


# ---------------------------------------------------------------------------
# BATCH_STOP_UPLOAD_AND_QUEUED (transport half)
# ---------------------------------------------------------------------------
def test_an_aborted_upload_leaves_no_intake_and_no_receipt(server, clips):
    httpd, store = server
    raw = clips["small"].read_bytes()
    # Announce the whole file but deliver only half of it, then drop the socket.
    sock = socket.create_connection(("127.0.0.1", httpd.server_port), timeout=30)
    head = (
        "POST " + PREFIX + "/imports?async=1 HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{httpd.server_port}\r\n"
        f"{TOKEN_HEADER}: {httpd.token}\r\n"
        f"Origin: {httpd.origin}\r\n"
        "Content-Type: application/octet-stream\r\n"
        f"Content-Length: {len(raw)}\r\n"
        "X-TraceAnnotate-Filename: small.mp4\r\n"
        "X-TraceAnnotate-Command-Id: upl-aborted-1\r\n\r\n"
    ).encode("ascii")
    sock.sendall(head + raw[: len(raw) // 2])
    time.sleep(0.2)
    sock.close()
    _wait(lambda: not any(Path(p).name.startswith("browser-upload-")
                          for p in (store.tmp_dir.iterdir() if store.tmp_dir.exists() else [])),
          budget=30, message="the partial upload temp file was never discarded")
    assert _get(httpd, "/intakes")[1]["intakes"] == []
    assert _get(httpd, "/commands/upl-aborted-1")[0] == 404
    # The same command id is still usable for the complete file.
    status, accepted = _post_upload(httpd, clips["small"], command_id="upl-aborted-1")
    assert status == 202, accepted
    assert len(_get(httpd, "/intakes")[1]["intakes"]) == 1
    _wait(lambda: (_intake(store, accepted["id"]) or {}).get("state") == "succeeded", budget=120)
    assert _task_count(store) == 1


# ---------------------------------------------------------------------------
# BATCH_MIXED_IDENTITIES
# ---------------------------------------------------------------------------
def test_a_corrupt_file_fails_alone_and_does_not_block_the_next_upload(server, clips, tmp_path):
    httpd, store = server
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a container" * 64)
    status, bad = _post_upload(httpd, corrupt, command_id="upl-corrupt")
    assert status == 202, bad
    status, good = _post_upload(httpd, clips["small"], command_id="upl-good")
    assert status == 202, good
    _wait(lambda: (_intake(store, bad["id"]) or {}).get("state") == "failed",
          budget=120, message="the corrupt upload never failed")
    failed = _intake(store, bad["id"])
    assert failed["error_code"] and failed["error_code"].isupper()
    assert failed["task_id"] is None and failed["media_id"] is None
    _wait(lambda: (_intake(store, good["id"]) or {}).get("state") == "succeeded", budget=120)
    assert _task_count(store) == 1


def test_the_same_bytes_twice_reuse_one_media_but_keep_two_tasks(server, clips):
    httpd, store = server
    status, first = _post_upload(httpd, clips["small"], command_id="upl-same-1")
    assert status == 202, first
    status, second = _post_upload(httpd, clips["small"], command_id="upl-same-2")
    assert status == 202, second
    assert second["source_sha256"] == first["source_sha256"]
    _wait(lambda: (_intake(store, first["id"]) or {}).get("state") == "succeeded", budget=120)
    _wait(lambda: (_intake(store, second["id"]) or {}).get("state") == "succeeded", budget=120)
    assert _intake(store, first["id"])["media_id"] == _intake(store, second["id"])["media_id"]
    assert _intake(store, first["id"])["task_id"] != _intake(store, second["id"])["task_id"]
    with store.transaction():
        assert int(store.conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]) == 1
    assert _task_count(store) == 2


# ---------------------------------------------------------------------------
# BATCH_STOP_PREPARING_ISOLATION (server half)
# ---------------------------------------------------------------------------
def test_cancelling_one_batch_leaves_another_batch_and_the_editing_task_alone(server, clips):
    httpd, store = server
    status, editing = _post_upload(httpd, clips["other"], command_id="upl-editing")
    assert status == 202, editing
    _wait(lambda: (_intake(store, editing["id"]) or {}).get("state") == "succeeded", budget=180)
    editing_task = _intake(store, editing["id"])["task_id"]
    with store.transaction():
        before = store.get_task(editing_task)

    # Batch A and batch B, both accepted but not yet prepared.
    status, a = _post_upload(httpd, clips["slow"], command_id="upl-a")
    assert status == 202, a
    status, b = _post_upload(httpd, clips["small"], command_id="upl-b")
    assert status == 202, b

    status, cancelled = _post_json(httpd, f"/intakes/{a['id']}/cancel")
    assert status == 200 and cancelled["state"] in {"cancelled", "cancel_requested"}, cancelled
    _wait(lambda: (_intake(store, a["id"]) or {}).get("state") in {"cancelled"}, budget=180,
          message="the cancelled intake never settled")

    _wait(lambda: (_intake(store, b["id"]) or {}).get("state") == "succeeded", budget=180,
          message="the untouched batch item never succeeded")
    assert _intake(store, a["id"])["task_id"] is None
    assert _intake(store, b["id"])["task_id"]
    with store.transaction():
        after = store.get_task(editing_task)
    assert after == before, "cancelling a batch changed the editing task"
    assert _task_count(store) == 2
