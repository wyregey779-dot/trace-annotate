"""P03: the command receipt is the only idempotency boundary.

The invariants under test:

* replaying a control command returns its first response and never re-applies
  the action to work that was accepted later (a cancel receipt must not cancel a
  retry that followed it);
* the same holds for both terminal shapes (``succeeded`` and ``cancelled``) and
  across a real workspace restart;
* one command id may not be reused for another intake, another kind of command
  or another upload;
* concurrent replays of one command produce exactly one action and one receipt;
* a malformed control body (array, null, unknown field, empty or illegal id,
  duplicate JSON key) is refused without touching any row;
* the legacy synchronous import endpoint still works.
"""
from __future__ import annotations

import http.client
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.workspace import media as media_module
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.server import TOKEN_HEADER, make_server
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools

PREFIX = "/api/v1/projects/current"


def _ffmpeg() -> str:
    return require_media_tools()[0]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("receipt-clips") / "clip.mp4"
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=10",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, timeout=300)
    return out


@pytest.fixture()
def broken(tmp_path) -> Path:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not media" * 16)
    return path


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


def _upload(httpd, path: Path, *, command_id=None, name=None):
    raw = path.read_bytes()
    headers = {TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
               "Content-Type": "application/octet-stream",
               "Content-Length": str(len(raw)),
               "X-TraceAnnotate-Filename": name or path.name}
    if command_id:
        headers["X-TraceAnnotate-Command-Id"] = command_id
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=120)
    conn.request("POST", PREFIX + "/imports?async=1", body=raw, headers=headers)
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, payload


def _post(httpd, path, body=None, *, raw=None, content_type="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=60)
    data = raw if raw is not None else json.dumps(body if body is not None else {}).encode()
    conn.request("POST", path, body=data,
                 headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                          "Content-Type": content_type, "Content-Length": str(len(data))})
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


def _wait(predicate, *, budget=120.0, interval=0.05, message="condition not reached"):
    deadline = time.monotonic() + budget
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"{message} within {budget:.0f}s (last={last!r})")


def _row(store, intake_id):
    with store.transaction():
        cursor = store.conn.execute("SELECT * FROM intakes WHERE id=?", (intake_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        names = [column[0] for column in cursor.description]
    return dict(zip(names, row))


def _wait_state(store, intake_id, states, *, message="intake never reached the expected state"):
    return _wait(lambda: (lambda row: row if row and row["state"] in states else None)(
        _row(store, intake_id)), message=message)


def _receipt_rows(store, command_id):
    with store.transaction():
        return store.conn.execute(
            "SELECT COUNT(*) FROM command_receipts WHERE command_id=?", (command_id,)).fetchone()[0]


def _transient_failure(monkeypatch, code="PREPARE_UNAVAILABLE"):
    """Fail the first real preparation, then let later attempts through."""
    original = media_module.prepare_workspace_media
    calls = {"count": 0}

    def wrapper(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise WorkspaceError(code, "decoder temporarily unavailable", retryable=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(media_module, "prepare_workspace_media", wrapper)
    return calls


# ---------------------------------------------------------------------------
# COMMAND_REPLAY_AFTER_RETRY
# ---------------------------------------------------------------------------
def test_a_cancel_receipt_cannot_cancel_a_retry_that_followed_it(server, clip, monkeypatch):
    httpd, store = server
    calls = _transient_failure(monkeypatch)
    status, accepted = _upload(httpd, clip, command_id="upl-retry-source")
    assert status == 202, accepted
    intake_id = accepted["id"]
    failed = _wait_state(store, intake_id, {"failed"}, message="the first attempt never failed")
    assert failed["error_code"] == "PREPARE_UNAVAILABLE"
    first_attempt = failed["attempt"]
    assert calls["count"] == 1

    cancel_id = "cmd-cancel-before-retry"
    status, cancelled = _post(httpd, f"{PREFIX}/intakes/{intake_id}/cancel", {"command_id": cancel_id})
    assert status == 200 and cancelled["state"] == "failed", cancelled

    status, queued = _post(httpd, f"{PREFIX}/intakes/{intake_id}/retry", {"command_id": "cmd-retry-after"})
    assert status == 200 and queued["state"] == "queued", queued

    # The replay of the earlier cancel must return its original response and
    # must not touch the queue entry the retry just created.
    status, replay = _post(httpd, f"{PREFIX}/intakes/{intake_id}/cancel", {"command_id": cancel_id})
    assert status == 200 and replay == cancelled, replay
    current = _row(store, intake_id)
    assert current["state"] not in {"cancelled", "cancel_requested"}, current
    assert _receipt_rows(store, cancel_id) == 1

    # The retry really ran: a second attempt is recorded and it succeeded.
    done = _wait_state(store, intake_id, {"succeeded", "failed"},
                       message="the retried intake never settled")
    assert done["state"] == "succeeded", done
    assert done["attempt"] == first_attempt + 1
    assert done["task_id"]
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    # ... and the cancel receipt still describes the first, failed attempt.
    status, receipt = _get(httpd, f"{PREFIX}/commands/{cancel_id}")
    assert status == 200 and receipt["response"] == cancelled


def test_terminal_receipts_replay_unchanged_for_succeeded_and_cancelled(server, clip):
    httpd, store = server
    status, accepted = _upload(httpd, clip, command_id="upl-terminal")
    assert status == 202, accepted
    succeeded = _wait_state(store, accepted["id"], {"succeeded"})
    status, first = _post(httpd, f"{PREFIX}/intakes/{accepted['id']}/cancel",
                          {"command_id": "cmd-cancel-succeeded"})
    assert status == 200 and first["state"] == "succeeded", first
    status, replay = _post(httpd, f"{PREFIX}/intakes/{accepted['id']}/cancel",
                           {"command_id": "cmd-cancel-succeeded"})
    assert status == 200 and replay == first
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert _row(store, accepted["id"])["task_id"] == succeeded["task_id"]

    # A queued intake that the user cancels keeps that answer on replay.
    status, queued = _upload(httpd, clip, command_id="upl-terminal-queued")
    assert status == 202, queued
    status, cancelled = _post(httpd, f"{PREFIX}/intakes/{queued['id']}/cancel",
                              {"command_id": "cmd-cancel-queued"})
    assert status == 200 and cancelled["state"] in {"cancelled", "cancel_requested"}, cancelled
    settled = _wait_state(store, queued["id"], {"cancelled"})
    status, replay = _post(httpd, f"{PREFIX}/intakes/{queued['id']}/cancel",
                           {"command_id": "cmd-cancel-queued"})
    assert status == 200 and replay == cancelled
    assert _row(store, queued["id"])["state"] == settled["state"] == "cancelled"


# ---------------------------------------------------------------------------
# COMMAND_CONFLICT_AND_RESTART
# ---------------------------------------------------------------------------
def test_one_command_id_cannot_address_another_intake_kind_or_upload(server, clip, broken):
    httpd, store = server
    status, first = _upload(httpd, broken, command_id="upl-identity-a")
    assert status == 202, first
    status, second = _upload(httpd, broken, command_id="upl-identity-b")
    assert status == 202, second
    _wait_state(store, first["id"], {"failed"})
    _wait_state(store, second["id"], {"failed"})

    command_id = "cmd-shared"
    status, ok = _post(httpd, f"{PREFIX}/intakes/{first['id']}/cancel", {"command_id": command_id})
    assert status == 200, ok
    before = _row(store, second["id"])

    status, conflict = _post(httpd, f"{PREFIX}/intakes/{second['id']}/cancel", {"command_id": command_id})
    assert status == 409 and conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT", conflict
    status, conflict = _post(httpd, f"{PREFIX}/intakes/{first['id']}/retry", {"command_id": command_id})
    assert status == 409 and conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT", conflict
    status, conflict = _upload(httpd, clip, command_id=command_id)
    assert status == 409 and conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT", conflict
    assert _row(store, second["id"]) == before, "a refused command changed the other intake"
    assert _receipt_rows(store, command_id) == 1


def test_a_receipt_survives_a_real_restart(server, clip):
    httpd, store = server
    status, accepted = _upload(httpd, clip, command_id="upl-restart")
    assert status == 202, accepted
    _wait_state(store, accepted["id"], {"succeeded"})
    command_id = "cmd-restart-cancel"
    status, original = _post(httpd, f"{PREFIX}/intakes/{accepted['id']}/cancel",
                             {"command_id": command_id})
    assert status == 200, original
    root = store.root

    # Close everything the way an operator would, then reopen the same project.
    httpd.shutdown()
    httpd.server_close()
    store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    second = make_server(reopened)
    thread = threading.Thread(target=second.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        status, receipt = _get(second, f"{PREFIX}/commands/{command_id}")
        assert status == 200, receipt
        assert receipt["response"] == original
        status, replay = _post(second, f"{PREFIX}/intakes/{accepted['id']}/cancel",
                               {"command_id": command_id})
        assert status == 200 and replay == original
        with reopened.transaction():
            assert reopened.conn.execute(
                "SELECT COUNT(*) FROM command_receipts WHERE command_id=?", (command_id,)).fetchone()[0] == 1
    finally:
        second.shutdown()
        thread.join(timeout=5)
        second.server_close()
        reopened.close()


def test_concurrent_replays_of_one_command_apply_it_once(server, clip):
    httpd, store = server
    status, accepted = _upload(httpd, clip, command_id="upl-concurrent")
    assert status == 202, accepted
    command_id = "cmd-concurrent"
    results = []
    barrier = threading.Barrier(6)

    def replay():
        barrier.wait()
        results.append(_post(httpd, f"{PREFIX}/intakes/{accepted['id']}/cancel",
                             {"command_id": command_id}))

    threads = [threading.Thread(target=replay) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert len(results) == 6
    statuses = {status for status, _ in results}
    assert statuses == {200}, results
    bodies = {json.dumps(body, sort_keys=True) for _, body in results}
    assert len(bodies) == 1, bodies
    assert _receipt_rows(store, command_id) == 1
    row = _row(store, accepted["id"])
    assert row["state"] in {"cancelled", "cancel_requested", "succeeded"}, row
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] <= 1


# ---------------------------------------------------------------------------
# HTTP_STRICT_INPUT (control bodies and path ids)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    (b"[]", 422),
    (b"null", 422),
    (b"\"command\"", 422),
    (b'{"command_id": "cmd-ok", "extra": 1}', 422),
    (b'{"command_id": ""}', 422),
    (b'{"command_id": 7}', 422),
    (b'{"command_id": "cmd-a", "command_id": "cmd-b"}', 400),
    (b"", 400),
])
def test_a_malformed_control_body_is_refused_without_touching_the_row(server, clip, raw, expected):
    httpd, store = server
    status, accepted = _upload(httpd, clip, command_id="upl-strict")
    assert status == 202, accepted
    _wait_state(store, accepted["id"], {"succeeded"})
    before = _row(store, accepted["id"])
    with store.transaction():
        receipts = store.conn.execute("SELECT COUNT(*) FROM command_receipts").fetchone()[0]
    status, payload = _post(httpd, f"{PREFIX}/intakes/{accepted['id']}/cancel", raw=raw)
    assert status == expected, payload
    assert payload["error"]["code"] in {"SCHEMA_INVALID", "INVALID_JSON"}, payload
    assert _row(store, accepted["id"]) == before
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM command_receipts").fetchone()[0] == receipts


@pytest.mark.parametrize("bad_id", ["1leading", "..", "with%20space", "x" * 97])
def test_an_illegal_intake_id_in_the_path_is_refused(server, clip, bad_id):
    httpd, store = server
    status, payload = _post(httpd, f"{PREFIX}/intakes/{bad_id}/cancel", {"command_id": "cmd-path"})
    assert status == 422, payload
    assert payload["error"]["code"] == "SCHEMA_INVALID"
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM command_receipts").fetchone()[0] == 0


def test_the_legacy_synchronous_import_endpoint_still_works(server, clip):
    httpd, store = server
    raw = clip.read_bytes()
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=120)
    conn.request("POST", PREFIX + "/imports", body=raw,
                 headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                          "Content-Type": "application/octet-stream",
                          "Content-Length": str(len(raw)),
                          "X-TraceAnnotate-Filename": "legacy.mp4"})
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    conn.close()
    assert response.status == 200, payload
    assert payload["task_id"]
    status, task = _get(httpd, f"{PREFIX}/tasks/{payload['task_id']}")
    assert status == 200 and task["media_id"] == payload["media_id"]
