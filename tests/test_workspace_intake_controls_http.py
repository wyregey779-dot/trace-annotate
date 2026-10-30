"""R05 HTTP idempotency for intake control commands (cancel/retry receipts)."""
from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from trace_annotate.workspace.server import make_server, TOKEN_HEADER
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")
    out = tmp_path_factory.mktemp("ctl") / "clip.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=10",
                    "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, timeout=30)
    return out


@pytest.fixture()
def server(tmp_path, media):
    store = WorkspaceStore(tmp_path / "ws")
    store.create()
    httpd = make_server(store)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    t.start()
    try:
        yield httpd, store, media
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        httpd.server_close()
        store.close()


def _bad_upload(httpd, path: Path) -> str:
    bad = path / "bad.mp4"
    bad.write_bytes(b"not media" * 16)
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    conn.request("POST", "/api/v1/projects/current/imports?async=1", body=bad.read_bytes(),
                 headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                          "Content-Type": "application/octet-stream",
                          "Content-Length": str(bad.stat().st_size),
                          "X-TraceAnnotate-Filename": "bad.mp4"})
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    return body["id"]


def _post(httpd, path, payload: dict):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    conn.request("POST", path, body=json.dumps(payload),
                 headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                          "Content-Type": "application/json"})
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    return resp.status, body


def _get(httpd, path):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    conn.request("GET", path, headers={TOKEN_HEADER: httpd.token})
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    return resp.status, body


def _wait_failed(httpd, intake_id: str):
    row = None
    for _ in range(40):
        _, row = _get(httpd, f"/api/v1/projects/current/intakes/{intake_id}")
        if row["state"] == "failed":
            return row
        time.sleep(0.25)
    raise AssertionError(f"intake never failed: {row}")


def test_terminal_cancel_stores_receipt_and_replays_same_response(server, tmp_path):
    httpd, store, media = server
    intake_id = _bad_upload(httpd, tmp_path)
    _wait_failed(httpd, intake_id)
    cid = "cmd-cancel-terminal-1"
    s1, r1 = _post(httpd, f"/api/v1/projects/current/intakes/{intake_id}/cancel",
                   {"command_id": cid})
    assert s1 == 200 and r1["state"] == "failed", r1
    # replay the same command -> identical receipt, no new state change
    s2, r2 = _post(httpd, f"/api/v1/projects/current/intakes/{intake_id}/cancel",
                   {"command_id": cid})
    assert s2 == 200 and r2 == r1
    # receipt is persisted and retrievable
    s3, receipt = _get(httpd, f"/api/v1/projects/current/commands/{cid}")
    assert s3 == 200 and receipt["response"]["state"] == "failed"


def test_reused_command_id_for_different_kind_conflicts(server, tmp_path):
    httpd, store, media = server
    intake_id = _bad_upload(httpd, tmp_path)
    _wait_failed(httpd, intake_id)
    cid = "cmd-reuse-1"
    s1, r1 = _post(httpd, f"/api/v1/projects/current/intakes/{intake_id}/cancel",
                   {"command_id": cid})
    assert s1 == 200
    # same command id but retry request -> conflict, not a second action
    s2, r2 = _post(httpd, f"/api/v1/projects/current/intakes/{intake_id}/retry",
                   {"command_id": cid})
    assert s2 >= 400 and r2["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_retry_command_id_replay_returns_queued_receipt(server, tmp_path):
    httpd, store, media = server
    intake_id = _bad_upload(httpd, tmp_path)
    _wait_failed(httpd, intake_id)
    cid = "cmd-retry-1"
    s1, r1 = _post(httpd, f"/api/v1/projects/current/intakes/{intake_id}/retry",
                   {"command_id": cid})
    assert s1 == 200 and r1["state"] == "queued"
    s2, r2 = _post(httpd, f"/api/v1/projects/current/intakes/{intake_id}/retry",
                   {"command_id": cid})
    assert s2 == 200 and r2 == r1
