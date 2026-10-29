"""HTTP boundary for async staged admission (T09 routes)."""
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
    out = tmp_path_factory.mktemp("http-intake") / "clip.mp4"
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


def _upload(httpd, path: Path, *, async_flag=True):
    raw = path.read_bytes()
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    conn.request(
        "POST", "/api/v1/projects/current/imports?async=1" if async_flag
        else "/api/v1/projects/current/imports",
        body=raw,
        headers={TOKEN_HEADER: httpd.token, "Origin": httpd.origin,
                 "Content-Type": "application/octet-stream",
                 "Content-Length": str(len(raw)),
                 "X-TraceAnnotate-Filename": "clip.mp4"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def _get(httpd, path):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    conn.request("GET", path, headers={TOKEN_HEADER: httpd.token})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def test_async_import_returns_202_then_background_admits(server):
    httpd, store, media = server
    status, body = _upload(httpd, media)
    assert status == 202, body
    intake_id = body["id"]
    assert body["state"] == "queued" and body["task_id"] is None
    # The drain thread processes it; poll until terminal.
    final = None
    for _ in range(40):
        _, final = _get(httpd, f"/api/v1/projects/current/intakes/{intake_id}")
        if final["state"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.25)
    assert final["state"] == "succeeded", final
    assert final["task_id"] and final["media_id"]
    assert store.get_task(final["task_id"])["revision"] == 0


def test_sync_import_still_returns_task_immediately(server):
    httpd, store, media = server
    status, body = _upload(httpd, media, async_flag=False)
    assert status == 200, body
    assert body["task_id"] and int(body["frames"]) >= 1
