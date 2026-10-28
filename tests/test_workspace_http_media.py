"""Real sockets, uploads and decoded images for the workspace media routes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import socket
import threading
import time
from urllib.parse import quote

import pytest

from tests.test_workspace_server import api, assert_error, command, http_video, request  # shared real-server fixtures
from trace_annotate.workspace.http_media import FILENAME_HEADER, resolve_media_object
from trace_annotate.workspace.server import TOKEN_HEADER, make_server
from trace_annotate.workspace.store import WorkspaceStore

PREFIX = "/api/v1/projects/current"


def media_id(store):
    return store.list_tasks()[0]["media_id"]


def cookie(server):
    return {"Cookie": f"{server.media_cookie_name}={server.media_cookie}"}


def upload(server, data, name="browser.mp4", **kwargs):
    return request(server, "POST", PREFIX + "/imports", data, headers={
        "Content-Type": "application/octet-stream", FILENAME_HEADER: quote(name, safe=""),
        **kwargs,
    })


def test_bootstrap_sets_media_only_strict_cookie_and_playback_range(api):
    server, store = api
    status, headers, _ = request(server, "GET", "/", authorized=False)
    assert status == 200
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    assert "/media/" in headers["Set-Cookie"]
    assert server.token not in headers["Set-Cookie"]
    ident = media_id(store)
    path, digest, size = resolve_media_object(store, ident, "playback_path")
    expected = path.read_bytes()
    url = PREFIX + f"/media/{ident}/playback"
    status, headers, actual = request(server, "GET", url, authorized=False, headers=cookie(server))
    assert status == 200 and actual == expected
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["X-TraceAnnotate-Media-SHA256"] == digest
    for value, start, stop in [("bytes=0-23", 0, 24), ("bytes=20-", 20, size), ("bytes=-32", size - 32, size)]:
        status, headers, actual = request(server, "GET", url, headers={"Range": value})
        assert status == 206 and actual == expected[start:stop]
        assert headers["Content-Range"] == f"bytes {start}-{stop - 1}/{size}"
    status, headers, actual = request(server, "HEAD", url)
    assert status == 200 and int(headers["Content-Length"]) == size and actual == b""


@pytest.mark.parametrize("value", ["bytes=999999999-", "bytes=4-2", "bytes=-0", "bytes=0-1,4-5", "bits=0-10"])
def test_invalid_range_returns_416_not_full_video(api, value):
    server, store = api
    status, headers, data = request(server, "GET", PREFIX + f"/media/{media_id(store)}/playback",
                                    headers={"Range": value})
    assert status == 416 and data == b""
    assert headers["Content-Range"].startswith("bytes */")


def test_media_cookie_never_authorizes_api_or_writes(api):
    server, store = api
    assert_error(request(server, "GET", PREFIX + "/snapshot", authorized=False, headers=cookie(server)),
                 403, "INVALID_TOKEN")
    path, body = command(store)
    assert_error(request(server, "POST", path, body, authorized=False, headers=cookie(server)),
                 403, "INVALID_TOKEN")
    playback = PREFIX + f"/media/{media_id(store)}/playback"
    assert_error(request(server, "GET", playback, authorized=False), 403, "INVALID_TOKEN")
    assert_error(request(server, "GET", playback, authorized=False,
                         headers={"Cookie": f"{server.media_cookie_name}=wrong"}), 403, "INVALID_TOKEN")
    for extra in ({"Origin": "https://attacker.invalid"}, {"Sec-Fetch-Site": "cross-site"},
                  {"Sec-Fetch-Site": "same-site"}, {"Host": "attacker.invalid"}):
        assert_error(request(server, "GET", playback, authorized=False, headers={**cookie(server), **extra}),
                     403, "FORBIDDEN_ORIGIN")


def test_duplicate_media_cookie_or_bad_token_cannot_fallback(api):
    server, store = api
    path = PREFIX + f"/media/{media_id(store)}/playback"
    twice = cookie(server)["Cookie"] + "; " + cookie(server)["Cookie"]
    assert_error(request(server, "GET", path, authorized=False, headers={"Cookie": twice}), 403, "INVALID_TOKEN")
    assert_error(request(server, "GET", path, headers={**cookie(server), TOKEN_HEADER: "wrong"}),
                 403, "INVALID_TOKEN")


def test_frame_page_and_png_bind_actual_decoded_frame(api):
    server, store = api
    ident = media_id(store)
    path = PREFIX + f"/media/{ident}/frames"
    status, _, page = request(server, "GET", path + "?offset=3&limit=2")
    assert status == 200 and page["offset"] == "3" and page["total"] == "20"
    assert [row["frame_index"] for row in page["frames"]] == ["3", "4"]
    assert page["frames"][0]["seconds"] == {"n": "3", "d": "10"}
    assert page["frame_map_sha256"]
    images = []
    for row in page["frames"]:
        status, headers, data = request(server, "GET", path + f"/{row['frame_index']}/image",
                                        authorized=False, headers=cookie(server))
        assert status == 200 and data.startswith(b"\x89PNG\r\n\x1a\n")
        assert headers["X-TraceAnnotate-Image-SHA256"] == hashlib.sha256(data).hexdigest()
        assert headers["X-TraceAnnotate-Frame-Index"] == row["frame_index"]
        assert headers["X-TraceAnnotate-Source-PTS"] == row["source_pts"]
        assert headers["X-TraceAnnotate-Proxy-PTS"] == row["proxy_pts"]
        images.append(data)
    assert images[0] != images[1]
    status, _, tail = request(server, "GET", path + "?offset=20&limit=1")
    assert status == 200 and tail["frames"] == []
    assert_error(request(server, "GET", path + "/20/image"), 404, "FRAME_NOT_FOUND")


def test_image_delivery_uses_pinned_bytes_after_cache_file_is_evicted(api, monkeypatch):
    server, store = api
    original = server.image_service.image

    def evict_after_pin(*args):
        image = original(*args)
        image["path"].unlink()
        return image

    monkeypatch.setattr(server.image_service, "image", evict_after_pin)
    status, headers, data = request(server, "GET", PREFIX + f"/media/{media_id(store)}/frames/2/image")
    assert status == 200 and data.startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(data).hexdigest() == headers["X-TraceAnnotate-Image-SHA256"]


@pytest.mark.parametrize("query", ["offset=-1", "offset=01", "offset=1.0", "offset=1&offset=2", "limit=0",
                                   "limit=1001", "limit=", "limit=true", "unknown=1", "offset=%FF"])
def test_frame_query_rejects_ambiguous_or_unbounded_pagination(api, query):
    server, store = api
    assert_error(request(server, "GET", PREFIX + f"/media/{media_id(store)}/frames?{query}"),
                 400, "INVALID_QUERY")


def test_playback_identity_failure_refuses_changed_bytes(api):
    server, store = api
    ident = media_id(store)
    path, _, _ = resolve_media_object(store, ident, "playback_path")
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))
    try:
        assert_error(request(server, "GET", PREFIX + f"/media/{ident}/playback"), 409, "MEDIA_CHANGED")
    finally:
        path.write_bytes(original)


def test_real_upload_retains_name_and_creates_separate_task_for_same_media(api, http_video):
    server, store = api
    data = http_video.read_bytes()
    before = store.get_snapshot()
    status, _, first = upload(server, data, "测试 🎬.mp4")
    assert status == 200 and first["reused_existing"] is True
    assert first["task_id"] != before["tasks"][0]["id"]
    assert first["media_id"] == before["media"][0]["id"]
    status, _, second = upload(server, data, "same-content.mp4")
    assert status == 200 and second["task_id"] != first["task_id"]
    assert len(store.get_snapshot()["media"]) == 1 and len(store.list_tasks()) == 3
    intakes = store.list_intakes()
    assert intakes[-2]["display_name"] == "测试 🎬.mp4"
    assert intakes[-1]["state"] == "succeeded"
    assert not list(store.tmp_dir.glob("browser-upload-*"))
    assert http_video.read_bytes() == data


def test_invalid_complete_upload_is_persisted_and_next_file_can_succeed(api, http_video):
    server, store = api
    assert_error(upload(server, b"this is not a video", "broken.mp4"), 422, "UNSUPPORTED_MEDIA")
    intake = store.list_intakes()[-1]
    assert intake["state"] == "failed" and intake["source_sha256"] == hashlib.sha256(b"this is not a video").hexdigest()
    assert intake["task_id"] is None
    assert upload(server, http_video.read_bytes())[0] == 200
    assert not list(store.tmp_dir.glob("browser-upload-*"))


@pytest.mark.parametrize("name", ["../outside.mp4", "C:\\outside.mp4", ".", "..", "", "a\x00.mp4", "a\n.mp4", "a" * 256])
def test_upload_filename_is_metadata_not_a_path(api, name):
    server, store = api
    before = len(store.list_intakes())
    assert_error(upload(server, b"abc", name), 400, "INVALID_UPLOAD")
    assert len(store.list_intakes()) == before
    assert not list(store.tmp_dir.glob("browser-upload-*"))


def test_upload_requires_header_origin_and_explicit_size_limit(api):
    server, store = api
    server.max_upload_bytes = 3
    assert_error(upload(server, b"1234"), 413, "LIMIT_EXCEEDED")
    assert_error(upload(server, b"abc", **{"Origin": "null"}), 403, "FORBIDDEN_ORIGIN")
    assert_error(upload(server, b"abc", **{TOKEN_HEADER: "bad"}), 403, "INVALID_TOKEN")
    assert_error(upload(server, b"abc", **{"Content-Type": "multipart/form-data"}), 400, "INVALID_UPLOAD")
    assert_error(upload(server, b"abc", **{FILENAME_HEADER: "%GG"}), 400, "INVALID_UPLOAD")
    assert request(server, "GET", PREFIX + "/imports")[2] == {"busy": False, "max_upload_bytes": "3"}
    assert len(store.list_tasks()) == 1


def test_interrupted_upload_never_admits_task_and_releases_import_slot(api):
    server, store = api
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.putrequest("POST", PREFIX + "/imports")
        for key, value in {"Origin": server.origin, TOKEN_HEADER: server.token,
                           "Content-Type": "application/octet-stream", FILENAME_HEADER: "partial.mp4",
                           "Content-Length": "100"}.items():
            connection.putheader(key, value)
        connection.endheaders(b"incomplete")
        connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 400 and body["error"]["code"] == "UPLOAD_INCOMPLETE"
    finally:
        connection.close()
    assert len(store.list_tasks()) == 1
    assert not server.import_lock.locked()
    assert not list(store.tmp_dir.glob("browser-upload-*"))


def test_concurrent_import_is_bounded_and_snapshot_remains_readable(api, http_video, monkeypatch):
    server, store = api
    entered, resume = threading.Event(), threading.Event()
    original = store.import_source

    def hold(*args, **kwargs):
        entered.set()
        assert resume.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "import_source", hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(upload, server, http_video.read_bytes())
        assert entered.wait(timeout=5)
        try:
            assert request(server, "GET", PREFIX + "/imports")[2]["busy"] is True
            assert request(server, "GET", PREFIX + "/snapshot")[0] == 200
            assert_error(upload(server, b"abc"), 409, "IMPORT_BUSY")
        finally:
            resume.set()
        assert pending.result(timeout=10)[0] == 200
    assert not server.import_lock.locked()


def test_snapshot_export_contains_saved_revision_and_no_source_paths(api):
    server, store = api
    path, body = command(store)
    assert request(server, "POST", path, body)[0] == 200
    status, headers, exported = request(server, "GET", PREFIX + "/export")
    assert status == 200
    assert exported.pop("intakes") == store.list_intakes()
    assert exported.pop("export_scope") == "development_snapshot_without_media_or_portable_import"
    assert exported == store.get_snapshot()
    assert headers["Content-Disposition"].startswith("attachment; filename=\"TraceAnnotate-")
    assert exported["tasks"][0]["revision"] == 1
    assert str(store.root) not in json.dumps(exported)
    assert_error(request(server, "GET", PREFIX + "/export", authorized=False, headers=cookie(server)),
                 403, "INVALID_TOKEN")


def test_server_close_cancels_preparation_before_store_is_closed(tmp_path, http_video, monkeypatch):
    from trace_annotate.workspace import media
    from trace_annotate.workspace.errors import WorkspaceError

    store = WorkspaceStore(tmp_path / "closing-workspace")
    store.create()
    server = make_server(store)
    serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    entered, cancelled_seen = threading.Event(), threading.Event()

    def cancellable_preparation(source, output_dir, *, cancelled=None):
        assert callable(cancelled)
        entered.set()
        deadline = time.monotonic() + 5
        while not cancelled():
            if time.monotonic() > deadline:
                raise AssertionError("Shutdown never reached the preparation callback")
            time.sleep(0.01)
        cancelled_seen.set()
        raise WorkspaceError("CANCELLED", "Controlled preparation stopped")

    monkeypatch.setattr(media, "prepare_workspace_media", cancellable_preparation)
    serving.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(upload, server, http_video.read_bytes())
            assert entered.wait(timeout=5)
            server.shutdown()
            serving.join(timeout=5)
            started = time.monotonic()
            server.server_close()
            assert time.monotonic() - started < 3
            assert cancelled_seen.is_set() and not server.import_lock.locked()
            assert future.result(timeout=5)[2]["error"]["code"] == "CANCELLED"
            assert store.list_tasks() == []
            assert store.list_intakes()[-1]["state"] == "cancelled"
    finally:
        if serving.is_alive():
            server.shutdown()
            serving.join(timeout=5)
        server.server_close()
        store.close()
