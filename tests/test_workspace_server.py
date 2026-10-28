"""Real loopback HTTP regressions for persistence and request boundaries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import pytest

from trace_annotate.workspace.server import make_server, MAX_COMMAND_BYTES, TOKEN_HEADER, REJECT_DRAIN_MAX_BYTES
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def http_video(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for real HTTP media admission tests")
    output = tmp_path_factory.mktemp("workspace-http-media") / "clip.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi",
                    "-i", "testsrc2=duration=2:size=160x120:rate=10", "-pix_fmt", "yuv420p",
                    str(output)], check=True, capture_output=True, timeout=30)
    return output


@contextmanager
def running(store):
    server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


@pytest.fixture()
def api(tmp_path, http_video):
    store = WorkspaceStore(tmp_path / "workspace")
    store.create()
    store.import_source(http_video)
    try:
        with running(store) as server:
            yield server, store
    finally:
        store.close()


def request(server, method, path, body=None, headers=None, authorized=True):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    prepared = {}
    if authorized:
        prepared[TOKEN_HEADER] = server.token
    if method == "POST":
        prepared["Origin"] = server.origin
        prepared["Content-Type"] = "application/json"
    if headers:
        prepared.update(headers)
    if isinstance(body, dict):
        body = json.dumps(body).encode("utf-8")
    try:
        connection.request(method, path, body=body, headers=prepared)
        response = connection.getresponse()
        data = response.read()
        content = json.loads(data) if response.getheader("Content-Type", "").startswith("application/json") and data else data
        return response.status, dict(response.getheaders()), content
    finally:
        connection.close()


def command(store, command_id="cmd-save"):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    return f"/api/v1/projects/current/tasks/{task['id']}/commands", {
        "command_id": command_id, "expected_revision": task["revision"],
        "expected_sha256": task["content_sha256"], "operation": "add_annotation",
        "payload": {"annotation": {
            "id": "ann-first", "type": "segment", "label": "working", "text": "reviewed",
            "tags": [], "track": "main", "attributes": {}, "evidence_ids": [],
            "provenance": {"origin": "manual", "actor_id": "user", "job_id": None},
            "start": {"seconds": {"n": "0", "d": "1"}},
            "end": {"seconds": {"n": "1", "d": "1"}},
        }},
    }


def assert_error(response, status, code):
    actual, headers, payload = response
    assert actual == status, payload
    assert headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers
    assert payload["error"]["code"] == code
    assert re.fullmatch(r"req-[0-9a-f]+", payload["error"]["request_id"])
    assert isinstance(payload["error"]["retryable"], bool)


def test_http_page_bootstrap_and_snapshot_use_actual_store_thread(api):
    server, store = api
    status, headers, html = request(server, "GET", "/", authorized=False)
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    token = json.loads(re.search(rb"const token=(.*?);", html).group(1))
    assert token == server.token
    assert server.token not in server.editor_url
    assert TOKEN_HEADER.encode() in html
    status, _, snapshot = request(server, "GET", "/api/v1/projects/current/snapshot")
    assert status == 200
    assert snapshot["project_id"] == server.project_id
    assert len(snapshot["tasks"]) == 1
    assert snapshot["tasks"][0]["id"] == store.list_tasks()[0]["task_id"]
    status, _, same_snapshot = request(server, "GET", f"/api/v1/projects/{server.project_id}/snapshot")
    assert status == 200 and same_snapshot == snapshot


def test_http_commit_replay_conflict_and_receipt(api):
    server, store = api
    path, body = command(store)
    first = request(server, "POST", path, body)
    assert first[0] == 200, first[2]
    assert first[2]["revision"] == 1 and first[2]["commit_id"]
    replay = request(server, "POST", path, body)
    assert replay[0] == 200 and replay[2] == first[2]
    receipt = request(server, "GET", "/api/v1/projects/current/commands/cmd-save")
    assert receipt[0] == 200
    assert receipt[2]["state"] == "committed" and receipt[2]["response"] == first[2]
    assert re.fullmatch(r"[0-9a-f]{64}", receipt[2]["request_sha256"])
    different = copy.deepcopy(body)
    different["payload"]["annotation"]["text"] = "changed request"
    assert_error(request(server, "POST", path, different), 409, "IDEMPOTENCY_CONFLICT")
    stale = copy.deepcopy(body)
    stale["command_id"] = "cmd-stale"
    assert_error(request(server, "POST", path, stale), 409, "REVISION_CONFLICT")
    assert_error(request(server, "GET", "/api/v1/projects/current/commands/cmd-stale"), 404, "COMMAND_UNKNOWN")
    task = store.get_task(store.list_tasks()[0]["task_id"])
    assert task["revision"] == 1 and len(task["annotations"]) == 1


def test_http_save_survives_server_and_store_restart(tmp_path, http_video):
    root = tmp_path / "persistent"
    store = WorkspaceStore(root)
    store.create()
    store.import_source(http_video)
    try:
        with running(store) as server:
            path, body = command(store)
            result = request(server, "POST", path, body)
            assert result[0] == 200
    finally:
        store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        with running(reopened) as server:
            status, _, receipt = request(server, "GET", "/api/v1/projects/current/commands/cmd-save")
            assert status == 200 and receipt["response"] == result[2]
            status, _, snapshot = request(server, "GET", "/api/v1/projects/current/snapshot")
            assert status == 200 and snapshot["tasks"][0]["revision"] == 1
    finally:
        reopened.close()


def test_http_concurrent_writes_have_one_winner(api):
    server, store = api
    path, first = command(store, "cmd-a")
    second = copy.deepcopy(first)
    second["command_id"] = "cmd-b"
    second["payload"]["annotation"]["id"] = "ann-second"
    barrier = threading.Barrier(2)

    def save(body):
        barrier.wait(timeout=5)
        return request(server, "POST", path, body)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, [first, second]))
    assert sorted(result[0] for result in results) == [200, 409]
    loser = next(result for result in results if result[0] == 409)
    assert_error(loser, 409, "REVISION_CONFLICT")
    task = store.get_task(store.list_tasks()[0]["task_id"])
    assert task["revision"] == 1 and len(task["annotations"]) == 1


def test_http_discarded_success_response_recovers_through_receipt(api):
    server, store = api
    path, body = command(store, "cmd-lost-response")
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("POST", path, body=json.dumps(body).encode("utf-8"), headers={
            TOKEN_HEADER: server.token, "Origin": server.origin, "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        # The client deliberately discards the command result, as when a tab
        # closes between HTTP headers and consuming the JSON response body.
        response.close()
    finally:
        connection.close()
    status, _, receipt = request(server, "GET", "/api/v1/projects/current/commands/cmd-lost-response")
    assert status == 200 and receipt["state"] == "committed"
    retry = request(server, "POST", path, body)
    assert retry[0] == 200 and retry[2] == receipt["response"]
    task = store.get_task(store.list_tasks()[0]["task_id"])
    assert task["revision"] == 1 and len(task["annotations"]) == 1


@pytest.mark.parametrize("headers", [
    {"Host": "attacker.example"}, {"Origin": "null"},
    {"Origin": "https://attacker.example"}, {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
])
def test_bootstrap_rejects_foreign_host_and_origin(api, headers):
    server, _ = api
    response = request(server, "GET", "/", headers=headers, authorized=False)
    assert_error(response, 403, "FORBIDDEN_ORIGIN")
    assert server.token not in json.dumps(response[2])


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_bootstrap_accepts_top_level_external_navigation(api, site):
    server, _ = api
    # Matches the observed Chrome extension navigation: exact Host, no Origin,
    # document/navigate/cross-site, and an explicit user navigation marker.
    headers = {"Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
               "Sec-Fetch-Site": site, "Sec-Fetch-User": "?1"}
    status, response_headers, html = request(server, "GET", "/", headers=headers, authorized=False)
    assert status == 200
    assert json.loads(re.search(rb"const token=(.*?);", html).group(1)) == server.token
    assert response_headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response_headers["Content-Security-Policy"]
    assert response_headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert "Access-Control-Allow-Origin" not in response_headers
    # Loading the page then using its token through a same-origin fetch works.
    status, _, snapshot = request(server, "GET", "/api/v1/projects/current/snapshot",
                                   headers={"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "same-origin",
                                            "Sec-Fetch-Dest": "empty"})
    assert status == 200 and snapshot["project_id"] == server.project_id


@pytest.mark.parametrize("destination", ["iframe", "frame", "object", "embed", "image", "empty"])
@pytest.mark.parametrize("site", ["cross-site", "same-origin"])
def test_bootstrap_rejects_embedded_and_non_document_requests(api, destination, site):
    server, _ = api
    headers = {"Sec-Fetch-Dest": destination, "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": site}
    response = request(server, "GET", "/", headers=headers, authorized=False)
    assert_error(response, 403, "FORBIDDEN_ORIGIN")
    assert server.token not in json.dumps(response[2])


@pytest.mark.parametrize("change", [{"Host": "attacker.example"}, {"Origin": "https://attacker.example"},
                                    {"Origin": "null"}, {"Sec-Fetch-Mode": "cors"},
                                    {"Sec-Fetch-Mode": "no-cors"}])
def test_navigation_exception_keeps_host_origin_and_mode_checks(api, change):
    server, _ = api
    headers = {"Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "cross-site"}
    headers.update(change)
    assert_error(request(server, "GET", "/", headers=headers, authorized=False), 403, "FORBIDDEN_ORIGIN")


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_navigation_headers_never_bypass_api_read_or_write_guard(api, site):
    server, store = api
    headers = {"Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
               "Sec-Fetch-Site": site, "Sec-Fetch-User": "?1"}
    assert_error(request(server, "GET", "/api/v1/projects/current/snapshot", headers=headers),
                 403, "FORBIDDEN_ORIGIN")
    path, body = command(store)
    # Even an otherwise valid token and matching Origin cannot override the
    # cross-site signal on a write endpoint.
    assert_error(request(server, "POST", path, body, headers=headers), 403, "FORBIDDEN_ORIGIN")
    assert store.get_task(store.list_tasks()[0]["task_id"])["revision"] == 0


@pytest.mark.parametrize("headers,code", [
    ({"Sec-Fetch-Site": "cross-site"}, "FORBIDDEN_ORIGIN"),
    ({"Origin": "https://attacker.example"}, "FORBIDDEN_ORIGIN"),
    ({TOKEN_HEADER: "incorrect"}, "INVALID_TOKEN"),
])
def test_repeated_complete_rejected_posts_deliver_json_without_reset(api, headers, code):
    server, store = api
    path, body = command(store)
    # Repeated real sockets cover the previous Windows close-with-unread-body
    # race. A transport abort is a failure, never an acceptable rejection.
    for _ in range(32):
        assert_error(request(server, "POST", path, body, headers=headers), 403, code)
    assert store.get_task(store.list_tasks()[0]["task_id"])["revision"] == 0


def test_rejected_post_with_headers_and_body_sent_separately_delivers_json(api):
    server, store = api
    path, body = command(store)
    raw = json.dumps(body).encode("utf-8")
    for _ in range(8):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.putrequest("POST", path)
            connection.putheader("Origin", "https://attacker.example")
            connection.putheader("Content-Length", str(len(raw)))
            connection.putheader("Content-Type", "application/json")
            connection.endheaders()
            connection.send(raw[:1])
            time.sleep(0.02)
            connection.send(raw[1:])
            response = connection.getresponse()
            data = json.loads(response.read())
            assert response.status == 403 and data["error"]["code"] == "FORBIDDEN_ORIGIN"
        finally:
            connection.close()


def test_rejected_partial_body_has_bounded_wait_and_still_returns_json(api):
    server, store = api
    path, _ = command(store)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Origin", "https://attacker.example")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "512")
        connection.endheaders(b"{")
        started = time.monotonic()
        response = connection.getresponse()
        data = json.loads(response.read())
        assert response.status == 403 and data["error"]["code"] == "FORBIDDEN_ORIGIN"
        assert time.monotonic() - started < 2
    finally:
        connection.close()
    # This connection's short drain timeout must not affect later requests.
    status, _, snapshot = request(server, "GET", "/api/v1/projects/current/snapshot")
    assert status == 200 and snapshot["tasks"][0]["revision"] == 0


@pytest.mark.parametrize("length", [REJECT_DRAIN_MAX_BYTES + 1, MAX_COMMAND_BYTES + 1, 9999999999])
def test_rejected_large_declared_body_does_not_wait_for_upload(api, length):
    server, store = api
    path, _ = command(store)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Origin", "https://attacker.example")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(length))
        connection.endheaders()
        response = connection.getresponse()
        data = json.loads(response.read())
        assert response.status == 403 and data["error"]["code"] == "FORBIDDEN_ORIGIN"
    finally:
        connection.close()
    assert store.get_task(store.list_tasks()[0]["task_id"])["revision"] == 0
    assert_error(request(server, "GET", "/api/v1/projects/current/commands/cmd-save"), 404, "COMMAND_UNKNOWN")


@pytest.mark.parametrize("authorized,status", [(True, 200), (False, 403)])
def test_connection_abort_after_headers_never_starts_a_second_response(api, monkeypatch, authorized, status):
    server, _ = api
    handler = server.RequestHandlerClass
    original_headers = handler._headers
    sent_statuses = []
    finished = threading.Event()
    original_finish = server.finish_request

    def abort_after_headers(instance, code, length, content_type):
        original_headers(instance, code, length, content_type)
        sent_statuses.append(code)
        raise ConnectionAbortedError("Injected Windows transport abort after response headers")

    def finish_request(*args):
        try:
            return original_finish(*args)
        finally:
            finished.set()

    monkeypatch.setattr(handler, "_headers", abort_after_headers)
    monkeypatch.setattr(server, "finish_request", finish_request)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/v1/projects/current/snapshot",
                           headers={TOKEN_HEADER: server.token} if authorized else {})
        response = connection.getresponse()
        assert response.status == status
        # This deliberately injected transport failure occurs after headers;
        # a second JSON/HTTP error cannot repair the incomplete first response.
        with pytest.raises(http.client.IncompleteRead):
            response.read()
        assert finished.wait(timeout=2)
        assert sent_statuses == [status]
    finally:
        connection.close()


def test_api_requires_private_header_and_writes_require_origin(api):
    server, store = api
    path, body = command(store)
    assert_error(request(server, "GET", "/api/v1/projects/current/snapshot", authorized=False), 403, "INVALID_TOKEN")
    assert_error(request(server, "POST", path, body, authorized=False), 403, "INVALID_TOKEN")
    assert_error(request(server, "POST", path, body, headers={TOKEN_HEADER: "wrong"}), 403, "INVALID_TOKEN")
    assert_error(request(server, "POST", path, body, headers={"Origin": ""}), 403, "FORBIDDEN_ORIGIN")
    # No legacy Authorization fallback may accidentally bypass the dedicated header.
    assert_error(request(server, "GET", "/api/v1/projects/current/snapshot", authorized=False,
                         headers={"Authorization": "Bearer " + server.token}), 403, "INVALID_TOKEN")


@pytest.mark.parametrize("path", [
    "/other/snapshot", "/api/v1/projects/wrong/snapshot", "/api/v1/projects/current/snapshot/",
    "/api/v1/projects/current/snapshot?token=bad", "/api/v1/projects/current/../snapshot",
    "/workspace.sqlite3", "/objects/anything",
])
def test_only_exact_scoped_routes_are_served(api, path):
    server, _ = api
    assert_error(request(server, "GET", path), 404, "NOT_FOUND")


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}',
    b'{"a":9007199254740992}', b'{"a":"\\ud800"}', b'{"a":"\xff"}', b'{',
    b'[' * 80 + b'0' + b']' * 80,
])
def test_invalid_json_rejected_without_write(api, raw):
    server, store = api
    path, _ = command(store)
    assert_error(request(server, "POST", path, raw), 400, "INVALID_JSON")
    assert store.get_task(store.list_tasks()[0]["task_id"])["revision"] == 0


@pytest.mark.parametrize("change", [
    {"unexpected": 1}, {"expected_revision": False}, {"expected_revision": -1},
    {"command_id": "bad\n"}, {"expected_sha256": "no"}, {"payload": []},
    {"operation": ["add_annotation"]}, {"payload": {"annotation": []}},
])
def test_invalid_command_shapes_rejected_without_write(api, change):
    server, store = api
    path, body = command(store)
    body.update(change)
    assert_error(request(server, "POST", path, body), 422, "SCHEMA_INVALID")
    assert store.get_task(store.list_tasks()[0]["task_id"])["revision"] == 0


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT", "BREW"])
def test_unsupported_methods_are_typed_and_no_cors(api, method):
    server, _ = api
    assert_error(request(server, method, "/api/v1/projects/current/snapshot"), 405, "METHOD_NOT_ALLOWED")


def test_body_limits_are_checked_before_reading(api):
    server, store = api
    path, _ = command(store)
    # Send only headers: the server must reject a huge declared body immediately.
    assert_error(request(server, "POST", path, b"", headers={"Content-Length": str(MAX_COMMAND_BYTES + 1)}),
                 413, "LIMIT_EXCEEDED")
    for headers in ({"Content-Length": "-1"}, {"Content-Type": "text/plain"},
                    {"Transfer-Encoding": "chunked"}, {"Content-Encoding": "gzip"}):
        assert_error(request(server, "POST", path, b"{}", headers=headers), 400, "INVALID_JSON")


@pytest.mark.parametrize("header", ["Host", "Origin", TOKEN_HEADER, "Content-Length", "Content-Type"])
def test_duplicate_security_and_framing_headers_rejected(api, header):
    server, store = api
    path, _ = command(store)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    values = {"Host": server.authority, "Origin": server.origin, TOKEN_HEADER: server.token,
              "Content-Length": "2", "Content-Type": "application/json"}
    try:
        connection.putrequest("POST", path, skip_host=True)
        for name, value in values.items():
            connection.putheader(name, value)
        connection.putheader(header, values[header])
        connection.endheaders(b"{}")
        response = connection.getresponse()
        data = json.loads(response.read())
        if header in ("Host", "Origin"):
            assert response.status == 403 and data["error"]["code"] == "FORBIDDEN_ORIGIN"
        elif header == TOKEN_HEADER:
            assert response.status == 403 and data["error"]["code"] == "INVALID_TOKEN"
        else:
            assert response.status == 400 and data["error"]["code"] == "INVALID_JSON"
    finally:
        connection.close()


def test_server_instances_do_not_share_credentials_or_project(api, tmp_path):
    first, _ = api
    other = WorkspaceStore(tmp_path / "other")
    other.create()
    try:
        with running(other) as second:
            assert first.token != second.token
            assert first.project_id != second.project_id
            assert_error(request(second, "GET", "/api/v1/projects/current/snapshot",
                                 headers={TOKEN_HEADER: first.token}), 403, "INVALID_TOKEN")
            assert request(first, "GET", "/api/v1/projects/current/snapshot")[2]["project_id"] == first.project_id
            assert request(second, "GET", "/api/v1/projects/current/snapshot")[2]["project_id"] == second.project_id
    finally:
        other.close()


def test_internal_errors_do_not_expose_paths_or_tokens(api, monkeypatch):
    server, store = api

    def fail():
        raise RuntimeError(str(store.root) + " " + server.token)

    monkeypatch.setattr(store, "get_snapshot", fail)
    result = request(server, "GET", "/api/v1/projects/current/snapshot")
    assert_error(result, 500, "INTERNAL_ERROR")
    assert result[2]["error"]["details"]["commit_status"] == "unknown"
    assert str(store.root) not in json.dumps(result[2])
    assert server.token not in json.dumps(result[2])


@pytest.mark.parametrize("host,port", [("0.0.0.0", 0), ("localhost", 0), ("127.0.0.1", True),
                                      ("127.0.0.1", -1), ("127.0.0.1", 65536)])
def test_bind_is_loopback_only(api, host, port):
    _, store = api
    with pytest.raises(ValueError):
        make_server(store, host, port)


def batch_body(store):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    return {
        "command_id": "cmd-batch", "operation": "synthetic-http-candidate", "processor_version": "test-v1",
        "parameters": {"controlled_test": True}, "items": [{"task_id": task["id"],
            "expected_revision": task["revision"], "expected_sha256": task["content_sha256"]}],
    }


def test_http_batch_pause_resume_cancel_and_frozen_receipt(api):
    server, store = api
    body = batch_body(store)
    prefix = "/api/v1/projects/current"
    status, _, created = request(server, "POST", prefix + "/batches", body)
    assert status == 200, created
    bid, job_id = created["id"], created["jobs"][0]["id"]
    assert created["jobs"][0]["state"] == "queued"
    status, _, listing = request(server, "GET", prefix + "/batches")
    assert status == 200 and listing["batches"] == [created]
    status, _, intakes = request(server, "GET", prefix + "/intakes")
    assert status == 200 and len(intakes["intakes"]) == 1
    control = prefix + f"/batches/{bid}/control"
    status, _, pause = request(server, "POST", control, {"command_id": "cmd-pause", "action": "pause_dispatch"})
    assert status == 200 and pause["action"] == "pause_dispatch"
    assert server.queue.claim("http-test-worker") is None
    status, _, current = request(server, "GET", prefix + f"/batches/{bid}")
    assert status == 200 and current["dispatch_paused"] is True
    status, _, replay = request(server, "POST", prefix + "/batches", body)
    assert status == 200 and replay == created
    status, _, receipt = request(server, "GET", prefix + "/commands/cmd-batch")
    assert status == 200 and receipt["response"] == created
    status, _, _ = request(server, "POST", control, {"command_id": "cmd-resume", "action": "resume_dispatch"})
    assert status == 200
    claimed = server.queue.claim("http-test-worker")
    assert claimed["id"] == job_id
    status, _, cancelled = request(server, "POST", prefix + f"/jobs/{job_id}/control",
                                   {"command_id": "cmd-cancel", "action": "cancel"})
    assert status == 200 and cancelled["state"] == "cancel_requested"
    server.queue.acknowledge_cancel(job_id, claimed["lease_generation"], "http-test-worker")
    status, _, job = request(server, "GET", prefix + f"/jobs/{job_id}")
    assert status == 200 and job["state"] == "cancelled" and job["commit"] is None
    # Repeating the accepted request returns its receipt, not a new mutation.
    status, _, replay = request(server, "POST", prefix + f"/jobs/{job_id}/control",
                                {"command_id": "cmd-cancel", "action": "cancel"})
    assert status == 200 and replay == cancelled
    assert store.get_task(body["items"][0]["task_id"])["revision"] == 0


def test_http_batch_stale_input_and_unknown_fields_do_not_enqueue(api):
    server, store = api
    body = batch_body(store)
    body["items"][0]["expected_revision"] = 9
    path = "/api/v1/projects/current/batches"
    assert_error(request(server, "POST", path, body), 409, "REVISION_CONFLICT")
    body = batch_body(store)
    body["source_path"] = "must-not-read-any-path"
    assert_error(request(server, "POST", path, body), 422, "SCHEMA_INVALID")
    body = batch_body(store)
    body["items"][0]["extra"] = True
    assert_error(request(server, "POST", path, body), 422, "SCHEMA_INVALID")
    assert server.queue.list_batches() == []


def test_http_retry_failed_preserves_attempt_history(api):
    server, store = api
    prefix = "/api/v1/projects/current"
    status, _, batch = request(server, "POST", prefix + "/batches", batch_body(store))
    assert status == 200
    claimed = server.queue.claim("worker-first")
    server.queue.fail(claimed["id"], claimed["lease_generation"], "worker-first")
    status, _, retried = request(server, "POST", prefix + f"/jobs/{claimed['id']}/control",
                                {"command_id": "cmd-retry", "action": "retry_failed"})
    assert status == 200 and retried["state"] == "queued"
    next_attempt = server.queue.claim("worker-next")
    assert next_attempt["attempt"] == 2 and next_attempt["id"] == claimed["id"]
    server.queue.finish(claimed["id"], next_attempt["lease_generation"], "worker-next", {"synthetic": True})
    status, _, job = request(server, "GET", prefix + f"/jobs/{claimed['id']}")
    assert status == 200 and job["state"] == "succeeded"
    assert [attempt["state"] for attempt in job["attempts"]] == ["failed", "succeeded"]
    assert job["commit"]["disposition"] == "candidate"
