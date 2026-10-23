"""Real HTTP coverage for a fixed, read-only prepared-project capability."""
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import threading
from urllib.parse import urlsplit

import pytest

from trace_annotate.temporal.project_server import (
    MAX_ANNOTATION_BYTES, MAX_MANIFEST_BYTES, STREAM_CHUNK_BYTES,
    make_project_server,
)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    # Not decoder evidence: this module only serves already-prepared bytes.
    payload = bytes(range(256)) * 900
    (root / "playback.mp4").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    document = {
        "kind": "temporal", "schema_version": 2, "clip": "playback.mp4",
        "duration": 12.5, "media_size": len(payload), "media_sha256": digest,
        "segments": [{"id": "中文-1", "start": 0, "end": 12.5,
                      "label": "测试", "text": "a\r\nb", "tags": [],
                      "track": "场景", "review_status": "unreviewed"}],
        "metadata": {"original_source": "原片.mov"},
    }
    manifest = {
        "kind": "traceannotate.prepared-media", "version": 1,
        "video": {"file": "playback.mp4", "size": len(payload), "sha256": digest, "duration": 12.5},
        "annotation_file": "annotations.json",
        "source": {"name": "原片.mov", "size": 123456, "sha256": "a" * 64},
        "timeline": {"origin": "first_video_pts", "start": 0},
        "mapping_file": "frame-map.jsonl",
    }
    (root / "project.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (root / "annotations.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    (root / "frame-map.jsonl").write_text('{"frame":0,"source_pts":1.5}\n', encoding="utf-8")
    (root / "private-source.mov").write_bytes(b"must never be served")
    return root, manifest, document, payload


@pytest.fixture
def running(prepared):
    root, manifest, document, payload = prepared
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    server = make_project_server(root)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, root, manifest, document, payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        # Mutation tests opt out; every ordinary workflow must be read-only.
        if not getattr(server, "test_changes_files", False):
            assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def request(server, name="editor.html", *, method="GET", headers=None, raw_path=None):
    parsed = urlsplit(server.editor_url)
    path = raw_path if raw_path is not None else parsed.path.rsplit("/", 1)[0] + "/" + name
    connection = http.client.HTTPConnection("127.0.0.1", parsed.port, timeout=5)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_real_http_serves_only_five_declared_resources_without_mutation(running):
    server, root, manifest, document, payload = running
    parsed = urlsplit(server.editor_url)
    assert parsed.hostname == "127.0.0.1" and parsed.port == server.server_port
    assert len(parsed.path.split("/")[1]) >= 40
    for name, content_type, expected in [
        ("project.json", "application/json", (root / "project.json").read_bytes()),
        ("annotations.json", "application/json", (root / "annotations.json").read_bytes()),
        ("playback.mp4", "video/mp4", payload),
        ("editor.html", "text/html", None), ("media-hash.js", "application/javascript", None),
    ]:
        status, headers, data = request(server, name)
        assert status == 200
        assert headers["Content-Type"].startswith(content_type)
        assert int(headers["Content-Length"]) == len(data)
        assert headers["Cache-Control"] == "no-store"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert "Access-Control-Allow-Origin" not in headers
        if expected is not None:
            assert data == expected
    assert json.loads(request(server, "project.json")[2]) == manifest
    assert json.loads(request(server, "annotations.json")[2]) == document


@pytest.mark.parametrize("byte_range,start,end", [
    ("bytes=0-9", 0, 9), ("bytes=5-", 5, 230399),
    ("bytes=-17", 230383, 230399), ("bytes=230390-999999", 230390, 230399),
    ("bytes=-999999", 0, 230399),
])
def test_real_range_has_exact_206_bytes_and_headers(running, byte_range, start, end):
    server, _, _, _, payload = running
    status, headers, data = request(server, "playback.mp4", headers={"Range": byte_range})
    assert status == 206 and data == payload[start:end + 1]
    assert headers["Content-Range"] == f"bytes {start}-{end}/{len(payload)}"
    assert int(headers["Content-Length"]) == end - start + 1
    assert headers["Accept-Ranges"] == "bytes"


@pytest.mark.parametrize("byte_range", [
    "bytes=230400-", "bytes=8-7", "bytes=-0", "bytes=-", "bytes=0-1,5-6",
    "bytes=foo-bar", "items=0-1", "bytes=0-1 ", "bytes=" + "9" * 200 + "-",
])
def test_real_invalid_range_returns_416_without_video_body(running, byte_range):
    server = running[0]
    status, headers, data = request(server, "playback.mp4", headers={"Range": byte_range})
    assert status == 416 and data == b""
    assert headers["Content-Range"] == "bytes */230400"
    assert headers["Content-Length"] == "0"


def test_head_full_and_range_are_bodyless_with_real_lengths(running):
    server = running[0]
    status, headers, data = request(server, "playback.mp4", method="HEAD")
    assert status == 200 and data == b"" and headers["Content-Length"] == "230400"
    status, headers, data = request(server, "playback.mp4", method="HEAD", headers={"Range": "bytes=3-8"})
    assert status == 206 and data == b"" and headers["Content-Length"] == "6"
    assert headers["Content-Range"] == "bytes 3-8/230400"


@pytest.mark.parametrize("path", [
    "/", "/editor.html", "/wrong-token/playback.mp4", "/TOKEN/", "/TOKEN/../project.json",
    "/TOKEN/%2e%2e/project.json", "/TOKEN/%70roject.json", "/TOKEN/%2fproject.json",
    "/TOKEN/..%5cproject.json", "/TOKEN/private-source.mov", "/TOKEN/frame-map.jsonl",
    "/TOKEN/annotations.json/extra", "/TOKEN/project.json?file=private-source.mov",
    "http://example.org/TOKEN/project.json",
])
def test_token_paths_traversal_and_other_files_cannot_be_served(running, path):
    server = running[0]
    token = urlsplit(server.editor_url).path.split("/")[1]
    status, _, data = request(server, raw_path=path.replace("TOKEN", token),
                              headers={"Host": f"127.0.0.1:{server.server_port}"})
    assert status == 404 and data == b""


@pytest.mark.parametrize("headers", [
    {"Host": "example.org"}, {"Host": "localhost"}, {"Host": "127.0.0.1"},
    {"Origin": "https://example.org"}, {"Origin": "null"}, {"Origin": "http://localhost"},
])
def test_foreign_host_and_origin_are_rejected(running, headers):
    assert request(running[0], "project.json", headers=headers)[0] == 403


def test_same_origin_is_allowed_but_duplicate_host_and_range_are_not(running):
    server = running[0]
    origin = f"http://127.0.0.1:{server.server_port}"
    assert request(server, "project.json", headers={"Origin": origin})[0] == 200
    for header_lines, expected in [
        (["Host: example.org"], 403),
        (["Range: bytes=0-1", "Range: bytes=2-3"], 416),
        ([f"Origin: {origin}", f"Origin: {origin}"], 403),
    ]:
        path = urlsplit(server.editor_url).path.replace("editor.html", "playback.mp4")
        wire = (f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                + "\r\n".join(header_lines) + "\r\nConnection: close\r\n\r\n")
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as stream:
            stream.sendall(wire.encode("ascii"))
            assert stream.recv(1024).startswith(f"HTTP/1.1 {expected} ".encode("ascii"))


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"])
def test_all_mutating_or_extra_methods_are_read_only(running, method):
    status, headers, data = request(running[0], "annotations.json", method=method)
    assert status == 405 and data == b""
    assert headers["Allow"] == "GET, HEAD"


@pytest.mark.parametrize("change", ["replace", "delete", "append"])
def test_changed_media_is_refused_after_startup(running, change):
    server, root, _, _, payload = running
    server.test_changes_files = True
    path = root / "playback.mp4"
    if change == "replace":
        other = root / "replacement.mp4"
        other.write_bytes(payload)
        os.replace(other, path)
    elif change == "delete":
        path.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    assert request(server, "playback.mp4")[0] == 409


@pytest.mark.parametrize("kind", ["video-file", "video-size", "video-sha", "source-name", "source-sha",
                                  "annotation-file", "mapping-file", "timeline", "version", "duration"])
def test_bad_manifest_is_refused_before_binding(prepared, kind):
    root, original, _, _ = prepared
    manifest = copy.deepcopy(original)
    if kind == "video-file": manifest["video"]["file"] = "../playback.mp4"
    elif kind == "video-size": manifest["video"]["size"] += 1
    elif kind == "video-sha": manifest["video"]["sha256"] = "0" * 64
    elif kind == "source-name": manifest["source"]["name"] = "C:\\private.mov"
    elif kind == "source-sha": manifest["source"]["sha256"] = "A" * 64
    elif kind == "annotation-file": manifest["annotation_file"] = "../secret.json"
    elif kind == "mapping-file": manifest["mapping_file"] = "../mapping.jsonl"
    elif kind == "timeline": manifest["timeline"] = []
    elif kind == "version": manifest["version"] = True
    elif kind == "duration": manifest["video"]["duration"] = 10 ** 100
    (root / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        make_project_server(root)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 1), ("clip", "original.mov"), ("media_size", 1),
    ("media_sha256", "f" * 64), ("duration", 13),
])
def test_annotation_must_be_fully_bound_to_proxy(prepared, field, value):
    root, _, original, _ = prepared
    document = copy.deepcopy(original)
    document[field] = value
    if field == "schema_version":
        del document["media_sha256"]
    (root / "annotations.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        make_project_server(root)


@pytest.mark.parametrize("filename,limit", [("project.json", MAX_MANIFEST_BYTES),
                                           ("annotations.json", MAX_ANNOTATION_BYTES)])
def test_oversized_json_is_rejected_before_it_is_loaded(prepared, filename, limit, monkeypatch):
    root = prepared[0]
    with (root / filename).open("wb") as stream:
        stream.truncate(limit + 1)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == root / filename:
            pytest.fail("Oversized JSON was opened")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="size limit"):
        make_project_server(root)


def test_large_video_is_streamed_with_bounded_reads(running, monkeypatch):
    server, root, _, _, payload = running
    original_open = Path.open
    counts = []
    class BoundedReader:
        def __init__(self, raw): self.raw = raw
        def __enter__(self): self.raw.__enter__(); return self
        def __exit__(self, *args): return self.raw.__exit__(*args)
        def fileno(self): return self.raw.fileno()
        def seek(self, *args): return self.raw.seek(*args)
        def read(self, count=-1):
            counts.append(count)
            assert 0 < count <= STREAM_CHUNK_BYTES
            return self.raw.read(count)
    def wrapped(path, *args, **kwargs):
        raw = original_open(path, *args, **kwargs)
        return BoundedReader(raw) if path == root / "playback.mp4" else raw
    monkeypatch.setattr(Path, "open", wrapped)
    assert request(server, "playback.mp4")[2] == payload
    assert len(counts) >= 4


def test_duplicate_manifest_keys_are_rejected(prepared):
    root = prepared[0]
    path = root / "project.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"version": 1', '"version": 1, "version": 1'), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid prepared project JSON"):
        make_project_server(root)


def test_same_size_tail_corruption_fails_full_file_identity(prepared):
    root = prepared[0]
    path = root / "playback.mp4"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="size/SHA256 mismatch"):
        make_project_server(root)


def test_unsafe_integer_video_size_is_rejected_without_hashing(prepared, monkeypatch):
    root, original, _, _ = prepared
    manifest = copy.deepcopy(original)
    manifest["video"]["size"] = 9007199254740992
    (root / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    def refuse_hash(*args, **kwargs):
        pytest.fail("Impossible-size manifest reached media hashing")
    monkeypatch.setattr("trace_annotate.temporal.project_server.stream_sha256", refuse_hash)
    with pytest.raises(ValueError, match="bind playback.mp4"):
        make_project_server(root)


def test_fresh_servers_issue_distinct_tokens(prepared):
    first = make_project_server(prepared[0])
    second = make_project_server(prepared[0])
    try:
        assert urlsplit(first.editor_url).path != urlsplit(second.editor_url).path
    finally:
        first.server_close()
        second.server_close()


@pytest.mark.parametrize("host,port", [("0.0.0.0", 0), ("localhost", 0), ("::1", 0),
                                      ("127.0.0.1", True), ("127.0.0.1", -1), ("127.0.0.1", 65536)])
def test_nonloopback_and_invalid_ports_are_refused_without_binding(prepared, host, port):
    with pytest.raises(ValueError):
        make_project_server(prepared[0], host=host, port=port)


def test_symlink_file_and_project_directory_are_rejected(prepared, tmp_path):
    root = prepared[0]
    target = tmp_path / "outside.mp4"
    target.write_bytes((root / "playback.mp4").read_bytes())
    link = tmp_path / "linked-project"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"OS does not allow symlink creation: {exc}")
    with pytest.raises(ValueError, match="non-linked directory"):
        make_project_server(link)
    (root / "playback.mp4").unlink()
    (root / "playback.mp4").symlink_to(target)
    with pytest.raises(ValueError, match="non-linked file"):
        make_project_server(root)
