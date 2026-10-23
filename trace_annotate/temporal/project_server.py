"""Read-only, loopback-only delivery of one prepared annotation project.

This is deliberately not a directory server. A fresh token names five fixed
resources; original source paths and other project files are never served.
Media identity is checked before binding a socket, and file changes observed
after startup invalidate delivery. The server is for a local, trusted desktop,
not a guarantee against a hostile local process racing filesystem operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import ntpath
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

from .formats import _bounded_json_float, _reject_json_constant, _unique_json_object
from .media import DURATION_TOLERANCE_SECONDS, stream_sha256
from .schema import MAX_SAFE_INTEGER, require_valid_document


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ANNOTATION_BYTES = 20 * 1024 * 1024
STREAM_CHUNK_BYTES = 64 * 1024


def _snapshot(info: os.stat_result) -> Tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _is_link(info: os.stat_result) -> bool:
    # Windows junctions and other reparse entries must not bypass the boundary.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _basename(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip())
            and value not in {".", ".."}
            and not ntpath.splitdrive(value)[0]
            and not any(c in value for c in "/\\:")
            and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value))


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_size(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_SAFE_INTEGER


@dataclass(frozen=True)
class _Resource:
    path: Path
    info: Tuple[int, int, int, int]
    ctime_ns: int
    content_type: str


def _resource(root: Path, name: str, content_type: str,
              limit: Optional[int] = None) -> _Resource:
    if not _basename(name):
        raise ValueError("Prepared project resource must be a plain filename")
    path = root / name
    info = path.lstat()
    if _is_link(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Prepared project resource must be a regular, non-linked file: {name}")
    if path.resolve(strict=True).parent != root:
        raise ValueError("Prepared project resource escaped its directory")
    if info.st_size > MAX_SAFE_INTEGER or (limit is not None and info.st_size > limit):
        raise ValueError(f"Prepared project resource exceeds its size limit: {name}")
    return _Resource(path, _snapshot(info), info.st_ctime_ns, content_type)


def _current(resource: _Resource) -> bool:
    info = resource.path.lstat()
    return (not _is_link(info) and stat.S_ISREG(info.st_mode)
            and _snapshot(info) == resource.info and info.st_ctime_ns == resource.ctime_ns
            and resource.path.resolve(strict=True) == resource.path)


def _read_json(resource: _Resource):
    # The stat limit is rechecked around a bounded read, including one extra byte
    # so an active writer cannot turn a small manifest into an unbounded read.
    with resource.path.open("rb") as stream:
        if _snapshot(os.fstat(stream.fileno())) != resource.info or not _current(resource):
            raise ValueError("Prepared project changed before validation")
        raw = stream.read(resource.info[2] + 1)
        if len(raw) != resource.info[2] or not _current(resource):
            raise ValueError("Prepared project changed during validation")
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_json_object,
                          parse_float=_bounded_json_float, parse_constant=_reject_json_constant)
    except (UnicodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"Invalid prepared project JSON: {resource.path.name}") from exc


def _project_resources(directory) -> Tuple[Dict[str, _Resource], dict]:
    requested = Path(directory).absolute()
    info = requested.lstat()
    if _is_link(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("Prepared project must be a non-linked directory")
    root = requested.resolve(strict=True)
    # Reject links in its lexical ancestry too, including Windows junctions.
    if any(_is_link(parent.lstat()) for parent in requested.parents):
        raise ValueError("Prepared project directory must not pass through links")
    project = _resource(root, "project.json", "application/json; charset=utf-8", MAX_MANIFEST_BYTES)
    manifest = _read_json(project)
    if (not isinstance(manifest, dict) or manifest.get("kind") != "traceannotate.prepared-media"
            or type(manifest.get("version")) is not int or manifest["version"] != 1):
        raise ValueError("Unsupported prepared project kind/version")
    video = manifest.get("video")
    if (not isinstance(video, dict) or video.get("file") != "playback.mp4"
            or not _valid_size(video.get("size")) or not _valid_digest(video.get("sha256"))):
        raise ValueError("Prepared project video must bind playback.mp4 size and SHA256")
    duration = video.get("duration")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not 0 < duration <= MAX_SAFE_INTEGER or not math.isfinite(duration)):
        raise ValueError("Prepared project video duration must be finite and positive")
    if manifest.get("annotation_file") != "annotations.json":
        raise ValueError("Prepared project annotation_file must be annotations.json")
    if manifest.get("mapping_file") != "frame-map.jsonl" or not isinstance(manifest.get("timeline"), dict):
        raise ValueError("Prepared project must declare its timeline and frame-map.jsonl")
    source = manifest.get("source")
    if (not isinstance(source, dict) or not _basename(source.get("name"))
            or not _valid_size(source.get("size")) or not _valid_digest(source.get("sha256"))):
        raise ValueError("Prepared project source metadata must bind name, size and SHA256")
    media = _resource(root, "playback.mp4", "video/mp4")
    annotation = _resource(root, "annotations.json", "application/json; charset=utf-8", MAX_ANNOTATION_BYTES)
    # Mapping is checked as an ordinary contained file, but deliberately has no
    # public route. Its semantic validation belongs to media preparation.
    _resource(root, "frame-map.jsonl", "application/x-ndjson")
    if media.info[2] != video["size"] or stream_sha256(media.path) != video["sha256"]:
        raise ValueError("Prepared playback media size/SHA256 mismatch")
    document = _read_json(annotation)
    require_valid_document(document)
    if (document["schema_version"] != 2 or document["clip"] != "playback.mp4"
            or document["media_size"] != video["size"] or document["media_sha256"] != video["sha256"]
            or abs(document["duration"] - duration) > DURATION_TOLERANCE_SECONDS
            or any(segment["end"] > duration for segment in document["segments"])):
        raise ValueError("Prepared annotation must be schema 2 and fully bound to playback.mp4")
    ui_root = Path(__file__).resolve().parent / "ui"
    resources = {
        "project.json": project, "playback.mp4": media, "annotations.json": annotation,
        "editor.html": _resource(ui_root, "annotator.html", "text/html; charset=utf-8"),
        "media-hash.js": _resource(ui_root, "media-hash.js", "application/javascript; charset=utf-8"),
    }
    if not all(_current(item) for item in resources.values()):
        raise ValueError("Prepared project changed during validation")
    return resources, manifest


def _byte_range(value: str, size: int) -> Tuple[int, int]:
    if len(value) > 160:
        raise ValueError("Invalid byte range")
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", value)
    if match is None or not any(match.groups()) or size == 0:
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if first:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            raise ValueError("Unsatisfiable byte range")
    else:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Unsatisfiable byte range")
        start, end = max(0, size - suffix), size - 1
    return start, end


class PreparedProjectServer(ThreadingHTTPServer):
    """Standard HTTPServer lifecycle plus the tokenized ``editor_url``."""

    daemon_threads = True
    allow_reuse_address = False


def make_project_server(directory, host: str = "127.0.0.1", port: int = 0) -> PreparedProjectServer:
    """Validate one project and bind loopback; callers manage its lifecycle.

    ``server.editor_url`` is the only starting URL. Call ``serve_forever()``;
    on interruption call ``server_close()`` (and ``shutdown()`` when serving in
    a separate thread). No browser is opened, files written or remote bound.
    """
    if host != "127.0.0.1":
        raise ValueError("Prepared projects can be served only on 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 through 65535")
    resources, manifest = _project_resources(directory)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            # Paths include the private capability token; do not log them.
            return

        def _headers(self, status: int, length: int, content_type: Optional[str] = None,
                     content_range: Optional[str] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(length))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if content_type:
                self.send_header("Content-Type", content_type)
            if content_range:
                self.send_header("Content-Range", content_range)
            if status in (200, 206):
                self.send_header("Accept-Ranges", "bytes")
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            self._response_started = True
            self.close_connection = True

        def _authorized(self) -> bool:
            authority = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get_all("Host", []) != [authority]:
                self._headers(403, 0)
                return False
            origins = self.headers.get_all("Origin", [])
            if origins and origins != [f"http://{authority}"]:
                self._headers(403, 0)
                return False
            return True

        def _serve(self, head: bool = False) -> None:
            if not self._authorized():
                return
            try:
                parsed = urlsplit(self.path)
            except ValueError:
                self._headers(400, 0)
                return
            prefix = f"/{token}/"
            if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                    or not parsed.path.startswith(prefix)):
                self._headers(404, 0)
                return
            resource = resources.get(parsed.path[len(prefix):])
            if resource is None:
                self._headers(404, 0)
                return
            try:
                if not _current(resource):
                    self._headers(409, 0)
                    return
                with resource.path.open("rb") as stream:
                    if _snapshot(os.fstat(stream.fileno())) != resource.info or not _current(resource):
                        self._headers(409, 0)
                        return
                    size = resource.info[2]
                    start, end, status = 0, size - 1, 200
                    ranges = self.headers.get_all("Range", [])
                    if ranges:
                        try:
                            if len(ranges) != 1:
                                raise ValueError("Multiple Range headers")
                            start, end = _byte_range(ranges[0], size)
                            status = 206
                        except ValueError:
                            self._headers(416, 0, content_range=f"bytes */{size}")
                            return
                    length = max(0, end - start + 1)
                    self._headers(status, length, resource.content_type,
                                  f"bytes {start}-{end}/{size}" if status == 206 else None)
                    if head:
                        return
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(STREAM_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (OSError, ValueError):
                # Includes a cancelled browser Range stream. Close rather than
                # append a second status line after headers were sent.
                if not getattr(self, "_response_started", False):
                    self._headers(409, 0)
                self.close_connection = True

        def do_GET(self):
            self._serve()

        def do_HEAD(self):
            self._serve(head=True)

        def _read_only(self):
            if self._authorized():
                self._headers(405, 0)

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _read_only

    server = PreparedProjectServer((host, port), Handler)
    server.editor_url = f"http://127.0.0.1:{server.server_port}/{token}/editor.html"
    server.project_manifest = manifest
    return server
