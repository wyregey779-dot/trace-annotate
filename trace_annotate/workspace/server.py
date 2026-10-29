"""Loopback HTTP access to one opened, revisioned workspace.

The editor serves only registered media and accepts uploads as bytes, never
filesystem paths. Write requests require an exact origin and private header.
"""
from __future__ import annotations
import json
import math
import os
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .errors import BadRequest, Conflict, LimitExceeded, NotFound, SchemaInvalid, WorkspaceError
from .queue import DurableQueue
from .store import WorkspaceStore
from .http_media import (MAX_UPLOAD_BYTES, UPLOAD_TIMEOUT_SECONDS, STREAM_CHUNK_BYTES,
                         _identity, receive_and_import, resolve_media_object)
from ..temporal.project_server import _byte_range
from .store import _safe
from .reporting import export_task, task_report
from .classic import classic_resource, CLASSIC_DOCUMENTS, HASH_SCRIPT

MAX_COMMAND_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 10
REJECT_DRAIN_MAX_BYTES = 64 * 1024
REJECT_DRAIN_TIMEOUT_SECONDS = 0.25
MAX_SAFE_INTEGER = 9007199254740991
TOKEN_HEADER = "X-TraceAnnotate-Token"
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,95}")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")

from .ui import EDITOR_HTML


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def _validate_json_tree(value, depth=0):
    if depth > 64:
        raise ValueError("JSON nesting limit exceeded")
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("Unpaired Unicode surrogate")
    elif type(value) in (int, float):
        if not math.isfinite(value) or abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("JSON number outside the supported range")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_json_tree(key, depth + 1)
            _validate_json_tree(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth + 1)


def _require_id(value, field):
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise SchemaInvalid("SCHEMA_INVALID", f"Invalid {field}")


def _command(body):
    fields = {"command_id", "expected_revision", "expected_sha256", "operation", "payload"}
    if not isinstance(body, dict) or set(body) != fields:
        raise SchemaInvalid("SCHEMA_INVALID", "Command fields are missing or unknown")
    _require_id(body["command_id"], "command_id")
    if type(body["expected_revision"]) is not int or not 0 <= body["expected_revision"] <= MAX_SAFE_INTEGER:
        raise SchemaInvalid("SCHEMA_INVALID", "expected_revision must be a nonnegative integer")
    if not isinstance(body["expected_sha256"], str) or HASH_PATTERN.fullmatch(body["expected_sha256"]) is None:
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid expected_sha256")
    payload = body["payload"]
    if not isinstance(body["operation"], str) or not isinstance(payload, dict):
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid operation or payload")
    required = {
        "add_annotation": {"annotation"},
        "update_annotation": {"annotation_id", "changes"},
        "delete_annotation": {"annotation_id"},
        "replace_annotations": {"annotations"},
        "revert_command": {"target_command_id"},
        "review_annotation": {"annotation_id", "reviewed"},
        "import_temporal": {"annotations", "legacy_document_metadata", "legacy_document_envelope"},
    }.get(body["operation"])
    if required is None:
        raise SchemaInvalid("UNSUPPORTED_OPERATION", "Operation is not implemented in this slice")
    if set(payload) != required:
        raise SchemaInvalid("SCHEMA_INVALID", "Operation payload fields are missing or unknown")
    if "annotation_id" in payload:
        _require_id(payload["annotation_id"], "annotation_id")
    for field in ("annotation", "changes"):
        if field in payload and not isinstance(payload[field], dict):
            raise SchemaInvalid("SCHEMA_INVALID", f"{field} must be an object")
    if "annotations" in payload and not isinstance(payload["annotations"], list):
        raise SchemaInvalid("SCHEMA_INVALID", "annotations must be a list")
    return body


def _fields(body, expected):
    if not isinstance(body, dict) or set(body) != expected:
        raise SchemaInvalid("SCHEMA_INVALID", "Request fields are missing or unknown")
    _require_id(body["command_id"], "command_id")
    return body


class WorkspaceHTTPServer(ThreadingHTTPServer):
    """Caller-managed HTTP lifecycle; token and store belong to this instance."""
    # Closing must wait for admitted writes and frame deliveries before the
    # caller closes SQLite or removes the decoder's owned cache images.
    daemon_threads = False
    allow_reuse_address = False

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(REQUEST_TIMEOUT_SECONDS)
        return connection, address

    def server_close(self):
        if hasattr(self, "closing"):
            self.closing.set()
        super().server_close()
        if hasattr(self, "image_service"):
            self.image_service.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        # Neither capability values nor private paths belong in access logs.
        pass

    def send_error(self, code, message=None, explain=None):
        # BaseHTTPRequestHandler otherwise emits an HTML error containing the
        # submitted method. Keep malformed/unknown HTTP requests typed as well.
        if code == 501:
            try:
                self._authorize()
            except WorkspaceError as exc:
                self._error(exc, "req-" + secrets.token_hex(12))
                return
        exc = WorkspaceError("METHOD_NOT_ALLOWED" if code == 501 else "BAD_REQUEST", "Request rejected")
        exc.http_status = 405 if code == 501 else code
        self._error(exc, "req-" + secrets.token_hex(12))

    def _headers(self, status, length, content_type):
        self._response_started = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        classic = getattr(self, "_classic_document", False)
        self.send_header("Content-Security-Policy", "default-src 'none'; connect-src 'self'; "
                         f"script-src 'nonce-{self.server.nonce}'; style-src 'unsafe-inline'; "
                         + ("media-src 'self' blob:; worker-src blob:; " if classic else "media-src 'self'; ")
                         + "img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        for name, value in getattr(self, "_extra_headers", []):
            self.send_header(name, value)
        if status == 405:
            self.send_header("Allow", "GET, HEAD, POST")
        self.end_headers()
        self.close_connection = True

    def _send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, len(data), "application/json; charset=utf-8")
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, exc, request_id):
        if getattr(self, "_response_started", False):
            self.close_connection = True
            return
        self._discard_rejected_body()
        # Store errors can contain local paths. Expose stable codes and only
        # public conflict metadata, never an exception or submitted input.
        messages = {
            "FORBIDDEN_ORIGIN": "Request origin or host is not allowed",
            "INVALID_TOKEN": "Workspace token is missing or invalid",
            "INVALID_JSON": "Request must contain valid bounded JSON",
            "SCHEMA_INVALID": "Request does not match the command schema",
            "LIMIT_EXCEEDED": "Request exceeds the configured size limit",
            "COMMAND_UNKNOWN": "No durable receipt exists for this command",
            "REVISION_CONFLICT": "Task revision changed; reload before retrying",
            "IDEMPOTENCY_CONFLICT": "Command ID was already used for another request",
            "UNSUPPORTED_OPERATION": "Operation is not supported by this server",
            "METHOD_NOT_ALLOWED": "HTTP method is not supported",
            "INTERNAL_ERROR": "Request failed; query its command receipt before retrying",
        }
        body = exc.body(request_id)
        body["error"]["message"] = messages.get(exc.code, "Workspace request was rejected")
        body["error"]["details"] = {
            key: value for key, value in exc.details.items()
            if key in {"expected_revision", "current_revision", "expected_sha256", "current_sha256"}
            and type(value) in (int, str)
        }
        status = "unknown" if exc.http_status >= 500 or exc.code == "COMMAND_UNKNOWN" else "not_committed"
        body["error"]["details"]["commit_status"] = "committed" if exc.code == "ALREADY_COMMITTED" else status
        try:
            self._send_json(exc.http_status, body)
        except (ConnectionError, socket.timeout):
            # A peer that has already left cannot receive an error. In
            # particular, never append another HTTP response to a partial one.
            self.close_connection = True

    def _discard_rejected_body(self):
        """Consume a small, well-framed rejected body before closing the socket.

        Closing a Windows socket with unread incoming data can reset it before
        the peer receives the JSON error. Draining has a byte cap and one total
        deadline; it must neither wait for an arbitrary upload nor read a body
        already consumed (or partially consumed by a timed-out normal read).
        """
        if getattr(self, "_body_read_started", False) or not hasattr(self, "headers"):
            return
        if self.headers.get_all("Transfer-Encoding", []):
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None:
            return
        remaining = int(lengths[0])
        if not 0 < remaining <= min(REJECT_DRAIN_MAX_BYTES, self.server.max_command_bytes):
            return
        self._body_read_started = True
        previous_timeout = self.connection.gettimeout()
        deadline = time.monotonic() + REJECT_DRAIN_TIMEOUT_SECONDS
        try:
            while remaining:
                available_time = deadline - time.monotonic()
                if available_time <= 0:
                    break
                self.connection.settimeout(available_time)
                chunk = self.rfile.read1(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            # The rejection stands even when the peer abandons its upload.
            pass
        finally:
            try:
                self.connection.settimeout(previous_timeout)
            except OSError:
                pass

    def _deny(self, code, status=403):
        exc = WorkspaceError(code, "Request rejected")
        exc.http_status = status
        raise exc

    def _authorize(self, write=False, bootstrap=False):
        if self.headers.get_all("Host", []) != [self.server.authority]:
            self._deny("FORBIDDEN_ORIGIN")
        origins = self.headers.get_all("Origin", [])
        if (origins and origins != [self.server.origin]) or (write and not origins):
            self._deny("FORBIDDEN_ORIGIN")
        sites = self.headers.get_all("Sec-Fetch-Site", [])
        destinations = self.headers.get_all("Sec-Fetch-Dest", [])
        modes = self.headers.get_all("Sec-Fetch-Mode", [])
        if bootstrap and destinations and destinations != ["document"]:
            self._deny("FORBIDDEN_ORIGIN")
        # Opening a local app from a browser extension or external link is a
        # cross-site top-level navigation, not a cross-origin API request. Only
        # the GET / document may bootstrap this way; Host and Origin above still
        # apply. CSP/X-Frame-Options separately prevent embedding the page.
        document_navigation = (
            bootstrap and not write and self.command == "GET"
            and modes == ["navigate"] and destinations == ["document"]
            and sites in (["cross-site"], ["same-site"])
        )
        if sites and sites not in (["same-origin"], ["none"]) and not document_navigation:
            self._deny("FORBIDDEN_ORIGIN")

    def _token(self):
        tokens = self.headers.get_all(TOKEN_HEADER, [])
        if (len(tokens) != 1 or not tokens[0].isascii()
                or not secrets.compare_digest(tokens[0], self.server.token)):
            self._deny("INVALID_TOKEN")

    def _media_token(self):
        """The media-only cookie cannot authorize API reads or any write."""
        if self.headers.get_all(TOKEN_HEADER, []):
            self._token()
            return
        values = self.headers.get_all("Cookie", [])
        if len(values) != 1:
            self._deny("INVALID_TOKEN")
        matches = []
        for part in values[0].split(";"):
            key, separator, value = part.strip().partition("=")
            if separator and key == self.server.media_cookie_name:
                matches.append(value)
        if (len(matches) != 1 or not matches[0].isascii()
                or not secrets.compare_digest(matches[0], self.server.media_cookie)):
            self._deny("INVALID_TOKEN")

    def _path(self):
        try:
            parsed = urlsplit(self.path)
        except ValueError as exc:
            raise BadRequest("INVALID_JSON", "Invalid request target") from exc
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise NotFound("NOT_FOUND", "Unknown route")
        self._query = {}
        if parsed.query:
            frame_query = re.fullmatch(r"/api/v1/projects/[A-Za-z0-9_-]+/media/[A-Za-z0-9_-]+/frames", parsed.path)
            export_query = re.fullmatch(r"/api/v1/projects/[A-Za-z0-9_-]+/tasks/[A-Za-z0-9_-]+/export", parsed.path)
            if (self.command != "GET" or not (frame_query or export_query)
                    or len(parsed.query) > (160 if frame_query else 4096)):
                raise NotFound("NOT_FOUND", "Unknown route")
            try:
                if re.search(r"%(?![0-9a-fA-F]{2})", parsed.query):
                    raise ValueError("Malformed percent escape")
                self._query = parse_qs(parsed.query, strict_parsing=True, keep_blank_values=True, errors="strict")
                allowed = {"offset", "limit"} if frame_query else {"format", "track", "track_id"}
                if set(self._query) - allowed or any(len(value) != 1 for value in self._query.values()):
                    raise ValueError("Unexpected query parameter")
                if export_query:
                    # A track is an exact stored string, not a filename or
                    # response header. Encoded tabs/newlines are legal track
                    # values under the existing annotation contract. Reject
                    # malformed query framing, but do not make such stored
                    # tracks impossible to select for a multi-track export.
                    format_value = self._query.get("format", [""])[0]
                    if any(ord(char) < 32 or ord(char) == 127 for char in format_value):
                        raise ValueError("Unsupported format control character")
            except (ValueError, UnicodeError) as exc:
                raise BadRequest("INVALID_QUERY", "Invalid query parameters") from exc
        return parsed.path

    def _pagination(self):
        values = {"offset": "0", "limit": "100"}
        values.update({name: value[0] for name, value in self._query.items()})
        if any(re.fullmatch(r"0|[1-9][0-9]{0,18}", value) is None for value in values.values()):
            raise BadRequest("INVALID_QUERY", "Pagination requires decimal integers")
        offset, limit = int(values["offset"]), int(values["limit"])
        if not 1 <= limit <= 1000:
            raise BadRequest("INVALID_QUERY", "Frame page limit must be 1 through 1000")
        return offset, limit

    def _send_file(self, path, content_type, *, byte_ranges=False, extra_headers=()):
        _safe(path)
        before = path.stat()
        with path.open("rb") as source:
            # Python 3.13 on Windows may report creation time through stat's
            # ctime but change time through fstat's ctime. Compare handle
            # identity/size/mtime and separately recheck the path snapshot.
            if (_identity(os.fstat(source.fileno()))[:4] != _identity(before)[:4]
                    or _identity(path.stat()) != _identity(before)):
                raise Conflict("MEDIA_CHANGED", "Resource changed before delivery")
            _safe(path)
            size = before.st_size
            start, end, status = 0, size - 1, 200
            self._extra_headers = list(extra_headers)
            if byte_ranges:
                self._extra_headers.append(("Accept-Ranges", "bytes"))
                ranges = self.headers.get_all("Range", [])
                if ranges:
                    try:
                        if len(ranges) != 1:
                            raise ValueError("Multiple ranges")
                        start, end = _byte_range(ranges[0], size)
                        status = 206
                    except ValueError:
                        self._extra_headers.append(("Content-Range", f"bytes */{size}"))
                        self._headers(416, 0, content_type)
                        return
                    self._extra_headers.append(("Content-Range", f"bytes {start}-{end}/{size}"))
            remaining = max(0, end - start + 1)
            self._headers(status, remaining, content_type)
            if self.command == "HEAD":
                return
            source.seek(start)
            while remaining:
                chunk = source.read(min(STREAM_CHUNK_BYTES, remaining))
                if not chunk:
                    # A truncated response cannot be reported as complete.
                    self.close_connection = True
                    return
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _media_request(self, parts):
        """Return True only for the fixed playback/image routes."""
        playback = len(parts) == 3 and parts[0] == "media" and parts[2] == "playback"
        frame = len(parts) == 5 and parts[0] == "media" and parts[2] == "frames" and parts[4] == "image"
        if not (playback or frame):
            return False
        self._media_token()
        if self.command not in {"GET", "HEAD"}:
            self._deny("METHOD_NOT_ALLOWED", 405)
        _require_id(parts[1], "media_id")
        if playback:
            path, digest, _ = resolve_media_object(self.server.store, parts[1], "playback_path")
            self._send_file(path, "video/mp4", byte_ranges=True,
                            extra_headers=[("X-TraceAnnotate-Media-SHA256", digest)])
        else:
            if re.fullmatch(r"0|[1-9][0-9]{0,18}", parts[3]) is None:
                raise BadRequest("INVALID_QUERY", "Frame index must be a decimal integer")
            image = self.server.image_service.image(parts[1], int(parts[3]))
            row = image["row"]
            self._extra_headers = [
                ("X-TraceAnnotate-Frame-Index", row["frame_index"]),
                ("X-TraceAnnotate-Source-PTS", row["source_pts"]),
                ("X-TraceAnnotate-Proxy-PTS", row["proxy_pts"]),
                ("X-TraceAnnotate-Image-SHA256", image["sha256"]),
                ("X-TraceAnnotate-Media-SHA256", image["playback_sha256"]),
            ]
            # Bytes are pinned and hashed under the decoder lock. Reopening a
            # cache filename here would race eviction by another request.
            self._headers(200, len(image["data"]), "image/png")
            if self.command != "HEAD":
                self.wfile.write(image["data"])
        return True

    def _api_parts(self, path):
        parts = path.split("/")
        if (len(parts) < 6 or parts[:4] != ["", "api", "v1", "projects"]
                or parts[4] not in ("current", self.server.project_id)):
            raise NotFound("NOT_FOUND", "Unknown project or route")
        return parts[5:]

    def _read_body(self):
        if self.headers.get_all("Transfer-Encoding", []) or self.headers.get_all("Content-Encoding", []):
            raise BadRequest("INVALID_JSON", "Encoded request bodies are not supported")
        types = self.headers.get_all("Content-Type", [])
        if len(types) != 1 or types[0].lower().replace(" ", "") not in ("application/json", "application/json;charset=utf-8"):
            raise BadRequest("INVALID_JSON", "Content-Type must be application/json")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None:
            raise BadRequest("INVALID_JSON", "A single valid Content-Length is required")
        length = int(lengths[0])
        if length > self.server.max_command_bytes:
            raise LimitExceeded("LIMIT_EXCEEDED", "Command body is too large")
        try:
            self._body_read_started = True
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            body = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
            _validate_json_tree(body)
        except (UnicodeError, ValueError, RecursionError, OverflowError, socket.timeout) as exc:
            raise BadRequest("INVALID_JSON", "Invalid JSON body") from exc
        return raw, body

    def _dispatch(self):
        path = self._path()
        self._authorize(write=self.command == "POST", bootstrap=self.command == "GET"
                        and (path == "/" or path in CLASSIC_DOCUMENTS))
        if path in CLASSIC_DOCUMENTS or path == HASH_SCRIPT:
            if self.command not in {"GET", "HEAD"}:
                self._deny("METHOD_NOT_ALLOWED", 405)
            data, content_type = classic_resource(path, self.server.nonce)
            self._classic_document = path in CLASSIC_DOCUMENTS
            self._headers(200, len(data), content_type)
            if self.command != "HEAD":
                self.wfile.write(data)
            return
        if self.command == "GET" and path == "/":
            html = EDITOR_HTML.replace("__TOKEN__", json.dumps(self.server.token)).replace("__NONCE__", self.server.nonce)
            data = html.encode("utf-8")
            self._extra_headers = [("Set-Cookie", f"{self.server.media_cookie_name}={self.server.media_cookie}; "
                                    f"Path=/api/v1/projects/{project}/media/; HttpOnly; SameSite=Strict")
                                   for project in ("current", self.server.project_id)]
            self._headers(200, len(data), "text/html; charset=utf-8")
            self.wfile.write(data)
            return
        parts = self._api_parts(path)
        if self._media_request(parts):
            return
        self._token()
        store = self.server.store
        if self.command == "POST" and parts == ["imports"]:
            result = receive_and_import(self)
            self._send_json(200, result)
            return
        if self.command == "GET" and parts == ["export"]:
            with store.transaction():
                snapshot = store.get_snapshot()
                snapshot["intakes"] = store.list_intakes()
                snapshot["export_scope"] = "development_snapshot_without_media_or_portable_import"
            self._extra_headers = [("Content-Disposition", 'attachment; filename="'
                                    f'TraceAnnotate-{self.server.project_id}-r{snapshot["workspace_revision"]}.json"')]
            self._send_json(200, snapshot)
            return
        if self.command == "GET":
            if parts == ["drafts"]:
                self._send_json(200, {"drafts": self.server.drafts.list()})
                return
            if len(parts) == 2 and parts[0] == "drafts":
                _require_id(parts[1], "draft_id")
                self._send_json(200, self.server.drafts.get(parts[1]))
                return
            if len(parts) == 3 and parts[0] == "tasks" and parts[2] in {"report", "export"}:
                _require_id(parts[1], "task_id")
                if parts[2] == "report":
                    self._send_json(200, task_report(store, parts[1]))
                else:
                    if "format" not in self._query:
                        raise BadRequest("INVALID_QUERY", "Task export requires an explicit format")
                    item = export_task(store, parts[1], self._query["format"][0],
                                       track=self._query.get("track", [None])[0],
                                       track_id=self._query.get("track_id", [None])[0])
                    self._extra_headers = [("Content-Disposition", f'attachment; filename="{item["filename"]}"'),
                                           ("X-TraceAnnotate-Task-Revision", str(item["binding"]["revision"])),
                                           ("X-TraceAnnotate-Content-SHA256", item["binding"]["content_sha256"]),
                                           ("X-TraceAnnotate-Export-Losses", item["losses"])]
                    self._headers(200, len(item["data"]), item["content_type"])
                    self.wfile.write(item["data"])
                return
            if len(parts) == 3 and parts[0] == "media" and parts[2] == "frames":
                _require_id(parts[1], "media_id")
                offset, limit = self._pagination()
                self._send_json(200, self.server.image_service.page(parts[1], offset, limit))
                return
            with store.transaction():
                if parts == ["snapshot"]:
                    result = store.get_snapshot()
                elif len(parts) == 2 and parts[0] == "tasks":
                    _require_id(parts[1], "task_id")
                    result = store.get_task(parts[1])
                elif len(parts) == 2 and parts[0] == "commands":
                    _require_id(parts[1], "command_id")
                    result = store.get_command_receipt(parts[1])
                elif parts == ["batches"]:
                    result = {"batches": self.server.queue.list_batches()}
                elif parts == ["intakes"]:
                    result = {"intakes": store.list_intakes()}
                elif parts == ["imports"]:
                    result = {"busy": self.server.import_lock.locked(),
                              "max_upload_bytes": str(self.server.max_upload_bytes)}
                elif len(parts) == 2 and parts[0] in ("batches", "jobs"):
                    _require_id(parts[1], "resource_id")
                    result = (self.server.queue.get_batch if parts[0] == "batches"
                              else self.server.queue.get_job)(parts[1])
                else:
                    raise NotFound("NOT_FOUND", "Unknown route")
        elif self.command == "POST":
            task_command = len(parts) == 3 and parts[0] == "tasks" and parts[2] == "commands"
            task_import_temporal = len(parts) == 3 and parts[0] == "tasks" and parts[2] == "import-temporal"
            control = len(parts) == 3 and parts[0] in ("batches", "jobs") and parts[2] == "control"
            if not (task_command or task_import_temporal or control or parts in (["batches"], ["drafts"])):
                raise NotFound("NOT_FOUND", "Unknown route")
            if len(parts) > 1:
                _require_id(parts[1], "resource_id")
            raw, body = self._read_body()
            with store.transaction():
                if parts == ["drafts"]:
                    expected = {"draft_id", "editor_session_id", "task_id", "media_id", "expected_draft_revision", "body"}
                    if not isinstance(body, dict) or set(body) != expected:
                        raise SchemaInvalid("SCHEMA_INVALID", "Invalid draft save request")
                    result = self.server.drafts.save(**body)
                elif task_command:
                    body = _command(body)
                    result = store.apply_task_command(task_id=parts[1], request_body=raw, **body)["response"]
                elif task_import_temporal:
                    task = store.get_task(parts[1])
                    media = store.get_media(task["media_id"])
                    doc = body.get("document", body)
                    if not isinstance(doc, dict):
                        raise SchemaInvalid("SCHEMA_INVALID", "Temporal document must be an object")
                    # D00/D04 stopgap: the client must pin the command identity and the
                    # exact base revision it saw. The server must not invent a fresh
                    # command id or silently replay against the current revision, or a
                    # repeated import would overwrite later human edits.
                    cmd_id = body.get("command_id")
                    if not isinstance(cmd_id, str) or not cmd_id:
                        raise SchemaInvalid("IMPORT_REQUIRES_COMMAND_ID",
                                            "Temporal import requires an explicit command_id")
                    exp_rev = body.get("expected_revision")
                    exp_sha = body.get("expected_sha256")
                    if type(exp_rev) is not int or not isinstance(exp_sha, str) or not exp_sha:
                        raise SchemaInvalid("IMPORT_REQUIRES_BASE_REVISION",
                                            "Temporal import requires expected_revision and expected_sha256")
                    # D00/D03 stopgap: a v2 document pins the source media by hash and
                    # size. Never replace task annotations when those do not match the
                    # media this task actually references; name and duration are hints only.
                    declared_sha = doc.get("media_sha256")
                    if declared_sha is not None:
                        source_sha = (media.get("source") or {}).get("sha256")
                        if (not isinstance(declared_sha, str) or not declared_sha
                                or source_sha is None or source_sha != declared_sha.lower()):
                            raise SchemaInvalid("MEDIA_IDENTITY_MISMATCH",
                                                "Document media_sha256 does not match the task media")
                    declared_size = doc.get("media_size")
                    if declared_size is not None:
                        source_size = (media.get("source") or {}).get("size_bytes")
                        try:
                            if source_size is None or int(source_size) != int(declared_size):
                                raise SchemaInvalid("MEDIA_IDENTITY_MISMATCH",
                                                    "Document media_size does not match the task media")
                        except (TypeError, ValueError) as exc:
                            raise SchemaInvalid("MEDIA_IDENTITY_MISMATCH",
                                                "Invalid media_size") from exc
                    from .legacy_adapter import temporal_import_payload
                    import_payload = temporal_import_payload(doc, media)
                    result = store.apply_task_command(
                        command_id=cmd_id,
                        task_id=task["id"],
                        expected_revision=exp_rev,
                        expected_sha256=exp_sha,
                        operation="import_temporal",
                        payload=import_payload,
                        request_body=raw,
                    )["response"]
                    result["imported_count"] = len(import_payload["annotations"])
                    result["endpoint_adjustments"] = len(import_payload["legacy_document_envelope"].get("endpoint_adjustments", []))
                elif control:
                    body = _fields(body, {"command_id", "action"})
                    if not isinstance(body["action"], str):
                        raise SchemaInvalid("SCHEMA_INVALID", "action must be a string")
                    method = self.server.queue.control_batch if parts[0] == "batches" else self.server.queue.control_job
                    result = method(parts[1], **body)
                else:
                    body = _fields(body, {"command_id", "items", "operation", "processor_version", "parameters"})
                    result = self.server.queue.create_batch(**body)
        else:
            self._deny("METHOD_NOT_ALLOWED", 405)
        # Report success only after the outer transaction has committed.
        self._send_json(200, result)

    def _serve(self):
        request_id = "req-" + secrets.token_hex(12)
        try:
            if self.command not in ("GET", "POST", "HEAD"):
                self._authorize()
                self._deny("METHOD_NOT_ALLOWED", 405)
            self._dispatch()
        except (ConnectionError, socket.timeout):
            self.close_connection = True
        except WorkspaceError as exc:
            self._error(exc, request_id)
        except Exception:
            self._error(WorkspaceError("INTERNAL_ERROR", "Request failed"), request_id)

    do_GET = do_POST = do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _serve


def make_server(store: WorkspaceStore, host: str = "127.0.0.1", port: int = 0) -> WorkspaceHTTPServer:
    """Bind one opened store on loopback; caller owns shutdown and store.close()."""
    if host != "127.0.0.1":
        raise ValueError("Workspaces can be served only on 127.0.0.1")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 through 65535")
    with store.transaction():
        queue = DurableQueue(store)
        project_id = store.get_snapshot()["project_id"]
    server = WorkspaceHTTPServer((host, port), _Handler)
    server.store = store
    from .drafts import WorkspaceDrafts
    server.drafts = WorkspaceDrafts(store)
    server.queue = queue
    server.project_id = project_id
    server.token = secrets.token_urlsafe(32)
    server.nonce = secrets.token_urlsafe(24)
    server.media_cookie = secrets.token_urlsafe(32)
    server.media_cookie_name = f"ta_media_{server.server_port}_{secrets.token_hex(6)}"
    server.max_command_bytes = MAX_COMMAND_BYTES
    server.max_upload_bytes = MAX_UPLOAD_BYTES
    server.upload_timeout_seconds = UPLOAD_TIMEOUT_SECONDS
    server.import_lock = threading.Lock()
    server.closing = threading.Event()
    from .frames import FrameService
    server.image_service = FrameService(store)
    server.authority = f"127.0.0.1:{server.server_port}"
    server.origin = f"http://{server.authority}"
    server.editor_url = server.origin + "/"
    return server


def serve(store: WorkspaceStore, host: str = "127.0.0.1", port: int = 0,
          open_browser: bool = True) -> None:
    httpd = make_server(store, host, port)
    print(f"TraceAnnotate workspace server: {httpd.editor_url}")
    if open_browser:
        webbrowser.open(httpd.editor_url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
