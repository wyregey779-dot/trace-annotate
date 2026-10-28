"""Loopback HTTP access to one opened, revisioned workspace.

The initial page is a snapshot viewer. It serves no arbitrary files and accepts
no filesystem paths. Write requests require an exact origin and private header.
"""
from __future__ import annotations
import json
import math
import re
import secrets
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import webbrowser

from .errors import BadRequest, LimitExceeded, NotFound, SchemaInvalid, WorkspaceError
from .queue import DurableQueue
from .store import WorkspaceStore

MAX_COMMAND_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 10
MAX_SAFE_INTEGER = 9007199254740991
TOKEN_HEADER = "X-TraceAnnotate-Token"
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,95}")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")

EDITOR_HTML = """<!doctype html><meta charset=utf-8><title>TraceAnnotate Workspace</title>
<body style=font-family:system-ui;padding:16px>
<h2>TraceAnnotate Workspace</h2><pre id=out>loading…</pre>
<script nonce=__NONCE__>
const token=__TOKEN__;
async function j(path, opts={}) {
  const headers=new Headers(opts.headers);
  headers.set('X-TraceAnnotate-Token',token);
  const r=await fetch(path,{...opts,headers,mode:'same-origin',cache:'no-store'});
  return {status:r.status, body: await r.json().catch(()=>null)};
}
(async()=>{
  try {
    const s=await j('/api/v1/projects/current/snapshot');
    document.getElementById('out').textContent=JSON.stringify(s.body,null,2);
  } catch (_) {
    document.getElementById('out').textContent='Workspace connection failed. Reload to retry.';
  }
})();
</script></body>"""


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
    return body


def _fields(body, expected):
    if not isinstance(body, dict) or set(body) != expected:
        raise SchemaInvalid("SCHEMA_INVALID", "Request fields are missing or unknown")
    _require_id(body["command_id"], "command_id")
    return body


class WorkspaceHTTPServer(ThreadingHTTPServer):
    """Caller-managed HTTP lifecycle; token and store belong to this instance."""
    daemon_threads = True
    allow_reuse_address = False

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(REQUEST_TIMEOUT_SECONDS)
        return connection, address


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
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; connect-src 'self'; "
                         f"script-src 'nonce-{self.server.nonce}'; style-src 'unsafe-inline'; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        if status == 405:
            self.send_header("Allow", "GET, POST")
        self.end_headers()
        self.close_connection = True

    def _send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, len(data), "application/json; charset=utf-8")
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, exc, request_id):
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
        self._send_json(exc.http_status, body)

    def _deny(self, code, status=403):
        exc = WorkspaceError(code, "Request rejected")
        exc.http_status = status
        raise exc

    def _authorize(self, write=False):
        if self.headers.get_all("Host", []) != [self.server.authority]:
            self._deny("FORBIDDEN_ORIGIN")
        origins = self.headers.get_all("Origin", [])
        if (origins and origins != [self.server.origin]) or (write and not origins):
            self._deny("FORBIDDEN_ORIGIN")
        sites = self.headers.get_all("Sec-Fetch-Site", [])
        if sites and sites not in (["same-origin"], ["none"]):
            self._deny("FORBIDDEN_ORIGIN")

    def _token(self):
        tokens = self.headers.get_all(TOKEN_HEADER, [])
        if (len(tokens) != 1 or not tokens[0].isascii()
                or not secrets.compare_digest(tokens[0], self.server.token)):
            self._deny("INVALID_TOKEN")

    def _path(self):
        try:
            parsed = urlsplit(self.path)
        except ValueError as exc:
            raise BadRequest("INVALID_JSON", "Invalid request target") from exc
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise NotFound("NOT_FOUND", "Unknown route")
        return parsed.path

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
        self._authorize(write=self.command == "POST")
        path = self._path()
        if self.command == "GET" and path == "/":
            html = EDITOR_HTML.replace("__TOKEN__", json.dumps(self.server.token)).replace("__NONCE__", self.server.nonce)
            data = html.encode("utf-8")
            self._headers(200, len(data), "text/html; charset=utf-8")
            self.wfile.write(data)
            return
        self._token()
        parts = self._api_parts(path)
        store = self.server.store
        if self.command == "GET":
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
                elif len(parts) == 2 and parts[0] in ("batches", "jobs"):
                    _require_id(parts[1], "resource_id")
                    result = (self.server.queue.get_batch if parts[0] == "batches"
                              else self.server.queue.get_job)(parts[1])
                else:
                    raise NotFound("NOT_FOUND", "Unknown route")
        elif self.command == "POST":
            task_command = len(parts) == 3 and parts[0] == "tasks" and parts[2] == "commands"
            control = len(parts) == 3 and parts[0] in ("batches", "jobs") and parts[2] == "control"
            if not (task_command or control or parts == ["batches"]):
                raise NotFound("NOT_FOUND", "Unknown route")
            if len(parts) > 1:
                _require_id(parts[1], "resource_id")
            raw, body = self._read_body()
            with store.transaction():
                if task_command:
                    body = _command(body)
                    result = store.apply_task_command(task_id=parts[1], request_body=raw, **body)["response"]
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
            if self.command not in ("GET", "POST"):
                self._authorize()
                self._deny("METHOD_NOT_ALLOWED", 405)
            self._dispatch()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
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
    server.queue = queue
    server.project_id = project_id
    server.token = secrets.token_urlsafe(32)
    server.nonce = secrets.token_urlsafe(24)
    server.max_command_bytes = MAX_COMMAND_BYTES
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
