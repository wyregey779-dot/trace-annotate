"""Minimal loopback HTTP server for the workspace editor (W02 slice).

Bound to 127.0.0.1 with a per-start capability token. Not production-grade; the
full contract (CSRF, Origin check, upload quotas) lands with the W03 queue work.
"""
from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
import urllib.parse
import webbrowser

from .errors import WorkspaceError
from .store import WorkspaceStore

EDITOR_HTML = """<!doctype html><meta charset=utf-8><title>TraceAnnotate Workspace</title>
<body style=font-family:system-ui;padding:16px>
<h2>TraceAnnotate Workspace</h2><pre id=out>loading…</pre>
<script>
const token=__TOKEN__;
async function j(path, opts={}) {
  const r=await fetch(path,{headers:{'Authorization':'Bearer '+token,...(opts.headers||{})},...opts});
  return {status:r.status, body: await r.json().catch(()=>null)};
}
(async()=>{
  const s=await j('/api/v1/projects/current/snapshot');
  document.getElementById('out').textContent=JSON.stringify(s.body,null,2);
})();
</script></body>"""


class _Handler(BaseHTTPRequestHandler):
    store: WorkspaceStore = None  # type: ignore
    token: str = ""

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, exc: WorkspaceError, request_id: str) -> None:
        self._send_json(exc.http_status, exc.body(request_id))

    def _check_token(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            html = EDITOR_HTML.replace("__TOKEN__", self.token).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if not self._check_token():
            self._send_json(403, {"error": {"code": "INVALID_TOKEN"}})
            return
        try:
            if parsed.path.endswith("/snapshot"):
                self._send_json(200, self.store.get_snapshot())
            else:
                self._send_json(404, {"error": {"code": "NOT_FOUND"}})
        except WorkspaceError as exc:
            self._error(exc, "req")

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_token():
            self._send_json(403, {"error": {"code": "INVALID_TOKEN"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"code": "INVALID_JSON"}})
            return
        try:
            # /api/v1/projects/current/tasks/{tid}/commands
            parts = self.path.strip("/").split("/")
            if len(parts) == 6 and parts[-1] == "commands":
                task_id = parts[-2]
                resp = self.store.apply_task_command(
                    command_id=body["command_id"], task_id=task_id,
                    expected_revision=body["expected_revision"],
                    expected_sha256=body["expected_sha256"],
                    operation=body["operation"], payload=body["payload"],
                    request_body=raw,
                )
                self._send_json(200, resp["response"])
            else:
                self._send_json(404, {"error": {"code": "NOT_FOUND"}})
        except WorkspaceError as exc:
            self._error(exc, "req")


def serve(store: WorkspaceStore, host: str = "127.0.0.1", port: int = 0,
          open_browser: bool = True) -> None:
    token = secrets.token_urlsafe(24)
    _Handler.store = store
    _Handler.token = token
    httpd = ThreadingHTTPServer((host, port), _Handler)
    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"TraceAnnotate workspace server: {url}  (token: {token})")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
