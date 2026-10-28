"""Bounded browser media I/O for an already opened workspace.

Only registered immutable playback objects are readable. Browser filenames are
display metadata, never paths; incoming bytes go to a uniquely owned temporary
file before the ordinary verified media admission pipeline runs.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import unquote_to_bytes

from ..temporal.media import stream_sha256
from .errors import BadRequest, Conflict, LimitExceeded, StorageFull, WorkspaceError
from .store import _safe

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
UPLOAD_TIMEOUT_SECONDS = 120
STREAM_CHUNK_BYTES = 64 * 1024
FILENAME_HEADER = "X-TraceAnnotate-Filename"


def resolve_media_object(store, media_id, key):
    """Resolve a bound artifact, refusing corrupt metadata or changed bytes."""
    if key not in {"playback_path", "frame_map_path"}:
        raise ValueError("Only playback and frame maps are HTTP media resources")
    with store.transaction():
        media = store.get_media(media_id)
        row = store.conn.execute("SELECT value FROM meta WHERE key=?",
                                 (f"media_objects:{media_id}",)).fetchone()
        if row is None:
            raise WorkspaceError("RECOVERY_REQUIRED", "Media object binding is missing")
        try:
            reference = json.loads(row[0])[key]
            artifact = store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE id=?",
                                          (reference["id"],)).fetchone()
            if (artifact is None or re.fullmatch(r"[0-9a-f]{64}", artifact[1]) is None
                    or artifact[0] != f"objects/{artifact[1]}"
                    or (artifact[1], str(artifact[2])) != (reference["sha256"], reference["size_bytes"])):
                raise ValueError("Inconsistent artifact binding")
            if key == "playback_path" and (artifact[1], str(artifact[2])) != (
                    media["playback"]["sha256"], media["playback"]["size_bytes"]):
                raise ValueError("Playback identity differs from media")
            if key == "frame_map_path" and reference["id"] != media["frame_map_artifact"]:
                raise ValueError("Frame-map identity differs from media")
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkspaceError("RECOVERY_REQUIRED", "Media object binding is inconsistent") from exc
    path = store.root / artifact[0]
    _safe(path)
    try:
        before = path.stat()
        if (not stat.S_ISREG(before.st_mode) or before.st_size != artifact[2]
                or stream_sha256(path) != artifact[1]):
            raise ValueError("Media object changed")
        after = path.stat()
        if _identity(before) != _identity(after):
            raise ValueError("Media object changed while being checked")
    except (OSError, ValueError) as exc:
        raise Conflict("MEDIA_CHANGED", "Media object is unavailable or corrupt") from exc
    return path, artifact[1], artifact[2]


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def upload_name(headers):
    values = headers.get_all(FILENAME_HEADER, [])
    if len(values) != 1 or len(values[0]) > 4096:
        raise BadRequest("INVALID_UPLOAD", "An encoded filename header is required")
    value = values[0]
    if not value.isascii() or re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise BadRequest("INVALID_UPLOAD", "Filename encoding is invalid")
    try:
        name = unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise BadRequest("INVALID_UPLOAD", "Filename encoding is invalid") from exc
    if (not name.strip() or len(name) > 255 or name in {".", ".."}
            or any(char in name for char in "/\\:")
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise BadRequest("INVALID_UPLOAD", "Filename must be a plain display name")
    return name


def upload_length(headers, maximum):
    if headers.get_all("Transfer-Encoding", []) or headers.get_all("Content-Encoding", []):
        raise BadRequest("INVALID_UPLOAD", "Encoded or chunked uploads are unsupported")
    if headers.get_all("Content-Type", []) != ["application/octet-stream"]:
        raise BadRequest("INVALID_UPLOAD", "Upload Content-Type must be application/octet-stream")
    values = headers.get_all("Content-Length", [])
    if len(values) != 1 or re.fullmatch(r"[0-9]{1,12}", values[0]) is None:
        raise BadRequest("INVALID_UPLOAD", "Upload requires a single Content-Length")
    size = int(values[0])
    if size > maximum:
        raise LimitExceeded("LIMIT_EXCEEDED", "Upload exceeds the configured size limit")
    if size == 0:
        raise BadRequest("INVALID_UPLOAD", "Upload must contain bytes")
    return size


def receive_and_import(handler):
    """One synchronous import; there is deliberately no automatic POST replay.

    A transport interruption before full upload creates no admitted task. A
    completed upload uses store.import_source, which persists preparation
    failures. A lost response after admission must be reconciled from intakes
    and the snapshot before the user explicitly starts a new import.
    """
    server = handler.server
    name = upload_name(handler.headers)
    size = upload_length(handler.headers, server.max_upload_bytes)
    if not server.import_lock.acquire(blocking=False):
        raise Conflict("IMPORT_BUSY", "Another upload or media preparation is running")
    path = None
    previous_timeout = handler.connection.gettimeout()
    try:
        _safe(server.store.tmp_dir)
        descriptor, filename = tempfile.mkstemp(prefix="browser-upload-", suffix=".source", dir=server.store.tmp_dir)
        path = Path(filename)
        deadline = time.monotonic() + server.upload_timeout_seconds
        handler._body_read_started = True
        remaining = size
        with os.fdopen(descriptor, "wb") as output:
            while remaining:
                if server.closing.is_set():
                    raise WorkspaceError("CANCELLED", "Server shutdown cancelled the upload")
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise BadRequest("UPLOAD_INCOMPLETE", "Upload deadline expired")
                handler.connection.settimeout(min(10, timeout))
                chunk = handler.rfile.read1(min(STREAM_CHUNK_BYTES, remaining))
                if not chunk:
                    raise BadRequest("UPLOAD_INCOMPLETE", "Upload ended before all bytes arrived")
                output.write(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        return server.store.import_source(path, display_name=name, cancelled=server.closing.is_set)
    except OSError as exc:
        if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
            raise StorageFull("STORAGE_FULL", "Upload requires more free storage", retryable=True) from exc
        if isinstance(exc, TimeoutError):
            raise BadRequest("UPLOAD_INCOMPLETE", "Upload timed out") from exc
        raise
    finally:
        try:
            handler.connection.settimeout(previous_timeout)
        except OSError:
            pass
        try:
            if path is not None:
                # This is the exact mkstemp entry owned by this request, never a
                # client-supplied path or another job's preparation directory.
                _safe(path.parent)
                path.unlink(missing_ok=True)
        finally:
            server.import_lock.release()
