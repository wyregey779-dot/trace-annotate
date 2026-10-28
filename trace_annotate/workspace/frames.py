"""Exact frame lookup and decoded proxy images; player seeks are not evidence.

Indices are zero-based decoded display order. Images are decoded from the start
with select(n), never time seeking or average FPS. Output is the admitted lossy
playback's pixels, not a claim of original color/pixel fidelity. Frame anchors
are checked against the immutable map before accepting an annotation revision.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

from ..temporal.media import stream_sha256
from . import schema as S
from .errors import DependencyUnavailable, NotFound, SchemaInvalid, WorkspaceError
from .store import _safe

STRIDE = 256
MAX_LINE = 4096
MAX_FRAMES = 2_000_000
MAX_CACHED_IMAGES = 32
MAX_CACHED_BYTES = 256 * 1024 * 1024
DECODE_TIMEOUT = 120


def _integer(value, field):
    if not isinstance(value, str) or len(value) > 40 or re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None or value == "-0":
        raise SchemaInvalid("INVALID_FRAME_MAP", f"Invalid {field}")
    return int(value)


def _binding(store, media_id, kind):
    S.require_id(media_id, "media_id")
    with store.transaction():
        media = store.get_media(media_id)
        raw = store.conn.execute("SELECT value FROM meta WHERE key=?", (f"media_objects:{media_id}",)).fetchone()
        if raw is None:
            raise WorkspaceError("RECOVERY_REQUIRED", "Missing media object binding")
        item = json.loads(raw[0]).get(kind, {})
        row = store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE id=?", (item.get("id"),)).fetchone()
    if (row is None or row[0] != f"objects/{row[1]}" or item.get("sha256") != row[1]
            or item.get("size_bytes") != str(row[2])):
        raise WorkspaceError("RECOVERY_REQUIRED", "Invalid media object binding")
    if kind == "frame_map_path" and item["id"] != media["frame_map_artifact"]:
        raise WorkspaceError("RECOVERY_REQUIRED", "Frame map identity differs")
    if kind == "playback_path" and (row[1], str(row[2])) != (media["playback"]["sha256"], media["playback"]["size_bytes"]):
        raise WorkspaceError("RECOVERY_REQUIRED", "Playback identity differs")
    path = store.root / row[0]
    _safe(path)
    if not path.is_file() or path.stat().st_size != row[2]:
        raise WorkspaceError("RECOVERY_REQUIRED", "Media object is missing or changed")
    return media, path, row[1]


class _Index:
    def __init__(self, path, digest, media):
        self.path, self.digest = path, digest
        self.offsets = []
        self.count = 0
        checksum = hashlib.sha256()
        previous = None
        base, source_base = S.rational_value(media["proxy_time_base"]), S.rational_value(media["source_time_base"])
        origin = S.rational_value(media["common_origin"])
        first, endpoint = S.rational_value(media["video_start"]), S.rational_value(media["video_end"])
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE or self.count >= MAX_FRAMES:
                    raise SchemaInvalid("INVALID_FRAME_MAP", "Frame map exceeds the supported bound")
                checksum.update(line)
                try:
                    row = json.loads(line)
                    if set(row) != {"frame_index", "source_pts", "proxy_pts", "seconds"}:
                        raise ValueError
                    index = _integer(row["frame_index"], "frame_index")
                    source_pts = _integer(row["source_pts"], "source_pts")
                    proxy_pts = _integer(row["proxy_pts"], "proxy_pts")
                    seconds = S.rational_value(row["seconds"])
                    if (index != self.count or S.rational(seconds) != row["seconds"]
                            or seconds != proxy_pts * base
                            or abs(source_pts * source_base - origin - seconds) > min(base, source_base)
                            or not first <= seconds < endpoint
                            or (self.count == 0 and seconds != first)
                            or (previous is not None and (seconds <= previous[0] or source_pts <= previous[1]))):
                        raise ValueError
                except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                    raise SchemaInvalid("INVALID_FRAME_MAP", "Frame map does not preserve its admitted timeline") from exc
                if self.count % STRIDE == 0:
                    self.offsets.append(offset)
                self.count += 1
                previous = seconds, source_pts
        if checksum.hexdigest() != digest or str(self.count) != media["frame_count"] or self.count == 0:
            raise WorkspaceError("RECOVERY_REQUIRED", "Frame map bytes or count changed")

    def rows(self, offset, limit):
        if offset == self.count:
            return []
        with self.path.open("rb") as stream:
            block = offset // STRIDE
            stream.seek(self.offsets[block])
            for _ in range(offset - block * STRIDE):
                stream.readline(MAX_LINE + 1)
            return [json.loads(stream.readline(MAX_LINE + 1)) for _ in range(min(limit, self.count - offset))]


class FrameService:
    def __init__(self, store):
        self.store = store
        self._lock = threading.RLock()
        self._decoder = threading.Lock()
        self._indices = OrderedDict()
        self._images = OrderedDict()

    def page(self, media_id, offset=0, limit=100):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise SchemaInvalid("SCHEMA_INVALID", "Invalid frame page")
        # Acquire DB metadata before the cache lock. Store validation may call
        # this service while holding its transaction; the inverse order deadlocks.
        media, path, digest = _binding(self.store, media_id, "frame_map_path")
        with self._lock:
            # Hash on every read, including cache hits: an external same-size
            # edit must not turn a previously verified index into false evidence.
            if stream_sha256(path) != digest:
                raise WorkspaceError("RECOVERY_REQUIRED", "Frame map bytes changed")
            key = (media_id, digest, S.sha256_text(media))
            index = self._indices.get(key)
            if index is None:
                index = _Index(path, digest, media)
                self._indices[key] = index
            self._indices.move_to_end(key)
            while len(self._indices) > 8:
                self._indices.popitem(last=False)
            if offset > index.count:
                raise NotFound("FRAME_NOT_FOUND", "Frame offset is beyond the media")
            rows = index.rows(offset, limit)
            if stream_sha256(path) != digest:
                raise WorkspaceError("RECOVERY_REQUIRED", "Frame map changed during lookup")
            return {"media_id": media_id, "offset": str(offset), "total": str(index.count),
                    "frame_map_sha256": digest, "frames": rows}

    def row(self, media_id, index):
        rows = self.page(media_id, index, 1)["frames"]
        if not rows:
            raise NotFound("FRAME_NOT_FOUND", "No such decoded frame")
        return rows[0]

    def image(self, media_id, index):
        row = self.row(media_id, index)
        media, path, playback_hash = _binding(self.store, media_id, "playback_path")
        if stream_sha256(path) != playback_hash:
            raise WorkspaceError("RECOVERY_REQUIRED", "Playback bytes changed")
        key = (media_id, playback_hash, media["source_video_stream"], index, "proxy-decoded-rgb-v1")
        if not self._decoder.acquire(blocking=False):
            raise DependencyUnavailable("FRAME_SERVICE_BUSY", "A frame is being decoded; retry after it finishes", retryable=True)
        try:
            if key in self._images:
                item = self._images[key]
                _safe(item["path"])
                data = item["path"].read_bytes()
                if S.sha256_bytes(data) != item["sha256"]:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Cached frame bytes changed")
                self._images.move_to_end(key)
                return dict(item, data=data)
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "FFmpeg is needed to decode frames")
            _safe(self.store.tmp_dir)
            work = Path(tempfile.mkdtemp(prefix="frame-", dir=self.store.tmp_dir))
            output, diagnostic = work / "frame.png", work / "decoder.stderr"
            command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", "-xerror",
                       "-protocol_whitelist", "file,pipe", "-err_detect", "explode", "-i", str(path),
                       "-map", "0:v:0", "-vf", f"select=eq(n\\,{index})", "-fps_mode", "passthrough",
                       "-frames:v", "1", "-c:v", "png", "-pix_fmt", "rgb24", "-f", "image2", str(output)]
            exit_code, outcome = None, "failed"
            try:
                with diagnostic.open("xb") as errors:
                    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=errors, timeout=DECODE_TIMEOUT)
                exit_code = result.returncode
                if exit_code or diagnostic.stat().st_size or not output.is_file():
                    raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder did not produce the requested frame")
                if output.stat().st_size > MAX_CACHED_BYTES:
                    raise WorkspaceError("FRAME_TOO_LARGE", "Decoded frame exceeds the cache bound")
                with output.open("rb") as stream:
                    if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                        raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder output is not PNG")
                if stream_sha256(path) != playback_hash:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Playback changed during frame decode")
                outcome = "succeeded"
            except subprocess.TimeoutExpired as exc:
                outcome = "timeout"
                raise WorkspaceError("FRAME_DECODE_TIMEOUT", "Exact frame decode exceeded the time limit") from exc
            finally:
                (work / "decode.json").write_bytes(S.canonical_json({"media_id": media_id,
                    "playback_sha256": playback_hash, "row": row, "transform": key[-1],
                    "command": command, "exit_code": exit_code, "outcome": outcome}))
            data = output.read_bytes()
            item = {"path": output, "sha256": S.sha256_bytes(data), "row": row,
                    "size_bytes": len(data), "playback_sha256": playback_hash}
            self._images[key] = item
            while len(self._images) > MAX_CACHED_IMAGES or sum(i["size_bytes"] for i in self._images.values()) > MAX_CACHED_BYTES:
                _, old = self._images.popitem(last=False)
                _safe(old["path"])
                # Remove only an owned cache image whose bytes are unchanged;
                # retain its small decode log. Never traverse an external link.
                if old["path"].is_file() and stream_sha256(old["path"]) == old["sha256"]:
                    old["path"].unlink()
            # HTTP must send these pinned bytes, not reopen a path after the
            # cache lock is released and a later request may evict that image.
            return dict(item, data=data)
        finally:
            self._decoder.release()

    def close(self):
        """Remove only this session's unchanged successful cache images.

        Failed/abruptly interrupted decodes and small logs remain diagnostic
        evidence. No recursive deletion or cleanup of another session is used.
        """
        with self._decoder:
            for item in self._images.values():
                _safe(item["path"])
                if item["path"].is_file() and stream_sha256(item["path"]) == item["sha256"]:
                    item["path"].unlink()
            self._images.clear()
        with self._lock:
            self._indices.clear()


def validate_anchor(store, media, value):
    anchor = value["anchor"]
    if not isinstance(anchor, dict) or set(anchor) != {"frame_index", "source_pts", "proxy_pts"}:
        raise SchemaInvalid("INVALID_FRAME_ANCHOR", "Frame anchor fields are invalid")
    index = _integer(anchor["frame_index"], "frame_index")
    for name in ("source_pts", "proxy_pts"):
        _integer(anchor[name], name)
    if index < 0 or index >= int(media["frame_count"]):
        raise SchemaInvalid("INVALID_FRAME_ANCHOR", "Frame anchor is outside the admitted media")
    service = getattr(store, "_frame_service", None)
    if service is None:
        service = store._frame_service = FrameService(store)
    row = service.row(media["id"], index)
    if any(anchor[key] != row[key] for key in anchor) or value["seconds"] != row["seconds"]:
        raise SchemaInvalid("INVALID_FRAME_ANCHOR", "Frame anchor differs from the admitted map")
