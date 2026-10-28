"""Explicit queue processors. Integrity verification does not label video content."""
from __future__ import annotations

import json
import time

from ..temporal.media import stream_sha256
from . import schema as S
from .errors import WorkspaceError
from .store import _safe

OPERATION = "verify-media"
VERSION = "1"


def media_verifier(store):
    def verify(job, context):
        if job["operation"] != OPERATION or job["processor_version"] != VERSION or job["parameters"]:
            raise WorkspaceError("UNSUPPORTED_PROCESSOR", "No installed processor matches this frozen job configuration")
        media = job["media_snapshot"]
        last_heartbeat = time.monotonic()

        def cancelled():
            nonlocal last_heartbeat
            if context.cancelled():
                return True
            if time.monotonic() - last_heartbeat >= 5:
                context.heartbeat()
                last_heartbeat = time.monotonic()
            return False

        with store.transaction():
            mapping = store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE id=?", (media["frame_map_artifact"],)).fetchone()
            playback = store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE sha256=?", (media["playback"]["sha256"],)).fetchone()
        if mapping is None or playback is None:
            raise WorkspaceError("RECOVERY_REQUIRED", "Media objects are not registered")
        for row in (mapping, playback):
            if row[0] != f"objects/{row[1]}":
                raise WorkspaceError("RECOVERY_REQUIRED", "Media object path is invalid")
            path = store.root / row[0]
            _safe(path)
            if not path.is_file() or path.stat().st_size != row[2] or stream_sha256(path, cancelled=cancelled) != row[1]:
                raise WorkspaceError("MEDIA_CHANGED", "Media object no longer matches its admitted hash")
        count, previous = 0, None
        base = S.rational_value(media["proxy_time_base"])
        source_base = S.rational_value(media["source_time_base"])
        origin = S.rational_value(media["common_origin"])
        with (store.root / mapping[0]).open(encoding="utf-8") as stream:
            for line in stream:
                context.raise_if_cancelled()
                if cancelled():
                    raise WorkspaceError("CANCELLED", "Verification was cancelled")
                row = json.loads(line)
                seconds = S.rational_value(row["seconds"])
                if (row["frame_index"] != str(count) or seconds != int(row["proxy_pts"]) * base
                        or abs(int(row["source_pts"]) * source_base - origin - seconds) > min(base, source_base)
                        or (previous is not None and seconds <= previous)):
                    raise WorkspaceError("INVALID_FRAME_MAP", "Frame map does not preserve its admitted timeline")
                count += 1
                previous = seconds
        if str(count) != media["frame_count"]:
            raise WorkspaceError("INVALID_FRAME_MAP", "Frame count differs from the admitted media")
        return {"kind": "traceannotate.media-integrity", "version": 1, "media_id": media["id"],
                "frame_count": str(count), "playback_sha256": playback[1], "frame_map_sha256": mapping[1],
                "scope": "immutable playback bytes and exact frame-map arithmetic",
                "semantic_annotation": "not_performed"}
    return verify
