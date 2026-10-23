"""Content identity for local media; no decoder or label verification is implied.

Hashing reads a bounded chunk at a time and rejects observed stat changes. That
is an ordinary concurrent-write check, not a guarantee against a hostile process
that changes bytes and restores filesystem timestamps during the read.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .schema import MAX_SAFE_INTEGER, require_valid_document


DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_CHUNK_SIZE = 16 * 1024 * 1024
DURATION_TOLERANCE_SECONDS = 0.001
MediaPath = Union[str, os.PathLike]


@dataclass(frozen=True)
class MediaIdentity:
    """Full-file hash and size, with a caller-declared duration in seconds."""

    clip: str
    duration: float
    media_size: int
    media_sha256: str


def _snapshot(info: os.stat_result) -> Tuple[int, int, int, int]:
    # On Windows/Python versions, stat and fstat can expose different ctime
    # semantics. Compare ctime only within the same API before/after the read.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _check_cancelled(cancelled: Optional[Callable[[], bool]]) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("Media hashing cancelled")


def _hash_file(path: MediaPath, chunk_size: int,
               cancelled: Optional[Callable[[], bool]]) -> Tuple[str, int]:
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int)
            or not 1 <= chunk_size <= MAX_CHUNK_SIZE):
        raise ValueError(f"chunk_size must be an integer between 1 and {MAX_CHUNK_SIZE}")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be a callable or None")
    _check_cancelled(cancelled)
    source = Path(path)
    initial = source.stat()
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("Media source must be a regular file")
    if not 0 <= initial.st_size <= MAX_SAFE_INTEGER:
        raise ValueError("Media size must be between 0 and 9007199254740991 bytes")
    expected = _snapshot(initial)
    digest = hashlib.sha256()
    read_size = 0
    with source.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or _snapshot(opened) != expected:
            raise OSError("Media changed before hashing; retry after writes have stopped")
        while True:
            _check_cancelled(cancelled)
            # Read at most one byte beyond the original boundary. An active
            # recorder must fail promptly instead of being followed until EOF.
            remaining = initial.st_size - read_size
            chunk = stream.read(min(chunk_size, remaining + 1))
            if not chunk:
                break
            if len(chunk) > remaining:
                raise OSError("Media changed during hashing; file grew beyond its initial size")
            digest.update(chunk)
            read_size += len(chunk)
        _check_cancelled(cancelled)
        after_read = os.fstat(stream.fileno())
        if (_snapshot(after_read) != expected
                or after_read.st_ctime_ns != opened.st_ctime_ns):
            raise OSError("Media changed during hashing; retry after writes have stopped")
    final = source.stat()
    if (_snapshot(final) != expected or final.st_ctime_ns != initial.st_ctime_ns
            or read_size != initial.st_size):
        raise OSError("Media changed during hashing; retry after writes have stopped")
    return digest.hexdigest(), read_size


def stream_sha256(path: MediaPath, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
                  cancelled: Optional[Callable[[], bool]] = None) -> str:
    """Hash one regular file with bounded reads; cancellation raises InterruptedError."""
    return _hash_file(path, chunk_size, cancelled)[0]


def identify_media(path: MediaPath, duration: float, *,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   cancelled: Optional[Callable[[], bool]] = None) -> MediaIdentity:
    """Hash media bytes and pair them with a declared, structurally valid duration.

    The caller must obtain duration from its decoder or another trusted source.
    This function does not decode media or establish the duration itself.
    """
    clip = Path(path).name
    # Check caller-controlled fields before spending time hashing a large file.
    require_valid_document({"kind": "temporal", "schema_version": 1,
                            "clip": clip, "duration": duration, "segments": []})
    digest, size = _hash_file(path, chunk_size, cancelled)
    return MediaIdentity(clip, duration, size, digest)


def bind_document(doc: Dict[str, Any], identity: MediaIdentity, *,
                  allow_rename: bool = False) -> Dict[str, Any]:
    """Return a validated copy bound to full-file identity, never mutate inputs.

    Version 1 has only weak matching data: basename, declared duration within
    one millisecond and optional byte size must match. Explicit migration adds
    identity and resets every segment to unreviewed. Version 2 checks hash and
    size, allows the same duration tolerance and preserves review decisions.
    Only a matching v2 hash permits explicit filename retargeting. Stored times,
    metadata, IDs and the document's declared duration are never rewritten.
    """
    require_valid_document(doc)
    if not isinstance(identity, MediaIdentity):
        raise ValueError("identity must be a MediaIdentity")
    if not isinstance(allow_rename, bool):
        raise ValueError("allow_rename must be a boolean")
    require_valid_document({"kind": "temporal", "schema_version": 2,
                            "clip": identity.clip, "duration": identity.duration,
                            "media_size": identity.media_size,
                            "media_sha256": identity.media_sha256, "segments": []})
    if abs(doc["duration"] - identity.duration) > DURATION_TOLERANCE_SECONDS:
        raise ValueError("Annotation duration differs from declared media duration by more than 1 millisecond")
    if any(segment["end"] > identity.duration for segment in doc["segments"]):
        raise ValueError("Annotation segment.end exceeds declared media duration; no timestamps were clipped")
    if "media_size" in doc and doc["media_size"] != identity.media_size:
        raise ValueError("Annotation media_size does not match the actual file size")
    if doc["schema_version"] == 1:
        if doc["clip"] != identity.clip:
            raise ValueError("Version 1 filename must match; weak identity cannot authorize a rename")
    else:
        if doc["media_sha256"] != identity.media_sha256:
            raise ValueError("Annotation media_sha256 does not match the full-file SHA256")
        if doc["clip"] != identity.clip and not allow_rename:
            raise ValueError("Media filename differs; use allow_rename only after a matching full-file hash")
    # The schema permits only JSON values. A JSON copy preserves those values
    # without deepcopy's lower effective recursion limit for nested mappings.
    try:
        result = json.loads(json.dumps(doc, ensure_ascii=False, allow_nan=False))
    except (RecursionError, ValueError) as exc:
        raise ValueError("Temporal JSON could not be copied safely; check values or reduce excessive nesting") from exc
    if doc["schema_version"] == 1:
        result["schema_version"] = 2
        result["media_size"] = identity.media_size
        result["media_sha256"] = identity.media_sha256
        for segment in result["segments"]:
            segment["review_status"] = "unreviewed"
    elif allow_rename:
        result["clip"] = identity.clip
    require_valid_document(result)
    return result
