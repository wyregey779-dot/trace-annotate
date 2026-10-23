"""Validate general-video documents without coercing user data.

Intervals are half-open [start, end), measured in seconds. Overlap is allowed,
including on the same track. Unknown fields are rejected; extensions belong in
the document's optional JSON metadata object. No video decoding is performed.
"""
from __future__ import annotations

import math
import ntpath
import re
from typing import Any, Dict, List, Set


DOCUMENT_FIELDS = {
    "kind", "schema_version", "clip", "duration", "media_size", "segments", "metadata",
}
V2_DOCUMENT_FIELDS = DOCUMENT_FIELDS | {"media_sha256"}
REQUIRED_DOCUMENT_FIELDS = {"kind", "schema_version", "clip", "duration", "segments"}
V2_REQUIRED_DOCUMENT_FIELDS = REQUIRED_DOCUMENT_FIELDS | {"media_size", "media_sha256"}
SEGMENT_FIELDS = {
    "id", "start", "end", "label", "text", "tags", "track", "review_status",
}
REVIEW_STATUSES = {"unreviewed", "reviewed"}
MAX_SAFE_INTEGER = 9007199254740991


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and abs(value) <= MAX_SAFE_INTEGER
    except (OverflowError, ValueError):
        return False


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and _valid_unicode(value)


def _valid_unicode(value: str) -> bool:
    # Escaped JSON surrogate pairs are decoded to a normal non-BMP codepoint.
    # Direct Python strings with surrogate code units are not valid UTF-8 text.
    return not any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _check_fields(value: Dict[str, Any], allowed: Set[str], required: Set[str],
                  location: str, errors: List[str]) -> None:
    for key in value:
        if not isinstance(key, str) or key not in allowed:
            errors.append(f"{location}: unknown field {key!r}; use document metadata for extensions")
    for key in sorted(required):
        if key not in value:
            errors.append(f"{location}.{key} is required")


def _check_json(value: Any, location: str, active: Set[int], errors: List[str]) -> None:
    """Reject values JSON would coerce, nonfinite values and container cycles."""
    if isinstance(value, str):
        if not _valid_unicode(value):
            errors.append(f"{location} must contain valid Unicode without surrogate code units")
        return
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not _finite_number(value):
            errors.append(f"{location} must contain finite JSON numbers within +/-9007199254740991")
        return
    if not isinstance(value, (list, dict)):
        errors.append(f"{location} must contain only JSON values")
        return
    identity = id(value)
    if identity in active:
        errors.append(f"{location} contains a circular reference")
        return
    active.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    errors.append(f"{location} must have string object keys")
                elif not _valid_unicode(key):
                    errors.append(f"{location} must have valid Unicode object keys without surrogate code units")
                else:
                    _check_json(item, f"{location}[{key!r}]", active, errors)
        else:
            for index, item in enumerate(value):
                _check_json(item, f"{location}[{index}]", active, errors)
    finally:
        active.remove(identity)


def validate_document(doc: Any) -> List[str]:
    """Return structural violations; ordinary malformed input never raises.

    This checks annotation structure, not label correctness, video identity or
    whether the declared duration matches an actual media file.
    """
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    errors: List[str] = []
    version = doc.get("schema_version")
    is_v2 = isinstance(version, int) and not isinstance(version, bool) and version == 2
    _check_fields(doc, V2_DOCUMENT_FIELDS if is_v2 else DOCUMENT_FIELDS,
                  V2_REQUIRED_DOCUMENT_FIELDS if is_v2 else REQUIRED_DOCUMENT_FIELDS,
                  "document", errors)
    if doc.get("kind") != "temporal":
        errors.append("document.kind must be 'temporal'")
    if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2}:
        errors.append("document.schema_version must be integer 1 or 2")
    if is_v2:
        if "media_size" not in doc or "media_sha256" not in doc:
            errors.append("document.schema_version 2 requires media_size and media_sha256")
        digest = doc.get("media_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append("document.media_sha256 must be exactly 64 lowercase hexadecimal characters")

    clip = doc.get("clip")
    if not _nonempty_string(clip):
        errors.append("document.clip must be a non-empty filename with valid Unicode")
    elif (clip in {".", ".."} or "/" in clip or "\\" in clip
          or ":" in clip or ntpath.splitdrive(clip)[0]
          or any(ord(character) < 32 for character in clip)):
        errors.append("document.clip must be a basename without paths, drive names or control characters")

    duration = doc.get("duration")
    duration_valid = _finite_number(duration) and duration > 0
    if not duration_valid:
        errors.append("document.duration must be a finite positive number <= 9007199254740991, not a boolean")
    if "media_size" in doc:
        size = doc["media_size"]
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_SAFE_INTEGER:
            errors.append("document.media_size must be an integer between 0 and 9007199254740991")
    if "metadata" in doc:
        if not isinstance(doc["metadata"], dict):
            errors.append("document.metadata must be a JSON object")
        else:
            try:
                _check_json(doc["metadata"], "document.metadata", set(), errors)
            except RecursionError:
                errors.append("document.metadata nesting is too deep")

    segments = doc.get("segments")
    if not isinstance(segments, list):
        errors.append("document.segments must be an array")
        return errors
    seen_ids: Set[str] = set()
    for index, segment in enumerate(segments):
        location = f"segments[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{location} must be a JSON object")
            continue
        _check_fields(segment, SEGMENT_FIELDS, SEGMENT_FIELDS, location, errors)
        for field in ("id", "label", "track"):
            if not _nonempty_string(segment.get(field)):
                errors.append(f"{location}.{field} must be a non-empty string with valid Unicode")
        segment_id = segment.get("id")
        if _nonempty_string(segment_id):
            if segment_id in seen_ids:
                errors.append(f"{location}.id must be unique: {segment_id!r}")
            seen_ids.add(segment_id)
        if not isinstance(segment.get("text"), str) or not _valid_unicode(segment["text"]):
            errors.append(f"{location}.text must be a string with valid Unicode")
        tags = segment.get("tags")
        if not isinstance(tags, list) or any(not _nonempty_string(tag) for tag in tags):
            errors.append(f"{location}.tags must be an array of non-empty strings with valid Unicode")
        status = segment.get("review_status")
        if not isinstance(status, str) or status not in REVIEW_STATUSES:
            errors.append(f"{location}.review_status must be 'unreviewed' or 'reviewed'")
        start, end = segment.get("start"), segment.get("end")
        if not _finite_number(start) or not _finite_number(end):
            errors.append(f"{location}.start/end must be finite numbers within +/-9007199254740991, not booleans")
        else:
            if start < 0 or end <= start:
                errors.append(f"{location} requires 0 <= start < end")
            if duration_valid and end > duration:
                errors.append(f"{location}.end must not exceed document.duration")
    return errors


def require_valid_document(doc: Any) -> None:
    """Raise ValueError with all supported structural errors, without mutation."""
    errors = validate_document(doc)
    if errors:
        raise ValueError("Invalid temporal annotation:\n" + "\n".join(errors))
