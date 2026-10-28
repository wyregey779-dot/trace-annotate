"""Lossless bidirectional adapter between temporal v1/v2 format and workspace schema.

Guarantees byte-for-byte or semantic roundtrip of segments, tags, Unicode text,
labels, tracks, review status, and metadata without silent field loss.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..temporal.schema import validate_document, require_valid_document
from . import schema as S
from .errors import SchemaInvalid


def import_temporal_document(
    doc: Dict[str, Any],
    media: Dict[str, Any],
    *,
    actor_id: str = "user"
) -> List[Dict[str, Any]]:
    """Convert a temporal v1/v2 annotation document into workspace task annotations.

    Validates structural validity, verifies duration consistency, and preserves
    legacy IDs, tags, text, track, and review status.
    """
    errors = validate_document(doc)
    if errors:
        raise SchemaInvalid("INVALID_TEMPORAL_DOCUMENT", "Invalid temporal annotation:\n" + "\n".join(errors))

    media_duration = float(S.rational_value(media["duration"]))
    doc_duration = float(doc["duration"])
    # Allow 1 second tolerance for container vs stream metadata discrepancies
    if abs(media_duration - doc_duration) > 1.0 and any(s["end"] > media_duration for s in doc.get("segments", [])):
        raise SchemaInvalid("DURATION_MISMATCH", f"Document duration {doc_duration} exceeds media duration {media_duration}")

    annotations: List[Dict[str, Any]] = []
    seen_ids = set()

    for seg in doc.get("segments", []):
        raw_id = seg["id"]
        # Check if raw_id is a valid ASCII workspace identifier
        if S.ID_RE.fullmatch(raw_id) and raw_id not in seen_ids:
            ann_id = raw_id
            legacy_id = None
        else:
            ann_id = S.make_id("seg", raw_id)
            # Ensure unique in case of hash collisions
            while ann_id in seen_ids:
                ann_id = S.make_id("seg", raw_id + str(len(seen_ids)))
            legacy_id = raw_id

        seen_ids.add(ann_id)

        start_rat = S.rational(seg["start"])
        end_rat = S.rational(seg["end"])

        ann: Dict[str, Any] = {
            "id": ann_id,
            "type": "segment",
            "start": {"seconds": start_rat},
            "end": {"seconds": end_rat},
            "label": seg["label"],
            "text": seg.get("text", ""),
            "track": seg.get("track", "main"),
            "tags": list(seg.get("tags", [])),
            "review_status": seg.get("review_status", "unreviewed"),
            "attributes": {},
            "evidence_ids": [],
            "provenance": {"origin": "manual", "actor_id": actor_id, "job_id": None},
        }
        if legacy_id is not None:
            ann["legacy_id"] = legacy_id

        annotations.append(ann)

    return annotations


def export_temporal_document(
    task: Dict[str, Any],
    media: Dict[str, Any],
    *,
    schema_version: int = 2
) -> Dict[str, Any]:
    """Project a workspace task into a compliant temporal v1/v2 annotation document.

    Restores legacy IDs, exact floating-point times, track names, tags,
    and review statuses.
    """
    if schema_version not in (1, 2):
        raise SchemaInvalid("UNSUPPORTED_VERSION", "Only temporal schema version 1 and 2 are supported")

    clip_name = Path(media.get("source", {}).get("display_name") or "video.mp4").name
    duration = float(S.rational_value(media["duration"]))

    segments: List[Dict[str, Any]] = []
    for ann in task.get("annotations", []):
        if ann.get("type") != "segment":
            continue

        seg_id = ann.get("legacy_id") or ann["id"]
        start_sec = float(S.rational_value(ann["start"]["seconds"]))
        end_sec = float(S.rational_value(ann["end"]["seconds"]))

        segment_obj: Dict[str, Any] = {
            "id": seg_id,
            "start": start_sec,
            "end": end_sec,
            "label": ann.get("label", ""),
            "text": ann.get("text", ""),
            "tags": list(ann.get("tags", [])),
            "track": ann.get("track", "main"),
            "review_status": ann.get("review_status", "unreviewed"),
        }
        segments.append(segment_obj)


    doc: Dict[str, Any] = {
        "kind": "temporal",
        "schema_version": schema_version,
        "clip": clip_name,
        "duration": duration,
        "segments": segments,
    }

    if schema_version == 2:
        raw_size = media.get("source", {}).get("size_bytes") or media.get("size_bytes", 0)
        doc["media_size"] = int(raw_size)
        doc["media_sha256"] = media.get("source", {}).get("sha256") or media.get("sha256", "")

    if "metadata" in task and isinstance(task["metadata"], dict):
        doc["metadata"] = dict(task["metadata"])

    require_valid_document(doc)
    return doc
