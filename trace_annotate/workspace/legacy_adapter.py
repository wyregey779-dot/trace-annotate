"""Temporal adapters with persisted source metadata and explicit projection loss.

Temporal documents describe segments only. They cannot carry workspace events,
frame anchors, command history or independent review evidence. The full task
snapshot remains authoritative for those fields.
"""
from __future__ import annotations

import copy
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List

from ..temporal.schema import validate_document, require_valid_document
from . import schema as S
from .errors import SchemaInvalid


def _import_end(doc, media, end):
    """Map only a hash-bound, six-decimal representation of the exact endpoint.

    This is not general clipping: internal boundaries, v1/unbound documents and
    errors larger than half a microsecond are never adjusted.
    """
    value = S.rational(end)
    actual = S.rational_value(media["duration"])
    delta = S.rational_value(value) - actual
    if delta <= 0:
        return value
    source = media.get("source", {})
    if (doc.get("schema_version") == 2 and doc.get("media_sha256") == source.get("sha256")
            and str(doc.get("media_size")) == str(source.get("size_bytes"))
            and end == doc.get("duration")
            and S.rational(doc["duration"]) == S.rational(round(float(actual), 6))
            and delta <= Fraction(1, 2_000_000)):
        return S.rational(actual)
    raise SchemaInvalid("TEMPORAL_ENDPOINT_MISMATCH", "Temporal end exceeds the admitted media endpoint",
                        details={"declared_end": value, "media_end": S.rational(actual),
                                 "policy": "no_general_clipping"})


def temporal_import_payload(doc, media, *, actor_id="user"):
    """Build the atomic import payload, including the original source envelope.

    The store persists ``legacy_document_metadata`` and the envelope under
    ``extensions.temporal_source`` in the same revision as the annotations.
    Both locations already participate in the task digest. No caller should
    reconstruct metadata after saving the annotations.
    """
    annotations = import_temporal_document(doc, media, actor_id=actor_id)
    envelope = {key: copy.deepcopy(doc[key]) for key in
                ("kind", "schema_version", "clip", "duration", "media_size", "media_sha256")
                if key in doc}
    envelope["metadata_present"] = "metadata" in doc
    envelope["media_size_present"] = "media_size" in doc
    envelope["source_reviews"] = [
        {"annotation_id": annotation["id"], "legacy_id": segment["id"],
         "review_status": segment["review_status"]}
        for annotation, segment in zip(annotations, doc["segments"])
    ]
    adjustments = [{"annotation_id": annotation["id"], "field": "end",
                    "original": segment["end"], "mapped_seconds": annotation["end"]["seconds"],
                    "rule": "verified-source-six-decimal-endpoint-v1"}
                   for annotation, segment in zip(annotations, doc["segments"])
                   if annotation["end"]["seconds"] != S.rational(segment["end"])]
    if adjustments:
        envelope["endpoint_adjustments"] = adjustments
    payload = {"annotations": annotations,
               "legacy_document_metadata": copy.deepcopy(doc.get("metadata", {})),
               "legacy_document_envelope": envelope}
    validate_temporal_import_payload(payload, media)
    return payload


def validate_temporal_import_payload(payload, media):
    """Validate direct command callers as strictly as the HTTP document route."""
    def invalid():
        raise SchemaInvalid("INVALID_TEMPORAL_ENVELOPE", "Invalid persisted temporal source envelope")
    if (not isinstance(payload, dict) or set(payload) != {
            "annotations", "legacy_document_metadata", "legacy_document_envelope"}):
        invalid()
    annotations, metadata, envelope = [payload[key] for key in
        ("annotations", "legacy_document_metadata", "legacy_document_envelope")]
    required = {"kind", "schema_version", "clip", "duration", "metadata_present",
                "media_size_present", "source_reviews"}
    if (not isinstance(annotations, list) or not isinstance(metadata, dict)
            or not isinstance(envelope, dict) or not required <= set(envelope)
            or set(envelope) - required - {"media_size", "media_sha256", "endpoint_adjustments"}
            or type(envelope["metadata_present"]) is not bool
            or type(envelope["media_size_present"]) is not bool
            or envelope["media_size_present"] != ("media_size" in envelope)
            or (not envelope["metadata_present"] and metadata)):
        invalid()
    reviews = envelope["source_reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(annotations):
        invalid()
    doc = {key: copy.deepcopy(envelope[key]) for key in
           ("kind", "schema_version", "clip", "duration", "media_size", "media_sha256")
           if key in envelope}
    if envelope["metadata_present"]:
        doc["metadata"] = copy.deepcopy(metadata)
    doc["segments"] = []
    adjustments = envelope.get("endpoint_adjustments", [])
    if not isinstance(adjustments, list):
        invalid()
    adjustment_map = {}
    for adjustment in adjustments:
        if (not isinstance(adjustment, dict) or set(adjustment) != {"annotation_id", "field", "original", "mapped_seconds", "rule"}
                or adjustment["field"] != "end" or adjustment["rule"] != "verified-source-six-decimal-endpoint-v1"
                or not isinstance(adjustment["annotation_id"], str) or adjustment["annotation_id"] in adjustment_map):
            invalid()
        adjustment_map[adjustment["annotation_id"]] = adjustment
    for annotation, review in zip(annotations, reviews):
        if (not isinstance(annotation, dict) or annotation.get("type") != "segment"
                or not isinstance(review, dict)
                or set(review) != {"annotation_id", "legacy_id", "review_status"}
                or review["annotation_id"] != annotation.get("id")
                or review["legacy_id"] != (annotation.get("legacy_id") or annotation.get("id"))
                or review["review_status"] != annotation.get("review_status")):
            invalid()
        try:
            adjustment = adjustment_map.pop(annotation["id"], None)
            end = float(S.rational_value(annotation["end"]["seconds"]))
            if adjustment:
                end = adjustment["original"]
                if (adjustment["mapped_seconds"] != annotation["end"]["seconds"]
                        or _import_end(doc, media, end) != annotation["end"]["seconds"]
                        or S.rational(end) == annotation["end"]["seconds"]):
                    invalid()
            doc["segments"].append({"id": review["legacy_id"],
                "start": float(S.rational_value(annotation["start"]["seconds"])),
                "end": end,
                **{key: copy.deepcopy(annotation[key]) for key in
                   ("label", "text", "tags", "track", "review_status")}})
        except (KeyError, TypeError):
            invalid()
    if adjustment_map or validate_document(doc):
        invalid()
    if doc["schema_version"] == 1:
        raise SchemaInvalid("MEDIA_BINDING_REQUIRED", "Bind legacy v1 to the verified source in the classic tool before importing as v2")
    source = media.get("source", {})
    if doc["schema_version"] == 2:
        if (doc["media_sha256"] != source.get("sha256")
                or str(doc["media_size"]) != str(source.get("size_bytes"))):
            raise SchemaInvalid("MEDIA_IDENTITY_MISMATCH", "Temporal source envelope does not match admitted media")


def _source_envelope(task):
    extensions = task.get("extensions", {})
    source = extensions.get("temporal_source", {}) if isinstance(extensions, dict) else {}
    return source if isinstance(source, dict) else {}


def temporal_projection_losses(task, media):
    """Return ASCII HTTP-header-safe disclosures for the actual projection."""
    losses = ["Temporal v2 segment projection; excludes workspace revisions, receipts, "
              "template, evidence, content-bound review records and provenance; not a complete workspace backup."]
    omitted = Counter(ann.get("type", "unknown") for ann in task.get("annotations", [])
                      if ann.get("type") != "segment")
    if omitted:
        # Validated annotation kinds are a finite ASCII vocabulary. Do not put
        # arbitrary labels, metadata or track names into an HTTP header.
        losses.append("Omitted annotation types: " + ", ".join(
            f"{kind}={omitted[kind]}" for kind in ("event", "whole_clip") if omitted[kind]) + ".")
    segments = [ann for ann in task.get("annotations", []) if ann.get("type") == "segment"]
    anchors = sum("anchor" in ann.get(key, {}) for ann in segments for key in ("start", "end"))
    if anchors:
        losses.append(f"Omitted frame anchors: {anchors}.")
    rounded = 0
    for ann in segments:
        for key in ("start", "end"):
            value = S.rational_value(ann[key]["seconds"])
            if S.rational_value(S.rational(float(value))) != value:
                rounded += 1
    if rounded:
        losses.append(f"Rational times rounded to JSON decimal numbers: {rounded}.")
    source = _source_envelope(task)
    if source.get("endpoint_adjustments"):
        losses.append(f"Six-decimal legacy endpoints mapped to verified exact media end: {len(source['endpoint_adjustments'])}; "
                      "original numbers and mapping rule remain in the workspace source envelope.")
    if source.get("schema_version") == 1:
        losses.append("Original temporal v1 projected as v2 with admitted media identity.")
    if any(item.get("review_status") == "reviewed" for item in source.get("source_reviews", [])):
        losses.append("Historical review provenance remains in the workspace source envelope; "
                      "exported review statuses are current states, not a new independent approval.")
    declared = source.get("duration")
    if declared is not None and any(
            S.rational_value(ann["end"]["seconds"]) > S.rational_value(S.rational(declared))
            for ann in segments):
        losses.append("Original declared duration extended to admitted duration for current segments.")
    return losses


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
        end_rat = _import_end(doc, media, seg["end"])

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

    source = _source_envelope(task)
    clip_name = source.get("clip") or Path(media.get("source", {}).get("display_name") or "video.mp4").name
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

    # Preserve the imported declaration when it still contains every segment.
    # A later edit extending beyond it uses the admitted duration, with the
    # corresponding loss disclosure instead of silently exporting invalid JSON.
    declared_duration = source.get("duration")
    if declared_duration is not None and all(
            S.rational_value(ann["end"]["seconds"]) <= S.rational_value(S.rational(declared_duration))
            for ann in task.get("annotations", []) if ann.get("type") == "segment"):
        duration = copy.deepcopy(declared_duration)

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

    if "legacy_document_metadata" in task:
        metadata = task["legacy_document_metadata"]
        if not isinstance(metadata, dict):
            raise SchemaInvalid("INVALID_TEMPORAL_METADATA", "Stored temporal metadata is not an object")
        if source.get("metadata_present", True) or metadata:
            doc["metadata"] = copy.deepcopy(metadata)
    elif "metadata" in task and isinstance(task["metadata"], dict):
        # Compatibility for callers predating persisted workspace imports.
        doc["metadata"] = copy.deepcopy(task["metadata"])

    if schema_version == 1 and source.get("media_size_present"):
        doc["media_size"] = int(media.get("source", {}).get("size_bytes") or media.get("size_bytes", 0))

    try:
        require_valid_document(doc)
    except ValueError as exc:
        raise SchemaInvalid("TEMPORAL_PROJECTION_UNSUPPORTED",
                            "Current task cannot be represented by the temporal format; use the complete task snapshot",
                            details={"violations": validate_document(doc)}) from exc
    return doc
