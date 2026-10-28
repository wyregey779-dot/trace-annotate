"""Read-only task reports and explicitly lossy export projections.

Coverage describes saved intervals on the admitted video's active clock. It
does not infer Idle, inspect image meaning, or create a review verdict. Exact
rational arithmetic is retained until the documented SRT millisecond boundary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
from fractions import Fraction
import hashlib
import io
import json
import re

from ..temporal.formats import _csv_text, _plain_text, _timestamp
from . import schema as S
from .errors import NotFound, SchemaInvalid, WorkspaceError

# These are analysis budgets, not restrictions on valid stored annotations or
# raw exports. Input rational limits cannot be reused here: subtracting two
# valid 40-digit fractions can itself require an 80-digit denominator.
MAX_STATISTICS_SEGMENTS = 20_000
MAX_STATISTICS_BITS = 4096


def _track_id(name):
    # A short selector for exact stored names, including names whose encoded
    # query would exceed HTTP limits. Do not normalize Unicode or case here.
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class _StatisticsBudgetExceeded(Exception):
    pass


def _bounded_statistic(value):
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > MAX_STATISTICS_BITS:
        raise _StatisticsBudgetExceeded
    return value


def _derived_rational(value):
    value = _bounded_statistic(value)
    return {"n": str(value.numerator), "d": str(value.denominator)}


def _duration(intervals):
    total = Fraction(0)
    for start, end in intervals:
        # Every intermediate sum is bounded. Each added difference derives
        # from two bounded input rationals; no unbounded big-int accumulation.
        total = _bounded_statistic(total + (end - start))
    return _derived_rational(total)


def _unavailable_statistics():
    return {"statistics_status": "budget_exceeded", "covered_seconds": None,
            "uncovered_seconds": None, "overlap_seconds": None, "max_concurrent_segments": None,
            "covered_intervals": None, "gaps": None, "overlaps": None}


def _read_subject(store, task_id):
    S.require_id(task_id, "task_id")
    # The caller holds one store transaction across subject and digest reads.
    task = store.get_task(task_id)
    media = store.get_media(task["media_id"])
    templates = store.list_templates()
    template = next((item for item in templates if (item["id"], item["version"]) ==
                     (task["template_id"], task["template_version"])), None)
    if template is None:
        raise NotFound("TEMPLATE_NOT_FOUND", "Task template is missing")
    return task, media, template


def _binding(task, media):
    return {"task_id": task["id"], "revision": task["revision"],
            "content_sha256": task["content_sha256"], "media_id": media["id"],
            "source_sha256": media["source"]["sha256"],
            "playback_sha256": media["playback"]["sha256"]}


def _interval(start, end):
    return {"start": _derived_rational(start), "end": _derived_rational(end),
            "duration": _derived_rational(end - start)}


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _coverage(intervals, first, endpoint):
    if len(intervals) > MAX_STATISTICS_SEGMENTS:
        raise _StatisticsBudgetExceeded
    # Clip only the statistical intersection, never the saved annotations.
    events = defaultdict(int)
    for start, end in intervals:
        start, end = max(first, start), min(endpoint, end)
        if start < end:
            events[start] += 1
            events[end] -= 1
    covered, overlaps = [], []
    previous, count, maximum = first, 0, 0
    for point in sorted(events):
        if point > previous:
            if count:
                covered.append((previous, point))
            if count > 1:
                overlaps.append((previous, point))
        count += events[point]
        maximum = max(maximum, count)
        previous = point
    covered, overlaps = _merge(covered), _merge(overlaps)
    gaps, previous = [], first
    for start, end in covered:
        if previous < start:
            gaps.append((previous, start))
        previous = end
    if previous < endpoint:
        gaps.append((previous, endpoint))
    return {"statistics_status": "complete", "covered_seconds": _duration(covered),
            "uncovered_seconds": _duration(gaps), "overlap_seconds": _duration(overlaps),
            "max_concurrent_segments": maximum,
            "covered_intervals": [_interval(*item) for item in covered],
            "gaps": [_interval(*item) for item in gaps],
            "overlaps": [_interval(*item) for item in overlaps]}


def _report(store, task, media, template):
    first, endpoint = S.rational_value(media["video_start"]), S.rational_value(media["video_end"])
    if not 0 <= first < endpoint <= S.rational_value(media["duration"]):
        raise SchemaInvalid("INVALID_MEDIA_TIMELINE", "Invalid video active interval")
    errors, warnings, valid = [], [], []
    if store.task_digest(task) != task["content_sha256"]:
        errors.append({"code": "CONTENT_DIGEST_MISMATCH", "annotation_id": None})
    seen = set()
    for original in task["annotations"]:
        annotation = copy.deepcopy(original)
        ident = annotation.get("id") if isinstance(annotation, dict) else None
        try:
            if not isinstance(annotation, dict):
                raise SchemaInvalid("SCHEMA_INVALID", "Annotation is not an object")
            store._validate_annotation(annotation, media, template)
            if ident in seen:
                raise SchemaInvalid("DUPLICATE_ANNOTATION_ID", "Annotation ID is duplicated")
            seen.add(ident)
            valid.append(annotation)
        except WorkspaceError as exc:
            errors.append({"code": exc.code, "annotation_id": ident if isinstance(ident, str) else None})
    kinds = Counter(item["type"] for item in valid)
    tags = Counter(tag for item in valid for tag in item["tags"])
    grouped = defaultdict(list)
    for item in valid:
        grouped[item["track"]].append(item)
    tracks = []
    for name in sorted(grouped):
        selected = grouped[name]
        segments = [item for item in selected if item["type"] == "segment"]
        intervals = [(S.rational_value(item["start"]["seconds"]), S.rational_value(item["end"]["seconds"]))
                     for item in segments]
        for item, (start, end) in zip(segments, intervals):
            if start < first or end > endpoint:
                warnings.append({"code": "SEGMENT_EXTENDS_OUTSIDE_VIDEO_ACTIVE", "annotation_id": item["id"]})
        try:
            if kinds["segment"] > MAX_STATISTICS_SEGMENTS:
                raise _StatisticsBudgetExceeded
            stats = _coverage(intervals, first, endpoint)
        except _StatisticsBudgetExceeded:
            stats = _unavailable_statistics()
            warnings.append({"code": "STATISTICS_BUDGET_EXCEEDED", "track": name})
        tracks.append({"track": name, "track_id": _track_id(name), "segment_count": len(segments),
                       "event_count": sum(item["type"] == "event" for item in selected),
                       "whole_clip_count": sum(item["type"] == "whole_clip" for item in selected), **stats})
    if template.get("coverage_policy") != "sparse":
        # A structural report must not silently certify an unimplemented
        # coverage policy as a release/review gate.
        warnings.append({"code": "COVERAGE_POLICY_NOT_CERTIFIED"})
    return {"kind": "traceannotate.task-structural-report", "version": 1,
            "binding": _binding(task, media), "scope": "structure_and_video_interval_statistics_only",
            "semantic_review": "not_performed", "coverage_policy": template.get("coverage_policy"),
            "video_active": _interval(first, endpoint),
            "counts": {"annotations": len(task["annotations"]), "valid_annotations": len(valid),
                       "segments": kinds["segment"], "events": kinds["event"], "whole_clip": kinds["whole_clip"],
                       "tag_occurrences": sum(tags.values()), "unique_tags": len(tags)},
            "tags": [{"tag": tag, "count": count} for tag, count in sorted(tags.items())],
            "checks": {"status": "failed" if errors else "passed", "errors": errors},
            "statistics_status": "budget_exceeded" if any(
                track["statistics_status"] == "budget_exceeded" for track in tracks) else "complete",
            "statistics_budget": {"max_segments": MAX_STATISTICS_SEGMENTS,
                                  "max_rational_bits": MAX_STATISTICS_BITS},
            "tracks": tracks, "warnings": warnings,
            "limitations": ["No image meaning or label correctness was reviewed.",
                            "Gaps mean unannotated, not Idle; sparse overlaps are not errors.",
                            "Coverage is per track on video_start/video_end only; events and whole-clip labels do not cover intervals.",
                            "Derived statistics exceeding the declared analysis budget are unavailable, never rounded or replaced with zero; original JSON/CSV exports remain available.",
                            "This report does not create an approval or certify complete-partition delivery."]}


def task_report(store, task_id):
    with store.transaction():
        task, media, template = _read_subject(store, task_id)
        return _report(store, task, media, template)


CSV_FIELDS = ["task_id", "revision", "content_sha256", "media_id", "annotation_id", "type",
              "label", "text", "track", "tags_json", "start_n", "start_d", "end_n", "end_d", "at_n", "at_d",
              "annotation_json"]


def _csv(task):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writeheader()
    for annotation in task["annotations"]:
        row = {"task_id": task["id"], "revision": task["revision"], "content_sha256": task["content_sha256"],
               "media_id": task["media_id"], "annotation_id": annotation["id"], "type": annotation["type"],
               "label": annotation["label"], "text": annotation["text"], "track": annotation["track"],
               "tags_json": json.dumps(annotation["tags"], ensure_ascii=False, separators=(",", ":")),
               # Preserve the original values alongside spreadsheet-safe text
               # projections. JSON starts with '{', so it is not a formula.
               "annotation_json": S.canonical_json(annotation).decode("utf-8")}
        for key in ("start", "end", "at"):
            for part in ("n", "d"):
                row[f"{key}_{part}"] = annotation.get(key, {}).get("seconds", {}).get(part, "")
        writer.writerow({key: _csv_text(value) if isinstance(value, str) else value for key, value in row.items()})
    return stream.getvalue()


def _milliseconds(value):
    numerator, denominator = value.numerator, value.denominator
    whole, remainder = divmod(numerator * 1000, denominator)
    return whole + int(remainder * 2 >= denominator)


def _srt(task, track, track_id=None):
    segments = [item for item in task["annotations"] if item["type"] == "segment"]
    tracks = {item["track"] for item in segments}
    if track_id is not None:
        matches = [name for name in tracks if _track_id(name) == track_id]
        if not matches:
            raise SchemaInvalid("TRACK_NOT_FOUND", "Selected segment track does not exist")
        if len(matches) != 1:
            raise SchemaInvalid("TRACK_ID_AMBIGUOUS", "Track ID does not uniquely identify a segment track")
        track = matches[0]
    if track is None and len(tracks) > 1:
        raise SchemaInvalid("TRACK_REQUIRED", "Select one track for SRT export")
    if track is not None and (not isinstance(track, str) or track not in tracks):
        raise SchemaInvalid("TRACK_NOT_FOUND", "Selected segment track does not exist")
    segments = [item for item in segments if track is None or item["track"] == track]
    segments.sort(key=lambda item: (S.rational_value(item["start"]["seconds"]),
                                    S.rational_value(item["end"]["seconds"]), item["id"]))
    cues = []
    for number, item in enumerate(segments, 1):
        start, end = [_milliseconds(S.rational_value(item[key]["seconds"])) for key in ("start", "end")]
        if end <= start:
            raise SchemaInvalid("SUBTITLE_PRECISION_LOSS", "A segment collapses at millisecond precision")
        try:
            text = _plain_text(item["text"] if item["text"] else item["label"], "srt", "annotation")
        except ValueError as exc:
            raise SchemaInvalid("SUBTITLE_TEXT_UNSUPPORTED", "SRT cannot represent this annotation text safely") from exc
        cues.append(f"{number}\n{_timestamp(start, 'srt')} --> {_timestamp(end, 'srt')}\n{text}")
    return "\n\n".join(cues) + ("\n" if cues else "")


def export_task(store, task_id, format, *, track=None, track_id=None):
    if format not in {"json", "csv", "srt"}:
        raise SchemaInvalid("UNSUPPORTED_EXPORT_FORMAT", "Supported task exports are json, csv and srt")
    if track is not None and track_id is not None:
        raise SchemaInvalid("TRACK_SELECTION_CONFLICT", "Select a track by either name or ID, not both")
    if (track is not None or track_id is not None) and format != "srt":
        raise SchemaInvalid("UNSUPPORTED_EXPORT_OPTION", "Track selection applies only to SRT")
    if track_id is not None and (not isinstance(track_id, str) or re.fullmatch(r"[0-9a-f]{64}", track_id) is None):
        raise SchemaInvalid("INVALID_TRACK_ID", "Track ID must be a lowercase SHA256 hex digest")
    with store.transaction():
        task, media, template = _read_subject(store, task_id)
        report = _report(store, task, media, template)
        if report["checks"]["status"] != "passed":
            raise SchemaInvalid("TASK_REPORT_FAILED", "Task has structural errors; repair it before export")
        binding = _binding(task, media)
        if format == "json":
            data = S.canonical_json({"kind": "traceannotate.task-snapshot", "version": 1, "binding": binding,
                                    "export_scope": "development_task_snapshot_without_media_or_portable_import",
                                    "task": task, "media": media, "template": template, "report": report})
            content_type = "application/json; charset=utf-8"
            losses = "No media bytes; not a portable import format; no semantic review."
        elif format == "csv":
            data = _csv(task).encode("utf-8-sig")
            content_type = "text/csv; charset=utf-8"
            losses = "Spreadsheet text is formula-escaped; annotation_json preserves originals; no media or review history."
        else:
            data = _srt(task, track, track_id).encode("utf-8")
            content_type = "application/x-subrip; charset=utf-8"
            losses = ("One segment track; round-half-up milliseconds; text else label; excludes events, whole-clip labels, "
                      "IDs, tags, anchors, metadata and review history; overlapping cues retained.")
    return {"data": data, "content_type": content_type, "filename": f"TraceAnnotate-{task_id}-r{binding['revision']}.{format}",
            "binding": binding, "losses": losses}
