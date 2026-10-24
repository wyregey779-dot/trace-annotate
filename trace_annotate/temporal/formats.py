"""Explicit projections of general temporal annotations.

JSON is the complete document format. CSV is a table projection and prefixes
formula-like text with an apostrophe for spreadsheet safety. Subtitles discard
labels when text is present, identifiers, tags, review state and metadata; their
timings are rounded half up to milliseconds. Neither projection is lossless.

Subtitle import accepts numbered SRT or unstyled WebVTT with no cue identifiers,
settings, notes, regions or styles. WebVTT accepts the six named text escapes;
SRT markup and ASS override/control syntax are rejected. Unsupported syntax
fails rather than being discarded.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from html import escape, unescape
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

from trace_annotate.temporal.schema import MAX_SAFE_INTEGER, require_valid_document


_CSV_FIELDS = ["id", "start", "end", "label", "text", "tags", "track", "review_status"]
_SRT_TIME = r"([0-9]{2,}):([0-5][0-9]):([0-5][0-9]),([0-9]{3})"
_VTT_TIME = r"(?:([0-9]{2,}):)?([0-5][0-9]):([0-5][0-9])\.([0-9]{3})"
_VTT_ESCAPE = re.compile(r"&(?:amp|lt|gt|nbsp|lrm|rlm);")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON value is not supported: {value}")


def _bounded_json_float(value):
    """Check the JSON lexeme before IEEE-754 rounding can hide an overflow."""
    try:
        number = Decimal(value)
        if not number.is_finite() or number.copy_abs() > MAX_SAFE_INTEGER:
            raise ValueError("JSON number must be within +/-9007199254740991 before float rounding")
    except InvalidOperation as exc:
        raise ValueError("JSON number must be finite and within +/-9007199254740991") from exc
    return float(value)


def load_document(path) -> Dict[str, Any]:
    """Load strict UTF-8 JSON and validate it; duplicates and nonfinite fail.

    Invalid JSON/documents raise ValueError. File access errors remain OSError;
    invalid UTF-8 remains UnicodeDecodeError. No input is rewritten.
    """
    try:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            document = json.load(stream, object_pairs_hook=_unique_json_object,
                                 parse_constant=_reject_json_constant,
                                 parse_float=_bounded_json_float)
    except RecursionError as exc:
        raise ValueError("Temporal JSON nesting is too deep") from exc
    require_valid_document(document)
    return document


def _csv_text(value: str) -> str:
    stripped = value.lstrip()
    leading = value[:len(value) - len(stripped)]
    if ((stripped and stripped[0] in "=+-@")
            or any(character in leading for character in "\t\r\n")):
        return "'" + value
    return value


def export_csv(doc: Dict[str, Any]) -> str:
    """Return segment rows; document metadata is omitted and formula text escaped."""
    require_valid_document(doc)
    output = io.StringIO(newline="")
    # Python 3.9's minimal quoting does not quote bare CR with an LF-only
    # terminator. Quote every field so multiline text round-trips consistently.
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for segment in doc["segments"]:
        row = dict(segment)
        row["tags"] = json.dumps(row["tags"], ensure_ascii=False, separators=(",", ":"))
        writer.writerow({key: _csv_text(value) if isinstance(value, str) else value
                         for key, value in row.items()})
    return output.getvalue()


def _format_name(value: str) -> str:
    if not isinstance(value, str) or value.lower() not in {"srt", "vtt", "webvtt"}:
        raise ValueError("subtitle format must be 'srt' or 'vtt'")
    return "vtt" if value.lower() == "webvtt" else value.lower()


def _milliseconds(seconds) -> int:
    # Work with the ratio so valid, very large times do not exceed Decimal's
    # default precision during quantize. All callers have non-negative times.
    numerator, denominator = Decimal(str(seconds)).as_integer_ratio()
    whole, remainder = divmod(numerator * 1000, denominator)
    return whole + int(remainder * 2 >= denominator)


def _timestamp(milliseconds: int, fmt: str) -> str:
    seconds, fraction = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    delimiter = "," if fmt == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{delimiter}{fraction:03d}"


def _plain_text(value: str, fmt: str, location: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if not value or any(not line.strip() for line in value.split("\n")):
        raise ValueError(f"{location}: empty or blank subtitle lines are unsupported; use JSON to retain them")
    if "-->" in value:
        raise ValueError(f"{location}: '-->' is not supported in subtitle text; use JSON to retain it")
    if any((ord(character) < 32 and character != "\n") or ord(character) == 127 for character in value):
        raise ValueError(f"{location}: subtitle text contains unsupported control characters")
    if fmt == "srt" and any(character in value for character in "<>"):
        raise ValueError(f"{location}: SRT markup/angle brackets are unsupported; use WebVTT or JSON")
    if fmt == "srt" and ("{\\" in value or any(token in value for token in (r"\N", r"\n", r"\h"))):
        raise ValueError(f"{location}: SRT ASS overrides/control escapes are unsupported; use WebVTT or JSON")
    return value


def export_subtitles(doc: Dict[str, Any], format: str = "srt",
                     track: Optional[str] = None) -> str:
    """Project one track as plain subtitles, with explicit millisecond rounding.

    A multi-track document requires a track argument. Overlapping cues are
    retained; exact simultaneous rendering depends on the destination player.
    The projection can round the last cue up beyond a sub-millisecond media
    duration; strict import never silently clips that cue to fit.
    """
    require_valid_document(doc)
    fmt = _format_name(format)
    tracks = {segment["track"] for segment in doc["segments"]}
    if track is None:
        if len(tracks) > 1:
            raise ValueError("Multiple tracks require an explicit track selection for subtitles")
    elif not isinstance(track, str) or not track.strip() or track not in tracks:
        raise ValueError("Selected subtitle track does not exist in this document")
    segments = [segment for segment in doc["segments"]
                if track is None or segment["track"] == track]
    segments.sort(key=lambda segment: (segment["start"], segment["end"], segment["id"]))
    cues = []
    for index, segment in enumerate(segments, 1):
        location = f"segment {segment['id']!r}"
        start, end = _milliseconds(segment["start"]), _milliseconds(segment["end"])
        if end <= start:
            raise ValueError(f"{location}: interval collapses at millisecond subtitle precision")
        text = _plain_text(segment["text"] if segment["text"] else segment["label"], fmt, location)
        if fmt == "vtt":
            text = escape(text, quote=False)
        timing = f"{_timestamp(start, fmt)} --> {_timestamp(end, fmt)}"
        cues.append(f"{index}\n{timing}\n{text}" if fmt == "srt" else f"{timing}\n{text}")
    body = "\n\n".join(cues)
    if fmt == "vtt":
        return "WEBVTT\n\n" + body + ("\n" if body else "")
    return body + ("\n" if body else "")


def _parse_time(value: str, fmt: str) -> float:
    pattern = _SRT_TIME if fmt == "srt" else _VTT_TIME
    match = re.fullmatch(pattern, value)
    if match is None:
        raise ValueError(f"Invalid {fmt.upper()} timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = match.groups()
    try:
        total = ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(milliseconds)
        return total / 1000
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Subtitle timestamp is out of range: {value!r}") from exc


def _decode_vtt_text(value: str, location: str) -> str:
    if "<" in value or ">" in value:
        raise ValueError(f"{location}: WebVTT markup is unsupported; escape literal angle brackets")
    if "&" in _VTT_ESCAPE.sub("", value):
        raise ValueError(f"{location}: only WebVTT amp/lt/gt/nbsp/lrm/rlm named escapes are supported")
    return unescape(value)


def import_subtitles(content: str, format: str, clip: str, duration,
                     track: str = "subtitles") -> Dict[str, Any]:
    """Import a documented plain-text subtitle subset into unreviewed segments.

    Clip identity and duration are supplied by the caller, not inferred from
    subtitle boundaries. No fps, speech recognition or review verdict is added.
    """
    fmt = _format_name(format)
    document: Dict[str, Any] = {"kind": "temporal", "schema_version": 1, "clip": clip,
                                "duration": duration, "segments": []}
    require_valid_document(document)
    if not isinstance(track, str) or not track.strip():
        raise ValueError("Subtitle track must be a non-empty string")
    if not isinstance(content, str):
        raise ValueError("Subtitle content must be a string")
    body = content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if fmt == "vtt":
        if body == "WEBVTT" or body == "WEBVTT\n":
            body = ""
        elif body.startswith("WEBVTT\n\n"):
            body = body[len("WEBVTT\n\n"):]
        else:
            raise ValueError("WebVTT requires a plain WEBVTT header and blank line; header metadata is unsupported")
    if not body.strip():
        return document
    blocks = [block for block in re.split(r"\n(?:[ \t]*\n)+", body.strip("\n")) if block.strip()]
    seen_numbers = set()
    time_pattern = _SRT_TIME if fmt == "srt" else _VTT_TIME
    timing_pattern = re.compile(rf"({time_pattern}) --> ({time_pattern})")
    for index, block in enumerate(blocks, 1):
        lines = block.split("\n")
        location = f"cue {index}"
        if fmt == "srt":
            if not lines or re.fullmatch(r"[0-9]+", lines[0]) is None:
                raise ValueError(f"{location}: SRT requires a numeric cue index")
            try:
                number = int(lines.pop(0))
            except ValueError as exc:
                raise ValueError(f"{location}: SRT cue index is out of range") from exc
            if number < 1 or number in seen_numbers:
                raise ValueError(f"{location}: SRT cue indices must be positive and unique")
            seen_numbers.add(number)
        if not lines or timing_pattern.fullmatch(lines[0]) is None:
            raise ValueError(f"{location}: expected plain timing line; cue identifiers/settings/styles/notes are unsupported")
        timing = lines.pop(0)
        start_text, end_text = timing.split(" --> ")
        text = "\n".join(lines)
        if fmt == "vtt":
            text = _decode_vtt_text(text, location)
        text = _plain_text(text, fmt, location)
        document["segments"].append({
            "id": f"subtitle-{index}", "start": _parse_time(start_text, fmt),
            "end": _parse_time(end_text, fmt), "label": "字幕", "text": text,
            "tags": [], "track": track, "review_status": "unreviewed",
        })
    require_valid_document(document)
    return document
