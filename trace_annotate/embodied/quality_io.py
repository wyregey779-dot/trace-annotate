"""Strict batch input and non-destructive output for embodied quality tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class QualityInputError(ValueError):
    """An unreadable, empty or malformed input, rather than an annotation verdict."""


class QualityViolation(ValueError):
    """A readable annotation failed a quality gate."""


@dataclass
class BatchRecord:
    doc: Dict[str, Any]
    path: str
    location: str
    sha256: str


@dataclass
class AnnotationBatch:
    records: List[BatchRecord] = field(default_factory=list)
    files: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def docs(self) -> List[Dict[str, Any]]:
        return [record.doc for record in self.records]


class _NonFinite:
    def __init__(self, token):
        self.token = token


def _constant(value: str):
    return _NonFinite(value)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _require_unicode_scalars(value: str, location: str) -> None:
    # json.loads merges a valid escaped surrogate pair into one Unicode scalar.
    # Any remaining surrogate cannot be encoded in the UTF-8 reports/exports.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{location}: unpaired Unicode surrogate is not allowed") from None


def _decode_json(text: str):
    value = json.loads(text, parse_constant=_constant, object_pairs_hook=_unique_pairs)
    pending = [("$", value, 0)]
    while pending:
        location, item, depth = pending.pop()
        if depth > 64:
            raise ValueError(f"{location}: annotation JSON exceeds 64 nesting levels")
        if isinstance(item, _NonFinite):
            raise ValueError(f"{location}: non-finite JSON constant {item.token} is not allowed")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{location}: non-finite JSON number is not allowed")
        if isinstance(item, str):
            _require_unicode_scalars(item, location)
        if isinstance(item, dict):
            for key, entry in item.items():
                _require_unicode_scalars(key, f"{location}: object key")
                pending.append((f"{location}.{key}", entry, depth + 1))
        elif isinstance(item, list):
            pending.extend((f"{location}[{index}]", entry, depth + 1) for index, entry in enumerate(item))
    return value


def _objects(value, label: str):
    items = value if isinstance(value, list) else [value]
    if not items:
        raise QualityInputError(f"{label}: empty annotation batch")
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise QualityInputError(f"{label} record {index}: annotation must be a JSON object")
    return items


def read_annotation_file(path: str) -> AnnotationBatch:
    """Accept JSON object/array or JSONL. A bad line invalidates the entire file.

    Parsing the complete document first also accepts pretty-printed JSON with a
    .jsonl suffix. Failed lines are never silently discarded or reported as PASS.
    """
    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise QualityInputError(f"{source}: {error}") from error
    if not text.strip():
        raise QualityInputError(f"{source}: empty annotation file")
    digest = hashlib.sha256(raw).hexdigest()
    info = {"path": str(source.absolute()), "sha256": digest}
    try:
        parsed = _decode_json(text)
    except (ValueError, RecursionError) as complete_error:
        records, errors = [], []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            label = f"{source}:{line_number}"
            try:
                item = _decode_json(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL record must be an object")
                records.append(BatchRecord(item, info["path"], f"line {line_number}", digest))
            except (ValueError, RecursionError) as error:
                errors.append(f"{label}: {error}")
        if errors or not records:
            detail = "; ".join(errors) if errors else str(complete_error)
            raise QualityInputError(f"{source}: invalid JSON/JSONL: {detail}") from complete_error
        return AnnotationBatch(records=records, files=[info])
    items = _objects(parsed, str(source))
    return AnnotationBatch(
        records=[BatchRecord(doc, info["path"], f"record {index}", digest)
                 for index, doc in enumerate(items, 1)], files=[info])


def collect_annotations(target: str) -> AnnotationBatch:
    """Read a file or the JSON/JSONL files directly within a batch directory."""
    source = Path(target)
    if source.is_dir():
        paths = sorted((p for p in source.iterdir()
                        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}), key=str)
    elif source.is_file():
        paths = [source]
    else:
        return AnnotationBatch(errors=[f"{source}: input file or batch directory does not exist"])
    if not paths:
        return AnnotationBatch(errors=[f"{source}: no JSON/JSONL annotation files"])
    batch = AnnotationBatch()
    for path in paths:
        try:
            loaded = read_annotation_file(str(path))
            batch.records.extend(loaded.records)
            batch.files.extend(loaded.files)
        except QualityInputError as error:
            batch.errors.append(str(error))
    return batch


def load_docs_from_file(file_path: str) -> List[Dict[str, Any]]:
    return read_annotation_file(file_path).docs


def load_annotations(target: str) -> Dict[str, Dict[str, Any]]:
    """Compatibility mapping loader: reject duplicate IDs before mapping can lose data."""
    batch = collect_annotations(target)
    if batch.errors:
        raise QualityInputError("; ".join(batch.errors))
    result = {}
    for record in batch.records:
        clip = record.doc.get("clip")
        if not isinstance(clip, str) or not clip.strip():
            raise QualityInputError(f"{record.path} {record.location}: non-empty clip is required")
        if clip in result:
            raise QualityInputError(f"duplicate clip ID: {clip}")
        result[clip] = record.doc
    return result


def _reject_link_parents(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
            raise QualityInputError(f"Output may not traverse a symbolic link or junction: {ancestor}")


def require_new_output(path: str, input_paths=()) -> Path:
    target = Path(os.path.abspath(path))
    _reject_link_parents(target)
    if os.path.lexists(target):
        raise QualityInputError(f"Output already exists; choose a new report/output path: {target}")
    target_key = os.path.normcase(str(target))
    if target_key in {os.path.normcase(os.path.abspath(p)) for p in input_paths}:
        raise QualityInputError(f"Output must not replace an input: {target}")
    return target


def write_new_text(path: str, text: str, input_paths=()) -> str:
    """Exclusive creation: never truncate an old report or an annotation input."""
    target = require_new_output(path, input_paths)
    # Fail before creating a file or parent directories. TextIOWrapper would
    # create an empty report before noticing an unencodable character.
    encoded = _encode_output_text(text)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_parents(target)
    # 'x' also catches a new file/symlink introduced after the preflight check.
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return str(target)


def _encode_output_text(text: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        raise QualityInputError("Output contains an unpaired Unicode surrogate; no output created") from None


def replace_annotation_explicitly(path: str, original: bytes, text: str) -> str:
    """Explicit --inplace only: backup then atomic replace after complete validation.

    Rechecks the source bytes before publishing. This is not a multi-writer CAS;
    a concurrent writer must use separate output paths instead.
    """
    source = Path(os.path.abspath(path))
    _reject_link_parents(source)
    if source.read_bytes() != original:
        raise QualityInputError("Input changed during idle preparation; no replacement made")
    encoded = _encode_output_text(text)
    backup = source.with_name(source.name + f".before-idle-{uuid.uuid4().hex}.bak")
    with backup.open("xb") as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor, temporary = tempfile.mkstemp(prefix=f".{source.name}.idle-", suffix=".tmp", dir=source.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_link_parents(source)
        if source.read_bytes() != original:
            raise QualityInputError("Input changed before publication; original and backup retained")
        os.replace(temporary, source)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(backup)


def annotation_json(docs: List[Dict[str, Any]], *, single: bool = False, jsonl: bool = False) -> str:
    if jsonl:
        return "".join(json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n" for doc in docs)
    return json.dumps(docs[0] if single and len(docs) == 1 else docs,
                      ensure_ascii=False, indent=2, allow_nan=False) + "\n"
