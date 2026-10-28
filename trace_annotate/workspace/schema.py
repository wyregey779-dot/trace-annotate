"""Workspace v3 data primitives: IDs, rational time, canonical digests.

Single source of truth for identity, time and hash rules from
contracts/first-batch-contract.md. No filesystem/network side effects.
"""
from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any, Dict, Tuple

from .errors import SchemaInvalid

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,95}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SAFE_INTEGER = 9007199254740991


def require_id(value: Any, field: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise SchemaInvalid(
            "INVALID_ID", f"{field} must be an ASCII identifier", details={"field": field}
        )
    return value


def require_hash(value: Any, field: str = "sha256") -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise SchemaInvalid("INVALID_HASH", f"{field} must be 64 lowercase hex chars", details={"field": field})
    return value


def _parse_intstr(value: Any, field: str) -> int:
    if (not isinstance(value, str) or len(value) > 40
            or not re.fullmatch(r"-?(0|[1-9][0-9]*)", value) or value == "-0"):
        raise SchemaInvalid("INVALID_RATIONAL", f"{field} must be a decimal integer string", details={"field": field})
    return int(value)


def reduce_rational(n: int, d: int) -> Tuple[str, str]:
    if type(n) is not int or type(d) is not int or d == 0:
        raise SchemaInvalid("INVALID_RATIONAL", "rational requires integers and a nonzero denominator")
    if d < 0:
        n, d = -n, -d
    if n == 0:
        return "0", "1"
    a, b = abs(n), d
    while b:
        a, b = b, a % b
    n //= a
    d //= a
    if len(str(n)) > 40 or len(str(d)) > 40:
        raise SchemaInvalid("INVALID_RATIONAL", "rational exceeds 40 decimal characters")
    return str(n), str(d)


def rational(seconds: Any) -> Dict[str, str]:
    if isinstance(seconds, dict):
        if set(seconds) != {"n", "d"}:
            raise SchemaInvalid("INVALID_RATIONAL", "rational requires exactly n and d")
        n = _parse_intstr(seconds.get("n"), "n")
        d = _parse_intstr(seconds.get("d"), "d")
        if d <= 0:
            raise SchemaInvalid("INVALID_RATIONAL", "denominator must be positive")
    elif isinstance(seconds, tuple):
        if len(seconds) != 2:
            raise SchemaInvalid("INVALID_RATIONAL", "rational tuple requires two integers")
        n, d = seconds
    else:
        try:
            if isinstance(seconds, bool) or (isinstance(seconds, str) and len(seconds) > 80):
                raise ValueError
            if isinstance(seconds, str) and re.search(r"[eE]", seconds):
                raise ValueError
            fr = seconds if isinstance(seconds, Fraction) else Fraction(str(seconds))
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            raise SchemaInvalid("INVALID_RATIONAL", "time must be finite and rational") from exc
        n, d = fr.numerator, fr.denominator
    rn, rd = reduce_rational(n, d)
    return {"n": rn, "d": rd}


def rational_value(rat: Dict[str, str]) -> Fraction:
    reduced = rational(rat)
    return Fraction(int(reduced["n"]), int(reduced["d"]))


def canonical_json(obj: Any) -> bytes:
    def validate(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif type(value) in (int, float) and abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("Large exact values must use decimal strings")
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Not a JSON value")
    try:
        validate(obj)
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise SchemaInvalid("SCHEMA_INVALID", "Value is not finite UTF-8 JSON") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def make_id(prefix: str, token: str) -> str:
    require_id(prefix, "prefix")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    ident = f"{prefix}-{digest}"
    require_id(ident)
    return ident


def task_subject(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task["id"],
        "revision": task["revision"],
        "template_id": task["template_id"],
        "template_version": task["template_version"],
        "media_id": task["media_id"],
        "annotations": task["annotations"],
        "evidence": sorted(task.get("evidence", []), key=lambda e: e["id"]),
        "extensions": task.get("extensions", {}),
        "legacy_document_metadata": task.get("legacy_document_metadata", {}),
    }


def task_content_sha256(task: Dict[str, Any]) -> str:
    return sha256_text(task_subject(task))
