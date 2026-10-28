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

from .errors import WorkspaceError

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,95}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SAFE_INTEGER = 9007199254740991


def require_id(value: Any, field: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise WorkspaceError(
            "INVALID_ID", f"{field} must match ^[A-Za-z][A-Za-z0-9_-]{{0,95}}$", field=field
        )
    return value


def require_hash(value: Any, field: str = "sha256") -> str:
    if not isinstance(value, str) or not HASH_RE.match(value):
        raise WorkspaceError("INVALID_HASH", f"{field} must be 64 lowercase hex chars", field=field)
    return value


def _parse_intstr(value: Any, field: str) -> int:
    if not isinstance(value, str) or not re.match(r"^-?(0|[1-9][0-9]*)$", value) or len(value) > 40:
        raise WorkspaceError("INVALID_RATIONAL", f"{field} must be a decimal integer string", field=field)
    ivalue = int(value)
    if abs(ivalue) > MAX_SAFE_INTEGER:
        raise WorkspaceError("INVALID_RATIONAL", f"{field} exceeds safe integer range", field=field)
    return ivalue


def reduce_rational(n: int, d: int) -> Tuple[str, str]:
    if d == 0:
        raise WorkspaceError("INVALID_RATIONAL", "rational denominator must not be zero")
    if d < 0:
        n, d = -n, -d
    if n == 0:
        return "0", "1"
    a, b = abs(n), d
    while b:
        a, b = b, a % b
    n //= a
    d //= a
    if abs(n) > MAX_SAFE_INTEGER or d > MAX_SAFE_INTEGER:
        raise WorkspaceError("INVALID_RATIONAL", "rational out of safe integer range")
    return str(n), str(d)


def rational(seconds: Any) -> Dict[str, str]:
    if isinstance(seconds, dict):
        n = _parse_intstr(seconds.get("n"), "n")
        d = _parse_intstr(seconds.get("d"), "d")
    elif isinstance(seconds, tuple):
        n, d = int(seconds[0]), int(seconds[1])
    else:
        fr = Fraction(float(seconds)).limit_denominator(10 ** 9)
        n, d = fr.numerator, fr.denominator
    rn, rd = reduce_rational(n, d)
    return {"n": rn, "d": rd}


def rational_value(rat: Dict[str, str]) -> float:
    return _parse_intstr(rat["n"], "n") / _parse_intstr(rat["d"], "d")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def make_id(prefix: str, token: str) -> str:
    require_id(prefix, "prefix")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    ident = f"{prefix}-{digest[:10]}"
    require_id(ident)
    return ident


def task_subject(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task["id"],
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
