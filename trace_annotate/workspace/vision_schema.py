"""Candidate annotations that carry versioned semantic claims.

Every helper here produces ``model_candidate`` + ``unreviewed``. A model may
declare observations and measurements; interpretations require a human author.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import schema as S
from . import semantic_schema as SS
from .errors import SchemaInvalid

CANDIDATE_ORIGIN = "model_candidate"
CANDIDATE_REVIEW = "unreviewed"
MAX_TAGS = 12


def _ratio_pair(seconds: Any) -> Dict[str, Any]:
    return {"seconds": S.rational(seconds)}


def _tags(tags: Iterable[str]) -> List[str]:
    out: List[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag or len(tag) > 120:
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Tags must be bounded non-empty strings")
        if tag not in out:
            out.append(tag)
    if len(out) > MAX_TAGS:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "A candidate carries too many tags")
    return out


def _method(method_id: str, version: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    S.require_id(method_id, "method id")
    if not isinstance(parameters, dict):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Method parameters must be an object")
    SS.S.canonical_json(parameters)
    return {"id": method_id, "version": str(version), "parameters": parameters}


def claim(*, claim_type: str, statement: str, method_id: str, method_version: str,
          parameters: Optional[Dict[str, Any]] = None, evidence_ids: Sequence[str] = (),
          subject_ref: Optional[str] = None, start: Any = None, end: Any = None,
          score: Optional[float] = None, score_kind: str = "none",
          uncertainty_reason: Optional[str] = None,
          model_run_id: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    """Build one claim; the caller keeps the token stable for reproducible ids."""
    if claim_type not in SS.CLAIM_TYPES:
        raise SchemaInvalid("INVALID_CLAIM_TYPE", "Unknown claim type")
    window = None
    if start is not None or end is not None:
        if start is None or end is None:
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "A claim window needs both ends")
        window = {"start": _ratio_pair(start), "end": _ratio_pair(end)}
    body = {
        "id": S.make_id("claim", token or f"{claim_type}|{statement}|{subject_ref}|{start}|{end}"),
        "claim_type": claim_type,
        "statement": statement,
        "subject_ref": subject_ref,
        "time_window": window,
        "evidence_ids": list(evidence_ids),
        "method": _method(method_id, method_version, parameters or {}),
        "model_run_id": model_run_id,
        "score": score,
        "score_kind": score_kind,
        "uncertainty_reason": uncertainty_reason,
    }
    return SS.validate_claim(body, actor_type="tool", model_run_ids=(), evidence_ids=evidence_ids)


def observable(**kwargs: Any) -> Dict[str, Any]:
    return claim(claim_type="observable", **kwargs)


def measured(**kwargs: Any) -> Dict[str, Any]:
    return claim(claim_type="measured", **kwargs)


def interpretation(**kwargs: Any) -> Dict[str, Any]:
    body = claim(claim_type="interpretation", **kwargs)
    return SS.validate_claim(body, actor_type="human", evidence_ids=body["evidence_ids"])


def build_annotation(*, annotation_id: str, kind: str, label: str, text: str,
                     track: str, claims: Sequence[Dict[str, Any]],
                     tags: Iterable[str] = (), evidence_ids: Sequence[str] = (),
                     start: Any = None, end: Any = None, at: Any = None,
                     actor_id: str = "vision-extractor", job_id: Optional[str] = None,
                     origin: str = CANDIDATE_ORIGIN,
                     review_status: str = CANDIDATE_REVIEW) -> Dict[str, Any]:
    """Assemble a candidate annotation whose semantics obey the frozen contract."""
    S.require_id(annotation_id)
    if kind not in ("segment", "event", "whole_clip"):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Unknown annotation type")
    for value in (label, track):
        if not isinstance(value, str) or not value:
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Label and track must be non-empty")
    if not isinstance(text, str):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Text must be a string")
    semantic = SS.build_semantic(claims, actor_type="tool" if origin != "manual" else "human",
                                 evidence_ids=evidence_ids)
    if origin == CANDIDATE_ORIGIN and review_status != CANDIDATE_REVIEW:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Model candidates stay unreviewed")
    if origin == CANDIDATE_ORIGIN:
        for item in semantic["claims"]:
            if item["claim_type"] == "interpretation":
                raise SchemaInvalid("INTERPRETATION_REQUIRES_HUMAN",
                                    "Model candidates must not carry interpretations")
    annotation: Dict[str, Any] = {
        "id": annotation_id, "type": kind, "label": label, "text": text,
        "tags": _tags(tags), "track": track,
        "attributes": {"semantic": semantic},
        "evidence_ids": [S.require_id(item, "evidence id") for item in evidence_ids],
        "provenance": {"origin": origin, "actor_id": actor_id, "job_id": job_id},
        "review_status": review_status,
    }
    if kind == "segment":
        if start is None or end is None:
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "A segment needs start and end")
        annotation["start"] = _ratio_pair(start)
        annotation["end"] = _ratio_pair(end)
    elif kind == "event":
        if at is None:
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "An event needs its anchor time")
        annotation["at"] = _ratio_pair(at)
    return annotation


def validate_candidate(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the candidate-side rules that matter before a store write."""
    if not isinstance(annotation, dict) or not set(annotation) <= {
            "id", "type", "label", "text", "tags", "track", "attributes", "evidence_ids",
            "provenance", "review_status", "start", "end", "at", "legacy_id"}:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Candidate annotation fields are invalid")
    provenance = annotation.get("provenance")
    if not isinstance(provenance, dict):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Candidate provenance is missing")
    origin = provenance.get("origin")
    if origin not in ("manual", "imported", "model_candidate", "derived"):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Unknown candidate origin")
    if origin == "model_candidate" and annotation.get("review_status") != CANDIDATE_REVIEW:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Model candidates stay unreviewed")
    if origin == "model_candidate" and provenance.get("job_id") is None:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "A model candidate requires its job id")
    attributes = annotation.get("attributes")
    if not isinstance(attributes, dict) or set(attributes) != {"semantic"}:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "attributes may only carry semantic")
    actor_type = "human" if origin == "manual" else "tool"
    semantic = SS.validate_semantic(attributes["semantic"], actor_type=actor_type,
                                    evidence_ids=annotation.get("evidence_ids", []))
    if origin == "model_candidate":
        if any(item["claim_type"] == "interpretation" for item in semantic["claims"]):
            raise SchemaInvalid("INTERPRETATION_REQUIRES_HUMAN",
                                "Model candidates must not carry interpretations")
    return annotation


def annotation_claims(annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    semantic = (annotation.get("attributes") or {}).get("semantic") or {}
    return list(semantic.get("claims", []))


def replace_claims(annotation: Dict[str, Any], claims: Sequence[Dict[str, Any]],
                   *, actor_type: str = "human") -> Dict[str, Any]:
    """Return a copy with revalidated claims; original evidence set is preserved."""
    updated = dict(annotation)
    attributes = dict(updated.get("attributes") or {})
    attributes["semantic"] = SS.build_semantic(claims, actor_type=actor_type,
                                               evidence_ids=updated.get("evidence_ids", []))
    updated["attributes"] = attributes
    return updated


def summarize(annotation: Dict[str, Any]) -> Dict[str, Any]:
    semantic = (annotation.get("attributes") or {}).get("semantic")
    return {"id": annotation.get("id"), "label": annotation.get("label"),
            "origin": (annotation.get("provenance") or {}).get("origin"),
            "review_status": annotation.get("review_status"),
            "claims": SS.semantic_summary(semantic),
            "evidence": len(annotation.get("evidence_ids", []))}
