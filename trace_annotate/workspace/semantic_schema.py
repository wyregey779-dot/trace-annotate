"""Versioned semantic contract: Claims, evidence refs, model runs, lineage.

The contract separates four statement kinds and never upgrades a model result
to a reviewed fact. Validation here is structural: it can prove that a claim is
labelled, authored and evidenced, not that a sentence is true.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import schema as S
from .errors import SchemaInvalid, WorkspaceError

SEMANTIC_VERSION = "visual-semantics-v3"
SEMANTIC_VERSION_V1 = "visual-semantics-v1"
SEMANTIC_VERSION_V2 = "visual-semantics-v2"
SEMANTIC_VERSIONS: Tuple[str, ...] = (
    SEMANTIC_VERSION_V1, SEMANTIC_VERSION_V2, SEMANTIC_VERSION)
DIGEST_VERSION = "visual-semantics-v1"
DIGEST_VERSION_V2 = "visual-semantics-v2"
DIGEST_VERSION_V3 = "visual-semantics-v3"
CLAIM_TYPES: Tuple[str, ...] = ("observable", "measured", "interpretation", "unspecified")
SCORE_KINDS: Tuple[str, ...] = ("similarity", "softmax", "rule_score", "confidence", "none")
MEASUREMENT_BASES: Tuple[str, ...] = ("measured", "declared")
MEASUREMENT_FIELDS: Tuple[str, ...] = ("value", "unit", "basis", "analysis_window")
MODEL_RUN_STATUSES: Tuple[str, ...] = ("success", "failed", "not_run")
ACTOR_TYPES: Tuple[str, ...] = ("human", "assistant", "tool")
REVIEW_DECISIONS: Tuple[str, ...] = ("accepted", "edited", "rejected")
MAX_CLAIMS = 32
MAX_EVIDENCE_PER_CLAIM = 32
# Remain bounded while leaving room for the explicit object, hand, state,
# auxiliary-video and composer receipts required by the manipulation pipeline.
MAX_MODEL_RUNS = 32
MAX_STATEMENT_CHARS = 2000
MAX_UNCERTAINTY_CHARS = 500
MAX_UNIT_CHARS = 120

CLAIM_FIELDS_V1 = ("id", "claim_type", "statement", "subject_ref", "time_window",
                   "evidence_ids", "method", "model_run_id", "score", "score_kind",
                   "uncertainty_reason")
# v2 adds a first-class measurement object so a measured value is never smuggled
# into ``score``. ``score`` remains confidence/rule score and is independent.
CLAIM_FIELDS_V2 = CLAIM_FIELDS_V1 + ("measurement",)
# v3 binds the object and recipient identities used by action claims.  Both
# fields are always present and nullable so absence is explicit and digestible.
CLAIM_FIELDS_V3 = CLAIM_FIELDS_V2 + ("object_ref", "recipient_ref")
CLAIM_FIELDS = CLAIM_FIELDS_V3
MODEL_RUN_FIELDS = ("id", "provider", "model_name", "model_version", "weights_sha256",
                    "license", "preprocessing", "parameters", "input_sha256",
                    "output_sha256", "code_version", "status", "failure_reason",
                    "score_kind")
LINEAGE_FIELDS = ("job_id", "model_run_ids", "input_sha256", "result_sha256",
                  "parameters", "created_at")
DECISION_FIELDS = ("actor_id", "actor_type", "decision", "reviewed_at",
                   "content_sha256", "previous_content_sha256", "command_id",
                   "diff_summary")
SEMANTIC_FIELDS = ("version", "claims")

# Fields that participate in each digest. Self-referential summaries are excluded.
CLAIM_DIGEST_FIELDS = ("claim_type", "statement", "subject_ref", "time_window",
                       "evidence_ids", "method", "model_run_id", "score", "score_kind",
                       "uncertainty_reason")
CLAIM_DIGEST_FIELDS_V2 = CLAIM_DIGEST_FIELDS + ("measurement",)
CLAIM_DIGEST_FIELDS_V3 = CLAIM_DIGEST_FIELDS_V2 + ("object_ref", "recipient_ref")
MODEL_RUN_DIGEST_FIELDS = ("provider", "model_name", "model_version", "weights_sha256",
                           "license", "preprocessing", "parameters", "input_sha256",
                           "code_version", "status", "failure_reason", "score_kind")
ANNOTATION_REVIEW_SUBJECT_FIELDS = ("id", "type", "label", "text", "tags", "track",
                                    "attributes", "evidence_ids", "provenance",
                                    "start", "end", "at", "legacy_id")
ANNOTATION_REVIEW_EXCLUDED_FIELDS = ("review_status", "review_binding")

DIGEST_PROJECTIONS = {
    "digest_version": DIGEST_VERSION,
    "claim": {"fields": list(CLAIM_DIGEST_FIELDS), "excludes": ["id"],
              "input": {"digest_version": DIGEST_VERSION, "claim": "<projection>"}},
    "claim_v2": {"fields": list(CLAIM_DIGEST_FIELDS_V2), "excludes": ["id"],
                 "input": {"digest_version": DIGEST_VERSION_V2, "claim": "<projection>"}},
    "claim_v3": {"fields": list(CLAIM_DIGEST_FIELDS_V3), "excludes": ["id"],
                 "input": {"digest_version": DIGEST_VERSION_V3, "claim": "<projection>"}},
    "semantic": {"fields": ["version", "claims"],
                 "claims_sorted_by": "id",
                 "input": {"digest_version": DIGEST_VERSION, "version": "visual-semantics-v1",
                           "claims": ["<claim digest projections>"]}},
    "semantic_v2": {"fields": ["version", "claims"],
                    "claims_sorted_by": "id",
                    "input": {"digest_version": DIGEST_VERSION_V2, "version": "visual-semantics-v2",
                              "claims": ["<claim digest projections>"]}},
    "semantic_v3": {"fields": ["version", "claims"],
                     "claims_sorted_by": "id",
                     "input": {"digest_version": DIGEST_VERSION_V3, "version": "visual-semantics-v3",
                               "claims": ["<claim digest projections>"]}},
    "model_run": {"fields": list(MODEL_RUN_DIGEST_FIELDS),
                  "excludes": ["id", "output_sha256"],
                  "input": {"digest_version": DIGEST_VERSION, "run": "<projection>"}},
    "annotation_review": {
        "fields": list(ANNOTATION_REVIEW_SUBJECT_FIELDS),
        "excludes": list(ANNOTATION_REVIEW_EXCLUDED_FIELDS),
        "input": {"annotation": "<annotation without review fields>", "media_id": "...",
                  "template_id": "...", "template_version": 1},
        "note": "identical to the existing workspace/professional review binding",
    },
    "task": {"note": "task digest is owned by workspace/schema.task_content_sha256 and professional_schema.task_content_digest; semantic fields travel inside annotations and therefore already participate",
             "excludes": ["content_sha256", "review_binding"]},
}


def _invalid(message: str, code: str = "INVALID_SEMANTIC_CLAIM") -> None:
    raise SchemaInvalid(code, message)


def _object(value: Any, fields: Sequence[str], what: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields):
        _invalid(f"{what} must contain exactly {list(fields)}")
    return value


def _bounded_text(value: Any, field: str, maximum: int, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or len(value) > maximum or (nonempty and not value):
        _invalid(f"{field} must be a bounded non-empty string")
    return value


def _nullable_id(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    return S.require_id(value, field)


def _time_window(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    _object(value, ("start", "end"), "time_window")
    start = value["start"]
    end = value["end"]
    _object(start, ("seconds",), "time_window.start")
    _object(end, ("seconds",), "time_window.end")
    start_seconds = S.rational(start["seconds"])
    end_seconds = S.rational(end["seconds"])
    if start_seconds != start["seconds"] or end_seconds != end["seconds"]:
        _invalid("time_window seconds must be reduced rationals")
    if S.rational_value(start_seconds) >= S.rational_value(end_seconds):
        _invalid("time_window start must precede end")
    return {"start": {"seconds": start_seconds}, "end": {"seconds": end_seconds}}


def _score(value: Any, score_kind: Any) -> Tuple[Optional[float], str]:
    if score_kind not in SCORE_KINDS:
        _invalid("score_kind must be one of %s" % (SCORE_KINDS,), "INVALID_SCORE_KIND")
    if value is None:
        if score_kind != "none":
            _invalid("A missing score requires score_kind=none", "INVALID_SCORE_KIND")
        return None, score_kind
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _invalid("score must be a finite number or null")
    if score_kind == "none":
        _invalid("A present score requires an explicit score kind", "INVALID_SCORE_KIND")
    return float(value), score_kind


def _measurement(value: Any, claim_type: str) -> Optional[Dict[str, Any]]:
    """Validate a v2 measurement independently from confidence/score."""
    if claim_type != "measured":
        if value is not None:
            _invalid("Only a measured claim may carry a measurement")
        return None
    _object(value, MEASUREMENT_FIELDS, "measurement")
    number = value["value"]
    if (isinstance(number, bool) or not isinstance(number, (int, float))
            or not math.isfinite(float(number))):
        _invalid("measurement.value must be a finite number")
    unit = _bounded_text(value["unit"], "measurement.unit", MAX_UNIT_CHARS)
    if value["basis"] not in MEASUREMENT_BASES:
        _invalid("measurement.basis must be one of %s" % (MEASUREMENT_BASES,))
    analysis = _time_window(value["analysis_window"])
    return {"value": float(number), "unit": unit, "basis": value["basis"],
            "analysis_window": analysis}


def validate_claim(claim: Any, *, actor_type: str = "human", model_run_ids: Iterable[str] = (),
                   evidence_ids: Iterable[str] = (), version: str = SEMANTIC_VERSION) -> Dict[str, Any]:
    """Validate one claim. Actor type decides whether inference is permitted."""
    if actor_type not in ACTOR_TYPES:
        _invalid("Unknown actor type")
    if version not in SEMANTIC_VERSIONS:
        _invalid("Unknown semantic version", "UNKNOWN_SEMANTIC_VERSION")
    fields = (CLAIM_FIELDS_V1 if version == SEMANTIC_VERSION_V1
              else CLAIM_FIELDS_V2 if version == SEMANTIC_VERSION_V2
              else CLAIM_FIELDS_V3)
    item = _object(claim, fields, "claim")
    S.require_id(item["id"], "claim id")
    claim_type = item["claim_type"]
    if claim_type not in CLAIM_TYPES:
        _invalid("claim_type must be one of %s" % (CLAIM_TYPES,), "INVALID_CLAIM_TYPE")
    statement = _bounded_text(item["statement"], "statement", MAX_STATEMENT_CHARS)
    subject = _nullable_id(item["subject_ref"], "subject_ref")
    object_ref = (_nullable_id(item["object_ref"], "object_ref")
                  if version == SEMANTIC_VERSION else None)
    recipient_ref = (_nullable_id(item["recipient_ref"], "recipient_ref")
                     if version == SEMANTIC_VERSION else None)
    window = _time_window(item["time_window"])
    if not isinstance(item["evidence_ids"], list) or len(item["evidence_ids"]) > MAX_EVIDENCE_PER_CLAIM:
        _invalid("evidence_ids must be a bounded array")
    allowed_evidence = set(evidence_ids)
    for value in item["evidence_ids"]:
        ref = S.require_id(value, "evidence id")
        if ref not in allowed_evidence:
            _invalid(f"evidence id {ref} is not declared on the annotation",
                     "UNSUPPORTED_PROFESSIONAL_EVIDENCE")
    method = item["method"]
    _object(method, ("id", "version", "parameters"), "method")
    S.require_id(method["id"], "method id")
    _bounded_text(method["version"], "method version", 120)
    if not isinstance(method["parameters"], dict):
        _invalid("method parameters must be an object")
    S.canonical_json(method["parameters"])
    run_id = _nullable_id(item["model_run_id"], "model_run_id")
    if model_run_ids is not None and run_id is not None and run_id not in set(model_run_ids):
        _invalid(f"model run {run_id} is not declared", "INVALID_SEMANTIC_CLAIM")
    score, score_kind = _score(item["score"], item["score_kind"])
    measurement = (_measurement(item.get("measurement"), claim_type)
                   if version != SEMANTIC_VERSION_V1 else None)
    uncertainty = item["uncertainty_reason"]
    if uncertainty is not None:
        _bounded_text(uncertainty, "uncertainty_reason", MAX_UNCERTAINTY_CHARS)
    if claim_type == "interpretation" and actor_type != "human":
        _invalid("Only a human author may store an interpretation",
                 "INTERPRETATION_REQUIRES_HUMAN")
    action_contract = method["parameters"].get("action_contract")
    if version == SEMANTIC_VERSION and action_contract is not None:
        if action_contract != "action-candidate-v3":
            _invalid("Unknown action claim contract")
        from ..extractors.actions import ACTION_REQUIREMENTS, ACTION_VOCAB, provider_coverage
        action = method["parameters"].get("action")
        if action not in ACTION_VOCAB:
            _invalid("Action claim uses a class outside the frozen vocabulary")
        coverage = provider_coverage(method["id"])
        if coverage.get("status") != "success" or action not in set(coverage.get("supported") or ()):
            _invalid("Action provider does not declare support for this action")
        required = ACTION_REQUIREMENTS[action]
        if subject is None:
            _invalid("An action claim requires subject_ref")
        if "object_ref" in required and object_ref is None:
            _invalid("This action claim requires object_ref")
        if "recipient_ref" in required and recipient_ref is None:
            _invalid("This action claim requires recipient_ref")
        if action == "递交" and recipient_ref == subject:
            _invalid("A hand-over recipient must differ from its subject")
        if window is None or not item["evidence_ids"] or run_id is None or score_kind == "none":
            _invalid("An action claim requires a window, evidence, model run and scored provider output")
        pts = method["parameters"].get("window_pts")
        if not isinstance(pts, dict) or set(pts) != {"start", "end"}:
            _invalid("An action claim requires its canonical window_pts")
        try:
            pts_start, pts_end = S.rational(pts["start"]), S.rational(pts["end"])
            if (pts_start != pts["start"] or pts_end != pts["end"]
                    or S.rational_value(pts_start) >= S.rational_value(pts_end)
                    or window["start"]["seconds"] != pts_start
                    or window["end"]["seconds"] != pts_end):
                _invalid("Action PTS must be reduced, ordered and equal to its time window")
        except (KeyError, TypeError, ValueError, WorkspaceError):
            _invalid("Action PTS must be reduced, ordered and equal to its time window")
        lower = -1.0 if score_kind == "similarity" else 0.0
        if score is None or not lower <= score <= 1.0:
            _invalid("Action claim score is outside the range admitted by its score kind")
    if claim_type == "measured":
        if version == SEMANTIC_VERSION_V1:
            if score_kind == "none":
                _invalid("A measured claim requires a score kind, not 'none'", "INVALID_SCORE_KIND")
            if score is None:
                _invalid("A measured claim requires its numeric result", "INVALID_SEMANTIC_CLAIM")
        elif measurement is None:
            _invalid("A v2 measured claim requires its measurement object", "INVALID_SEMANTIC_CLAIM")
    checked = {
        "id": item["id"], "claim_type": claim_type, "statement": statement,
        "subject_ref": subject, "time_window": window,
        "evidence_ids": list(item["evidence_ids"]), "method": method,
        "model_run_id": run_id, "score": score, "score_kind": score_kind,
        "uncertainty_reason": uncertainty,
    }
    if version != SEMANTIC_VERSION_V1:
        checked["measurement"] = measurement
    if version == SEMANTIC_VERSION:
        checked["object_ref"] = object_ref
        checked["recipient_ref"] = recipient_ref
    return checked


def validate_semantic(value: Any, *, actor_type: str = "human",
                      model_run_ids: Iterable[str] = (),
                      evidence_ids: Iterable[str] = ()) -> Dict[str, Any]:
    """Validate the whole attributes.semantic object and freeze its version."""
    item = _object(value, SEMANTIC_FIELDS, "attributes.semantic")
    version = item["version"]
    if version not in SEMANTIC_VERSIONS:
        _invalid("Unknown semantic version", "UNKNOWN_SEMANTIC_VERSION")
    claims = item["claims"]
    if not isinstance(claims, list) or len(claims) > MAX_CLAIMS:
        _invalid("claims must be a bounded array")
    allowed_runs = None if model_run_ids is None else tuple(model_run_ids)
    allowed_evidence = tuple(evidence_ids)
    seen = set()
    validated: List[Dict[str, Any]] = []
    for claim in claims:
        checked = validate_claim(claim, actor_type=actor_type, model_run_ids=allowed_runs,
                                 evidence_ids=allowed_evidence, version=version)
        if checked["id"] in seen:
            _invalid("Claim identifiers must be unique")
        seen.add(checked["id"])
        validated.append(checked)
    return {"version": version, "claims": validated}


def validate_model_run(run: Any) -> Dict[str, Any]:
    item = _object(run, MODEL_RUN_FIELDS, "model run")
    S.require_id(item["id"], "model run id")
    _bounded_text(item["provider"], "provider", 120)
    _bounded_text(item["model_name"], "model_name", 200)
    _bounded_text(item["model_version"], "model_version", 120)
    if item["weights_sha256"] is not None:
        S.require_hash(item["weights_sha256"], "weights_sha256")
    _bounded_text(item["license"], "license", 200)
    S.canonical_json(item["preprocessing"])
    if not isinstance(item["preprocessing"], dict):
        _invalid("preprocessing must be an object")
    if not isinstance(item["parameters"], dict):
        _invalid("parameters must be an object")
    S.canonical_json(item["parameters"])
    S.require_hash(item["input_sha256"], "input_sha256")
    if item["output_sha256"] is not None:
        S.require_hash(item["output_sha256"], "output_sha256")
    _bounded_text(item["code_version"], "code_version", 120)
    status = item["status"]
    if status not in MODEL_RUN_STATUSES:
        _invalid("status must be one of %s" % (MODEL_RUN_STATUSES,))
    if status == "success":
        if item["failure_reason"] is not None or item["output_sha256"] is None:
            _invalid("A successful run needs an output hash and no failure reason")
    else:
        _bounded_text(item["failure_reason"], "failure_reason", 500)
    if item["score_kind"] not in SCORE_KINDS:
        _invalid("score_kind must be one of %s" % (SCORE_KINDS,), "INVALID_SCORE_KIND")
    return dict(item)


def validate_lineage(value: Any, *, model_run_ids: Iterable[str] = ()) -> Dict[str, Any]:
    item = _object(value, LINEAGE_FIELDS, "generation lineage")
    S.require_id(item["job_id"], "job_id")
    if not isinstance(item["model_run_ids"], list):
        _invalid("model_run_ids must be an array")
    allowed = set(model_run_ids)
    for ref in item["model_run_ids"]:
        if S.require_id(ref, "model run id") not in allowed:
            _invalid(f"lineage run {ref} is not declared")
    S.require_hash(item["input_sha256"], "input_sha256")
    S.require_hash(item["result_sha256"], "result_sha256")
    if not isinstance(item["parameters"], dict):
        _invalid("lineage parameters must be an object")
    _bounded_text(item["created_at"], "created_at", 80)
    return dict(item)


def validate_review_decision(value: Any, *, content_sha256: Optional[str] = None) -> Dict[str, Any]:
    item = _object(value, DECISION_FIELDS, "review decision")
    S.require_id(item["actor_id"], "actor_id")
    if item["actor_type"] not in ACTOR_TYPES:
        _invalid("actor_type must be one of %s" % (ACTOR_TYPES,))
    if item["decision"] not in REVIEW_DECISIONS:
        _invalid("decision must be one of %s" % (REVIEW_DECISIONS,))
    _bounded_text(item["reviewed_at"], "reviewed_at", 80)
    S.require_hash(item["content_sha256"], "content_sha256")
    if item["previous_content_sha256"] is not None:
        S.require_hash(item["previous_content_sha256"], "previous_content_sha256")
    S.require_id(item["command_id"], "command_id")
    if item["diff_summary"] is not None:
        _bounded_text(item["diff_summary"], "diff_summary", 500)
    if content_sha256 is not None and item["content_sha256"] != content_sha256:
        _invalid("Review decision does not bind current content", "STALE_REVIEW_BINDING")
    return dict(item)


def claim_digest(claim: Dict[str, Any]) -> str:
    if "object_ref" in claim or "recipient_ref" in claim:
        projection = {key: claim[key] for key in CLAIM_DIGEST_FIELDS_V3}
        return S.sha256_text({"digest_version": DIGEST_VERSION_V3, "claim": projection})
    if "measurement" in claim:
        projection = {key: claim[key] for key in CLAIM_DIGEST_FIELDS_V2}
        return S.sha256_text({"digest_version": DIGEST_VERSION_V2, "claim": projection})
    projection = {key: claim[key] for key in CLAIM_DIGEST_FIELDS}
    return S.sha256_text({"digest_version": DIGEST_VERSION, "claim": projection})


def semantic_digest(semantic: Dict[str, Any]) -> str:
    claims = [claim_digest(claim) for claim in sorted(semantic["claims"], key=lambda c: c["id"])]
    version = (DIGEST_VERSION_V3 if semantic["version"] == SEMANTIC_VERSION
               else DIGEST_VERSION_V2 if semantic["version"] == SEMANTIC_VERSION_V2
               else DIGEST_VERSION)
    return S.sha256_text({"digest_version": version, "version": semantic["version"],
                          "claims": claims})


def annotation_review_digest(annotation: Dict[str, Any], *, media_id: str,
                             template_id: str, template_version: int) -> str:
    """Same rule as the workspace store: review binds content, not its own summary."""
    subject = {key: value for key, value in annotation.items()
               if key not in ("review_status", "review_binding")}
    return S.sha256_text({"annotation": subject, "media_id": media_id,
                          "template_id": template_id, "template_version": template_version})


def model_run_digest(run: Dict[str, Any]) -> str:
    projection = {key: run[key] for key in MODEL_RUN_DIGEST_FIELDS}
    return S.sha256_text({"digest_version": DIGEST_VERSION, "run": projection})


def build_semantic(claims: Sequence[Dict[str, Any]], *, actor_type: str = "human",
                   model_run_ids: Iterable[str] = (), evidence_ids: Iterable[str] = (),
                   version: str = SEMANTIC_VERSION) -> Dict[str, Any]:
    return validate_semantic({"version": version, "claims": list(claims)},
                             actor_type=actor_type, model_run_ids=model_run_ids,
                             evidence_ids=evidence_ids)


def empty_semantic(version: str = SEMANTIC_VERSION) -> Dict[str, Any]:
    return {"version": version, "claims": []}


def claim_requires_human(claim_type: str) -> bool:
    return claim_type == "interpretation"


def semantic_summary(semantic: Optional[Dict[str, Any]]) -> Dict[str, int]:
    counts = {name: 0 for name in CLAIM_TYPES}
    for claim in (semantic or {}).get("claims", []):
        counts[claim["claim_type"]] += 1
    return counts
