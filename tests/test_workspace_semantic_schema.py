"""The versioned semantic contract: label taxonomy, authorship, evidence and digests."""
from __future__ import annotations

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace import semantic_schema as SS
from trace_annotate.workspace.errors import SchemaInvalid


def _evidence():
    return ["art-e1"]


def _claim(claim_type="observable", **overrides):
    body = {
        "id": "claim-0000000000000001",
        "claim_type": claim_type,
        "statement": "画面中人物嘴角上扬",
        "subject_ref": None,
        "time_window": None,
        "evidence_ids": _evidence(),
        "method": {"id": "smile-rule", "version": "1", "parameters": {"threshold": 0.4}},
        "model_run_id": None,
        "score": 0.72,
        "score_kind": "rule_score",
        "measurement": None,
        "object_ref": None,
        "recipient_ref": None,
        "uncertainty_reason": None,
    }
    body.update(overrides)
    return body


def test_claim_taxonomy_and_authorship_are_enforced():
    assert SS.validate_claim(_claim("observable"), evidence_ids={"art-e1"})["claim_type"] == "observable"
    with pytest.raises(SchemaInvalid) as exc:
        SS.validate_claim(_claim("interpretation"), actor_type="tool", evidence_ids={"art-e1"})
    assert exc.value.code == "INTERPRETATION_REQUIRES_HUMAN"
    assert SS.validate_claim(_claim("interpretation"), actor_type="human",
                             evidence_ids={"art-e1"})["claim_type"] == "interpretation"


def test_claim_rejects_unknown_keys_types_and_scores():
    body = _claim()
    body["extra"] = 1
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(body, evidence_ids={"art-e1"})
    with pytest.raises(SchemaInvalid) as exc:
        SS.validate_claim(_claim("emotion"), evidence_ids={"art-e1"})
    assert exc.value.code == "INVALID_CLAIM_TYPE"
    with pytest.raises(SchemaInvalid) as exc:
        SS.validate_claim(_claim(score_kind="accuracy"), evidence_ids={"art-e1"})
    assert exc.value.code == "INVALID_SCORE_KIND"
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(_claim(score=None, score_kind="rule_score"), evidence_ids={"art-e1"})


def test_claim_evidence_must_be_declared_on_the_annotation():
    with pytest.raises(SchemaInvalid) as exc:
        SS.validate_claim(_claim(evidence_ids=["art-other"]), evidence_ids={"art-e1"})
    assert exc.value.code == "UNSUPPORTED_PROFESSIONAL_EVIDENCE"


def test_measured_claim_needs_a_first_class_measurement():
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(_claim("measured", score=None, score_kind="none"), evidence_ids={"art-e1"})
    checked = SS.validate_claim(_claim("measured", statement="窗口 RMS -23.4 dBFS",
                                       score=None, score_kind="none",
                                       measurement={"value": -23.4, "unit": "dBFS",
                                                    "basis": "measured", "analysis_window": None}),
                                evidence_ids={"art-e1"})
    assert checked["measurement"]["value"] == -23.4 and checked["score"] is None


def test_v1_claim_still_validates_and_new_v2_is_explicit():
    legacy = _claim("measured", score=-23.4, score_kind="rule_score")
    for field in ("measurement", "object_ref", "recipient_ref"):
        legacy.pop(field)
    checked = SS.validate_claim(legacy, evidence_ids={"art-e1"}, version=SS.SEMANTIC_VERSION_V1)
    assert checked["score"] == -23.4 and "measurement" not in checked
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(legacy, evidence_ids={"art-e1"})


def test_semantic_version_is_frozen_and_claims_are_unique():
    with pytest.raises(SchemaInvalid) as exc:
        SS.validate_semantic({"version": "visual-semantics-v4", "claims": []})
    assert exc.value.code == "UNKNOWN_SEMANTIC_VERSION"
    with pytest.raises(SchemaInvalid):
        SS.validate_semantic({"version": SS.SEMANTIC_VERSION_V1,
                              "claims": [_claim(), _claim()]}, evidence_ids={"art-e1"})


def test_digest_excludes_identifiers_and_self_summaries():
    one = SS.validate_claim(_claim(), evidence_ids={"art-e1"})
    other = dict(one)
    other["id"] = "claim-0000000000000002"
    assert SS.claim_digest(one) == SS.claim_digest(other)
    assert SS.claim_digest(one) != SS.claim_digest(dict(one, score=0.71))
    assert SS.claim_digest(one) != SS.claim_digest(dict(one, object_ref="object-1"))


def test_v2_remains_valid_and_v3_requires_nullable_action_references():
    v2 = _claim()
    v2.pop("object_ref")
    v2.pop("recipient_ref")
    assert SS.validate_claim(v2, version=SS.SEMANTIC_VERSION_V2,
                             evidence_ids={"art-e1"})["measurement"] is None
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(v2, evidence_ids={"art-e1"})


def test_v3_action_claim_requires_participants_evidence_run_and_window():
    base = _claim(
        statement="候选动作：走", subject_ref="subject-1", model_run_id="run-actions",
        time_window={"start": {"seconds": {"n": "0", "d": "1"}},
                     "end": {"seconds": {"n": "1", "d": "1"}}},
        method={"id": "subject-translation-v1", "version": "1", "parameters": {
            "action_contract": "action-candidate-v3", "action": "走",
            "window_pts": {"start": {"n": "0", "d": "1"},
                           "end": {"n": "1", "d": "1"}}}})
    checked = SS.validate_claim(base, evidence_ids={"art-e1"},
                                model_run_ids={"run-actions"})
    assert checked["subject_ref"] == "subject-1"
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(dict(base, evidence_ids=[]), evidence_ids=set(),
                          model_run_ids={"run-actions"})
    with pytest.raises(SchemaInvalid):
        SS.validate_claim(dict(base, score=1.01), evidence_ids={"art-e1"},
                          model_run_ids={"run-actions"})
    missing_pts = dict(base)
    missing_pts["method"] = {
        **base["method"], "parameters": {
            key: value for key, value in base["method"]["parameters"].items()
            if key != "window_pts"}}
    with pytest.raises(SchemaInvalid, match="window_pts"):
        SS.validate_claim(missing_pts, evidence_ids={"art-e1"},
                          model_run_ids={"run-actions"})
    mismatched_pts = dict(base)
    mismatched_pts["method"] = {
        **base["method"], "parameters": {
            **base["method"]["parameters"],
            "window_pts": {"start": {"n": "0", "d": "1"},
                           "end": {"n": "2", "d": "1"}}}}
    with pytest.raises(SchemaInvalid, match="equal to its time window"):
        SS.validate_claim(mismatched_pts, evidence_ids={"art-e1"},
                          model_run_ids={"run-actions"})
    unsupported = dict(base)
    unsupported.update({"statement": "候选动作：递交", "object_ref": "object-1",
                        "recipient_ref": "recipient-1"})
    unsupported["method"] = {"id": "subject-translation-v1", "version": "1",
                             "parameters": {"action_contract": "action-candidate-v3",
                                            "action": "递交",
                                            "window_pts": {
                                                "start": {"n": "0", "d": "1"},
                                                "end": {"n": "1", "d": "1"}}}}
    with pytest.raises(SchemaInvalid, match="does not declare support"):
        SS.validate_claim(unsupported, evidence_ids={"art-e1"},
                          model_run_ids={"run-actions"})


def test_review_digest_tracks_content_but_not_review_fields():
    annotation = {"id": "ann-1", "type": "segment", "label": "镜头", "text": "",
                  "tags": [], "track": "镜头", "attributes": {"semantic": {"version": SS.SEMANTIC_VERSION,
                  "claims": []}}, "evidence_ids": [], "provenance": {"origin": "manual",
                  "actor_id": "user", "job_id": None}, "review_status": "reviewed",
                  "review_binding": {"content_sha256": "0" * 64, "actor_id": "user",
                                     "command_id": "cmd-1", "reviewed_at": "now",
                                     "task_revision": 2},
                  "start": {"seconds": {"n": "0", "d": "1"}},
                  "end": {"seconds": {"n": "1", "d": "1"}}}
    first = SS.annotation_review_digest(annotation, media_id="media-1",
                                       template_id="tpl-1", template_version=1)
    stripped = {key: value for key, value in annotation.items()
                if key not in ("review_status", "review_binding")}
    assert SS.annotation_review_digest(stripped, media_id="media-1",
                                       template_id="tpl-1", template_version=1) == first
    assert SS.annotation_review_digest(dict(stripped, text="改写"),
                                       media_id="media-1", template_id="tpl-1",
                                       template_version=1) != first


def test_digest_projection_declares_exclusions():
    assert SS.DIGEST_PROJECTIONS["claim"]["excludes"] == ["id"]
    assert SS.DIGEST_PROJECTIONS["claim_v2"]["excludes"] == ["id"]
    assert SS.DIGEST_PROJECTIONS["claim_v3"]["excludes"] == ["id"]
    assert set(SS.DIGEST_PROJECTIONS["annotation_review"]["excludes"]) == {
        "review_status", "review_binding"}
    assert "output_sha256" in SS.DIGEST_PROJECTIONS["model_run"]["excludes"]


def test_model_run_status_semantics():
    base = {"id": "run-1", "provider": "local", "model_name": "tiny", "model_version": "1",
            "weights_sha256": None, "license": "Apache-2.0", "preprocessing": {},
            "parameters": {}, "input_sha256": "0" * 64, "output_sha256": "1" * 64,
            "code_version": "unknown", "status": "success", "failure_reason": None,
            "score_kind": "none"}
    assert SS.validate_model_run(base)["status"] == "success"
    with pytest.raises(SchemaInvalid):
        SS.validate_model_run(dict(base, failure_reason="boom"))
    assert SS.validate_model_run(dict(base, status="not_run", output_sha256=None,
                                      failure_reason="weights missing"))["status"] == "not_run"
    with pytest.raises(SchemaInvalid):
        SS.validate_model_run(dict(base, status="partial"))


def test_review_decision_binds_its_content_hash():
    decision = {"actor_id": "local-user", "actor_type": "human", "decision": "accepted",
                "reviewed_at": "2026-11-05T00:00:00Z", "content_sha256": "a" * 64,
                "previous_content_sha256": None, "command_id": "cmd-1",
                "diff_summary": None}
    assert SS.validate_review_decision(decision, content_sha256="a" * 64)["decision"] == "accepted"
    with pytest.raises(SchemaInvalid) as exc:
        SS.validate_review_decision(decision, content_sha256="b" * 64)
    assert exc.value.code == "STALE_REVIEW_BINDING"


def test_published_examples_match_the_code_digests():
    """The contract's fixed JSON must be reproducible, not hand-written prose."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "docs" / "contracts"
    examples = json.loads((root / "visual-semantics-v1.examples.json").read_text(encoding="utf-8"))
    semantic = SS.validate_semantic(examples["annotation"]["attributes"]["semantic"],
                                    actor_type="human",
                                    model_run_ids=[examples["model_run"]["id"]],
                                    evidence_ids=examples["annotation"]["evidence_ids"])
    run = SS.validate_model_run(examples["model_run"])
    assert SS.semantic_digest(semantic) == examples["expected"]["semantic_digest"]
    assert SS.model_run_digest(run) == examples["expected"]["model_run_digest"]
    assert SS.claim_digest(semantic["claims"][0]) == examples["expected"]["claim_digests"][
        semantic["claims"][0]["id"]]
    assert SS.annotation_review_digest(examples["annotation"], media_id="media-0001",
                                       template_id="generic-segment",
                                       template_version=1) == examples["expected"][
        "annotation_review_digest"]
    projections = json.loads((root / "visual-semantics-v1.digest-projections.json").read_text(encoding="utf-8"))
    assert projections["claim"]["excludes"] == ["id"]


def test_published_v2_examples_separate_measurement_from_score():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "docs" / "contracts"
    examples = json.loads((root / "visual-semantics-v2.examples.json").read_text(encoding="utf-8"))
    semantic = SS.validate_semantic(examples["annotation"]["attributes"]["semantic"],
                                    actor_type="human",
                                    model_run_ids=[examples["model_run"]["id"]],
                                    evidence_ids=examples["annotation"]["evidence_ids"])
    measured = next(c for c in semantic["claims"] if c["claim_type"] == "measured")
    assert measured["measurement"]["value"] == -23.4
    assert measured["measurement"]["unit"] == "dBFS"
    assert measured["score"] == 0.9
    assert SS.semantic_digest(semantic) == examples["expected"]["semantic_digest"]
    for claim in semantic["claims"]:
        assert SS.claim_digest(claim) == examples["expected"]["claim_digests"][claim["id"]]
    assert SS.annotation_review_digest(examples["annotation"], media_id="med-example",
                                       template_id="tpl-example",
                                       template_version=1) == examples["expected"][
        "annotation_review_digest"]
