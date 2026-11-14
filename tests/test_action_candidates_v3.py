"""Evidence-bound action-candidate-v3 contract and honest per-class outcomes."""
from __future__ import annotations

import pytest

from trace_annotate.extractors import actions as AC


def _v3(action="走", **overrides):
    values = {"subject_ref": "subject-1", "start_frame": "1", "end_frame": "4",
              "window_pts": {"start": {"n": "1", "d": "1"},
                             "end": {"n": "4", "d": "1"}},
              "evidence_ids": ["art-1"], "score": .7, "score_kind": "rule_score",
              "provider": "subject-translation-v1", "model_run_id": "run-actions",
              "uncertainty": "requires human review"}
    values.update(overrides)
    return AC.build_candidate_v3(action, **values)


def test_v3_walk_binds_subject_pts_evidence_provider_and_run():
    candidate = _v3()
    assert candidate["contract_version"] == AC.ACTION_CANDIDATE_VERSION
    assert candidate["subject_ref"] == "subject-1"
    assert candidate["window_pts"]["start"] == {"n": "1", "d": "1"}
    assert candidate["evidence_ids"] == ["art-1"]
    assert candidate["provider"] == "subject-translation-v1"
    assert candidate["model_run_id"] == "run-actions"


def test_v3_object_and_recipient_requirements_fail_closed():
    with pytest.raises(ValueError, match="ACTION_REQUIRES_OBJECT"):
        _v3("拿起", provider="kinetics400")
    with pytest.raises(ValueError, match="ACTION_REQUIRES_RECIPIENT"):
        _v3("递交", provider="kinetics400", object_ref="object-1")
    with pytest.raises(ValueError, match="ACTION_REQUIRES_DISTINCT_RECIPIENT"):
        _v3("递交", provider="kinetics400", object_ref="object-1", recipient_ref="subject-1")


def test_v3_rejects_unordered_ranges_missing_evidence_and_missing_run():
    with pytest.raises(ValueError, match="INVALID_ACTION_FRAME_RANGE"):
        _v3(start_frame="5", end_frame="4")
    with pytest.raises(ValueError, match="INVALID_ACTION_PTS"):
        _v3(window_pts={"start": {"n": "4", "d": "1"}, "end": {"n": "1", "d": "1"}})
    with pytest.raises(ValueError, match="INVALID_ACTION_PTS"):
        _v3(window_pts={"start": {"n": "02", "d": "04"}, "end": {"n": "2", "d": "1"}})
    with pytest.raises(ValueError, match="ACTION_REQUIRES_EVIDENCE"):
        _v3(evidence_ids=[])
    with pytest.raises(ValueError, match="INVALID_ACTION_REFERENCE"):
        _v3(model_run_id="")
    with pytest.raises(ValueError, match="INVALID_ACTION_FRAME_RANGE"):
        _v3(start_frame="01")
    with pytest.raises(ValueError, match="INVALID_ACTION_SCORE"):
        _v3(score=True)
    with pytest.raises(ValueError, match="INVALID_ACTION_REFERENCE"):
        _v3(evidence_ids=[{"bad": "shape"}])


def test_every_class_has_success_unknown_or_provider_unavailable():
    candidates = [_v3()]
    outcomes = AC.action_class_results("subject-translation-v1", status="success",
                                      candidates=candidates)
    assert set(outcomes) == set(AC.ACTION_VOCAB)
    assert outcomes["走"]["status"] == "success"
    assert outcomes["跑"] == {"action": "跑", "status": "unknown",
                              "reason": "no_candidate_met_threshold",
                              "provider": "subject-translation-v1", "candidate_count": 0,
                              "requires": ["subject_ref"]}
    for action in ("拿起", "放下", "打开", "关闭", "递交"):
        assert outcomes[action]["status"] == "not_run"
        assert outcomes[action]["reason"] == "provider_unavailable"
    with pytest.raises(ValueError, match="INVALID_ACTION_RESULT_STATUS"):
        AC.action_class_results("subject-translation-v1", status="complete")


def test_v3_merge_refuses_adjacent_frame_numbers_across_a_pts_hole():
    first = _v3(start_frame="0", end_frame="0",
                window_pts={"start": {"n": "0", "d": "1"},
                            "end": {"n": "1", "d": "1"}},
                evidence_ids=["artifact-one"])
    second = _v3(start_frame="1", end_frame="1",
                 window_pts={"start": {"n": "100", "d": "1"},
                             "end": {"n": "101", "d": "1"}},
                 evidence_ids=["artifact-two"])
    assert len(AC.merge_candidates([first, second])) == 2


def test_v3_merge_accepts_bounded_pts_continuity_and_rejects_bad_gap_budget():
    first = _v3(start_frame="0", end_frame="0",
                window_pts={"start": {"n": "0", "d": "1"},
                            "end": {"n": "1", "d": "1"}},
                evidence_ids=["artifact-one"])
    second = _v3(start_frame="1", end_frame="1",
                 window_pts={"start": {"n": "3", "d": "2"},
                             "end": {"n": "2", "d": "1"}},
                 evidence_ids=["artifact-two"])
    merged = AC.merge_candidates([first, second])
    assert len(merged) == 1
    assert merged[0]["window_pts"]["end"] == {"n": "2", "d": "1"}
    with pytest.raises(ValueError, match="max_pts_gap_seconds"):
        AC.merge_candidates([first, second], max_pts_gap_seconds=float("nan"))


def test_strict_translation_rejects_reversed_or_mismatched_pts_identity():
    track = []
    for frame, (seconds, pts) in enumerate(((0, 2), (1, 1), (2, 0))):
        track.append({"frame_index": str(frame), "seconds": seconds,
                      "pts": {"n": str(pts), "d": "1"},
                      "box": [frame * 50, 0, frame * 50 + 80, 100],
                      "evidence_sha256": f"{frame + 1:064x}"})
    result = AC.detect_translation_candidates(
        track, subject_ref="subject-1", model_run_id="run-actions")
    assert result == {"status": "not_run", "candidates": [],
                      "reason": "non_monotonic_frame_or_pts_range"}

    for point in track:
        point["pts"] = {"n": point["frame_index"], "d": "1"}
    track[-1]["seconds"] = 2.25
    result = AC.detect_translation_candidates(
        track, subject_ref="subject-1", model_run_id="run-actions")
    assert result == {"status": "not_run", "candidates": [],
                      "reason": "pts_seconds_identity_mismatch"}


@pytest.mark.parametrize("track,min_samples,reason", [
    ([], 0, "invalid_min_samples"),
    ([], True, "invalid_min_samples"),
    ("not-a-track", 3, "invalid_or_excessive_track_samples"),
    ([None, None], 2, "insufficient_evidence_samples"),
])
def test_translation_input_limits_fail_closed_without_index_errors(track, min_samples, reason):
    result = AC.detect_translation_candidates(track, min_samples=min_samples)
    assert result == {"status": "not_run", "candidates": [], "reason": reason}


def test_translation_and_merge_enforce_public_resource_budgets(monkeypatch):
    monkeypatch.setattr(AC, "MAX_TRANSLATION_SAMPLES", 1)
    assert AC.detect_translation_candidates([{}, {}])["reason"] == (
        "invalid_or_excessive_track_samples")
    monkeypatch.setattr(AC, "MAX_ACTION_CANDIDATES", 1)
    with pytest.raises(ValueError, match="candidate budget"):
        AC.merge_candidates([_v3(), _v3(start_frame="5", end_frame="6",
            window_pts={"start": {"n": "5", "d": "1"},
                        "end": {"n": "6", "d": "1"}})])
    monkeypatch.setattr(AC, "MAX_ACTION_CANDIDATES", 10)
    monkeypatch.setattr(AC, "MAX_ACTION_CANDIDATE_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        AC.merge_candidates([_v3()])
    with pytest.raises(ValueError, match="must be an object"):
        AC.merge_candidates([None])
