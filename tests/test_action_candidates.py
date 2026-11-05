"""Action candidates: frozen vocabulary, declared provider coverage, merging."""
from __future__ import annotations

import pytest

from trace_annotate.extractors import actions as AC


def _track(speeds, height=100.0, start_seconds=0):
    """A subject box moving right at `speed` pixels per second."""
    track = []
    x = 200.0
    for index, speed in enumerate(speeds):
        track.append({"frame_index": str(index),
                      "seconds": {"n": str(start_seconds + index), "d": "1"},
                      "box": [x, 100.0, x + 80.0, 100.0 + height]})
        x += speed
    return track


def test_slow_subject_movement_is_a_walk_candidate():
    result = AC.detect_translation_candidates(_track([50, 50, 50, 50]), subject_id="s1")
    assert result["status"] == "success"
    assert result["candidates"][0]["action"] == "走"
    assert result["candidates"][0]["score_kind"] == "rule_score"
    assert result["candidates"][0]["uncertainty"]


def test_fast_subject_movement_is_a_run_candidate():
    result = AC.detect_translation_candidates(_track([200, 200, 200, 200]), subject_id="s1")
    assert result["candidates"][0]["action"] == "跑"


def test_static_subject_produces_no_candidate():
    result = AC.detect_translation_candidates(_track([0, 0, 0, 0]), subject_id="s1")
    assert result["status"] == "success"
    assert result["candidates"] == []


def test_too_few_points_is_not_run():
    result = AC.detect_translation_candidates(_track([50, 50]), subject_id="s1")
    assert result["status"] == "not_run"


def test_provider_without_declared_class_is_not_run_for_that_action():
    coverage = AC.provider_coverage("kinetics400")
    assert coverage["supported"] == []
    assert "拿起" in coverage["unsupported"]


def test_build_candidate_rejects_action_outside_declared_coverage():
    with pytest.raises(ValueError, match="PROVIDER_DOES_NOT_SUPPORT"):
        AC.build_candidate("拿起", subject_id="s1", start_frame="10", end_frame="20",
                           evidence_ids=["art-10"], score=0.7, score_kind="rule_score",
                           provider="kinetics400")


def test_build_candidate_rejects_unknown_vocabulary():
    with pytest.raises(ValueError, match="UNSUPPORTED_ACTION_CLASS"):
        AC.build_candidate("跳舞", subject_id="s1", start_frame="1", end_frame="2",
                           evidence_ids=["art-1"], score=0.5, score_kind="rule_score",
                           provider="subject-translation-v1")


def test_candidate_requires_evidence():
    with pytest.raises(ValueError, match="ACTION_REQUIRES_EVIDENCE"):
        AC.build_candidate("走", subject_id="s1", start_frame="1", end_frame="2",
                           evidence_ids=[], score=0.5, score_kind="rule_score",
                           provider="subject-translation-v1")


def test_merge_keeps_windows_and_requires_same_subject_and_class():
    first = AC.build_candidate("走", subject_id="s1", start_frame="1", end_frame="4",
                               evidence_ids=["art-4"], score=0.5, score_kind="rule_score",
                               provider="subject-translation-v1")
    second = AC.build_candidate("走", subject_id="s1", start_frame="5", end_frame="9",
                                evidence_ids=["art-9"], score=0.6, score_kind="rule_score",
                                provider="subject-translation-v1")
    other = AC.build_candidate("跑", subject_id="s1", start_frame="10", end_frame="12",
                               evidence_ids=["art-12"], score=0.7, score_kind="rule_score",
                               provider="subject-translation-v1")
    merged = AC.merge_candidates([first, second, other])
    assert len(merged) == 2
    assert merged[0]["merged_windows"][0]["start_frame"] == "5"


def test_candidates_map_to_observable_claims():
    candidate = AC.build_candidate("走", subject_id="s1", start_frame="1", end_frame="4",
                                   evidence_ids=["art-4"], score=0.5,
                                   score_kind="rule_score",
                                   provider="subject-translation-v1")
    claims = AC.candidates_to_claims([candidate])
    assert claims[0]["claim_type"] == "observable"
    assert claims[0]["uncertainty_reason"]
