"""Same-frame hand/object measurements never become action assertions."""
from __future__ import annotations

import copy
import hashlib
from fractions import Fraction

import pytest

from trace_annotate.extractors.hand_object_evidence import build_hand_object_evidence
from trace_annotate.extractors.object_tracks import build_object_tracks


EVIDENCE = "a" * 64


def _ratio(value):
    value = Fraction(value)
    return {"n": str(value.numerator), "d": str(value.denominator)}


def _frame(frame=10, *, scene=0, pts=1, evidence=EVIDENCE, hands=True):
    result = {"scene_index": scene, "frame_index": str(frame),
              "timestamp": _ratio(pts), "evidence_sha256": evidence,
              "object": {"status": "success", "truncated": False,
                         "objects_detected": [{"class_id": 39}],
                         "detector_object_count": 1}}
    if hands:
        result["hands"] = {"status": "success", "hands": [_hand(frame, scene=scene,
                                                                    pts=pts, evidence=evidence)]}
    return result


def _hand(frame=10, *, scene=0, pts=1, evidence=EVIDENCE, outside=False):
    landmarks = [[12 + index * .5, 12 + index * .25, 0] for index in range(21)]
    return {"observation_id": f"handobs-{scene}-{frame}",
            "scene_index": scene, "frame_index": str(frame),
            "pts": _ratio(pts), "evidence_sha256": evidence,
            "landmarks_pixel": landmarks, "box_xyxy": [12, 12, 22, 17],
            "outside_frame": outside}


def _object(frame=10, x=20, *, scene=0, pts=1, evidence=EVIDENCE,
            observation_id=None):
    return {"scene_index": scene, "frame_index": str(frame),
            "pts": _ratio(pts), "class_id": 39, "class_label": "bottle",
            "box_xyxy": [x, 12, x + 10, 22], "score": .8,
            "evidence_sha256": evidence, "model_run_id": "run-object",
            "observation_id": observation_id or f"objobs-{scene}-{frame}-{x}"}


def _sampling(*observations, status="success"):
    tracking = build_object_tracks(list(observations))
    return {"status": status, "tracking": tracking}


def test_same_decoded_frame_yields_measurements_only_and_does_not_mutate_inputs():
    frame = _frame()
    sampling = _sampling(_object())
    before = copy.deepcopy((frame, sampling))
    result = build_hand_object_evidence([frame], sampling)
    assert (frame, sampling) == before
    assert result["status"] == "success"
    assert result["contract"] == "hand-object-frame-geometry-v1"
    assert result["coverage"]["sampled_module_coverage_complete"] is True
    assert result["coverage"]["comparable_pairs"] == 1
    assert result["coverage"]["omitted_pairs"] == 0
    pair = result["pairs"][0]
    assert pair["id"].startswith("hog-")
    assert pair["hand_observation_id"] == "handobs-0-10"
    assert pair["object_observation_id"] == "objobs-0-10-20"
    assert pair["frame_index"] == "10" and pair["pts"] == _ratio(1)
    assert pair["evidence_sha256"] == EVIDENCE
    assert pair["landmark_to_object_box_pixels"] == 0
    assert pair["hand_box_to_object_box_pixels"] == 0
    assert 0 < pair["box_iou"] < 1
    assert pair["review_status"] == "unreviewed"
    assert not ({"action", "touch", "person_ref", "subject_ref", "held_by"} & set(pair))
    assert "does not establish contact" in pair["uncertainty"]


@pytest.mark.parametrize("change", [
    {"frame": 11}, {"scene": 1}, {"pts": Fraction(11, 10)},
    {"evidence": "b" * 64},
])
def test_different_frame_scene_pts_or_evidence_never_pair(change):
    result = build_hand_object_evidence([_frame()], _sampling(_object(**change)))
    assert result["status"] == "success"
    assert result["pairs"] == []
    assert result["reason"] == "no_co_visible_observations"


def test_outside_frame_and_ambiguous_tracks_are_not_compared():
    frame = _frame()
    frame["hands"]["hands"][0]["outside_frame"] = True
    result = build_hand_object_evidence([frame], _sampling(_object()))
    assert result["pairs"] == []
    assert result["coverage"]["skipped_outside_frame_hands"] == 1
    frame["hands"]["hands"][0]["outside_frame"] = False
    sampling = _sampling(_object())
    sampling["tracking"]["tracks"][0]["ambiguity"] = True
    sampling["tracking"]["tracks"][0]["ambiguity_reason"] = "near_tied_association"
    result = build_hand_object_evidence([frame], sampling)
    assert result["pairs"] == []
    assert result["coverage"]["skipped_ambiguous_tracks"] == 1


def test_missing_provider_and_empty_success_are_explicitly_distinct():
    assert build_hand_object_evidence([_frame(hands=False)], _sampling(_object()))["status"] == "not_run"
    frame = _frame()
    frame["hands"] = {"status": "not_run", "hands": [], "reason": "weights_missing"}
    assert build_hand_object_evidence([frame], _sampling(_object()))["reason"] == "hand_landmarks_unavailable"
    assert build_hand_object_evidence([_frame()],
                                      {"status": "not_run", "tracking": {"status": "not_run"}})["reason"] == "object_tracking_unavailable"
    empty_frame = _frame()
    empty_frame["object"]["objects_detected"] = []
    empty_frame["object"]["detector_object_count"] = 0
    empty = build_hand_object_evidence([empty_frame], _sampling())
    assert empty["status"] == "success"
    assert empty["reason"] == "no_co_visible_observations"
    assert empty["pairs"] == []


def test_nearest_three_objects_are_kept_with_explicit_omission_and_stable_order():
    frame = _frame()
    observations = [_object(x=x) for x in (60, 20, 40, 30)]
    first = build_hand_object_evidence([frame], _sampling(*observations))
    second = build_hand_object_evidence([frame], _sampling(*reversed(observations)))
    assert first == second
    assert first["coverage"]["comparable_pairs"] == 4
    assert first["coverage"]["emitted_pairs"] == 3
    assert first["coverage"]["omitted_pairs"] == 1
    assert first["coverage"]["truncated"] is True
    assert {pair["object_observation_id"] for pair in first["pairs"]} == {
        "objobs-0-10-20", "objobs-0-10-30", "objobs-0-10-40"}


@pytest.mark.parametrize("change,match", [
    (lambda hand: hand.update(pts=_ratio(2)), "differs from its decoded frame"),
    (lambda hand: hand["landmarks_pixel"][0].__setitem__(0, float("nan")), "finite"),
    (lambda hand: hand["landmarks_pixel"][0].__setitem__(0, 1000), "differ from their pixel box"),
    (lambda hand: hand.update(outside_frame="false"), "outside_frame"),
])
def test_invalid_hand_identity_or_geometry_fails_closed(change, match):
    frame = _frame()
    change(frame["hands"]["hands"][0])
    with pytest.raises(ValueError, match=match):
        build_hand_object_evidence([frame], _sampling(_object()))


def test_noncanonical_object_pts_and_invalid_object_geometry_fail_closed():
    sampling = _sampling(_object())
    observation = sampling["tracking"]["tracks"][0]["observations"][0]
    observation["pts"] = {"n": "2", "d": "2"}
    with pytest.raises(ValueError, match="reduced rational"):
        build_hand_object_evidence([_frame()], sampling)
    observation["pts"] = _ratio(1)
    observation["box_xyxy"][2] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        build_hand_object_evidence([_frame()], sampling)


def test_partial_source_coverage_is_not_reported_as_complete():
    first = _frame()
    second = _frame(frame=11, pts=Fraction(11, 10))
    second["hands"] = {"status": "failed", "hands": [], "reason": "worker_timeout"}
    result = build_hand_object_evidence([first, second],
                                        _sampling(_object(), status="partial"))
    assert result["status"] == "success"
    assert result["coverage"]["hand_successful_keyframes"] == 1
    assert result["coverage"]["sampled_module_coverage_complete"] is False


def test_object_truncation_is_counted_without_inventing_unseen_pairs():
    frame = _frame()
    frame["object"].update(truncated=True, detector_object_count=5,
                           objects_detected=[{"class_id": 39}])
    result = build_hand_object_evidence([frame], _sampling(_object()))
    coverage = result["coverage"]
    assert result["status"] == "success"
    assert len(result["pairs"]) == 1
    assert coverage["sampled_module_coverage_complete"] is False
    assert coverage["object_truncated_keyframes"] == 1
    assert coverage["object_detection_omissions"] == 4
    assert coverage["object_detection_omissions_unknown_keyframes"] == 0
    assert coverage["comparable_pairs"] == 1
    assert coverage["omitted_pairs"] == 0
    frame["object"]["truncated"] = False
    inconsistent = build_hand_object_evidence([frame], _sampling(_object()))
    assert inconsistent["coverage"]["sampled_module_coverage_complete"] is False


def test_not_run_retains_known_sample_and_hand_counts():
    frame = _frame()
    result = build_hand_object_evidence(
        [frame], {"status": "not_run", "tracking": {"status": "not_run"}})
    assert result["status"] == "not_run"
    assert result["coverage"]["total_keyframes"] == 1
    assert result["coverage"]["hand_successful_keyframes"] == 1
    assert result["coverage"]["hand_observations"] == 1
    assert result["coverage"]["object_successful_keyframes"] == 1
    assert result["coverage"]["object_observations"] == 1
    assert result["coverage"]["eligible_object_observations"] == 0
    assert result["coverage"]["sampled_module_coverage_complete"] is False
    unavailable_tracking = build_hand_object_evidence(
        [frame], {"status": "success", "tracking": {"status": "not_run"}})
    assert unavailable_tracking["status"] == "not_run"
    assert unavailable_tracking["coverage"]["sampled_module_coverage_complete"] is False
    legacy = build_hand_object_evidence([_frame(hands=False)], _sampling(_object()))
    assert legacy["status"] == "not_run"
    assert legacy["coverage"]["total_keyframes"] == 1
    assert legacy["coverage"]["hand_successful_keyframes"] == 0
    assert legacy["coverage"]["object_observations"] == 1


def test_keyframe_budget_is_bounded():
    with pytest.raises(ValueError, match="keyframe budget"):
        build_hand_object_evidence([{}] * 513, _sampling())


def test_global_pair_budget_is_deterministic_and_reports_omissions():
    frames, tracks = [], []
    for index in range(342):
        frame_index = index + 10
        evidence = hashlib.sha256(str(index).encode("ascii")).hexdigest()
        pts = Fraction(frame_index, 10)
        frames.append(_frame(frame_index, pts=pts, evidence=evidence))
        for x in (20, 30, 40, 50):
            observation = _object(frame_index, x, pts=pts, evidence=evidence)
            observation["evidence_id"] = observation.pop("evidence_sha256")
            tracks.append({"scene_index": 0, "ambiguity": False,
                           "termination_reason": "end_of_scene",
                           "observations": [observation]})
    sampling = {"status": "success", "tracking": {"status": "success", "tracks": tracks}}
    forward = build_hand_object_evidence(frames, sampling)
    reverse = build_hand_object_evidence(list(reversed(frames)),
                                         {"status": "success", "tracking": {
                                             "status": "success", "tracks": list(reversed(tracks))}})
    assert forward == reverse
    assert forward["coverage"]["comparable_pairs"] == 1368
    assert forward["coverage"]["emitted_pairs"] == 1024
    assert forward["coverage"]["omitted_pairs"] == 344
    assert forward["coverage"]["truncated"] is True
