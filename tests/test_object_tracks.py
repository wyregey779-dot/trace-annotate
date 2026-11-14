"""Object tracking and relation evidence stay below the action layer."""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from trace_annotate.extractors import object_tracks as OT


def obs(frame, x, *, scene=0, class_id=39, label="bottle", token=None,
        pts=None, model_run_id="run-nanodet", observation_id=None):
    default_pts = Fraction(frame, 10)
    value = {"scene_index": scene, "frame_index": str(frame),
            "pts": pts if pts is not None else {
                "n": str(default_pts.numerator), "d": str(default_pts.denominator)},
            "class_id": class_id, "class_label": label,
            "box_xyxy": [x, 0, x + 10, 10], "score": .8,
            "score_kind": "model_score_uncalibrated",
            "evidence_id": token or f"ev-frame-{scene}-{frame}",
            "model_run_id": model_run_id}
    if observation_id is not None:
        value["observation_id"] = observation_id
    return value


def subject(name, boxes, *, scene=0):
    return {"subject_ref": name, "scene_index": scene,
            "observations": [
                {"frame_index": str(frame), "pts": {
                    "n": str(Fraction(frame, 10).numerator),
                    "d": str(Fraction(frame, 10).denominator)},
                 "box_xyxy": box, "evidence_id": f"ev-frame-{scene}-{frame}"}
                for frame, box in boxes]}


def test_track_result_is_input_order_independent_and_has_stable_identity():
    values = [obs(0, 0), obs(1, 4), obs(2, 8)]
    forward = OT.build_object_tracks(values)
    reverse = OT.build_object_tracks(list(reversed(values)))
    assert forward == reverse
    assert len(forward["tracks"]) == 1
    track = forward["tracks"][0]
    assert track["track_id"].startswith("objtrk-")
    assert [item["frame_index"] for item in track["observations"]] == ["0", "1", "2"]
    assert all(item["track_id"] == track["track_id"] for item in track["observations"])
    assert track["termination_reason"] == "end_of_scene"


def test_two_near_tied_crossing_objects_split_instead_of_switching_identity():
    values = [obs(0, 0, token="a0"), obs(0, 20, token="b0"),
              obs(1, 9, token="a1"), obs(1, 11, token="b1")]
    result = OT.build_object_tracks(values, max_centre_distance=3, ambiguity_margin=.12)
    assert result["ambiguous_tracks"] >= 2
    old = [track for track in result["tracks"] if track["observations"][0]["frame_index"] == "0"]
    new = [track for track in result["tracks"] if track["observations"][0]["frame_index"] == "1"]
    assert len(old) == 2 and len(new) == 2
    assert all(track["termination_reason"] == "ambiguous_association" for track in old)
    assert all(track["ambiguity"] for track in old + new)
    assert all(len(track["observations"]) == 1 for track in old + new)


def test_short_occlusion_is_retained_but_late_reentry_starts_a_new_track():
    values = [obs(0, 0), obs(1, 3), obs(3, 7), obs(8, 9)]
    result = OT.build_object_tracks(values, max_gap_frames=2)
    assert len(result["tracks"]) == 2
    assert [item["frame_index"] for item in result["tracks"][0]["observations"]] == ["0", "1", "3"]
    assert result["tracks"][0]["termination_reason"] == "gap_exceeded"
    assert result["tracks"][1]["observations"][0]["frame_index"] == "8"


def test_sparse_frame_stride_is_allowed_only_with_an_explicit_time_bound():
    values = [obs(0, 0, pts={"n": "0", "d": "1"}),
              obs(90, 2, pts={"n": "3", "d": "1"}),
              obs(300, 4, pts={"n": "10", "d": "1"})]
    result = OT.build_object_tracks(values, max_gap_frames=300, max_gap_seconds=5)
    assert [len(track["observations"]) for track in result["tracks"]] == [2, 1]
    assert result["tracks"][0]["termination_reason"] == "time_gap_exceeded"
    assert result["parameters"]["max_gap_frames"] == 300
    assert result["parameters"]["max_gap_seconds"] == 5.0
    with pytest.raises(ValueError, match="max_gap_seconds"):
        OT.build_object_tracks(values, max_gap_seconds=float("inf"))


def test_tracks_never_merge_observations_from_different_model_runs():
    first = obs(0, 0, model_run_id="run-one")
    second = obs(1, 1, model_run_id="run-two")
    result = OT.build_object_tracks([first, second])
    assert len(result["tracks"]) == 2
    assert {track["model_run_id"] for track in result["tracks"]} == {"run-one", "run-two"}
    forged = copy.deepcopy(result["tracks"][0])
    foreign = copy.deepcopy(result["tracks"][1]["observations"][0])
    foreign["track_id"] = forged["track_id"]
    forged["observations"].append(foreign)
    with pytest.raises(ValueError, match="retain scene and class identity"):
        OT.relation_evidence([forged], [])


def test_hard_cut_and_class_identity_are_absolute_track_boundaries():
    values = [obs(0, 0, scene=0), obs(1, 1, scene=1),
              obs(1, 1, scene=1, class_id=41, label="cup")]
    result = OT.build_object_tracks(values)
    assert len(result["tracks"]) == 3
    scene_zero = next(track for track in result["tracks"] if track["scene_index"] == 0)
    assert scene_zero["termination_reason"] == "hard_cut"
    assert {track["class_id"] for track in result["tracks"] if track["scene_index"] == 1} == {39, 41}


def test_coco_class_label_and_identifier_provenance_are_fail_closed():
    forged = obs(0, 0, class_id=39, label="cup")
    with pytest.raises(ValueError, match="frozen COCO-80"):
        OT.build_object_tracks([forged])
    for field, invalid in (("model_run_id", "run with spaces"),
                           ("observation_id", "9-not-letter-led"),
                           ("evidence_id", "e" * 97)):
        value = obs(0, 0)
        value[field] = invalid
        with pytest.raises(ValueError, match="bounded ASCII"):
            OT.build_object_tracks([value])
    hashed = obs(0, 0)
    hashed.pop("evidence_id")
    hashed["evidence_sha256"] = "0" * 64
    hashed_tracks = OT.build_object_tracks([hashed])
    assert hashed_tracks["observations"] == 1
    matching_subject = subject("subject-a", [(0, [0, 0, 20, 20])])
    matching_subject["observations"][0]["evidence_id"] = "0" * 64
    assert OT.relation_evidence(
        hashed_tracks["tracks"], [matching_subject])["status"] == "success"
    hashed["evidence_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        OT.build_object_tracks([hashed])
    bad_score = obs(0, 0)
    bad_score["score"] = 1.1
    with pytest.raises(ValueError, match="between zero and one"):
        OT.build_object_tracks([bad_score])
    bad_kind = obs(0, 0)
    bad_kind["score_kind"] = "probability"
    with pytest.raises(ValueError, match="score_kind"):
        OT.build_object_tracks([bad_kind])


def test_track_time_axes_reject_duplicate_or_non_increasing_frame_and_pts():
    for noncanonical in (
            {"n": "2", "d": "20"},
            {"n": "01", "d": "10"},
            {"n": "-1", "d": "-10"},
    ):
        with pytest.raises(ValueError, match="finite rational"):
            OT.build_object_tracks([obs(0, 0, pts=noncanonical)])
    with pytest.raises(ValueError, match="strictly increasing"):
        OT.build_object_tracks([
            obs(0, 0, pts={"n": "1", "d": "10"}),
            obs(1, 1, pts={"n": "1", "d": "10"}),
        ])
    with pytest.raises(ValueError, match="same PTS"):
        OT.build_object_tracks([
            obs(0, 0, token="frame-a", pts={"n": "0", "d": "1"}),
            obs(0, 20, token="frame-b", pts={"n": "1", "d": "10"}),
        ])

    track = OT.build_object_tracks([obs(0, 0)])["tracks"][0]
    duplicate = copy.deepcopy(track["observations"][0])
    duplicate["observation_id"] = "object-observation-two"
    duplicate["evidence_id"] = "object-evidence-two"
    track["observations"].append(duplicate)
    with pytest.raises(ValueError, match="strictly increasing"):
        OT.relation_evidence([track], [])

    duplicate_subject = subject("subject-a", [(0, [0, 0, 20, 20]),
                                                (0, [1, 0, 21, 20])])
    duplicate_subject["observations"][1]["evidence_id"] = "ev-subject-a-duplicate"
    with pytest.raises(ValueError, match="strictly increasing"):
        OT.relation_evidence([], [duplicate_subject])
    repeated_pts = subject("subject-a", [(0, [0, 0, 20, 20]),
                                          (1, [1, 0, 21, 20])])
    repeated_pts["observations"][1]["pts"] = {"n": "0", "d": "1"}
    with pytest.raises(ValueError, match="strictly increasing"):
        OT.relation_evidence([], [repeated_pts])


def test_subject_and_track_references_reject_unbounded_or_non_ascii_ids():
    with pytest.raises(ValueError, match="subject_ref"):
        OT.relation_evidence([], [subject("subject name", [])])
    bad_evidence = subject("subject-a", [(0, [0, 0, 20, 20])])
    bad_evidence["observations"][0]["evidence_id"] = "证据"
    with pytest.raises(ValueError, match="bounded ASCII"):
        OT.relation_evidence([], [bad_evidence])
    track = OT.build_object_tracks([obs(0, 0)])["tracks"][0]
    track["track_id"] = "track id"
    with pytest.raises(ValueError, match="track_id"):
        OT.relation_evidence([track], [])


def test_static_proximity_is_not_held_or_control_transfer():
    objects = OT.build_object_tracks([obs(0, 10), obs(1, 10), obs(2, 10)])["tracks"]
    person = subject("subject-a", [(0, [9, 0, 29, 20]),
                                    (1, [9, 0, 29, 20]),
                                    (2, [9, 0, 29, 20])])
    result = OT.relation_evidence(objects, [person])
    kinds = [item["relation"] for item in result["relations"]]
    assert "near" in kinds and "contact_candidate" in kinds
    assert "co_moving" not in kinds
    assert "held_by" not in kinds
    assert "control_transfer" not in kinds
    assert result["scope"] == "geometric_relation_candidates_not_action_recognition"
    assert "pick_up" in result["unsupported_action_assertions"]


def test_same_motion_and_contact_produce_relation_evidence_not_an_action():
    objects = OT.build_object_tracks([obs(frame, 10 * frame) for frame in range(4)])["tracks"]
    person = subject("subject-a", [(frame, [10 * frame + 9, 0, 10 * frame + 49, 20])
                                    for frame in range(4)])
    result = OT.relation_evidence(objects, [person])
    kinds = {item["relation"] for item in result["relations"]}
    assert {"near", "contact_candidate", "co_moving", "held_by"} <= kinds
    assert all("action" not in item for item in result["relations"])
    held = next(item for item in result["relations"] if item["relation"] == "held_by")
    assert held["review_status"] == "unreviewed"
    assert held["score_kind"] == "rule_score"
    assert held["uncertainty"]


def test_control_transfer_needs_early_giver_release_and_late_recipient_control():
    objects = OT.build_object_tracks([obs(frame, 10 * frame) for frame in range(6)])["tracks"]
    giver_boxes = []
    recipient_boxes = []
    for frame in range(6):
        x = 10 * frame
        # Giver overlaps through frame 3, then has a real image-plane gap.
        giver_boxes.append((frame, [x + (9 if frame < 4 else 11), 0,
                                    x + (49 if frame < 4 else 51), 20]))
        # Recipient is separate in frames 0-1 and overlaps from frame 2 onward.
        recipient_boxes.append((frame, [x + (-41 if frame < 2 else -39), 0,
                                        x + (-1 if frame < 2 else 1), 20]))
    result = OT.relation_evidence(objects, [subject("giver", giver_boxes),
                                            subject("recipient", recipient_boxes)])
    transfer = [item for item in result["relations"] if item["relation"] == "control_transfer"]
    assert len(transfer) == 1
    assert transfer[0]["subject_ref"] == "giver"
    assert transfer[0]["recipient_ref"] == "recipient"
    assert transfer[0]["object_ref"] == objects[0]["track_id"]
    assert transfer[0]["review_status"] == "unreviewed"


def test_no_release_means_no_control_transfer_even_when_two_people_are_near():
    objects = OT.build_object_tracks([obs(frame, 10 * frame) for frame in range(6)])["tracks"]
    first = subject("first", [(frame, [10 * frame + 9, 0, 10 * frame + 49, 20])
                               for frame in range(6)])
    second = subject("second", [(frame, [10 * frame - 39, 0, 10 * frame + 1, 20])
                                 for frame in range(6)])
    result = OT.relation_evidence(objects, [first, second])
    assert not [item for item in result["relations"] if item["relation"] == "control_transfer"]


@pytest.mark.parametrize("kwargs", [
    {"near_gap": float("nan")},
    {"near_gap": -0.1},
    {"contact_gap": float("inf")},
    {"contact_gap": -0.1},
    {"near_gap": 0.1, "contact_gap": 0.2},
    {"held_min_samples": 3.0},
    {"held_min_samples": True},
])
def test_relation_thresholds_are_finite_ordered_and_typed(kwargs):
    with pytest.raises(ValueError):
        OT.relation_evidence([], [], **kwargs)


def test_public_track_and_relation_apis_enforce_input_budgets(monkeypatch):
    monkeypatch.setattr(OT, "MAX_TRACK_OBSERVATIONS", 2)
    with pytest.raises(ValueError, match="observation budget"):
        OT.build_object_tracks([obs(0, 0), obs(1, 1), obs(2, 2)])
    monkeypatch.setattr(OT, "MAX_TRACKS", 1)
    with pytest.raises(ValueError, match="track budget"):
        OT.relation_evidence([{}, {}], [])
    monkeypatch.setattr(OT, "MAX_TRACKS", 10)
    monkeypatch.setattr(OT, "MAX_SUBJECT_TRACKS", 10)
    monkeypatch.setattr(OT, "MAX_RELATION_PAIRS", 1)
    with pytest.raises(ValueError, match="pair budget"):
        OT.relation_evidence([{}], [{}, {}])


def test_relation_work_and_serialized_output_budgets_fail_before_growth(monkeypatch):
    objects = OT.build_object_tracks([obs(frame, frame) for frame in range(3)])["tracks"]
    person = subject("subject-budget", [
        (frame, [frame, 0, frame + 20, 20]) for frame in range(3)])
    monkeypatch.setattr(OT, "MAX_RELATION_PAIR_SAMPLE_REFERENCES", 2)
    with pytest.raises(ValueError, match="pair sample budget"):
        OT.relation_evidence(objects, [person])
    monkeypatch.setattr(OT, "MAX_RELATION_PAIR_SAMPLE_REFERENCES", 10)
    monkeypatch.setattr(OT, "MAX_RELATION_OUTPUT_SAMPLE_REFERENCES", 1)
    with pytest.raises(ValueError, match="sample output budget"):
        OT.relation_evidence(objects, [person])
    monkeypatch.setattr(OT, "MAX_RELATION_OUTPUT_SAMPLE_REFERENCES", 10)
    monkeypatch.setattr(OT, "MAX_RELATION_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="serialized output budget"):
        OT.relation_evidence(objects, [person])


def test_pts_mismatch_and_cross_scene_inputs_do_not_create_relations():
    objects = OT.build_object_tracks([obs(frame, frame) for frame in range(3)])["tracks"]
    other_scene = subject("subject-other-scene", [
        (frame, [frame, 0, frame + 20, 20]) for frame in range(3)], scene=1)
    cross_scene = OT.relation_evidence(objects, [other_scene])
    assert cross_scene["status"] == "unknown"
    assert cross_scene["reason"] == "insufficient_evidence"

    misaligned = subject("subject-misaligned", [
        (frame, [frame, 0, frame + 20, 20]) for frame in range(3)])
    for frame, observation in enumerate(misaligned["observations"]):
        shifted = Fraction(frame + 1, 10)
        observation["pts"] = {
            "n": str(shifted.numerator), "d": str(shifted.denominator)}
    mismatched = OT.relation_evidence(objects, [misaligned])
    assert mismatched["status"] == "unknown"
    assert mismatched["relations"] == []


def test_equal_frame_and_pts_from_different_evidence_cannot_share_geometry():
    objects = OT.build_object_tracks([obs(frame, frame) for frame in range(3)])["tracks"]
    person = subject("subject-different-decode", [
        (frame, [frame, 0, frame + 20, 20]) for frame in range(3)])
    for frame, observation in enumerate(person["observations"]):
        observation["evidence_id"] = f"ev-other-decode-{frame}"
    result = OT.relation_evidence(objects, [person])
    assert result["status"] == "unknown"
    assert result["relations"] == []


def test_missing_or_ambiguous_evidence_returns_unknown_without_guessing():
    empty = OT.relation_evidence([], [])
    assert empty["status"] == "unknown" and empty["reason"] == "insufficient_evidence"
    ambiguous = OT.build_object_tracks(
        [obs(0, 0, token="a0"), obs(0, 20, token="b0"),
         obs(1, 9, token="a1"), obs(1, 11, token="b1")],
        max_centre_distance=3, ambiguity_margin=.12)["tracks"]
    assert all(track["ambiguity"] for track in ambiguous)
    assert OT.relation_evidence(ambiguous, [subject("s", [(1, [0, 0, 30, 20])])])[
        "status"] == "unknown"
