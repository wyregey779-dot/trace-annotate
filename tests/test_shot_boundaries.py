"""Boundary events from synthetic frames: cut, flash, fade, dissolve, no-op."""
from __future__ import annotations

import numpy as np
import pytest

from trace_annotate.extractors import shot_boundaries as SB


def _samples(colors, size=(48, 64)):
    """BGR frames with frame-table-like identity."""
    frames = []
    for index, color in enumerate(colors):
        frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        frame[..., 0] = color[0]
        frame[..., 1] = color[1]
        frame[..., 2] = color[2]
        frames.append({"frame_index": str(index), "frame": frame,
                       "seconds": {"n": str(index), "d": "1"}})
    return frames


BLUE = (200, 20, 20)
RED = (20, 20, 200)
GRAY = (120, 120, 120)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _kinds(result):
    return [event["kind"] for event in result["events"]]


def test_hard_cut_anchors_on_new_shot_first_frame():
    result = SB.detect_shot_boundaries(_samples([BLUE] * 6 + [RED] * 6))
    assert _kinds(result) == ["cut"]
    event = result["events"][0]
    assert event["anchor_frame"] == "6"
    assert event["previous_frame"] == "5"
    assert event["split_point"] == "6"
    assert event["uncertainty"] is None


def test_single_frame_flash_does_not_split_the_shot():
    result = SB.detect_shot_boundaries(_samples([GRAY] * 5 + [WHITE] + [GRAY] * 5))
    assert _kinds(result) == ["flash"]
    event = result["events"][0]
    assert event["anchor_frame"] == "5"
    assert event["split_point"] is None
    assert event["uncertainty"]


def test_fade_in_from_black_owns_an_interval():
    colors = [BLACK] * 3 + [(60, 60, 60), (120, 120, 120), (180, 180, 180),
                            (240, 240, 240)] + [(240, 240, 240)] * 3
    result = SB.detect_shot_boundaries(_samples(colors))
    assert _kinds(result) == ["fade_in"]
    event = result["events"][0]
    interval = event["interval"]
    assert interval["start_frame"] == "0"
    assert event["representative_frame"] == event["anchor_frame"]
    assert event["uncertainty_frames"] >= 2


def test_dissolve_between_two_scenes_is_a_transition_not_two_cuts():
    start_color = (150, 20, 10)
    end_color = (10, 20, 240)
    blended = []
    for step in range(1, 6):
        alpha = step / 6.0
        blended.append(tuple(int((1 - alpha) * start_color[i] + alpha * end_color[i])
                             for i in range(3)))
    colors = [start_color, start_color] + blended + [end_color, end_color]
    result = SB.detect_shot_boundaries(_samples(colors))
    kinds = _kinds(result)
    assert "cut" not in kinds
    assert "dissolve" in kinds or "unknown_transition" in kinds
    event = [item for item in result["events"] if item["kind"] in {"dissolve", "unknown_transition"}][0]
    assert event["interval"]["start_frame"] != event["interval"]["end_frame"]


def test_small_global_brightness_drift_is_not_a_boundary():
    colors = [tuple([20 + step * 4] * 3) for step in range(12)]
    result = SB.detect_shot_boundaries(_samples(colors))
    assert result["events"] == []


def test_empty_sequence_returns_no_events():
    result = SB.detect_shot_boundaries([])
    assert result["events"] == []
    assert result["params"]["cut_threshold"] == SB.CUT_THRESHOLD


def test_events_map_to_observable_claims_with_evidence():
    result = SB.detect_shot_boundaries(_samples([BLUE] * 4 + [RED] * 4))
    claims = SB.events_to_claims(result["events"], lambda frame: f"art-{frame}" if frame else None)
    assert len(claims) == 1
    assert claims[0]["claim_type"] == "observable"
    assert set(claims[0]["evidence_ids"]) == {"art-4", "art-3"}
    assert claims[0]["score_kind"] == "rule_score"


def test_track_reports_real_frame_identity_not_average_fps():
    result = SB.detect_shot_boundaries(_samples([BLUE, BLUE, RED]))
    track = result["track"]
    assert [entry["frame_index"] for entry in track] == ["0", "1", "2"]
    assert track[2]["seconds"] == {"n": "2", "d": "1"}


def test_scene_cut_classifier_accepts_a_persistent_hard_cut():
    colors = [(20, 20, 20)] * 7 + [(220, 220, 220)] * 8
    event = SB.classify_scene_cut(_samples(colors), candidate_frame=7)
    assert event is not None
    assert event["kind"] == "cut"
    assert event["anchor_frame"] == "7"
    assert event["previous_frame"] == "6"


def test_scene_cut_classifier_rejects_a_low_contrast_subtitle_step():
    colors = [(100, 100, 100)] * 7 + [(112, 112, 112)] * 8
    assert SB.classify_scene_cut(_samples(colors), candidate_frame=7) is None


def test_scene_cut_classifier_rejects_a_brightness_exposure_swing():
    colors = [(200, 200, 200)] * 7 + [(80, 80, 80)] * 2 + [(140, 140, 140)] * 6
    assert SB.classify_scene_cut(_samples(colors), candidate_frame=7) is None
