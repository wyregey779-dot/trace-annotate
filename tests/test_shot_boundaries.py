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
    assert interval["start_frame"] == "2"
    assert interval["end_frame"] == "6"
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


def test_dissolve_fit_rejects_a_continuous_textured_pan():
    y, x = np.indices((48, 64))
    frames = [{"frame_index": str(i),
               "frame": (np.where(x < 20 + i * 3, 225, 25) + (y % 3) * 5).astype(np.uint8)}
              for i in range(8)]
    track = SB.build_track(frames)
    assert SB._dissolve_fit(track, 0, len(track) - 1) is None


def test_dissolve_fit_requires_observed_internal_mixtures():
    first = np.full((48, 64), 35, dtype=np.uint8)
    last = np.full((48, 64), 180, dtype=np.uint8)
    frames = [first, first, last, last]
    track = SB.build_track([{"frame_index": str(i), "frame": frame} for i, frame in enumerate(frames)])
    assert SB._dissolve_fit(track, 0, 3) is None


def test_dissolve_fit_accepts_a_textured_nonlinear_crossfade():
    y, x = np.indices((48, 64))
    first = (((x // 4 + y // 6) % 2) * 180 + 35).astype(float)
    last = (((x // 9 + y // 3) % 2) * 150 + 70).astype(float)
    alphas = [0.0, .1, .24, .43, .62, .79, .93, 1.0]
    frames = [np.round((1 - alpha) * first + alpha * last).astype(np.uint8) for alpha in alphas]
    track = SB.build_track([{"frame_index": str(i), "frame": frame} for i, frame in enumerate(frames)])
    fit = SB._dissolve_fit(track, 0, len(track) - 1)
    assert fit is not None
    assert fit["interior_mixture_frames"] >= 3
    assert fit["relative_residual"] < .02


@pytest.mark.parametrize("color", [BLACK, WHITE])
def test_static_polarity_hold_is_not_a_fade(color):
    assert SB.detect_shot_boundaries(_samples([color] * 12))["events"] == []


@pytest.mark.parametrize("colors,kind", [
    ([(240, 240, 240)] * 3 + [(180, 180, 180), (120, 120, 120),
                              (60, 60, 60)] + [BLACK] * 3, "fade_out"),
    ([(40, 40, 40)] * 3 + [(100, 100, 100), (160, 160, 160),
                           (220, 220, 220)] + [WHITE] * 3, "fade_out"),
    ([WHITE] * 3 + [(220, 220, 220), (160, 160, 160),
                    (100, 100, 100), (40, 40, 40)] + [(40, 40, 40)] * 2,
     "fade_in"),
])
def test_fade_owns_the_complete_ramp_without_static_holds(colors, kind):
    result = SB.detect_shot_boundaries(_samples(colors))
    assert _kinds(result) == [kind]
    event = result["events"][0]
    assert event["interval"]["start_frame"] == "2"
    assert event["interval"]["end_frame"] == "6"
    assert event["previous_frame"] == "1"
    assert event["representative_frame"] == event["anchor_frame"] == "6"


@pytest.mark.parametrize("polarity,normal,ramp", [
    (BLACK, (240, 240, 240), [(180, 180, 180), (120, 120, 120), (60, 60, 60)]),
    (WHITE, (40, 40, 40), [(100, 100, 100), (160, 160, 160), (220, 220, 220)]),
])
def test_polarity_bridge_preserves_incoming_and_outgoing_fades(polarity, normal, ramp):
    colors = [normal] * 3 + ramp + [polarity] * 3 + list(reversed(ramp)) + [normal] * 3
    result = SB.detect_shot_boundaries(_samples(colors))
    assert _kinds(result) == ["fade_out", "fade_in"]
    assert [(event["interval"]["start_frame"], event["interval"]["end_frame"])
            for event in result["events"]] == [("2", "6"), ("8", "12")]


def test_single_black_endpoint_can_be_shared_by_two_complete_fades():
    colors = ([(240, 240, 240)] * 3 + [(180, 180, 180), (120, 120, 120),
                                      (60, 60, 60), BLACK, (60, 60, 60),
                                      (120, 120, 120), (180, 180, 180)]
              + [(240, 240, 240)] * 3)
    result = SB.detect_shot_boundaries(_samples(colors))
    assert _kinds(result) == ["fade_out", "fade_in"]
    assert [(event["interval"]["start_frame"], event["interval"]["end_frame"])
            for event in result["events"]] == [("2", "6"), ("6", "10")]


def test_black_white_hard_cut_stays_a_hard_cut():
    result = SB.detect_shot_boundaries(_samples([BLACK] * 5 + [WHITE] * 5))
    assert _kinds(result) == ["cut"]
    assert result["events"][0]["anchor_frame"] == "5"


def test_missing_frames_cannot_establish_a_single_frame_flash():
    samples = _samples([GRAY, WHITE, GRAY])
    for sample, frame_index in zip(samples, (20, 24, 29)):
        sample["frame_index"] = str(frame_index)
    result = SB.detect_shot_boundaries(samples)
    assert result["events"] == []
    assert all(entry["step"] is None for entry in result["track"])


def test_real_flash_does_not_swallow_a_following_cut():
    result = SB.detect_shot_boundaries(_samples([GRAY] * 5 + [WHITE] + [GRAY] * 2 + [RED] * 5))
    assert _kinds(result) == ["flash", "cut"]
    assert result["events"][0]["split_point"] is None
    assert result["events"][1]["anchor_frame"] == "8"


def test_vfr_fade_preserves_exact_frame_identity_and_event_times():
    samples = _samples([WHITE] * 3 + [(220, 220, 220), (160, 160, 160),
                                     (100, 100, 100), (40, 40, 40)] + [(40, 40, 40)] * 2)
    for index, sample in enumerate(samples):
        sample["frame_index"] = str(70 + index)
        sample["source_pts"] = str(index * index + 1000)
        sample["proxy_pts"] = str(index * index + 100)
        sample["seconds"] = {"n": str(index * index + 100), "d": "1000"}
    result = SB.detect_shot_boundaries(samples)
    event = result["events"][0]
    assert event["interval"] == {"start_frame": "72", "end_frame": "76",
                                 "start_seconds": samples[2]["seconds"],
                                 "end_seconds": samples[6]["seconds"]}
    assert event["anchor_frame"] == "76"
    assert event["previous_frame"] == "71"
    assert [entry["frame_index"] for entry in result["track"]] == [sample["frame_index"] for sample in samples]
    assert [entry["source_pts"] for entry in result["track"]] == [sample["source_pts"] for sample in samples]
    assert [entry["proxy_pts"] for entry in result["track"]] == [sample["proxy_pts"] for sample in samples]
