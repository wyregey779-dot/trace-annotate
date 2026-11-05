"""Shot framing: geometry bands, face-only caution, multi-subject refusal."""
from __future__ import annotations

from trace_annotate.extractors import shot_attributes as SA

FRAME = (1920, 1080)


def _box(height_ratio, width_ratio=0.2, center=True):
    height = height_ratio * FRAME[1]
    width = width_ratio * FRAME[0]
    if center:
        x1 = (FRAME[0] - width) / 2
        y1 = (FRAME[1] - height) / 2
    else:
        x1, y1 = 0, 0
    return [x1, y1, x1 + width, y1 + height]


def test_small_subject_is_a_long_shot():
    result = SA.classify_framing([_box(0.08)], FRAME)
    assert result["framing"] == "远景"
    assert result["confidence"] >= 0.8


def test_half_frame_subject_is_a_close_shot():
    result = SA.classify_framing([_box(0.6)], FRAME)
    assert result["framing"] == "近景"
    assert result["uncertainty"] is None


def test_full_frame_subject_is_an_extreme_close_up():
    result = SA.classify_framing([_box(0.9)], FRAME)
    assert result["framing"] == "特写"


def test_face_only_measurement_carries_uncertainty():
    result = SA.classify_framing([_box(0.85)], FRAME, subject_kind="face")
    assert result["framing"] == "特写"
    assert result["uncertainty"]
    assert result["confidence"] < 0.6


def test_no_subject_is_not_applicable_not_a_guess():
    result = SA.classify_framing([], FRAME)
    assert result["framing"] == "不适用"
    assert result["subjects"] == 0
    assert result["uncertainty"]


def test_different_subjects_report_disagreement():
    result = SA.classify_framing([_box(0.06), _box(0.6)], FRAME)
    assert result["framing"] == "多主体不一致"


def test_clipped_subject_is_uncertain():
    result = SA.classify_framing([_box(0.7, center=False)], FRAME)
    assert result["framing"] == "近景"
    assert result["confidence"] <= 0.5
    assert "裁切" in result["uncertainty"]


def test_window_summary_counts_dominant_and_reports_changes():
    frames = [{"frame_index": str(index), "frame_size": FRAME,
               "boxes": [_box(0.6 if index < 3 else 0.25)]}
              for index in range(6)]
    result = SA.summarise_framing(frames)
    assert result["status"] == "success"
    assert result["framing"] == "近景"
    assert result["changes"][0]["to"] == "全景"
    assert result["changes"][0]["at_frame"] == "3"


def test_empty_window_reports_not_run():
    result = SA.summarise_framing([])
    assert result["status"] == "not_run"


def test_framing_claims_are_measured_not_interpretation():
    result = SA.summarise_framing([{"frame_index": "4", "frame_size": FRAME,
                                    "boxes": [_box(0.6)]}])
    claims = SA.framing_to_claims(result, lambda name: "art-4")
    assert claims[0]["claim_type"] == "measured"
    assert claims[0]["measurement"]["unit"] == "framing_class"
    assert claims[0]["score_kind"] == "rule_score"
