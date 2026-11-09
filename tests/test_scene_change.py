"""In-shot content change detector: structure/chroma/persistence gates."""
from __future__ import annotations

import numpy as np

from trace_annotate.extractors import scene_change as SC


def _checker(size=64, square=16, low=60, high=200):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            value = high if ((x // square) + (y // square)) % 2 == 0 else low
            frame[y, x, :] = value
    return frame


def _stripes(size=64, width=2, low=40, high=230):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            value = high if (x // width) % 2 == 0 else low
            frame[y, x, :] = value
    return frame


def _samples(frames):
    return [{"frame_index": str(index), "frame": frame,
             "seconds": {"n": str(index), "d": "1"}}
            for index, frame in enumerate(frames)]


def test_structural_change_persisting_is_a_content_change():
    frames = [_checker()] * 6 + [_stripes()] * 12
    result = SC.detect_content_changes(_samples(frames))
    assert result["changes"]
    change = result["changes"][0]
    assert change["kind"] == "content_change"
    assert change["start_frame"] == "6"
    assert change["sustained"] > SC.CONTENT_THRESHOLD


def test_uniform_brightness_step_is_lighting_not_content():
    base = _checker()
    brighter = np.clip(base.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    frames = [base] * 6 + [brighter] * 12
    result = SC.detect_content_changes(_samples(frames))
    assert result["changes"] == []
    kinds = [item["kind"] for item in result["filtered"]]
    assert "lighting_change" in kinds


def test_one_frame_overlay_is_transient_not_content():
    base = _checker()
    overlay = base.copy()
    overlay[8:40, 8:40, :] = 255
    frames = [base] * 6 + [overlay] + [base] * 12
    result = SC.detect_content_changes(_samples(frames))
    assert result["changes"] == []
    kinds = [item["kind"] for item in result["filtered"]]
    assert "transient_change" in kinds


def test_static_frames_report_nothing():
    frames = [_checker()] * 15
    result = SC.detect_content_changes(_samples(frames))
    assert result["changes"] == []
    assert result["filtered"] == []


def test_too_short_sequence_is_reported_as_empty():
    result = SC.detect_content_changes(_samples([_checker()] * 5))
    assert result["changes"] == []
    assert result["params"]["persistence"] == SC.PERSISTENCE


def test_eight_samples_reports_the_required_nine_explicitly():
    result = SC.detect_content_changes(_samples([_checker()] * 8))
    assert result["changes"] == []
    assert result["status"] == "insufficient_samples"
    assert result["provided_samples"] == 8
    assert result["required_samples"] == 9
    assert "9" in result["message"]


def test_nine_samples_still_detects_a_persistent_structural_change():
    frames = [_checker()] * 4 + [_stripes()] * 5
    result = SC.detect_content_changes(_samples(frames))
    assert result["status"] == "success"
    assert result["changes"]
    assert result["changes"][0]["kind"] == "content_change"


def test_changes_map_to_observable_claims():
    frames = [_checker()] * 6 + [_stripes()] * 12
    result = SC.detect_content_changes(_samples(frames))
    claims = SC.changes_to_claims(result["changes"], lambda frame: f"art-{frame}")
    assert claims[0]["claim_type"] == "observable"
    assert claims[0]["score_kind"] == "rule_score"
    assert claims[0]["evidence_ids"]
