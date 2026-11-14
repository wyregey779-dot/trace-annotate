"""Camera motion measurement on synthetic shifts and zoom."""
from __future__ import annotations

import numpy as np
import pytest

from trace_annotate.extractors import camera_motion as CM

cv2 = pytest.importorskip("cv2")


def _scene(size=(120, 160)):
    """Deterministic textured frame with unique trackable corners."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(noise, (5, 5), 0)
    cv2.circle(frame, (40, 40), 12, (255, 255, 255), -1)
    cv2.circle(frame, (110, 80), 18, (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 90), (70, 110), (250, 250, 250), 2)
    return frame


def _samples(frames):
    return [{"frame_index": str(index), "frame": frame,
             "seconds": {"n": str(index), "d": "1"}}
            for index, frame in enumerate(frames)]


def test_static_frames_report_static():
    frame = _scene()
    result = CM.analyse_motion(_samples([frame] * 4))
    assert result["status"] == "success"
    assert result["summary"]["class"] == "static"


def test_rightward_image_shift_is_pan_left():
    frame = _scene()
    shifted = np.roll(frame, 10, axis=1)
    result = CM.analyse_motion(_samples([frame, shifted, shifted]))
    assert result["status"] == "success"
    assert result["summary"]["class"] == "pan_left"
    assert result["pairs"][0]["metrics"]["tx"] == pytest.approx(10, abs=1.5)


def test_zoom_in_is_measured_as_scale_change():
    frame = _scene()
    scaled = cv2.resize(frame, None, fx=1.08, fy=1.08, interpolation=cv2.INTER_LINEAR)
    center = (scaled.shape[1] // 2, scaled.shape[0] // 2)
    cropped = scaled[center[1] - 60:center[1] + 60, center[0] - 80:center[0] + 80]
    result = CM.analyse_motion(_samples([frame, cropped]))
    assert result["status"] == "success"
    pair = result["pairs"][0]
    assert pair["class"] == "zoom_in"
    assert pair["uncertainty"]


def test_flat_frames_report_not_run_instead_of_a_guess():
    flat = np.full((80, 80, 3), 128, dtype=np.uint8)
    result = CM.analyse_motion(_samples([flat, flat, flat]))
    assert result["status"] == "not_run"
    assert result["summary"] is None


def test_speed_uses_real_seconds_not_average_fps():
    frame = _scene()
    shifted = np.roll(frame, 10, axis=1)
    samples = [{"frame_index": "0", "frame": frame, "seconds": {"n": "0", "d": "1"}},
               {"frame_index": "1", "frame": shifted, "seconds": {"n": "2", "d": "1"}}]
    result = CM.analyse_motion(samples)
    assert result["pairs"][0]["seconds"] == pytest.approx(2.0)
    assert result["pairs"][0]["speed_px_s"] == pytest.approx(5.0, abs=1.0)


def test_low_global_fit_is_unknown_not_a_handheld_claim():
    classified = CM.classify_pair({
        "status": "success", "tx": 8.0, "ty": 1.0, "scale": 1.0,
        "rotation": 0.0, "inlier_ratio": 0.25, "residual": 2.0,
    }, 1.0)
    assert classified["class"] == "unknown_motion"
    assert "不能据此判断手持" in classified["reason"]
    assert classified["uncertainty"]


def test_competing_static_tracks_refuse_subject_motion_as_camera_motion():
    classified = CM.classify_pair({
        "status": "success", "tx": 10.0, "ty": 0.0, "scale": 1.0,
        "rotation": 0.0, "inlier_ratio": 0.9, "residual": 0.3,
        "static_track_ratio": 0.4, "static_track_cells": 5,
    }, 1.0)
    assert classified["class"] == "unknown_motion"
    assert classified["uncertainty"]


def test_fixed_camera_with_moving_subject_stays_static():
    rng = np.random.default_rng(42)
    background = cv2.GaussianBlur(
        rng.integers(0, 256, (120, 160, 3), dtype=np.uint8), (5, 5), 0)
    subject = cv2.GaussianBlur(
        rng.integers(0, 256, (60, 50, 3), dtype=np.uint8), (3, 3), 0)
    left, right = background.copy(), background.copy()
    left[30:90, 25:75] = subject
    right[30:90, 65:115] = subject

    result = CM.analyse_motion(_samples([left, right]))

    assert result["status"] == "success"
    assert result["summary"]["class"] == "static"


def test_unknown_summary_preserves_the_pair_refusal_reason(monkeypatch):
    monkeypatch.setattr(CM, "measure_pair", lambda _left, _right: {
        "status": "success", "tx": 8.0, "ty": 1.0, "scale": 1.0,
        "rotation": 0.0, "inlier_ratio": 0.25, "residual": 2.0,
    })
    frame = _scene()
    result = CM.analyse_motion(_samples([frame, frame]))
    assert result["summary"]["class"] == "unknown_motion"
    assert result["summary"]["uncertainty"]
    claims = CM.motion_to_claims(result, lambda name: f"art-{name}", "0", "1")
    assert claims[0]["uncertainty_reason"]


def test_motion_maps_to_a_measured_claim():
    frame = _scene()
    shifted = np.roll(frame, 10, axis=1)
    result = CM.analyse_motion(_samples([frame, shifted]))
    claims = CM.motion_to_claims(result, lambda name: f"art-{name}", "0", "1")
    assert claims[0]["claim_type"] == "measured"
    assert claims[0]["measurement"]["unit"] == "class"
    assert claims[0]["score_kind"] == "rule_score"
