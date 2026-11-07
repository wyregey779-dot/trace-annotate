"""Expressive cues: acoustic measurements and observable face landmarks."""
from __future__ import annotations

import numpy as np
import pytest

from trace_annotate.extractors import expressive_cues as EC


def _sine(frequency, seconds, sample_rate=22050, amplitude=0.5):
    times = np.arange(int(seconds * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * times)


def test_rms_matches_the_known_sine_level():
    measurement = EC.measure_acoustic_window(_sine(440, 1.0), 22050)
    assert measurement["status"] == "success"
    assert measurement["rms_dbfs"] == pytest.approx(-9.03, abs=0.1)


def test_f0_matches_the_known_pure_tone_within_one_percent():
    measurement = EC.measure_acoustic_window(_sine(440, 1.0), 22050)
    assert measurement["f0_hz"] == pytest.approx(440, rel=0.01)


def test_silence_has_no_f0_and_no_active_speech():
    measurement = EC.measure_acoustic_window(np.zeros(22050), 22050)
    assert measurement["f0_hz"] is None
    assert measurement["active_ratio"] == 0.0


def test_speech_interval_is_detected_between_silences():
    sample_rate = 22050
    quiet = np.zeros(int(0.4 * sample_rate))
    tone = _sine(300, 0.5, sample_rate, amplitude=0.4)
    audio = np.concatenate([quiet, tone, quiet])
    measurement = EC.measure_acoustic_window(audio, sample_rate)
    assert len(measurement["active_intervals"]) == 1
    interval = measurement["active_intervals"][0]
    assert interval["start"] >= 0.35
    assert interval["end"] <= 1.0


def test_continuous_loud_window_is_active_despite_flat_floor():
    measurement = EC.measure_acoustic_window(_sine(440, 1.0), 22050)
    assert measurement["active_ratio"] > 0.9
    assert len(measurement["active_intervals"]) == 1


def test_window_reports_origin_and_units():
    measurement = EC.measure_acoustic_window(_sine(300, 0.5), 22050, origin_seconds=12.5)
    assert measurement["window_seconds"]["start"] == 12.5
    assert measurement["sample_rate"] == 22050
    assert "db" in "dbfs"


def test_face_without_landmarks_is_not_run():
    face = {"box": [100, 100, 180, 180], "frame_size": [1920, 1080]}
    result = EC.face_cue_candidates(face, frame_index="3", evidence_id="art-3")
    assert result["status"] == "not_run"


def test_tiny_face_is_not_applicable():
    face = {"box": [10, 10, 40, 40], "frame_size": [1920, 1080], "landmarks": {}}
    result = EC.face_cue_candidates(face, frame_index="3", evidence_id="art-3")
    assert result["status"] == "not_applicable"


def test_landmarks_need_frame_evidence():
    face = {"box": [100, 100, 200, 220], "frame_size": [1920, 1080],
            "landmarks": {"mouth": [[120, 180], [180, 180], [150, 190], [150, 210]]}}
    result = EC.face_cue_candidates(face, frame_index="3", evidence_id=None)
    assert result["status"] == "not_run"


def test_mouth_and_brow_cues_are_observable_with_evidence():
    face = {"box": [100, 100, 200, 220], "frame_size": [1920, 1080],
            "landmarks": {"mouth": [[120, 180], [180, 180], [150, 190], [150, 215]],
                          "mouth_corner_lift": 0.05,
                          "brows": [[140, 130], [160, 130]], "eye_gap": 60}}
    result = EC.face_cue_candidates(face, frame_index="3", evidence_id="art-3")
    kinds = {cue["kind"] for cue in result["cues"]}
    assert kinds == {"mouth_open", "smile", "brow_furrow"}
    claims = EC.cues_to_claims(result["cues"])
    assert all(claim["claim_type"] == "observable" for claim in claims)
    assert all(claim["evidence_ids"] == ["art-3"] for claim in claims)


def test_acoustic_claim_is_measured_and_refuses_interpretation():
    measurement = EC.measure_acoustic_window(_sine(440, 0.5), 22050)
    claims = EC.acoustic_claim(measurement, "art-audio")
    assert claims[0]["claim_type"] == "measured"
    assert claims[0]["measurement"]["f0_hz"] == pytest.approx(440, rel=0.01)
    assert "情绪" in claims[0]["uncertainty_reason"]
