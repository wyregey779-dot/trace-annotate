"""Bounded acoustic decoding and honest energy-activity semantics."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.errors import WorkspaceError


def _window(start: float, end: float):
    return {"start": S.rational(start), "end": S.rational(end)}


def test_measure_audio_decodes_only_the_requested_window_without_pcm_pipe_capture(
        tmp_path, monkeypatch):
    commands = []

    def child(command, output, diagnostic, *, cancelled, timeout):
        commands.append(list(command))
        diagnostic.write_text("", encoding="utf-8")
        if output is not None:
            seconds = 1.5
            samples = np.sin(2 * np.pi * 220 * np.arange(int(seconds * V.AUDIO_SAMPLE_RATE))
                             / V.AUDIO_SAMPLE_RATE).astype("<f4")
            output.write_bytes(samples.tobytes())
        else:
            Path(command[-1]).write_bytes(b"encoded-audio")
        return 0, ""

    monkeypatch.setattr(V, "_audio_child", child)
    result = V._measure_audio(tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
                              window=_window(12.5, 14.0), media_duration=S.rational(30),
                              cancelled=lambda: False)
    assert result["status"] == "success"
    assert result["window_seconds"] == {"start": 12.5, "end": 14.0}
    assert result["measurement"]["window_seconds"]["start"] == 12.5
    assert result["measurement"]["activity_kind"] == "energy_above_adaptive_threshold"
    decode = commands[0]
    assert decode[decode.index("-ss") + 1] == "12.500000"
    assert decode[decode.index("-t") + 1] == "1.500000"
    assert decode[-1] == "pipe:1"
    assert not (tmp_path / "job" / "audio" / "acoustic.f32le").exists()
    assert not (tmp_path / "job" / "audio" / "decode.stderr").exists()
    assert (tmp_path / "job" / "audio" / "acoustic.m4a").is_file()


@pytest.mark.parametrize(("detail", "expected"), [
    ("Stream map '0:a:0' matches no streams.", ("not_run", "no_decodable_audio_track")),
    ("Invalid data found when processing input", ("failed", "audio_decode_failed")),
])
def test_measure_audio_separates_absent_track_from_decode_failure(
        tmp_path, monkeypatch, detail, expected):
    def child(command, output, diagnostic, *, cancelled, timeout):
        diagnostic.write_text(detail, encoding="utf-8")
        if output is not None:
            output.write_bytes(b"")
        return 1, detail

    monkeypatch.setattr(V, "_audio_child", child)
    result = V._measure_audio(tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
                              window=_window(1, 2), media_duration=S.rational(3))
    assert (result["status"], result["reason"]) == expected
    assert not (tmp_path / "job" / "audio" / "acoustic.f32le").exists()


def test_measure_audio_propagates_typed_cancellation_and_removes_pcm(tmp_path, monkeypatch):
    def child(command, output, diagnostic, *, cancelled, timeout):
        diagnostic.write_text("cancelled", encoding="utf-8")
        if output is not None:
            output.write_bytes(np.zeros(16, dtype="<f4").tobytes())
        raise WorkspaceError("CANCELLED", "test cancellation")

    monkeypatch.setattr(V, "_audio_child", child)
    with pytest.raises(WorkspaceError, match="test cancellation") as failure:
        V._measure_audio(tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
                         window=_window(1, 2), media_duration=S.rational(3))
    assert failure.value.code == "CANCELLED"
    assert not (tmp_path / "job" / "audio" / "acoustic.f32le").exists()


def test_long_request_uses_bounded_prefix_and_declares_actual_coverage(tmp_path, monkeypatch):
    commands = []

    def child(command, output, diagnostic, *, cancelled, timeout):
        commands.append(list(command))
        diagnostic.write_text("", encoding="utf-8")
        if output is not None:
            count = int(V.AUDIO_SAMPLE_RATE)
            output.write_bytes(np.zeros(count, dtype="<f4").tobytes())
        else:
            Path(command[-1]).write_bytes(b"bounded-evidence")
        return 0, ""

    monkeypatch.setattr(V, "AUDIO_MAX_ANALYSIS_SECONDS", 1.0)
    monkeypatch.setattr(V, "_audio_child", child)
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(10, 1010), media_duration=S.rational(1200))
    assert result["status"] == "success"
    assert commands[0][commands[0].index("-t") + 1] == "1.000000"
    assert result["requested_window_seconds"] == {"start": 10.0, "end": 1010.0}
    assert result["window_seconds"] == {"start": 10.0, "end": 11.0}
    assert result["coverage"] == {
        "kind": "bounded_prefix", "requested_seconds": 1000.0,
        "analysed_seconds": 1.0, "ratio": 0.001,
        "truncated": True, "max_analysis_seconds": 1.0,
    }
    assert result["measurement"]["coverage"] == result["coverage"]


def test_audio_disk_check_preserves_workspace_reserve(tmp_path, monkeypatch):
    called = False

    def child(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("decode must not start below the reserve")

    duration = 1.0
    pcm = int(duration * V.AUDIO_SAMPLE_RATE) * 4
    free = pcm + V.AUDIO_WORKSPACE_RESERVE_BYTES - 1
    monkeypatch.setattr(V, "_audio_child", child)
    monkeypatch.setattr(V.shutil, "disk_usage",
                        lambda path: shutil._ntuple_diskusage(10**12, 10**12 - free, free))
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(0, duration), media_duration=S.rational(duration))
    assert result["status"] == "failed"
    assert result["reason"] == "insufficient_disk_for_audio_window"
    assert result["pcm_bytes"] == pcm
    assert result["reserve_bytes"] == V.AUDIO_WORKSPACE_RESERVE_BYTES
    assert result["required_bytes"] == pcm + V.AUDIO_WORKSPACE_RESERVE_BYTES
    assert not called


def test_audio_disk_status_failure_is_explicit_and_decode_never_starts(tmp_path, monkeypatch):
    called = False

    def child(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("decode must not start without a measured reserve")

    def unavailable(path):
        raise OSError("disk status unavailable")

    monkeypatch.setattr(V, "_audio_child", child)
    monkeypatch.setattr(V.shutil, "disk_usage", unavailable)
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(0, 1), media_duration=S.rational(1))
    assert result == {"status": "failed", "reason": "audio_storage_status_unavailable",
                      "error": "OSError", "measurement": None}
    assert not called


def test_audio_decode_rejects_pcm_beyond_the_frozen_byte_budget(tmp_path, monkeypatch):
    def child(command, output, diagnostic, *, cancelled, timeout):
        diagnostic.write_text("", encoding="utf-8")
        assert output is not None
        count = V.AUDIO_SAMPLE_RATE + V.AUDIO_PCM_TOLERANCE_SAMPLES + 1
        output.write_bytes(np.zeros(count, dtype="<f4").tobytes())
        return 0, ""

    monkeypatch.setattr(V, "AUDIO_MAX_ANALYSIS_SECONDS", 1.0)
    monkeypatch.setattr(V, "_audio_child", child)
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(0, 10), media_duration=S.rational(10))
    assert result["status"] == "failed"
    assert result["reason"] == "audio_decode_exceeded_budget"
    assert result["expected_pcm_bytes"] == V.AUDIO_SAMPLE_RATE * 4
    assert result["actual_pcm_bytes"] > result["expected_pcm_bytes"]
    assert not (tmp_path / "job" / "audio" / "acoustic.f32le").exists()
    assert not (tmp_path / "job" / "audio" / "acoustic.m4a").exists()


def test_audio_rechecks_reserve_after_decode_before_encoding(tmp_path, monkeypatch):
    encoded = False

    def child(command, output, diagnostic, *, cancelled, timeout):
        nonlocal encoded
        diagnostic.write_text("", encoding="utf-8")
        if output is None:
            encoded = True
            raise AssertionError("evidence encoding must not start below the reserve")
        output.write_bytes(np.zeros(V.AUDIO_SAMPLE_RATE, dtype="<f4").tobytes())
        return 0, ""

    calls = 0

    def capacity(path):
        nonlocal calls
        calls += 1
        free = (V.AUDIO_WORKSPACE_RESERVE_BYTES + V.AUDIO_SAMPLE_RATE * 4
                if calls == 1 else V.AUDIO_WORKSPACE_RESERVE_BYTES - 1)
        return shutil._ntuple_diskusage(10**12, 10**12 - free, free)

    monkeypatch.setattr(V, "_audio_child", child)
    monkeypatch.setattr(V.shutil, "disk_usage", capacity)
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(0, 1), media_duration=S.rational(1))
    assert result["status"] == "failed"
    assert result["reason"] == "insufficient_disk_for_audio_evidence"
    assert not encoded
    assert not (tmp_path / "job" / "audio" / "acoustic.m4a").exists()


def test_failed_audio_encoder_partial_output_is_not_evidence(tmp_path, monkeypatch):
    def child(command, output, diagnostic, *, cancelled, timeout):
        diagnostic.write_text("encoder failed" if output is None else "", encoding="utf-8")
        if output is not None:
            output.write_bytes(np.zeros(V.AUDIO_SAMPLE_RATE, dtype="<f4").tobytes())
            return 0, ""
        Path(command[-1]).write_bytes(b"partial-unusable-audio")
        return 1, "encoder failed"

    monkeypatch.setattr(V, "_audio_child", child)
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(0, 1), media_duration=S.rational(1))
    assert result["status"] == "failed"
    assert result["reason"] == "acoustic_evidence_encode_failed"
    assert not (tmp_path / "job" / "audio" / "acoustic.m4a").exists()


def test_measure_audio_cancels_inside_cpu_analysis_and_removes_pcm(tmp_path, monkeypatch):
    checks = 0

    def child(command, output, diagnostic, *, cancelled, timeout):
        diagnostic.write_text("", encoding="utf-8")
        if output is not None:
            output.write_bytes(np.ones(V.AUDIO_SAMPLE_RATE * 3, dtype="<f4").tobytes())
        else:
            raise AssertionError("cancelled analysis must not encode evidence")
        return 0, ""

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 8

    monkeypatch.setattr(V, "_audio_child", child)
    with pytest.raises(WorkspaceError) as failure:
        V._measure_audio(
            tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
            window=_window(0, 3), media_duration=S.rational(3), cancelled=cancelled)
    assert failure.value.code == "CANCELLED"
    assert not (tmp_path / "job" / "audio" / "acoustic.f32le").exists()


def test_measure_audio_cpu_deadline_fails_without_publishing_evidence(tmp_path, monkeypatch):
    def child(command, output, diagnostic, *, cancelled, timeout):
        diagnostic.write_text("", encoding="utf-8")
        if output is not None:
            output.write_bytes(np.ones(V.AUDIO_SAMPLE_RATE, dtype="<f4").tobytes())
        else:
            raise AssertionError("timed-out analysis must not encode evidence")
        return 0, ""

    monkeypatch.setattr(V, "AUDIO_ANALYSIS_TIMEOUT_SECONDS", -1.0)
    monkeypatch.setattr(V, "_audio_child", child)
    result = V._measure_audio(
        tmp_path / "input.mp4", "ffmpeg", tmp_path / "job",
        window=_window(0, 1), media_duration=S.rational(1))
    assert result == {"status": "failed", "reason": "audio_analysis_timeout",
                      "measurement": None}
    assert not (tmp_path / "job" / "audio" / "acoustic.f32le").exists()
    assert not (tmp_path / "job" / "audio" / "acoustic.m4a").exists()


def test_projected_acoustic_claim_calls_energy_activity_energy_not_speech():
    measurement = {
        "status": "success", "window_seconds": {"start": 1.0, "end": 2.0},
        "rms_dbfs": -18.0, "f0_hz": 220.0, "f0_variation_hz": 1.0,
        "activity_kind": "energy_above_adaptive_threshold",
        "energy_active_ratio": 0.5,
        "energy_active_intervals": [{"start": 1.2, "end": 1.7}],
    }
    candidate = {"job_id": "job-acoustic", "processor_version": "1.1",
                 "actions_sampling": {"candidates": []},
                 "acoustics_sampling": {"status": "success", "measurement": measurement,
                                         "stream_selection": "0:a:0", "sample_rate": 16000}}
    body = V._candidate_claims(candidate, 0, {"start": 0.0, "end": 3.0},
                               S.rational(0), S.rational(3),
                               {"frame_index": "0", "source_pts": "0", "roles": ["first"]})
    claim = next(item for item in body["claims"]
                 if item["method"]["id"] == "acoustic-window-v2")
    assert "声能活动" in claim["statement"]
    assert "语音活动" not in claim["statement"]
    assert claim["method"]["parameters"]["activity_kind"] == (
        "energy_above_adaptive_threshold")
    assert "不代表语音" in claim["uncertainty_reason"]


def test_projected_acoustic_claim_discloses_bounded_prefix_coverage():
    coverage = {"kind": "bounded_prefix", "requested_seconds": 600.0,
                "analysed_seconds": 120.0, "ratio": 0.2,
                "truncated": True, "max_analysis_seconds": 120.0}
    measurement = {
        "status": "success", "window_seconds": {"start": 0.0, "end": 120.0},
        "rms_dbfs": -18.0, "f0_hz": None, "f0_variation_hz": None,
        "activity_kind": "energy_above_adaptive_threshold",
        "energy_active_ratio": 0.5, "energy_active_intervals": [],
        "active_ratio": 0.5, "active_intervals": [], "coverage": coverage,
    }
    candidate = {"job_id": "job-acoustic-prefix", "processor_version": "1.1",
                 "actions_sampling": {"candidates": []},
                 "acoustics_sampling": {"status": "success", "measurement": measurement,
                                          "coverage": coverage,
                                          "stream_selection": "0:a:0", "sample_rate": 16000}}
    body = V._candidate_claims(candidate, 0, {"start": 0.0, "end": 600.0},
                               S.rational(0), S.rational(600),
                               {"frame_index": "0", "source_pts": "0", "roles": ["first"]})
    claim = next(item for item in body["claims"]
                 if item["method"]["id"] == "acoustic-window-v2")
    assert claim["method"]["parameters"]["coverage"] == coverage
    assert "仅测量并绑定已声明的前缀窗口" in claim["uncertainty_reason"]


def test_real_ffmpeg_audio_window_preserves_origin_and_duration(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg unavailable")
    source = tmp_path / "source.m4a"
    generated = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=2", "-c:a", "aac", str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert generated.returncode == 0, generated.stderr.decode("utf-8", "replace")
    result = V._measure_audio(source, ffmpeg, tmp_path / "real-job",
                              window=_window(1.0, 1.5), media_duration=S.rational(2))
    assert result["status"] == "success", result
    assert result["measurement"]["window_seconds"]["start"] == 1.0
    assert result["measurement"]["window_seconds"]["end"] == pytest.approx(1.5, abs=0.02)
    assert result["samples"] == pytest.approx(V.AUDIO_SAMPLE_RATE // 2, abs=1024)
    assert (tmp_path / "real-job" / "audio" / "acoustic.m4a").stat().st_size > 0


def test_owned_audio_child_observes_cancellation_and_reaps_process(tmp_path):
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    started = time.monotonic()
    with pytest.raises(WorkspaceError) as failure:
        V._audio_child([sys.executable, "-c", "import time; time.sleep(30)"], None,
                       tmp_path / "cancel.stderr", cancelled=cancelled, timeout=10)
    assert failure.value.code == "CANCELLED"
    assert time.monotonic() - started < 3
