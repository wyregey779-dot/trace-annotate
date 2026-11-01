"""Real-source objective analysis, strict scope and owned-child cancellation."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.workspace import analysis_service as A
from trace_annotate.workspace import professional_schema as PS
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def media_factory(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Objective analysis requires FFmpeg and ffprobe")
    root = tmp_path_factory.mktemp("analysis-fixtures")
    cache = {}
    def make(kind):
        if kind in cache:
            return cache[kind]
        folder = root / kind
        folder.mkdir()
        source = folder / ("source.mkv" if kind == "unknown-layout" else "source.mov")
        command = ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=blue:size=32x32:rate=10:duration=2"]
        audio = {
            "mono": "sine=frequency=997:sample_rate=48000:duration=2,volume=4",
            "stereo": "aevalsrc=0.5*sin(2*PI*997*t)|0.25*sin(2*PI*997*t):s=48000:d=2:c=stereo",
            "silence": "anullsrc=r=48000:cl=stereo:d=1",
            "short": "sine=frequency=997:sample_rate=48000:duration=0.1,volume=4",
            "quiet": "sine=frequency=997:sample_rate=48000:duration=2,volume=0.008",
            "below-gate": "sine=frequency=997:sample_rate=48000:duration=2,volume=0.00008",
            "offset": "sine=frequency=997:sample_rate=48000:duration=1,volume=4",
            "six": "anullsrc=r=48000:cl=5.1:d=1",
            "unknown-layout": "sine=frequency=997:sample_rate=48000:duration=1,volume=4",
        }.get(kind)
        if audio:
            if kind == "offset":
                command += ["-itsoffset", "0.25"]
            command += ["-f", "lavfi", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "pcm_f32le" if kind == "unknown-layout" else "pcm_s24le"]
        else:
            command += ["-map", "0:v"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if kind == "vfr":
            command += ["-vf", "select='not(eq(n,2))'", "-fps_mode", "vfr"]
        command += [str(source)]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        store = WorkspaceStore(folder / "admitted-workspace")
        store.create()
        try:
            task = store.import_source(source)
            media = store.get_media(task["media_id"])
        finally:
            store.close()
        cache[kind] = (source, media)
        return cache[kind]
    return make


def values(report):
    return {v["name"]: v for v in report["values"]}


def loudness(result):
    return next(r for r in result["reports"] if r["kind"] == "loudness")


def test_known_pcm_mono_is_measured_and_all_reports_are_contract_valid(media_factory, tmp_path):
    source, media = media_factory("mono")
    result = A.analyze_media(source, media, tmp_path / "analysis")
    report = loudness(result)
    measured = values(report)
    assert abs(measured["integrated_lufs"]["value"] - (-9.0306)) <= .15
    assert abs(measured["sample_peak_dbfs"]["value"] - 20 * math.log10(.5)) <= .06
    assert measured["decoded_samples_per_channel"]["value"] == "96000"
    assert measured["channels"]["value"] == 1
    assert json.loads(measured["silence_intervals_json"]["value"]) == []
    frame = next(r for r in result["reports"] if r["kind"] == "frame_timing")
    assert values(frame)["decoded_frame_count"]["value"] == 20
    assert values(frame)["observed_timing"]["value"] == "CFR"
    assert values(frame)["observed_fps"]["value"] == 10
    for item in result["reports"]:
        PS.validate_analysis(item, media)
    assert result["semantic_listening"] == "not_performed"
    assert result["source_unchanged"]
    assert (tmp_path / "analysis/commands.jsonl").is_file()
    assert not P.live_media_processes()


def test_stereo_channels_and_interval_are_explicit(media_factory, tmp_path):
    source, media = media_factory("stereo")
    window = {"start": S.rational("0.5"), "end": S.rational("1.5")}
    result = A.analyze_media(source, media, tmp_path / "interval", window=window, subject_annotation_id="ann-audio")
    report = loudness(result)
    measured = values(report)
    assert report["scope"] == "interval" and report["window"] == window
    assert report["subject_annotation_id"] == "ann-audio"
    assert measured["channels"]["value"] == 2
    assert measured["decoded_samples_per_channel"]["value"] == "48000"
    assert abs(measured["integrated_lufs"]["value"] - (-3.01 + 10 * math.log10(.5 ** 2 + .25 ** 2))) <= .15
    assert measured["effective_audio_start"]["value"] == .5
    assert measured["effective_audio_end"]["value"] == 1.5


@pytest.mark.parametrize("kind,reason", [("silence", "digital_silence"), ("short", "insufficient_400ms_gating_block"),
                                         ("below-gate", "at_or_below_absolute_gate_summary_resolution")])
def test_silence_short_and_below_gate_do_not_export_false_lufs(media_factory, tmp_path, kind, reason):
    source, media = media_factory(kind)
    measured = values(loudness(A.analyze_media(source, media, tmp_path / kind)))
    assert measured["integrated_lufs"]["value"] is None
    assert measured["integrated_lufs"]["reason"] == reason
    if kind == "silence":
        assert measured["sample_peak_dbfs"]["value"] is None
        assert measured["true_peak_estimate_dbtp"]["value"] is None
        intervals = json.loads(measured["silence_intervals_json"]["value"])
        assert intervals == [{"start": S.rational(0), "end": S.rational(1)}]


def test_quiet_but_gated_signal_is_not_mistaken_for_ebur_sentinel(media_factory, tmp_path):
    source, media = media_factory("quiet")
    measured = values(loudness(A.analyze_media(source, media, tmp_path / "quiet")))
    assert measured["integrated_lufs"]["status"] == "available"
    assert -65 < measured["integrated_lufs"]["value"] < -60


@pytest.mark.parametrize("kind,reason", [("six", "unsupported_channel_count"), ("unknown-layout", "unsupported_or_unknown_channel_layout")])
def test_unsupported_audio_mapping_is_null_not_guessed(media_factory, tmp_path, kind, reason):
    source, media = media_factory(kind)
    report = loudness(A.analyze_media(source, media, tmp_path / kind))
    assert report["status"] == "unavailable"
    assert all(v["value"] is None and v["reason"] == reason for v in report["values"])


def test_no_audio_keeps_video_and_metadata_usable(media_factory, tmp_path):
    source, media = media_factory("silent-video")
    result = A.analyze_media(source, media, tmp_path / "video")
    report = loudness(result)
    assert report["scope"] == "file" and report["stream_index"] is None
    assert report["values"][0]["reason"] == "no_audio_stream"
    assert any(r["kind"] == "frame_timing" and r["status"] == "available" for r in result["reports"])


def test_actual_vfr_pts_are_not_inferred_from_nominal_fps(media_factory, tmp_path):
    source, media = media_factory("vfr")
    result = A.analyze_media(source, media, tmp_path / "vfr")
    report = next(r for r in result["reports"] if r["kind"] == "frame_timing")
    measured = values(report)
    assert measured["observed_timing"]["value"] == "VFR"
    assert measured["observed_fps"]["value"] is None
    assert measured["decoded_frame_count"]["value"] == 19


def test_audio_pts_offset_and_interval_no_audio_are_not_zero_padded(media_factory, tmp_path):
    source, media = media_factory("offset")
    no_audio = A.analyze_media(source, media, tmp_path / "before-audio", window={"start": S.rational(0), "end": S.rational(.2)})
    assert loudness(no_audio)["values"][0]["reason"] == "no_audio_in_window"
    overlap = A.analyze_media(source, media, tmp_path / "audio-overlap", window={"start": S.rational(.2), "end": S.rational(.5)})
    measured = values(loudness(overlap))
    assert measured["effective_audio_start"]["value"] == .25
    assert measured["effective_audio_end"]["value"] == .5
    assert measured["decoded_samples_per_channel"]["value"] == "12000"


def test_missing_tools_produce_typed_unavailable_without_fabrication(media_factory, tmp_path, monkeypatch):
    source, media = media_factory("mono")
    monkeypatch.setattr(A.shutil, "which", lambda _: None)
    result = A.analyze_media(source, media, tmp_path / "no-tools")
    assert all(r["status"] == "unavailable" for r in result["reports"])
    assert all(v["value"] is None for r in result["reports"] for v in r["values"])
    assert result["commands"] is None


def test_missing_ffmpeg_does_not_remove_probe_or_frame_timing(media_factory, tmp_path, monkeypatch):
    source, media = media_factory("mono")
    which = A.shutil.which
    monkeypatch.setattr(A.shutil, "which", lambda name: None if name == "ffmpeg" else which(name))
    result = A.analyze_media(source, media, tmp_path / "no-ffmpeg")
    assert any(r["kind"] == "frame_timing" and r["status"] == "available" for r in result["reports"])
    assert loudness(result)["values"][0]["reason"] == "ffmpeg_unavailable"


def test_source_identity_mismatch_is_rejected_before_new_output(media_factory, tmp_path):
    source, media = media_factory("mono")
    bad = dict(media, source=dict(media["source"], sha256="0" * 64))
    with pytest.raises(WorkspaceError) as error:
        A.analyze_media(source, bad, tmp_path / "wrong-source")
    assert error.value.code == "MEDIA_IDENTITY_MISMATCH"
    assert not (tmp_path / "wrong-source").exists()


def test_source_change_is_rejected_without_success_manifest(media_factory, tmp_path, monkeypatch):
    original, media = media_factory("mono")
    source = tmp_path / "mutable.mov"
    shutil.copyfile(original, source)
    probe = P._probe
    def changing_probe(*args, **kwargs):
        result = probe(*args, **kwargs)
        with source.open("ab") as out:
            out.write(b"changed after identity check")
        return result
    monkeypatch.setattr(P, "_probe", changing_probe)
    with pytest.raises(WorkspaceError) as error:
        A.analyze_media(source, media, tmp_path / "changed-analysis")
    assert error.value.code == "MEDIA_CHANGED"
    assert not (tmp_path / "changed-analysis/result.json").exists()


@pytest.mark.parametrize("kind,code", [("timeout", "ANALYSIS_TIMEOUT"), ("space", "STORAGE_FULL"), ("output", "ANALYSIS_LIMIT_EXCEEDED")])
def test_analysis_resource_guards_stop_without_false_result(media_factory, tmp_path, monkeypatch, kind, code):
    source, media = media_factory("mono")
    if kind == "timeout":
        monkeypatch.setattr(A, "TIMEOUT_SECONDS", -1)
    elif kind == "space":
        monkeypatch.setattr(A, "RESERVE_BYTES", 2 ** 62)
    else:
        monkeypatch.setattr(A, "MAX_WORK_BYTES", 0)
    with pytest.raises(WorkspaceError) as error:
        A.analyze_media(source, media, tmp_path / kind)
    assert error.value.code == code
    assert not (tmp_path / kind / "result.json").exists()
    assert not P.live_media_processes()


def test_actual_running_owned_child_is_cancelled_and_not_left_alive(media_factory, tmp_path):
    source, media = media_factory("mono")
    before = {p["pid"] for p in P.live_media_processes()}
    def cancelled():
        return any(p["name"].endswith("decode-measure") for p in P.live_media_processes() if p["pid"] not in before)
    with pytest.raises(WorkspaceError) as error:
        A.analyze_media(source, media, tmp_path / "cancelled", cancelled=cancelled)
    assert error.value.code == "CANCELLED"
    assert {p["pid"] for p in P.live_media_processes()} == before
    commands = [json.loads(line) for line in (tmp_path / "cancelled/commands.jsonl").read_text("utf-8").splitlines()]
    assert any(c["name"].endswith("decode-measure") and c["cancelled_or_interrupted"] for c in commands)
    assert not (tmp_path / "cancelled/result.json").exists()


def test_preexisting_output_is_not_overwritten(media_factory, tmp_path):
    source, media = media_factory("mono")
    target = tmp_path / "existing"
    target.mkdir(); (target / "user.txt").write_text("preserve")
    with pytest.raises(WorkspaceError) as error:
        A.analyze_media(source, media, target)
    assert error.value.code == "OUTPUT_EXISTS"
    assert (target / "user.txt").read_text() == "preserve"


def test_pcm_resource_budget_returns_explicit_unavailable(media_factory, tmp_path, monkeypatch):
    source, media = media_factory("mono")
    monkeypatch.setattr(A, "MAX_PCM_BYTES", 32)
    report = loudness(A.analyze_media(source, media, tmp_path / "small-budget"))
    assert report["values"][0]["reason"] == "decoded_pcm_budget_exceeded"
    assert not list((tmp_path / "small-budget").glob("*.f64le"))


def test_unavailable_audio_filter_does_not_revoke_valid_metadata(media_factory, tmp_path, monkeypatch):
    source, media = media_factory("mono")
    def unavailable(*args, **kwargs):
        raise P.MediaPreparationError("ebur128 unavailable in installed tool")
    monkeypatch.setattr(A, "_audio_report", unavailable)
    result = A.analyze_media(source, media, tmp_path / "no-audio-filter")
    assert any(r["kind"] == "technical_declared" and r["status"] == "available" for r in result["reports"])
    assert loudness(result)["values"][0]["reason"] == "audio_timeline_or_measurement_unavailable"


def test_unavailable_frame_timing_does_not_revoke_valid_audio(media_factory, tmp_path, monkeypatch):
    source, media = media_factory("mono")
    def unavailable(*args, **kwargs):
        raise P.MediaPreparationError("verified tail duration unavailable")
    monkeypatch.setattr(A, "_frame_report", unavailable)
    result = A.analyze_media(source, media, tmp_path / "no-frame-timing")
    assert values(loudness(result))["integrated_lufs"]["status"] == "available"
    report = next(r for r in result["reports"] if r["kind"] == "frame_timing")
    assert report["status"] == "unavailable"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_decoded_samples_never_become_json_values(tmp_path, bad):
    path = tmp_path / "bad.f64le"
    np.array([0, bad, 0], dtype="<f8").tofile(path)
    with pytest.raises(P.MediaPreparationError, match="nonfinite"):
        A._pcm_statistics(path, 48000, 1, None)


def test_pcm_silence_window_quantization_and_one_active_channel(tmp_path):
    path = tmp_path / "windows.f64le"
    audio = np.zeros((1205, 2), dtype="<f8")
    audio[305:805, 1] = .5
    audio.tofile(path)
    stats = A._pcm_statistics(path, 1000, 2, None)
    assert stats["silence"] == [(0, 300), (810, 1205)]
    assert stats["peak_dbfs"] == 20 * math.log10(.5)


@pytest.mark.parametrize("argument,value", [("stream_index", True), ("stream_index", -1), ("window", {"start": S.rational(0), "end": S.rational(0)}),
                                           ("time_domain", "guessed"), ("cancelled", 1), ("subject_annotation_id", "ann-no-window")])
def test_invalid_request_is_rejected_without_starting_a_tool(media_factory, tmp_path, argument, value):
    source, media = media_factory("mono")
    with pytest.raises(WorkspaceError):
        A.analyze_media(source, media, tmp_path / "invalid", **{argument: value})
    assert not (tmp_path / "invalid").exists()
