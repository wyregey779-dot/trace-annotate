"""Scene helper regressions, including real decoding when FFmpeg is installed."""
import json
import re
import shutil
import subprocess

import pytest

from trace_annotate.config import MetricThresholds
from trace_annotate.extractors.scene import SceneDetectionError, SceneDetector, merge_adjacent_cuts


def _filtered_frame_pts(command):
    """Decode the production seek and trim, observing its surviving frames."""
    probe = command.copy()
    probe[probe.index("-v") + 1] = "info"
    filter_index = probe.index("-vf") + 1
    trim = probe[filter_index].split(",select=", 1)[0]
    assert trim.startswith("trim=start=")
    probe[filter_index] = f"{trim},showinfo"
    result = subprocess.run(probe, check=True, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    return [float(match.group(1)) for match in re.finditer(
        r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*(-?[\d.]+)", result.stderr)]


def _first_filtered_frame_pts(command):
    frames = _filtered_frame_pts(command)
    assert frames, "FFmpeg decoded no frames in the selected window"
    return frames[0]


@pytest.fixture
def make_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real scene tests require FFmpeg and ffprobe")
    counter = 0

    def generate(durations):
        nonlocal counter
        counter += 1
        path = tmp_path / f"video-{counter}.mp4"
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error"]
        for index, duration in enumerate(durations):
            color = "white" if index % 2 else "black"
            command += ["-f", "lavfi", "-i", f"color={color}:s=160x90:r=10:d={duration}"]
        inputs = "".join(f"[{index}:v]" for index in range(len(durations)))
        command += ["-filter_complex", f"{inputs}concat=n={len(durations)}:v=1:a=0[out]",
                    "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-n", str(path)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return path
    return generate


@pytest.mark.parametrize("durations,edges", [
    ([3, 4, 5], [0, 3, 7, 12]),
    ([0.2, 0.3, 0.2], [0, 0.2, 0.5, 0.7]),
    ([2], [0, 2]),
])
def test_real_video_is_fully_partitioned_including_short_and_final_shots(make_video, durations, edges):
    path = make_video(durations)
    result = SceneDetector().extract({"duration": 999}, str(path))
    assert result["status"] == "success"
    segments = result["scene_sub_segments"]
    assert [segments[0]["start"]] + [item["end"] for item in segments] == pytest.approx(edges)
    assert result["analyzed_duration"] == pytest.approx(edges[-1])
    assert all(item["duration"] == pytest.approx(item["end"] - item["start"]) for item in segments)
    assert all(previous["end"] == current["start"] for previous, current in zip(segments, segments[1:]))


@pytest.mark.parametrize("declared_duration", [0, 2, 12])
def test_nonexistent_video_cannot_become_a_valid_whole_clip(tmp_path, declared_duration):
    detector = SceneDetector()
    path = tmp_path / "missing.mp4"
    with pytest.raises(SceneDetectionError, match="existing regular file"):
        detector.detect_scenes_ffmpeg(str(path))
    result = detector.extract({"duration": declared_duration}, str(path))
    assert result["status"] == "failed"
    assert result["scene_sub_segments"] == []
    assert "existing regular file" in result["error"]


def test_merge_adjacent_cuts_drops_a_weak_duplicate() -> None:
    cuts = [(0.0, float("inf")), (1.0, 0.8), (1.02, 0.2), (2.0, 0.9), (2.2, float("inf"))]

    assert merge_adjacent_cuts(cuts, window=0.07, weak_ratio=0.5) == [(0.0, float("inf")), (1.0, 0.8), (2.0, 0.9), (2.2, float("inf"))]


def test_merge_adjacent_cuts_keeps_two_strong_cuts_within_one_frame() -> None:
    cuts = [(1.0, 0.8), (1.033, 0.9)]

    assert merge_adjacent_cuts(cuts, window=0.07, weak_ratio=0.5) == cuts


def test_nonvideo_bytes_fail_real_probe(tmp_path):
    if not shutil.which("ffprobe"):
        pytest.skip("real invalid-input test requires ffprobe")
    path = tmp_path / "not-video.mp4"
    path.write_bytes(b"This is not a video.")
    result = SceneDetector().extract({"duration": 12}, str(path))
    assert result["status"] == "failed"
    assert result["scene_sub_segments"] == []
    assert "ffprobe failed" in result["error"]


def test_real_corrupt_video_with_readable_metadata_does_not_fall_back(make_video):
    path = make_video([3, 4, 5])
    raw = bytearray(path.read_bytes())
    marker = raw.index(b"mdat")
    size = int.from_bytes(raw[marker - 4:marker], "big")
    raw[marker + 4:marker - 4 + size] = bytes(size - 8)
    path.write_bytes(raw)
    result = SceneDetector().extract({"duration": 12}, str(path))
    assert result["status"] == "failed"
    assert result["scene_sub_segments"] == []
    assert "ffmpeg failed" in result["error"]


def test_duration_cap_refuses_incomplete_analysis_instead_of_discarding_the_tail(make_video):
    path = make_video([3, 4, 5])
    detector = SceneDetector(MetricThresholds(scene_cap_seconds=5))
    result = detector.extract({"duration": 2}, str(path))
    assert result["status"] == "failed"
    assert result["scene_sub_segments"] == []
    assert "exceeds cap_seconds" in result["error"]
    complete = detector.detect_scenes_ffmpeg(str(path), cap_seconds=12)
    assert complete[-1]["end"] == pytest.approx(12)


def test_decoder_failure_discards_partial_detected_cuts(tmp_path, monkeypatch):
    path = tmp_path / "exists.mp4"
    path.write_bytes(b"stand-in for a decoder failure after valid metadata")

    def run(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "streams": [{"codec_type": "video", "duration": "12"}]}), "")
        return subprocess.CompletedProcess(command, 69, "frame:0 pts:30 pts_time:3\n", "corrupt packet")

    monkeypatch.setattr(subprocess, "run", run)
    result = SceneDetector().extract({"duration": 12}, str(path))
    assert result["status"] == "failed"
    assert result["scene_sub_segments"] == []
    assert "exit 69" in result["error"]


@pytest.mark.parametrize("error", [FileNotFoundError("ffprobe unavailable"), subprocess.TimeoutExpired("ffprobe", 30)])
def test_unavailable_or_timed_out_tool_is_an_explicit_failure(tmp_path, monkeypatch, error):
    path = tmp_path / "exists.mp4"
    path.write_bytes(b"stand-in")

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    result = SceneDetector().extract({"duration": 12}, str(path))
    assert result["status"] == "failed"
    assert result["scene_sub_segments"] == []
    assert "could not complete" in result["error"]


@pytest.mark.parametrize("kwargs", [{"threshold": float("nan")}, {"threshold": -1}, {"threshold": True},
                                    {"cap_seconds": 0}, {"cap_seconds": float("inf")},
                                    {"cap_seconds": 10**1000}])
def test_invalid_limits_are_not_submitted_to_ffmpeg(kwargs):
    with pytest.raises(ValueError):
        SceneDetector().detect_scenes_ffmpeg("unused.mp4", **kwargs)


def test_default_scene_threshold_keeps_short_shot_recall():
    assert SceneDetector().thresholds.scene_threshold == 0.15


def test_real_window_scans_only_selected_decoded_timeline(make_video):
    path = make_video([1, 1, 1, 1])
    detector = SceneDetector(MetricThresholds(scene_cap_seconds=2))
    with pytest.raises(SceneDetectionError, match="exceeds cap_seconds"):
        detector.detect_scenes_ffmpeg(str(path))
    parts = detector.detect_scenes_ffmpeg(str(path), window=(1.4, 3.4))
    assert [parts[0]["start"]] + [part["end"] for part in parts] == pytest.approx(
        [1.4, 2, 3, 3.4])
    assert all(left["end"] == right["start"] for left, right in zip(parts, parts[1:]))


def test_empty_decoded_window_is_not_reported_as_a_valid_partition(make_video, monkeypatch):
    from trace_annotate.extractors import scene as module

    path = make_video([1])  # 10 fps: frames at 0.0, 0.1, ...; none in [0.01, 0.02).
    original_run = module._run
    decoder_commands = []

    def capture_run(command, **kwargs):
        if command[0] == "ffmpeg":
            decoder_commands.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(module, "_run", capture_run)
    with pytest.raises(SceneDetectionError, match="no decoded video frames"):
        SceneDetector().detect_scenes_ffmpeg(str(path), window=(.01, .02))
    assert len(decoder_commands) == 1
    assert _filtered_frame_pts(decoder_commands[0]) == []


@pytest.mark.parametrize("durations,window,edges", [
    ([1], (0.0, 0.01), [0.0, 0.01]),  # Exactly one decoded frame.
    ([1], (0.2, 0.8), [0.2, 0.8]),   # Several frames, no scene cut.
    ([1, 1, 1], (0.2, 2.8), [0.2, 1.0, 2.0, 2.8]),
])
def test_decoded_windows_keep_only_real_scene_cuts(make_video, durations, window, edges):
    path = make_video(durations)
    parts = SceneDetector().detect_scenes_ffmpeg(str(path), window=window)
    assert [parts[0]["start"]] + [part["end"] for part in parts] == pytest.approx(edges)


def test_scene_window_uses_registered_video_offset(make_video, tmp_path):
    source = make_video([1, 1, 1, 1])
    shifted = tmp_path / "offset.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(source),
                    "-vf", "setpts=PTS+0.5/TB", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-avoid_negative_ts", "disabled", "-y", str(shifted)],
                   check=True, capture_output=True, timeout=30)
    detector = SceneDetector()
    parts = detector.detect_scenes_ffmpeg(
        str(shifted), window=(1.6, 3.6), timeline_start=.5)
    assert [parts[0]["start"]] + [part["end"] for part in parts] == pytest.approx(
        [1.6, 2.5, 3.5, 3.6])


@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_window_seek_decodes_first_selected_frame(make_video, tmp_path, monkeypatch, offset):
    from trace_annotate.extractors import scene as module

    source = make_video([1, 1, 1, 1])
    path = source
    if offset:
        path = tmp_path / "nonzero-format-start.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(source),
            "-vf", "setpts=PTS+0.5/TB", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-avoid_negative_ts", "disabled", "-y", str(path),
        ], check=True, capture_output=True, timeout=30)
    original_run = module._run
    decoder_commands = []

    def capture_run(command, **kwargs):
        if command[0] == "ffmpeg":
            decoder_commands.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(module, "_run", capture_run)
    start, end = 1.1 + offset, 3.1 + offset
    parts = SceneDetector().detect_scenes_ffmpeg(
        str(path), window=(start, end), timeline_start=offset)
    assert parts[0]["start"] == pytest.approx(start)
    assert parts[-1]["end"] == pytest.approx(end)
    assert len(decoder_commands) == 1
    assert _first_filtered_frame_pts(decoder_commands[0]) == pytest.approx(start, abs=.001)


def test_admitted_vfr_with_leading_audio_keeps_frame_map_time(make_video, tmp_path, monkeypatch):
    from trace_annotate.extractors import scene as module
    from trace_annotate.workspace import schema as S
    from trace_annotate.workspace.frames import FrameService, _binding
    from trace_annotate.workspace.store import WorkspaceStore

    source = make_video([1, 1, 1, 1])
    shifted = tmp_path / "offset-vfr-audio.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(source),
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=5",
        "-vf", r"select=not(eq(n\,3)+eq(n\,7)),setpts=PTS+0.5/TB",
        "-fps_mode", "passthrough", "-c:v", "libx264", "-bf", "0",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(shifted),
    ], check=True, capture_output=True, timeout=30)
    store = WorkspaceStore(tmp_path / "admitted")
    store.create()
    try:
        item = store.import_source(shifted)
        media = store.get_media(item["media_id"])
        assert S.rational_value(media["video_start"]) == .5
        rows = FrameService(store).page(item["media_id"], 0, 40)["frames"]
        seconds = [float(S.rational_value(row["seconds"])) for row in rows]
        assert len({round(right - left, 4) for left, right in zip(seconds, seconds[1:])}) > 1
        playback = _binding(store, item["media_id"], "playback_path")[1]
        original_run = module._run
        decoder_commands = []

        def capture_run(command, **kwargs):
            if command[0] == "ffmpeg":
                decoder_commands.append(command)
            return original_run(command, **kwargs)

        monkeypatch.setattr(module, "_run", capture_run)
        parts = SceneDetector().detect_scenes_ffmpeg(
            str(playback), window=(1.6, 3.6),
            timeline_start=float(S.rational_value(media["video_start"])))
        assert [parts[0]["start"]] + [part["end"] for part in parts] == pytest.approx(
            [1.6, 2.5, 3.5, 3.6])
        assert len(decoder_commands) == 1
        assert _first_filtered_frame_pts(decoder_commands[0]) == pytest.approx(1.6, abs=.001)
    finally:
        store.close()


def test_owned_scene_decoder_is_terminated_on_cancellation(make_video, monkeypatch):
    from trace_annotate.extractors import scene as module

    path = make_video([1, 1])
    real_popen = subprocess.Popen
    spawned = []
    checks = 0

    class StalledDecoder:
        returncode = None

        def __init__(self):
            self.terminated = False

        def communicate(self, timeout=None):
            if not self.terminated:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            self.returncode = -15
            return "", ""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

    def popen(command, **kwargs):
        if command[0] == "ffmpeg":
            process = StalledDecoder()
            spawned.append(process)
            return process
        return real_popen(command, **kwargs)

    def cancelled():
        nonlocal checks
        checks += 1
        return bool(spawned) and checks >= 3

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(InterruptedError, match="cancelled"):
        module.SceneDetector().detect_scenes_ffmpeg(str(path), cancelled=cancelled)
    assert len(spawned) == 1 and spawned[0].terminated


def test_vision_scene_stage_does_not_convert_cancellation_to_failed_result(make_video):
    from trace_annotate.workspace.vision_jobs import _run_scenes

    path = make_video([1, 1])
    with pytest.raises(InterruptedError, match="cancelled"):
        _run_scenes(SceneDetector(), path, MetricThresholds(), None, 0.0, lambda: True)


def test_vision_scene_stage_reports_empty_window_as_failed(make_video):
    from trace_annotate.workspace import schema as S
    from trace_annotate.workspace.vision_jobs import _run_scenes

    path = make_video([1])
    result = _run_scenes(
        SceneDetector(), path, MetricThresholds(),
        {"start": S.rational("0.01"), "end": S.rational("0.02")}, 0.0,
        lambda: False)
    assert result == {"status": "failed", "error": "scene_detection_failed", "scenes": []}
