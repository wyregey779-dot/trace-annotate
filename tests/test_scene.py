"""Scene helper regressions, including real decoding when FFmpeg is installed."""
import json
import shutil
import subprocess

import pytest

from trace_annotate.config import MetricThresholds
from trace_annotate.extractors.scene import SceneDetectionError, SceneDetector


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
