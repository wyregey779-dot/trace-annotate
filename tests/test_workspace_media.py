"""Real decoder evidence for workspace media admission and precision."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from trace_annotate.temporal.media import stream_sha256
from trace_annotate.workspace import media
from trace_annotate.workspace.errors import Conflict, DependencyUnavailable, NotFound, SchemaInvalid, WorkspaceError
from trace_annotate.workspace.schema import rational_value


@pytest.fixture()
def real_tools():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("real ffmpeg and ffprobe required for media admission")
    return ffmpeg, ffprobe


def generate(tools, path, *arguments):
    subprocess.run([tools[0], "-hide_banner", "-v", "error", "-nostdin", "-n", *arguments, str(path)],
                   check=True, capture_output=True, timeout=30)
    return path


def frames(prepared):
    return [json.loads(line) for line in prepared["frame_map_path"].read_text(encoding="utf-8").splitlines()]


def assert_identity_and_real_decode(source, prepared):
    assert prepared["source_sha256"] == stream_sha256(source)
    assert prepared["source_size"] == source.stat().st_size
    fields = prepared["media_fields"]
    assert fields["playback"]["sha256"] == stream_sha256(prepared["playback_path"])
    assert int(fields["playback"]["size_bytes"]) == prepared["playback_path"].stat().st_size
    assert prepared["frame_map_sha256"] == stream_sha256(prepared["frame_map_path"])
    evidence = json.loads(prepared["preparation_log_path"].read_text(encoding="utf-8"))
    commands = {item["name"]: item for item in evidence["commands"]}
    assert {"source-video-frames", "playback-video-frames", "playback-full-decode"} <= commands.keys()
    assert all(item["exit_code"] == 0 for item in commands.values())
    assert prepared["manifest_path"].is_file()
    rows = frames(prepared)
    assert fields["frame_count"] == str(len(rows))
    source_base = rational_value(fields["source_time_base"])
    proxy_base = rational_value(fields["proxy_time_base"])
    origin = rational_value(fields["common_origin"])
    for index, row in enumerate(rows):
        assert row["frame_index"] == str(index)
        assert isinstance(row["source_pts"], str) and isinstance(row["proxy_pts"], str)
        time = rational_value(row["seconds"])
        assert time == int(row["proxy_pts"]) * proxy_base
        assert abs(time - (int(row["source_pts"]) * source_base - origin)) <= min(source_base, proxy_base)
    return rows


def test_real_cfr_admission_has_actual_frames_and_full_decode_evidence(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "source.mp4", "-f", "lavfi", "-i",
                      "testsrc2=size=160x96:rate=10:duration=2", "-c:v", "libx264", "-bf", "0")
    prepared = media.prepare_workspace_media(source, tmp_path / "prepared")
    rows = assert_identity_and_real_decode(source, prepared)
    assert [rational_value(row["seconds"]) for row in rows] == [Fraction(i, 10) for i in range(20)]
    fields = prepared["media_fields"]
    assert rational_value(fields["video_start"]) == 0
    assert rational_value(fields["video_end"]) == rational_value(fields["duration"]) == 2
    assert (fields["source_width"], fields["source_height"]) == (160, 96)
    assert fields["rotation_degrees"] == 0


def test_real_vfr_admission_preserves_individual_pts_instead_of_average_fps(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "vfr.mp4", "-f", "lavfi", "-i",
                      "testsrc2=size=160x96:rate=10:duration=3", "-vf",
                      r"settb=1/1000,setpts=if(lt(N\,10)\,N*100\,1000+(N-10)*200)",
                      "-fps_mode", "passthrough", "-enc_time_base", "1:1000", "-c:v", "libx264", "-bf", "0")
    prepared = media.prepare_workspace_media(source, tmp_path / "prepared")
    rows = assert_identity_and_real_decode(source, prepared)
    expected = [Fraction(i, 10) for i in range(10)] + [1 + Fraction(i, 5) for i in range(20)]
    assert [rational_value(row["seconds"]) for row in rows] == expected
    assert len(rows) == 30
    assert rational_value(prepared["media_fields"]["video_end"]) > expected[-1]
    manifest = json.loads(prepared["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["timeline"]["variable_frame_rate"] is True


def test_real_audio_leading_and_outlasting_video_uses_shared_clock(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "audio-leading.mp4", "-f", "lavfi", "-i",
                      "testsrc2=size=160x96:rate=10:duration=1", "-f", "lavfi", "-i",
                      "sine=frequency=440:sample_rate=48000:duration=3", "-vf", "setpts=PTS+1/TB",
                      "-fps_mode", "passthrough", "-c:v", "libx264", "-bf", "0", "-c:a", "aac")
    prepared = media.prepare_workspace_media(source, tmp_path / "prepared")
    rows = assert_identity_and_real_decode(source, prepared)
    fields = prepared["media_fields"]
    assert rational_value(fields["common_origin"]) == 0
    assert rational_value(fields["video_start"]) == 1
    assert rational_value(fields["video_end"]) == 2
    assert rational_value(fields["duration"]) >= 3
    assert [rational_value(row["seconds"]) for row in rows] == [1 + Fraction(i, 10) for i in range(10)]


def test_real_rotation_and_aspect_keep_source_pixel_coordinates(tmp_path, real_tools):
    original = generate(real_tools, tmp_path / "original.mp4", "-f", "lavfi", "-i",
                        "testsrc2=size=160x96:rate=10:duration=0.3", "-vf", "setsar=4/3", "-c:v", "libx264")
    source = generate(real_tools, tmp_path / "rotated.mp4", "-display_rotation:v:0", "90",
                      "-i", str(original), "-c", "copy")
    prepared = media.prepare_workspace_media(source, tmp_path / "prepared")
    assert_identity_and_real_decode(source, prepared)
    fields = prepared["media_fields"]
    assert (fields["source_width"], fields["source_height"]) == (160, 96)
    assert fields["rotation_degrees"] == 90
    assert rational_value(fields["pixel_aspect_ratio"]) == Fraction(4, 3)
    assert (prepared["display_dimensions"]["playback_width"], prepared["display_dimensions"]["playback_height"]) == (96, 160)
    assert rational_value(prepared["display_dimensions"]["display_width"]) == 96
    assert rational_value(prepared["display_dimensions"]["display_height"]) == Fraction(640, 3)


def test_real_corrupt_media_has_typed_failure_and_no_admission(tmp_path, real_tools):
    source = tmp_path / "corrupt.mp4"
    source.write_bytes(b"not a self-contained video")
    with pytest.raises(SchemaInvalid) as error:
        media.prepare_workspace_media(source, tmp_path / "prepared")
    assert error.value.code == "UNSUPPORTED_MEDIA"
    assert str(tmp_path) not in json.dumps(error.value.body("request"))
    assert source.read_bytes() == b"not a self-contained video"
    assert not (tmp_path / "prepared").exists()


def test_existing_output_is_preserved_without_invoking_tools(tmp_path, monkeypatch):
    target = tmp_path / "existing"
    target.mkdir()
    note = target / "user.txt"
    note.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(media, "prepare_media", lambda *a, **kw: pytest.fail("must not run preparation"))
    with pytest.raises(Conflict) as error:
        media.prepare_workspace_media(tmp_path / "missing.mp4", target)
    assert error.value.code == "OUTPUT_EXISTS"
    assert note.read_text(encoding="utf-8") == "keep"


def test_missing_source_has_typed_failure(tmp_path):
    with pytest.raises(NotFound) as error:
        media.prepare_workspace_media(tmp_path / "missing.mp4", tmp_path / "prepared")
    assert error.value.code == "NOT_FOUND"
    assert not (tmp_path / "prepared").exists()


def test_missing_decoder_is_dependency_failure_without_false_media(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(DependencyUnavailable) as error:
        media.prepare_workspace_media(source, tmp_path / "prepared")
    assert error.value.code == "DEPENDENCY_UNAVAILABLE"
    assert error.value.details["tools"] == ["ffmpeg", "ffprobe"]
    assert not (tmp_path / "prepared").exists()


def test_cancellation_is_typed_and_does_not_publish(tmp_path, real_tools):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    with pytest.raises(WorkspaceError) as error:
        media.prepare_workspace_media(source, tmp_path / "prepared", cancelled=lambda: True)
    assert error.value.code == "CANCELLED"
    assert not (tmp_path / "prepared").exists()
