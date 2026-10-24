"""PTS, rollback and real decoder coverage for compatibility preparation."""
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from trace_annotate.temporal import preparation as prep
from trace_annotate.temporal.media import stream_sha256
from trace_annotate.temporal.schema import validate_document


def rows(path, values):
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path


def test_pts_mapping_uses_real_vfr_timestamps_and_shared_origin(tmp_path):
    source = rows(tmp_path / "source", ["pts=7007|duration=1001", "pts=8008|duration=3003", "pts=11011|duration=1001"])
    proxy = rows(tmp_path / "proxy", ["pts=7|duration=1001", "pts=1008|duration=3003", "pts=4011|duration=1001"])
    output = tmp_path / "map.jsonl"
    assert prep._pair_frames(source, proxy, Fraction(1, 1000), Fraction(1, 1000), Fraction(7), output, None) == 3
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["source_pts"] for item in actual] == [7007, 8008, 11011]
    assert [item["proxy_time"] for item in actual] == [0.007, 1.008, 4.011]


@pytest.mark.parametrize("proxy_values,message", [
    (["pts=0|duration=1"], "lost"),
    (["pts=0|duration=1", "pts=100|duration=1", "pts=200|duration=1"], "extra"),
    (["pts=0|duration=1", "pts=110|duration=1"], "timestamp changed"),
])
def test_mapping_rejects_frame_loss_insertion_and_drift(tmp_path, proxy_values, message):
    original = rows(tmp_path / "source", ["pts=10|duration=1", "pts=110|duration=1"])
    proxy = rows(tmp_path / "proxy", proxy_values)
    with pytest.raises(prep.MediaPreparationError, match=message):
        prep._pair_frames(original, proxy, Fraction(1, 1000), Fraction(1, 1000), Fraction(1, 100), tmp_path / "map", None)


@pytest.mark.parametrize("values,message", [
    ([], "No decodable"),
    (["pts=0|duration=1", "pts=0|duration=1"], "strictly increasing"),
    (["pts=4|duration=1", "pts=3|duration=1"], "strictly increasing"),
    (["pts=0|duration=0"], "Final video frame duration"),
    (["best_effort_timestamp=0|duration=1"], "no original presentation timestamp"),
])
def test_video_summary_fails_closed_on_ambiguous_timeline(tmp_path, values, message):
    with pytest.raises(prep.MediaPreparationError, match=message):
        prep._video_summary(rows(tmp_path / "frames", values), Fraction(1, 1000), None)


def test_explicit_packet_tail_requires_matching_pts_even_when_packets_are_reordered(tmp_path):
    frames = rows(tmp_path / "frames", ["pts=100|duration=10", "pts=110|duration=0", "pts=140"])
    packets = rows(tmp_path / "packets", ["pts=100|duration=10", "pts=140|duration=7", "pts=110|duration=30"])
    result = prep._video_summary(frames, Fraction(1, 1000), None, packets=packets)
    assert result["count"] == 3 and result["last_duration"] == 7
    assert result["duration"] == Fraction(47, 1000)
    assert result["last_duration_basis"] == "explicit_demux_packet_duration_with_one_to_one_decoded_pts"


@pytest.mark.parametrize("packets", [
    ["pts=100|duration=10", "pts=110|duration=30"],
    ["pts=100|duration=10", "pts=140|duration=0", "pts=110|duration=30"],
    ["pts=100|duration=10", "pts=140|duration=7", "pts=140|duration=7"],
    ["pts=100|duration=10", "pts=140|duration=7", "pts=120|duration=30"],
    ["pts=100|duration=10", "pts=140|duration=7", "pts=110|duration=30", "pts=160|duration=9"],
    ["pts=100|duration=10", "pts=140|duration=7", "duration=30"],
])
def test_missing_ambiguous_or_incomplete_packet_tail_cannot_replace_missing_frame_duration(tmp_path, packets):
    frames = rows(tmp_path / "frames", ["pts=100|duration=10", "pts=110|duration=0", "pts=140"])
    with pytest.raises(prep.MediaPreparationError):
        prep._video_summary(frames, Fraction(1, 1000), None, packets=rows(tmp_path / "packets", packets))


def test_zero_frame_duration_can_use_its_explicit_packet_duration(tmp_path):
    frames = rows(tmp_path / "frames", ["pts=100|duration=0|pkt_duration=7"])
    result = prep._video_summary(frames, Fraction(1, 1000), None)
    assert result["duration"] == Fraction(7, 1000)


def test_missing_decoded_duration_reads_real_packets_without_guessing_fps(tmp_path, monkeypatch):
    frames = rows(tmp_path / "frames", ["pts=100", "pts=140"])
    packets = rows(tmp_path / "packets", ["pts=140|duration=7", "pts=100|duration=40"])
    calls = []
    def explicit_packets(*args):
        calls.append(args)
        return packets
    monkeypatch.setattr(prep, "_packets", explicit_packets)
    result = prep._verified_video_summary("ffprobe", tmp_path / "video.mp4", 2, frames,
                                          Fraction(1, 1000), tmp_path, "source-packets", None)
    assert len(calls) == 1 and calls[0][2] == 2
    assert result["last_duration"] == 7 and result["duration"] == Fraction(47, 1000)


@pytest.mark.parametrize("tail", ["pts=140|duration=-1", "pts=140|pkt_duration=-1",
                                   "pts=140|duration=-1|pkt_duration=7", "pts=140|duration=6|pkt_duration=7"])
def test_invalid_frame_duration_cannot_be_hidden_by_positive_packet_metadata(tmp_path, tail):
    frames = rows(tmp_path / "frames", ["pts=100|duration=40", tail])
    packets = rows(tmp_path / "packets", ["pts=100|duration=40", "pts=140|duration=7"])
    with pytest.raises(prep.MediaPreparationError, match="negative|conflict"):
        prep._video_summary(frames, Fraction(1, 1000), None, packets=packets)


def test_packet_tail_verification_honors_cancellation(tmp_path):
    frames = rows(tmp_path / "frames", ["pts=100", "pts=140"])
    packets = rows(tmp_path / "packets", ["pts=100|duration=40", "pts=140|duration=7"])
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls > 2
    with pytest.raises(InterruptedError, match="cancelled"):
        prep._video_summary(frames, Fraction(1, 1000), cancel, packets=packets)


def test_packet_tail_verification_refuses_its_resource_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "_MAX_PACKET_VERIFICATION_FRAMES", 1)
    frames = rows(tmp_path / "frames", ["pts=100", "pts=140"])
    packets = rows(tmp_path / "packets", ["pts=100|duration=40", "pts=140|duration=7"])
    with pytest.raises(prep.MediaPreparationError, match="resource limit"):
        prep._video_summary(frames, Fraction(1, 1000), None, packets=packets)


def test_audio_allows_one_container_tick_but_rejects_real_gap(tmp_path):
    normal = rows(tmp_path / "normal", ["pts=0|nb_samples=648", "pts=14|nb_samples=960", "pts=34|nb_samples=960"])
    actual = prep._audio_summary(normal, Fraction(1, 1000), 48000, None)
    assert actual["samples"] == 2568
    assert actual["end"] == Fraction(54, 1000)
    gap = rows(tmp_path / "gap", ["pts=0|nb_samples=648", "pts=30|nb_samples=960"])
    with pytest.raises(prep.MediaPreparationError, match="Discontinuous"):
        prep._audio_summary(gap, Fraction(1, 1000), 48000, None)


def test_known_audio_tail_flush_inference_is_explicit_and_middle_missing_pts_fail(tmp_path):
    tail = rows(tmp_path / "tail", ["pts=0|nb_samples=960", "pts=20|nb_samples=960", "nb_samples=960"])
    result = prep._audio_summary(tail, Fraction(1, 1000), 48000, None, allow_trailing_flush_inference=True)
    assert result["samples"] == 2880 and result["end"] == Fraction(60, 1000)
    assert result["inferred_tail_frames"] == [{"frame": 2, "start": .04, "end": .06, "samples": 960}]
    assert "inferred" in result["timestamp_basis"]
    middle = rows(tmp_path / "middle", ["pts=0|nb_samples=960", "nb_samples=960", "pts=40|nb_samples=960"])
    with pytest.raises(prep.MediaPreparationError, match="no middle timestamps"):
        prep._audio_summary(middle, Fraction(1, 1000), 48000, None, allow_trailing_flush_inference=True)
    with pytest.raises(prep.MediaPreparationError, match="no original presentation timestamp"):
        prep._audio_summary(tail, Fraction(1, 1000), 48000, None)


def test_audio_missing_first_or_two_flush_timestamps_cannot_be_inferred(tmp_path):
    for index, values in enumerate((["nb_samples=960"], ["pts=0|nb_samples=960", "nb_samples=960", "nb_samples=960"])):
        with pytest.raises(prep.MediaPreparationError, match="Only one known-codec trailing"):
            prep._audio_summary(rows(tmp_path / str(index), values), Fraction(1, 1000), 48000, None,
                                allow_trailing_flush_inference=True)


def test_existing_output_is_rejected_before_tools_and_keeps_user_files(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"untouched")
    target = tmp_path / "existing"
    target.mkdir()
    note = target / "user.txt"
    note.write_bytes(b"existing user data")
    monkeypatch.setattr(prep, "_tool", lambda name: pytest.fail("must not look for tools"))
    with pytest.raises(FileExistsError, match="already exists"):
        prep.prepare_media(source, target)
    assert note.read_bytes() == b"existing user data"
    assert source.read_bytes() == b"untouched"


def test_missing_tool_and_immediate_cancellation_create_nothing(tmp_path, monkeypatch):
    source = tmp_path / "media"
    source.write_bytes(b"source")
    target = tmp_path / "new"
    monkeypatch.setattr(prep.shutil, "which", lambda name: None)
    with pytest.raises(prep.MediaPreparationError, match="ffmpeg is required"):
        prep.prepare_media(source, target)
    with pytest.raises(InterruptedError, match="cancelled"):
        prep.prepare_media(source, target, cancelled=lambda: True)
    assert not target.exists()
    assert not list(tmp_path.glob(".*-prepare-*"))


def test_cancelled_tool_is_terminated_and_cancellation_logged(tmp_path):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(InterruptedError):
        prep._run([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, "sleep", cancel)
    entry = json.loads((tmp_path / "commands.jsonl").read_text())
    assert entry["cancelled_or_interrupted"] is True
    assert entry["exit_code"] is not None
    assert entry["elapsed_seconds"] < 5


def test_publication_cancellation_removes_only_own_entries(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "playback.mp4").write_bytes(b"video")
    (staged / "project.json").write_bytes(b"manifest")
    target = tmp_path / "target"
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            (target / "user-added.txt").write_bytes(b"must survive")
            return True
        return False

    with pytest.raises(InterruptedError):
        prep._publish_new(staged, target, cancel)
    assert (target / "user-added.txt").read_bytes() == b"must survive"
    assert not (target / "playback.mp4").exists()
    assert not (target / "project.json").exists()
    assert (staged / "playback.mp4").read_bytes() == b"video"


@pytest.fixture
def real_tools():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg/ffprobe are not installed; real preparation checks require both")
    return ffmpeg, ffprobe


def generate(real_tools, output, *options):
    command = [real_tools[0], "-hide_banner", "-v", "error", "-nostdin", "-n", *options, str(output)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    return output


@pytest.mark.parametrize("codec,suffix", [("ffv1", ".mkv"), ("mpeg4", ".avi"), ("prores_ks", ".mov")])
def test_real_preparation_converts_unsupported_browser_codecs_without_source_changes(tmp_path, real_tools, codec, suffix):
    source = generate(real_tools, tmp_path / ("source" + suffix), "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.6", "-c:v", codec)
    before = stream_sha256(source)
    target = tmp_path / "project"
    manifest = prep.prepare_media(source, target)
    assert manifest["timeline"]["frame_count"] == 6
    assert manifest["video"]["sha256"] == stream_sha256(target / "playback.mp4")
    assert manifest["source"]["sha256"] == before == stream_sha256(source)
    doc = json.loads((target / "annotations.json").read_text(encoding="utf-8"))
    assert validate_document(doc) == [] and doc["segments"] == []
    assert doc["media_sha256"] == manifest["video"]["sha256"]
    assert json.loads((target / "project.json").read_text(encoding="utf-8")) == manifest
    log = json.loads((target / "preparation-log.json").read_text(encoding="utf-8"))
    assert log["source_before"]["sha256"] == log["source_after"]["sha256"] == before
    assert all(item["exit_code"] == 0 for item in log["commands"])
    assert log["tools"]["ffmpeg"].startswith("ffmpeg version")
    assert len((target / "frame-map.jsonl").read_text().splitlines()) == 6


def test_real_vfr_preserves_all_frame_pts_and_explicit_tail(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "vfr.mp4", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.8",
                      "-vf", "settb=1/30000,setpts=if(lt(N\\,4)\\,N/(10*TB)\\,(0.4+(N-4)/30)/TB)",
                      "-c:v", "libx264", "-fps_mode", "passthrough", "-enc_time_base", "1/30000")
    manifest = prep.prepare_media(source, tmp_path / "project")
    assert manifest["timeline"]["variable_frame_rate"] is True
    mapping = [json.loads(line) for line in (tmp_path / "project" / "frame-map.jsonl").read_text().splitlines()]
    assert len(mapping) == 8
    source_base = Fraction(manifest["timeline"]["source_time_base"])
    proxy_base = Fraction(manifest["timeline"]["proxy_time_base"])
    origin = Fraction(manifest["timeline"]["origin_time_fraction"])
    assert all(abs(row["source_pts"] * source_base - origin - row["proxy_pts"] * proxy_base) <= proxy_base for row in mapping)


def test_real_packet_tail_is_verified_when_decoder_duration_fields_are_unavailable(tmp_path, real_tools, monkeypatch):
    """Exercise FFmpeg 6's absent-frame-duration shape using real packet data.

    Only the decoded-summary view omits duration fields. Original media,
    demuxer packets, full decoding and mapping checks remain real and unchanged.
    """
    original_summary = prep._video_summary
    calls = []
    def without_decoded_durations(path, base, cancelled, *, packets=None):
        view = path.with_name(path.name + ".no-decoded-duration")
        view.write_text("\n".join("|".join(part for part in line.split("|")
                                          if part.partition("=")[0] not in {"duration", "pkt_duration"})
                                  for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        result = original_summary(view, base, cancelled, packets=packets)
        calls.append(result["last_duration_basis"])
        return result
    monkeypatch.setattr(prep, "_video_summary", without_decoded_durations)
    source = generate(real_tools, tmp_path / "vfr-with-audio.mp4", "-f", "lavfi", "-i",
                      "testsrc2=size=160x90:rate=10:duration=0.8", "-f", "lavfi", "-i",
                      "sine=frequency=440:sample_rate=48000:duration=0.8",
                      "-vf", "settb=1/30000,setpts=if(lt(N\\,4)\\,N/(10*TB)\\,(0.4+(N-4)/30)/TB)",
                      "-c:v", "libx264", "-fps_mode", "passthrough", "-enc_time_base", "1/30000", "-c:a", "aac")
    before = stream_sha256(source)
    target = tmp_path / "project"
    manifest = prep.prepare_media(source, target)
    assert manifest["timeline"]["frame_count"] == 8
    assert manifest["timeline"]["variable_frame_rate"] is True
    assert manifest["timeline"]["audio"]["stream_count"] == 1
    assert calls == ["explicit_demux_packet_duration_with_one_to_one_decoded_pts"] * 2
    assert manifest["timeline"]["last_frame_duration"]["source_basis"] == calls[0]
    assert manifest["timeline"]["last_frame_duration"]["playback_basis"] == calls[1]
    assert manifest["source"]["sha256"] == before == stream_sha256(source)
    log = json.loads((target / "preparation-log.json").read_text(encoding="utf-8"))
    assert {entry["name"] for entry in log["commands"]} >= {"source-video-packets", "playback-video-packets", "playback-full-decode"}
    assert all(entry["exit_code"] == 0 for entry in log["commands"])


def test_real_audio_leading_video_is_preserved_on_common_zero_based_timeline(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "offset.mp4", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=1",
                      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                      "-vf", "setpts=PTS+0.1/TB", "-c:v", "libx264", "-fps_mode", "passthrough", "-c:a", "aac")
    manifest = prep.prepare_media(source, tmp_path / "project")
    timeline = manifest["timeline"]
    assert timeline["origin_time"] == 0
    assert timeline["first_video_time"] == pytest.approx(0.1)
    assert timeline["playback_first_video_time"] == pytest.approx(0.1)
    track = timeline["audio"]["tracks"][0]
    assert track["expected_start"] == 0
    assert track["playback_start"] == pytest.approx(0, abs=track["tolerance_seconds"])


def test_real_multiple_audio_tracks_are_retained(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "multi.mkv", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.6",
                      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.6",
                      "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=0.6",
                      "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "ffv1", "-c:a", "pcm_s16le")
    manifest = prep.prepare_media(source, tmp_path / "project")
    assert manifest["timeline"]["audio"]["stream_count"] == 2
    assert len(manifest["timeline"]["audio"]["tracks"]) == 2


def test_real_wmv_wma_retains_video_pts_and_records_audio_flush_inference(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "source.wmv", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=2",
                      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2", "-c:v", "wmv2", "-c:a", "wmav2")
    manifest = prep.prepare_media(source, tmp_path / "project")
    assert manifest["timeline"]["frame_count"] == 50
    track = manifest["timeline"]["audio"]["tracks"][0]
    assert "trailing_decoder_flush_inferred" in track["source_timestamp_basis"]
    assert len(track["source_inferred_tail_frames"]) == 1
    assert track["source_decoded_samples"] > 80000
    assert abs(track["source_decoded_samples"] - track["playback_decoded_samples"]) < 2048


def test_real_full_range_mjpeg_uses_pixel_conversion_and_limited_range_output(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "source.mov", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.4",
                      "-c:v", "mjpeg", "-pix_fmt", "yuvj420p")
    manifest = prep.prepare_media(source, tmp_path / "project")
    assert manifest["conversion"]["source_color_range"] == "pc"
    assert manifest["conversion"]["playback_color_range"] == "tv"
    assert manifest["conversion"]["range_conversion"] == "full_to_limited_pixel_scaling"
    log = json.loads((tmp_path / "project" / "preparation-log.json").read_text(encoding="utf-8"))
    command = next(item["args"] for item in log["commands"] if item["name"] == "transcode")
    assert "scale=in_range=full:out_range=limited" in command[command.index("-vf") + 1]


@pytest.mark.parametrize("source_kind", ["audio", "text", "empty", "truncated"])
def test_real_invalid_sources_never_publish_partial_projects(tmp_path, real_tools, source_kind):
    if source_kind == "audio":
        source = generate(real_tools, tmp_path / "sound.mp4", "-f", "lavfi", "-i", "sine=duration=0.2", "-c:a", "aac")
    elif source_kind == "truncated":
        source = generate(real_tools, tmp_path / "bad.mp4", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.5", "-c:v", "libx264")
        source.write_bytes(source.read_bytes()[:100])
    else:
        source = tmp_path / "bad.mp4"
        source.write_bytes(b"not a video" if source_kind == "text" else b"")
    before = source.read_bytes()
    target = tmp_path / "project"
    with pytest.raises(prep.MediaPreparationError):
        prep.prepare_media(source, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".*-prepare-*"))
    assert source.read_bytes() == before


def test_real_cancellation_during_transcode_leaves_no_output(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "video.mkv", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=0.5", "-c:v", "ffv1")
    before = stream_sha256(source)

    def transcode_started():
        return any((entry / "transcode.stdout").exists() for entry in tmp_path.glob(".*-prepare-*"))

    with pytest.raises(InterruptedError):
        prep.prepare_media(source, tmp_path / "project", cancelled=transcode_started)
    assert not (tmp_path / "project").exists()
    assert not list(tmp_path.glob(".*-prepare-*"))
    assert stream_sha256(source) == before


@pytest.mark.parametrize("name,payload", [
    ("playlist.m3u8", b"#EXTM3U\n#EXTINF:1,\nhttps://example.invalid/private.ts\n"),
    ("disguised.mp4", b"ffconcat version 1.0\nfile 'other-video.mp4'\n"),
    ("manifest.mp4", b'<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011"/>'),
])
def test_manifests_are_rejected_before_ffprobe_can_follow_dependencies(tmp_path, real_tools, monkeypatch, name, payload):
    source = tmp_path / name
    source.write_bytes(payload)
    monkeypatch.setattr(prep, "_probe", lambda *args: pytest.fail("external dependencies must be rejected before probing"))
    with pytest.raises(prep.MediaPreparationError, match="not self-contained"):
        prep.prepare_media(source, tmp_path / "project")
    assert source.read_bytes() == payload and not (tmp_path / "project").exists()


def test_real_hdr_requires_explicit_opt_in_and_marks_lossy_sdr_conversion(tmp_path, real_tools):
    source = generate(real_tools, tmp_path / "pq.mp4", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.4",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020",
                      "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
                      "-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc")
    before = stream_sha256(source)
    with pytest.raises(prep.MediaPreparationError, match="no silent SDR"):
        prep.prepare_media(source, tmp_path / "default-must-fail")
    assert not (tmp_path / "default-must-fail").exists()
    manifest = prep.prepare_media(source, tmp_path / "opt-in-project", tone_map_hdr=True)
    assert manifest["timeline"]["frame_count"] == 4
    assert manifest["conversion"]["hdr_tone_mapping"]["applied"] is True
    assert manifest["conversion"]["lossy"] is True
    log = json.loads((tmp_path / "opt-in-project" / "preparation-log.json").read_text(encoding="utf-8"))
    video = next(item for item in log["playback_probe"]["streams"] if item["codec_type"] == "video")
    assert video["color_primaries"] == video["color_transfer"] == video["color_space"] == "bt709"
    assert video["pix_fmt"] == "yuv420p"
    assert stream_sha256(source) == before


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_hdr_opt_in_requires_an_actual_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="tone_map_hdr must be a boolean"):
        prep.prepare_media(tmp_path / "absent", tmp_path / "new", tone_map_hdr=value)
