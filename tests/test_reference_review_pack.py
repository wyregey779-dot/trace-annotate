"""Review packs bind decoded PNGs to actual VFR frame PTS without sign-off."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_reference_review_pack as pack  # noqa: E402
import build_reference_truth as reference  # noqa: E402


@pytest.fixture
def observed_vfr(tmp_path):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg and ffprobe are required for actual decoded-frame evidence")
    video = tmp_path / "vfr.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n",
                    "-f", "lavfi", "-i", "testsrc2=size=80x48:rate=10:duration=0.6",
                    "-vf", "settb=1/1000,setpts=10+N*N*7", "-c:v", "libx264",
                    "-fps_mode", "passthrough", "-enc_time_base", "1/1000",
                    str(video)], check=True, timeout=30)
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                            "-show_streams", "-show_frames", "-of", "json", str(video)],
                           check=True, capture_output=True, timeout=30)
    document = json.loads(probe.stdout)
    frame_map = tmp_path / "frame-map.json"
    frame_map.write_text(json.dumps(document), encoding="utf-8")
    streams = [item for item in document["streams"] if item["codec_type"] == "video"]
    assert len(streams) == 1
    pts = [item["pts"] for item in document["frames"]]
    assert pts[0] > 0 and len({b - a for a, b in zip(pts, pts[1:])}) > 1
    base = reference.rational(streams[0]["time_base"])
    digest = reference.sha256(video)
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({
        "media": {"source": {"sha256": digest}, "frame_count": str(len(pts)),
                  "common_origin": {"n": "0", "d": "1"}},
        "task": {"annotations": [{"id": "shot-2", "track": "Shots",
                                   "label": "observed shot boundary", "tags": [],
                                   "start": {"seconds": reference.packed(pts[2] * base),
                                             "anchor": {"frame_index": "2", "source_pts": str(pts[2]),
                                                        "proxy_pts": str(pts[2])}},
                                   "end": {"seconds": reference.packed(pts[3] * base)},
                                   "review_status": "unreviewed",
                                   "provenance": {"origin": "synthetic_test", "actor_id": "pytest"}}]}
    }), encoding="utf-8")
    candidate = reference.build_reference(annotation, video, frame_map, digest, "fixture")
    source = tmp_path / "reference.json"
    source.write_text(json.dumps(candidate), encoding="utf-8")
    return source, video, frame_map, annotation, pts


def test_vfr_review_pack_uses_actual_before_and_current_pts(observed_vfr, tmp_path):
    source, video, frame_map, annotation, pts = observed_vfr
    output = tmp_path / "review-pack"
    manifest = pack.build_pack(source, output)
    assert manifest["actual_decoded_frames_verified"] == len(pts)
    assert manifest["human_review"] is False and manifest["holdout_sealed"] is False
    event = manifest["events"][0]
    assert event["review_status"] == "unreviewed"
    assert event["before"]["pts"] == pts[1] and event["current"]["pts"] == pts[2]
    assert event["ineligible_reasons"]
    for key in ("before", "current"):
        entry = event[key]
        image = output / entry["path"]
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert reference.sha256(image) == entry["sha256"]
    assert (output / "manifest.json").is_file()
    assert manifest["video_sha256"] == reference.sha256(video)
    assert manifest["frame_map_sha256"] == reference.sha256(frame_map)
    assert manifest["annotation_sha256"] == reference.sha256(annotation)
    with pytest.raises(ValueError, match="already exists"):
        pack.build_pack(source, output)


def test_tampered_frame_map_is_rejected_by_actual_decoded_pts(observed_vfr, tmp_path):
    _, video, frame_map, annotation, pts = observed_vfr
    document = json.loads(frame_map.read_text(encoding="utf-8"))
    document["frames"][0]["pts"] = pts[0] - 1
    frame_map.write_text(json.dumps(document), encoding="utf-8")
    candidate = reference.build_reference(annotation, video, frame_map,
                                          reference.sha256(video), "fixture")
    source = tmp_path / "tampered-reference.json"
    source.write_text(json.dumps(candidate), encoding="utf-8")
    output = tmp_path / "failed-pack"
    with pytest.raises(ValueError, match="actual decoded frame PTS"):
        pack.build_pack(source, output)
    assert (output / "FAILED.json").is_file()
    assert not (output / "manifest.json").exists()


def test_large_selection_is_rejected_before_png_decode(observed_vfr, tmp_path, monkeypatch):
    source = observed_vfr[0]
    monkeypatch.setattr(pack, "MAX_IMAGE_BYTES", 1)
    output = tmp_path / "over-budget"
    with pytest.raises(ValueError, match="uncompressed image budget"):
        pack.build_pack(source, output)
    assert (output / "FAILED.json").is_file()
    assert not list(output.glob("decoded-*.png"))


def test_probe_stdout_limit_and_reference_change_fail_closed(observed_vfr, tmp_path, monkeypatch):
    out, err = tmp_path / "oversize.stdout", tmp_path / "oversize.stderr"
    with pytest.raises(ValueError, match="bounded evidence size"):
        pack._run([sys.executable, "-c", "import sys;sys.stdout.write('x'*1000)"],
                  out, err, 5, max_stdout=10)
    source = observed_vfr[0]
    original = pack._run_images

    def change_reference_after_decode(command, work, timeout):
        original(command, work, timeout)
        source.write_bytes(source.read_bytes() + b" ")

    monkeypatch.setattr(pack, "_run_images", change_reference_after_decode)
    output = tmp_path / "changed-reference"
    with pytest.raises(ValueError, match="source bytes changed"):
        pack.build_pack(source, output)
    assert (output / "FAILED.json").is_file()
    assert not (output / "manifest.json").exists()


def test_changed_source_video_and_empty_map_fail_closed(observed_vfr, tmp_path):
    source, video, frame_map, annotation, _ = observed_vfr
    video.write_bytes(video.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="video bytes"):
        pack.build_pack(source, tmp_path / "changed-video-pack")
    assert not (tmp_path / "changed-video-pack").exists()
    video.write_bytes(video.read_bytes()[:-7])
    empty = json.loads(frame_map.read_text(encoding="utf-8"))
    empty["frames"] = []
    empty["streams"][0]["nb_frames"] = "0"
    frame_map.write_text(json.dumps(empty), encoding="utf-8")
    with pytest.raises(ValueError, match="empty decoded frame map"):
        reference.build_reference(annotation, video, frame_map,
                                  reference.sha256(video), "fixture")
