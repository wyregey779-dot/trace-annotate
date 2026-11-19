"""Review packs bind decoded PNGs to actual VFR frame PTS without sign-off."""
from __future__ import annotations

import json
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_reference_review_pack as pack  # noqa: E402
import build_reference_truth as reference  # noqa: E402


class _PageTags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


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
    page = output / manifest["review_page"]["path"]
    css = output / manifest["review_page"]["stylesheet"]["path"]
    assert reference.sha256(page) == manifest["review_page"]["sha256"]
    assert reference.sha256(css) == manifest["review_page"]["stylesheet"]["sha256"]
    assert b"\r\n" not in page.read_bytes()
    assert b"\r\n" not in css.read_bytes()
    page_text = page.read_text(encoding="utf-8")
    assert "未人工复核" in page_text
    assert page_text.index('<div class="frames">') < page_text.index('<details class="metadata">')
    assert '<details class="sources">' in page_text
    assert "源视频 PTS 时间基" in page_text and "标注时间轴秒数" in page_text
    assert str(reference.rational(event["current"]["time_base"])) in page_text
    assert str(reference.rational(event["current"]["seconds"])) in page_text
    assert css.read_text(encoding="utf-8").startswith(":root")
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


def test_local_review_page_escapes_untrusted_annotation_and_has_no_active_content(observed_vfr, tmp_path):
    source, video, frame_map, annotation, _ = observed_vfr
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["task"]["annotations"][0]["label"] = '<script>alert("x")</script>'
    payload["task"]["annotations"][0]["provenance"]["actor_id"] = '"><img src=x onerror=alert(1)>'
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    source.write_text(json.dumps(reference.build_reference(
        annotation, video, frame_map, reference.sha256(video), "fixture")), encoding="utf-8")
    output = tmp_path / "safe-page"
    manifest = pack.build_pack(source, output)
    page = (output / "index.html").read_text(encoding="utf-8")
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in page
    assert '&lt;img src=x onerror=alert(1)&gt;' in page
    assert "default-src 'none'" in page and "img-src 'self'" in page
    assert manifest["human_review"] is False
    parsed = _PageTags()
    parsed.feed(page)
    assert not any(tag in {"script", "iframe", "form", "base"} for tag, _ in parsed.tags)
    assert not any(name.startswith("on") for _, attrs in parsed.tags for name in attrs)
    assert all(attrs["src"].startswith("frame-index-") for tag, attrs in parsed.tags if tag == "img")
    assert all(attrs["href"].startswith("#") or attrs["href"] == "review.css"
               for _, attrs in parsed.tags if "href" in attrs)
    assert sum(tag == "img" for tag, _ in parsed.tags) == 2


def test_review_page_rejects_foreign_image_path_and_failed_render_has_no_manifest(observed_vfr, tmp_path, monkeypatch):
    source = observed_vfr[0]
    output = tmp_path / "path-check"
    manifest = pack.build_pack(source, output)
    manifest["events"][0]["current"]["path"] = "../private.png"
    with pytest.raises(ValueError, match="generated frame name"):
        pack._render_review_html(manifest)
    manifest["events"][0]["current"]["path"] = "frame-index-000002.png"
    manifest["decoded_video_role"] = "prepared_playback_copy"
    assert "播放副本 PTS 时间基" in pack._render_review_html(manifest)

    def render_failure(_):
        raise ValueError("render stopped")

    monkeypatch.setattr(pack, "_render_review_html", render_failure)
    failed = tmp_path / "failed-render"
    with pytest.raises(ValueError, match="render stopped"):
        pack.build_pack(source, failed)
    assert (failed / "FAILED.json").is_file()
    assert not (failed / "manifest.json").exists()
    assert not (failed / "index.html").exists()


def test_partial_page_write_cannot_leave_success_manifest(observed_vfr, tmp_path, monkeypatch):
    original = Path.write_bytes

    def incomplete_html(self, data):
        if self.name == "index.html.incomplete":
            original(self, b"incomplete")
            raise OSError("simulated write interruption")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", incomplete_html)
    output = tmp_path / "interrupted-page"
    with pytest.raises(OSError, match="simulated write interruption"):
        pack.build_pack(observed_vfr[0], output)
    assert (output / "FAILED.json").is_file()
    assert not (output / "manifest.json").exists()
    assert not (output / "index.html").exists()
    assert not (output / "review.css").exists()
    assert (output / "index.html.incomplete").exists()


def test_manifest_promotion_failure_demotes_page_and_stylesheet(observed_vfr, tmp_path, monkeypatch):
    original = Path.rename

    def fail_manifest_promotion(self, target):
        if self.name == "manifest.json.incomplete":
            raise OSError("simulated manifest promotion failure")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", fail_manifest_promotion)
    output = tmp_path / "failed-promotion"
    with pytest.raises(OSError, match="simulated manifest promotion failure"):
        pack.build_pack(observed_vfr[0], output)
    assert (output / "FAILED.json").is_file()
    for name in ("manifest.json", "index.html", "review.css"):
        assert not (output / name).exists()
        assert (output / f"{name}.incomplete").exists()


def test_partial_manifest_write_never_promotes_page(observed_vfr, tmp_path, monkeypatch):
    def incomplete_manifest(_, stream, **_kwargs):
        stream.write('{"partial":')
        raise OSError("simulated manifest write interruption")

    monkeypatch.setattr(pack.json, "dump", incomplete_manifest)
    output = tmp_path / "interrupted-manifest"
    with pytest.raises(OSError, match="simulated manifest write interruption"):
        pack.build_pack(observed_vfr[0], output)
    assert (output / "FAILED.json").is_file()
    for name in ("manifest.json", "index.html", "review.css"):
        assert not (output / name).exists()
        assert (output / f"{name}.incomplete").exists()


def test_offline_review_page_loads_verified_images_in_chromium(observed_vfr, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    source = observed_vfr[0]
    output = tmp_path / "browser-review-pack"
    manifest = pack.build_pack(source, output)
    assert manifest["human_review"] is False
    evidence_dir = os.getenv("TRACE_ANNOTATE_REVIEW_PAGE_EVIDENCE_DIR")
    if evidence_dir:
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            for label, viewport in (("desktop", {"width": 1366, "height": 768}),
                                    ("mobile", {"width": 390, "height": 844})):
                page = browser.new_page(viewport=viewport)
                requests, errors = [], []
                page.on("request", lambda request: requests.append(request.url))
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto((output / "index.html").as_uri(), wait_until="load")
                if evidence_dir:
                    page.screenshot(path=str(Path(evidence_dir) / f"review-page-{label}.png"))
                images = page.locator("article.event .frames img")
                assert images.count() == 2
                assert images.evaluate_all(
                    "nodes=>nodes.every(img=>img.complete && img.naturalWidth>0 && img.naturalHeight>0)")
                assert page.locator(".frames").evaluate("node=>getComputedStyle(node).display") == "grid"
                assert "未人工复核" in page.locator(".warning").inner_text()
                assert page.locator("script").count() == 0
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                if label == "desktop":
                    assert images.evaluate_all(
                        "nodes=>nodes.every(img=>img.getBoundingClientRect().top < window.innerHeight)")
                assert requests and all(url.startswith("file:") for url in requests), requests
                assert not errors, errors
                page.close()
        finally:
            browser.close()
