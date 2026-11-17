"""Extract exact boundary frames for reviewing existing reference candidates.

The pack is read-only evidence. It does not decide transition types, certify
ground truth, or turn an assistant's observations into human review.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import build_reference_truth as R


MAX_EVENTS = 128
MAX_FRAMES = 200_000
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MIN_FREE_RESERVE = 2 * 1024 * 1024 * 1024
SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b")


def _select_expression(indices: list[int]) -> str:
    ranges = []
    first = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((first, previous))
            first = index
        previous = index
    ranges.append((first, previous))
    terms = [f"eq(n\\,{start})" if start == end else f"between(n\\,{start}\\,{end})"
             for start, end in ranges]
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i+1]})" if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return "select=" + terms[0]


def _run(command: list[str], stdout: Path, stderr: Path, timeout: int,
         max_stdout: int = 32 * 1024 * 1024) -> None:
    started = time.monotonic()
    with stdout.open("xb") as output, stderr.open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors)
        try:
            while process.poll() is None:
                if stdout.stat().st_size > max_stdout:
                    raise ValueError("frame probe output exceeds the bounded evidence size")
                if time.monotonic() - started > timeout:
                    raise ValueError(f"decoder exceeded {timeout}s: {command[0]}")
                time.sleep(0.05)
            if stdout.stat().st_size > max_stdout:
                raise ValueError("frame probe output exceeds the bounded evidence size")
            if process.returncode:
                raise ValueError(f"decoder failed with exit {process.returncode}: {command[0]}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def _run_images(command: list[str], work: Path, timeout: int) -> None:
    """Stop the owned decoder before its PNGs exhaust the output budget."""
    started = time.monotonic()
    with (work / "ffmpeg.stdout").open("xb") as output, (work / "ffmpeg.stderr").open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors)
        try:
            while process.poll() is None:
                image_bytes = sum(path.stat().st_size for path in work.glob("decoded-*.png"))
                if image_bytes > MAX_IMAGE_BYTES or shutil.disk_usage(work).free < MIN_FREE_RESERVE:
                    raise ValueError("review images exceeded the byte budget or 2 GiB disk reserve")
                if time.monotonic() - started > timeout:
                    raise ValueError(f"decoder exceeded {timeout}s: {command[0]}")
                time.sleep(0.05)
            if process.returncode:
                raise ValueError(f"decoder failed with exit {process.returncode}: {command[0]}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def _actual_frame_pts(video: Path, rows: list[dict], work: Path, timeout: int,
                      selected_count: int) -> tuple[list[int], list[list[str]], dict]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValueError("ffprobe is required")
    streams = [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
               "-show_streams", "-of", "json", str(video)]
    _run(streams, work / "ffprobe-streams.json", work / "ffprobe-streams.stderr", timeout)
    document = json.loads((work / "ffprobe-streams.json").read_text(encoding="utf-8"))
    video_streams = [entry for entry in document.get("streams", [])
                     if entry.get("codec_type") == "video"]
    if len(video_streams) != 1 or R.rational(video_streams[0]["time_base"]) != R.rational(rows[0]["time_base"]):
        raise ValueError("actual video stream or time base disagrees with the supplied frame map")
    width = R.integer(video_streams[0]["width"])
    height = R.integer(video_streams[0]["height"])
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise ValueError("invalid actual video dimensions")
    estimated_bytes = selected_count * width * height * 4
    if estimated_bytes > MAX_IMAGE_BYTES:
        raise ValueError("selected frames exceed the uncompressed image budget; review fewer candidates")
    frames = [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
              "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=pts",
              "-of", "json", str(video)]
    output = work / "ffprobe-frame-pts.json"
    _run(frames, output, work / "ffprobe-frames.stderr", timeout)
    if output.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("decoded frame probe exceeds the bounded evidence size")
    frame_entries = json.loads(output.read_text(encoding="utf-8")).get("frames", [])
    if len(frame_entries) > MAX_FRAMES:
        raise ValueError("decoded frame count exceeds review-pack limit")
    pts = []
    for entry in frame_entries:
        value = entry.get("pts")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("actual decoded frame has no integer PTS")
        pts.append(value)
    if len(pts) != len(rows) or any(value != row["pts"] for value, row in zip(pts, rows)):
        raise ValueError("actual decoded frame PTS or count disagrees with supplied frame map")
    return pts, [streams, frames], {"width": width, "height": height,
                                   "estimated_uncompressed_bytes": estimated_bytes}


def build_pack(reference_path: Path, output: Path, *, timeout: int = 300) -> dict:
    """Create a new evidence directory; on error retain it with FAILED.json."""
    reference_path = reference_path.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise ValueError("review-pack output already exists")
    if not output.parent.is_dir():
        raise ValueError("review-pack parent directory does not exist")
    if shutil.disk_usage(output.parent).free < MIN_FREE_RESERVE + MAX_IMAGE_BYTES:
        raise ValueError("insufficient free disk for bounded images and 2 GiB reserve")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_sha256 = R.sha256(reference_path)
    if reference.get("schema") != "trace-annotate.reference-candidates.v1":
        raise ValueError("unsupported reference-candidates schema")
    annotation = Path(reference["annotation"]["path"]).resolve(strict=True)
    video = Path(reference["video"]["path"]).resolve(strict=True)
    frame_map = Path(reference["frame_map"]["path"]).resolve(strict=True)
    if len(reference.get("events", [])) > MAX_EVENTS:
        raise ValueError(f"review pack accepts at most {MAX_EVENTS} candidate events")
    regenerated = R.build_reference(annotation, video, frame_map,
                                    reference["frame_map"]["declared_source_sha256"],
                                    reference["source_id"], reference["split"])
    if regenerated != reference:
        raise ValueError("reference candidates differ from current annotation or frame map")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    _, _, media, _, layout = R.annotation_parts(payload)
    rows = R.load_frame_map(frame_map, media, layout)
    if len(rows) > MAX_FRAMES:
        raise ValueError("decoded frame map exceeds review-pack limit")
    selected = set()
    for event in reference["events"]:
        boundary = event["frame_index"]
        if boundary is not None:
            selected.update(index for index in (boundary - 1, boundary) if index >= 0)
    indices = sorted(selected)
    if not indices:
        raise ValueError("reference has no exact frame boundary to review")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required")
    output.mkdir(exist_ok=False)
    try:
        pts, probe_commands, dimensions = _actual_frame_pts(video, rows, output,
                                                             timeout, len(indices))
        filter_expression = _select_expression(indices) + ",showinfo"
        command = [ffmpeg, "-hide_banner", "-nostdin", "-n", "-xerror",
                   "-err_detect", "explode", "-protocol_whitelist", "file,pipe",
                   "-threads", "1", "-filter_threads", "1", "-copyts", "-i", str(video),
                   "-map", "0:v:0", "-vf", filter_expression,
                   "-fps_mode", "passthrough", "-c:v", "png", "-pix_fmt", "rgb24",
                   "-threads:v", "1", "-frames:v", str(len(indices)),
                   "-f", "image2", str(output / "decoded-%06d.png")]
        _run_images(command, output, timeout)
        emitted = [int(match.group(2)) for line in (output / "ffmpeg.stderr").read_text(
            encoding="utf-8", errors="replace").splitlines()
                   if (match := SHOWINFO.search(line))]
        produced = sorted(output.glob("decoded-*.png"))
        if len(produced) != len(indices) or emitted != [pts[index] for index in indices]:
            raise ValueError("FFmpeg selected frame count or PTS disagrees with actual decoded map")
        images = {}
        total_bytes = 0
        for index, file in zip(indices, produced):
            with file.open("rb") as stream:
                header = stream.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                raise ValueError("decoded output is not PNG")
            size = file.stat().st_size
            total_bytes += size
            if total_bytes > MAX_IMAGE_BYTES:
                raise ValueError("review images exceed the bounded output budget")
            named = output / f"frame-index-{index:06d}.png"
            file.rename(named)
            images[index] = {"path": named.name, "sha256": R.sha256(named),
                             "size_bytes": size, "frame_index": index,
                             "pts": pts[index], "time_base": rows[index]["time_base"],
                             "seconds": rows[index]["seconds"]}
        if (R.sha256(video) != reference["video"]["sha256"]
                or R.sha256(frame_map) != reference["frame_map"]["sha256"]
                or R.sha256(annotation) != reference["annotation"]["sha256"]
                or R.sha256(reference_path) != reference_sha256):
            raise ValueError("source bytes changed while review frames were decoded")
        events = []
        for event in reference["events"]:
            index = event["frame_index"]
            events.append({"event_id": event["event_id"], "annotation_id": event["annotation_id"],
                           "declared_kind": event["kind"], "ineligible_reasons": event["ineligible_reasons"],
                           "source_provenance": event["source_annotation"].get("provenance"),
                           "review_status": "unreviewed", "human_review": False,
                           "before": images.get(index - 1) if index is not None else None,
                           "current": images.get(index) if index is not None else None,
                           "frame_identity_basis": event["frame_identity_basis"]})
        manifest = {"schema": "trace-annotate.reference-review-pack.v1",
                    "reference_sha256": reference_sha256,
                    "video_sha256": reference["video"]["sha256"],
                    "frame_map_sha256": reference["frame_map"]["sha256"],
                    "annotation_sha256": reference["annotation"]["sha256"],
                    "actual_decoded_frames_verified": len(pts),
                    "actual_dimensions": dimensions,
                    "scope": "observed boundary frames only; no transition adjudication or human signoff",
                    "human_review": False, "holdout_sealed": False,
                    "commands": probe_commands + [command], "total_image_bytes": total_bytes,
                    "events": events}
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return manifest
    except Exception as exc:
        (output / "FAILED.json").write_text(json.dumps({"error": type(exc).__name__,
            "reason": str(exc), "scope": "incomplete review pack; not usable as evidence"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.timeout <= 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")
        result = build_pack(args.reference, args.out, timeout=args.timeout)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"review pack failed: {exc}\n")
    print(json.dumps({"out": str(args.out), "events": len(result["events"]),
                      "actual_decoded_frames_verified": result["actual_decoded_frames_verified"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
