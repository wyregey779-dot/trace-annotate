"""Retain synthetic timing fixtures and raw probes for FFmpeg-version diagnosis.

This is a source-checkout diagnostic, not a quality bypass. It leaves preparation
errors unchanged and preserves its own temporary work solely for inspection.
It never reads user media, rewrites existing outputs or uses remote media URLs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; existing paths refused")
    parser.add_argument("--ffmpeg-dir", type=Path, help="Optional directory containing both ffmpeg and ffprobe")
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    if args.ffmpeg_dir:
        os.environ["PATH"] = str(args.ffmpeg_dir.absolute()) + os.pathsep + os.environ.get("PATH", "")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("ffmpeg and ffprobe must both be available")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from trace_annotate.temporal import preparation
    report = {"schema": "traceannotate.media-timing-diagnostic/1", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "source_root": str(root), "preparation_source_sha256": sha256(Path(preparation.__file__)),
              "ffmpeg": ffmpeg, "ffprobe": ffprobe, "fixtures": [], "commands": [], "probe_summaries": [],
              "scope": "Synthetic media and raw metadata diagnosis only; business validation remains fail-closed."}

    def run(command, directory, name):
        started = datetime.now(timezone.utc).isoformat()
        stdout, stderr = directory / (name + ".stdout"), directory / (name + ".stderr")
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.run([str(value) for value in command], stdout=out, stderr=err,
                                     stdin=subprocess.DEVNULL, timeout=120)
        report["commands"].append({"args": [str(value) for value in command], "started_at": started,
                                   "exit_code": process.returncode, "stdout": str(stdout), "stderr": str(stderr)})
        return process.returncode, stdout

    for name, tool in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        run([tool, "-version"], output, name + "-version")

    @contextmanager
    def retained_preparation_work(*, prefix, dir):
        # Each caller directory is owned by this newly created diagnostic run.
        # Production prepare_media still uses its ordinary cleanup everywhere else.
        work = tempfile.mkdtemp(prefix=prefix, dir=dir)
        yield work

    def probe(media, directory, name):
        _, info_path = run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_streams", "-show_format",
                            "-of", "json", media], directory, name + "-streams")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        for stream in info.get("streams", []):
            index = stream.get("index")
            if stream.get("codec_type") != "video":
                continue
            metadata = {}
            for option, entries in (("frames", "frame=pts,best_effort_timestamp,duration,pkt_duration,pkt_pos"),
                                    ("packets", "packet=pts,dts,duration,pos,flags")):
                _, rows_path = run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", index,
                     "-show_" + option, "-show_entries", entries, "-of", "json", media],
                    directory, name + "-video-" + str(index) + "-" + option)
                try:
                    metadata[option] = json.loads(rows_path.read_text(encoding="utf-8")).get(option, [])
                except ValueError:
                    metadata[option] = []
            frames, packets = metadata["frames"], metadata["packets"]
            frame_pts = [row["pts"] for row in frames if isinstance(row.get("pts"), int)]
            packet_pts = [row["pts"] for row in packets if isinstance(row.get("pts"), int)]
            final_pts = max(frame_pts) if frame_pts else None
            report["probe_summaries"].append({
                "media": str(media), "stream_index": index, "time_base": stream.get("time_base"),
                "stream_start_pts": stream.get("start_pts"), "stream_duration_ts": stream.get("duration_ts"),
                "frame_count": len(frames), "packet_count": len(packets),
                "frames_without_pts": len(frames) - len(frame_pts), "packets_without_pts": len(packets) - len(packet_pts),
                "duplicate_frame_pts": len(frame_pts) != len(set(frame_pts)),
                "duplicate_packet_pts": len(packet_pts) != len(set(packet_pts)),
                "pts_sets_match": set(frame_pts) == set(packet_pts),
                "frame_only_pts": sorted(set(frame_pts) - set(packet_pts)), "packet_only_pts": sorted(set(packet_pts) - set(frame_pts)),
                "final_decoded_frame": [row for row in frames if row.get("pts") == final_pts],
                "matching_final_packets": [row for row in packets if row.get("pts") == final_pts],
            })

    vfr = "settb=1/30000,setpts=if(lt(N\\,4)\\,N/(10*TB)\\,(0.4+(N-4)/30)/TB)"
    fixtures = [
        ("vfr", ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.8", "-vf", vfr,
                  "-c:v", "libx264", "-fps_mode", "passthrough", "-enc_time_base", "1/30000"]),
        ("audio-leading", ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=1",
                            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-vf", "setpts=PTS+0.1/TB",
                            "-c:v", "libx264", "-fps_mode", "passthrough", "-c:a", "aac"]),
        ("vfr-audio", ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.8",
                       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.8", "-vf", vfr,
                       "-c:v", "libx264", "-fps_mode", "passthrough", "-enc_time_base", "1/30000", "-c:a", "aac"]),
    ]
    for name, options in fixtures:
        directory = output / name
        directory.mkdir()
        source = directory / "source.mp4"
        entry = {"name": name, "source": str(source)}
        report["fixtures"].append(entry)
        code, _ = run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", *options, source], directory, "generate")
        entry["generation_exit_code"] = code
        if code:
            continue
        entry["source_sha256_before"] = sha256(source)
        probe(source, directory, "source")
        try:
            with patch.object(preparation.tempfile, "TemporaryDirectory", retained_preparation_work):
                manifest = preparation.prepare_media(source, directory / "project")
            entry["preparation"] = "PASS_SCOPED"
            entry["timeline"] = manifest["timeline"]
        except Exception as error:
            entry["preparation"] = "FAILED"
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            (directory / "preparation-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        entry["source_sha256_after"] = sha256(source)
        # A failure before publication can still leave a transcoded file in the
        # retained private work directory; inspect those exact bytes too.
        for index, playback in enumerate(sorted(directory.rglob("playback.mp4"))):
            probe(playback, directory, "playback-" + str(index))
    report["files"] = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)}
                       for path in sorted(output.rglob("*")) if path.is_file()]
    report["status"] = "PASS_SCOPED" if all(entry.get("preparation") == "PASS_SCOPED" for entry in report["fixtures"]) else "FAILED"
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "fixtures": [{"name": entry["name"], "preparation": entry.get("preparation"), "error": entry.get("error")}
                                   for entry in report["fixtures"]]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS_SCOPED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
