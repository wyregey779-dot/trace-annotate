"""Prepare a local, independently verified browser playback copy.

This is a lossy compatibility conversion, not annotation or color-fidelity
certification. Frames are paired using decoded PTS, never nominal frame rate.
Only the new project directory is published; the source is never rewritten.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Iterator, Optional

from .media import stream_sha256
from .schema import require_valid_document

_MAX_PACKET_VERIFICATION_FRAMES = 2_000_000

class MediaPreparationError(ValueError):
    """The source cannot be prepared without breaking the declared contract."""


class _MissingFinalFrameDuration(MediaPreparationError):
    """Decoded frame metadata may need an explicit demux-packet duration."""


def _cancel(cancelled: Optional[Callable[[], bool]]) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("Media preparation cancelled")


def _tool(name: str) -> str:
    result = shutil.which(name)
    if not result:
        raise MediaPreparationError(f"{name} is required on PATH for video preparation")
    return result


def _run(command: list, directory: Path, name: str,
         cancelled: Optional[Callable[[], bool]], *, strict: bool = True) -> Path:
    """Capture arbitrarily long tool output on disk, polling for cancellation."""
    _cancel(cancelled)
    stdout, stderr = directory / f"{name}.stdout", directory / f"{name}.stderr"
    started = time.monotonic()
    was_cancelled = False
    with stdout.open("xb") as output, stderr.open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors)
        try:
            while process.poll() is None:
                _cancel(cancelled)
                time.sleep(0.05)
            _cancel(cancelled)
        except BaseException:
            was_cancelled = True
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            output.flush()
            errors.flush()
            with (directory / "commands.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"name": name, "args": command, "exit_code": process.returncode,
                                      "elapsed_seconds": time.monotonic() - started,
                                      "cancelled_or_interrupted": was_cancelled,
                                      "stdout_file": stdout.name, "stderr_file": stderr.name}) + "\n")
    # Every caller uses -v error; a decoder reporting an error while returning
    # zero must still fail. Read only a bounded diagnostic tail into memory.
    if process.returncode or (strict and stderr.stat().st_size):
        with stderr.open("rb") as stream:
            stream.seek(max(0, stderr.stat().st_size - 4096))
            detail = stream.read().decode("utf-8", errors="replace").strip()
        raise MediaPreparationError(f"{name} failed (exit {process.returncode}): {detail}")
    return stdout


def _probe(ffprobe: str, media: Path, work: Path, name: str,
           cancelled: Optional[Callable[[], bool]]) -> Dict[str, Any]:
    result = _run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_streams", "-show_format",
                   "-of", "json", str(media)], work, name, cancelled)
    try:
        value = json.loads(result.read_text(encoding="utf-8"))
        if not isinstance(value.get("streams"), list):
            raise ValueError
        return value
    except (ValueError, AttributeError) as exc:
        raise MediaPreparationError(f"{name}: invalid ffprobe stream metadata") from exc


def _time_base(stream: Dict[str, Any]) -> Fraction:
    try:
        result = Fraction(stream["time_base"])
        if result <= 0 or result.denominator > 2147483647:
            raise ValueError
        return result
    except (ValueError, KeyError, ZeroDivisionError, TypeError) as exc:
        raise MediaPreparationError("Video has an unsupported or missing time base") from exc


def _frames(ffprobe: str, media: Path, index: int, work: Path, name: str,
            cancelled: Optional[Callable[[], bool]]) -> Path:
    return _run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-err_detect", "explode",
                 "-select_streams", str(index), "-show_frames", "-show_entries",
                 "frame=pts,best_effort_timestamp,duration,pkt_duration,nb_samples",
                 "-of", "compact=p=0:nk=0", str(media)], work, name, cancelled)


def _packets(ffprobe: str, media: Path, index: int, work: Path, name: str,
             cancelled: Optional[Callable[[], bool]]) -> Path:
    return _run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
                 "-select_streams", str(index), "-show_packets", "-show_entries",
                 "packet=pts,duration", "-of", "compact=p=0:nk=0", str(media)],
                work, name, cancelled)


def _frame_rows(path: Path, *, allow_missing_pts: bool = False) -> Iterator[Dict[str, int]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = {}
            for part in line.strip().split("|"):
                key, separator, value = part.partition("=")
                if separator and key in {"pts", "best_effort_timestamp", "duration", "pkt_duration", "nb_samples"}:
                    try:
                        fields[key] = int(value)
                    except ValueError:
                        pass
            # Side-data-only rows are permitted. Actual frames must carry PTS;
            # best-effort timestamps alone would hide missing source timestamps.
            if not fields:
                continue
            if "pts" not in fields and not allow_missing_pts:
                raise MediaPreparationError("Decoded frame has no original presentation timestamp")
            yield fields


def _video_summary(path: Path, base: Fraction,
                   cancelled: Optional[Callable[[], bool]], *,
                   packets: Optional[Path] = None) -> Dict[str, Any]:
    count = 0
    first = previous = last_duration = first_delta = None
    variable = False
    frame_pts = set() if packets is not None else None
    for row in _frame_rows(path):
        _cancel(cancelled)
        pts = row["pts"]
        if previous is not None:
            delta = pts - previous
            if delta <= 0:
                raise MediaPreparationError("Video PTS must be strictly increasing; duplicate or reversed frames cannot be mapped")
            if first_delta is None:
                first_delta = delta
            elif delta != first_delta:
                variable = True
        else:
            first = pts
        previous = pts
        durations = [row.get(key, 0) for key in ("duration", "pkt_duration")]
        if any(value < 0 for value in durations):
            raise MediaPreparationError("Decoded video frame duration cannot be negative")
        if all(value > 0 for value in durations) and durations[0] != durations[1]:
            raise MediaPreparationError("Decoded frame and attached packet duration metadata conflict")
        # A present zero duration is not usable, but older ffprobe versions
        # may still expose the positive packet duration on the decoded frame.
        last_duration = row.get("duration", 0) or row.get("pkt_duration", 0)
        duration_basis = ("explicit_decoded_frame_duration" if row.get("duration", 0) > 0
                          else "explicit_decoded_frame_attached_packet_duration")
        if frame_pts is not None:
            if count >= _MAX_PACKET_VERIFICATION_FRAMES:
                raise MediaPreparationError("Packet duration verification exceeds the 2000000-frame resource limit")
            frame_pts.add(pts)
        count += 1
    if count == 0:
        raise MediaPreparationError("No decodable video frames were found")
    if last_duration is None or last_duration <= 0:
        if packets is None:
            # A last-frame hold cannot be inferred from average FPS or the
            # prior frame interval, particularly for VFR.
            raise _MissingFinalFrameDuration("Final video frame duration is missing; an exact playable endpoint cannot be established")
        # FFmpeg 6 may omit a decoded H.264 tail duration even though the
        # demuxer reports it explicitly. Accept it only after proving a unique
        # one-packet-per-displayed-frame PTS mapping for the selected stream.
        # Reordered packet DTS is irrelevant; no FPS/container estimate is used.
        packet_pts = set()
        for row in _frame_rows(packets):
            _cancel(cancelled)
            pts = row["pts"]
            if row.get("duration", 0) < 0:
                raise MediaPreparationError("Demux packet duration cannot be negative")
            if pts in packet_pts or len(packet_pts) >= count:
                raise MediaPreparationError("Explicit packet duration requires one unique packet PTS per decoded frame")
            packet_pts.add(pts)
            if pts == previous:
                last_duration = row.get("duration", 0)
        if packet_pts != frame_pts:
            raise MediaPreparationError("Packet-to-frame PTS mapping is ambiguous; the final hold cannot be verified")
        if not last_duration or last_duration <= 0:
            raise MediaPreparationError(
                "FINAL_FRAME_DURATION_UNAVAILABLE: decoded frames and matching packets expose no explicit final duration. "
                "Use a matching newer FFmpeg/ffprobe toolchain (9.0.1 is locally verified) and retry; "
                "if it still lacks timing metadata, obtain a source with reliable timing. No FPS or container-duration estimate is used.")
        duration_basis = "explicit_demux_packet_duration_with_one_to_one_decoded_pts"
    end = (previous - first + last_duration) * base
    if end <= 0 or not math.isfinite(float(end)):
        raise MediaPreparationError("Video timeline is empty or invalid")
    return {"count": count, "first": first, "last": previous,
            "last_duration": last_duration, "duration": end, "variable": variable,
            "last_duration_basis": duration_basis}


def _verified_video_summary(ffprobe: str, media: Path, index: int, frames: Path,
                            base: Fraction, work: Path, name: str,
                            cancelled: Optional[Callable[[], bool]]) -> Dict[str, Any]:
    try:
        return _video_summary(frames, base, cancelled)
    except _MissingFinalFrameDuration:
        packets = _packets(ffprobe, media, index, work, name, cancelled)
        return _video_summary(frames, base, cancelled, packets=packets)


def _audio_summary(path: Path, base: Fraction, sample_rate: int,
                   cancelled: Optional[Callable[[], bool]], *,
                   allow_trailing_flush_inference: bool = False) -> Dict[str, Any]:
    first = previous = end = None
    samples = 0
    count = 0
    inferred = []
    for row in _frame_rows(path, allow_missing_pts=allow_trailing_flush_inference):
        _cancel(cancelled)
        size = row.get("nb_samples", 0)
        if size <= 0:
            raise MediaPreparationError("Decoded audio frame has no sample count")
        if "pts" not in row:
            if not allow_trailing_flush_inference or end is None or inferred:
                raise MediaPreparationError("Only one known-codec trailing audio flush frame may have an inferred timestamp")
            current = end
            inferred.append({"frame": count, "start": float(current),
                             "end": float(current + Fraction(size, sample_rate)), "samples": size})
        else:
            if inferred:
                raise MediaPreparationError("Audio timestamp is missing before the final flush frame; no middle timestamps may be inferred")
            current = row["pts"] * base
        if previous is not None and current <= previous:
            raise MediaPreparationError("Audio PTS must be strictly increasing")
        # Matroska/WebM timestamps are often quantized to milliseconds even
        # when decoded audio samples have submillisecond duration. One tick is
        # permitted, while real missing sections still fail explicitly.
        if end is not None and abs(current - end) > max(base, Fraction(2, sample_rate)):
            raise MediaPreparationError("Discontinuous audio timestamps require a separate synchronization workflow")
        if first is None:
            first = current
        end = current + Fraction(size, sample_rate)
        previous = current
        samples += size
        count += 1
    if not count:
        raise MediaPreparationError("An advertised audio stream has no decodable samples")
    return {"start": first, "end": end, "frames": count, "samples": samples,
            "timestamp_basis": "original_pts_with_one_trailing_decoder_flush_inferred_from_samples" if inferred else "original_pts",
            "inferred_tail_frames": inferred}


def _pair_frames(source: Path, proxy: Path, source_base: Fraction,
                 proxy_base: Fraction, origin: Fraction, mapping: Path,
                 cancelled: Optional[Callable[[], bool]]) -> int:
    count = 0
    rows = _frame_rows(proxy)
    with mapping.open("x", encoding="utf-8", newline="\n") as output:
        for count, original in enumerate(_frame_rows(source), 1):
            _cancel(cancelled)
            converted = next(rows, None)
            if converted is None:
                raise MediaPreparationError("Playback conversion lost video frames")
            expected = original["pts"] * source_base - origin
            actual = converted["pts"] * proxy_base
            # MP4 time-base quantization may differ by a single output tick;
            # it must never accumulate drift or silently turn VFR into CFR.
            if abs(expected - actual) > min(source_base, proxy_base):
                raise MediaPreparationError(f"Playback frame {count - 1} timestamp changed during conversion")
            output.write(json.dumps({"frame": count - 1, "source_pts": original["pts"],
                                     "proxy_pts": converted["pts"],
                                     "source_time": float(original["pts"] * source_base),
                                     "proxy_time": float(actual)}, allow_nan=False) + "\n")
        if next(rows, None) is not None:
            raise MediaPreparationError("Playback conversion introduced extra video frames")
    return count


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _publish_new(project: Path, target: Path,
                 cancelled: Optional[Callable[[], bool]]) -> None:
    """Reserve a new directory exclusively, publishing the ready manifest last."""
    _cancel(cancelled)
    target.mkdir()  # refuses existing directories, files and symlinks
    identity = target.stat().st_dev, target.stat().st_ino
    created = []
    try:
        names = sorted(item.name for item in project.iterdir() if item.name != "project.json")
        for name in names + ["project.json"]:
            _cancel(cancelled)
            destination = target / name
            # Hard-link publication cannot replace an existing file. Both
            # paths are in the same parent filesystem; source links are removed
            # only after the destination identity is recorded.
            os.link(project / name, destination)
            info = destination.stat()
            created.append((destination, (info.st_dev, info.st_ino)))
        _cancel(cancelled)
    except BaseException:
        for destination, expected in reversed(created):
            try:
                info = destination.lstat()
                if (info.st_dev, info.st_ino) == expected:
                    destination.unlink()
            except FileNotFoundError:
                pass
        try:
            info = target.lstat()
            if (info.st_dev, info.st_ino) == identity:
                target.rmdir()  # refuses deletion if anyone added other files
        except OSError:
            pass
        raise


def prepare_media(source: os.PathLike, output_dir: os.PathLike,
                  cancelled: Optional[Callable[[], bool]] = None, *,
                  tone_map_hdr: bool = False) -> Dict[str, Any]:
    """Return a newly created, fully decoded compatibility project's manifest.

    Requires FFmpeg/ffprobe with libx264 and AAC encoders and a filesystem with
    hard links. A successful result preserves decoded frame count and PTS order;
    visual pixels and audio samples are lossy. HDR requires the explicit
    tone_map_hdr opt-in and is converted to SDR using Hable tone mapping;
    this does not preserve HDR appearance or dynamic metadata. Unsupported
    discontinuous timelines fail explicitly. No semantic annotation is implied.
    """
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be a callable or None")
    if not isinstance(tone_map_hdr, bool):
        raise ValueError("tone_map_hdr must be a boolean")
    _cancel(cancelled)
    source = Path(source).resolve(strict=True)
    target = Path(output_dir).absolute()
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {target.parent}")
    ffmpeg, ffprobe = _tool("ffmpeg"), _tool("ffprobe")
    original_hash = stream_sha256(source, cancelled=cancelled)
    original_size = source.stat().st_size
    # Reject common multi-file manifests before a demuxer can follow even local
    # references. The protocol whitelist independently blocks network protocols.
    with source.open("rb") as stream:
        prefix = stream.read(4096).lstrip(b"\xef\xbb\xbf \r\n\t")
    if (source.suffix.lower() in {".m3u", ".m3u8", ".mpd", ".ffconcat"}
            or prefix.startswith((b"#EXTM3U", b"ffconcat version"))
            or b"<MPD" in prefix or b"urn:mpeg:dash:schema:mpd" in prefix):
        raise MediaPreparationError("Playlists and multi-file/network manifests are not self-contained media; prepare a local merged video first")
    with tempfile.TemporaryDirectory(prefix=f".{target.name[:40]}-prepare-", dir=target.parent) as temporary:
        work = Path(temporary)
        project = work / "project"
        project.mkdir()
        versions = {}
        for tool_name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
            version_output = _run([executable, "-version"], work, f"{tool_name}-version", cancelled, strict=False)
            with version_output.open(encoding="utf-8", errors="replace") as stream:
                versions[tool_name] = stream.readline().strip()
        info = _probe(ffprobe, source, work, "source-probe", cancelled)
        formats = set(info.get("format", {}).get("format_name", "").split(","))
        if formats & {"hls", "dash", "concat", "image2", "image2pipe", "sdp", "rtsp", "sap", "smoothstreaming"}:
            raise MediaPreparationError("Multi-file, sequence and network demuxers are not supported as a single hash-bound video")
        videos = [item for item in info["streams"] if item.get("codec_type") == "video"
                  and not item.get("disposition", {}).get("attached_pic")]
        if not videos:
            raise MediaPreparationError("Source has no video stream; audio-only and cover-art files are not videos")
        video = next((item for item in videos if item.get("disposition", {}).get("default")), videos[0])
        is_hdr = video.get("color_transfer") in {"smpte2084", "arib-std-b67"}
        if is_hdr and not tone_map_hdr:
            raise MediaPreparationError("HDR/PQ/HLG needs a verified tone-mapping workflow; no silent SDR conversion is allowed")
        hdr_filter = None
        if is_hdr:
            if video.get("color_primaries") != "bt2020" or video.get("color_space") not in {"bt2020nc", "bt2020c"}:
                raise MediaPreparationError("HDR tone mapping requires explicit supported BT.2020 primaries and matrix metadata")
            # This policy uses the backwards-compatible HDR base layer. Dolby
            # Vision dynamic RPU / enhancement layers are not applied or kept.
            available = _run([ffmpeg, "-hide_banner", "-v", "error", "-filters"], work, "hdr-filters", cancelled, strict=False)
            with available.open(encoding="utf-8", errors="replace") as stream:
                names = {parts[1] for line in stream if len(parts := line.split()) >= 2}
            if not {"zscale", "tonemap", "sidedata"}.issubset(names):
                raise MediaPreparationError("Explicit HDR conversion requires FFmpeg zscale, tonemap and sidedata filters")
            hdr_filter = "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:r=limited,format=yuv420p,sidedata=mode=delete"
        base = _time_base(video)
        source_frames = _frames(ffprobe, source, video["index"], work, "source-video-frames", cancelled)
        source_summary = _verified_video_summary(ffprobe, source, video["index"], source_frames,
                                                 base, work, "source-video-packets", cancelled)
        first = source_summary["first"]
        first_time = first * base
        audio = [item for item in info["streams"] if item.get("codec_type") == "audio"]
        source_audio = []
        for index, stream in enumerate(audio):
            try:
                rate = int(stream["sample_rate"])
                if rate <= 0:
                    raise ValueError
            except (KeyError, ValueError, TypeError) as exc:
                raise MediaPreparationError("Audio sample rate is invalid") from exc
            rows = _frames(ffprobe, source, stream["index"], work, f"source-audio-{index}", cancelled)
            summary = _audio_summary(rows, _time_base(stream), rate, cancelled,
                                     allow_trailing_flush_inference=stream.get("codec_name") in {"wmav1", "wmav2"})
            source_audio.append((stream, summary, rate))
        # A shared origin preserves audio that begins before the first video
        # frame (for example WebM Opus at 0 with video at 7 ms). Forcing the
        # video itself to zero would silently trim that earlier audio.
        origin = min([first_time] + [summary["start"] for _, summary, _ in source_audio])
        timescale = base.denominator * origin.denominator // math.gcd(base.denominator, origin.denominator)
        if timescale > 2147483647:
            raise MediaPreparationError("Combined audio/video timeline needs an unsupported MP4 timescale")
        encoder_base = Fraction(1, timescale)
        playback = project / "playback.mp4"
        command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", "-xerror",
                   "-protocol_whitelist", "file,pipe", "-err_detect", "explode", "-copyts", "-i", str(source),
                   "-map", f"0:{video['index']}"]
        for stream in audio:
            command += ["-map", f"0:{stream['index']}"]
        # Padding leaves all odd-sized source pixels present instead of cropping.
        last_pts = int((source_summary["last"] * base - origin) / encoder_base)
        last_duration = int(source_summary["last_duration"] * base / encoder_base)
        # With reordered B frames MP4's final decode-order packet can differ
        # from its final displayed frame. Disable reordering and carry the
        # source's explicit last-frame hold, rather than guessing it from FPS.
        final_hold = f"setts=pts=PTS:dts=DTS:duration=if(eq(PTS\\,{last_pts})\\,{last_duration}\\,DURATION)"
        video_filter = f"settb=1/{timescale},setpts=PTS-({origin.numerator}/{origin.denominator})/TB,pad=ceil(iw/2)*2:ceil(ih/2)*2"
        full_range = video.get("color_range") == "pc" or video.get("pix_fmt", "").startswith("yuvj")
        if hdr_filter:
            video_filter += "," + hdr_filter
        elif full_range:
            # MJPEG commonly decodes to yuvj420p (full range). A pixel-value
            # conversion is essential; merely changing the tag would crush
            # shadows/highlights or leave yuvj420p outside the declared format.
            video_filter += ",scale=in_range=full:out_range=limited,format=yuv420p,setparams=range=limited"
        command += ["-vf", video_filter,
                    "-c:v", "libx264", "-bf", "0", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-fps_mode:v", "passthrough", "-enc_time_base:v", str(encoder_base),
                    "-video_track_timescale", str(timescale), "-bsf:v", final_hold]
        if hdr_filter:
            command += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
        elif full_range:
            command += ["-color_range", "tv"]
        for index, stream in enumerate(audio):
            # Source frames have already proved contiguous within their time
            # base's quantization. Use the decoded sample clock, preserving all
            # samples and the verified first-audio offset. This removes source
            # millisecond quantization jitter without resampling or filling gaps
            # and gives a declared timestamp to a permitted WMA flush frame.
            audio_offset = source_audio[index][1]["start"] - origin
            command += [f"-filter:a:{index}", f"asetpts=N/SR/TB+({audio_offset.numerator}/{audio_offset.denominator})/TB",
                        f"-c:a:{index}", "aac", f"-b:a:{index}", "192k"]
        command += ["-map_metadata", "-1", "-map_chapters", "-1", "-avoid_negative_ts", "disabled",
                    "-movflags", "+faststart", str(playback)]
        _run(command, work, "transcode", cancelled)
        # Decode the final multiplexed file independently of its encoder.
        _run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode",
              "-protocol_whitelist", "file,pipe", "-i", str(playback), "-map", "0:v", "-map", "0:a?", "-fps_mode:v", "passthrough",
              "-enc_time_base:v", "demux", "-f", "null", "-"],
             work, "playback-full-decode", cancelled)
        converted = _probe(ffprobe, playback, work, "playback-probe", cancelled)
        converted_video = [item for item in converted["streams"] if item.get("codec_type") == "video"]
        converted_audio = [item for item in converted["streams"] if item.get("codec_type") == "audio"]
        if len(converted_video) != 1 or len(converted_audio) != len(audio):
            raise MediaPreparationError("Playback stream count differs; no audio track may disappear")
        proxy_video = converted_video[0]
        if proxy_video.get("codec_name") != "h264" or proxy_video.get("pix_fmt") != "yuv420p":
            raise MediaPreparationError("Playback is not the declared H264/yuv420p compatibility format")
        if hdr_filter and any(proxy_video.get(key) != value for key, value in (
                ("color_transfer", "bt709"), ("color_primaries", "bt709"), ("color_space", "bt709"))):
            raise MediaPreparationError("HDR conversion did not produce the declared BT.709 SDR metadata")
        proxy_base = _time_base(proxy_video)
        proxy_frames = _frames(ffprobe, playback, proxy_video["index"], work, "playback-video-frames", cancelled)
        proxy_summary = _verified_video_summary(ffprobe, playback, proxy_video["index"], proxy_frames,
                                                proxy_base, work, "playback-video-packets", cancelled)
        expected_first = first_time - origin
        if abs(proxy_summary["first"] * proxy_base - expected_first) > proxy_base:
            raise MediaPreparationError("Playback video start does not preserve its offset from the common timeline origin")
        mapping = project / "frame-map.jsonl"
        paired = _pair_frames(source_frames, proxy_frames, base, proxy_base, origin, mapping, cancelled)
        if abs(source_summary["duration"] - proxy_summary["duration"]) > max(base, proxy_base):
            raise MediaPreparationError("Playback final frame duration changed")
        checks = []
        for index, ((original, original_summary, rate), changed) in enumerate(zip(source_audio, converted_audio)):
            if changed.get("codec_name") != "aac" or int(changed.get("channels", 0)) != int(original.get("channels", 0)):
                raise MediaPreparationError("Playback audio codec or channel count does not match the declared conversion")
            changed_rate = int(changed["sample_rate"])
            rows = _frames(ffprobe, playback, changed["index"], work, f"playback-audio-{index}", cancelled)
            audio_summary = _audio_summary(rows, _time_base(changed), changed_rate, cancelled)
            tolerance = Fraction(2048, changed_rate) + Fraction(2, 1000)
            expected_start = original_summary["start"] - origin
            expected_end = original_summary["end"] - origin
            if abs(audio_summary["start"] - expected_start) > tolerance or abs(audio_summary["end"] - expected_end) > tolerance:
                raise MediaPreparationError(f"Audio track {index} synchronization changed during conversion")
            checks.append({"source_stream": original["index"], "playback_stream": changed["index"],
                           "source_start": float(original_summary["start"]), "source_end": float(original_summary["end"]),
                           "expected_start": float(expected_start), "expected_end": float(expected_end),
                           "playback_start": float(audio_summary["start"]), "playback_end": float(audio_summary["end"]),
                           "tolerance_seconds": float(tolerance), "channels": changed["channels"],
                           "source_decoded_samples": original_summary["samples"], "playback_decoded_samples": audio_summary["samples"],
                           "source_timestamp_basis": original_summary["timestamp_basis"],
                           "source_inferred_tail_frames": original_summary["inferred_tail_frames"],
                           "playback_timestamp_policy": "decoded_sample_clock_from_verified_source_start; source-tick quantization removed; samples not resampled",
                           "language": original.get("tags", {}).get("language")})
        # The HTML media element uses container duration, including a longer
        # audio tail. Keep that timeline, while mapping only actual video frames.
        try:
            duration = float(converted["format"]["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError
        except (KeyError, ValueError, TypeError) as exc:
            raise MediaPreparationError("Prepared playback has no finite positive duration") from exc
        playback_hash = stream_sha256(playback, cancelled=cancelled)
        source_identity = {"name": source.name, "size": original_size, "sha256": original_hash}
        timeline = {"frame_count": paired, "first_video_pts": first, "first_video_time": float(first_time),
                    "origin_time": float(origin), "origin_time_fraction": str(origin),
                    "playback_first_video_time": float(proxy_summary["first"] * proxy_base),
                    "source_time_base": str(base), "proxy_time_base": str(proxy_base),
                    "mapping": "proxy_time = source_time - origin_time",
                    "video_duration": float(proxy_summary["duration"]), "duration": duration,
                    "last_frame_duration": {"source_ticks": source_summary["last_duration"],
                                            "source_basis": source_summary["last_duration_basis"],
                                            "playback_ticks": proxy_summary["last_duration"],
                                            "playback_basis": proxy_summary["last_duration_basis"]},
                    "variable_frame_rate": source_summary["variable"],
                    "preservation": "decoded_frame_order_and_timestamps_verified",
                    "audio": {"policy": "all_audio_streams_aac", "stream_count": len(audio),
                              "offset_policy": "source_audio_pts_minus_origin_time",
                              "verification": "decoded_start_end_and_contiguous_samples_within_tolerance",
                              "tracks": checks}}
        selected = {video["index"]} | {item["index"] for item in audio}
        manifest = {"kind": "traceannotate.prepared-media", "version": 1,
                    "video": {"file": "playback.mp4", "size": playback.stat().st_size,
                              "sha256": playback_hash, "duration": duration},
                    "annotation_file": "annotations.json", "source": source_identity,
                    "timeline": timeline, "mapping_file": "frame-map.jsonl", "preparation_log": "preparation-log.json",
                    "conversion": {"video_codec": "h264", "pixel_format": "yuv420p", "lossy": True,
                                   "source_video_stream": video["index"], "source_pixel_format": video.get("pix_fmt"),
                                   "source_color_transfer": video.get("color_transfer"),
                                   "source_color_range": video.get("color_range"), "playback_color_range": proxy_video.get("color_range"),
                                   "range_conversion": "HDR linear-light tone mapping to limited SDR" if hdr_filter else
                                                       "full_to_limited_pixel_scaling" if full_range else "decoder/encoder default; no full-range retagging",
                                   "hdr_tone_mapping": {"requested": tone_map_hdr, "applied": bool(hdr_filter),
                                                        "algorithm": "linear-light Hable to BT.709 SDR" if hdr_filter else None,
                                                        "nominal_peak_luminance": 100 if hdr_filter else None,
                                                        "filter": hdr_filter,
                                                        "dynamic_hdr_metadata": "not applied or preserved; HDR base layer only" if hdr_filter else None},
                                   "source_size": [video.get("width"), video.get("height")],
                                   "playback_size": [proxy_video.get("width"), proxy_video.get("height")],
                                   "rotation": "ffmpeg_autorotate_applied", "odd_size_policy": "pad_right_or_bottom_to_even",
                                   "omitted_streams": [{"index": item["index"], "type": item.get("codec_type"),
                                                        "codec": item.get("codec_name")} for item in info["streams"] if item["index"] not in selected],
                                   "limitations": ["Lossy SDR 8-bit compatibility copy; original pixels and audio samples are not preserved.",
                                                   "Frame PTS correspondence does not certify browser frame-accurate seeking or semantic labels.",
                                                   "All audio tracks are retained; browser audio-track selection depends on browser support."]}}
        document = {"kind": "temporal", "schema_version": 2, "clip": "playback.mp4", "duration": duration,
                    "media_size": manifest["video"]["size"], "media_sha256": playback_hash, "segments": [],
                    "metadata": {"prepared_media": {"project_file": "project.json", "source": source_identity,
                                                    "timeline": timeline, "mapping_file": "frame-map.jsonl"}}}
        require_valid_document(document)
        _write_json(project / "annotations.json", document)
        _write_json(project / "project.json", manifest)
        final_source_hash = stream_sha256(source, cancelled=cancelled)
        if final_source_hash != original_hash or source.stat().st_size != original_size:
            raise MediaPreparationError("Source changed during preparation; no project was published")
        command_log = []
        with (work / "commands.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                for channel in ("stdout", "stderr"):
                    raw = work / entry[f"{channel}_file"]
                    entry[f"{channel}_size"] = raw.stat().st_size
                    entry[f"{channel}_sha256"] = stream_sha256(raw, cancelled=cancelled)
                # Decoder logs are empty on success; preserve any version
                # diagnostics without reading long frame lists into memory.
                errors = work / entry["stderr_file"]
                with errors.open("rb") as diagnostic:
                    entry["stderr_tail"] = diagnostic.read(4096).decode("utf-8", errors="replace")
                command_log.append(entry)
        _write_json(project / "preparation-log.json", {
            "kind": "traceannotate.preparation-log", "version": 1, "tools": versions,
            "source_before": source_identity, "source_after": {"size": source.stat().st_size, "sha256": final_source_hash},
            "commands": command_log, "source_probe": info, "playback_probe": converted,
            "frame_mapping_sha256": stream_sha256(mapping, cancelled=cancelled),
            "raw_tool_output_policy": "Temporary raw output is removed; hashes, stream probes, command exits and the full frame map are retained."})
        _publish_new(project, target, cancelled)
        return manifest
