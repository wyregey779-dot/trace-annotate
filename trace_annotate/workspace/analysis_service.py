"""Hash-bound objective media analysis; never semantic listening or source gain.

Reports use the professional typed command contract. This service returns data
for a caller to commit under revision/CAS protection and does not mutate a task.
All subprocesses use preparation's owned-child/cancellation runner. Original
paths must come from an explicit local selection, never an imported JSON path.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

from ..temporal import preparation as P
from ..temporal.media import stream_sha256
from . import professional_schema as PS
from . import schema as S
from .errors import Conflict, SchemaInvalid, WorkspaceError
from .store import _safe

VERSION = "1"
MAX_STREAMS = 16
MAX_PCM_BYTES = 512 * 1024 * 1024
MAX_WORK_BYTES = MAX_PCM_BYTES + 128 * 1024 * 1024
RESERVE_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 2_000_000
MAX_SILENCE_INTERVALS = 128
TIMEOUT_SECONDS = 600


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _value(name, value, unit="none", *, basis="measured", reason=None):
    return {"name": name, "value": value, "unit": unit, "basis": basis,
            "status": "available" if value is not None else "unavailable",
            "reason": None if value is not None else (reason or "not_declared")}


def _report(kind, media, domain, index, method, values, window=None, subject=None):
    scope = "interval" if window is not None else ("stream" if index is not None else "file")
    report = {"id": "analysis-placeholder", "kind": kind, "scope": scope,
              "file_sha256": media["playback" if domain == "playback_seconds" else "source"]["sha256"],
              "stream_index": index, "time_domain": domain, "window": window,
              "method": method, "input_digest": "0" * 64,
              "status": "available" if any(v["status"] == "available" for v in values) else "unavailable",
              "values": values, "evidence_ids": [], "subject_annotation_id": subject}
    report["input_digest"] = PS.analysis_input_digest(report)
    # Identity contains both dependencies and results; a changed result cannot
    # silently replace an already immutable report of the same identity.
    report["id"] = S.make_id("analysis", S.sha256_text({k: v for k, v in report.items() if k != "id"}))
    PS.validate_analysis(report, media)
    return report


def _method(name, tool_versions, parameters=None):
    return {"id": "ffmpeg-" + name, "version": VERSION,
            "parameters": {"tool_versions": tool_versions, **(parameters or {})}}


def _loudness_unavailable(media, domain, stream, versions, reason, window, subject):
    return _report("loudness", media, domain, stream, _method("audio-objective", versions),
        [_value(name, None, unit, reason=reason) for name, unit in (
            ("integrated_lufs", "LUFS"), ("sample_peak_dbfs", "dBFS"),
            ("true_peak_estimate_dbtp", "dBTP"), ("silence_intervals_json", "none"))], window, subject)


def _parse_ebur(stderr):
    patterns = {
        "integrated_lufs": r"\bI:\s*([-+\w.]+)\s+LUFS",
        "absolute_threshold_lufs": r"Integrated loudness:\s*I:\s*[-+\w.]+\s+LUFS\s*Threshold:\s*([-+\w.]+)",
        "sample_peak_dbfs": r"Sample peak:\s*Peak:\s*([-+\w.]+)\s+dBFS",
        "true_peak_estimate_dbtp": r"True peak:\s*Peak:\s*([-+\w.]+)\s+dBFS",
    }
    result = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, stderr)
        if not matches:
            raise P.MediaPreparationError("ebur128 summary is missing " + key)
        result[key] = _finite(matches[-1])
    return result


def _pcm_statistics(path, rate, channels, cancelled):
    if path.stat().st_size % (8 * channels):
        raise P.MediaPreparationError("Decoded PCM contains a partial sample")
    count = path.stat().st_size // (8 * channels)
    if not count:
        return {"samples": 0, "peak_dbfs": None, "silence": [], "rms_dbfs": None}
    audio = np.memmap(path, dtype="<f8", mode="r", shape=(count, channels))
    peak = 0.0
    energy = 0.0
    try:
        for start in range(0, count, 65536):
            P._cancel(cancelled)
            chunk = np.asarray(audio[start:start + 65536])
            if not np.all(np.isfinite(chunk)):
                raise P.MediaPreparationError("Decoded PCM contains nonfinite samples")
            peak = max(peak, float(np.max(np.abs(chunk))))
            energy += float(np.sum(chunk * chunk, dtype=np.float64))
        window = max(1, round(rate * .01))
        minimum = max(1, round(rate * .2))
        threshold = 10 ** (-50 / 20)
        intervals = []
        active = None
        for start in range(0, count, window):
            P._cancel(cancelled)
            end = min(start + window, count)
            quiet = float(np.max(np.abs(audio[start:end]))) < threshold
            if quiet and active is None:
                active = start
            if not quiet and active is not None:
                if start - active >= minimum:
                    intervals.append((active, start))
                active = None
        if active is not None and count - active >= minimum:
            intervals.append((active, count))
        return {"samples": count, "peak_dbfs": 20 * math.log10(peak) if peak > 0 else None,
                "silence": intervals, "rms_dbfs": 10 * math.log10(energy / (count * channels)) if energy > 0 else None}
    finally:
        # Release mmap before a caller can clean a completed analysis directory
        # on Windows. The source and PCM bytes are never changed here.
        del audio


def _technical_reports(probe, media, domain, versions, selected):
    reports = []
    format_info = probe.get("format", {})
    values = [_value("format_name", format_info.get("format_name"), basis="declared"),
              _value("declared_duration", _finite(format_info.get("duration")), "s", basis="declared"),
              _value("stream_count", len(probe["streams"]), basis="declared")]
    reports.append(_report("technical_declared", media, domain, None, _method("probe-declared", versions,
                           {"meaning": "container/stream declarations, not device gamut or pixel-distribution measurement"}), values))
    for stream in selected:
        kind = stream.get("codec_type")
        values = [_value(name, stream.get(name), basis="declared") for name in
                  ("codec_name", "profile", "codec_type", "pix_fmt", "sample_fmt", "channel_layout")]
        values += [_value(name, _finite(stream.get(name)), unit, basis="declared") for name, unit in
                   (("width", "px"), ("height", "px"), ("sample_rate", "Hz"), ("channels", "none"),
                    ("start_time", "s"), ("duration", "s"))]
        for name in ("color_primaries", "color_transfer", "color_space", "color_range", "sample_aspect_ratio"):
            value = stream.get(name)
            if value in ("unknown", "unspecified", "N/A", "0:1"):
                value = None
            values.append(_value(name, value, basis="declared"))
        for name in ("avg_frame_rate", "r_frame_rate"):
            value = stream.get(name)
            try:
                rate = float(Fraction(value)) if value is not None else None
                if rate is not None and rate <= 0:
                    rate = None
            except (ValueError, ZeroDivisionError, TypeError):
                rate = None
            values.append(_value(name, rate, "fps", basis="declared"))
        bits = _finite(stream.get("bits_per_sample"))
        # A codec's0 field means unspecified; it never means0-bit audio.
        values.append(_value("bits_per_sample", bits if bits and bits > 0 else None,
                             basis="declared", reason="not_declared_for_codec"))
        values.append(_value("device_gamut_measurement", None, basis="declared", reason="not_a_file_declaration"))
        reports.append(_report("technical_declared", media, domain, stream["index"],
                               _method("probe-declared", versions, {"stream_type": kind}), values))
    return reports


def _frame_report(ffprobe, source, stream, media, domain, versions, work, cancelled):
    index = stream["index"]
    name = "frames-" + str(index)
    path = P._frames(ffprobe, source, index, work, name, cancelled)
    base = P._time_base(stream)
    summary = P._verified_video_summary(ffprobe, source, index, path, base, work, name + "-packets", cancelled)
    deltas = Counter()
    previous = None
    count = 0
    for row in P._frame_rows(path):
        P._cancel(cancelled)
        count += 1
        if count > MAX_FRAMES:
            raise P.MediaPreparationError("Frame timing exceeds2000000-frame analysis budget")
        if previous is not None:
            deltas[row["pts"] - previous] += 1
        previous = row["pts"]
    values = [_value("decoded_frame_count", count),
              _value("pts_strictly_increasing", True),
              _value("observed_timing", "insufficient_intervals" if not deltas else ("CFR" if len(deltas) == 1 else "VFR")),
              _value("first_frame_pts", str(summary["first"])), _value("last_frame_pts", str(summary["last"])),
              _value("time_base", str(base)), _value("last_frame_hold", float(summary["last_duration"] * base), "s"),
              _value("last_duration_basis", summary["last_duration_basis"])]
    values.append(_value("observed_fps", float(1 / (next(iter(deltas)) * base)) if len(deltas) == 1 else None,
                         "fps", reason="variable_or_insufficient_frame_intervals"))
    values.append(_value("delta_ticks_histogram_json", json.dumps(dict(sorted(deltas.items())), separators=(",", ":"))
                         if len(deltas) <= 128 else None, reason="timing_histogram_budget_exceeded"))
    return _report("frame_timing", media, domain, index, _method("decoded-pts", versions,
                   {"every_frame_checked": True, "frame_limit": MAX_FRAMES,
                    "nominal_fps_is_not_timing_evidence": True}), values)


def _audio_report(ffmpeg, ffprobe, source, stream, media, domain, versions, work, cancelled, window, subject):
    index = stream["index"]
    channels = stream.get("channels")
    layout = stream.get("channel_layout")
    try:
        rate = int(stream.get("sample_rate", 0))
    except (ValueError, TypeError):
        rate = 0
    reason = None
    if type(channels) is not int or channels not in (1, 2):
        reason = "unsupported_channel_count"
    elif layout not in ("mono", "stereo"):
        reason = "unsupported_or_unknown_channel_layout"
    elif (layout == "mono" and channels != 1) or (layout == "stereo" and channels != 2):
        reason = "channel_layout_count_mismatch"
    elif not 1 <= rate <= 384000:
        reason = "unsupported_sample_rate"
    elif ffmpeg is None:
        reason = "ffmpeg_unavailable"
    if reason:
        return _loudness_unavailable(media, domain, index, versions, reason, window, subject)
    name = "audio-" + str(index)
    rows = P._frames(ffprobe, source, index, work, name + "-frames", cancelled)
    summary = P._audio_summary(rows, P._time_base(stream), rate, cancelled,
                               allow_trailing_flush_inference=stream.get("codec_name") in {"wmav1", "wmav2"})
    start, end = 0, summary["samples"]
    origin = Fraction(0) if domain in ("playback_seconds", "original_source_seconds") else S.rational_value(media["common_origin"])
    if window is not None:
        absolute_start = S.rational_value(window["start"]) + origin
        absolute_end = S.rational_value(window["end"]) + origin
        start = max(0, math.ceil((absolute_start - summary["start"]) * rate))
        end = min(end, max(0, math.ceil((absolute_end - summary["start"]) * rate)))
    if end <= start:
        return _loudness_unavailable(media, domain, index, versions, "no_audio_in_window", window, subject)
    if (end - start) * channels * 8 > MAX_PCM_BYTES:
        return _loudness_unavailable(media, domain, index, versions, "decoded_pcm_budget_exceeded", window, subject)
    pcm = work / (name + ".f64le")
    filters = f"atrim=start_sample={start}:end_sample={end},ebur128=peak=sample+true:framelog=verbose"
    P._run([ffmpeg, "-hide_banner", "-v", "info", "-nostdin", "-n", "-xerror", "-err_detect", "explode",
            "-protocol_whitelist", "file,pipe", "-copyts", "-i", str(source), "-map", f"0:{index}",
            "-vn", "-af", filters, "-c:a", "pcm_f64le", "-f", "f64le", str(pcm)],
           work, name + "-decode-measure", cancelled, strict=False)
    if pcm.stat().st_size > MAX_PCM_BYTES:
        raise WorkspaceError("ANALYSIS_LIMIT_EXCEEDED", "Decoded audio exceeds its analysis resource budget")
    stats = _pcm_statistics(pcm, rate, channels, cancelled)
    if stats["samples"] != end - start:
        raise P.MediaPreparationError("Decoded sample count differs from verified audio timeline")
    metrics = _parse_ebur((work / (name + "-decode-measure.stderr")).read_text("utf-8", errors="replace"))
    lufs_reason = None
    if stats["samples"] < math.ceil(.4 * rate):
        lufs_reason = "insufficient_400ms_gating_block"
    elif stats["peak_dbfs"] is None:
        lufs_reason = "digital_silence"
    elif metrics["integrated_lufs"] is None or metrics["integrated_lufs"] <= -70:
        lufs_reason = "at_or_below_absolute_gate_summary_resolution"
    effective_start = summary["start"] + Fraction(start, rate) - origin
    effective_end = summary["start"] + Fraction(end, rate) - origin
    silence = [{"start": S.rational(effective_start + Fraction(a, rate)),
                "end": S.rational(effective_start + Fraction(b, rate))} for a, b in stats["silence"]]
    values = [
        _value("integrated_lufs", None if lufs_reason else metrics["integrated_lufs"], "LUFS", reason=lufs_reason),
        _value("sample_peak_dbfs", stats["peak_dbfs"], "dBFS", reason="digital_silence"),
        _value("true_peak_estimate_dbtp", metrics["true_peak_estimate_dbtp"], "dBTP", reason="digital_silence_or_unavailable_true_peak"),
        _value("rms_dbfs", stats["rms_dbfs"], "dBFS", reason="digital_silence"),
        _value("decoded_samples_per_channel", str(stats["samples"])),
        _value("sample_rate", rate, "Hz"), _value("channels", channels),
        _value("effective_audio_start", float(effective_start), "s"),
        _value("effective_audio_end", float(effective_end), "s"),
        _value("silence_interval_count", len(silence)),
        _value("silence_intervals_json", json.dumps(silence, separators=(",", ":")) if len(silence) <= MAX_SILENCE_INTERVALS else None,
               reason="silence_interval_budget_exceeded"),
    ]
    return _report("loudness", media, domain, index, _method("audio-objective", versions, {
        "ffmpeg_filter": filters, "channel_layout": layout, "sample_rate": rate,
        "requested_source_sample_range": {"start": str(start), "end_exclusive": str(end)},
        "audio_stream_start_pts_seconds": S.rational(summary["start"]),
        "shared_origin_seconds": S.rational(origin), "normalization": False,
        "loudness_window_ms": 400, "loudness_step_ms": 100,
        "absolute_gate_lufs": -70, "relative_gate_lu": -10,
        "silence_threshold_dbfs": -50, "silence_window_ms": 10, "silence_minimum_seconds": .2,
        "silence_rule": "all samples in all channels below threshold;10ms window boundaries; no semantic inference",
        "true_peak_is_estimate": True, "meter_conformance_certified": False,
        "audio_summary_timestamp_basis": summary["timestamp_basis"],
        "pcm_sha256": stream_sha256(pcm, cancelled=cancelled),
    }), values, window, subject)


def analyze_media(source_path, media, output_dir, *, stream_index=None, window=None,
                  subject_annotation_id=None, cancelled=None, time_domain="admitted_source_seconds"):
    """Analyze an explicitly selected hash-matching local file without task writes.

    `window` uses reduced rational half-open boundaries on the admitted source
    or playback clock. Audio sample selection uses the verified original PTS
    plus common origin, never nominal FPS or a player's approximate position.
    Whole-file metadata/frame timing accompany interval audio when requested.
    """
    PS.validate_media(media)
    if time_domain not in {"admitted_source_seconds", "playback_seconds", "original_source_seconds"}:
        raise SchemaInvalid("SCHEMA_INVALID", "Unsupported analysis time domain")
    if cancelled is not None and not callable(cancelled):
        raise SchemaInvalid("SCHEMA_INVALID", "Cancellation must be callable")
    if stream_index is not None and (type(stream_index) is not int or stream_index < 0):
        raise SchemaInvalid("SCHEMA_INVALID", "Stream index must be a nonnegative integer")
    if subject_annotation_id is not None:
        S.require_id(subject_annotation_id)
        if window is None:
            raise SchemaInvalid("SCHEMA_INVALID", "An annotation subject needs an interval window")
    if window is not None:
        if (time_domain == "original_source_seconds" or not isinstance(window, dict) or set(window) != {"start", "end"}
                or any(S.rational(window[k]) != window[k] for k in ("start", "end"))
                or not 0 <= S.rational_value(window["start"]) < S.rational_value(window["end"]) <= S.rational_value(media["duration"])):
            raise SchemaInvalid("SCHEMA_INVALID", "Analysis window must be exact and within admitted/playback media duration")
    source, work = Path(source_path).absolute(), Path(output_dir).absolute()
    _safe(source); _safe(work)
    if not source.is_file():
        raise WorkspaceError("SOURCE_MISSING", "Selected analysis source is unavailable")
    expected = media["playback" if time_domain == "playback_seconds" else "source"]
    started = time.monotonic()
    created = False
    def control():
        if cancelled is not None and cancelled():
            return True
        if time.monotonic() - started > TIMEOUT_SECONDS:
            raise WorkspaceError("ANALYSIS_TIMEOUT", "Objective analysis exceeded its time budget", retryable=True)
        if created:
            if shutil.disk_usage(work).free < RESERVE_BYTES:
                raise WorkspaceError("STORAGE_FULL", "Objective analysis reached its minimum free-space reserve", retryable=True)
            if sum(p.stat().st_size for p in work.iterdir() if p.is_file()) > MAX_WORK_BYTES:
                raise WorkspaceError("ANALYSIS_LIMIT_EXCEEDED", "Objective analysis output exceeded its disk budget")
        return False
    try:
        if source.stat().st_size != int(expected["size_bytes"]) or stream_sha256(source, cancelled=control) != expected["sha256"]:
            raise Conflict("MEDIA_IDENTITY_MISMATCH", "Selected media differs from the admitted identity")
        if work.exists():
            raise Conflict("OUTPUT_EXISTS", "Analysis requires a new output directory")
        work.mkdir(parents=True, exist_ok=False)
        created = True
        ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
        versions = {}
        for name, executable in (("ffprobe", ffprobe), ("ffmpeg", ffmpeg)):
            if executable:
                path = P._run([executable, "-version"], work, name + "-version", control, strict=False)
                versions[name] = path.read_text("utf-8", errors="replace").splitlines()[0]
            else:
                versions[name] = "unavailable"
        if ffprobe is None:
            reports = [_report("technical_declared", media, time_domain, None, _method("probe-declared", versions),
                               [_value("metadata", None, basis="declared", reason="ffprobe_unavailable")]),
                       _loudness_unavailable(media, time_domain, stream_index, versions, "ffprobe_unavailable", window, subject_annotation_id)]
        else:
            probe = P._probe(ffprobe, source, work, "source-probe", control)
            streams = probe["streams"]
            if len(streams) > MAX_STREAMS:
                raise WorkspaceError("ANALYSIS_LIMIT_EXCEEDED", "Media stream count exceeds16")
            selected = streams if stream_index is None else [s for s in streams if s.get("index") == stream_index]
            if not selected:
                raise SchemaInvalid("STREAM_NOT_FOUND", "Selected media stream does not exist")
            reports = _technical_reports(probe, media, time_domain, versions, selected)
            audio_count = 0
            for stream in selected:
                P._cancel(control)
                if stream.get("codec_type") == "video":
                    try:
                        reports.append(_frame_report(ffprobe, source, stream, media, time_domain, versions, work, control))
                    except P.MediaPreparationError:
                        reports.append(_report("frame_timing", media, time_domain, stream["index"],
                            _method("decoded-pts", versions), [_value("verified_frame_timing", None,
                                reason="verified_frame_timing_unavailable")]))
                elif stream.get("codec_type") == "audio":
                    audio_count += 1
                    try:
                        reports.append(_audio_report(ffmpeg, ffprobe, source, stream, media, time_domain, versions,
                                                     work, control, window, subject_annotation_id))
                    except P.MediaPreparationError:
                        reports.append(_loudness_unavailable(media, time_domain, stream["index"], versions,
                            "audio_timeline_or_measurement_unavailable", window, subject_annotation_id))
            if not audio_count and stream_index is None:
                reports.append(_loudness_unavailable(media, time_domain, None, versions, "no_audio_stream", window, subject_annotation_id))
        if source.stat().st_size != int(expected["size_bytes"]) or stream_sha256(source, cancelled=control) != expected["sha256"]:
            raise Conflict("MEDIA_CHANGED", "Analysis source changed during measurement")
        result = {"kind": "traceannotate.objective-analysis-result", "version": 1,
                  "file_sha256": expected["sha256"], "time_domain": time_domain, "reports": reports,
                  "semantic_listening": "not_performed", "source_unchanged": True,
                  "evidence_directory": str(work), "commands": str(work / "commands.jsonl") if (work / "commands.jsonl").exists() else None}
        (work / "result.json").write_bytes(S.canonical_json(result))
        return result
    except InterruptedError as exc:
        raise WorkspaceError("CANCELLED", "Objective analysis was cancelled") from exc
    except P.MediaPreparationError as exc:
        raise WorkspaceError("ANALYSIS_FAILED", "Objective media analysis could not verify the requested measurements",
                             details={"stage": "decode_or_measure"}) from exc
