"""Official TraceAnnotate vision benchmark entry.

The historical QC scripts under ``02_PROJECT_DATA/04_QC`` are retained as
evidence and are not modified. This module is the reproducible replacement for
their two known defects:

* FFmpeg prints ``frame/pts_time`` first and ``lavfi.scene_score`` second. The
  historical parser stored the score first and attached it to the *next*
  timestamp, shifting every row and dropping the tail.
* The historical boundary metric read only ``seedCuts`` and used ``1.5/fps`` as
  tolerance. This module freezes ``shots`` boundaries, excludes gradual
  transitions from the hard-cut set, and matches with actual frame tables at
  ±1 frame, one-to-one.

Subcommands
-----------
``scene-scores``
    Run FFmpeg scene detection and write a JSON document containing every
    decoded frame's ``frame_index``, ``pts``, ``pts_time`` and
    ``scene_score``. Parser issues are explicit and cause a nonzero exit.

``freeze-reference``
    Freeze a shots.json report into a benchmark reference. ``hard_cut`` uses
    only shot starts whose ``transitionIn`` is ``cut``. Fade-in/fade-out/
    dissolve starts are kept in ``gradual`` with a shot-span interval and a
    dispute note; ``seedCuts``/``manualCuts`` remain the authoring trail.

``boundary-metrics``
    Evaluate scene-score predictions against the frozen hard-cut reference
    using an actual frame table. Reports precision/recall/F1 per threshold and
    emits the full match/false-positive/missed lists.

Examples
--------
python scripts/vision_benchmark.py scene-scores --video demo.mp4 --out scores.json
python scripts/vision_benchmark.py freeze-reference --shots shots.json \
    --source demo.mp4 --frame-table frames.csv --out reference-freeze.json
python scripts/vision_benchmark.py boundary-metrics --scores scores.json \
    --reference-freeze reference-freeze.json --frame-table frames.csv \
    --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
    --tolerance-frames 1 --out metrics.json
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
FRAME_RE = re.compile(r"\bframe:\s*(\d+)")
PTS_RE = re.compile(r"\bpts:\s*(-?\d+)")
PTS_TIME_RE = re.compile(r"\bpts_time:\s*([^\s]+)")
SCORE_RE = re.compile(r"lavfi\.scene_score=\s*([^\s]+)")


class BenchmarkError(RuntimeError):
    """A benchmark input or output could not be produced reproducibly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise BenchmarkError(f"{label} is not finite: {value!r}")
    return result


def _parse_frame_line(line: str) -> dict[str, Any]:
    """Parse one ``frame:... pts:... pts_time:...`` metadata line."""
    frame_match = FRAME_RE.search(line)
    pts_match = PTS_RE.search(line)
    time_match = PTS_TIME_RE.search(line)
    if not frame_match:
        raise BenchmarkError(f"frame line has no frame index: {line!r}")
    if not time_match:
        raise BenchmarkError(f"frame line has no pts_time: {line!r}")
    raw_time = time_match.group(1).strip().rstrip(",")
    if raw_time in {"N/A", "nan", "NaN", "inf", "-inf"}:
        raise BenchmarkError(f"frame line has unusable pts_time: {line!r}")
    pts_time = _finite_float(raw_time, "pts_time")
    pts = int(pts_match.group(1)) if pts_match else None
    return {
        "frame_index": int(frame_match.group(1)),
        "pts": pts,
        "pts_time": pts_time,
        "score": None,
        "raw": line,
    }


def parse_scene_metadata(text: str) -> dict[str, Any]:
    """Pair each ``pts_time`` with the immediately following scene score.

    FFmpeg's actual order is::

        frame:0    pts:0       pts_time:0
        lavfi.scene_score=0.000000

    A score is accepted only for the current frame. A missing score, an orphan
    score, a duplicate score, a malformed score, or an unparsable frame line is
    recorded as an issue. Complete frames are returned in ``frames``; frames
    with a missing score are returned in ``incomplete`` so callers can audit
    the tail instead of silently dropping it.
    """
    frames: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def close_current(reason: str, line_number: int) -> None:
        nonlocal current
        if current is None:
            return
        if current.get("score") is None:
            current["missing_score_line"] = line_number
            current["missing_score_reason"] = reason
            incomplete.append(current)
            issues.append({
                "kind": "missing_score",
                "line": line_number,
                "frame_index": current.get("frame_index"),
                "pts_time": current.get("pts_time"),
                "reason": reason,
            })
        else:
            frames.append(current)
        current = None

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("frame:"):
            close_current("next frame started before a scene score", line_number)
            try:
                current = _parse_frame_line(line)
            except BenchmarkError as exc:
                issues.append({"kind": "unparsable_frame_line", "line": line_number,
                               "raw": line, "reason": str(exc)})
                current = None
            continue
        if SCORE_RE.search(line):
            raw_score = SCORE_RE.search(line).group(1).strip().rstrip(",")
            try:
                score = _finite_float(raw_score, "scene_score")
                if not 0.0 <= score <= 1.0:
                    raise BenchmarkError(f"scene_score outside [0,1]: {score!r}")
            except BenchmarkError as exc:
                issues.append({"kind": "unparsable_score", "line": line_number,
                               "raw": line, "reason": str(exc)})
                continue
            if current is None:
                issues.append({"kind": "orphan_score", "line": line_number,
                               "score": score, "reason": "score without a frame line"})
                continue
            if current.get("score") is not None:
                issues.append({"kind": "duplicate_score", "line": line_number,
                               "frame_index": current.get("frame_index"),
                               "first_score": current.get("score"),
                               "duplicate_score": score})
                continue
            current["score"] = score
            continue

    close_current("end of metadata before a scene score", len(text.splitlines()) + 1)
    return {
        "frames": frames,
        "incomplete": incomplete,
        "issues": issues,
        "complete": bool(frames) and not incomplete and not issues,
        "frame_count": len(frames),
        "incomplete_count": len(incomplete),
        "issue_count": len(issues),
    }


def build_scene_command(video: Path, ffmpeg: str = "ffmpeg") -> list[str]:
    """Build a command that prints metadata for every decoded video frame."""
    return [
        ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "error",
        "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "setpts=PTS-STARTPTS,select='gte(scene,0)',metadata=print:file=-",
        "-f", "null", "-",
    ]


def run_scene_scores(video: Path, out: Path, *, ffmpeg: str = "ffmpeg",
                     timeout: float = 3600.0,
                     raw_out: Path | None = None) -> dict[str, Any]:
    """Run FFmpeg and write the paired score document plus real diagnostics."""
    if not video.is_file():
        raise BenchmarkError(f"video is not a regular file: {video}")
    command = build_scene_command(video, ffmpeg)
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload = {
            "schema": "traceannotate.vision-benchmark.scene-scores.v1",
            "ok": False, "video": str(video), "command": command,
            "error": f"FFmpeg could not complete: {exc}",
        }
        _write_json(out, payload)
        return payload
    parsed = parse_scene_metadata(proc.stdout)
    if raw_out is not None:
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(proc.stdout, encoding="utf-8", newline="\n")
    payload = {
        "schema": "traceannotate.vision-benchmark.scene-scores.v1",
        "ok": proc.returncode == 0 and parsed["complete"],
        "video": str(video),
        "video_sha256": _sha256(video),
        "video_size_bytes": video.stat().st_size,
        "ffmpeg": ffmpeg,
        "command": command,
        "returncode": proc.returncode,
        "raw_output": str(raw_out) if raw_out is not None else None,
        "raw_output_sha256": _sha256(raw_out) if raw_out is not None and raw_out.is_file() else None,
        "stderr_tail": proc.stderr.strip()[-4000:],
        **parsed,
    }
    _write_json(out, payload)
    return payload


def read_frame_table(path: Path) -> list[dict[str, Any]]:
    """Read actual frame PTS from JSON or ffprobe CSV.

    Accepted forms:
    * JSON list of numbers (times)
    * JSON list of dicts with ``pts_time``/``best_effort_timestamp_time``
    * JSON object with ``frames`` using either form
    * CSV with or without a header; ffprobe's trailing comma is tolerated
    """
    if not path.is_file():
        raise BenchmarkError(f"frame table is not a regular file: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("frames") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise BenchmarkError("frame table JSON must contain a frames list")
        normalized = []
        for index, row in enumerate(rows):
            if isinstance(row, (int, float)):
                normalized.append({"frame_index": index, "pts_time": _finite_float(row, "pts_time")})
            elif isinstance(row, dict):
                time_value = row.get("pts_time", row.get("best_effort_timestamp_time"))
                if time_value is None:
                    raise BenchmarkError(f"frame row {index} has no pts_time")
                frame_index = row.get("frame_index", index)
                normalized.append({"frame_index": int(frame_index),
                                   "pts_time": _finite_float(time_value, "pts_time")})
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                normalized.append({"frame_index": int(row[0]),
                                   "pts_time": _finite_float(row[1], "pts_time")})
            else:
                raise BenchmarkError(f"frame row {index} has unsupported shape: {row!r}")
    else:
        normalized = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise BenchmarkError("frame CSV is empty")
            header_text = " ".join(header).lower()
            has_header = any("time" in cell.lower() or "pts" in cell.lower()
                             or "frame" in cell.lower() for cell in header)
            rows = list(reader) if has_header else [header, *reader]
            time_column = 0
            frame_column = None
            if has_header:
                for index, cell in enumerate(header):
                    lowered = cell.strip().lower()
                    if lowered in {"pts_time", "best_effort_timestamp_time", "time"}:
                        time_column = index
                    if lowered in {"frame_index", "frame", "index"}:
                        frame_column = index
            for row_number, row in enumerate(rows, 2 if has_header else 1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if time_column >= len(row):
                    raise BenchmarkError(f"frame CSV row {row_number} has no time column")
                raw_time = row[time_column].strip().rstrip(",")
                if raw_time in {"", "N/A"}:
                    raise BenchmarkError(f"frame CSV row {row_number} has empty time")
                frame_index = int(row[frame_column]) if frame_column is not None and frame_column < len(row) else row_number - 1
                normalized.append({"frame_index": frame_index,
                                   "pts_time": _finite_float(raw_time, "pts_time")})
    if not normalized:
        raise BenchmarkError("frame table is empty")
    times = [row["pts_time"] for row in normalized]
    for left, right in zip(times, times[1:]):
        if right < left:
            raise BenchmarkError("frame PTS is not monotonically increasing")
    if len(set(times)) != len(times):
        raise BenchmarkError("frame PTS contains duplicate timestamps")
    return normalized


def nearest_frame_index(frame_times: Sequence[float], timestamp: float) -> int:
    """Return the nearest actual frame index for a timestamp."""
    if not frame_times:
        raise BenchmarkError("frame table is empty")
    position = bisect.bisect_left(frame_times, timestamp)
    if position <= 0:
        return 0
    if position >= len(frame_times):
        return len(frame_times) - 1
    before = position - 1
    after = position
    if abs(frame_times[after] - timestamp) < abs(timestamp - frame_times[before]):
        return after
    return before


def match_boundaries(prediction_times: Iterable[float], reference_times: Iterable[float],
                     frame_table: Sequence[dict[str, Any]], *, tolerance_frames: int = 1) -> dict[str, Any]:
    """One-to-one nearest match with actual-frame tolerance."""
    if tolerance_frames < 0:
        raise BenchmarkError("tolerance_frames must be non-negative")
    frame_times = [row["pts_time"] for row in frame_table]
    predictions = [_finite_float(value, "prediction") for value in prediction_times]
    references = [_finite_float(value, "reference") for value in reference_times]
    pred_frames = [nearest_frame_index(frame_times, value) for value in predictions]
    ref_frames = [nearest_frame_index(frame_times, value) for value in references]
    candidates = []
    for pred_index, pred_frame in enumerate(pred_frames):
        for ref_index, ref_frame in enumerate(ref_frames):
            distance = abs(pred_frame - ref_frame)
            if distance <= tolerance_frames:
                candidates.append((distance, pred_index, ref_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches = []
    for distance, pred_index, ref_index in candidates:
        if pred_index in used_predictions or ref_index in used_references:
            continue
        used_predictions.add(pred_index)
        used_references.add(ref_index)
        matches.append({
            "prediction_index": pred_index,
            "reference_index": ref_index,
            "prediction_time": predictions[pred_index],
            "reference_time": references[ref_index],
            "prediction_frame": pred_frames[pred_index],
            "reference_frame": ref_frames[ref_index],
            "distance_frames": distance,
        })
    false_positives = [{"index": index, "time": predictions[index], "frame": pred_frames[index]}
                       for index in range(len(predictions)) if index not in used_predictions]
    missed = [{"index": index, "time": references[index], "frame": ref_frames[index]}
              for index in range(len(references)) if index not in used_references]
    true_positives = len(matches)
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(references) if references else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_frames": tolerance_frames,
        "prediction_count": len(predictions),
        "reference_count": len(references),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed": missed,
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "frame_table_count": len(frame_table),
        "frame_table_start": frame_times[0],
        "frame_table_end": frame_times[-1],
    }


def _reference_times(payload: dict[str, Any], kind: str) -> list[float]:
    if kind == "hard_cut":
        rows = payload.get("hard_cut", [])
    elif kind == "all_boundary":
        rows = payload.get("all_boundary", payload.get("all_boundary_events", []))
    else:
        raise BenchmarkError(f"unknown reference kind: {kind}")
    if isinstance(rows, list):
        values = []
        for row in rows:
            if isinstance(row, dict):
                value = row.get("start", row.get("time"))
            else:
                value = row
            if value is None:
                raise BenchmarkError(f"reference row has no start/time: {row!r}")
            values.append(_finite_float(value, "reference time"))
        return values
    if isinstance(rows, int):
        raise BenchmarkError(f"reference {kind} is a count, not a list")
    raise BenchmarkError(f"reference {kind} is not a list")


def evaluate_thresholds(scores: dict[str, Any], reference: dict[str, Any],
                        frame_table: Sequence[dict[str, Any]],
                        thresholds: Sequence[float], *,
                        reference_kind: str = "hard_cut",
                        tolerance_frames: int = 1) -> dict[str, Any]:
    frames = scores.get("frames")
    if not isinstance(frames, list) or not frames:
        raise BenchmarkError("scores document has no complete frames")
    numeric_scores = [_finite_float(row.get("score"), "score") for row in frames]
    times = [_finite_float(row.get("pts_time"), "pts_time") for row in frames]
    ref_times = _reference_times(reference, reference_kind)
    results = []
    for threshold in thresholds:
        detected = [time for time, score in zip(times, numeric_scores) if score >= threshold]
        result = match_boundaries(detected, ref_times, frame_table,
                                 tolerance_frames=tolerance_frames)
        result["threshold"] = threshold
        results.append(result)
    best = max(results, key=lambda item: (item["f1"], item["precision"], -item["threshold"]))
    return {
        "schema": "traceannotate.vision-benchmark.boundary-metrics.v1",
        "reference_kind": reference_kind,
        "reference_count": len(ref_times),
        "tolerance_frames": tolerance_frames,
        "frame_table_count": len(frame_table),
        "thresholds": results,
        "best": best,
        "parameters": {"thresholds": list(thresholds), "tolerance_frames": tolerance_frames},
    }


def freeze_reference(shots_path: Path, *, source: Path | None = None,
                     frame_table: Sequence[dict[str, Any]] | None = None,
                     source_id: str | None = None,
                     reference_author_type: str = "human_assisted_external_tool") -> dict[str, Any]:
    """Freeze a shots.json report into hard-cut, gradual and authoring lists."""
    if not shots_path.is_file():
        raise BenchmarkError(f"shots file is not a regular file: {shots_path}")
    payload = json.loads(shots_path.read_text(encoding="utf-8"))
    shots = payload.get("shots")
    if not isinstance(shots, list) or not shots:
        raise BenchmarkError("shots.json must contain a non-empty shots list")
    frame_times = [row["pts_time"] for row in frame_table] if frame_table else None
    all_boundary = []
    hard_cut = []
    gradual = []
    for index, shot in enumerate(shots):
        start = _finite_float(shot.get("start"), "shot.start")
        end = _finite_float(shot.get("end"), "shot.end")
        transition = str(shot.get("transitionIn", "cut"))
        entry = {
            "shot_id": shot.get("id"),
            "start": start,
            "end": end,
            "transition_in": transition,
            "frame_index": nearest_frame_index(frame_times, start) if frame_times else None,
        }
        if start > 0:
            all_boundary.append(entry)
            if transition == "cut":
                hard_cut.append(entry)
            elif transition in {"fade-in", "fade-out", "dissolve", "flash"}:
                gradual_entry = dict(entry)
                gradual_entry["interval"] = {"start": start, "end": end}
                gradual_entry["interval_basis"] = "shot_span"
                gradual_entry["status"] = "candidate_reference_interval_needs_review"
                gradual.append(gradual_entry)
            else:
                gradual_entry = dict(entry)
                gradual_entry["status"] = "unknown_transition_kept"
                gradual.append(gradual_entry)
    seed = [_finite_float(value, "seedCut") for value in payload.get("seedCuts", [])]
    manual = [_finite_float(value, "manualCut") for value in payload.get("manualCuts", [])]
    shot_starts = [entry["start"] for entry in all_boundary]
    seed_not_in_shots = [value for value in seed if value not in shot_starts]
    manual_not_in_shots = [value for value in manual if value not in shot_starts]
    shots_not_in_authoring = [value for value in shot_starts if value not in set(seed) | set(manual)]
    disputed = [
        {"time": value, "source": "seedCuts", "status": "outside_final_shots",
         "reason": "seed cut occurs inside a final shot span; kept visible, not counted as hard-cut truth"}
        for value in seed_not_in_shots
    ] + [
        {"time": value, "source": "manualCuts", "status": "outside_final_shots",
         "reason": "manual cut occurs outside final shot starts; kept visible, not counted as hard-cut truth"}
        for value in manual_not_in_shots
    ]
    source_sha = _sha256(source) if source and source.is_file() else None
    reference_sha = _sha256(shots_path)
    document = {
        "schema": "traceannotate.vision-benchmark.reference-freeze.v1",
        "source_id": source_id or payload.get("source") or shots_path.stem,
        "source_path": str(source) if source else None,
        "source_sha256": source_sha,
        "reference_path": str(shots_path),
        "reference_sha256": reference_sha,
        "reference_author_type": reference_author_type,
        "frozen_at_utc": None,
        "shots_count": len(shots),
        "all_boundary_count": len(all_boundary),
        "hard_cut_count": len(hard_cut),
        "gradual_count": len(gradual),
        "all_boundary": all_boundary,
        "hard_cut": hard_cut,
        "gradual": gradual,
        "seed_cuts": seed,
        "manual_cuts": manual,
        "reconciliation": {
            "seed_cuts_not_in_shots": seed_not_in_shots,
            "manual_cuts_not_in_shots": manual_not_in_shots,
            "shots_not_in_seed_or_manual": shots_not_in_authoring,
            "rule": "shots starts are the final segmentation; seedCuts/manualCuts are authoring trail; no union truth",
        },
        "disputed": disputed,
        "transition_counts": _transition_counts(shots),
        "frame_table_count": len(frame_table) if frame_table else None,
        "notes": [
            "Hard cut excludes fade-in/fade-out/dissolve/flash starts.",
            "Gradual intervals currently use the shot span and remain disputed unless a human supplies the true transition interval.",
            "The previous current_corpus estimate of 56 all-boundary events is not the hard-cut denominator; hard-cut count is 53 across the two samples.",
        ],
        "status": "FROZEN_HARD_CUT_GRADUAL_INTERVALS_DISPUTED" if gradual else "FROZEN",
    }
    return document


def _transition_counts(shots: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shot in shots:
        key = str(shot.get("transitionIn", "cut"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _parse_thresholds(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = _finite_float(part, "threshold")
        if not 0.0 <= value <= 1.0:
            raise BenchmarkError(f"threshold outside [0,1]: {value}")
        values.append(value)
    if not values:
        raise BenchmarkError("at least one threshold is required")
    return values


def _cmd_scene_scores(args: argparse.Namespace) -> int:
    payload = run_scene_scores(args.video, args.out, ffmpeg=args.ffmpeg,
                               timeout=args.timeout, raw_out=args.raw_out)
    print(f"scene-scores ok={payload.get('ok')} frames={payload.get('frame_count')} "
          f"incomplete={payload.get('incomplete_count')} issues={payload.get('issue_count')} "
          f"exit={payload.get('returncode')} out={args.out}")
    if not payload.get("ok"):
        return 2 if payload.get("returncode", 0) != 0 else 3
    return 0


def _cmd_freeze_reference(args: argparse.Namespace) -> int:
    table = read_frame_table(args.frame_table) if args.frame_table else None
    payload = freeze_reference(args.shots, source=args.source, frame_table=table,
                               source_id=args.source_id,
                               reference_author_type=args.reference_author_type)
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.out, payload)
    print(f"freeze-reference shots={payload['shots_count']} all={payload['all_boundary_count']} "
          f"hard_cut={payload['hard_cut_count']} gradual={payload['gradual_count']} out={args.out}")
    return 0


def _cmd_boundary_metrics(args: argparse.Namespace) -> int:
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_freeze.read_text(encoding="utf-8"))
    table = read_frame_table(args.frame_table)
    thresholds = _parse_thresholds(args.thresholds)
    payload = evaluate_thresholds(scores, reference, table, thresholds,
                                  reference_kind=args.reference_kind,
                                  tolerance_frames=args.tolerance_frames)
    payload["inputs"] = {
        "scores": str(args.scores),
        "scores_sha256": _sha256(args.scores),
        "reference_freeze": str(args.reference_freeze),
        "reference_sha256": _sha256(args.reference_freeze),
        "frame_table": str(args.frame_table),
        "frame_table_sha256": _sha256(args.frame_table),
    }
    _write_json(args.out, payload)
    best = payload["best"]
    print(f"boundary-metrics hard_cut_ref={payload['reference_count']} "
          f"best_th={best['threshold']:.2f} p={best['precision']:.3f} "
          f"r={best['recall']:.3f} f1={best['f1']:.3f} out={args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TraceAnnotate official vision benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    scene = sub.add_parser("scene-scores", help="Run FFmpeg and pair frame PTS with scene scores")
    scene.add_argument("--video", type=Path, required=True)
    scene.add_argument("--out", type=Path, required=True)
    scene.add_argument("--ffmpeg", default="ffmpeg")
    scene.add_argument("--timeout", type=float, default=3600.0)
    scene.add_argument("--raw-out", type=Path)
    scene.set_defaults(func=_cmd_scene_scores)

    freeze = sub.add_parser("freeze-reference", help="Freeze shots.json into a benchmark reference")
    freeze.add_argument("--shots", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--source", type=Path)
    freeze.add_argument("--frame-table", type=Path)
    freeze.add_argument("--source-id")
    freeze.add_argument("--reference-author-type", default="human_assisted_external_tool")
    freeze.set_defaults(func=_cmd_freeze_reference)

    metrics = sub.add_parser("boundary-metrics", help="Evaluate boundaries with actual frames")
    metrics.add_argument("--scores", type=Path, required=True)
    metrics.add_argument("--reference-freeze", type=Path, required=True)
    metrics.add_argument("--frame-table", type=Path, required=True)
    metrics.add_argument("--thresholds", default=",".join(f"{v:.2f}" for v in DEFAULT_THRESHOLDS))
    metrics.add_argument("--tolerance-frames", type=int, default=1)
    metrics.add_argument("--reference-kind", choices=("hard_cut", "all_boundary"), default="hard_cut")
    metrics.add_argument("--out", type=Path, required=True)
    metrics.set_defaults(func=_cmd_boundary_metrics)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchmarkError as exc:
        print(f"BENCHMARK_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
