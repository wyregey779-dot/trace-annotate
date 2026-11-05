"""Shot boundary events: cut, fade in/out, dissolve, flash, unknown transition.

The detector works on a sequence of sampled frames that already carry their real
frame identity from the workspace frame table (frame_index / source_pts /
proxy_pts / seconds), so nothing here invents a time from an average frame rate.

Rules that come from the rea-video shots port and the project contract:

* a cut anchors on the *first frame of the new shot* and keeps the previous frame;
* a gradual transition owns an interval plus a representative anchor and an
  explicit uncertainty, and its split point is reported separately from the full
  transition interval;
* a flash (one bright/dark frame with the following frame back to the old scene)
  never splits the shot by default;
* short shots are never silently merged into a neighbour.

The module is pure numpy so it can be tested against synthetic frame sequences
without FFmpeg; decoding and artifact storage stay the caller's job.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ANALYSIS_WIDTH = 96
HIST_BINS = 32
CUT_THRESHOLD = 0.30
GRADUAL_LOW = 0.05
STRONG_CUT = 0.60
FLASH_RATIO = 0.5
MIN_GRADUAL_FRAMES = 3
DISSOLVE_RESIDUAL = 0.20
DISSOLVE_FAR = 0.15
BLACK_LEVEL = 10
WHITE_LEVEL = 248
NEIGHBOUR_RADIUS = 6

TRANSITION_KINDS = ("cut", "fade_in", "fade_out", "dissolve", "flash",
                    "unknown_transition")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Downscale one BGR/RGB/gray frame to a small luma patch."""
    if frame is None:
        raise ValueError("frame is empty")
    array = np.asarray(frame)
    if array.ndim == 3:
        # the pipeline decodes BGR frames; channel order is documented here
        gray = (0.114 * array[..., 0] + 0.587 * array[..., 1]
                + 0.299 * array[..., 2])
    else:
        gray = array.astype(np.float64)
    height, width = gray.shape
    step_y = max(1, height // 64)
    step_x = max(1, width // ANALYSIS_WIDTH)
    return gray[::step_y, ::step_x].astype(np.float64)


def _histogram(gray: np.ndarray) -> np.ndarray:
    """Smoothed luma histogram.

    A flat synthetic frame puts every pixel in one bin, so an unsmoothed
    histogram reports a full-distance change for a four-level brightness drift.
    Smoothing with a [1,2,1] kernel makes adjacent-bin shifts cheap while a real
    scene change (bins far apart) stays expensive.
    """
    values, _ = np.histogram(gray, bins=HIST_BINS, range=(0, 255))
    total = values.sum()
    if not total:
        return values.astype(np.float64)
    smoothed = np.convolve(values.astype(np.float64) / total, [0.25, 0.5, 0.25], mode="same")
    return smoothed


def frame_features(frame: np.ndarray) -> Dict[str, Any]:
    """Measurements that every later decision is allowed to cite."""
    gray = _to_gray(frame)
    return {"gray": gray, "luma": float(gray.mean()), "hist": _histogram(gray)}


def pair_metrics(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, float]:
    """Histogram + pixel measurements between two adjacent sampled frames."""
    a, b = previous["gray"], current["gray"]
    if a.shape != b.shape:
        raise ValueError("frames differ in size; resample before comparing")
    hist_a, hist_b = previous["hist"], current["hist"]
    hist_distance = float(np.sqrt(np.sum((np.sqrt(hist_a) - np.sqrt(hist_b)) ** 2)) / np.sqrt(2))
    pixel = float(np.abs(a - b).mean() / 255.0)
    changed = float((np.abs(a - b) > 12).mean())
    score = 0.6 * hist_distance + 0.4 * min(1.0, pixel * 4)
    return {"hist": hist_distance, "pixel": pixel, "changed_ratio": changed,
            "score": score, "luma_delta": float(current["luma"] - previous["luma"])}


def build_track(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-sample metrics plus the score of the step that *arrived* at it."""
    track: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        entry = {"frame_index": str(sample["frame_index"]),
                 "seconds": sample.get("seconds"),
                 "source_pts": sample.get("source_pts"),
                 "proxy_pts": sample.get("proxy_pts")}
        if "gray" in sample:
            entry.update({"gray": sample["gray"], "luma": sample["luma"],
                          "hist": sample["hist"]})
        else:
            entry.update(frame_features(sample["frame"]))
        if index:
            entry["step"] = pair_metrics(track[-1], entry)
        track.append(entry)
    return track


def _neighbour_median(track: List[Dict[str, Any]], index: int,
                      radius: int = NEIGHBOUR_RADIUS) -> float:
    neighbours = [entry["step"]["score"]
                  for entry in track[max(1, index - radius):index + radius + 1]
                  if "step" in entry]
    return float(np.median(neighbours)) if neighbours else 0.0


def _dissolve_fit(track: List[Dict[str, Any]], start: int, end: int) -> Optional[Dict[str, float]]:
    """Test whether frames between start/end are a linear blend of the endpoints."""
    if end - start < 2:
        return None
    first, last = track[start]["gray"], track[end]["gray"]
    far = float(np.abs(first - last).mean() / 255.0)
    if far < DISSOLVE_FAR:
        return None
    residuals = []
    count = end - start
    for offset, index in enumerate(range(start + 1, end + 1)):
        alpha = (offset + 1) / count
        blend = (1 - alpha) * first + alpha * last
        residuals.append(float(np.abs(track[index]["gray"] - blend).mean() / 255.0))
    residual = max(residuals) if residuals else 1.0
    if residual > DISSOLVE_RESIDUAL:
        return None
    return {"endpoint_distance": round(far, 4), "max_residual": round(residual, 4),
            "mean_residual": round(float(np.mean(residuals)), 4)}


def _gradual_candidates(track: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Runs of sustained change that are not one-frame spikes.

    The band reaches above the cut threshold so a dissolve whose per-step score
    momentarily looks like a cut is still treated as one transition instead of a
    string of hard cuts.
    """
    low, high = GRADUAL_LOW, CUT_THRESHOLD * 2.5
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index in range(1, len(track)):
        score = track[index]["step"]["score"]
        moderate = low <= score <= high
        if moderate and start is None:
            start = index - 1
        elif not moderate and start is not None:
            if index - start >= MIN_GRADUAL_FRAMES:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(track) - 1 - start >= MIN_GRADUAL_FRAMES:
        runs.append((start, len(track) - 1))
    return runs


def _polarity_runs(track: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    """Contiguous runs of near-black or near-white frames."""
    runs: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    polarity: Optional[str] = None
    for index, entry in enumerate(track):
        current = None
        if entry["luma"] <= BLACK_LEVEL:
            current = "black"
        elif entry["luma"] >= WHITE_LEVEL:
            current = "white"
        if current and current == polarity and start is not None:
            continue
        if start is not None:
            runs.append((start, index - 1, polarity))
            start, polarity = None, None
        if current:
            start, polarity = index, current
    if start is not None:
        runs.append((start, len(track) - 1, polarity))
    return [run for run in runs if run[1] - run[0] >= 1]


def _expand_ramp(track: List[Dict[str, Any]], start: int, end: int) -> Tuple[int, int]:
    """Extend a black/white run across the monotonic luma ramp beside it."""
    brightening = end + 1 < len(track) and track[end + 1]["luma"] > track[end]["luma"] + 1
    if brightening:
        while (end + 1 < len(track)
               and track[end + 1]["luma"] > track[end]["luma"] + 1
               and track[end + 1]["luma"] < 250):
            end += 1
        return start, end
    while (start - 1 >= 0
           and track[start - 1]["luma"] < track[start]["luma"] - 1
           and track[start - 1]["luma"] > 5):
        start -= 1
    return start, end


def detect_shot_boundaries(samples: Sequence[Dict[str, Any]], *,
                           neighbour_radius: int = NEIGHBOUR_RADIUS) -> Dict[str, Any]:
    """Detect boundary events from sampled frames.

    Each sample needs frame_index, and optionally seconds/source_pts/proxy_pts
    taken from the workspace frame table, plus either a decoded `frame` array or
    pre-computed `gray`/`luma`/`hist` features. Returns events, the measured
    track and the parameters so a reviewer can re-run the same decision.
    """
    if not samples:
        return {"events": [], "track": [], "params": _params(neighbour_radius)}
    track = build_track(samples)
    events: List[Dict[str, Any]] = []
    consumed: set[int] = set()

    # 1) Flash: one frame spikes and the frame after it returns to the old scene.
    for index in range(1, len(track) - 1):
        step = track[index]["step"]["score"]
        if step <= CUT_THRESHOLD:
            continue
        # Compare the entry and exit side of the spike: if they agree, the spike
        # is a flash and must not become two cuts around one white frame.
        left = pair_metrics(track[index - 1], track[index + 1])
        if left["score"] < CUT_THRESHOLD:
            events.append(_event("flash", track, index, index,
                                 track[index]["frame_index"],
                                 confidence=round(min(1.0, step), 3),
                                 uncertainty="单帧异常亮度/颜色，未切断镜头",
                                 metrics={"in_score": round(step, 4),
                                          "out_score": round(track[index + 1]["step"]["score"], 4),
                                          "endpoint_score": round(left["score"], 4)}))
            consumed.add(index)
            consumed.add(index + 1)

    # 2) Fades: a black/white run plus the monotonic ramp beside it.
    for start, end, polarity in _polarity_runs(track):
        start, end = _expand_ramp(track, start, end)
        before = track[start - 1]["luma"] if start else None
        after = track[end + 1]["luma"] if end + 1 < len(track) else None
        if polarity == "black":
            kind = "fade_in" if (after or 0) >= (before or 0) else "fade_out"
        else:
            kind = "fade_out" if (before or 0) >= (after or 0) else "fade_in"
        events.append(_event(kind, track, start, end, track[end]["frame_index"],
                             confidence=0.9,
                             uncertainty="端点帧可能落在转场内部",
                             metrics={"polarity": polarity,
                                      "start_luma": round(track[start]["luma"], 2),
                                      "end_luma": round(track[end]["luma"], 2)},
                             interval=True))
        consumed.update(range(start, end + 1))

    # 3) Dissolve / unknown: sustained moderate change that is not a spike.
    for start, end in _gradual_candidates(track):
        indexes = set(range(start, end + 1))
        if indexes & consumed:
            continue
        fit = _dissolve_fit(track, start, end)
        if fit:
            events.append(_event("dissolve", track, start, end,
                                 track[end]["frame_index"], confidence=0.85,
                                 uncertainty="线性混合假设；若含运动或遮挡需人工复核",
                                 metrics=fit, interval=True))
        else:
            events.append(_event("unknown_transition", track, start, end,
                                 track[end]["frame_index"], confidence=0.5,
                                 uncertainty="持续变化但不满足叠化/淡变模型",
                                 metrics={"frames": end - start + 1}, interval=True))
        consumed.update(indexes)

    # 4) Hard cuts last, so a transition interval owns its frames.
    for index in range(1, len(track)):
        if index in consumed or track[index]["step"]["score"] <= CUT_THRESHOLD:
            continue
        median = _neighbour_median(track, index, neighbour_radius)
        strong = track[index]["step"]["score"] >= STRONG_CUT
        relative = track[index]["step"]["score"] >= max(CUT_THRESHOLD * 3, median * 3)
        if not (strong or relative):
            continue
        uncertainty = None if strong else "相对阈值触发，建议并排核对前后帧"
        events.append(_event("cut", track, index, index, track[index]["frame_index"],
                             confidence=round(min(1.0, track[index]["step"]["score"]), 3),
                             uncertainty=uncertainty,
                             metrics={"score": round(track[index]["step"]["score"], 4),
                                      "neighbour_median": round(median, 4)}))

    events.sort(key=lambda item: int(item["anchor_frame"]))
    _fill_split_points(events)
    return {"events": events, "track": _public_track(track),
            "params": _params(neighbour_radius)}


def _public_track(track: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The measurable part of the track (arrays stay out of JSON)."""
    return [{"frame_index": entry["frame_index"], "seconds": entry["seconds"],
             "luma": round(entry["luma"], 2),
             "step": {key: round(value, 4) for key, value in entry["step"].items()}
             if "step" in entry else None}
            for entry in track]


def _params(neighbour_radius: int = NEIGHBOUR_RADIUS) -> Dict[str, Any]:
    return {"analysis_width": ANALYSIS_WIDTH, "hist_bins": HIST_BINS,
            "cut_threshold": CUT_THRESHOLD, "strong_cut": STRONG_CUT,
            "flash_ratio": FLASH_RATIO, "min_gradual_frames": MIN_GRADUAL_FRAMES,
            "dissolve_residual": DISSOLVE_RESIDUAL, "dissolve_far": DISSOLVE_FAR,
            "neighbour_radius": neighbour_radius}


def _event(kind: str, track: List[Dict[str, Any]], start: int, end: int,
           anchor: str, *, confidence: float, uncertainty: Optional[str],
           metrics: Dict[str, Any], interval: bool = False) -> Dict[str, Any]:
    assert kind in TRANSITION_KINDS
    event = {"kind": kind,
             "anchor_frame": anchor,
             "previous_frame": track[start - 1]["frame_index"] if start else None,
             "confidence": confidence,
             "uncertainty": uncertainty,
             "metrics": metrics,
             "needs_review": uncertainty is not None}
    if interval:
        event["interval"] = {"start_frame": track[start]["frame_index"],
                             "end_frame": track[end]["frame_index"],
                             "start_seconds": track[start]["seconds"],
                             "end_seconds": track[end]["seconds"]}
        event["representative_frame"] = anchor
        event["uncertainty_frames"] = max(1, end - start)
    return event


def _fill_split_points(events: List[Dict[str, Any]]) -> None:
    """Report the new-shot split point separately from the full interval."""
    for event in events:
        event["split_point"] = None if event["kind"] == "flash" else event["anchor_frame"]


def events_to_claims(events: List[Dict[str, Any]], evidence_id_for) -> List[Dict[str, Any]]:
    """Map boundary events onto observable claims a reviewer can accept."""
    statements = {"cut": "在锚点帧处发生硬切",
                  "fade_in": "在区间内画面由黑/白渐入",
                  "fade_out": "在区间内画面渐出至黑/白",
                  "dissolve": "在区间内发生叠化",
                  "flash": "出现单帧闪光，未切断镜头",
                  "unknown_transition": "发生未建模的持续转场"}
    claims = []
    for event in events:
        evidence = [item for item in (evidence_id_for(event["anchor_frame"]),
                                      evidence_id_for(event.get("previous_frame")))
                    if item]
        interval = event.get("interval") or {}
        claims.append({"claim_type": "observable",
                       "statement": statements[event["kind"]],
                       "evidence_ids": sorted(set(evidence)),
                       "method": {"id": "shot-boundaries-v1", "version": "1",
                                  "parameters": _params()},
                       "time_window": {"start": interval.get("start_seconds"),
                                       "end": interval.get("end_seconds")},
                       "score": event["confidence"], "score_kind": "rule_score",
                       "uncertainty_reason": event["uncertainty"]})
    return claims
