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
DISSOLVE_RELATIVE_RESIDUAL = 0.20
DISSOLVE_MIN_INTERIOR = 3
DISSOLVE_ALPHA_MARGIN = 0.10
DISSOLVE_REVERSAL_TOLERANCE = 0.03
BLACK_LEVEL = 10
WHITE_LEVEL = 248
NEIGHBOUR_RADIUS = 6

SCENE_CUT_MIN_RAW = 25.0
SCENE_CUT_MIN_HIST = 0.12
SCENE_CUT_MIN_HIST_RAW = 32.0
SCENE_CUT_FLASH_DROP = -45.0
SCENE_CUT_FLASH_HIST = 0.50
SCENE_CUT_FLASH_NEXT = 10.0

TRANSITION_KINDS = ("cut", "fade_in", "fade_out", "dissolve", "flash",
                    "unknown_transition")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Downscale one BGR/RGB/gray frame to a small luma patch."""
    if frame is None:
        raise ValueError("frame is empty")
    array = np.asarray(frame)
    height, width = array.shape[:2]
    step_y = max(1, height // 64)
    step_x = max(1, width // ANALYSIS_WIDTH)
    # Point sampling commutes with the channel-wise luma calculation. Slice
    # first so a 4K source never allocates several full-size float64 planes
    # merely to retain a roughly 64x96 patch.
    small = array[::step_y, ::step_x]
    if small.ndim == 3:
        # the pipeline decodes BGR frames; channel order is documented here
        gray = (0.114 * small[..., 0] + 0.587 * small[..., 1]
                + 0.299 * small[..., 2])
    else:
        gray = small.astype(np.float64)
    return gray.astype(np.float64, copy=False)


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
    """Require multiple observed mixtures progressing between both endpoints.

    A small absolute residual alone also fits a translated edge or an abrupt
    cut. Compare residual to the actual endpoint change and require internal
    blend weights, allowing nonlinear timing while rejecting reversals.
    """
    if end - start < 2:
        return None
    first = np.asarray(track[start]["gray"], dtype=np.float64)
    last = np.asarray(track[end]["gray"], dtype=np.float64)
    far = float(np.abs(first - last).mean() / 255.0)
    if far < DISSOLVE_FAR:
        return None
    delta = last - first
    energy = float(np.sum(delta * delta))
    if energy <= 0:
        return None
    residuals, alphas = [], []
    for index in range(start + 1, end):
        current = np.asarray(track[index]["gray"], dtype=np.float64)
        alpha = float(np.sum((current - first) * delta) / energy)
        if not -DISSOLVE_REVERSAL_TOLERANCE <= alpha <= 1 + DISSOLVE_REVERSAL_TOLERANCE:
            return None
        alpha = max(0.0, min(1.0, alpha))
        alphas.append(alpha)
        blend = (1 - alpha) * first + alpha * last
        residuals.append(float(np.abs(current - blend).mean() / 255.0))
    residual = max(residuals) if residuals else 1.0
    interior = sum(DISSOLVE_ALPHA_MARGIN < value < 1 - DISSOLVE_ALPHA_MARGIN for value in alphas)
    relative = residual / far
    if (residual > DISSOLVE_RESIDUAL or relative > DISSOLVE_RELATIVE_RESIDUAL
            or interior < DISSOLVE_MIN_INTERIOR
            or any(right + DISSOLVE_REVERSAL_TOLERANCE < left for left, right in zip(alphas, alphas[1:]))):
        return None
    return {"endpoint_distance": round(far, 4), "max_residual": round(residual, 4),
            "mean_residual": round(float(np.mean(residuals)), 4),
            "relative_residual": round(relative, 4), "interior_mixture_frames": interior}


def _gradual_candidates(track: List[Dict[str, Any]], low: float = GRADUAL_LOW) -> List[Tuple[int, int]]:
    """Runs of sustained change that are not one-frame spikes.

    The band reaches above the cut threshold so a dissolve whose per-step score
    momentarily looks like a cut is still treated as one transition instead of a
    string of hard cuts.
    """
    high = CUT_THRESHOLD * 2.5
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
    return runs


def _expand_ramp(track: List[Dict[str, Any]], edge: int, *, polarity: str,
                 direction: int) -> Optional[Tuple[int, int]]:
    """Follow one monotonic ramp away from its black/white endpoint.

    Each side of a polarity hold is independent. The incoming ramp fades out
    and the outgoing ramp fades in, so a black bridge can own both events.
    Static holds and one-step black/white cuts have no sustained ramp.
    """
    sign = 1 if polarity == "black" else -1
    outer = edge
    neighbour = outer + direction
    while (0 <= neighbour < len(track)
           and sign * (track[neighbour]["luma"] - track[outer]["luma"]) > 1):
        outer = neighbour
        neighbour += direction
    if abs(outer - edge) < MIN_GRADUAL_FRAMES:
        return None
    return min(edge, outer), max(edge, outer)


def detect_shot_boundaries(samples: Sequence[Dict[str, Any]], *,
                           neighbour_radius: int = NEIGHBOUR_RADIUS,
                           gradual_low: float = GRADUAL_LOW) -> Dict[str, Any]:
    """Detect boundary events from sampled frames.

    Each sample needs frame_index, and optionally seconds/source_pts/proxy_pts
    taken from the workspace frame table, plus either a decoded `frame` array or
    pre-computed `gray`/`luma`/`hist` features. Returns events, the measured
    track and the parameters so a reviewer can re-run the same decision.

    Missing admitted frame indices split the measurement track. In particular,
    three sparse samples cannot establish a single-frame flash between them.
    """
    events: List[Dict[str, Any]] = []
    if not isinstance(gradual_low, (int, float)) or isinstance(gradual_low, bool) or not 0 < gradual_low < CUT_THRESHOLD:
        raise ValueError("gradual proposal threshold must be positive and below the cut threshold")
    public_track: List[Dict[str, Any]] = []
    start = 0
    previous = None
    for index, sample in enumerate(samples):
        current = int(sample["frame_index"])
        if previous is not None:
            if current <= previous:
                raise ValueError("sample frame indices must be strictly increasing")
            if current != previous + 1:
                result = _detect_contiguous_samples(samples[start:index], neighbour_radius, gradual_low)
                events.extend(result["events"])
                public_track.extend(result["track"])
                start = index
        previous = current
    result = _detect_contiguous_samples(samples[start:], neighbour_radius, gradual_low)
    events.extend(result["events"])
    public_track.extend(result["track"])
    return {"events": events, "track": public_track, "params": _params(neighbour_radius, gradual_low)}


def _detect_contiguous_samples(samples: Sequence[Dict[str, Any]],
                               neighbour_radius: int, gradual_low: float) -> Dict[str, Any]:
    if not samples:
        return {"events": [], "track": [], "params": _params(neighbour_radius, gradual_low)}
    track = build_track(samples)
    events: List[Dict[str, Any]] = []
    consumed: set[int] = set()
    fade_ramps = []
    fade_intervals = set()
    fade_frames: set[int] = set()
    for first, last, polarity in _polarity_runs(track):
        for edge, direction, kind in ((first, -1, "fade_out"), (last, 1, "fade_in")):
            ramp = _expand_ramp(track, edge, polarity=polarity, direction=direction)
            if ramp is not None and ramp not in fade_intervals:
                fade_ramps.append((*ramp, kind, polarity))
                fade_intervals.add(ramp)
                fade_frames.update(range(ramp[0], ramp[1] + 1))

    # 1) Flash: one frame spikes and the frame after it returns to the old scene.
    for index in range(1, len(track) - 1):
        if index in fade_frames:
            continue
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

    # 2) Fades: actual incoming/outgoing ramps, excluding their static holds.
    flash_consumed = set(consumed)
    for start, end, kind, polarity in fade_ramps:
        if set(range(start, end + 1)) & flash_consumed:
            continue
        events.append(_event(kind, track, start, end, track[end]["frame_index"],
                             confidence=0.9,
                             uncertainty="端点帧可能落在转场内部",
                             metrics={"polarity": polarity,
                                      "start_luma": round(track[start]["luma"], 2),
                                      "end_luma": round(track[end]["luma"], 2)},
                             interval=True))
        consumed.update(range(start, end + 1))

    # 3) Dissolve / unknown: sustained moderate change that is not a spike.
    for start, end in _gradual_candidates(track, gradual_low):
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
            "params": _params(neighbour_radius, gradual_low)}


def _public_track(track: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The measurable part of the track (arrays stay out of JSON)."""
    return [{"frame_index": entry["frame_index"], "seconds": entry["seconds"],
             "source_pts": entry["source_pts"], "proxy_pts": entry["proxy_pts"],
             "luma": round(entry["luma"], 2),
             "step": {key: round(value, 4) for key, value in entry["step"].items()}
             if "step" in entry else None}
            for entry in track]


def _params(neighbour_radius: int = NEIGHBOUR_RADIUS, gradual_low: float = GRADUAL_LOW) -> Dict[str, Any]:
    return {"analysis_width": ANALYSIS_WIDTH, "hist_bins": HIST_BINS,
            "cut_threshold": CUT_THRESHOLD, "strong_cut": STRONG_CUT, "gradual_low": gradual_low,
            "flash_ratio": FLASH_RATIO, "min_gradual_frames": MIN_GRADUAL_FRAMES,
            "dissolve_residual": DISSOLVE_RESIDUAL, "dissolve_far": DISSOLVE_FAR,
            "dissolve_relative_residual": DISSOLVE_RELATIVE_RESIDUAL,
            "dissolve_min_interior": DISSOLVE_MIN_INTERIOR,
            "dissolve_alpha_margin": DISSOLVE_ALPHA_MARGIN,
            "dissolve_reversal_tolerance": DISSOLVE_REVERSAL_TOLERANCE,
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


def _plain_histogram(gray: np.ndarray) -> np.ndarray:
    values, _ = np.histogram(gray, bins=HIST_BINS, range=(0, 256))
    total = values.sum()
    return values.astype(np.float64) / total if total else values.astype(np.float64)


def classify_scene_cut(samples: Sequence[Dict[str, Any]], *,
                       candidate_frame: int) -> Optional[Dict[str, Any]]:
    """Confirm a scene-detector boundary as a hard cut from dense frames.

    The scene detector proposes boundaries from a sparse global score. This
    classifier re-measures the candidate on the decoded frames around it, snaps
    to the strongest adjacent step, and rejects steps that are too small,
    histogram-flat, or a brightness/exposure swing with an unstable next frame.
    """
    entries = sorted(
        ({"index": int(sample["frame_index"]),
          "gray": sample["gray"] if "gray" in sample else _to_gray(sample["frame"]),
          "seconds": sample.get("seconds"),
          "source_pts": sample.get("source_pts"),
          "proxy_pts": sample.get("proxy_pts")}
         for sample in samples),
        key=lambda item: item["index"])
    if len(entries) < 5:
        return None
    positions = {entry["index"]: offset for offset, entry in enumerate(entries)}
    candidate = int(candidate_frame)
    if candidate not in positions:
        return None

    def raw_step(left: int, right: int) -> float:
        return float(np.abs(entries[left]["gray"] - entries[right]["gray"]).mean())

    def step_score(offset: int) -> float:
        return raw_step(offset - 1, offset)

    base = positions[candidate]
    choices = [offset for offset in range(base - 2, base + 3)
               if 2 <= offset < len(entries) - 2]
    if not choices:
        return None
    chosen = max(choices, key=step_score)
    anchor = entries[chosen]["index"]
    before = np.mean([entry["gray"] for entry in entries[chosen - 2:chosen]], axis=0)
    after = np.mean([entry["gray"] for entry in entries[chosen:chosen + 2]], axis=0)
    avg_raw = float(np.abs(before - after).mean())
    avg_hist = float(np.abs(_plain_histogram(before) - _plain_histogram(after)).sum() / 2)
    flash = float(max(entries[chosen]["gray"].mean(), entries[chosen + 1]["gray"].mean())
                  - max(entries[chosen - 2]["gray"].mean(), entries[chosen + 2]["gray"].mean())
                  )
    next_raw = raw_step(chosen + 1, chosen + 2)
    metrics = {"candidate_frame": candidate, "anchor_frame": anchor,
               "avg_raw": round(avg_raw, 3), "avg_hist": round(avg_hist, 4),
               "flash": round(flash, 3), "next_raw": round(next_raw, 3),
               "source": "scene-candidate-structural-v1"}

    if avg_raw < SCENE_CUT_MIN_RAW:
        return None
    if avg_hist < SCENE_CUT_MIN_HIST and avg_raw < SCENE_CUT_MIN_HIST_RAW:
        return None
    if (flash < SCENE_CUT_FLASH_DROP and avg_hist > SCENE_CUT_FLASH_HIST
            and next_raw > SCENE_CUT_FLASH_NEXT):
        return None

    confidence = round(min(0.95, 0.5 + 0.5 * min(1.0, avg_raw / 120.0)), 4)
    return {"kind": "cut",
            "anchor_frame": str(anchor),
            "previous_frame": str(anchor - 1),
            "representative_frame": str(anchor),
            "split_point": str(anchor),
            "confidence": confidence,
            "uncertainty": None,
            "metrics": metrics,
            "needs_review": False,
            "interval": None}


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
