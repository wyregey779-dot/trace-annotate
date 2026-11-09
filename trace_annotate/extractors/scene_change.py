"""Continuous in-shot content changes, separated from lighting and transients.

A semantic scene change inside one shot must survive three filters:

* it is structural: the distribution of edge energy changes, not just brightness;
* it is chromatic: chroma (colour minus luma) moves, so a light dimmer does not
  qualify;
* it persists: the new state still differs one and two windows later, so an
  occlusion, a subtitle pop or a passing object does not qualify.

Camera pans keep the edge-energy distribution and the chroma roughly constant,
so they are reported as motion, not as a new scene. This module deliberately
reports what it can measure; it never labels a location or an event.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ANALYSIS_WIDTH = 96
EDGE_BINS = 16
CONTENT_THRESHOLD = 0.18
LIGHTING_LUMA = 18.0
PERSISTENCE = 4
MINIMUM_SAMPLES = PERSISTENCE * 2 + 1

CHANGE_KINDS = ("content_change", "lighting_change", "transient_change",
                "camera_motion", "no_change")


def _prepare(frame: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    height, width = array.shape[:2]
    step_y = max(1, height // 64)
    step_x = max(1, width // ANALYSIS_WIDTH)
    small = array[::step_y, ::step_x].astype(np.float64)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    chroma = np.array([float((b - luma).mean()), float((r - luma).mean())])
    gray = luma
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    hist, _ = np.histogram(magnitude, bins=EDGE_BINS, range=(0, 128))
    total = hist.sum()
    edge_hist = hist.astype(np.float64) / total if total else hist.astype(np.float64)
    edge_hist = np.convolve(edge_hist, [0.25, 0.5, 0.25], mode="same")
    return {"luma": float(luma.mean()), "chroma": chroma,
            "edge_hist": edge_hist, "gray": gray}


def features(sample: Dict[str, Any]) -> Dict[str, Any]:
    if "gray" in sample:
        return sample
    return _prepare(sample["frame"])


def _edge_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    left, right = a["edge_hist"], b["edge_hist"]
    return float(np.sqrt(np.sum((np.sqrt(left) - np.sqrt(right)) ** 2)) / np.sqrt(2))


def _chroma_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return float(np.linalg.norm(a["chroma"] - b["chroma"]) / 128.0)


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    edge = _edge_distance(a, b)
    chroma = _chroma_distance(a, b)
    return {"structural": edge, "chromatic": chroma,
            "score": 0.7 * edge + 0.3 * chroma,
            "luma_delta": abs(a["luma"] - b["luma"])}


def minimum_samples(persistence: int = PERSISTENCE) -> int:
    """Samples needed for one anchor plus two persistence windows."""
    return persistence * 2 + 1


def detect_content_changes(samples: Sequence[Dict[str, Any]], *,
                           threshold: float = CONTENT_THRESHOLD,
                           persistence: int = PERSISTENCE) -> Dict[str, Any]:
    """Return candidate content changes plus the negatives that were filtered."""
    required = minimum_samples(persistence)
    provided = len(samples)
    if provided < required:
        return {"changes": [], "filtered": [],
                "params": _params(threshold, persistence),
                "status": "insufficient_samples",
                "provided_samples": provided,
                "required_samples": required,
                "message": (f"内容变化检测至少需要 {required} 个覆盖同一镜头的采样帧，"
                            f"当前只有 {provided} 帧；未判定为无变化。")}
    track = [dict(features(sample), frame_index=str(sample["frame_index"]),
                  seconds=sample.get("seconds")) for sample in samples]
    changes: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []
    index = 0
    while index + persistence * 2 < len(track):
        anchor, later, latest = track[index], track[index + persistence], track[index + persistence * 2]
        immediate = compare(anchor, later)
        sustained = compare(anchor, latest)
        baseline = compare(track[index - 1], later) if index else None
        # A loud change that matches the pre-window frame as soon as the window
        # moves one step is a return, not a new scene: occlusion, subtitle pop,
        # a light flash. It is decided before the lighting rule so a white
        # overlay is not misfiled as an exposure change.
        if (baseline is not None
                and immediate["score"] > threshold * 0.5
                and baseline["score"] < immediate["score"] * 0.4
                and abs(later["luma"] - anchor["luma"]) >= LIGHTING_LUMA / 2):
            filtered.append({"kind": "transient_change",
                             "at_frame": anchor["frame_index"],
                             "reason": "变化在持久窗口内回到原状态（遮挡/提示/闪光）",
                             "metrics": {key: round(value, 4) for key, value in immediate.items()}})
            index += 1
            continue
        if immediate["score"] > threshold and sustained["score"] > threshold:
            if baseline is not None and baseline["score"] <= threshold:
                # the "new" state matches the frame before the window: the change
                # returned, i.e. an occlusion/subtitle/overlay, not a new scene.
                filtered.append({"kind": "transient_change",
                                 "at_frame": anchor["frame_index"],
                                 "reason": "提示/遮挡类变化在持久窗口内恢复",
                                 "metrics": {key: round(value, 4) for key, value in immediate.items()}})
                index += 1
                continue
            located = _locate_change(track, index, index + persistence * 2, threshold)
            change = {"kind": "content_change",
                      "start_frame": located,
                      "end_frame": latest["frame_index"],
                      "metrics": {key: round(value, 4) for key, value in immediate.items()},
                      "sustained": round(sustained["score"], 4),
                      "confidence": round(min(1.0, immediate["score"] / (threshold * 2)), 3),
                      "uncertainty": None}
            changes.append(change)
            index += persistence
            continue
        if (abs(later["luma"] - anchor["luma"]) >= LIGHTING_LUMA
                and immediate["structural"] < threshold
                and immediate["chromatic"] < threshold):
            filtered.append({"kind": "lighting_change",
                             "at_frame": anchor["frame_index"],
                             "reason": "亮度变化但边缘结构与色度基本不变",
                             "metrics": {"luma_delta": round(abs(later["luma"] - anchor["luma"]), 2),
                                         "structural": round(immediate["structural"], 4),
                                         "chromatic": round(immediate["chromatic"], 4)}})
            index += 1
            continue
        if immediate["structural"] > threshold and immediate["chromatic"] <= threshold:
            filtered.append({"kind": "camera_motion",
                             "at_frame": anchor["frame_index"],
                             "reason": "边缘能量分布未变，按运镜而非语义变化处理",
                             "metrics": {key: round(value, 4) for key, value in immediate.items()}})
        index += 1
    return {"changes": changes, "filtered": filtered,
            "params": _params(threshold, persistence),
            "status": "success",
            "provided_samples": provided,
            "required_samples": required}


def _locate_change(track: List[Dict[str, Any]], start: int, end: int,
                   threshold: float) -> str:
    """First frame where the new state is already established and the old one gone."""
    reference = track[end]
    for index in range(start + 1, end + 1):
        if (compare(track[index], reference)["score"] <= threshold
                and compare(track[index - 1], track[start])["score"] <= threshold):
            return track[index]["frame_index"]
    return track[start + 1]["frame_index"]


def _params(threshold: float, persistence: int) -> Dict[str, Any]:
    return {"analysis_width": ANALYSIS_WIDTH, "edge_bins": EDGE_BINS,
            "content_threshold": threshold, "lighting_luma": LIGHTING_LUMA,
            "persistence": persistence,
            "score": "0.7*edge_distribution + 0.3*chroma_dist"}


def changes_to_claims(changes: List[Dict[str, Any]], evidence_id_for) -> List[Dict[str, Any]]:
    claims = []
    for change in changes:
        evidence = [item for item in (evidence_id_for(change["start_frame"]),
                                      evidence_id_for(change.get("end_frame"))) if item]
        claims.append({"claim_type": "observable",
                       "statement": "镜头内部画面内容发生变化",
                       "evidence_ids": sorted(set(evidence)),
                       "method": {"id": "scene-change-v1", "version": "1",
                                  "parameters": _params(CONTENT_THRESHOLD, PERSISTENCE)},
                       "time_window": {"start": None, "end": None},
                       "score": change["confidence"], "score_kind": "rule_score",
                       "uncertainty_reason": change["uncertainty"]})
    return claims
