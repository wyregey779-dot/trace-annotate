"""Camera motion measured from consecutive sampled frames.

Ported from the rea-video LK + RANSAC approach, with the project's own rules:

* only the global transform is reported, never a shot-language label without it;
* a subject that moves inside a locked-off frame must not be called camera motion,
  so the inlier ratio of the global fit and the residual decide;
* speed uses the real PTS interval of the two frames, never an average frame rate;
* zoom and physical dolly are not distinguishable from one image, so the
  direction is reported with an explicit uncertainty instead of a guess.

OpenCV is optional: without it the module reports ``not_run`` rather than
inventing numbers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MOTION_CLASSES = ("static", "pan_left", "pan_right", "tilt_up", "tilt_down",
                  "zoom_in", "zoom_out", "handheld_like", "unknown_motion", "not_run")

MIN_POINTS = 15
MIN_INLIERS = 12
STATIC_PIXELS = 2.0
SCALE_THRESHOLD = 0.02
ROTATION_THRESHOLD = 1.5


def _seconds_of(sample: Dict[str, Any]) -> Optional[float]:
    value = sample.get("seconds")
    if isinstance(value, dict):
        numerator = float(value.get("n", 0))
        denominator = float(value.get("d", 1) or 1)
        return numerator / denominator
    if value is None:
        return None
    return float(value)


def _gray(frame: np.ndarray, width: int = 480) -> np.ndarray:
    """8-bit luma: LK and goodFeaturesToTrack both require CV_8U input."""
    array = np.asarray(frame)
    if array.ndim == 3:
        array = 0.114 * array[..., 0] + 0.587 * array[..., 1] + 0.299 * array[..., 2]
    height, frame_width = array.shape[:2]
    scale = min(1.0, width / float(frame_width))
    if scale < 1.0:
        step = int(round(1 / scale))
        array = array[::step, ::step]
    return np.clip(array, 0, 255).astype(np.uint8)


def measure_pair(previous: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
    """Global transform between two frames, or a measured refusal."""
    try:
        import cv2  # noqa: WPS433 - optional dependency, guarded
    except Exception as error:  # pragma: no cover - environment dependent
        return {"status": "not_run", "reason": f"cv2 unavailable: {error}"}
    left, right = _gray(previous), _gray(current)
    if left.shape != right.shape:
        height = min(left.shape[0], right.shape[0])
        width = min(left.shape[1], right.shape[1])
        left, right = left[:height, :width], right[:height, :width]
    try:
        return _measure_pair(cv2, left, right)
    except Exception as error:  # cv2 assertions must not crash a job
        return {"status": "failed", "reason": f"optical flow failed: {error}"}


def _measure_pair(cv2, left: np.ndarray, right: np.ndarray) -> Dict[str, Any]:
    points = cv2.goodFeaturesToTrack(left, maxCorners=300, qualityLevel=0.01,
                                     minDistance=7, blockSize=7)
    if points is None or len(points) < MIN_POINTS:
        return {"status": "not_run", "reason": "not enough trackable corners",
                "points": 0 if points is None else int(len(points))}
    forward, status, _ = cv2.calcOpticalFlowPyrLK(left, right, points, None)
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(right, left, forward, None)
    good = (status.ravel() == 1) & (status_back.ravel() == 1)
    if good.sum() < MIN_POINTS:
        return {"status": "not_run", "reason": "forward/backward check kept too few points",
                "points": int(good.sum())}
    source = points[good].reshape(-1, 2)
    target = forward[good].reshape(-1, 2)
    back = backward[good].reshape(-1, 2)
    consistent = np.linalg.norm(source - back, axis=1) < 1.0
    source, target = source[consistent], target[consistent]
    if len(source) < MIN_POINTS:
        return {"status": "not_run", "reason": "too few consistent tracks",
                "points": int(len(source))}
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                  ransacReprojThreshold=1.0,
                                                  maxIters=2000)
    if matrix is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return {"status": "not_run", "reason": "global transform not determined",
                "inlier_ratio": float(inliers.mean()) if inliers is not None else 0.0}
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale = float(np.sqrt(a * a + c * c))
    rotation = float(np.degrees(np.arctan2(c, a)))
    predicted = cv2.transform(source.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
    residual = float(np.abs(predicted - target).mean())
    inlier_ratio = float(inliers.mean())
    return {"status": "success", "tx": float(tx), "ty": float(ty),
            "scale": scale, "rotation": rotation, "residual": residual,
            "inlier_ratio": inlier_ratio, "points": int(len(source)),
            "frame_width": int(right.shape[1])}


def classify_pair(metrics: Dict[str, Any], seconds: Optional[float]) -> Dict[str, Any]:
    """Turn a measured transform into a motion class plus its measured speed."""
    if metrics.get("status") != "success":
        return {"class": "not_run", "reason": metrics.get("reason"),
                "speed_px_s": None, "confidence": 0.0}
    tx, ty = metrics["tx"], metrics["ty"]
    scale, rotation = metrics["scale"], metrics["rotation"]
    magnitude = float(np.hypot(tx, ty))
    speed = None if not seconds or seconds <= 0 else magnitude / seconds
    if metrics["inlier_ratio"] < 0.4 or metrics["residual"] > 1.5:
        return {"class": "handheld_like",
                "reason": "全局拟合内点率低或残差高",
                "speed_px_s": speed, "confidence": round(metrics["inlier_ratio"], 3)}
    if abs(scale - 1.0) >= SCALE_THRESHOLD and abs(rotation) < ROTATION_THRESHOLD:
        direction = "zoom_in" if scale > 1.0 else "zoom_out"
        return {"class": direction, "reason": "尺度变化主导",
                "scale_delta": round(scale - 1.0, 4),
                "speed_px_s": speed, "confidence": 0.75,
                "uncertainty": "单目画面无法区分变焦与物理推轨"}
    if abs(rotation) >= ROTATION_THRESHOLD and magnitude < STATIC_PIXELS * 2:
        return {"class": "unknown_motion", "reason": "旋转为主，无对应类别",
                "rotation_deg": round(rotation, 2), "speed_px_s": speed,
                "confidence": 0.4}
    if magnitude < STATIC_PIXELS:
        return {"class": "static", "reason": "位移低于两个像素",
                "speed_px_s": 0.0, "confidence": 0.9}
    if abs(tx) >= abs(ty):
        # a rightward image movement means the camera panned left
        direction = "pan_left" if tx > 0 else "pan_right"
    else:
        direction = "tilt_up" if ty > 0 else "tilt_down"
    return {"class": direction, "dx": round(tx, 2), "dy": round(ty, 2),
            "speed_px_s": None if speed is None else round(speed, 2),
            "confidence": round(min(1.0, metrics["inlier_ratio"] * 1.2), 3)}


def analyse_motion(samples: Sequence[Dict[str, Any]], *,
                   min_pair_confidence: float = 0.5) -> Dict[str, Any]:
    """Measure every adjacent pair and summarise the stable class of the window."""
    pairs: List[Dict[str, Any]] = []
    for index in range(1, len(samples)):
        previous, current = samples[index - 1], samples[index]
        seconds = None
        start, end = _seconds_of(previous), _seconds_of(current)
        if start is not None and end is not None and end > start:
            seconds = end - start
        measured = measure_pair(previous["frame"], current["frame"])
        classified = classify_pair(measured, seconds)
        classified.update({"from_frame": str(previous["frame_index"]),
                           "to_frame": str(current["frame_index"]),
                           "seconds": seconds,
                           "metrics": {key: measured.get(key)
                                       for key in ("tx", "ty", "scale", "rotation",
                                                   "residual", "inlier_ratio", "points")}})
        pairs.append(classified)
    classes = [pair["class"] for pair in pairs if pair["class"] != "not_run"]
    if not classes:
        return {"status": "not_run", "pairs": pairs, "summary": None}
    # A shot is described by the motion it actually contains: a hold frame at the
    # end must not cancel the pan that happened before it.
    moving = [name for name in classes if name != "static"]
    pool = moving or classes
    stable = max(set(pool), key=pool.count)
    agreement = pool.count(stable) / len(classes)
    summary = {"class": stable if agreement >= min_pair_confidence else "unknown_motion",
               "agreement": round(pool.count(stable) / len(pool), 3),
               "pairs": len(classes),
               "mean_speed_px_s": _mean_speed(pairs),
               "uncertainty": None if pool.count(stable) / len(pool) >= 0.8
               else "相邻帧分类不一致，需人工复核"}
    return {"status": "success", "pairs": pairs, "summary": summary}


def _mean_speed(pairs: List[Dict[str, Any]]) -> Optional[float]:
    speeds = [pair["speed_px_s"] for pair in pairs if pair.get("speed_px_s")]
    return round(float(np.mean(speeds)), 2) if speeds else None


def motion_to_claims(result: Dict[str, Any], evidence_id_for,
                     start_frame: str, end_frame: str) -> List[Dict[str, Any]]:
    summary = result.get("summary")
    if not summary:
        return []
    evidence = [item for item in (evidence_id_for(start_frame), evidence_id_for(end_frame)) if item]
    return [{"claim_type": "measured",
             "statement": f"窗口内画面全局运动为 {summary['class']}",
             "evidence_ids": sorted(set(evidence)),
             "method": {"id": "camera-motion-v1", "version": "1",
                        "parameters": {"min_points": MIN_POINTS,
                                       "min_inliers": MIN_INLIERS,
                                       "static_pixels": STATIC_PIXELS}},
             "measurement": {"value": summary["class"],
                             "unit": "class",
                             "mean_speed_px_s": summary["mean_speed_px_s"]},
             "score": summary["agreement"], "score_kind": "rule_score",
             "uncertainty_reason": summary.get("uncertainty")}]
