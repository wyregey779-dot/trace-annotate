"""Expressive cues: observable face cues plus independent acoustic measurements.

Two very different things live here on purpose:

* face cues are geometry on landmarks a detector already produced. A cue without
  a frame and without landmarks is ``not_applicable``, never an assumption; a
  small, side-facing or occluded face is refused rather than guessed; and a cue is
  never turned into an inner emotional state.
* acoustic measurements are deterministic signal numbers: window RMS in dBFS,
  energy-active intervals above the local noise floor, and f0 with its variation
  for voiced frames. Units, audio origin and the analysis window are part of the
  result. Timbre, identity or emotion are not measured here.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

CUE_KINDS = ("smile", "brow_furrow", "mouth_open", "eyes_closed", "eyes_narrowed")
F0_MIN = 60.0
F0_MAX = 500.0
FRAME_HOP_SECONDS = 0.02
MIN_FACE_HEIGHT = 48
MIN_FACE_RATIO = 0.06


def _analysis_guard(cancelled: Optional[Callable[[], bool]],
                    deadline_monotonic: Optional[float]) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("acoustic_analysis_cancelled")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError("acoustic_analysis_timeout")


def _rms_dbfs(samples: np.ndarray, *,
               cancelled: Optional[Callable[[], bool]] = None,
               deadline_monotonic: Optional[float] = None) -> float:
    source = np.asarray(samples).ravel()
    if source.size == 0:
        return -240.0
    # Accumulate in bounded chunks so a memory-mapped long recording is not
    # copied into one full float64 work array.
    total = 0.0
    step = 1_000_000
    for index in range(0, source.size, step):
        _analysis_guard(cancelled, deadline_monotonic)
        chunk = source[index:index + step].astype(np.float64, copy=False)
        total += float(np.dot(chunk, chunk))
    value = float(np.sqrt(total / source.size))
    return 20.0 * np.log10(max(value, 1e-12))


def measure_acoustic_window(samples: Sequence[float], sample_rate: int, *,
                            origin_seconds: float = 0.0,
                            cancelled: Optional[Callable[[], bool]] = None,
                            deadline_monotonic: Optional[float] = None) -> Dict[str, Any]:
    """RMS, energy-active intervals and f0 for one analysis window.

    The adaptive energy threshold is deliberately not called speech activity.
    Music, machinery and other non-speech sounds can all cross it.
    """
    audio = np.asarray(samples).ravel()
    if audio.size == 0 or sample_rate <= 0:
        return {"status": "not_run", "reason": "空窗口或非法采样率"}
    _analysis_guard(cancelled, deadline_monotonic)
    duration = audio.size / float(sample_rate)
    rms = _rms_dbfs(audio, cancelled=cancelled,
                    deadline_monotonic=deadline_monotonic)
    hop = max(1, int(round(FRAME_HOP_SECONDS * sample_rate)))
    energy_values = []
    for index in range(0, audio.size - hop + 1, hop):
        _analysis_guard(cancelled, deadline_monotonic)
        energy_values.append(_rms_dbfs(audio[index:index + hop]))
    energies = np.asarray(energy_values)
    if energies.size == 0:
        energies = np.array([rms])
    floor = float(np.percentile(energies, 10))
    # A window that is loud from end to end has no quieter reference: its
    # percentile floor equals the signal, so a pure relative threshold would
    # report zero activity. Cap the relative threshold below the window RMS.
    threshold = max(min(floor + 6.0, rms - 6.0), -60.0)
    active = []
    start: Optional[int] = None
    for index, energy in enumerate(energies):
        _analysis_guard(cancelled, deadline_monotonic)
        loud = energy >= threshold
        if loud and start is None:
            start = index
        elif not loud and start is not None:
            active.append(_interval(start, index - 1, hop, sample_rate, origin_seconds))
            start = None
    if start is not None:
        active.append(_interval(start, len(energies) - 1, hop, sample_rate, origin_seconds))
    f0 = _fundamental(audio, sample_rate, cancelled=cancelled,
                      deadline_monotonic=deadline_monotonic)
    active_ratio = (round(sum(item["end"] - item["start"] for item in active) / duration, 3)
                    if duration else 0.0)
    return {"status": "success", "window_seconds": {"start": origin_seconds,
                                                    "end": origin_seconds + duration},
            "sample_rate": sample_rate, "samples": int(audio.size),
            "rms_dbfs": round(rms, 3), "noise_floor_dbfs": round(floor, 3),
            "activity_kind": "energy_above_adaptive_threshold",
            "energy_active_intervals": active, "energy_active_ratio": active_ratio,
            # Backward-compatible numeric aliases.  Callers must use
            # ``activity_kind`` and must not describe these as speech.
            "active_intervals": active, "active_ratio": active_ratio,
            "f0_hz": f0["value"], "f0_variation_hz": f0["variation"],
            "f0_frames": f0["frames"], "method": "acoustic-window-v2"}


def _interval(start: int, end: int, hop: int, sample_rate: int, origin: float) -> Dict[str, float]:
    return {"start": round(origin + start * hop / sample_rate, 3),
            "end": round(origin + (end + 1) * hop / sample_rate, 3)}


def _fundamental(audio: np.ndarray, sample_rate: int, *,
                 cancelled: Optional[Callable[[], bool]] = None,
                 deadline_monotonic: Optional[float] = None) -> Dict[str, Any]:
    """Autocorrelation f0 on voiced frames; silence returns no value."""
    frame = int(0.04 * sample_rate)
    values: List[float] = []
    for index in range(0, audio.size - frame + 1, max(1, frame // 2)):
        _analysis_guard(cancelled, deadline_monotonic)
        chunk = audio[index:index + frame]
        if _rms_dbfs(chunk) < -45.0:
            continue
        chunk = chunk - chunk.mean()
        correlation = np.correlate(chunk, chunk, mode="full")[frame - 1:]
        low = int(sample_rate / F0_MAX)
        high = min(len(correlation) - 1, int(sample_rate / F0_MIN))
        if high <= low:
            continue
        peak = low + int(np.argmax(correlation[low:high]))
        if correlation[peak] <= 0:
            continue
        values.append(sample_rate / float(peak))
    if not values:
        return {"value": None, "variation": None, "frames": 0}
    return {"value": round(float(np.median(values)), 2),
            "variation": round(float(np.std(values)), 2), "frames": len(values)}


def face_cue_candidates(face: Dict[str, Any], *, frame_index: str,
                        evidence_id: Optional[str]) -> Dict[str, Any]:
    """Observable cues from one face's landmarks, or a measured refusal."""
    box = face.get("box")
    if not box or len(box) < 4:
        return {"status": "not_applicable", "reason": "没有主体框"}
    height = float(box[3]) - float(box[1])
    frame_area = face.get("frame_size")
    if height < MIN_FACE_HEIGHT:
        return {"status": "not_applicable", "reason": "人脸过小，线索不可信"}
    if frame_area and len(frame_area) >= 2 and height / float(frame_area[1]) < MIN_FACE_RATIO:
        return {"status": "not_applicable", "reason": "人脸占画面过小"}
    if face.get("occluded"):
        return {"status": "not_applicable", "reason": "遮挡"}
    quality = face.get("quality") or {}
    if quality.get("side_facing") or quality.get("excessive_roll"):
        return {"status": "not_applicable", "reason": "侧脸或旋转过大，几何线索不可信"}
    if quality.get("partial_outside_frame"):
        return {"status": "not_applicable", "reason": "人脸关键点越出画面，拒绝截断线索"}
    landmarks = face.get("landmarks")
    if not landmarks:
        return {"status": "not_run", "reason": "没有人脸关键点，不能推断表情线索"}
    if not evidence_id:
        return {"status": "not_run", "reason": "缺少帧证据，不生成肯定线索"}
    cues = []
    mouth = landmarks.get("mouth")
    if mouth and len(mouth) >= 4:
        width = float(np.hypot(*(np.array(mouth[0]) - np.array(mouth[1]))))
        opening = float(np.hypot(*(np.array(mouth[2]) - np.array(mouth[3]))))
        if width > 0 and opening / width >= 0.35:
            cues.append(_cue("mouth_open", frame_index, evidence_id,
                             {"mouth_opening_ratio": round(opening / width, 3)}))
        if width > 0 and opening / width >= 0.2 and landmarks.get("mouth_corner_lift", 0) > 0.02:
            cues.append(_cue("smile", frame_index, evidence_id,
                             {"mouth_width_ratio": round(width / float(box[2] - box[0] or 1), 3),
                              "corner_lift": landmarks.get("mouth_corner_lift")}))
    brows = landmarks.get("brows")
    if brows and len(brows) >= 2 and landmarks.get("brow_baseline_available") is not False:
        gap = float(np.hypot(*(np.array(brows[0]) - np.array(brows[1]))))
        eye_gap = float(landmarks.get("eye_gap", 0) or 0)
        if eye_gap and gap / eye_gap < 0.5:
            cues.append(_cue("brow_furrow", frame_index, evidence_id,
                             {"brow_gap_ratio": round(gap / eye_gap, 3)}))
    eyes = landmarks.get("eyes")
    if eyes and len(eyes) == 2:
        ratios = []
        for eye in eyes:
            points = np.asarray(eye, dtype=float)
            if points.shape != (6, 2) or not np.isfinite(points).all():
                ratios = []
                break
            width = float(np.linalg.norm(points[0] - points[3]))
            if width < 6:
                ratios = []
                break
            opening = float(np.linalg.norm(points[1] - points[5]) +
                            np.linalg.norm(points[2] - points[4]))
            ratios.append(opening / (2 * width))
        if len(ratios) == 2 and all(ratio <= 0.18 for ratio in ratios):
            kind = "eyes_closed" if all(ratio <= 0.10 for ratio in ratios) else "eyes_narrowed"
            candidate = _cue(kind, frame_index, evidence_id,
                             {"eye_aspect_ratios": [round(ratio, 3) for ratio in ratios],
                              "threshold": 0.10 if kind == "eyes_closed" else 0.18,
                              "measurement": "single_frame_not_blink_duration"})
            candidate["uncertainty"] = "未校准眼睑几何候选，低头或视线可影响结果；单帧不能判定眨眼或内心状态"
            cues.append(candidate)
    return {"status": "success", "frame_index": frame_index, "cues": cues,
            "uncertainty": "几何启发式线索，不表示真实情绪"}


def _cue(kind: str, frame_index: str, evidence_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "frame_index": frame_index,
            "evidence_ids": [evidence_id], "metrics": metrics,
            "score": 0.6, "score_kind": "rule_score",
            "uncertainty": "表情线索不代表内心状态"}


def cues_to_claims(cues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = {"smile": "嘴角上扬", "brow_furrow": "眉头收紧", "mouth_open": "嘴部张开",
             "eyes_closed": "双眼接近闭合（单帧候选）",
             "eyes_narrowed": "双眼睑间距较小（单帧候选）"}
    return [{"claim_type": "observable",
             "statement": f"可观察表情线索：{names.get(cue['kind'], cue['kind'])}",
             "evidence_ids": list(cue.get("evidence_ids", [])),
             "method": {"id": "expressive-cues-v1", "version": "1"},
             "score": cue["score"], "score_kind": cue["score_kind"],
             "uncertainty_reason": cue.get("uncertainty")} for cue in cues]


def acoustic_claim(measurement: Dict[str, Any], evidence_id: Optional[str]) -> List[Dict[str, Any]]:
    if measurement.get("status") != "success":
        return []
    return [{"claim_type": "measured",
             "statement": "分析窗口内的音量与基频测量",
             "evidence_ids": [evidence_id] if evidence_id else [],
             "method": {"id": measurement["method"], "version": "1",
                        "window": measurement["window_seconds"]},
             "measurement": {"rms_dbfs": measurement["rms_dbfs"],
                             "f0_hz": measurement["f0_hz"],
                             "active_ratio": measurement["active_ratio"],
                             "energy_active_ratio": measurement["energy_active_ratio"],
                             "activity_kind": measurement["activity_kind"]},
             "score": 1.0, "score_kind": "confidence",
             "uncertainty_reason": "仅声学测量，不代表音色、身份或情绪"}]
