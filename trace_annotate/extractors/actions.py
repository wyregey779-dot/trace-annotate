"""Action candidates from a frozen vocabulary and declared provider coverage.

The project vocabulary is 走, 跑, 拿起, 放下, 打开, 关闭, 递交. No provider is
assumed to cover them: a provider must declare which classes it actually
supports, and a class outside that declaration returns ``not_run`` instead of
being renamed into a neighbouring Kinetics label.

What this module can produce on its own is a *movement candidate* measured from a
subject box track with real PTS: translation speed and direction are measured, the
walk/run band is a rule with an explicit uncertainty, and the result is a
candidate for review, never a confirmed action. Overlapping actions are allowed;
merging requires the same subject, the same class and continuous evidence.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ACTION_VOCAB = ("走", "跑", "拿起", "放下", "打开", "关闭", "递交")

# provider name -> classes it declares support for, with the evidence it needs
ACTION_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "subject-translation-v1": {"supports": ("走", "跑"), "needs": "subject box track"},
    "kinetics400": {"supports": (), "needs": "video classifier",
                    "note": "Kinetics 类别不覆盖本项目词表，不能改名冒充"},
}

WALK_BAND = (0.15, 1.2)   # subject heights per second
RUN_BAND = (1.2, 6.0)


def provider_coverage(provider: str) -> Dict[str, Any]:
    if provider not in ACTION_PROVIDERS:
        return {"status": "not_run", "provider": provider,
                "reason": "provider 未登记，不能假设其支持词表"}
    declared = ACTION_PROVIDERS[provider]["supports"]
    missing = [name for name in ACTION_VOCAB if name not in declared]
    return {"status": "success", "provider": provider,
            "supported": list(declared), "unsupported": missing,
            "reason": None if not missing else "词表未全覆盖，未覆盖类别必须返回 not_run"}


def _seconds(sample: Dict[str, Any]) -> Optional[float]:
    value = sample.get("seconds")
    if isinstance(value, dict):
        denominator = float(value.get("d", 1) or 1)
        return float(value.get("n", 0)) / denominator if denominator else None
    return None if value is None else float(value)


def detect_translation_candidates(track: Sequence[Dict[str, Any]], *,
                                  subject_id: Optional[str] = None,
                                  min_samples: int = 3) -> Dict[str, Any]:
    """Measure a subject's box centre movement and emit walk/run candidates."""
    points = []
    for entry in track:
        if entry.get("box") is None or entry.get("frame_index") is None:
            continue
        seconds = _seconds(entry)
        if seconds is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in entry["box"]]
        points.append({"frame_index": str(entry["frame_index"]), "seconds": seconds,
                       "centre": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                       "height": max(1.0, y2 - y1),
                       "evidence_sha256": entry.get("evidence_sha256"),
                       "detection_confidence": entry.get("detection_confidence"),
                       "association": entry.get("association")})
    if len(points) < min_samples:
        return {"status": "not_run", "candidates": [],
                "reason": "主体框轨迹点不足，无法测量位移"}
    duration = points[-1]["seconds"] - points[0]["seconds"]
    if duration <= 0:
        return {"status": "not_run", "candidates": [],
                "reason": "轨迹时间跨度为零，PTS 不可用"}
    dx = points[-1]["centre"][0] - points[0]["centre"][0]
    dy = points[-1]["centre"][1] - points[0]["centre"][1]
    mean_height = float(np.mean([point["height"] for point in points]))
    speed = float(np.hypot(dx, dy) / mean_height / duration)
    candidates = []
    classification = None
    if WALK_BAND[0] <= speed < RUN_BAND[0]:
        classification = "走"
    elif RUN_BAND[0] <= speed <= RUN_BAND[1]:
        classification = "跑"
    if classification:
        candidates.append({
            "kind": "action_candidate",
            "action": classification,
            "subject_id": subject_id,
            "start_frame": points[0]["frame_index"],
            "end_frame": points[-1]["frame_index"],
            "window_seconds": {"start": points[0]["seconds"], "end": points[-1]["seconds"]},
            "metrics": {"speed_heights_per_s": round(speed, 3),
                        "dx": round(dx, 2), "dy": round(dy, 2),
                        "samples": len(points)},
            "sample_frames": [
                {key: point[key] for key in
                 ("frame_index", "seconds", "evidence_sha256",
                  "detection_confidence", "association")}
                for point in points],
            "method": "subject-translation-v1",
            "score": round(min(0.85, 0.4 + speed / 4.0), 3),
            "score_kind": "rule_score",
            "uncertainty": "仅由主体框位移推出，动作类别需人工确认",
            "object": None,
        })
    return {"status": "success", "candidates": candidates,
            "subject_id": subject_id,
            "metrics": {"speed_heights_per_s": round(speed, 3),
                        "duration_seconds": round(duration, 3), "samples": len(points)}}


def build_candidate(action: str, *, subject_id: Optional[str], start_frame: str,
                    end_frame: str, evidence_ids: Sequence[str], score: float,
                    score_kind: str, provider: str, object_label: Optional[str] = None,
                    window_seconds: Optional[Dict[str, Any]] = None,
                    uncertainty: Optional[str] = None) -> Dict[str, Any]:
    """Validate one candidate against the frozen vocabulary and evidence rules."""
    if action not in ACTION_VOCAB:
        raise ValueError(f"UNSUPPORTED_ACTION_CLASS: {action} 不在冻结词表内")
    coverage = provider_coverage(provider)
    if action not in coverage.get("supported", []):
        raise ValueError(f"PROVIDER_DOES_NOT_SUPPORT: {provider} 未声明支持 {action}")
    if not evidence_ids:
        raise ValueError("ACTION_REQUIRES_EVIDENCE")
    if score_kind not in {"similarity", "softmax", "rule_score", "confidence"}:
        raise ValueError(f"UNSUPPORTED_SCORE_KIND: {score_kind}")
    return {"kind": "action_candidate", "action": action,
            "subject_id": subject_id, "object": object_label,
            "start_frame": start_frame, "end_frame": end_frame,
            "window_seconds": window_seconds, "evidence_ids": list(evidence_ids),
            "provider": provider, "score": round(float(score), 3),
            "score_kind": score_kind, "uncertainty": uncertainty}


def merge_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge continuous candidates without crossing subject/object/method identity.

    Inputs can arrive from sampled windows in any order.  Shared evidence alone is
    not proof of temporal continuity, and candidates without a subject identity
    cannot safely be assumed to describe the same actor.
    """

    def stable(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=str)

    def first(candidate: Dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in candidate:
                return candidate[name]
        return None

    def identity(candidate: Dict[str, Any]) -> tuple[str, ...]:
        return (
            stable(candidate.get("action")),
            stable(first(candidate, "subject_ref", "subject_id")),
            stable(first(candidate, "object_ref", "object_id", "object")),
            stable(first(candidate, "recipient_ref", "recipient_id",
                         "receiver_ref", "receiver_id")),
            stable(candidate.get("provider")),
            stable(candidate.get("method")),
            stable(candidate.get("score_kind")),
        )

    def frame_bounds(candidate: Dict[str, Any]) -> Optional[tuple[int, int]]:
        try:
            start = int(candidate["start_frame"])
            end = int(candidate["end_frame"])
        except (KeyError, TypeError, ValueError):
            return None
        return (start, end) if end >= start else None

    def evidence(candidate: Dict[str, Any]) -> List[Any]:
        values = candidate.get("evidence_ids", [])
        return list(values) if isinstance(values, (list, tuple)) else []

    def append_unique(target: List[Any], additions: Sequence[Any]) -> None:
        seen = {stable(value) for value in target}
        for value in additions:
            marker = stable(value)
            if marker not in seen:
                target.append(copy.deepcopy(value))
                seen.add(marker)

    valid: Dict[tuple[str, ...], List[tuple[int, int, Dict[str, Any]]]] = {}
    separate: List[Dict[str, Any]] = []
    for original in candidates:
        candidate = copy.deepcopy(original)
        bounds = frame_bounds(candidate)
        subject = first(candidate, "subject_ref", "subject_id")
        if bounds is None or subject is None:
            separate.append(candidate)
            continue
        valid.setdefault(identity(candidate), []).append((bounds[0], bounds[1], candidate))

    merged: List[Dict[str, Any]] = []
    for group_identity in sorted(valid):
        group = sorted(valid[group_identity], key=lambda item: (
            item[0], item[1], stable(item[2])))
        current: Optional[Dict[str, Any]] = None
        current_end = -1
        for start, end, candidate in group:
            if current is None or start > current_end + 1:
                current = candidate
                current_end = end
                merged.append(current)
                continue

            window = {
                "start_frame": candidate["start_frame"],
                "end_frame": candidate["end_frame"],
            }
            for key in ("window_seconds", "evidence_ids", "provider", "method",
                        "score", "score_kind"):
                if key in candidate:
                    window[key] = copy.deepcopy(candidate[key])
            current.setdefault("merged_windows", []).append(window)
            previous_end = current_end
            current_end = max(current_end, end)
            current["end_frame"] = str(current_end)
            current_evidence = evidence(current)
            append_unique(current_evidence, evidence(candidate))
            if current_evidence or "evidence_ids" in current or "evidence_ids" in candidate:
                current["evidence_ids"] = current_evidence
            try:
                current["score"] = max(float(current["score"]), float(candidate["score"]))
            except (KeyError, TypeError, ValueError):
                pass
            if end > previous_end and isinstance(current.get("window_seconds"), dict) and isinstance(
                    candidate.get("window_seconds"), dict):
                if "end" in candidate["window_seconds"]:
                    current["window_seconds"]["end"] = copy.deepcopy(
                        candidate["window_seconds"]["end"])

    merged.extend(separate)
    merged.sort(key=lambda candidate: (
        frame_bounds(candidate) is None,
        frame_bounds(candidate) or (0, 0),
        identity(candidate),
        stable(candidate),
    ))
    return merged


def candidates_to_claims(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """An action candidate is an observable claim that still needs a human."""
    claims = []
    for candidate in candidates:
        claims.append({"claim_type": "observable",
                       "statement": f"候选动作：{candidate['action']}",
                       "evidence_ids": list(candidate.get("evidence_ids", [])),
                       "method": {"id": candidate["method"] if "method" in candidate
                                  else candidate["provider"], "version": "1"},
                       "time_window": candidate.get("window_seconds"),
                       "score": candidate["score"], "score_kind": candidate["score_kind"],
                       "uncertainty_reason": candidate.get("uncertainty")
                       or "候选动作需人工确认，不能直接视为已识别"})
    return claims
