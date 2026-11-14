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
import math
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ACTION_VOCAB = ("走", "跑", "拿起", "放下", "打开", "关闭", "递交")
ACTION_CANDIDATE_VERSION = "action-candidate-v3"
ACTION_REQUIREMENTS = {
    "走": ("subject_ref",),
    "跑": ("subject_ref",),
    "拿起": ("subject_ref", "object_ref"),
    "放下": ("subject_ref", "object_ref"),
    "打开": ("subject_ref", "object_ref"),
    "关闭": ("subject_ref", "object_ref"),
    "递交": ("subject_ref", "object_ref", "recipient_ref"),
}
ACTION_RESULT_STATUSES = ("success", "unknown", "not_run")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,95}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# provider name -> classes it declares support for, with the evidence it needs
ACTION_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "subject-translation-v1": {"supports": ("走", "跑"), "needs": "subject box track"},
    "kinetics400": {"supports": (), "needs": "video classifier",
                    "note": "Kinetics 类别不覆盖本项目词表，不能改名冒充"},
}

WALK_BAND = (0.15, 1.2)   # subject heights per second
RUN_BAND = (1.2, 6.0)
MAX_TRANSLATION_SAMPLES = 4096
MAX_ACTION_CANDIDATES = 4096
MAX_ACTION_CANDIDATE_BYTES = 16 * 1024 * 1024


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


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"INVALID_ACTION_REFERENCE: {field} must be an ASCII identifier")
    return value


def _rational(value: Any, field: str) -> Dict[str, str]:
    try:
        if isinstance(value, dict):
            if set(value) != {"n", "d"}:
                raise ValueError
            fraction = Fraction(int(value["n"]), int(value["d"]))
            canonical = {"n": str(fraction.numerator), "d": str(fraction.denominator)}
            if value != canonical:
                raise ValueError
        elif isinstance(value, bool):
            raise ValueError
        else:
            fraction = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"INVALID_ACTION_PTS: {field} must be a finite rational") from exc
    if fraction.denominator <= 0:
        raise ValueError(f"INVALID_ACTION_PTS: {field} denominator must be positive")
    return {"n": str(fraction.numerator), "d": str(fraction.denominator)}


def _window(value: Optional[Dict[str, Any]], field: str) -> Dict[str, Dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        raise ValueError(f"INVALID_ACTION_PTS: {field} requires start and end")
    start = _rational(value["start"], field + ".start")
    end = _rational(value["end"], field + ".end")
    if Fraction(int(start["n"]), int(start["d"])) >= Fraction(int(end["n"]), int(end["d"])):
        raise ValueError("INVALID_ACTION_PTS: action start must precede end")
    return {"start": start, "end": end}


def action_class_results(provider: str, *, status: str,
                         candidates: Sequence[Dict[str, Any]] = (),
                         reason: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return an explicit outcome for every frozen class.

    Absence of a candidate is ``unknown`` only when the provider actually ran.
    A class outside declared coverage is always ``not_run/provider_unavailable``.
    """
    if status not in ACTION_RESULT_STATUSES:
        raise ValueError("INVALID_ACTION_RESULT_STATUS")
    coverage = provider_coverage(provider)
    supported = set(coverage.get("supported") or ())
    counts = {name: 0 for name in ACTION_VOCAB}
    for candidate in candidates:
        action = candidate.get("action")
        if action in counts:
            counts[action] += 1
    results: Dict[str, Dict[str, Any]] = {}
    for action in ACTION_VOCAB:
        if action not in supported:
            outcome, why = "not_run", "provider_unavailable"
        elif status != "success":
            outcome, why = "not_run", reason or "provider_not_executed"
        elif counts[action]:
            outcome, why = "success", None
        else:
            outcome, why = "unknown", "no_candidate_met_threshold"
        results[action] = {
            "action": action,
            "status": outcome,
            "reason": why,
            "provider": provider if action in supported else None,
            "candidate_count": counts[action],
            "requires": list(ACTION_REQUIREMENTS[action]),
        }
    return results


def detect_translation_candidates(track: Sequence[Dict[str, Any]], *,
                                  subject_id: Optional[str] = None,
                                  subject_ref: Optional[str] = None,
                                  model_run_id: Optional[str] = None,
                                  min_samples: int = 3) -> Dict[str, Any]:
    """Measure a subject's box centre movement and emit walk/run candidates."""
    if (not isinstance(track, (list, tuple))
            or len(track) > MAX_TRANSLATION_SAMPLES):
        return {"status": "not_run", "candidates": [],
                "reason": "invalid_or_excessive_track_samples"}
    if (type(min_samples) is not int
            or not 2 <= min_samples <= MAX_TRANSLATION_SAMPLES):
        return {"status": "not_run", "candidates": [],
                "reason": "invalid_min_samples"}
    strict = subject_ref is not None or model_run_id is not None
    subject = subject_ref if subject_ref is not None else subject_id
    run_id = None
    if strict:
        try:
            subject = _identifier(subject, "subject_ref")
            run_id = _identifier(model_run_id, "model_run_id")
        except ValueError as exc:
            return {"status": "not_run", "candidates": [], "reason": str(exc)}
    points = []
    for entry in track:
        if not isinstance(entry, dict):
            continue
        if entry.get("box") is None or entry.get("frame_index") is None:
            continue
        try:
            seconds = _seconds(entry)
        except (TypeError, ValueError, OverflowError):
            continue
        if seconds is None:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in entry["box"]]
        except (TypeError, ValueError):
            continue
        if (not all(math.isfinite(value) for value in (seconds, x1, y1, x2, y2))
                or x2 <= x1 or y2 <= y1):
            continue
        raw_pts = entry.get("pts", entry.get("seconds"))
        try:
            pts = _rational(raw_pts, "sample.pts")
        except ValueError:
            continue
        digest = entry.get("evidence_sha256")
        if strict and (not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None):
            continue
        points.append({"frame_index": str(entry["frame_index"]), "seconds": seconds, "pts": pts,
                       "centre": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                       "height": max(1.0, y2 - y1),
                       "evidence_sha256": digest,
                       "detection_confidence": entry.get("detection_confidence"),
                       "association": entry.get("association")})
    if len(points) < min_samples:
        return {"status": "not_run", "candidates": [],
                "reason": "insufficient_evidence_samples"}
    try:
        frames = [int(point["frame_index"]) for point in points]
    except (TypeError, ValueError):
        return {"status": "not_run", "candidates": [],
                "reason": "non_monotonic_frame_or_pts_range"}
    if (any(str(frame) != str(point["frame_index"]) or frame < 0
            for frame, point in zip(frames, points))
            or any(points[index]["seconds"] >= points[index + 1]["seconds"]
                   or frames[index] >= frames[index + 1]
                   or Fraction(int(points[index]["pts"]["n"]),
                               int(points[index]["pts"]["d"]))
                   >= Fraction(int(points[index + 1]["pts"]["n"]),
                               int(points[index + 1]["pts"]["d"]))
                   for index in range(len(points) - 1))):
        return {"status": "not_run", "candidates": [],
                "reason": "non_monotonic_frame_or_pts_range"}
    if strict and any(abs(float(Fraction(int(point["pts"]["n"]),
                                         int(point["pts"]["d"])))
                          - point["seconds"]) > 1e-9 for point in points):
        return {"status": "not_run", "candidates": [],
                "reason": "pts_seconds_identity_mismatch"}
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
        candidate = {
            "kind": "action_candidate",
            "action": classification,
            "start_frame": points[0]["frame_index"],
            "end_frame": points[-1]["frame_index"],
            "window_seconds": {"start": points[0]["seconds"], "end": points[-1]["seconds"]},
            "metrics": {"speed_heights_per_s": round(speed, 3),
                        "dx": round(dx, 2), "dy": round(dy, 2),
                        "samples": len(points)},
            "sample_frames": [
                {key: point[key] for key in
                 ("frame_index", "seconds", "pts", "evidence_sha256",
                  "detection_confidence", "association")}
                for point in points],
            "method": "subject-translation-v1",
            "score": round(min(0.85, 0.4 + speed / 4.0), 3),
            "score_kind": "rule_score",
            "uncertainty": "仅由主体框位移推出，动作类别需人工确认",
        }
        if strict:
            candidate.update({
                "contract_version": ACTION_CANDIDATE_VERSION,
                "subject_ref": subject, "object_ref": None, "recipient_ref": None,
                "window_pts": {"start": points[0]["pts"], "end": points[-1]["pts"]},
                "provider": "subject-translation-v1", "model_run_id": run_id,
            })
        else:
            candidate.update({"subject_id": subject, "object": None})
        candidates.append(candidate)
    return {"status": "success", "candidates": candidates,
            ("subject_ref" if strict else "subject_id"): subject,
            "metrics": {"speed_heights_per_s": round(speed, 3),
                        "duration_seconds": round(duration, 3), "samples": len(points)}}


def build_candidate(action: str, *, subject_id: Optional[str], start_frame: str,
                    end_frame: str, evidence_ids: Sequence[str], score: float,
                    score_kind: str, provider: str, object_label: Optional[str] = None,
                    window_seconds: Optional[Dict[str, Any]] = None,
                    uncertainty: Optional[str] = None) -> Dict[str, Any]:
    """Build the backward-compatible v1/v2 candidate shape."""
    if action not in ACTION_VOCAB:
        raise ValueError(f"UNSUPPORTED_ACTION_CLASS: {action} 不在冻结词表内")
    coverage = provider_coverage(provider)
    if action not in coverage.get("supported", []):
        raise ValueError(f"PROVIDER_DOES_NOT_SUPPORT: {provider} 未声明支持 {action}")
    if not evidence_ids:
        raise ValueError("ACTION_REQUIRES_EVIDENCE")
    if (not isinstance(start_frame, str) or not start_frame.isdigit()
            or str(int(start_frame)) != start_frame
            or not isinstance(end_frame, str) or not end_frame.isdigit()
            or str(int(end_frame)) != end_frame
            or int(start_frame) > int(end_frame)):
        raise ValueError("INVALID_ACTION_FRAME_RANGE")
    if (not isinstance(evidence_ids, (list, tuple))
            or any(not isinstance(item, str) or _ID_RE.fullmatch(item) is None
                   for item in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)):
        raise ValueError("ACTION_REQUIRES_EVIDENCE")
    if score_kind not in {"similarity", "softmax", "rule_score", "confidence"}:
        raise ValueError(f"UNSUPPORTED_SCORE_KIND: {score_kind}")
    if isinstance(score, bool):
        raise ValueError("INVALID_ACTION_SCORE")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("INVALID_ACTION_SCORE") from exc
    lower = -1.0 if score_kind == "similarity" else 0.0
    if not math.isfinite(numeric_score) or not lower <= numeric_score <= 1.0:
        raise ValueError("INVALID_ACTION_SCORE")
    if window_seconds is not None:
        if (not isinstance(window_seconds, dict)
                or set(window_seconds) != {"start", "end"}):
            raise ValueError("INVALID_ACTION_TIME_RANGE")
        try:
            left, right = (float(window_seconds[key]) for key in ("start", "end"))
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_ACTION_TIME_RANGE") from exc
        if not math.isfinite(left) or not math.isfinite(right) or left >= right:
            raise ValueError("INVALID_ACTION_TIME_RANGE")
    return {"kind": "action_candidate", "action": action,
            "subject_id": subject_id, "object": object_label,
            "start_frame": start_frame, "end_frame": end_frame,
            "window_seconds": window_seconds, "evidence_ids": list(evidence_ids),
            "provider": provider, "score": round(numeric_score, 3),
            "score_kind": score_kind, "uncertainty": uncertainty}


def build_candidate_v3(action: str, *, subject_ref: str, start_frame: str,
                       end_frame: str, window_pts: Dict[str, Any],
                       evidence_ids: Sequence[str], score: float, score_kind: str,
                       provider: str, model_run_id: str,
                       object_ref: Optional[str] = None,
                       recipient_ref: Optional[str] = None,
                       uncertainty: str) -> Dict[str, Any]:
    """Build the evidence/provider-bound v3 candidate shape."""
    if action not in ACTION_VOCAB:
        raise ValueError(f"UNSUPPORTED_ACTION_CLASS: {action} 不在冻结词表内")
    subject = _identifier(subject_ref, "subject_ref")
    obj = _identifier(object_ref, "object_ref") if object_ref is not None else None
    recipient = _identifier(recipient_ref, "recipient_ref") if recipient_ref is not None else None
    required = ACTION_REQUIREMENTS[action]
    if "object_ref" in required and obj is None:
        raise ValueError("ACTION_REQUIRES_OBJECT")
    if "recipient_ref" in required and recipient is None:
        raise ValueError("ACTION_REQUIRES_RECIPIENT")
    if action == "递交" and recipient == subject:
        raise ValueError("ACTION_REQUIRES_DISTINCT_RECIPIENT")
    provider = _identifier(provider, "provider")
    run_id = _identifier(model_run_id, "model_run_id")
    coverage = provider_coverage(provider)
    if action not in coverage.get("supported", []):
        raise ValueError(f"PROVIDER_DOES_NOT_SUPPORT: {provider} 未声明支持 {action}")
    if not isinstance(evidence_ids, (list, tuple)) or not evidence_ids:
        raise ValueError("ACTION_REQUIRES_EVIDENCE")
    evidence = [_identifier(item, "evidence_id") for item in evidence_ids]
    if len(set(evidence)) != len(evidence):
        raise ValueError("ACTION_REQUIRES_EVIDENCE")
    if score_kind not in {"similarity", "softmax", "rule_score", "confidence"}:
        raise ValueError(f"UNSUPPORTED_SCORE_KIND: {score_kind}")
    if isinstance(score, bool):
        raise ValueError("INVALID_ACTION_SCORE")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("INVALID_ACTION_SCORE") from exc
    lower = -1.0 if score_kind == "similarity" else 0.0
    if not math.isfinite(numeric_score) or not lower <= numeric_score <= 1.0:
        raise ValueError("INVALID_ACTION_SCORE")
    if (not isinstance(start_frame, str) or not start_frame.isdigit()
            or str(int(start_frame)) != start_frame
            or not isinstance(end_frame, str) or not end_frame.isdigit()
            or str(int(end_frame)) != end_frame
            or int(start_frame) > int(end_frame)):
        raise ValueError("INVALID_ACTION_FRAME_RANGE")
    pts = _window(window_pts, "window_pts")
    if not isinstance(uncertainty, str) or not uncertainty.strip():
        raise ValueError("ACTION_REQUIRES_UNCERTAINTY")
    seconds = {key: float(Fraction(int(value["n"]), int(value["d"])))
               for key, value in pts.items()}
    return {"kind": "action_candidate", "contract_version": ACTION_CANDIDATE_VERSION,
            "action": action, "subject_ref": subject, "object_ref": obj,
            "recipient_ref": recipient, "start_frame": start_frame, "end_frame": end_frame,
            "window_seconds": seconds, "window_pts": pts, "evidence_ids": evidence,
            "provider": provider, "model_run_id": run_id, "score": round(numeric_score, 3),
            "score_kind": score_kind, "uncertainty": uncertainty}


def merge_candidates(candidates: Sequence[Dict[str, Any]], *,
                     max_pts_gap_seconds: float = 1.0) -> List[Dict[str, Any]]:
    """Merge continuous candidates without crossing subject/object/method identity.

    Inputs can arrive from sampled windows in any order.  Shared evidence alone is
    not proof of temporal continuity, and candidates without a subject identity
    cannot safely be assumed to describe the same actor.
    """

    if (not isinstance(candidates, (list, tuple))
            or len(candidates) > MAX_ACTION_CANDIDATES):
        raise ValueError("action candidate budget exceeded")
    if (isinstance(max_pts_gap_seconds, bool)
            or not isinstance(max_pts_gap_seconds, (int, float))
            or not math.isfinite(float(max_pts_gap_seconds))
            or float(max_pts_gap_seconds) < 0):
        raise ValueError("max_pts_gap_seconds must be a finite non-negative number")
    max_pts_gap = Fraction(str(float(max_pts_gap_seconds)))

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
            stable(candidate.get("model_run_id")),
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

    def pts_bounds(candidate: Dict[str, Any]) -> Optional[tuple[Fraction, Fraction]]:
        value = candidate.get("window_pts")
        if not isinstance(value, dict) or set(value) != {"start", "end"}:
            return None
        try:
            start = _rational(value["start"], "window_pts.start")
            end = _rational(value["end"], "window_pts.end")
            left = Fraction(int(start["n"]), int(start["d"]))
            right = Fraction(int(end["n"]), int(end["d"]))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return (left, right) if left < right else None

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
    candidate_bytes = 0
    for original in candidates:
        if not isinstance(original, dict):
            raise ValueError("action candidate must be an object")
        candidate_bytes += len(stable(original).encode("utf-8"))
        if candidate_bytes > MAX_ACTION_CANDIDATE_BYTES:
            raise ValueError("action candidate byte budget exceeded")
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
            candidate_pts = pts_bounds(candidate)
            current_pts = pts_bounds(current) if current is not None else None
            strict_pts = (candidate.get("contract_version") == ACTION_CANDIDATE_VERSION
                          or (current or {}).get("contract_version") == ACTION_CANDIDATE_VERSION)
            pts_continuous = (
                current is not None
                and candidate_pts is not None
                and current_pts is not None
                and candidate_pts[0] >= current_pts[0]
                and candidate_pts[1] >= candidate_pts[0]
                and candidate_pts[0] <= current_pts[1] + max_pts_gap
            )
            if (current is None or start > current_end + 1
                    or (strict_pts and not pts_continuous)
                    or (candidate_pts is not None and current_pts is not None and not pts_continuous)):
                current = candidate
                current_end = end
                merged.append(current)
                continue

            window = {
                "start_frame": candidate["start_frame"],
                "end_frame": candidate["end_frame"],
            }
            for key in ("window_seconds", "window_pts", "evidence_ids", "provider", "method",
                        "model_run_id", "score", "score_kind"):
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
            if end > previous_end and isinstance(current.get("window_pts"), dict) and isinstance(
                    candidate.get("window_pts"), dict) and "end" in candidate["window_pts"]:
                current["window_pts"]["end"] = copy.deepcopy(candidate["window_pts"]["end"])
            if isinstance(current.get("sample_frames"), list) and isinstance(candidate.get("sample_frames"), list):
                existing_samples = {(str(item.get("frame_index")), item.get("evidence_sha256"))
                                    for item in current["sample_frames"]}
                for sample in candidate["sample_frames"]:
                    marker = (str(sample.get("frame_index")), sample.get("evidence_sha256"))
                    if marker not in existing_samples:
                        current["sample_frames"].append(copy.deepcopy(sample))
                        existing_samples.add(marker)
                current["sample_frames"].sort(key=lambda item: int(item["frame_index"]))

    merged.extend(separate)
    merged.sort(key=lambda candidate: (
        frame_bounds(candidate) is None,
        frame_bounds(candidate) or (0, 0),
        identity(candidate),
        stable(candidate),
    ))
    return merged


def candidates_to_claims(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project legacy candidates without mislabelling an inference as observable."""
    claims = []
    for candidate in candidates:
        claims.append({"claim_type": "unspecified",
                       "statement": f"候选动作：{candidate['action']}",
                       "evidence_ids": list(candidate.get("evidence_ids", [])),
                       "method": {"id": candidate["method"] if "method" in candidate
                                  else candidate["provider"], "version": "1"},
                       "time_window": candidate.get("window_seconds"),
                       "subject_ref": candidate.get("subject_ref", candidate.get("subject_id")),
                       "object_ref": candidate.get("object_ref"),
                       "recipient_ref": candidate.get("recipient_ref"),
                       "model_run_id": candidate.get("model_run_id"),
                       "score": candidate["score"], "score_kind": candidate["score_kind"],
                       "uncertainty_reason": candidate.get("uncertainty")
                       or "候选动作需人工确认，不能直接视为已识别"})
    return claims
