"""Bounded, same-frame geometry between hand landmarks and object boxes.

The output describes pixels in sampled decoded frames. A zero distance or box
overlap does not establish touch, grasp, ownership, identity, or an action.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple


CONTRACT = "hand-object-frame-geometry-v1"
MAX_KEYFRAMES = 512
MAX_HANDS_PER_FRAME = 4
MAX_OBJECT_TRACKS = 16384
MAX_OBJECTS_PER_FRAME = 32
MAX_OBJECTS_PER_HAND = 3
MAX_PAIRS = 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,95}\Z")
_UNCERTAINTY = (
    "Sampled-frame image-plane geometry only; overlap or zero pixel distance "
    "does not establish contact, grasp, support, person identity, or action."
)


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded ASCII identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _frame(value: Any) -> str:
    if (not isinstance(value, str) or not value.isascii() or not value.isdecimal()
            or (len(value) > 1 and value[0] == "0")):
        raise ValueError("frame_index must be a canonical non-negative string")
    return value


def _pts(value: Any) -> Fraction:
    if (not isinstance(value, dict) or set(value) != {"n", "d"}
            or any(not isinstance(value[key], str) or len(value[key]) > 32
                   for key in ("n", "d"))):
        raise ValueError("PTS must be a reduced rational")
    try:
        result = Fraction(int(value["n"]), int(value["d"]))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("PTS must be a reduced rational") from exc
    if value != {"n": str(result.numerator), "d": str(result.denominator)}:
        raise ValueError("PTS must be a reduced rational")
    return result


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("geometry values must be finite numbers")
    result = float(value)
    if not math.isfinite(result) or abs(result) > 10_000_000:
        raise ValueError("geometry values must be finite, bounded numbers")
    return result


def _box(value: Any) -> Tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box_xyxy requires four finite values")
    left, top, right, bottom = (_number(item) for item in value)
    if right <= left or bottom <= top:
        raise ValueError("box_xyxy requires positive area")
    return left, top, right, bottom


def _point_to_box(point: Sequence[float], box: Tuple[float, ...]) -> float:
    x, y = point[:2]
    return math.hypot(max(box[0] - x, x - box[2], 0.0),
                      max(box[1] - y, y - box[3], 0.0))


def _box_gap(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    return math.hypot(max(left[0] - right[2], right[0] - left[2], 0.0),
                      max(left[1] - right[3], right[1] - left[3], 0.0))


def _iou(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    intersection = (max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
                    * max(0.0, min(left[3], right[3]) - max(left[1], right[1])))
    return intersection / (area(left) + area(right) - intersection)


def _source_counts(keyframes: Sequence[Dict[str, Any]],
                   object_sampling: Dict[str, Any]) -> Dict[str, Any]:
    hand_success = 0
    object_success = 0
    object_truncated = 0
    object_omissions = 0
    object_unknown = 0
    retained_objects = 0
    hand_observations = 0
    outside_hands = 0
    for frame in keyframes:
        hand = frame.get("hands") or {}
        obj = frame.get("object") or {}
        if hand.get("status") == "success":
            hand_success += 1
            observations = hand.get("hands")
            if isinstance(observations, list):
                hand_observations += len(observations)
                outside_hands += sum(item.get("outside_frame") is True
                                     for item in observations if isinstance(item, dict))
        if obj.get("status") == "success":
            object_success += 1
            object_truncated += obj.get("truncated") is True
            detections = obj.get("objects_detected")
            if isinstance(detections, list):
                retained_objects += len(detections)
            original_count = obj.get("detector_object_count")
            if (isinstance(detections, list) and type(original_count) is int
                    and original_count >= len(detections)):
                object_omissions += original_count - len(detections)
            else:
                object_unknown += 1
    return {
        "total_keyframes": len(keyframes),
        "hand_successful_keyframes": hand_success,
        "object_successful_keyframes": object_success,
        "object_truncated_keyframes": object_truncated,
        "object_detection_omissions": object_omissions,
        "object_detection_omissions_unknown_keyframes": object_unknown,
        "object_observations": retained_objects,
        "hand_observations": hand_observations,
        "skipped_outside_frame_hands": outside_hands,
        "sampled_module_coverage_complete": bool(keyframes)
        and hand_success == len(keyframes)
        and object_success == len(keyframes)
        and all((frame.get("object") or {}).get("truncated") is False
                for frame in keyframes)
        and object_omissions == 0
        and object_unknown == 0
        and object_sampling.get("status") == "success"
        and (object_sampling.get("tracking") or {}).get("status") == "success",
    }


def _empty(status: str, reason: str, counts: Dict[str, Any] | None = None) -> Dict[str, Any]:
    counts = counts or {}
    return {
        "contract": CONTRACT, "status": status, "reason": reason,
        "authority": "unreviewed_sampled_frame_pixel_geometry_only",
        "pairs": [],
        "coverage": {
            "total_keyframes": counts.get("total_keyframes", 0),
            "hand_successful_keyframes": counts.get("hand_successful_keyframes", 0),
            "object_successful_keyframes": counts.get("object_successful_keyframes", 0),
            "object_truncated_keyframes": counts.get("object_truncated_keyframes", 0),
            "object_detection_omissions": counts.get("object_detection_omissions", 0),
            "object_detection_omissions_unknown_keyframes": counts.get(
                "object_detection_omissions_unknown_keyframes", 0),
            "sampled_module_coverage_complete": counts.get(
                "sampled_module_coverage_complete", False),
            "hand_observations": counts.get("hand_observations", 0),
            "object_observations": counts.get("object_observations", 0),
            "eligible_object_observations": counts.get("eligible_object_observations", 0),
            "comparable_pairs": 0, "emitted_pairs": 0, "omitted_pairs": 0,
            "skipped_outside_frame_hands": counts.get("skipped_outside_frame_hands", 0),
            "skipped_ambiguous_tracks": counts.get("skipped_ambiguous_tracks", 0),
            "truncated": False,
        },
        "unsupported_assertions": ["touch", "grasp", "held_by", "pick_up",
                                   "put_down", "open", "close", "hand_over"],
    }


def build_hand_object_evidence(keyframes: Sequence[Dict[str, Any]],
                               object_sampling: Dict[str, Any]) -> Dict[str, Any]:
    """Measure co-visible hand and object geometry without inferring an action.

    Inputs are the validated keyframes and object sampling produced by one
    vision job. At most three nearest object boxes per hand are retained; all
    omitted comparisons are counted. No input is modified.
    """
    if not isinstance(keyframes, (list, tuple)) or len(keyframes) > MAX_KEYFRAMES:
        raise ValueError("keyframe budget exceeded")
    if not isinstance(object_sampling, dict):
        raise ValueError("object sampling must be an object")
    if any(not isinstance(frame, dict) for frame in keyframes):
        raise ValueError("keyframe is invalid")
    counts = _source_counts(keyframes, object_sampling)
    if not keyframes or not any("hands" in frame for frame in keyframes):
        return _empty("not_run", "hand_landmarks_unavailable", counts)
    if not any((frame.get("hands") or {}).get("status") == "success"
               for frame in keyframes):
        return _empty("not_run", "hand_landmarks_unavailable", counts)
    tracking = object_sampling.get("tracking") or {}
    if tracking.get("status") != "success":
        return _empty("not_run", "object_tracking_unavailable", counts)
    tracks = tracking.get("tracks")
    if not isinstance(tracks, list) or len(tracks) > MAX_OBJECT_TRACKS:
        raise ValueError("object track budget exceeded")

    objects: Dict[Tuple[int, str, Fraction, str], List[Dict[str, Any]]] = {}
    ambiguous = 0
    object_count = 0
    object_ids = set()
    for track in tracks:
        if not isinstance(track, dict) or type(track.get("scene_index")) is not int:
            raise ValueError("object track scene identity is invalid")
        if track.get("ambiguity") is True or track.get("termination_reason") == "ambiguous_association":
            ambiguous += 1
            continue
        if track.get("ambiguity") is not False:
            raise ValueError("object track ambiguity is invalid")
        observations = track.get("observations")
        if not isinstance(observations, list):
            raise ValueError("object track observations are invalid")
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError("object observation is invalid")
            if observation.get("scene_index") != track["scene_index"]:
                raise ValueError("object observation scene identity is invalid")
            observation_id = _identity(observation.get("observation_id"), "object observation_id")
            if observation_id in object_ids:
                raise ValueError("object observation_id is duplicated")
            object_ids.add(observation_id)
            evidence = _digest(observation.get("evidence_id"), "object evidence_id")
            key = (track["scene_index"], _frame(observation.get("frame_index")),
                   _pts(observation.get("pts")), evidence)
            objects.setdefault(key, []).append({
                "observation_id": observation_id,
                "box": _box(observation.get("box_xyxy")),
            })
            object_count += 1
            if len(objects[key]) > MAX_OBJECTS_PER_FRAME:
                raise ValueError("object frame budget exceeded")

    pairs: List[Dict[str, Any]] = []
    total_pairs = 0
    hand_count = 0
    outside = 0
    hand_ids = set()
    for frame in keyframes:
        if not isinstance(frame, dict):
            raise ValueError("keyframe is invalid")
        result = frame.get("hands") or {}
        if result.get("status") != "success":
            continue
        hands = result.get("hands")
        if not isinstance(hands, list) or len(hands) > MAX_HANDS_PER_FRAME:
            raise ValueError("hand frame budget exceeded")
        scene = frame.get("scene_index")
        if type(scene) is not int or scene < 0:
            raise ValueError("hand scene identity is invalid")
        key = (scene, _frame(frame.get("frame_index")),
               _pts(frame.get("timestamp")),
               _digest(frame.get("evidence_sha256"), "hand frame evidence_sha256"))
        for hand in hands:
            if not isinstance(hand, dict):
                raise ValueError("hand observation is invalid")
            hand_count += 1
            if (hand.get("scene_index") != scene
                    or _frame(hand.get("frame_index")) != key[1]
                    or _pts(hand.get("pts")) != key[2]
                    or _digest(hand.get("evidence_sha256"), "hand evidence_sha256") != key[3]):
                raise ValueError("hand observation differs from its decoded frame")
            hand_id = _identity(hand.get("observation_id"), "hand observation_id")
            if hand_id in hand_ids:
                raise ValueError("hand observation_id is duplicated")
            hand_ids.add(hand_id)
            if hand.get("outside_frame") is True:
                outside += 1
                continue
            if hand.get("outside_frame") is not False:
                raise ValueError("hand outside_frame is invalid")
            landmarks = hand.get("landmarks_pixel")
            if not isinstance(landmarks, list) or len(landmarks) != 21:
                raise ValueError("hand requires 21 pixel landmarks")
            points = []
            for landmark in landmarks:
                if not isinstance(landmark, (list, tuple)) or len(landmark) != 3:
                    raise ValueError("hand pixel landmark is invalid")
                points.append(tuple(_number(value) for value in landmark))
            hand_box = _box(hand.get("box_xyxy"))
            if any(not (hand_box[0] - 0.001 <= point[0] <= hand_box[2] + 0.001
                        and hand_box[1] - 0.001 <= point[1] <= hand_box[3] + 0.001)
                   for point in points):
                raise ValueError("hand landmarks differ from their pixel box")
            ranked = []
            for obj in objects.get(key, []):
                box = obj["box"]
                distances = [_point_to_box(point, box) for point in points]
                landmark_index = min(range(21), key=lambda index: (distances[index], index))
                ranked.append((distances[landmark_index], obj["observation_id"],
                               landmark_index, box))
            total_pairs += len(ranked)
            for distance, object_id, landmark_index, box in sorted(ranked)[:MAX_OBJECTS_PER_HAND]:
                token = {"scene_index": scene, "frame_index": key[1],
                         "pts": frame["timestamp"], "evidence_sha256": key[3],
                         "hand_observation_id": hand_id,
                         "object_observation_id": object_id}
                stable = json.dumps(token, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode("ascii")
                pairs.append({
                    "id": "hog-" + hashlib.sha256(stable).hexdigest()[:40],
                    **token,
                    "nearest_hand_landmark_index": landmark_index,
                    "landmark_to_object_box_pixels": round(distance, 6),
                    "hand_box_to_object_box_pixels": round(_box_gap(hand_box, box), 6),
                    "box_iou": round(_iou(hand_box, box), 6),
                    "review_status": "unreviewed",
                    "uncertainty": _UNCERTAINTY,
                })

    pairs.sort(key=lambda item: (item["scene_index"], int(item["frame_index"]),
                                 item["hand_observation_id"],
                                 item["landmark_to_object_box_pixels"],
                                 item["object_observation_id"]))
    pairs = pairs[:MAX_PAIRS]
    counts.update(hand_observations=hand_count,
                  eligible_object_observations=object_count,
                  skipped_outside_frame_hands=outside,
                  skipped_ambiguous_tracks=ambiguous)
    result = _empty("success", "no_co_visible_observations" if not pairs else "geometry_only", counts)
    result["pairs"] = pairs
    result["coverage"].update(comparable_pairs=total_pairs, emitted_pairs=len(pairs),
                              omitted_pairs=total_pairs - len(pairs),
                              truncated=total_pairs > len(pairs))
    return result
