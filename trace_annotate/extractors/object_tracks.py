"""Conservative object tracks and pre-action relation evidence.

This module deliberately stops below the action layer.  A COCO box track can
support observations such as ``near`` or ``co_moving``; it cannot, by itself,
prove that a person picked up, put down, opened, closed or handed over an
object.  Callers must retain the returned uncertainty and obtain the missing
state/hand/support evidence before turning a relation into an action candidate.

All association is confined to one scene and one detector class.  Ambiguous
one-to-many or many-to-one matches split tracks instead of silently switching
identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .objects import COCO80_LABELS


TRACK_CONTRACT = "object-tracks-v1"
RELATION_CONTRACT = "object-relations-v1"
RELATIONS = ("near", "contact_candidate", "co_moving", "held_by", "control_transfer")
MAX_TRACK_OBSERVATIONS = 16384
MAX_TRACKS = 16384
MAX_SUBJECT_TRACKS = 1024
MAX_SAMPLES_PER_TRACK = 4096
MAX_RELATION_PAIRS = 16384
MAX_RELATIONS = 65536
MAX_RELATION_PAIR_SAMPLE_REFERENCES = 131072
MAX_RELATION_OUTPUT_SAMPLE_REFERENCES = 65536
MAX_RELATION_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_TRANSFER_COMPARISONS = 65536
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,95}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _identifier(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_stable(value).encode("utf-8")).hexdigest()[:40]
    return f"{prefix}-{digest}"


def _ascii_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded ASCII identifier")
    return value


def _evidence_identity(value: Dict[str, Any], field: str) -> str:
    if "evidence_id" in value:
        identity = value["evidence_id"]
        if (not isinstance(identity, str)
                or (_ID_RE.fullmatch(identity) is None and _HASH_RE.fullmatch(identity) is None)):
            raise ValueError(f"{field}.evidence_id must be a bounded ASCII identifier or lowercase SHA-256")
        return identity
    digest = value.get("evidence_sha256")
    if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} requires an ASCII evidence_id or lowercase SHA-256")
    return digest


def _frame(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("frame_index must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_index must be a non-negative integer") from exc
    if str(result) != str(value) and not isinstance(value, int):
        raise ValueError("frame_index must be a canonical integer")
    if result < 0:
        raise ValueError("frame_index must be a non-negative integer")
    return result


def _pts(value: Any) -> Dict[str, str]:
    try:
        if isinstance(value, dict) and set(value) == {"n", "d"}:
            result = Fraction(int(value["n"]), int(value["d"]))
            canonical = {"n": str(result.numerator), "d": str(result.denominator)}
            if value != canonical:
                raise ValueError
        elif isinstance(value, bool):
            raise ValueError
        else:
            result = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("PTS must be a finite rational") from exc
    return {"n": str(result.numerator), "d": str(result.denominator)}


def _pts_fraction(value: Dict[str, str]) -> Fraction:
    return Fraction(int(value["n"]), int(value["d"]))


def _box(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box_xyxy must contain four values")
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("box_xyxy values must be finite numbers") from exc
    if not all(math.isfinite(item) for item in box) or not box[2] > box[0] or not box[3] > box[1]:
        raise ValueError("box_xyxy must be a finite positive-area box")
    return box


def _observation(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("object observation must be an object")
    scene = value.get("scene_index")
    class_id = value.get("class_id")
    if type(scene) is not int or scene < 0:
        raise ValueError("scene_index must be a non-negative integer")
    if type(class_id) is not int or not 0 <= class_id < 80:
        raise ValueError("class_id must be a COCO80 index")
    label = value.get("class_label")
    if not isinstance(label, str) or not label or len(label) > 120:
        raise ValueError("class_label must be a bounded non-empty string")
    if label != COCO80_LABELS[class_id]:
        raise ValueError("class_label must match the frozen COCO-80 class_id")
    score = value.get("score")
    if (not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
        raise ValueError("score must be finite and between zero and one")
    if value.get("score_kind", "model_score_uncalibrated") != "model_score_uncalibrated":
        raise ValueError("object observation score_kind is not supported")
    evidence = _evidence_identity(value, "object observation")
    model_run_id = _ascii_id(value.get("model_run_id"), "model_run_id")
    normalized = {
        "scene_index": scene,
        "frame_index": str(_frame(value.get("frame_index"))),
        "pts": _pts(value.get("pts", value.get("seconds"))),
        "class_id": class_id,
        "class_label": label,
        "box_xyxy": _box(value.get("box_xyxy", value.get("box"))),
        "score": float(score),
        "score_kind": value.get("score_kind", "model_score_uncalibrated"),
        "evidence_id": evidence,
        "model_run_id": model_run_id,
    }
    token = {key: normalized[key] for key in (
        "scene_index", "frame_index", "pts", "class_id", "class_label",
        "box_xyxy", "evidence_id", "model_run_id")}
    supplied = value.get("observation_id")
    normalized["observation_id"] = (_identifier("objobs", token) if supplied is None
                                    else _ascii_id(supplied, "observation_id"))
    return normalized


def _area(box: Sequence[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _centre(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _normalised_centre_distance(left: Sequence[float], right: Sequence[float]) -> float:
    one, two = _centre(left), _centre(right)
    scale = max(1.0, math.sqrt((_area(left) + _area(right)) / 2.0))
    return math.hypot(one[0] - two[0], one[1] - two[1]) / scale


def _match_score(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float, float]:
    overlap = _iou(left, right)
    distance = _normalised_centre_distance(left, right)
    size_ratio = min(_area(left), _area(right)) / max(_area(left), _area(right))
    score = 0.60 * overlap + 0.25 * max(0.0, 1.0 - distance) + 0.15 * size_ratio
    return score, overlap, distance


def _new_track(observation: Dict[str, Any]) -> Dict[str, Any]:
    token = {"scene_index": observation["scene_index"],
             "class_id": observation["class_id"],
             "model_run_id": observation["model_run_id"],
             "first_observation_id": observation["observation_id"]}
    return {
        "contract": TRACK_CONTRACT,
        "track_id": _identifier("objtrk", token),
        "scene_index": observation["scene_index"],
        "class_id": observation["class_id"],
        "class_label": observation["class_label"],
        "model_run_id": observation["model_run_id"],
        "observations": [dict(observation)],
        "ambiguity": False,
        "ambiguity_reason": None,
        "termination_reason": None,
    }


def build_object_tracks(observations: Sequence[Dict[str, Any]], *,
                        max_gap_frames: int = 2, min_iou: float = 0.10,
                        max_centre_distance: float = 1.25,
                        ambiguity_margin: float = 0.05,
                        max_gap_seconds: float = 5.0) -> Dict[str, Any]:
    """Associate sparse COCO observations without crossing class/scene/time bounds."""
    if type(max_gap_frames) is not int or max_gap_frames < 0:
        raise ValueError("max_gap_frames must be a non-negative integer")
    if (not isinstance(observations, (list, tuple))
            or len(observations) > MAX_TRACK_OBSERVATIONS):
        raise ValueError("object observation budget exceeded")
    for name, value in (("min_iou", min_iou), ("max_centre_distance", max_centre_distance),
                        ("ambiguity_margin", ambiguity_margin),
                        ("max_gap_seconds", max_gap_seconds)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    normalized = [_observation(value) for value in observations]
    identities = [value["observation_id"] for value in normalized]
    if len(set(identities)) != len(identities):
        raise ValueError("observation_id values must be unique")
    normalized.sort(key=lambda item: (item["scene_index"], int(item["frame_index"]),
                                      item["class_id"], item["observation_id"]))
    frame_pts: Dict[Tuple[int, int], Fraction] = {}
    for observation in normalized:
        key = (observation["scene_index"], int(observation["frame_index"]))
        point = _pts_fraction(observation["pts"])
        prior = frame_pts.setdefault(key, point)
        if prior != point:
            raise ValueError("all observations of one scene frame must have the same PTS")
    for scene in sorted({key[0] for key in frame_pts}):
        ordered = sorted((frame, point) for (candidate, frame), point in frame_pts.items()
                         if candidate == scene)
        if any(right[0] <= left[0] or right[1] <= left[1]
               for left, right in zip(ordered, ordered[1:])):
            raise ValueError("scene frame and PTS values must be strictly increasing")
    tracks: List[Dict[str, Any]] = []
    active: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    grouped: Dict[Tuple[int, int, int, str], List[Dict[str, Any]]] = {}
    for observation in normalized:
        grouped.setdefault((observation["scene_index"], int(observation["frame_index"]),
                            observation["class_id"], observation["model_run_id"]), []).append(observation)

    previous_scene: Optional[int] = None
    for (scene, frame_index, class_id, model_run_id), frame_observations in sorted(grouped.items()):
        if previous_scene is not None and scene != previous_scene:
            for key, values in list(active.items()):
                if key[0] == previous_scene:
                    for track in values:
                        if track["termination_reason"] is None:
                            track["termination_reason"] = "hard_cut"
                    del active[key]
        previous_scene = scene
        key = (scene, class_id, model_run_id)
        candidates = active.setdefault(key, [])
        still_active = []
        current_pts = _pts_fraction(frame_observations[0]["pts"])
        for track in candidates:
            gap = frame_index - int(track["observations"][-1]["frame_index"])
            time_gap = current_pts - _pts_fraction(track["observations"][-1]["pts"])
            if time_gap > Fraction(str(max_gap_seconds)):
                track["termination_reason"] = "time_gap_exceeded"
            elif gap > max_gap_frames + 1:
                track["termination_reason"] = "gap_exceeded"
            else:
                still_active.append(track)
        candidates[:] = still_active

        edges: List[Tuple[float, str, str, Dict[str, Any], Dict[str, Any], float, float]] = []
        for track in candidates:
            last = track["observations"][-1]
            for observation in frame_observations:
                score, overlap, distance = _match_score(last["box_xyxy"], observation["box_xyxy"])
                if overlap >= min_iou or distance <= max_centre_distance:
                    edges.append((score, track["track_id"], observation["observation_id"],
                                  track, observation, overlap, distance))

        by_track: Dict[str, List[Tuple[float, str]]] = {}
        by_observation: Dict[str, List[Tuple[float, str]]] = {}
        for score, track_id, observation_id, *_ in edges:
            by_track.setdefault(track_id, []).append((score, observation_id))
            by_observation.setdefault(observation_id, []).append((score, track_id))

        ambiguous_tracks = set()
        ambiguous_observations = set()
        for track_id, values in by_track.items():
            ordered = sorted(values, key=lambda item: (-item[0], item[1]))
            if len(ordered) > 1 and ordered[0][0] - ordered[1][0] <= ambiguity_margin:
                ambiguous_tracks.add(track_id)
                ambiguous_observations.update(item[1] for item in ordered[:2])
        for observation_id, values in by_observation.items():
            ordered = sorted(values, key=lambda item: (-item[0], item[1]))
            if len(ordered) > 1 and ordered[0][0] - ordered[1][0] <= ambiguity_margin:
                ambiguous_observations.add(observation_id)
                ambiguous_tracks.update(item[1] for item in ordered[:2])

        for track in list(candidates):
            if track["track_id"] in ambiguous_tracks:
                track["ambiguity"] = True
                track["ambiguity_reason"] = "near_tied_association"
                track["termination_reason"] = "ambiguous_association"
                candidates.remove(track)

        used_tracks, used_observations = set(), set()
        for score, track_id, observation_id, track, observation, overlap, distance in sorted(
                edges, key=lambda item: (-item[0], item[1], item[2])):
            if (track_id in ambiguous_tracks or observation_id in ambiguous_observations
                    or track_id in used_tracks or observation_id in used_observations
                    or track not in candidates):
                continue
            appended = dict(observation)
            appended["association"] = {"method": "class-iou-centre-v1",
                                       "score": round(score, 6),
                                       "iou": round(overlap, 6),
                                       "centre_distance": round(distance, 6)}
            track["observations"].append(appended)
            used_tracks.add(track_id)
            used_observations.add(observation_id)

        for observation in frame_observations:
            if observation["observation_id"] in used_observations:
                continue
            created = _new_track(observation)
            if observation["observation_id"] in ambiguous_observations:
                created["ambiguity"] = True
                created["ambiguity_reason"] = "near_tied_association"
            tracks.append(created)
            candidates.append(created)

    for values in active.values():
        for track in values:
            if track["termination_reason"] is None:
                track["termination_reason"] = "end_of_scene"
    tracks.sort(key=lambda item: (item["scene_index"], int(item["observations"][0]["frame_index"]),
                                  item["class_id"], item["model_run_id"], item["track_id"]))
    for track in tracks:
        for observation in track["observations"]:
            observation["track_id"] = track["track_id"]
    return {"contract": TRACK_CONTRACT, "status": "success", "tracks": tracks,
            "observations": sum(len(item["observations"]) for item in tracks),
            "ambiguous_tracks": sum(bool(item["ambiguity"]) for item in tracks),
            "parameters": {"max_gap_frames": max_gap_frames,
                           "max_gap_seconds": float(max_gap_seconds),
                           "min_iou": float(min_iou),
                           "max_centre_distance": float(max_centre_distance),
                           "ambiguity_margin": float(ambiguity_margin)},
            "scope": "sparse_observation_single_scene_same_class_geometric_association_only"}


def _strictly_increasing(observations: Sequence[Dict[str, Any]], field: str) -> None:
    previous_frame: Optional[int] = None
    previous_pts: Optional[Fraction] = None
    for observation in observations:
        frame = int(observation["frame_index"])
        point = _pts_fraction(observation["pts"])
        if (previous_frame is not None
                and (frame <= previous_frame or point <= previous_pts)):
            raise ValueError(f"{field} frame and PTS values must be strictly increasing")
        previous_frame, previous_pts = frame, point


def _object_track(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("contract") != TRACK_CONTRACT:
        raise ValueError("object track has an unsupported contract")
    track_id = _ascii_id(value.get("track_id"), "track_id")
    model_run_id = _ascii_id(value.get("model_run_id"), "model_run_id")
    scene = value.get("scene_index")
    class_id = value.get("class_id")
    if type(scene) is not int or scene < 0:
        raise ValueError("object track scene_index must be non-negative")
    if type(class_id) is not int or not 0 <= class_id < len(COCO80_LABELS):
        raise ValueError("object track class_id must be a COCO-80 index")
    if value.get("class_label") != COCO80_LABELS[class_id]:
        raise ValueError("object track class_label must match class_id")
    raw_observations = value.get("observations")
    if (not isinstance(raw_observations, (list, tuple)) or not raw_observations
            or len(raw_observations) > MAX_SAMPLES_PER_TRACK):
        raise ValueError("object track requires observations")
    observations = []
    identities = set()
    for raw in raw_observations:
        observation = _observation(raw)
        if (observation["scene_index"] != scene or observation["class_id"] != class_id
                or observation["class_label"] != value["class_label"]
                or observation["model_run_id"] != model_run_id):
            raise ValueError("object track observations must retain scene and class identity")
        if observation["observation_id"] in identities:
            raise ValueError("object track observation_id values must be unique")
        identities.add(observation["observation_id"])
        supplied_track = raw.get("track_id")
        if supplied_track is not None and _ascii_id(supplied_track, "observation.track_id") != track_id:
            raise ValueError("object observation track_id does not match its track")
        observation["track_id"] = track_id
        observations.append(observation)
    _strictly_increasing(observations, "object track")
    ambiguity = value.get("ambiguity")
    if type(ambiguity) is not bool:
        raise ValueError("object track ambiguity must be boolean")
    return {"contract": TRACK_CONTRACT, "track_id": track_id,
            "scene_index": scene, "class_id": class_id,
            "class_label": value["class_label"], "observations": observations,
            "model_run_id": model_run_id,
            "ambiguity": ambiguity,
            "ambiguity_reason": value.get("ambiguity_reason"),
            "termination_reason": value.get("termination_reason")}


def _subject_track(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("subject track must be an object")
    subject_ref = _ascii_id(value.get("subject_ref"), "subject_ref")
    scene = value.get("scene_index")
    if type(scene) is not int or scene < 0:
        raise ValueError("subject scene_index must be non-negative")
    raw_observations = value.get("observations", [])
    if (not isinstance(raw_observations, (list, tuple))
            or len(raw_observations) > MAX_SAMPLES_PER_TRACK):
        raise ValueError("subject observations must be a sequence")
    observations = []
    identities = set()
    for raw in raw_observations:
        if not isinstance(raw, dict):
            raise ValueError("subject observation must be an object")
        evidence = _evidence_identity(raw, "subject observation")
        frame = str(_frame(raw.get("frame_index")))
        point = _pts(raw.get("pts", raw.get("seconds")))
        supplied = raw.get("observation_id")
        observation_id = (_identifier("subobs", {"subject_ref": subject_ref,
                                                  "scene_index": scene,
                                                  "frame_index": frame,
                                                  "pts": point,
                                                  "evidence_id": evidence})
                          if supplied is None else _ascii_id(supplied, "observation_id"))
        if observation_id in identities:
            raise ValueError("subject observation_id values must be unique")
        identities.add(observation_id)
        observations.append({"observation_id": observation_id,
                             "frame_index": frame, "pts": point,
                             "box_xyxy": _box(raw.get("box_xyxy", raw.get("box"))),
                             "evidence_id": evidence})
    _strictly_increasing(observations, "subject track")
    return {"subject_ref": subject_ref, "scene_index": scene,
            "observations": observations}


def _edge_gap(left: Sequence[float], right: Sequence[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    scale = max(1.0, math.sqrt(min(_area(left), _area(right))))
    return math.hypot(dx, dy) / scale


def _relation(kind: str, *, subject_ref: str, object_ref: str,
              samples: Sequence[Dict[str, Any]], score: float,
              recipient_ref: Optional[str] = None,
              uncertainty: str) -> Dict[str, Any]:
    first, last = samples[0], samples[-1]
    evidence = []
    for sample in samples:
        for item in sample.get("evidence_ids", []):
            if item not in evidence:
                evidence.append(item)
    return {"contract": RELATION_CONTRACT, "relation": kind,
            "subject_ref": subject_ref, "object_ref": object_ref,
            "recipient_ref": recipient_ref,
            "start_frame": first["frame_index"], "end_frame": last["frame_index"],
            "window_pts": {"start": first["pts"], "end": last["pts"]},
            "evidence_ids": evidence, "samples": list(samples),
            "score": round(float(score), 6), "score_kind": "rule_score",
            "review_status": "unreviewed", "uncertainty": uncertainty}


def _pair_samples(object_track: Dict[str, Any], subject: Dict[str, Any]) -> List[Dict[str, Any]]:
    subjects = {item["frame_index"]: item for item in subject["observations"]}
    result = []
    for obj in object_track["observations"]:
        person = subjects.get(obj["frame_index"])
        if person is None:
            continue
        if _pts_fraction(person["pts"]) != _pts_fraction(obj["pts"]):
            continue
        # Geometry is comparable only when both boxes were decoded from the
        # same immutable frame.  Equal frame numbers/PTS from different
        # evidence sources do not establish a shared pixel coordinate space.
        if person["evidence_id"] != obj["evidence_id"]:
            continue
        gap = _edge_gap(obj["box_xyxy"], person["box_xyxy"])
        result.append({"frame_index": obj["frame_index"], "pts": obj["pts"],
                       "object_box": obj["box_xyxy"], "subject_box": person["box_xyxy"],
                       "edge_gap": round(gap, 6),
                       "evidence_ids": [obj["evidence_id"]]})
    return result


def _co_motion(samples: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, float]]:
    if len(samples) < 3:
        return False, {"samples": float(len(samples))}
    obj_a, obj_b = _centre(samples[0]["object_box"]), _centre(samples[-1]["object_box"])
    sub_a, sub_b = _centre(samples[0]["subject_box"]), _centre(samples[-1]["subject_box"])
    ov, sv = (obj_b[0] - obj_a[0], obj_b[1] - obj_a[1]), (sub_b[0] - sub_a[0], sub_b[1] - sub_a[1])
    om, sm = math.hypot(*ov), math.hypot(*sv)
    if min(om, sm) < 2.0:
        return False, {"samples": float(len(samples)), "object_motion": om, "subject_motion": sm}
    cosine = (ov[0] * sv[0] + ov[1] * sv[1]) / (om * sm)
    ratio = min(om, sm) / max(om, sm)
    start_offset = math.dist(obj_a, sub_a)
    end_offset = math.dist(obj_b, sub_b)
    scale = max(1.0, math.sqrt(_area(samples[-1]["subject_box"])))
    drift = abs(end_offset - start_offset) / scale
    return cosine >= 0.85 and ratio >= 0.45 and drift <= 0.35, {
        "samples": float(len(samples)), "motion_cosine": cosine,
        "motion_ratio": ratio, "relative_offset_drift": drift}


def relation_evidence(object_tracks: Sequence[Dict[str, Any]],
                      subject_tracks: Sequence[Dict[str, Any]], *,
                      near_gap: float = 0.35, contact_gap: float = 0.08,
                      held_min_samples: int = 3) -> Dict[str, Any]:
    """Create relation evidence, never action labels, from aligned track samples."""
    for name, value in (("near_gap", near_gap), ("contact_gap", contact_gap)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative number")
    if contact_gap > near_gap:
        raise ValueError("contact_gap must not exceed near_gap")
    if type(held_min_samples) is not int or held_min_samples < 3:
        raise ValueError("held_min_samples must be at least three")
    if (not isinstance(object_tracks, (list, tuple))
            or len(object_tracks) > MAX_TRACKS):
        raise ValueError("object track budget exceeded")
    if (not isinstance(subject_tracks, (list, tuple))
            or len(subject_tracks) > MAX_SUBJECT_TRACKS):
        raise ValueError("subject track budget exceeded")
    if len(object_tracks) * len(subject_tracks) > MAX_RELATION_PAIRS:
        raise ValueError("object/subject relation pair budget exceeded")
    objects = [_object_track(value) for value in object_tracks]
    subjects = [_subject_track(value) for value in subject_tracks]
    if (sum(len(item["observations"]) for item in objects)
            + sum(len(item["observations"]) for item in subjects)
            > MAX_TRACK_OBSERVATIONS * 2):
        raise ValueError("relation observation budget exceeded")
    if len({item["track_id"] for item in objects}) != len(objects):
        raise ValueError("object track_id values must be unique")
    if len({item["subject_ref"] for item in subjects}) != len(subjects):
        raise ValueError("subject_ref values must be unique")
    relations: List[Dict[str, Any]] = []
    pair_sample_references = 0
    output_sample_references = 0
    output_bytes = 0

    def emit(value: Dict[str, Any]) -> None:
        nonlocal output_sample_references, output_bytes
        if len(relations) >= MAX_RELATIONS:
            raise ValueError("relation output budget exceeded")
        samples = value.get("samples")
        if not isinstance(samples, list):
            raise ValueError("relation samples are invalid")
        output_sample_references += len(samples)
        if output_sample_references > MAX_RELATION_OUTPUT_SAMPLE_REFERENCES:
            raise ValueError("relation sample output budget exceeded")
        output_bytes += len(_stable(value).encode("utf-8"))
        if output_bytes > MAX_RELATION_OUTPUT_BYTES:
            raise ValueError("relation serialized output budget exceeded")
        relations.append(value)
    pair_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for obj in objects:
        if obj.get("contract") != TRACK_CONTRACT or obj.get("ambiguity"):
            continue
        for subject in subjects:
            if subject["scene_index"] != obj.get("scene_index"):
                continue
            pair_sample_references += min(
                len(obj["observations"]), len(subject["observations"]))
            if pair_sample_references > MAX_RELATION_PAIR_SAMPLE_REFERENCES:
                raise ValueError("relation pair sample budget exceeded")
            samples = _pair_samples(obj, subject)
            if not samples:
                continue
            near = [item for item in samples if item["edge_gap"] <= near_gap]
            contact = [item for item in samples if item["edge_gap"] <= contact_gap]
            moving, metrics = _co_motion(samples)
            state = {"samples": samples, "near": near, "contact": contact,
                     "co_moving": moving, "metrics": metrics,
                     "held": (moving and len(contact) >= held_min_samples
                              and len(contact) / len(samples) >= 0.60)}
            pair_state[(obj["track_id"], subject["subject_ref"])] = state
            if near:
                emit(_relation("near", subject_ref=subject["subject_ref"],
                    object_ref=obj["track_id"], samples=near,
                    score=len(near) / len(samples),
                    uncertainty="Box distance is only a proximity observation; it does not prove touch or intent."))
            if contact:
                emit(_relation("contact_candidate", subject_ref=subject["subject_ref"],
                    object_ref=obj["track_id"], samples=contact,
                    score=len(contact) / len(samples),
                    uncertainty="Overlapping/near boxes are not hand contact and may be occlusion or projection."))
            if moving:
                emit(_relation("co_moving", subject_ref=subject["subject_ref"],
                    object_ref=obj["track_id"], samples=samples,
                    score=max(0.0, min(1.0, metrics["motion_cosine"] * metrics["motion_ratio"])),
                    uncertainty="Similar image-plane motion does not prove that the subject controls the object."))
            if state["held"]:
                emit(_relation("held_by", subject_ref=subject["subject_ref"],
                    object_ref=obj["track_id"], samples=contact,
                    score=(len(contact) / len(samples)) * max(0.0, metrics["motion_cosine"]),
                    uncertainty="A held-by candidate still lacks hand pose, support and occlusion verification."))

    # Transfer is reported only when one subject is held/contacting early, is
    # later released for at least two observed frames, and a distinct subject
    # is held/contacting late.  Mere simultaneous proximity never qualifies.
    transfer_comparisons = 0
    for obj in objects:
        if obj.get("contract") != TRACK_CONTRACT or obj.get("ambiguity"):
            continue
        people = [subject for subject in subjects if subject["scene_index"] == obj.get("scene_index")]
        for giver in people:
            giver_state = pair_state.get((obj["track_id"], giver["subject_ref"]))
            if not giver_state or not giver_state["held"]:
                continue
            giver_contact_frames = {int(item["frame_index"]) for item in giver_state["contact"]}
            all_frames = [int(item["frame_index"]) for item in giver_state["samples"]]
            for recipient in people:
                if recipient["subject_ref"] == giver["subject_ref"]:
                    continue
                transfer_comparisons += 1
                if transfer_comparisons > MAX_TRANSFER_COMPARISONS:
                    raise ValueError("relation transfer comparison budget exceeded")
                recipient_state = pair_state.get((obj["track_id"], recipient["subject_ref"]))
                if not recipient_state or not recipient_state["held"]:
                    continue
                recipient_contact_frames = {int(item["frame_index"]) for item in recipient_state["contact"]}
                if not giver_contact_frames or not recipient_contact_frames:
                    continue
                transfer_frame = min(recipient_contact_frames)
                early = [frame for frame in giver_contact_frames if frame < transfer_frame]
                released = [frame for frame in all_frames if frame >= transfer_frame
                            and frame not in giver_contact_frames]
                late_recipient = [frame for frame in recipient_contact_frames if frame >= transfer_frame]
                if len(early) < 2 or len(released) < 2 or len(late_recipient) < 2:
                    continue
                sample_by_frame = {int(item["frame_index"]): item for item in
                                   giver_state["samples"] + recipient_state["samples"]}
                selected = [sample_by_frame[frame] for frame in sorted(set(
                    early[-2:] + released[:2] + late_recipient[:2]))]
                emit(_relation("control_transfer", subject_ref=giver["subject_ref"],
                    recipient_ref=recipient["subject_ref"], object_ref=obj["track_id"],
                    samples=selected, score=0.5,
                    uncertainty="This is a geometric control-transfer candidate; hand grip and intentional hand-over remain unverified."))

    relations.sort(key=lambda item: (int(item["start_frame"]), item["relation"],
                                     item["object_ref"], item["subject_ref"],
                                     item.get("recipient_ref") or ""))
    return {"contract": RELATION_CONTRACT,
            "status": "success" if relations else "unknown",
            "reason": None if relations else "insufficient_evidence",
            "relations": relations,
            "scope": "geometric_relation_candidates_not_action_recognition",
            "unsupported_action_assertions": ["pick_up", "put_down", "open", "close", "hand_over"]}
