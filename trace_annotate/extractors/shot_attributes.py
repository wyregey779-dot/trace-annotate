"""Shot framing (景别) from subject geometry, reported as evidence not truth.

The classifier consumes subject boxes that other extractors already produced
(person boxes, or face boxes when that is all the project has). It never
declares a close-up because a face is large: a face-only measurement is marked
as such, a subject cut by the frame edge is uncertain, no subject means
"not applicable", and disagreeing subjects mean "multi-subject disagreement"
rather than an average.

Geometry rule (subject extent divided by frame height):

* < 0.10 远景, 0.10-0.30 全景, 0.30-0.50 中景, 0.50-0.80 近景, >= 0.80 特写.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

FRAMING_CLASSES = ("远景", "全景", "中景", "近景", "特写",
                   "多主体不一致", "不适用", "不确定")

RATIO_BANDS = ((0.10, "远景"), (0.30, "全景"), (0.50, "中景"), (0.80, "近景"))


def _band(ratio: float) -> str:
    for limit, name in RATIO_BANDS:
        if ratio < limit:
            return name
    return "特写"


def _clipped(box: Sequence[float], frame_width: int, frame_height: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= 0 or y1 <= 0 or x2 >= frame_width or y2 >= frame_height


def classify_framing(boxes: Sequence[Sequence[float]], frame_size: Sequence[int], *,
                     subject_kind: str = "person") -> Dict[str, Any]:
    """Classify one frame's framing from its subject boxes."""
    if len(frame_size) < 2:
        raise ValueError("frame_size must be (width, height)")
    frame_width, frame_height = int(frame_size[0]), int(frame_size[1])
    if not boxes:
        return {"framing": "不适用", "reason": "画面内没有可用主体框",
                "confidence": 0.0, "subjects": 0,
                "uncertainty": "未检测到主体不等于画面中不存在主体"}
    measured: List[Dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        height = max(0.0, y2 - y1)
        ratio = height / float(frame_height) if frame_height else 0.0
        measured.append({"box": [round(value, 1) for value in (x1, y1, x2, y2)],
                         "height_ratio": round(ratio, 3),
                         "framing": _band(ratio),
                         "clipped": _clipped((x1, y1, x2, y2), frame_width, frame_height)})
    distinct = sorted({item["framing"] for item in measured})
    if len(distinct) > 1:
        return {"framing": "多主体不一致", "subjects": len(measured),
                "confidence": 0.4, "measurements": measured,
                "uncertainty": "不同主体占据的画面比例差异过大，无法给出单一景别"}
    framing = distinct[0]
    clipped = any(item["clipped"] for item in measured)
    uncertainty = None
    confidence = 0.85
    if subject_kind == "face":
        confidence = 0.55
        uncertainty = "仅有人脸框，景别由脸占比推出，需用人物框复核"
    if clipped:
        confidence = min(confidence, 0.5)
        uncertainty = "主体被画幅裁切，占比不可信"
    return {"framing": framing, "subjects": len(measured),
            "confidence": confidence, "measurements": measured,
            "uncertainty": uncertainty, "subject_kind": subject_kind}


def summarise_framing(frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Dominant framing over a window plus the transitions between classes."""
    if not frames:
        return {"status": "not_run", "framing": None, "changes": []}
    classified = []
    for entry in frames:
        result = classify_framing(entry.get("boxes") or [], entry["frame_size"],
                                  subject_kind=entry.get("subject_kind", "person"))
        result.update({"frame_index": str(entry["frame_index"]),
                       "seconds": entry.get("seconds")})
        classified.append(result)
    usable = [item for item in classified if item["framing"] not in {"不适用", "不确定"}]
    if not usable:
        return {"status": "not_run", "framing": None,
                "reason": "窗口内没有可判定的主体框", "frames": classified, "changes": []}
    counts: Dict[str, int] = {}
    for item in usable:
        counts[item["framing"]] = counts.get(item["framing"], 0) + 1
    dominant = max(counts, key=counts.get)
    changes = []
    for index in range(1, len(classified)):
        previous, current = classified[index - 1], classified[index]
        if previous["framing"] != current["framing"]:
            changes.append({"from": previous["framing"], "to": current["framing"],
                            "at_frame": current["frame_index"],
                            "seconds": current["seconds"]})
    return {"status": "success", "framing": dominant,
            "share": round(counts[dominant] / len(usable), 3),
            "frames": classified, "changes": changes}


def framing_to_claims(result: Dict[str, Any], evidence_id_for) -> List[Dict[str, Any]]:
    """Framing is a measured geometry claim, never a scene interpretation."""
    claims = []
    for item in result.get("frames", []):
        if item["framing"] in {"不适用", "不确定"}:
            continue
        evidence = evidence_id_for(item["frame_index"])
        claims.append({"claim_type": "measured",
                       "statement": f"该帧按主体占比判定为{item['framing']}",
                       "evidence_ids": [evidence] if evidence else [],
                       "method": {"id": "shot-attributes-v1", "version": "1",
                                  "parameters": {"ratio_bands": RATIO_BANDS}},
                       "measurement": {"value": item["framing"],
                                       "unit": "framing_class",
                                       "subject_ratio": item["measurements"][0]["height_ratio"]},
                       "score": item["confidence"], "score_kind": "rule_score",
                       "uncertainty_reason": item.get("uncertainty")})
    return claims
