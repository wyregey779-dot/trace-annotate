"""Import observed annotations as typed, evidence-gated reference candidates.

This is not an adjudicator. Shot starts are unknown until a typed decision with
hashed boundary evidence exists. Existing annotations, approximate times and
actor provenance are retained; imports never acquire human sign-off or an
unseen-holdout identity. Frame counts and timestamps come only from a supplied
decoded frame map, never FPS or container/audio duration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


KINDS = {"hard_cut", "dissolve", "unknown"}
SPLITS = ("observed_reference", "dev", "val", "observed_holdout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rational(value: Any) -> Fraction:
    if isinstance(value, dict) and "seconds" in value:
        value = value["seconds"]
    if isinstance(value, dict):
        return Fraction(int(value["n"]), int(value["d"]))
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    return Fraction(str(value))


def packed(value: Fraction) -> dict:
    return {"n": str(value.numerator), "d": str(value.denominator)}


def integer(value: Any) -> int:
    if isinstance(value, bool) or str(value) != str(int(value)):
        raise ValueError(f"not an integer: {value!r}")
    return int(value)


def annotation_parts(payload: dict) -> tuple:
    if "task" in payload and "media" in payload:
        media = payload["media"]
        return ([x for x in payload["task"]["annotations"] if x.get("track") == "Shots"],
                media["source"]["sha256"], media, {}, "professional")
    if "segments" in payload:
        prepared = payload.get("metadata", {}).get("prepared_media", {})
        timeline = prepared.get("timeline", {})
        indexes = payload.get("metadata", {}).get("assignment_review", {}).get("shot_frame_index", [])
        return ([x for x in payload["segments"] if x.get("track") == "镜头内容"],
                payload["media_sha256"], timeline, {x["id"]: x for x in indexes}, "temporal")
    raise ValueError("unsupported annotation schema")


def load_frame_map(path: Path, media: dict, layout: str) -> list[dict]:
    """Read ffprobe JSON or the existing admitted JSONL frame-map shape."""
    raw = path.read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = [json.loads(line) for line in raw.splitlines() if line.strip()]
    origin = rational(media.get("common_origin", media.get("origin_time_fraction", 0)))
    rows = []
    if isinstance(document, dict) and "streams" in document:
        streams = [s for s in document["streams"] if s.get("codec_type") == "video"]
        if len(streams) != 1:
            raise ValueError("frame probe must identify exactly one video stream")
        stream = streams[0]
        base = rational(stream["time_base"])
        if base <= 0:
            raise ValueError("invalid video time base")
        for index, item in enumerate(document["frames"]):
            if "stream_index" in item and item["stream_index"] != stream["index"]:
                raise ValueError("frame belongs to another stream")
            pts = integer(item.get("pts", item.get("best_effort_timestamp")))
            # Temporal documents bind the prepared playback copy. Its PTS are
            # already normalized, while professional documents bind the source.
            seconds = pts * base - origin if layout == "professional" else pts * base
            rows.append({"frame_index": index, "pts": pts, "time_base": packed(base),
                         "seconds": packed(seconds), "native": None})
        declared = stream.get("nb_frames")
        if declared not in (None, "N/A") and integer(declared) != len(rows):
            raise ValueError("decoded frame count disagrees with video stream nb_frames")
    else:
        entries = document.get("frames", []) if isinstance(document, dict) else document
        base = rational(media["proxy_time_base"])
        source_base = rational(media["source_time_base"])
        for index, item in enumerate(entries):
            if integer(item["frame_index"]) != index:
                raise ValueError("frame map indices must be contiguous decoded order")
            seconds = rational(item["seconds"])
            source_pts, proxy_pts = integer(item["source_pts"]), integer(item["proxy_pts"])
            if seconds != proxy_pts * base or abs(source_pts * source_base - origin - seconds) > min(base, source_base):
                raise ValueError("native frame map time-domain mismatch")
            rows.append({"frame_index": index, "pts": source_pts if layout == "professional" else proxy_pts,
                         "time_base": packed(source_base if layout == "professional" else base),
                         "seconds": packed(seconds), "native": dict(item)})
    if not rows:
        raise ValueError("empty decoded frame map")
    times = [rational(row["seconds"]) for row in rows]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("decoded frame PTS must be strictly increasing")
    if media.get("frame_count") is not None and integer(media["frame_count"]) != len(rows):
        raise ValueError("annotation and decoded map frame counts disagree")
    return rows


def locate(value: Any, rows: list[dict], layout: str, declared_index: Any = None) -> tuple:
    seconds = rational(value)
    anchor = value.get("anchor") if isinstance(value, dict) else None
    if anchor is not None:
        index = integer(anchor["frame_index"])
        if not 0 <= index < len(rows):
            raise ValueError("annotation anchor outside decoded map")
        row = rows[index]
        native = row["native"]
        fields = ("source_pts", "proxy_pts") if native else ("source_pts" if layout == "professional" else "proxy_pts",)
        for field in fields:
            expected = integer(native[field]) if native else row["pts"]
            if integer(anchor[field]) != expected:
                raise ValueError("annotation anchor PTS disagrees with decoded frame map")
        if seconds != rational(row["seconds"]):
            raise ValueError("annotation anchor seconds disagree with decoded frame map")
        return row, "native_anchor_verified"
    if declared_index is not None:
        index = integer(declared_index)
        if not 0 <= index < len(rows):
            return None, "declared_index_outside_map"
        row = rows[index]
        # Legacy JSON stores six-place seconds beside its explicit frame index.
        if abs(seconds - rational(row["seconds"])) > Fraction(1, 2_000_000):
            raise ValueError("legacy frame index disagrees with timestamp")
        return row, "legacy_index_verified_against_map"
    matches = [row for row in rows if rational(row["seconds"]) == seconds]
    return (matches[0], "exact_pts_match") if matches else (None, "unanchored_time_not_exact_pts")


def declared_kind(item: dict) -> tuple:
    decision = item.get("transition_decision", {})
    kind = decision.get("kind", item.get("transition_type", item.get("attributes", {}).get("transition_type")))
    if kind is not None:
        if kind not in KINDS:
            return "unknown", "unsupported_declared_type"
        return kind, "explicit_transition_type"
    label = str(item.get("label", "")).lower()
    if "dissolve" in label or "叠化" in label:
        return "dissolve", "annotation_label_requires_adjudication"
    return "unknown", "shot_start_is_not_a_transition_decision"


def decision_evidence(item: dict, row: dict | None, root: Path, video_sha: str, map_sha: str) -> tuple:
    decision = item.get("transition_decision", {})
    actor = decision.get("actor", {})
    if decision.get("kind") not in KINDS - {"unknown"} or decision.get("kind") != declared_kind(item)[0]:
        return [], "reviewed_decision_does_not_confirm_transition_type"
    if decision.get("status") != "reviewed" or actor.get("type") not in ("agent", "human") or not actor.get("id"):
        return [], "missing_explicit_reviewed_transition_decision"
    if actor.get("type") == "human" and (actor.get("identity_verification") != "verified" or "codex" in str(actor["id"]).lower()):
        return [], "unverified_human_actor"
    evidence = []
    for entry in decision.get("evidence", []):
        if entry.get("source_sha256") != video_sha or entry.get("frame_map_sha256") != map_sha:
            raise ValueError("transition evidence source or frame-map identity mismatch")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = root / path
        digest = entry.get("sha256", "").lower()
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("transition evidence missing or hash mismatch")
        evidence.append({**entry, "path": str(path.resolve()), "sha256": digest})
    expected = {row["frame_index"] - 1, row["frame_index"]} if row else set()
    actual = {integer(entry["frame_index"]) for entry in evidence}
    if not expected or min(expected) < 0 or not expected.issubset(actual):
        return evidence, "missing_before_and_current_frame_evidence"
    return evidence, None


def build_reference(annotation: Path, video: Path, frame_map: Path, frame_map_source_sha256: str,
                    source_id: str, split: str = "observed_reference") -> dict:
    if split not in SPLITS:
        raise ValueError("only explicitly observed reference splits are supported")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    items, expected_sha, media, legacy_indexes, layout = annotation_parts(payload)
    actual_sha = sha256(video)
    if actual_sha != str(expected_sha).lower():
        raise ValueError("video bytes do not match annotation media SHA256")
    if actual_sha != frame_map_source_sha256.lower():
        raise ValueError("frame-map source declaration does not match video SHA256")
    rows = load_frame_map(frame_map, media, layout)
    map_sha = sha256(frame_map)
    events = []
    ids = set()
    for item in sorted(items, key=lambda x: rational(x["start"])):
        identity = item.get("id")
        if not isinstance(identity, str) or identity in ids:
            raise ValueError("annotations require unique string ids")
        ids.add(identity)
        if rational(item["end"]) <= rational(item["start"]):
            raise ValueError("annotation end must follow start")
        legacy = legacy_indexes.get(identity, {})
        row, basis = locate(item["start"], rows, layout, legacy.get("start_frame"))
        end_row, end_basis = locate(item["end"], rows, layout)
        kind, type_basis = declared_kind(item)
        tags = item.get("tags", [])
        approximate = any("approximate" in str(tag) for tag in tags) or bool(item.get("approximate", False))
        evidence, evidence_reason = decision_evidence(item, row, annotation.parent, actual_sha, map_sha)
        reasons = []
        if row is None:
            reasons.append(basis)
        elif row["frame_index"] == 0:
            reasons.append("clip_start_is_not_a_transition")
        if kind == "unknown" or type_basis != "explicit_transition_type":
            reasons.append(type_basis)
        if approximate:
            reasons.append("approximate_boundary_requires_adjudication")
        if evidence_reason:
            reasons.append(evidence_reason)
        if kind == "dissolve":
            # An interval needs its own transition-specific adjudication; a pair
            # of stills cannot establish a complete gradual-transition interval.
            reasons.append("dissolve_interval_requires_separate_adjudication")
        eligible = not reasons
        events.append({
            "event_id": f"reference-{source_id}-{identity}", "annotation_id": identity,
            "kind": kind, "type_basis": type_basis,
            "adjudication_status": "confirmed" if eligible else "pending",
            "review_status": "agent_reviewed" if eligible and item["transition_decision"]["actor"]["type"] == "agent" else "unreviewed",
            "human_review": False, "evaluation_eligible": eligible,
            "ineligible_reasons": reasons, "approximate": approximate,
            "frame_index": row["frame_index"] if row else None,
            "pts": row["pts"] if row else None,
            "time_base": row["time_base"] if row else None,
            "seconds": row["seconds"] if row else None, "frame_identity_basis": basis,
            "declared_interval": {"start": item["start"], "end": item["end"]},
            "interval_end_frame": end_row["frame_index"] if end_row else None,
            "interval_end_identity_basis": end_basis,
            "source_annotation": item, "verified_decision_evidence": evidence,
            "legacy_frame_index": legacy or None,
        })
    return {
        "schema": "trace-annotate.reference-candidates.v1", "source_id": source_id,
        "split": split, "previously_observed": True, "holdout_sealed": False,
        "human_review": False, "authority": "imported_annotations_require_adjudication",
        "annotation": {"path": str(annotation.resolve()), "sha256": sha256(annotation), "layout": layout},
        "video": {"path": str(video.resolve()), "sha256": actual_sha},
        "frame_map": {"path": str(frame_map.resolve()), "sha256": map_sha,
                      "declared_source_sha256": frame_map_source_sha256.lower(),
                      "binding_basis": "caller_declared_source_sha256_and_verified_video_bytes",
                      "decoded_frame_count": len(rows), "first_frame": rows[0], "last_frame": rows[-1]},
        "counts": dict(Counter(event["kind"] for event in events)),
        "evaluation_eligible_count": sum(event["evaluation_eligible"] for event in events),
        "events": events,
        "limitations": ["Import does not perform visual adjudication or establish independent human sign-off.",
                        "Frame-map provenance is caller-declared; the importer does not decode video to authenticate map pixels.",
                        "No negative samples or complete transition coverage are inferred from shot intervals.",
                        "Dissolve intervals remain pending until a transition-specific adjudication workflow exists."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame-map", required=True, type=Path)
    parser.add_argument("--frame-map-source-sha256", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--split", choices=SPLITS, default="observed_reference")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_reference(args.annotation, args.video, args.frame_map,
                                 args.frame_map_source_sha256, args.source_id, args.split)
        # Preserve historical evidence: this tool never overwrites an existing output.
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"reference import failed: {exc}\n")
    print(json.dumps({"out": str(args.out), "counts": result["counts"],
                      "evaluation_eligible_count": result["evaluation_eligible_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
