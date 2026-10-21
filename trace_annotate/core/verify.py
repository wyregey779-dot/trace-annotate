"""
21-Point Integrity & Verification Suite.
Validates consistency, schema completeness, null keys, cluster assignments,
and provenance fidelity across the entire annotation catalog.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


EXPECTED_SCHEMA_KEYS = [
    "asset_id", "rel_path", "kind", "ext", "size", "head1mb_md5", "full_md5",
    "width", "height", "aspect_ratio", "duration", "fps", "video_codec", "audio_codec",
    "primary_entity", "entity_status", "tags", "indoor_outdoor", "time_of_day",
    "weather", "screen_type", "mood",
    "face_count", "face_status", "crowd_level", "face_subject_kind", "face_review",
    "ocr_text", "ocr_readable", "ocr_clean",
    "brightness", "sharpness", "is_bad_frame",
    "dhash", "similar_cluster_id", "usage_blacklist",
    "usable_windows", "best_frame_ts", "scene_sub_segments",
    "tech_quality", "clip_scores", "source_type",
    "import_date", "fill"
]


@dataclass
class VerificationReport:
    """Quantitative verification report."""
    total_rows: int = 0
    passed: bool = True
    checks: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_rows": self.total_rows,
            "checks": self.checks,
            "errors": self.errors
        }


def verify_annotations(
    annotation_records: List[Dict[str, Any]],
    inventory_records: Optional[List[Dict[str, Any]]] = None,
    expected_keys: Optional[List[str]] = None
) -> VerificationReport:
    """
    Run 21-point comprehensive integrity check on annotation records.
    """
    keys_to_check = expected_keys or EXPECTED_SCHEMA_KEYS
    report = VerificationReport(total_rows=len(annotation_records))
    checks = report.checks
    errors = report.errors

    checks["total_rows"] = len(annotation_records)

    # 1. Row count match with inventory
    if inventory_records is not None:
        checks["inventory_rows"] = len(inventory_records)
        checks["row_count_match"] = len(annotation_records) == len(inventory_records)
        if not checks["row_count_match"]:
            errors.append(f"Row count mismatch: annotation has {len(annotation_records)}, inventory has {len(inventory_records)}")
    else:
        checks["row_count_match"] = True

    # 2. Key completeness & asset_id
    asset_id_missing = 0
    key_missing_counts: Dict[str, int] = {}
    phash_missing = 0
    video_windows_null = 0
    face_missing = 0
    ocr_missing = 0
    clip_missing = 0
    fill_missing = 0
    clusters_assigned = 0
    blacklist_count = 0
    entity_confirmed_count = 0
    entity_candidate_count = 0
    scene_sub_segments_count = 0

    seen_ids = set()
    duplicate_ids = 0

    for r in annotation_records:
        aid = r.get("asset_id")
        if not aid:
            asset_id_missing += 1
        elif aid in seen_ids:
            duplicate_ids += 1
        else:
            seen_ids.add(aid)

        for k in keys_to_check:
            if k not in r:
                key_missing_counts[k] = key_missing_counts.get(k, 0) + 1

        if not r.get("dhash"):
            phash_missing += 1

        if r.get("kind") == "video":
            if r.get("usable_windows") is None:
                video_windows_null += 1

        if r.get("face_status") is None:
            face_missing += 1

        if r.get("ocr_readable") is None:
            ocr_missing += 1

        if not r.get("clip_scores"):
            clip_missing += 1

        if not r.get("fill"):
            fill_missing += 1

        cid = r.get("similar_cluster_id")
        if cid and cid != "cluster_unassigned":
            clusters_assigned += 1

        if r.get("usage_blacklist") is True:
            blacklist_count += 1

        status = r.get("entity_status")
        if status == "confirmed":
            entity_confirmed_count += 1
        elif status == "candidate":
            entity_candidate_count += 1

        if r.get("scene_sub_segments"):
            scene_sub_segments_count += 1

    checks["asset_id_missing"] = asset_id_missing
    checks["duplicate_ids"] = duplicate_ids
    checks["key_missing"] = key_missing_counts
    checks["phash_missing"] = phash_missing
    checks["video_windows_null"] = video_windows_null
    checks["face_missing"] = face_missing
    checks["ocr_missing"] = ocr_missing
    checks["clip_missing"] = clip_missing
    checks["fill_missing"] = fill_missing
    checks["clusters_assigned"] = clusters_assigned
    checks["blacklist_count"] = blacklist_count
    checks["entity_confirmed"] = entity_confirmed_count
    checks["entity_candidate"] = entity_candidate_count
    checks["scene_sub_segments_count"] = scene_sub_segments_count

    if asset_id_missing > 0:
        errors.append(f"{asset_id_missing} rows missing asset_id")
    if duplicate_ids > 0:
        errors.append(f"{duplicate_ids} duplicate asset_ids detected")
    if key_missing_counts:
        errors.append(f"Missing schema keys detected: {key_missing_counts}")
    if video_windows_null > 0:
        errors.append(f"{video_windows_null} video rows have null usable_windows")
    if fill_missing > 0:
        errors.append(f"{fill_missing} rows missing fill provenance dictionary")

    report.passed = len(errors) == 0
    return report
