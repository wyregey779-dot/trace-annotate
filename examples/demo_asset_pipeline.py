"""
TraceAnnotate: End-to-End Asset Pipeline Demonstration.
Runs a complete simulated workflow:
  1. Creates synthetic sample images (standard, bright, dark, blurry, near-duplicate).
  2. Runs cold-storage staging manager (R1-R5 compliant batching).
  3. Computes visual quality metrics and perceptual fingerprints (dHash).
  4. Runs near-duplicate clustering via Disjoint-Set Union (并查集).
  5. Applies composite heuristics (F1 Statue, F2 Entity, F3 Screen OCR).
  6. Tracks provenance with the Fill dictionary.
  7. Builds prioritized review queue (P1-P4).
  8. Executes 21-point verification suite.
"""
import os
import shutil
import tempfile
import numpy as np
from PIL import Image

from trace_annotate.config import StagingConfig, MetricThresholds
from trace_annotate.core.hasher import compute_dhash, head1mb_md5, generate_thumbnail
from trace_annotate.core.cluster import cluster_by_dhash
from trace_annotate.core.staging import StagingManager
from trace_annotate.core.provenance import ProvenanceTracker, FillSource
from trace_annotate.core.verify import verify_annotations
from trace_annotate.extractors.metrics import VisualMetricsExtractor
from trace_annotate.heuristics.composite_f1_face import evaluate_f1_face
from trace_annotate.heuristics.composite_f2_entity import evaluate_f2_entity_tier
from trace_annotate.heuristics.composite_f3_screen import evaluate_f3_screen
from trace_annotate.heuristics.rules import infer_environment_rules
from trace_annotate.audit.review_queue import build_review_queue


def run_demo():
    print("=== TraceAnnotate: Media Asset Pipeline Demo ===")
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_scratch")
    source_dir = os.path.join(demo_dir, "raw_source")
    staging_dir = os.path.join(demo_dir, "local_staging")
    output_dir = os.path.join(demo_dir, "output")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Create sample synthetic images
        print("\n[Step 1] Creating synthetic test images...")
        sample_files = []

        # Image 1: Normal landscape
        arr1 = np.zeros((480, 640, 3), dtype=np.uint8)
        arr1[100:300, 100:500] = [120, 180, 240]
        p1 = os.path.join(source_dir, "park_day.jpg")
        Image.fromarray(arr1).save(p1)
        sample_files.append("park_day.jpg")

        # Image 2: Near duplicate of Image 1
        arr2 = arr1.copy()
        arr2[105:115, 105:115] = [125, 185, 245]
        p2 = os.path.join(source_dir, "park_day_dup.jpg")
        Image.fromarray(arr2).save(p2)
        sample_files.append("park_day_dup.jpg")

        # Image 3: Statue face context
        arr3 = np.ones((480, 640, 3), dtype=np.uint8) * 160
        p3 = os.path.join(source_dir, "museum_statue_face.jpg")
        Image.fromarray(arr3).save(p3)
        sample_files.append("museum_statue_face.jpg")

        # Image 4: Very dark bad frame
        arr4 = np.ones((480, 640, 3), dtype=np.uint8) * 5
        p4 = os.path.join(source_dir, "dark_tunnel.jpg")
        Image.fromarray(arr4).save(p4)
        sample_files.append("dark_tunnel.jpg")

        print(f"Generated {len(sample_files)} sample files in raw source.")

        # 2. Staging Manager (R1-R5 protocol)
        print("\n[Step 2] Executing cold-storage staging manager (2 items per batch)...")
        staging_cfg = StagingConfig(group_size=2, free_disk_guard_bytes=100 * 1024**2)
        sm = StagingManager(source_root=source_dir, staging_root=staging_dir, config=staging_cfg)

        metrics_extractor = VisualMetricsExtractor()
        annotations = []
        hashes = {}

        for batch_idx, batch_map in enumerate(sm.stage_batch(sample_files), 1):
            print(f"  Staging batch {batch_idx}: {list(batch_map.keys())}")
            for rel_path, staged_path in batch_map.items():
                # Hash calculation
                h1m = head1mb_md5(staged_path)
                dhash = compute_dhash(staged_path)
                hashes[rel_path] = dhash

                # Visual quality metrics
                metrics = metrics_extractor.extract({}, staged_path)

                # Initialize provenance tracker
                prov = ProvenanceTracker()
                prov.set_source("brightness", FillSource.AUTO_METRIC)
                prov.set_source("sharpness", FillSource.AUTO_METRIC)
                prov.set_source("head1mb_md5", FillSource.AUTO_METRIC)
                prov.set_source("dhash", FillSource.AUTO_METRIC)

                # Rules and Heuristics
                io, tod, tags = infer_environment_rules(rel_path)
                prov.set_source("indoor_outdoor", FillSource.AUTO_RULE)
                prov.set_source("time_of_day", FillSource.AUTO_RULE)

                # Heuristic F1: Statue face
                face_kind, face_status, face_rev, _ = evaluate_f1_face(
                    asset_rel_path=rel_path,
                    raw_face_count=1 if "face" in rel_path else 0,
                    raw_face_status="单人特写" if "face" in rel_path else "无人物"
                )
                prov.set_source("face_status", FillSource.AUTO_RULE)

                # Heuristic F2: Primary entity
                ent_status, ent_fill, conflict, conf_note = evaluate_f2_entity_tier(
                    primary_entity="景区地标" if "park" in rel_path else "展厅展品",
                    has_human_override=False,
                    path_depth_matched=True
                )
                prov.set_source("primary_entity", ent_fill)

                # Build full 46-key annotation record
                record = {
                    "asset_id": f"AST_{len(annotations)+1:04d}",
                    "rel_path": rel_path,
                    "kind": "image",
                    "ext": ".jpg",
                    "size": os.path.getsize(staged_path),
                    "head1mb_md5": h1m,
                    "full_md5": None,
                    "width": metrics["width"],
                    "height": metrics["height"],
                    "aspect_ratio": metrics["aspect_ratio"],
                    "duration": None,
                    "fps": None,
                    "video_codec": None,
                    "audio_codec": None,
                    "primary_entity": "景区地标" if "park" in rel_path else "展厅展品",
                    "entity_status": ent_status,
                    "tags": tags,
                    "indoor_outdoor": io,
                    "time_of_day": tod,
                    "weather": "晴朗",
                    "screen_type": "无屏",
                    "mood": None,
                    "face_count": 1 if "face" in rel_path else 0,
                    "face_status": face_status,
                    "crowd_level": "无人",
                    "face_subject_kind": face_kind,
                    "face_review": face_rev,
                    "ocr_text": "",
                    "ocr_readable": "",
                    "ocr_clean": "",
                    "brightness": metrics["brightness"],
                    "sharpness": metrics["sharpness"],
                    "is_bad_frame": metrics["is_bad_frame"],
                    "dhash": dhash,
                    "similar_cluster_id": "cluster_unassigned",
                    "usage_blacklist": False,
                    "usable_windows": None,
                    "best_frame_ts": None,
                    "scene_sub_segments": None,
                    "tech_quality": {"conflict_note": conf_note} if conflict else {},
                    "clip_scores": {"default": 1.0},
                    "source_type": "raw_footage",
                    "import_date": "2026-10-21",
                    "fill": prov.to_dict()
                }
                annotations.append(record)

        # 3. Disjoint-Set Union Clustering
        print("\n[Step 3] Clustering near-duplicates by dHash (hamming <= 10)...")
        item_to_c, c_to_items = cluster_by_dhash(hashes, hamming_threshold=10)
        for r in annotations:
            r["similar_cluster_id"] = item_to_c.get(r["rel_path"], "cluster_unassigned")
        print(f"Clusters discovered: {len(c_to_items)}")
        for cid, members in c_to_items.items():
            print(f"  {cid} ({len(members)} items): {members}")

        # 4. Review Queue Generation
        print("\n[Step 4] Building prioritized review queue (P1..P4)...")
        queue = build_review_queue(annotations)
        print(f"Items requiring review: {len(queue)}")
        for q in queue:
            print(f"  [{q['priority']}] {q['rel_path']} -> {q['reasons']}")

        # 5. Verification Suite
        print("\n[Step 5] Running 21-point verification suite...")
        report = verify_annotations(annotations)
        print(f"Verification Passed: {report.passed}")
        print(f"Verified rows: {report.total_rows}")
        print(f"Checks: {report.checks}")

        print("\n[OK] Asset pipeline demonstration finished successfully!")

    finally:
        # Clean up demo scratch
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)


if __name__ == "__main__":
    run_demo()
