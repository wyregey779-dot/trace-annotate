"""
Comprehensive Reliability & Contract Test Suite (T01 - T14).
Validates security, atomicity, isolation, temporal precision, semantic consistency,
model status truthfulness, audit rollback, and provenance idempotency.
"""
from __future__ import annotations
import os
import tempfile
import pytest

from trace_annotate.core.staging import (
    StagingManager,
    resolve_safe_source_path,
    resolve_safe_staging_path
)
from trace_annotate.audit.overrides import OverrideManager
from trace_annotate.embodied.schema import ErrorType, Verdict, ClipStatus
from trace_annotate.embodied.validator import validate_embodied_doc
from trace_annotate.embodied.label_studio import parse_label_studio_export
from trace_annotate.embodied.exporters.cvat import export_to_cvat_xml
from trace_annotate.embodied.exporters.action_dataset import export_to_action_segmentation
from trace_annotate.embodied.differ import ProcedureSpec, diff_expected_steps
from trace_annotate.embodied.agreement import (
    calculate_segment_tiou,
    calculate_temporal_kappa,
    match_temporal_segments
)
from trace_annotate.extractors.base import ExtractorStatus
from trace_annotate.extractors.faces import FaceDetector
from trace_annotate.extractors.clip_engine import CLIPEngine
from trace_annotate.audit.provenance import ProvenanceTracker


def test_t01_safe_path_traversal_protection():
    """T01: Path traversal attacks must be strictly blocked with PermissionError."""
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as stg:
        # Normal relative path is allowed
        safe_src = resolve_safe_source_path(src, "clips/vid1.mp4")
        assert safe_src.startswith(os.path.realpath(src))

        # Dot-dot traversal escape
        with pytest.raises(PermissionError, match="Path traversal detected"):
            resolve_safe_source_path(src, "../../secret.txt")

        # Absolute path override
        with pytest.raises(PermissionError, match="Path traversal detected"):
            resolve_safe_staging_path(stg, "C:\\Windows\\System32\\calc.exe")

        with pytest.raises(PermissionError, match="Path traversal detected"):
            resolve_safe_staging_path(stg, "/etc/passwd")


def test_t02_atomic_write_and_recovery_overrides():
    """T02: Overrides must be saved atomically with .bak backup and revision tracking."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ov_mgr = OverrideManager(temp_dir)
        rel_path = "batch1/clip1.mp4"

        # Initial revision
        ov_mgr.record_override(
            rel_path=rel_path,
            field_name="entity_status",
            old_val="candidate",
            new_val="confirmed",
            author="lead_annotator",
            reason="verified on frame 35"
        )
        saved = ov_mgr.save_overrides(atomic=True)
        assert os.path.exists(saved)

        # Mutate to revision 2
        ov_mgr.record_override(
            rel_path=rel_path,
            field_name="entity_status",
            old_val="confirmed",
            new_val="candidate",
            author="auditor",
            reason="ambiguous face"
        )
        ov_mgr.save_overrides(atomic=True)

        history = ov_mgr.get_history(rel_path)
        assert len(history) == 2
        assert history[0]["changes"]["entity_status"]["new"] == "confirmed"
        assert history[1]["changes"]["entity_status"]["new"] == "candidate"

        # Backup file must exist
        bak_file = saved + ".bak"
        assert os.path.exists(bak_file)


def test_t03_staging_in_use_tracking_and_isolation():
    """T03: In-use staging files must not be deleted by background cleanup."""
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as stg:
        mgr = StagingManager(src, stg)
        test_file = os.path.join(stg, "active_task.mp4")
        with open(test_file, "w") as f:
            f.write("video content")

        mgr.lock_in_use("active_task.mp4")
        cleaned = mgr.cleanup_completed(dry_run=False)
        assert "active_task.mp4" not in cleaned
        assert os.path.exists(test_file)

        mgr.unlock_in_use("active_task.mp4")
        cleaned2 = mgr.cleanup_completed(dry_run=False)
        assert "active_task.mp4" in cleaned2
        assert not os.path.exists(test_file)


def test_t04_multi_framerate_consistency():
    """T04: CVAT export frame indices must map accurately across different FPS."""
    doc = {
        "clip": "motion.mp4",
        "working": [{"start": 1.0, "end": 2.5, "step_number": 1, "verdict": "对"}]
    }

    # At 30 FPS: start=30, end=75
    xml_30 = export_to_cvat_xml(doc, fps=30.0)
    assert 'frame="30"' in xml_30
    assert 'frame="75"' in xml_30

    # At 25 FPS: start=25, end=62 or 63 (2.5 * 25 = 62.5 -> 62)
    xml_25 = export_to_cvat_xml(doc, fps=25.0)
    assert 'frame="25"' in xml_25
    assert 'frame="62"' in xml_25 or 'frame="63"' in xml_25


def test_t05_invalid_timestamps_validation():
    """T05: Negative timestamps, inverted ranges, and overlapping working intervals must fail validation."""
    # Negative start
    bad_neg = {
        "clip": "neg.mp4",
        "clip_status": "正常",
        "working": [{"start": -0.5, "end": 2.0, "step_number": 1, "verdict": "对"}]
    }
    errs_neg = validate_embodied_doc(bad_neg)
    assert any("negative" in e for e in errs_neg)

    # Inverted range (start >= end)
    bad_inv = {
        "clip": "inv.mp4",
        "clip_status": "正常",
        "working": [{"start": 5.0, "end": 3.0, "step_number": 1, "verdict": "对"}]
    }
    errs_inv = validate_embodied_doc(bad_inv)
    assert any("strictly less than" in e for e in errs_inv)

    # Overlapping working intervals
    bad_overlap = {
        "clip": "overlap.mp4",
        "clip_status": "正常",
        "working": [
            {"start": 1.0, "end": 5.0, "step_number": 1, "verdict": "对"},
            {"start": 4.0, "end": 8.0, "step_number": 2, "verdict": "对"}
        ]
    }
    errs_overlap = validate_embodied_doc(bad_overlap)
    assert any("overlap in time" in e for e in errs_overlap)


def test_t06_label_studio_order_invariance_and_unlabelled_preservation():
    """T06: Label Studio parsing must be order-invariant and never hallucinate defaults for unlabelled attributes."""
    # choices before videorectangle
    ls_export = [
        {
            "id": 101,
            "data": {"video_url": "sample_video.mp4"},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "region_1",
                            "type": "choices",
                            "value": {"choices": ["错"]}
                        },
                        {
                            "id": "region_1",
                            "type": "choices",
                            "value": {"choices": ["用错物体"]}
                        },
                        {
                            "id": "region_1",
                            "type": "videorectangle",
                            "value": {"sequence": [{"time": 2.0}, {"time": 6.5}]}
                        }
                    ]
                }
            ]
        }
    ]

    docs = parse_label_studio_export(ls_export)
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc["working"]) == 1
    seg = doc["working"][0]
    assert seg["start"] == 2.0
    assert seg["end"] == 6.5
    assert seg["verdict"] == "错"
    assert seg["error_type"] == "用错物体"
    # step_number was NOT provided in export: must be None, NOT defaulted to 1!
    assert seg["step_number"] is None


def test_t07_action_segmentation_global_mapping_persistence():
    """T07: Exporter must preserve class index IDs across batches without mapping drift."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Batch 1 discovers step 1 and step 2
        batch1 = [
            {"clip": "c1.mp4", "working": [{"start": 0.0, "end": 2.0, "step_number": 1}]}
        ]
        mapping_path, _ = export_to_action_segmentation(batch1, temp_dir, fps=1.0)

        # Batch 2 exports step 2 and step 3 into the same directory
        batch2 = [
            {"clip": "c2.mp4", "working": [{"start": 0.0, "end": 2.0, "step_number": 2}]}
        ]
        export_to_action_segmentation(batch2, temp_dir, fps=1.0)

        # Verify mapping file retains stable IDs
        with open(mapping_path, "r", encoding="utf-8") as f:
            lines = [l.strip().split() for l in f if l.strip()]
        mapping_dict = {l[1]: int(l[0]) for l in lines}

        assert mapping_dict["idle"] == 0
        assert mapping_dict["step_1"] == 1
        assert mapping_dict["step_2"] == 2


def test_t08_temporal_vs_semantic_agreement_and_prevalence_paradox():
    """T08: High tIoU with conflicting verdicts/steps must flag semantic conflict, and Kappa handles prevalence paradox."""
    # Annotator A and B agree on time (0..10) but disagree on verdict
    records_a = {
        "clip1": {
            "working": [{"start": 0.0, "end": 10.0, "step_number": 1, "verdict": "对"}]
        }
    }
    records_b = {
        "clip1": {
            "working": [{"start": 0.0, "end": 10.0, "step_number": 1, "verdict": "错"}]
        }
    }

    res = calculate_temporal_kappa(records_a, records_b, sample_rate_hz=1.0)
    # Perfect temporal overlap
    assert res["mean_tiou"] == 1.0
    # Prevalence paradox detected: 100% of duration is working
    assert res["is_degenerate_single_class"] is True
    # Semantic conflict detected: verdict mismatch
    assert len(res["semantic_conflicts"]) == 1
    assert res["semantic_conflicts"][0]["type"] == "verdict_mismatch"
    assert res["verdict_agreement_rate"] == 0.0


def test_t09_invalid_and_out_of_range_step_detection():
    """T09: Step numbers outside procedure definition must be flagged as invalid, without causing artificial missing steps."""
    spec = ProcedureSpec(
        name="assembly",
        steps=[1, 2, 3, 4],
        step_names={1: "fetch", 2: "align", 3: "screw", 4: "inspect"}
    )

    # Annotated sequence has step 1, 999 (wild outlier), 3
    working = [
        {"step_number": 1},
        {"step_number": 999},
        {"step_number": 3}
    ]

    diff = diff_expected_steps(working, procedure=spec)
    assert diff["is_perfect_sequence"] is False
    assert len(diff["invalid_steps"]) == 1
    assert diff["invalid_steps"][0]["step_number"] == 999
    # Missing steps should only be 2 and 4 from the spec, NOT 2..998!
    assert diff["missing"] == [2, 4]


def test_t10_extractor_status_truthfulness():
    """T10: Uninitialized extractors must return NOT_RUN and None values instead of claiming negative detections."""
    # FaceDetector with no weights loaded
    detector = FaceDetector(weights_path=None)
    result = detector.detect_faces_in_video("dummy.mp4")
    assert result["status"] == ExtractorStatus.NOT_RUN.value
    assert result["face_count"] is None
    assert result["face_status"] is None

    # CLIPEngine with no model loaded
    clip = CLIPEngine(model_name="none")
    clip_res = clip.classify_scene("dummy.mp4", ["indoor", "outdoor"])
    assert clip_res["status"] == ExtractorStatus.NOT_RUN.value
    assert clip_res["top_category"] is None
    assert clip_res["scores"] == {}


def test_t11_action_segmentation_true_duration_tail_idle():
    """T11: True video duration must be respected, padding tail idle frames instead of truncating video length."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Video is 6.0 seconds long, but action ends at 2.0s
        docs = [
            {
                "clip": "short_action.mp4",
                "duration": 6.0,
                "working": [{"start": 0.0, "end": 2.0, "step_number": 1}]
            }
        ]
        _, gts = export_to_action_segmentation(docs, temp_dir, fps=1.0)
        assert len(gts) == 1
        with open(gts[0], "r", encoding="utf-8") as f:
            frames = [line.strip() for line in f]

        # Must have 6 frames total: 2 step_1 + 4 idle
        assert len(frames) == 6
        assert frames == ["step_1", "step_1", "idle", "idle", "idle", "idle"]


def test_t12_safe_rollback_preserves_full_audit_trail():
    """T12: Rollback restores target revision value while preserving all revision audit history."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ov_mgr = OverrideManager(temp_dir)
        rel_path = "clip_audit.mp4"

        # Revision 1
        ov_mgr.record_override(
            rel_path=rel_path,
            field_name="working_status",
            old_val="unlabeled",
            new_val="working",
            author="worker_1",
            reason="detected arm movement"
        )
        rev1_id = ov_mgr.get_history(rel_path)[0]["revision_id"]

        # Revision 2 (erroneous override)
        ov_mgr.record_override(
            rel_path=rel_path,
            field_name="working_status",
            old_val="working",
            new_val="idle",
            author="worker_2",
            reason="accidental toggle"
        )
        assert len(ov_mgr.get_history(rel_path)) == 2

        # Rollback to revision 1
        success = ov_mgr.rollback_revision(
            rel_path=rel_path,
            target_revision_id=rev1_id,
            author="qa_lead",
            reason="reverting accidental toggle"
        )
        assert success is True

        history = ov_mgr.get_history(rel_path)
        # History must contain all 3 revisions (append-only)
        assert len(history) == 3
        # Active value must be reverted to revision 1's new_val ("working")
        assert ov_mgr.get_effective_override(rel_path, "working_status") == "working"


def test_t13_staging_overlap_rejection():
    """T13: Staging directory overlap or nesting with source root must be rejected immediately."""
    with tempfile.TemporaryDirectory() as base_dir:
        # Same directory
        with pytest.raises(ValueError, match="Staging root cannot be identical"):
            StagingManager(base_dir, base_dir)

        # Staging inside source
        sub_stg = os.path.join(base_dir, "staging_subdir")
        os.makedirs(sub_stg, exist_ok=True)
        with pytest.raises(ValueError, match="Staging root cannot be inside source root"):
            StagingManager(base_dir, sub_stg)

        # Source inside staging
        with pytest.raises(ValueError, match="Source root cannot be inside staging root"):
            StagingManager(sub_stg, base_dir)


def test_t14_batch_provenance_idempotent_rerun():
    """T14: Provenance tracker accurately records input hashes and guarantees idempotent rerun tracking."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ProvenanceTracker(temp_dir)
        input_data = {"clip": "test1.mp4", "working": [{"start": 1.0, "end": 4.0}]}

        rec1 = tracker.record_stage(
            stage_name="label_studio_import",
            inputs={"raw_doc": input_data},
            outputs={"clean_doc": input_data},
            metadata={"version": "2.1"}
        )

        # Idempotent re-run on same inputs
        rec2 = tracker.record_stage(
            stage_name="label_studio_import",
            inputs={"raw_doc": input_data},
            outputs={"clean_doc": input_data},
            metadata={"version": "2.1"}
        )

        # Both records have identical input_hash
        assert rec1["input_hash"] == rec2["input_hash"]
        assert tracker.get_stage_record("label_studio_import") is not None
