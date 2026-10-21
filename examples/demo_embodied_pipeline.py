"""
TraceAnnotate: Embodied AI Annotation Pipeline Demonstration.
Demonstrates:
  1. Creation and validation of an egocentric task annotation document (Specification v2.1).
  2. Automatic idle complement interval generation.
  3. Workflow expected step reconciliation (detecting missing, out-of-order, duplicates).
  4. Inter-annotator temporal agreement and Cohen's Kappa score calculation.
  5. Label Studio task template export.
"""
import json
from trace_annotate.embodied.schema import (
    EmbodiedAnnotationDoc,
    WorkingSegment,
    MidSegment,
    Verdict,
    ErrorType,
)
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps
from trace_annotate.embodied.validator import validate_embodied_doc
from trace_annotate.embodied.agreement import calculate_temporal_kappa
from trace_annotate.embodied.label_studio import generate_label_studio_xml


def run_demo():
    print("=== TraceAnnotate: Embodied AI Temporal Annotation Demo ===")

    # 1. Create a structured embodied annotation document
    print("\n[Step 1] Constructing compliant EmbodiedAnnotationDoc (Specification v2.1)...")
    doc = EmbodiedAnnotationDoc(
        clip="assembly_task_wrist_001.mp4",
        text_description="1. 拾取左侧底板; 2. 安装支撑轴; 3. 拧紧固定螺母",
        clip_status="正常",
        working=[
            WorkingSegment(
                start=5.0,
                end=18.5,
                step_number=1,
                verdict=Verdict.CORRECT.value,
                mid_segments=[MidSegment(start=10.0, end=11.2, reason="idle")]
            ),
            WorkingSegment(
                start=22.0,
                end=35.0,
                step_number=3,  # Out of order! Step 3 executed before Step 2
                verdict=Verdict.WRONG.value,
                error_type=ErrorType.OUT_OF_ORDER.value,
                evidence="t=22s 第3步固定螺母出现在第2步支撑轴之前，未装轴先拧螺母"
            ),
            WorkingSegment(
                start=38.0,
                end=52.0,
                step_number=2,
                verdict=Verdict.CORRECT.value
            )
        ]
    )

    # 2. Compute timeline idle complement
    print("\n[Step 2] Computing automatic idle complement intervals...")
    total_clip_duration = 60.0
    working_dicts = [{"start": w.start, "end": w.end} for w in doc.working]
    idle_intervals = compute_idle_complement(working_dicts, total_duration=total_clip_duration)
    print(f"Working segments count: {len(doc.working)}")
    print(f"Generated {len(idle_intervals)} idle intervals:")
    for idx, interval in enumerate(idle_intervals, 1):
        print(f"  Idle #{idx}: [{interval['start']}s -> {interval['end']}s]")

    # 3. Reconcile expected steps S = [1, 2, 3]
    print("\n[Step 3] Reconciling expected workflow steps (S = [1, 2, 3])...")
    diff_report = diff_expected_steps(
        [{"step_number": w.step_number} for w in doc.working],
        expected_step_count=3
    )
    print("Step Diff Report:")
    print(f"  Missing steps:      {diff_report['missing']}")
    print(f"  Out-of-order steps: {diff_report['out_of_order']}")
    print(f"  Duplicate steps:    {diff_report['duplicate']}")
    print(f"  Perfect sequence:   {diff_report['is_perfect_sequence']}")

    # 4. Validate document compliance
    print("\n[Step 4] Validating document compliance against Specification v2.1...")
    doc_dict = doc.to_dict()
    violations = validate_embodied_doc(doc_dict)
    if not violations:
        print("  [OK] Document is 100% compliant with Specification v2.1 rules!")
    else:
        print(f"  [!] Violations found: {violations}")

    # 5. Inter-annotator agreement (Cohen's Kappa)
    print("\n[Step 5] Simulating two independent annotators and evaluating Kappa agreement...")
    annotator_a = {
        "clip_001": {
            "working": [{"start": 5.0, "end": 20.0}, {"start": 30.0, "end": 45.0}]
        }
    }
    annotator_b = {
        "clip_001": {
            "working": [{"start": 5.5, "end": 19.5}, {"start": 29.5, "end": 46.0}]
        }
    }

    kappa_res = calculate_temporal_kappa(annotator_a, annotator_b, sample_rate_hz=2.0)
    print(f"  Observed Agreement (p0): {kappa_res['observed_agreement_p0']:.3f}")
    print(f"  Expected Chance (pe):    {kappa_res['expected_chance_pe']:.3f}")
    print(f"  Cohen's Kappa:           {kappa_res['kappa']:.3f}")
    print(f"  Meets Pilot Bar (>=0.80):    {kappa_res['meets_trial_bar']}")
    print(f"  Meets Minimum Bar (>=0.70):  {kappa_res['meets_baseline_bar']}")

    # 6. Label Studio XML template
    print("\n[Step 6] Generating Label Studio v2.1 XML configuration...")
    xml_str = generate_label_studio_xml()
    print(f"  Generated XML ({len(xml_str)} characters, begins with '<View>')")

    print("\n[OK] Embodied AI annotation pipeline demo completed successfully!")


if __name__ == "__main__":
    run_demo()
