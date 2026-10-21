"""
Command-Line Interface (CLI) for TraceAnnotate.
Provides subcommands for both Media Asset Data Pipelines and Embodied Video Annotations.
"""
from __future__ import annotations
import sys
import os
import json
import argparse
from typing import List

from trace_annotate import __version__
from trace_annotate.core.hasher import head1mb_md5, compute_dhash
from trace_annotate.core.verify import verify_annotations
from trace_annotate.audit.contact_sheet import create_contact_sheet
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps
from trace_annotate.embodied.agreement import calculate_temporal_kappa
from trace_annotate.embodied.validator import validate_embodied_export
from trace_annotate.embodied.label_studio import generate_label_studio_xml


def cmd_asset_verify(args: argparse.Namespace) -> None:
    """Verify an asset annotation file."""
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    inv_records = None
    if args.inventory and os.path.exists(args.inventory):
        inv_records = []
        with open(args.inventory, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    inv_records.append(json.loads(line))

    print(f"Running 21-point verification on {len(records)} records...")
    report = verify_annotations(records, inv_records)

    print(f"Passed: {report.passed}")
    print(f"Total Rows: {report.total_rows}")
    print(f"Checks: {json.dumps(report.checks, ensure_ascii=False, indent=2)}")
    if report.errors:
        print(f"\n[!] Errors ({len(report.errors)}):")
        for err in report.errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n[OK] All verification checks passed!")


def cmd_asset_sheet(args: argparse.Namespace) -> None:
    """Generate visual contact sheet."""
    images = args.images
    labels = args.labels or [os.path.basename(p) for p in images]
    success = create_contact_sheet(
        image_paths=images,
        labels=labels,
        output_path=args.output,
        cols=args.cols,
        title=args.title
    )
    if success:
        print(f"[OK] Contact sheet written to {args.output}")
    else:
        print("[!] Failed to generate contact sheet", file=sys.stderr)
        sys.exit(1)


def cmd_embodied_idle(args: argparse.Namespace) -> None:
    """Compute idle complement for an embodied annotation doc."""
    doc = json.load(open(args.file, "r", encoding="utf-8"))
    working = doc.get("working", [])
    total_dur = args.duration or max([float(w["end"]) for w in working] or [0.0])

    idle = compute_idle_complement(working, total_dur)
    doc["idle"] = idle

    if args.inplace:
        with open(args.file, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"[OK] Updated {args.file} in-place: {len(working)} working, {len(idle)} idle segments.")
    else:
        print(json.dumps(doc, ensure_ascii=False, indent=2))


def cmd_embodied_diff(args: argparse.Namespace) -> None:
    """Diff working segments step numbers against expected procedure."""
    doc = json.load(open(args.file, "r", encoding="utf-8"))
    working = doc.get("working", [])
    result = diff_expected_steps(working, expected_step_count=args.nsteps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_embodied_kappa(args: argparse.Namespace) -> None:
    """Compute temporal Cohen's Kappa between two annotator files."""
    def load_jsonl(path: str):
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    clip_id = item.get("clip")
                    if clip_id:
                        data[clip_id] = item
        return data

    annotator_a = load_jsonl(args.annotator_a)
    annotator_b = load_jsonl(args.annotator_b)

    metrics = calculate_temporal_kappa(annotator_a, annotator_b, sample_rate_hz=args.fps)
    print("=== Inter-Annotator Agreement Evaluation ===")
    print(f"Common Clips: {metrics.get('common_clips')}")
    print(f"Observed Agreement (p0): {metrics.get('observed_agreement_p0')}")
    print(f"Expected Chance (pe):    {metrics.get('expected_chance_pe')}")
    print(f"Cohen's Kappa:           {metrics.get('kappa')}")
    print(f"Meets Trial Bar (>=0.80):    {metrics.get('meets_trial_bar')}")
    print(f"Meets Baseline Bar (>=0.70): {metrics.get('meets_baseline_bar')}")


def cmd_embodied_validate(args: argparse.Namespace) -> None:
    """Validate export compliance against Specification v2.1."""
    report = validate_embodied_export(args.target)
    print(f"Checked: {report['records_checked']} records across {report['files_checked']} files")
    print(f"Violations: {report['violations_count']}")
    if report["violations"]:
        print("\nViolations List:")
        for v in report["violations"][:20]:
            print(f"  - {v}")
        if len(report["violations"]) > 20:
            print(f"  ... and {len(report['violations']) - 20} more.")
        sys.exit(1)
    else:
        print("[OK] Specification v2.1 100% Compliant!")


def cmd_embodied_xml(args: argparse.Namespace) -> None:
    """Print or export Label Studio XML configuration."""
    xml_content = generate_label_studio_xml()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"[OK] Label Studio XML written to {args.output}")
    else:
        print(xml_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trace-annotate",
        description="Production-grade multi-modal asset staging and embodied AI annotation engine."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Asset Verify
    p_verify = subparsers.add_parser("asset-verify", help="Run 21-point verification suite")
    p_verify.add_argument("file", help="Path to annotation JSONL file")
    p_verify.add_argument("--inventory", help="Optional inventory JSONL file to match row counts")
    p_verify.set_defaults(func=cmd_asset_verify)

    # Asset Contact Sheet
    p_sheet = subparsers.add_parser("asset-sheet", help="Generate NxN visual audit contact sheet")
    p_sheet.add_argument("images", nargs="+", help="Image paths to include in sheet")
    p_sheet.add_argument("--labels", nargs="*", help="Optional caption labels for images")
    p_sheet.add_argument("-o", "--output", default="audit_sheet.jpg", help="Output JPG path")
    p_sheet.add_argument("--cols", type=int, default=4, help="Grid columns (default 4)")
    p_sheet.add_argument("--title", default="Visual Audit Sheet", help="Header title")
    p_sheet.set_defaults(func=cmd_asset_sheet)

    # Embodied Idle
    p_idle = subparsers.add_parser("embodied-idle", help="Generate timeline idle complement intervals")
    p_idle.add_argument("file", help="Path to embodied annotation JSON file")
    p_idle.add_argument("--duration", type=float, help="Total clip duration in seconds")
    p_idle.add_argument("--inplace", action="store_true", help="Modify file in-place")
    p_idle.set_defaults(func=cmd_embodied_idle)

    # Embodied Diff
    p_diff = subparsers.add_parser("embodied-diff", help="Diff expected steps S=[1..N]")
    p_diff.add_argument("file", help="Path to embodied annotation JSON file")
    p_diff.add_argument("--nsteps", type=int, help="Total expected steps count")
    p_diff.set_defaults(func=cmd_embodied_diff)

    # Embodied Kappa
    p_kappa = subparsers.add_parser("embodied-kappa", help="Compute inter-annotator temporal Cohen's Kappa")
    p_kappa.add_argument("annotator_a", help="Path to Annotator A JSONL")
    p_kappa.add_argument("annotator_b", help="Path to Annotator B JSONL")
    p_kappa.add_argument("--fps", type=float, default=2.0, help="Timeline sampling rate Hz (default 2.0)")
    p_kappa.set_defaults(func=cmd_embodied_kappa)

    # Embodied Validate
    p_val = subparsers.add_parser("embodied-validate", help="Validate export compliance against Specification v2.1")
    p_val.add_argument("target", help="JSON file or folder to validate")
    p_val.set_defaults(func=cmd_embodied_validate)

    # Embodied XML
    p_xml = subparsers.add_parser("embodied-xml", help="Generate Label Studio XML configuration")
    p_xml.add_argument("-o", "--output", help="Output file path (prints to stdout if omitted)")
    p_xml.set_defaults(func=cmd_embodied_xml)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
