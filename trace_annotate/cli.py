"""
Command-Line Interface (CLI) for TraceAnnotate.
Provides media review, general temporal annotation and embodied workflows.
"""
from __future__ import annotations
import sys
import os
import json
import argparse
import math
import webbrowser
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
from trace_annotate.embodied.exporters.srt import export_to_srt
from trace_annotate.embodied.exporters.cvat import export_to_cvat_xml
from trace_annotate.embodied.exporters.action_dataset import export_to_action_segmentation
from trace_annotate.embodied.qc_report import generate_qc_report
from trace_annotate.embodied.motion import extract_motion_energy, propose_idle_pauses
from trace_annotate.temporal.commands import register_video_commands
from trace_annotate.workspace.commands import register_workspace_commands
from trace_annotate.embodied.quality_pipeline import (
    add_pipeline_arguments, guarded, idle_command, diff_command, kappa_command,
    validate_command, pipeline_command,
)
from trace_annotate.embodied.quality_io import (
    QualityInputError, QualityViolation, load_docs_from_file, require_new_output, write_new_text,
)
from trace_annotate.embodied.validator import validate_embodied_doc


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


def _duration_seconds(value) -> float:
    """Require a usable clip length before generating timeline complements."""
    try:
        if isinstance(value, bool):
            raise ValueError
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("duration must be a finite positive number") from None
    return duration


def _duration_argument(value: str) -> float:
    try:
        return _duration_seconds(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def cmd_embodied_idle(args: argparse.Namespace) -> int:
    """Validated idle repair; stdout/new copy by default, explicit backed-up inplace."""
    return idle_command(args)

def cmd_embodied_diff(args: argparse.Namespace) -> int:
    """Strict procedure coverage and temporal-order gate."""
    return diff_command(args)

def cmd_embodied_kappa(args: argparse.Namespace) -> int:
    """Full-file coverage, semantic and non-degenerate temporal agreement gate."""
    return kappa_command(args)

def cmd_embodied_validate(args: argparse.Namespace) -> int:
    """Strict JSON/JSONL, fields, duration and timeline coverage validation."""
    return validate_command(args)

def cmd_embodied_xml(args: argparse.Namespace) -> None:
    """Print or export Label Studio XML configuration."""
    xml_content = generate_label_studio_xml()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"[OK] Label Studio XML written to {args.output}")
    else:
        print(xml_content)


def cmd_embodied_export(args: argparse.Namespace) -> None:
    """Export annotation document to SRT subtitles or CVAT XML."""
    docs = load_docs_from_file(args.file)
    fmt = args.format.lower()
    if fmt in {"srt", "cvat"} and len(docs) != 1:
        raise QualityInputError("SRT/CVAT export requires exactly one clip; choose individual files")
    violations = [f"record {index}: {error}" for index, doc in enumerate(docs, 1)
                  for error in validate_embodied_doc(doc) if not error.startswith("REVIEW_REQUIRED:")]
    if violations:
        raise QualityViolation("; ".join(violations))
    doc = docs[0]

    if fmt == "srt":
        srt_content = export_to_srt(doc)
        out = args.output or os.path.splitext(args.file)[0] + ".srt"
        write_new_text(out, srt_content, [args.file])
        print(f"[OK] Exported SRT subtitles to {out} (can be loaded in VLC/PotPlayer)")

    elif fmt == "cvat":
        xml_content = export_to_cvat_xml(doc)
        out = args.output or os.path.splitext(args.file)[0] + "_cvat.xml"
        write_new_text(out, xml_content, [args.file])
        print(f"[OK] Exported CVAT Track XML to {out}")

    elif fmt == "action":
        out_dir = args.output or "./action_dataset"
        # The library supports explicit incremental datasets. The end-user CLI
        # always creates a new destination, so it cannot overwrite prior labels.
        destination = require_new_output(out_dir, [args.file])
        destination.mkdir(parents=True, exist_ok=False)
        m_file, gts = export_to_action_segmentation(docs, str(destination))
        print(f"[OK] Exported Action Segmentation dataset to {out_dir}:")
        print(f"  Mapping: {m_file}")
        print(f"  Ground truth files: {len(gts)}")


def cmd_embodied_qc(args: argparse.Namespace) -> int:
    """Generate the same strict pipeline report used by the batch CLI."""
    args.target = args.input
    args.report = args.output
    return pipeline_command(args)


def cmd_embodied_pipeline(args: argparse.Namespace) -> int:
    """Run all requested embodied machine gates and write a new report."""
    return pipeline_command(args)

def cmd_embodied_ui(args: argparse.Namespace) -> None:
    """Open interactive Web video timeline annotator in browser."""
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embodied", "ui", "annotator.html")
    if os.path.exists(ui_path):
        print(f"Opening TraceAnnotate Web Timeline Annotator: {ui_path}")
        webbrowser.open(f"file://{ui_path}")
    else:
        print(f"Error: UI file not found: {ui_path}", file=sys.stderr)
        sys.exit(1)


def cmd_embodied_motion(args: argparse.Namespace) -> None:
    """Analyze video motion energy and propose idle intervals."""
    print(f"Analyzing motion energy on {args.video}...")
    profile = extract_motion_energy(args.video, sample_fps=args.fps)
    pauses = propose_idle_pauses(profile, min_pause_s=args.min_pause)
    print(f"Extracted {len(profile)} motion samples.")
    print(f"Discovered {len(pauses)} candidate idle intervals (>={args.min_pause}s):")
    for idx, p in enumerate(pauses, 1):
        print(f"  #{idx}: [{p['start']}s -> {p['end']}s] (Duration: {p['duration']}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trace-annotate",
        description="Local media review and temporal video annotation tools."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    register_video_commands(subparsers)
    register_workspace_commands(subparsers)

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
    p_idle.add_argument("--duration", type=_duration_argument, help="Finite positive total clip duration in seconds")
    p_idle.add_argument("--inplace", action="store_true", help="Explicit validated atomic replacement with original backup")
    p_idle.add_argument("--output", "-o", help="New corrected copy; existing files are never overwritten")
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
    p_kappa.add_argument("--min-kappa", type=float, default=0.80)
    p_kappa.set_defaults(func=cmd_embodied_kappa)

    # Embodied Validate
    p_val = subparsers.add_parser("embodied-validate", help="Validate TraceAnnotate annotation structure and segment fields")
    p_val.add_argument("target", help="JSON file or folder to validate")
    p_val.set_defaults(func=cmd_embodied_validate)

    # Embodied XML
    p_xml = subparsers.add_parser("embodied-xml", help="Generate Label Studio XML configuration")
    p_xml.add_argument("-o", "--output", help="Output file path (prints to stdout if omitted)")
    p_xml.set_defaults(func=cmd_embodied_xml)

    # Embodied Export
    p_exp = subparsers.add_parser("embodied-export", help="Export to SRT subtitles, CVAT XML, or ActionFormer")
    p_exp.add_argument("file", help="Path to embodied annotation JSON file")
    p_exp.add_argument("--format", default="srt", choices=["srt", "cvat", "action"], help="Export format")
    p_exp.add_argument("-o", "--output", help="Output file or directory path")
    p_exp.set_defaults(func=cmd_embodied_export)

    # Embodied QC Report
    p_qc = subparsers.add_parser("embodied-qc", help="Generate interactive HTML quality control report")
    p_qc.add_argument("input", help="Path to annotation JSON/JSONL file or folder")
    p_qc.add_argument("-o", "--output", default="qc_report.html", help="Output HTML file path")
    p_qc.add_argument("--nsteps", "-n", type=int)
    p_qc.add_argument("--compare", "-c")
    p_qc.add_argument("--auto-idle", action="store_true")
    p_qc.add_argument("--fps", type=float, default=2.0)
    p_qc.add_argument("--min-kappa", type=float, default=0.80)
    p_qc.set_defaults(func=cmd_embodied_qc)

    p_pipeline = subparsers.add_parser("embodied-pipeline", help="Strict batch quality gates; new report, no source mutation")
    add_pipeline_arguments(p_pipeline)
    p_pipeline.set_defaults(func=cmd_embodied_pipeline)

    # Embodied UI
    p_ui = subparsers.add_parser("embodied-ui", help="Launch interactive browser timeline annotator")
    p_ui.set_defaults(func=cmd_embodied_ui)

    # Embodied Motion
    p_mot = subparsers.add_parser("embodied-motion", help="Extract motion energy and propose idle intervals")
    p_mot.add_argument("video", help="Path to video file")
    p_mot.add_argument("--fps", type=float, default=4.0, help="Sampling FPS")
    p_mot.add_argument("--min-pause", type=float, default=2.0, help="Minimum pause duration in seconds")
    p_mot.set_defaults(func=cmd_embodied_motion)

    args = parser.parse_args()
    if hasattr(args, "func"):
        if args.command == "workspace":
            from .workspace.commands import run_workspace_command
            raise SystemExit(run_workspace_command(args))
        if args.command in {"embodied-idle", "embodied-diff", "embodied-kappa", "embodied-validate", "embodied-qc", "embodied-pipeline", "embodied-export"}:
            raise SystemExit(guarded(args.func, args))
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
