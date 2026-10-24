"""Run the documented annotation workflow using synthetic data and the real CLI.

Install TraceAnnotate first. No video, API key or model weights are required.
The destination must not exist so a rerun cannot overwrite your annotations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/quickstart"))
    args = parser.parse_args()
    destination = args.output.absolute()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"destination already exists; choose a new directory: {destination}")

    source = Path(__file__).parent / "annotations" / "assembly.json"
    annotation = destination / "assembly.json"
    shutil.copyfile(source, annotation)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"

    def cli(*arguments: str) -> str:
        result = subprocess.run(
            [sys.executable, "-m", "trace_annotate.cli", *arguments],
            check=True, capture_output=True, encoding="utf-8", env=environment,
        )
        print(result.stdout, end="")
        return result.stdout

    cli("embodied-idle", str(annotation), "--inplace")
    validation = json.loads(cli("embodied-validate", str(annotation)))
    if not validation["passed"] or validation["records_checked"] != 1 or validation["violations"]:
        raise RuntimeError("Strict annotation structure check did not pass")
    difference = json.loads(cli("embodied-diff", str(annotation), "--nsteps", "2"))
    quality = json.loads(cli("embodied-qc", str(annotation), "-o", str(destination / "qc-report.html")))
    if not quality["passed"] or quality["status"] != "PASS_SCOPED" or quality["exit_code"] != 0:
        raise RuntimeError("Declared annotation machine-check scope did not pass")
    cli("embodied-export", str(annotation), "--format", "srt", "-o", str(destination / "assembly.srt"))

    document = json.loads(annotation.read_text(encoding="utf-8"))
    expected_idle = [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 7.0}, {"start": 10.0, "end": 12.0}]
    if document["idle"] != expected_idle or document["duration"] != 12:
        raise RuntimeError("Unexpected idle intervals or duration in exported document")
    if difference["missing"] or difference["out_of_order"]:
        raise RuntimeError("Unexpected missing or out-of-order procedure steps")
    subtitles = (destination / "assembly.srt").read_text(encoding="utf-8")
    if subtitles.count(" --> ") != 5 or "00:00:10,000 --> 00:00:12,000" not in subtitles:
        raise RuntimeError("SRT export did not preserve all five timeline segments")
    print(f"Verified: 2 working segments, 3 idle intervals, 5 subtitle entries.\nOutputs: {destination}")
    print("Machine scope: PASS_SCOPED. Synthetic format example; video truth and independent human annotation are not verified.")


if __name__ == "__main__":
    main()
