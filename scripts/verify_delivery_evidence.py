"""Verify delivery evidence for the reliable-batch-import slice (P15).

Usage:
    python scripts/verify_delivery_evidence.py --run <RunDir>

The RunDir must contain:
  gate-ledger.json      - {cases: {P01..P15: {status, evidence, ...}}}
  packages/             - the candidate wheel/sdist (only one wheel)
  logs/pytest-full.xml  - a JUnit report
The verifier refuses to rubber-stamp: a missing case, a case not PASS, an empty
evidence path, a JUnit report with failures, or a wheel whose hash is not the
one recorded in the ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_CASES = [f"P{n:02d}" for n in range(1, 16)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(run: Path) -> list:
    problems: list[str] = []
    ledger_path = run / "gate-ledger.json"
    if not ledger_path.is_file():
        return [f"missing gate-ledger.json in {run}"]
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"gate-ledger.json is not valid JSON: {exc}"]
    cases = ledger.get("cases", {})

    for case_id in REQUIRED_CASES:
        entry = cases.get(case_id)
        if entry is None:
            problems.append(f"missing required case {case_id}")
            continue
        if entry.get("status") != "PASS":
            problems.append(f"{case_id} status is {entry.get('status')!r}, expected PASS")
        evidence = entry.get("evidence")
        if not evidence:
            problems.append(f"{case_id} has no evidence path")
        else:
            for rel in evidence if isinstance(evidence, list) else [evidence]:
                if not (run / rel).is_file() and not (run / rel).is_dir():
                    problems.append(f"{case_id} evidence not found: {rel}")

    # Wheel identity.
    pkg_dir = run / "packages"
    wheels = sorted(pkg_dir.glob("*.whl")) if pkg_dir.is_dir() else []
    if len(wheels) != 1:
        problems.append(f"expected exactly one candidate wheel under packages/, found {len(wheels)}")
    else:
        recorded = (ledger.get("candidate") or {}).get("wheel_sha256")
        actual = _sha256(wheels[0])
        if recorded and recorded.lower() != actual.lower():
            problems.append(f"wheel sha256 mismatch: ledger={recorded} actual={actual}")

    # JUnit.
    junit = run / "logs" / "pytest-full.xml"
    if not junit.is_file():
        problems.append("missing logs/pytest-full.xml")
    else:
        try:
            root = ET.parse(junit).getroot()
            suites = root.iter("testsuite")
            for suite in suites:
                failures = int(suite.get("failures", 0) or 0)
                errors = int(suite.get("errors", 0) or 0)
                if failures or errors:
                    problems.append(
                        f"junit {suite.get('name')} has {failures} failures / {errors} errors")
                    break
        except ET.ParseError as exc:
            problems.append(f"junit not parseable: {exc}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.expanduser().resolve()
    if not run.is_dir():
        print(f"Run directory not found: {run}")
        return 2
    problems = verify(run)
    if problems:
        print("DELIVERY EVIDENCE REJECTED:")
        for item in problems:
            print(" - " + item)
        return 1
    print("DELIVERY EVIDENCE ACCEPTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
