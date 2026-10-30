"""Freeze ``acceptance-policy.json`` from the required-case manifest (P15).

The policy is the authority the verifier reads. It is generated once from the
hand-off manifest and then treated as frozen input: a run may not add, remove,
rename or re-scope a required case or scenario.

Usage:
    python scripts/build_acceptance_policy.py \
        --cases <必需用例清单.json> \
        --out <Run>/acceptance-policy.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

POLICY_VERSION = 1

# Terminal states that the instruction text states explicitly. Anything not
# listed here has no frozen terminal state and the verifier does not invent one.
EXPECTED_TERMINAL_STATE = {
    "UI_CANCEL_QUEUED": "cancelled",
    "UI_CANCEL_PREPARING": "cancelled",
    "UI_CANCEL_TERMINAL_REPLAY": "cancelled",
    "ATOMIC_CANCEL_AND_STALE_FENCE": "cancelled",
    "SHUTDOWN_PREPARE_AND_WAIT": "failed",
}

# Cases whose evidence must be a real browser/native session rather than pytest.
NON_AUTOMATED_PREFIXES = ("UI_", "NATIVE_")

# The only case allowed to carry explained skips: the full regression run also
# executes symlink-dependent scenarios that this host cannot create. Every skip
# must still be listed with a reason in the result, and the count must match.
ALLOW_SKIPS = {"REGRESSION_FULL"}


def _phase(case_id: str) -> str:
    if case_id.startswith(NON_AUTOMATED_PREFIXES):
        return "browser_validation"
    if case_id.startswith("VERIFY_"):
        return "verifier_validation"
    return "product_validation"


def build(cases_path: Path, baseline_sha: str | None = None) -> dict:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if payload.get("document_type") != "PLANNED_REQUIRED_CASES_NOT_EXECUTION_EVIDENCE":
        raise SystemExit("unexpected required-case manifest document_type")
    required = payload["required_cases"]
    product, verifier = [], []
    for case in required:
        case_id = case["case_id"]
        gate = case["gate"]
        entry = {
            "case_id": case_id,
            "gate": gate,
            "phase": _phase(case_id),
            "required_scenarios": list(case["required_scenarios"]),
            "instruction_items": list(case.get("instruction_items", [])),
            "evidence_required": True,
        }
        if case_id in EXPECTED_TERMINAL_STATE:
            entry["expected_terminal_state"] = EXPECTED_TERMINAL_STATE[case_id]
        if case_id in ALLOW_SKIPS:
            entry["allow_skips"] = True
        if gate == "P15":
            verifier.append(entry)
        else:
            product.append(entry)
    policy = {
        "document_type": "ACCEPTANCE_POLICY",
        "policy_version": POLICY_VERSION,
        "baseline_sha": baseline_sha or payload.get("baseline_sha"),
        "source_root": payload.get("source_root"),
        "target": "READY_FOR_PRIMARY_REVIEW",
        "may_publish": False,
        "product_cases": product,
        "verifier_cases": verifier,
        "primary_review": [{
            "case_id": "PRIMARY_REVIEW",
            "gate": "P16",
            "phase": "primary_review",
            "required_scenarios": ["primary_replay"],
            "evidence_required": False,
            "owner": "original_chat_primary",
        }],
        "evidence_rules": {
            "evidence_must_be_inside_run": True,
            "evidence_requires_sha256": True,
            "directories_are_not_evidence": True,
            "junit_requires_nonzero_tests": True,
            "required_skip_is_failure": True,
            "ledger_is_derived_from_results": True,
        },
    }
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze acceptance-policy.json")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-sha", default=None)
    args = parser.parse_args(argv)
    policy = build(args.cases, args.baseline_sha)
    text = json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    digest = hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()
    print(f"wrote {args.out} cases={len(policy['product_cases']) + len(policy['verifier_cases'])} "
          f"scenarios={sum(len(c['required_scenarios']) for c in policy['product_cases'] + policy['verifier_cases'])} "
          f"sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
