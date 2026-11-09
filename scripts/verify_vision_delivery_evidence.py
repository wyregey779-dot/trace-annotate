"""Local T00-T16 acceptance verifier.

This verifier never runs the product. Default verification checks structure and
file hashes only. --strict-execution additionally requires an item-bound receipt
for each complete item, with current tested HEAD/tree, successful command exit,
hashed logs and a snapshot of the actual relevant source files. Historical and
source evidence may coexist, but neither satisfies current execution. ACCEPTED
means evidence consistency within the reported scope, never full product
completion or independent authentication that a command was executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPLETE = "FEATURE_COMPLETE_LOCAL"
STATUSES = {COMPLETE, "PARTIAL", "BLOCKED", "NOT_RUN"}
ITEM_IDS = {f"T{index:02d}" for index in range(17)}
HEX40 = set("0123456789abcdef")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise OSError("git executable is unavailable")
    result = subprocess.run([executable, "-C", str(repo), *args], check=True,
                            capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _problem(code: str, message: str, item_id: Optional[str] = None) -> Dict[str, Any]:
    return {"code": code, "item_id": item_id, "message": message}


def _failed_result(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "")).lower()
    return (status in {"fail", "failed", "not_run", "blocked", "error", "rejected"}
            or str(payload.get("verdict", "")).lower() == "rejected"
            or payload.get("passes") is False or payload.get("success") is False
            or payload.get("ok") is False)


def required_source_paths(repo: Path) -> set:
    """Tracked runtime, harness and test sources needed for execution identity."""
    paths = git(repo, "ls-files", "-z").split("\0")
    return {path for path in paths if path and (
        path.startswith(("trace_annotate/", "scripts/", "tests/"))
        or path in {"pyproject.toml", "MANIFEST.in"})}


def _receipt_problems(payload: Any, item_id: str, repo: Path, roots: List[Path],
                      head: str, tree: str, required_sources: set) -> List[Dict[str, Any]]:
    problems = []

    def problem(code, message):
        problems.append(_problem(code, message, item_id))

    if not isinstance(payload, dict) or payload.get("schema") != "traceannotate.execution-receipt.v1":
        problem("EXECUTION_RECEIPT_INVALID", "current_execution must reference a v1 execution receipt")
        return problems
    if payload.get("tested_head") != head or payload.get("tested_tree") != tree:
        problem("EXECUTION_IDENTITY_STALE", "receipt tested_head/tested_tree must match current code")
    if not isinstance(payload.get("item_ids"), list) or item_id not in payload["item_ids"]:
        problem("EXECUTION_ITEM_MISMATCH", "receipt does not declare support for this item")
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        problem("EXECUTION_COMMAND_MISSING", "receipt command must be a nonempty argv list")
    if type(payload.get("exit_code")) is not int or payload["exit_code"] != 0:
        problem("EXECUTION_EXIT_FAILED", "receipt requires actual integer exit_code=0")
    if str(payload.get("status", "")).lower() not in {"pass", "passed", "success", "succeeded"}:
        problem("EXECUTION_STATUS_FAILED", "receipt has no successful actual result status")

    def checked_artifact(record, category):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            problem(category + "_MISSING", "receipt needs a hashed artifact path")
            return
        path = Path(record["path"])
        if not path.is_absolute() or not any(root == path.resolve() or root in path.resolve().parents for root in roots):
            problem(category + "_OUTSIDE_SCOPE", "receipt artifact must be inside Run/Work/Repo")
            return
        if not path.is_file():
            problem(category + "_MISSING", str(path))
            return
        if record.get("sha256") != sha256_file(path):
            problem(category + "_HASH_MISMATCH", str(path))
        if path.suffix.lower() == ".json":
            try:
                result = _load(path)
            except (ValueError, UnicodeError):
                problem(category + "_INVALID", "result artifact is not JSON")
            else:
                if _failed_result(result):
                    problem("EXECUTION_RESULT_FAILED", "normal result artifact declares failure/not_run")

    checked_artifact(payload.get("log"), "EXECUTION_LOG")
    results = payload.get("results", [])
    if not isinstance(results, list):
        problem("EXECUTION_RESULTS_INVALID", "results must be a list of hashed result artifacts")
    else:
        for result in results:
            checked_artifact(result, "EXECUTION_RESULT")
    sources = payload.get("sources")
    seen = set()
    if not isinstance(sources, list) or not sources:
        problem("SOURCE_MANIFEST_MISSING", "receipt requires the tested source snapshot")
        sources = []
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            problem("SOURCE_PATH_INVALID", "source entry requires a repo-relative path")
            continue
        relative = Path(source["path"])
        resolved = (repo / relative).resolve()
        if relative.is_absolute() or repo.resolve() not in resolved.parents or ".." in relative.parts:
            problem("SOURCE_PATH_INVALID", "source entry escapes repository")
            continue
        name = relative.as_posix()
        if name in seen:
            problem("SOURCE_PATH_DUPLICATE", name)
        seen.add(name)
        if not resolved.is_file():
            problem("SOURCE_MISSING", name)
        elif source.get("sha256") != sha256_file(resolved):
            problem("SOURCE_HASH_MISMATCH", name)
    missing = sorted(required_sources - seen)
    if missing or not required_sources:
        problem("SOURCE_COVERAGE_MISSING", "missing required tracked source files: " + ", ".join(missing))
    return problems


def verify(acceptance_path: Path, repo: Path, run: Path, work: Path,
           strict_execution: bool = False) -> Dict[str, Any]:
    problems: List[Dict[str, Any]] = []
    checked = 0
    evidence_count = 0
    scope = "current_execution_identity_and_hash" if strict_execution else "structure_and_hash_only"
    common = {"kind": "traceannotate.local-acceptance-verification", "scope": scope,
              "overall_complete": None,
              "verdict_meaning": "evidence_consistency_only_not_full_product_completion"}
    if not acceptance_path.is_file():
        return {**common, "verdict": "REJECTED", "problems": [_problem("ACCEPTANCE_MISSING", str(acceptance_path))],
                "checked_items": 0, "checked_evidence": 0}
    try:
        acceptance = _load(acceptance_path)
        if not isinstance(acceptance, dict):
            raise ValueError("acceptance must be an object")
    except (ValueError, UnicodeError, OSError) as exc:
        return {**common, "verdict": "REJECTED", "problems": [_problem("ACCEPTANCE_INVALID", str(exc))],
                "checked_items": 0, "checked_evidence": 0}
    common["overall_complete"] = acceptance.get("overall_complete")
    if acceptance.get("kind") != "traceannotate.local-acceptance":
        problems.append(_problem("ACCEPTANCE_INVALID", "kind must be traceannotate.local-acceptance"))
    head = acceptance.get("head")
    tree = acceptance.get("tree")
    if not isinstance(head, str) or len(head) != 40 or set(head) - HEX40:
        problems.append(_problem("CODE_IDENTITY_INVALID", "head must be a 40-hex commit"))
    if not isinstance(tree, str) or len(tree) != 40 or set(tree) - HEX40:
        problems.append(_problem("CODE_IDENTITY_INVALID", "tree must be a 40-hex tree"))
    actual_head = git(repo, "rev-parse", "HEAD")
    actual_tree = git(repo, "rev-parse", "HEAD^{tree}")
    required_sources = required_source_paths(repo) if strict_execution else set()
    if head != actual_head or tree != actual_tree:
        problems.append(_problem("CODE_IDENTITY_STALE",
                                 f"acceptance {head}/{tree} != current {actual_head}/{actual_tree}"))

    items = acceptance.get("items")
    if not isinstance(items, list):
        return {**common, "verdict": "REJECTED", "problems": problems + [_problem("ACCEPTANCE_INVALID", "items must be a list")],
                "checked_items": 0, "checked_evidence": 0}
    seen: Dict[str, Dict[str, Any]] = {}
    for item in items:
        item_id = item.get("id") if isinstance(item, dict) else None
        if item_id not in ITEM_IDS:
            problems.append(_problem("ITEM_UNKNOWN", f"unknown item {item_id!r}"))
            continue
        if item_id in seen:
            problems.append(_problem("ITEM_DUPLICATE", "item appears twice", item_id))
            continue
        seen[item_id] = item
    missing = sorted(ITEM_IDS - set(seen))
    if missing:
        problems.append(_problem("ITEM_MISSING", f"items missing: {missing}"))

    for item_id, item in sorted(seen.items()):
        current_execution_count = 0
        checked += 1
        status = item.get("status")
        if status not in STATUSES:
            problems.append(_problem("STATUS_UNKNOWN", f"status {status!r}", item_id))
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            problems.append(_problem("EVIDENCE_INVALID", "evidence must be a list", item_id))
            evidence = []
        if status == COMPLETE and not evidence:
            problems.append(_problem("EVIDENCE_EMPTY", "complete item has no evidence", item_id))
        if status in {"PARTIAL", "BLOCKED", "NOT_RUN"} and not item.get("open"):
            problems.append(_problem("OPEN_REASON_MISSING", "non-complete item must record its open reason", item_id))
        for entry in evidence:
            evidence_count += 1
            rel = entry.get("path") if isinstance(entry, dict) else None
            recorded = entry.get("sha256") if isinstance(entry, dict) else None
            if not isinstance(rel, str) or not rel:
                problems.append(_problem("EVIDENCE_PATH_INVALID", "evidence needs a path", item_id))
                continue
            path = Path(rel)
            if not path.is_absolute():
                problems.append(_problem("EVIDENCE_PATH_INVALID", "evidence path must be absolute", item_id))
                continue
            roots = [run.resolve(), work.resolve(), repo.resolve()]
            try:
                resolved = path.resolve()
            except OSError:
                problems.append(_problem("EVIDENCE_MISSING", f"cannot resolve {rel}", item_id))
                continue
            if not any(resolved == root or root in resolved.parents for root in roots):
                problems.append(_problem("EVIDENCE_OUTSIDE_SCOPE", f"{rel} is outside Run/Work/Repo", item_id))
                continue
            if not resolved.is_file():
                problems.append(_problem("EVIDENCE_MISSING", f"missing {rel}", item_id))
                continue
            actual = sha256_file(resolved)
            role = entry.get("role")
            if role not in {None, "historical", "source", "current_execution"}:
                problems.append(_problem("EVIDENCE_ROLE_INVALID", f"unknown evidence role {role!r}", item_id))
            if actual != recorded:
                problems.append(_problem("EVIDENCE_HASH_MISMATCH",
                                         f"{rel} sha256 {recorded} != actual {actual}", item_id))
            if status == COMPLETE and resolved.suffix.lower() == ".json":
                try:
                    payload = _load(resolved)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    problems.append(_problem("EVIDENCE_INVALID", f"{rel} is not JSON", item_id))
                    continue
                if role not in {"historical", "source"} and _failed_result(payload):
                    problems.append(_problem("NOT_RUN_CLAIM",
                                             "completion evidence declares failed/not_run result",
                                             item_id))
                if strict_execution and role == "current_execution":
                    receipt_problems = _receipt_problems(payload, item_id, repo, roots,
                                                         actual_head, actual_tree, required_sources)
                    problems.extend(receipt_problems)
                    if not receipt_problems and actual == recorded:
                        current_execution_count += 1
            elif strict_execution and status == COMPLETE and role == "current_execution":
                problems.append(_problem("EXECUTION_RECEIPT_INVALID", "execution receipt must be JSON", item_id))
        if strict_execution and status == COMPLETE and not current_execution_count:
            problems.append(_problem("CURRENT_EXECUTION_MISSING", "complete item has no valid current execution receipt", item_id))

    overall = acceptance.get("overall_complete")
    all_complete = len(seen) == len(ITEM_IDS) and all(
        item.get("status") == COMPLETE for item in seen.values())
    if overall is True and not all_complete:
        problems.append(_problem("OVERALL_CLAIM_FALSE", "overall_complete=true while an item is not complete"))
    if overall is False and all_complete:
        problems.append(_problem("OVERALL_CLAIM_FALSE", "all items complete but overall_complete=false"))
    return {
        **common,
        "verdict": "ACCEPTED" if not problems else "REJECTED",
        "checked_items": checked,
        "checked_evidence": evidence_count,
        "problems": problems,
        "code_identity": {"acceptance": {"head": head, "tree": tree},
                          "current": {"head": actual_head, "tree": actual_tree}},
    }


def _negative_controls(repo: Path, run: Path, work: Path, valid: Dict[str, Any],
                       strict_execution: bool = False) -> List[Dict[str, Any]]:
    controls: List[Dict[str, Any]] = []
    temp = Path(tempfile.mkdtemp(prefix=".t16-negative-", dir=str(work)))
    try:
        good_path = temp / "good.json"
        good_path.write_text(json.dumps(valid, ensure_ascii=False), encoding="utf-8")

        empty = temp / "empty"
        empty.mkdir()
        result = verify(empty / "local-acceptance.json", repo, run, work, strict_execution)
        codes = {p["code"] for p in result["problems"]}
        controls.append({"case": "empty_directory", "rejected": "ACCEPTANCE_MISSING" in codes,
                         "codes": sorted(codes)})

        missing = json.loads(json.dumps(valid))
        missing["items"][0]["evidence"] = [{"path": str(temp / "absent.json"), "sha256": "0" * 64}]
        missing_path = temp / "missing.json"
        missing_path.write_text(json.dumps(missing, ensure_ascii=False), encoding="utf-8")
        result = verify(missing_path, repo, run, work, strict_execution)
        codes = {p["code"] for p in result["problems"]}
        controls.append({"case": "missing_evidence", "rejected": "EVIDENCE_MISSING" in codes,
                         "codes": sorted(codes)})

        tampered = json.loads(json.dumps(valid))
        artifact = temp / "artifact.txt"
        artifact.write_text("tampered", encoding="utf-8")
        tampered["items"][0]["evidence"] = [{"path": str(artifact), "sha256": "0" * 64}]
        tampered_path = temp / "tampered.json"
        tampered_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        result = verify(tampered_path, repo, run, work, strict_execution)
        codes = {p["code"] for p in result["problems"]}
        controls.append({"case": "tampered_artifact", "rejected": "EVIDENCE_HASH_MISMATCH" in codes,
                         "codes": sorted(codes)})

        stale = json.loads(json.dumps(valid))
        stale["head"] = "0" * 40
        stale_path = temp / "stale.json"
        stale_path.write_text(json.dumps(stale, ensure_ascii=False), encoding="utf-8")
        result = verify(stale_path, repo, run, work, strict_execution)
        codes = {p["code"] for p in result["problems"]}
        controls.append({"case": "reused_old_code_report", "rejected": "CODE_IDENTITY_STALE" in codes,
                         "codes": sorted(codes)})

        not_run = json.loads(json.dumps(valid))
        payload = temp / "not-run.json"
        payload.write_text(json.dumps({"status": "not_run"}), encoding="utf-8")
        not_run["items"][0]["status"] = COMPLETE
        not_run["items"][0]["evidence"] = [{"path": str(payload), "sha256": sha256_file(payload)}]
        not_run_path = temp / "not-run-acceptance.json"
        not_run_path.write_text(json.dumps(not_run, ensure_ascii=False), encoding="utf-8")
        result = verify(not_run_path, repo, run, work, strict_execution)
        codes = {p["code"] for p in result["problems"]}
        controls.append({"case": "only_not_run", "rejected": "NOT_RUN_CLAIM" in codes,
                         "codes": sorted(codes)})
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return controls


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify TraceAnnotate local T00-T16 acceptance")
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--strict-execution", action="store_true",
                        help="Require current item-bound execution receipts and actual source hashes")
    args = parser.parse_args(argv)
    result = verify(args.acceptance.expanduser(), args.repo.expanduser(),
                    args.run.expanduser(), args.work.expanduser(), args.strict_execution)
    if args.self_test:
        result["negative_controls"] = _negative_controls(
            args.repo.expanduser(), args.run.expanduser(), args.work.expanduser(),
            _load(args.acceptance.expanduser()), args.strict_execution)
        result["self_test_ok"] = all(control["rejected"] for control in result["negative_controls"])
        if not result["self_test_ok"]:
            result["verdict"] = "REJECTED"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["verdict"])
    print(f"items={result['checked_items']} evidence={result['checked_evidence']} "
          f"problems={len(result['problems'])}")
    if result.get("negative_controls"):
        for control in result["negative_controls"]:
            print(f"{control['case']}: {'REJECTED' if control['rejected'] else 'MISSED'}")
    return 0 if result["verdict"] == "ACCEPTED" and result.get("self_test_ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
