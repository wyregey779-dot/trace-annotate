"""Current execution evidence must survive independently targeted negatives."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "verify_vision_delivery_evidence", Path(__file__).resolve().parents[1] / "scripts" / "verify_vision_delivery_evidence.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
HEAD, TREE = "a" * 40, "b" * 40


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def artifact(path, role=None):
    result = {"path": str(path), "sha256": V.sha256_file(path)}
    if role:
        result["role"] = role
    return result


@pytest.fixture
def case(tmp_path, monkeypatch):
    repo, run, work = (tmp_path / name for name in ("repo", "run", "work"))
    for path in (repo, run, work):
        path.mkdir()
    sources = ["trace_annotate/engine.py", "scripts/check.py", "tests/test_engine.py", "pyproject.toml"]
    for name in sources:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source before execution\n", encoding="utf-8")

    def git(repository, *args):
        assert repository == repo
        if args == ("rev-parse", "HEAD"):
            return HEAD
        if args == ("rev-parse", "HEAD^{tree}"):
            return TREE
        assert args == ("ls-files", "-z")
        return "\0".join(sources) + "\0"

    monkeypatch.setattr(V, "git", git)
    log = run / "pytest.log"
    log.write_text("1 passed in 0.01s", encoding="utf-8")
    receipt = {
        "schema": "traceannotate.execution-receipt.v1", "tested_head": HEAD, "tested_tree": TREE,
        "item_ids": ["T14"], "command": ["python", "-m", "pytest"], "exit_code": 0, "status": "PASS",
        "log": artifact(log), "sources": [{"path": name, "sha256": V.sha256_file(repo / name)} for name in sources],
    }
    receipt_path = write(run / "execution.json", receipt)
    historical = write(run / "old-failure.json", {"status": "FAIL", "tested_head": "c" * 40})
    evidence = [artifact(receipt_path, "current_execution"), artifact(historical, "historical")]
    evidence[1]["tested_head"] = "c" * 40
    evidence[1]["tested_tree"] = "d" * 40
    acceptance = {"kind": "traceannotate.local-acceptance", "head": HEAD, "tree": TREE,
                  "overall_complete": False, "items": [{
                      "id": f"T{index:02d}", "status": V.COMPLETE if index == 14 else "PARTIAL",
                      "open": None if index == 14 else "Not completed", "evidence": evidence if index == 14 else [],
                  } for index in range(17)]}
    acceptance_path = write(run / "acceptance.json", acceptance)
    return {"repo": repo, "run": run, "work": work, "receipt": receipt,
            "receipt_path": receipt_path, "acceptance": acceptance, "path": acceptance_path, "log": log}


def save(case):
    write(case["receipt_path"], case["receipt"])
    # Test fixture re-registers only a deliberately mutated receipt; production
    # verifier never refreshes any hash or edits an input.
    case["acceptance"]["items"][14]["evidence"][0]["sha256"] = V.sha256_file(case["receipt_path"])
    write(case["path"], case["acceptance"])


def verify(case, strict=True):
    return V.verify(case["path"], case["repo"], case["run"], case["work"], strict)


def codes(result):
    return {problem["code"] for problem in result["problems"]}


def test_current_execution_and_historical_failure_can_coexist_without_rewriting_history(case):
    before = case["path"].read_bytes()
    result = verify(case)
    assert result["verdict"] == "ACCEPTED", result["problems"]
    assert result["scope"] == "current_execution_identity_and_hash"
    assert result["overall_complete"] is False
    assert result["verdict_meaning"] == "evidence_consistency_only_not_full_product_completion"
    assert case["path"].read_bytes() == before
    assert case["acceptance"]["items"][14]["evidence"][1]["tested_head"] == "c" * 40


def test_default_mode_explicitly_has_only_structure_and_hash_scope(case):
    case["acceptance"]["items"][14]["evidence"].pop(0)
    write(case["path"], case["acceptance"])
    assert verify(case, False)["scope"] == "structure_and_hash_only"
    assert verify(case, False)["verdict"] == "ACCEPTED"
    assert "CURRENT_EXECUTION_MISSING" in codes(verify(case))


@pytest.mark.parametrize("field,value", [("tested_head", "c" * 40), ("tested_tree", "d" * 40), ("tested_head", None)])
def test_old_or_absent_actual_execution_identity_cannot_be_rebound_by_ledger(case, field, value):
    case["receipt"][field] = value
    save(case)
    assert "EXECUTION_IDENTITY_STALE" in codes(verify(case))


@pytest.mark.parametrize("field,value,code", [
    ("exit_code", 1, "EXECUTION_EXIT_FAILED"),
    ("exit_code", False, "EXECUTION_EXIT_FAILED"),
    ("command", [], "EXECUTION_COMMAND_MISSING"),
    ("status", "FAIL", "EXECUTION_STATUS_FAILED"),
    ("status", "not_run", "EXECUTION_STATUS_FAILED"),
    ("item_ids", ["T00"], "EXECUTION_ITEM_MISMATCH"),
])
def test_non_success_or_unrelated_execution_is_rejected(case, field, value, code):
    case["receipt"][field] = value
    save(case)
    assert code in codes(verify(case))


def test_current_workspace_source_drift_is_not_hidden_by_current_head(case):
    (case["repo"] / "trace_annotate" / "engine.py").write_text("# changed after execution", encoding="utf-8")
    assert "SOURCE_HASH_MISMATCH" in codes(verify(case))


def test_snapshot_must_cover_all_tracked_runtime_scripts_and_tests(case):
    case["receipt"]["sources"] = case["receipt"]["sources"][:1]
    save(case)
    assert "SOURCE_COVERAGE_MISSING" in codes(verify(case))


def test_snapshot_cannot_escape_repository(case):
    case["receipt"]["sources"].append({"path": "../run/pytest.log", "sha256": V.sha256_file(case["log"])})
    save(case)
    assert "SOURCE_PATH_INVALID" in codes(verify(case))


def test_missing_snapshot_is_not_treated_as_current_execution(case):
    case["receipt"].pop("sources")
    save(case)
    assert "SOURCE_MANIFEST_MISSING" in codes(verify(case))


def test_log_hash_drift_is_rejected_without_refresh(case):
    case["log"].write_text("changed log", encoding="utf-8")
    assert "EXECUTION_LOG_HASH_MISMATCH" in codes(verify(case))


def test_receipt_hash_drift_is_rejected(case):
    changed = copy.deepcopy(case["receipt"])
    changed["extra"] = "changed after registration"
    write(case["receipt_path"], changed)
    assert "EVIDENCE_HASH_MISMATCH" in codes(verify(case))


@pytest.mark.parametrize("result", [{"status": "FAIL"}, {"status": "failed"},
                                   {"status": "not_run"}, {"ok": False}])
def test_normal_failed_result_json_cannot_be_outvoted_by_exit_zero(case, result):
    result_path = write(case["run"] / "result.json", result)
    case["receipt"]["results"] = [artifact(result_path)]
    save(case)
    assert "EXECUTION_RESULT_FAILED" in codes(verify(case))


def test_untyped_legacy_fail_result_does_not_support_complete(case):
    case["acceptance"]["items"][14]["evidence"][1].pop("role")
    write(case["path"], case["acceptance"])
    assert "NOT_RUN_CLAIM" in codes(verify(case, False))


def test_default_and_strict_negative_self_tests_still_check_all_five_classes(case):
    for strict in (False, True):
        controls = V._negative_controls(case["repo"], case["run"], case["work"], case["acceptance"], strict)
        assert len(controls) == 5 and all(control["rejected"] for control in controls), controls


def test_strict_cli_reports_scope_original_completion_and_exit_status(case):
    output = case["run"] / "verification.json"
    args = ["--acceptance", str(case["path"]), "--repo", str(case["repo"]), "--run", str(case["run"]),
            "--work", str(case["work"]), "--json-out", str(output), "--strict-execution"]
    assert V.main(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["overall_complete"] is False
    assert result["scope"] == "current_execution_identity_and_hash"
    case["receipt"]["exit_code"] = 2
    save(case)
    assert V.main(args) == 1


def test_missing_input_still_reports_verification_scope(case):
    case["path"].unlink()
    result = verify(case)
    assert result["verdict"] == "REJECTED"
    assert result["scope"] == "current_execution_identity_and_hash"
