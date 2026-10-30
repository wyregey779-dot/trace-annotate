"""P15: the delivery-evidence verifier must reject fakes and accept real results.

Each negative control mutates exactly one property of an otherwise valid Run and
asserts that the *specific* defect code is reported. A verifier that merely
returns non-zero for everything would fail the positive control; a verifier that
returns zero for the fixture would fail every negative control.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verify_delivery_evidence", ROOT / "scripts" / "verify_delivery_evidence.py")
verifier = importlib.util.module_from_spec(_spec)
# dataclasses resolves annotations through sys.modules, so register before exec.
sys.modules.setdefault("verify_delivery_evidence", verifier)
_spec.loader.exec_module(verifier)

CANDIDATE_ID = "cand-20261030-test"
SOURCE_SHA = "7cba20a5705599079bd3d41f3c56ac3324056306"
BUILT_AT = "2026-10-30T00:00:00+00:00"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy() -> dict:
    return {
        "document_type": "ACCEPTANCE_POLICY",
        "policy_version": 1,
        "baseline_sha": SOURCE_SHA,
        "product_cases": [
            {"case_id": "ALPHA", "gate": "P01", "phase": "product_validation",
             "required_scenarios": ["one", "two"]},
            {"case_id": "UI_CANCEL_QUEUED", "gate": "P09", "phase": "browser_validation",
             "required_scenarios": ["queued"], "expected_terminal_state": "cancelled"},
        ],
        "verifier_cases": [
            {"case_id": "VERIFY_ALPHA", "gate": "P15", "phase": "verifier_validation",
             "required_scenarios": ["reject"]},
        ],
        "primary_review": [
            {"case_id": "PRIMARY_REVIEW", "gate": "P16", "phase": "primary_review",
             "required_scenarios": ["primary_replay"]},
        ],
    }


def _junit(path: Path, *, tests: int = 2, failures: int = 0, errors: int = 0,
           skipped: int = 0, node: str = "tests/test_x.py::test_case") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tests == 0:
        path.write_text('<?xml version="1.0"?><testsuites/>', encoding="utf-8")
        return
    classname, name = node.split("::")
    path.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="%d" '
        'failures="%d" errors="%d" skipped="%d">'
        '<testcase classname="%s" name="%s"/></testsuite></testsuites>'
        % (tests, failures, errors, skipped, classname, name), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_run(root: Path) -> dict:
    """Create a complete, internally consistent Run; returns the results document."""
    policy = _policy()
    _write_json(root / "acceptance-policy.json", policy)

    wheel = root / "packages" / "trace_annotate-1.1.0.dev0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"wheel-bytes")
    sdist = root / "packages" / "trace_annotate-1.1.0.dev0.tar.gz"
    sdist.write_bytes(b"sdist-bytes")
    _write_json(root / "candidate-manifest.json", {
        "document_type": "CANDIDATE_MANIFEST",
        "manifest_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_sha": SOURCE_SHA,
        "source_tree_clean": True,
        "version": "1.1.0.dev0",
        "built_at": BUILT_AT,
        "wheel": {"path": "packages/" + wheel.name, "sha256": _sha(wheel),
                  "size": wheel.stat().st_size},
        "sdist": {"path": "packages/" + sdist.name, "sha256": _sha(sdist),
                  "size": sdist.stat().st_size},
    })

    results = []
    for spec in policy["product_cases"] + policy["verifier_cases"]:
        case_id = spec["case_id"]
        junit = root / "logs" / f"{case_id}.xml"
        _junit(junit)
        observations = [{"kind": "db_state", "state": spec.get("expected_terminal_state", "succeeded")}]
        results.append({
            "case_id": case_id,
            "gate": spec["gate"],
            "status": "PASS",
            "execution_mode": "pytest",
            "started_at": "2026-10-30T00:10:00+00:00",
            "finished_at": "2026-10-30T00:20:00+00:00",
            "scenarios": [{"scenario_id": item, "status": "PASS"} for item in spec["required_scenarios"]],
            "observations": observations,
            "commands": [{"argv": ["python", "-m", "pytest", "-q"], "exit_code": 0}],
            "evidence": [{"path": f"logs/{case_id}.xml", "sha256": _sha(junit), "kind": "junit"}],
        })
    document = {
        "document_type": "CASE_RESULTS",
        "results_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_sha": SOURCE_SHA,
        "results": results,
    }
    _write_json(root / "case-results.json", document)
    _write_json(root / "gate-ledger.json", {
        "document_type": "GATE_LEDGER",
        "ledger_version": 1,
        "gates": {**{f"P{n:02d}": {"status": "PASS"} for n in (1, 9, 15)},
                  "P16": {"status": "PENDING_PRIMARY_REVIEW"}},
    })
    return document


@pytest.fixture()
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    build_run(root)
    return root


def codes(report) -> set:
    return {problem.code for problem in report.problems}


def mutate_results(root: Path, case_id: str, **changes) -> None:
    document = json.loads((root / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == case_id:
            entry.update(changes)
    _write_json(root / "case-results.json", document)


# -- positive control ------------------------------------------------------
def test_verify_real_valid_result_is_accepted(run):
    report = verifier.verify(run)
    assert report.accepted, [p.render() for p in report.problems]
    assert report.checked_cases == 3
    assert report.checked_scenarios == 4
    assert report.checked_evidence == 3


def test_verify_binds_input_hashes(run):
    report = verifier.verify(run)
    assert "acceptance-policy.json" in report.inputs
    assert "candidate-manifest.json" in report.inputs
    assert "case-results.json" in report.inputs
    assert "gate-ledger.json" in report.inputs
    # The verdict itself is never an input.
    assert not any("verdict" in name for name in report.inputs)


# -- negative controls -----------------------------------------------------
def test_verify_empty_checks_is_rejected(run):
    mutate_results(run, "ALPHA", observations=[], checks=[])
    report = verifier.verify(run)
    assert "OBSERVATION_MISSING" in codes(report)


def test_verify_missing_case_is_rejected(run):
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    document["results"] = [e for e in document["results"] if e["case_id"] != "VERIFY_ALPHA"]
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "CASE_MISSING" in codes(report)


def test_verify_terminal_recorded_as_preparing_is_rejected(run):
    mutate_results(run, "UI_CANCEL_QUEUED",
                   observations=[{"kind": "db_state", "state": "preparing"}])
    report = verifier.verify(run)
    assert "DB_STATE_MISMATCH" in codes(report)


def test_verify_empty_evidence_is_rejected(run):
    mutate_results(run, "ALPHA", evidence=[])
    report = verifier.verify(run)
    assert "EVIDENCE_EMPTY" in codes(report)


def test_verify_zero_test_xml_is_rejected(run):
    _junit(run / "logs" / "ALPHA.xml", tests=0)
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == "ALPHA":
            entry["evidence"][0]["sha256"] = _sha(run / "logs" / "ALPHA.xml")
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "JUNIT_ZERO_TESTS" in codes(report)


def test_verify_missing_hash_is_rejected(run):
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == "ALPHA":
            entry["evidence"][0].pop("sha256")
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "EVIDENCE_HASH_MISSING" in codes(report)


def test_verify_wrong_candidate_is_rejected(run):
    manifest = json.loads((run / "candidate-manifest.json").read_text(encoding="utf-8"))
    manifest["wheel"]["sha256"] = "0" * 64
    _write_json(run / "candidate-manifest.json", manifest)
    report = verifier.verify(run)
    assert "CANDIDATE_HASH_MISMATCH" in codes(report)


def test_verify_source_fallback_is_rejected(run):
    mutate_results(run, "ALPHA", observations=[
        {"kind": "db_state", "state": "succeeded"},
        {"kind": "import_origin",
         "module_file": "D:/视频标注具身标注/265/trace_annotate/workspace/intake.py"},
    ])
    report = verifier.verify(run)
    assert "SOURCE_FALLBACK" in codes(report)


def test_verify_installed_origin_is_accepted(run):
    mutate_results(run, "ALPHA", observations=[
        {"kind": "db_state", "state": "succeeded"},
        {"kind": "import_origin",
         "module_file": "C:/run/wheel-venv/Lib/site-packages/trace_annotate/workspace/intake.py"},
    ])
    report = verifier.verify(run)
    assert report.accepted, [p.render() for p in report.problems]


def test_verify_stale_result_is_rejected(run):
    mutate_results(run, "ALPHA",
                   started_at="2026-10-29T00:00:00+00:00",
                   finished_at="2026-10-29T01:00:00+00:00")
    report = verifier.verify(run)
    assert "RESULT_STALE" in codes(report)


def test_verify_http_failure_is_rejected(run):
    mutate_results(run, "ALPHA", observations=[
        {"kind": "db_state", "state": "succeeded"},
        {"kind": "http_response", "status": 500},
    ])
    report = verifier.verify(run)
    assert "HTTP_FAILURE" in codes(report)


def test_verify_wrong_db_state_is_rejected(run):
    mutate_results(run, "UI_CANCEL_QUEUED",
                   observations=[{"kind": "db_state", "state": "failed"}])
    report = verifier.verify(run)
    assert "DB_STATE_MISMATCH" in codes(report)


def test_verify_process_still_running_is_rejected(run):
    mutate_results(run, "ALPHA", observations=[
        {"kind": "db_state", "state": "succeeded"},
        {"kind": "process", "pid": 4242, "alive": True},
    ])
    report = verifier.verify(run)
    assert "PROCESS_STILL_RUNNING" in codes(report)


def test_verify_junit_false_pass_is_rejected(run):
    _junit(run / "logs" / "ALPHA.xml", tests=2, failures=1)
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == "ALPHA":
            entry["evidence"][0]["sha256"] = _sha(run / "logs" / "ALPHA.xml")
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "JUNIT_FAILURES" in codes(report)


def test_verify_required_skip_is_rejected(run):
    _junit(run / "logs" / "ALPHA.xml", tests=2, skipped=1)
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == "ALPHA":
            entry["evidence"][0]["sha256"] = _sha(run / "logs" / "ALPHA.xml")
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "JUNIT_REQUIRED_SKIP" in codes(report)


def test_verify_bad_log_hash_is_rejected(run):
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == "ALPHA":
            entry["evidence"][0]["sha256"] = "a" * 64
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "EVIDENCE_HASH_MISMATCH" in codes(report)


def test_verify_evidence_outside_run_is_rejected(run):
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    for entry in document["results"]:
        if entry["case_id"] == "ALPHA":
            entry["evidence"][0]["path"] = "../escape.xml"
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "EVIDENCE_OUTSIDE_RUN" in codes(report)


def test_verify_ledger_cannot_outvote_results(run):
    ledger = json.loads((run / "gate-ledger.json").read_text(encoding="utf-8"))
    ledger["gates"]["P09"] = {"status": "PASS"}
    _write_json(run / "gate-ledger.json", ledger)
    mutate_results(run, "UI_CANCEL_QUEUED", status="FAIL")
    report = verifier.verify(run)
    assert "LEDGER_INCONSISTENT" in codes(report)
    assert "CASE_STATUS_NOT_PASS" in codes(report)


def test_verify_missing_scenario_is_rejected(run):
    mutate_results(run, "ALPHA", scenarios=[{"scenario_id": "one", "status": "PASS"}])
    report = verifier.verify(run)
    assert "SCENARIO_MISSING" in codes(report)


def test_verify_duplicate_json_key_is_rejected(run):
    path = run / "case-results.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"document_type": "CASE_RESULTS",',
                                 '"document_type": "CASE_RESULTS", "document_type": "CASE_RESULTS",', 1),
                    encoding="utf-8")
    report = verifier.verify(run)
    assert "DUPLICATE_KEY" in codes(report)


def test_verify_empty_directory_combo_is_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "logs").mkdir()
    _junit(empty / "logs" / "pytest-full.xml", tests=0)
    (empty / "packages").mkdir()
    shutil.copy2(Path(__file__).resolve(), empty / "packages" / "real.whl")
    report = verifier.verify(empty)
    assert not report.accepted
    assert "POLICY_MISSING" in codes(report)


def test_verify_unknown_policy_version_is_rejected(run):
    policy = json.loads((run / "acceptance-policy.json").read_text(encoding="utf-8"))
    policy["policy_version"] = 99
    _write_json(run / "acceptance-policy.json", policy)
    report = verifier.verify(run)
    assert "POLICY_VERSION_UNKNOWN" in codes(report)


def test_verify_unexpected_case_is_rejected(run):
    document = json.loads((run / "case-results.json").read_text(encoding="utf-8"))
    extra = dict(document["results"][0])
    extra["case_id"] = "EXTRA_CASE"
    document["results"].append(extra)
    _write_json(run / "case-results.json", document)
    report = verifier.verify(run)
    assert "CASE_UNEXPECTED" in codes(report)


def test_cli_exit_codes_and_json_output(run, tmp_path, capsys):
    out = tmp_path / "verdict.json"
    assert verifier.main(["--run", str(run), "--json-out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "ACCEPTED"
    assert payload["document_type"] == "PRODUCT_EVIDENCE_VERIFICATION"
    assert payload["summary"]["checked_cases"] == 3

    mutate_results(run, "ALPHA", evidence=[])
    assert verifier.main(["--run", str(run), "--json-out", str(out)]) == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "REJECTED"
    assert any(item["code"] == "EVIDENCE_EMPTY" for item in payload["problems"])
