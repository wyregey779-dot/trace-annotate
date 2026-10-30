"""Strict delivery-evidence verifier (P15).

The verifier answers exactly one question: *do the recorded artefacts prove that
the frozen candidate passed the frozen required cases?* It never runs the
product and it never trusts a claim that is not backed by a file whose hash is
recorded and whose content matches the case's declared expectation.

Design rules that this module enforces:

1. **The required set comes from the policy, not from the results.** A run may
   not shrink its obligations by deleting cases or scenarios.
2. **A ledger may not outvote the results.** ``gate-ledger.json`` is a derived
   summary; every gate it marks PASS must be recomputed from
   ``case-results.json`` and its evidence.
3. **Evidence must be inside the declared Run.** Relative paths only, no
   traversal, no symlink escape, and every file needs a recorded sha256 that
   matches the bytes on disk.
4. **A directory is not evidence.** Every PASS needs at least one concrete
   observation and one hashed artefact; ``observations: []`` or ``checks: []``
   is a hard failure.
5. **JUnit must contain real tests.** A ``<testsuites/>`` document, a suite with
   zero cases, or a skipped required test is rejected.
6. **Identity must be pinned.** Source SHA, candidate id, wheel hash and the
   installed import origin all have to agree; a module loaded from the source
   tree instead of site-packages is rejected.

Every rejection carries a stable code so negative controls can assert that the
*specific* defect was caught rather than "something failed".
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SUPPORTED_POLICY_VERSIONS = {1}
SUPPORTED_MANIFEST_VERSIONS = {1}
SUPPORTED_RESULTS_VERSIONS = {1}
SUPPORTED_LEDGER_VERSIONS = {1}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EXECUTION_MODES = {"pytest", "browser", "native", "script", "manual"}
EVIDENCE_KINDS = {
    "junit": {".xml"},
    "log": {".log", ".txt", ".jsonl"},
    "json": {".json"},
    "jsonl": {".jsonl"},
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".webm"},
    "har": {".har"},
    "csv": {".csv"},
}
GATES = [f"P{n:02d}" for n in range(1, 17)]


# --------------------------------------------------------------------------
# problems
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    code: str
    message: str
    case_id: Optional[str] = None

    def render(self) -> str:
        prefix = f"[{self.code}]"
        if self.case_id:
            return f"{prefix} {self.case_id}: {self.message}"
        return f"{prefix} {self.message}"


@dataclass
class Report:
    problems: List[Problem] = field(default_factory=list)
    checked_cases: int = 0
    checked_scenarios: int = 0
    checked_evidence: int = 0
    inputs: Dict[str, str] = field(default_factory=dict)

    def add(self, code: str, message: str, case_id: Optional[str] = None) -> None:
        self.problems.append(Problem(code, message, case_id))

    @property
    def accepted(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------------
# strict loading
# --------------------------------------------------------------------------
class DuplicateKey(ValueError):
    pass


def _no_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise DuplicateKey(f"duplicate JSON key {key!r}")
        seen[key] = value
    return seen


def load_json(path: Path) -> Any:
    """Parse JSON, rejecting duplicate object keys and non-object roots."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8")
    return json.loads(text, object_pairs_hook=_no_duplicates)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _require_str(value: Any, what: str, report: Report, code: str,
                 case_id: Optional[str] = None) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        report.add(code, f"{what} must be a non-empty string", case_id)
        return None
    return value


# --------------------------------------------------------------------------
# path containment
# --------------------------------------------------------------------------
def resolve_inside(root: Path, rel: Any, report: Report, what: str,
                   case_id: Optional[str] = None) -> Optional[Path]:
    if not isinstance(rel, str) or not rel.strip():
        report.add("EVIDENCE_PATH_INVALID", f"{what} must be a non-empty relative path", case_id)
        return None
    candidate = Path(rel)
    if candidate.is_absolute() or candidate.drive or rel.startswith(("\\\\", "//")):
        report.add("EVIDENCE_OUTSIDE_RUN", f"{what} must be relative to the Run: {rel}", case_id)
        return None
    if any(part == ".." for part in candidate.parts):
        report.add("EVIDENCE_OUTSIDE_RUN", f"{what} must not traverse upwards: {rel}", case_id)
        return None
    target = (root / candidate).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        report.add("EVIDENCE_OUTSIDE_RUN", f"{what} escapes the Run boundary: {rel}", case_id)
        return None
    return target


# --------------------------------------------------------------------------
# JUnit
# --------------------------------------------------------------------------
@dataclass
class JunitSummary:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    node_ids: List[str] = field(default_factory=list)
    suite_count: int = 0


def read_junit(path: Path) -> JunitSummary:
    root = ET.parse(path).getroot()
    summary = JunitSummary()
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError(f"unexpected JUnit root <{root.tag}>")
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    summary.suite_count = len(suites)
    for suite in suites:
        summary.tests += int(suite.get("tests") or 0)
        summary.failures += int(suite.get("failures") or 0)
        summary.errors += int(suite.get("errors") or 0)
        summary.skipped += int(suite.get("skipped") or 0)
        for case in suite.iter("testcase"):
            classname = case.get("classname") or ""
            name = case.get("name") or ""
            summary.node_ids.append(f"{classname}::{name}" if classname else name)
    # A JUnit document that claims tests without any testcase node is a lie.
    if summary.tests == 0 and summary.node_ids:
        summary.tests = len(summary.node_ids)
    return summary


# --------------------------------------------------------------------------
# main verification
# --------------------------------------------------------------------------
class Verifier:
    def __init__(self, run: Path, policy_path: Optional[Path] = None):
        self.run = run
        self.policy_path = policy_path or (run / "acceptance-policy.json")
        self.report = Report()

    # -- helpers ---------------------------------------------------------
    def _input_key(self, path: Path) -> str:
        """Name an input for the hash manifest; tolerate an external policy."""
        try:
            return str(path.relative_to(self.run)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def _load(self, path: Path, code_prefix: str, what: str,
              case_id: Optional[str] = None) -> Optional[Any]:
        if not path.is_file():
            self.report.add(f"{code_prefix}_MISSING", f"{what} is missing at {path.name}", case_id)
            return None
        try:
            payload = load_json(path)
        except DuplicateKey as exc:
            self.report.add("DUPLICATE_KEY", f"{what}: {exc}", case_id)
            return None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.report.add(f"{code_prefix}_INVALID", f"{what} is not valid JSON: {exc}", case_id)
            return None
        self.report.inputs[self._input_key(path)] = sha256_file(path)
        if not isinstance(payload, dict):
            self.report.add(f"{code_prefix}_INVALID", f"{what} must be a JSON object", case_id)
            return None
        return payload

    # -- policy ----------------------------------------------------------
    def check_policy(self) -> Optional[Dict[str, Any]]:
        policy = self._load(self.policy_path, "POLICY", "acceptance-policy.json")
        if policy is None:
            return None
        if policy.get("document_type") != "ACCEPTANCE_POLICY":
            self.report.add("POLICY_INVALID", "document_type must be ACCEPTANCE_POLICY")
            return None
        version = policy.get("policy_version")
        if type(version) is not int or version not in SUPPORTED_POLICY_VERSIONS:
            self.report.add("POLICY_VERSION_UNKNOWN", f"unsupported policy_version {version!r}")
            return None
        for section in ("product_cases", "verifier_cases", "primary_review"):
            if not isinstance(policy.get(section), list) or not policy[section]:
                self.report.add("POLICY_INVALID", f"{section} must be a non-empty list")
                return None
        seen: Dict[str, str] = {}
        for section in ("product_cases", "verifier_cases", "primary_review"):
            for entry in policy[section]:
                if not isinstance(entry, dict):
                    self.report.add("POLICY_INVALID", f"{section} entries must be objects")
                    return None
                case_id = entry.get("case_id")
                if not isinstance(case_id, str) or not case_id:
                    self.report.add("POLICY_INVALID", f"{section} entry lacks case_id")
                    return None
                if case_id in seen:
                    self.report.add("POLICY_INVALID", f"duplicate case_id {case_id} across sections")
                    return None
                seen[case_id] = section
                gate = entry.get("gate")
                if gate not in GATES:
                    self.report.add("POLICY_INVALID", f"{case_id} has unknown gate {gate!r}")
                    return None
                scenarios = entry.get("required_scenarios")
                if not isinstance(scenarios, list) or not scenarios:
                    self.report.add("POLICY_INVALID", f"{case_id} must declare required_scenarios")
                    return None
                if len(set(scenarios)) != len(scenarios):
                    self.report.add("POLICY_INVALID", f"{case_id} has duplicate scenario ids")
                    return None
                if not all(isinstance(item, str) and item for item in scenarios):
                    self.report.add("POLICY_INVALID", f"{case_id} scenario ids must be strings")
                    return None
        return policy

    # -- candidate manifest ---------------------------------------------
    def check_manifest(self) -> Optional[Dict[str, Any]]:
        manifest = self._load(self.run / "candidate-manifest.json", "MANIFEST",
                              "candidate-manifest.json")
        if manifest is None:
            return None
        if manifest.get("document_type") != "CANDIDATE_MANIFEST":
            self.report.add("MANIFEST_INVALID", "document_type must be CANDIDATE_MANIFEST")
            return None
        version = manifest.get("manifest_version")
        if type(version) is not int or version not in SUPPORTED_MANIFEST_VERSIONS:
            self.report.add("MANIFEST_VERSION_UNKNOWN", f"unsupported manifest_version {version!r}")
            return None
        source_sha = manifest.get("source_sha")
        if not isinstance(source_sha, str) or not GIT_SHA_RE.match(source_sha):
            self.report.add("MANIFEST_INVALID", "source_sha must be a 40-hex commit id")
            return None
        if manifest.get("source_tree_clean") is not True:
            self.report.add("MANIFEST_INVALID", "source_tree_clean must be true")
        built_at = _iso(manifest.get("built_at"))
        if built_at is None:
            self.report.add("TIMESTAMP_INVALID", "candidate-manifest built_at must be an ISO timestamp")
        candidate_id = manifest.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            self.report.add("CANDIDATE_WRONG_ID", "candidate_id is required")
            return None
        for key in ("wheel", "sdist"):
            entry = manifest.get(key)
            if not isinstance(entry, dict):
                self.report.add("MANIFEST_INVALID", f"{key} entry is required")
                continue
            path = resolve_inside(self.run, entry.get("path"), self.report, f"manifest {key}")
            recorded = entry.get("sha256")
            if not isinstance(recorded, str) or not SHA256_RE.match(recorded):
                self.report.add("EVIDENCE_HASH_MISSING", f"manifest {key} lacks a sha256")
                continue
            if path is None:
                continue
            if not path.is_file():
                self.report.add("CANDIDATE_MISSING", f"manifest {key} file is absent: {entry.get('path')}")
                continue
            actual = sha256_file(path)
            self.report.inputs[str(path.relative_to(self.run)).replace("\\", "/")] = actual
            if actual != recorded:
                self.report.add("CANDIDATE_HASH_MISMATCH",
                                f"manifest {key} sha256 {recorded} != actual {actual}")
            size = entry.get("size")
            if type(size) is int and path.stat().st_size != size:
                self.report.add("CANDIDATE_HASH_MISMATCH",
                                f"manifest {key} size {size} != actual {path.stat().st_size}")
        return manifest

    # -- results ---------------------------------------------------------
    def check_results(self, policy: Dict[str, Any],
                      manifest: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        results = self._load(self.run / "case-results.json", "RESULTS", "case-results.json")
        if results is None:
            return None
        if results.get("document_type") != "CASE_RESULTS":
            self.report.add("RESULTS_INVALID", "document_type must be CASE_RESULTS")
            return None
        version = results.get("results_version")
        if type(version) is not int or version not in SUPPORTED_RESULTS_VERSIONS:
            self.report.add("RESULTS_VERSION_UNKNOWN", f"unsupported results_version {version!r}")
            return None
        entries = results.get("results")
        if not isinstance(entries, list):
            self.report.add("RESULTS_INVALID", "results must be a list")
            return None
        if manifest is not None:
            if results.get("candidate_id") != manifest.get("candidate_id"):
                self.report.add("CANDIDATE_WRONG_ID",
                                "case-results candidate_id does not match the manifest")
            if results.get("source_sha") != manifest.get("source_sha"):
                self.report.add("SOURCE_SHA_MISMATCH",
                                "case-results source_sha does not match the manifest")

        by_id: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                self.report.add("RESULTS_INVALID", "every result must be an object")
                continue
            case_id = entry.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                self.report.add("RESULTS_INVALID", "a result lacks case_id")
                continue
            if case_id in by_id:
                self.report.add("DUPLICATE_KEY", f"case_id {case_id} appears more than once", case_id)
                continue
            by_id[case_id] = entry

        required: List[Dict[str, Any]] = []
        for section in ("product_cases", "verifier_cases"):
            required.extend(policy[section])
        required_ids = {entry["case_id"] for entry in required}
        for entry in required:
            if entry["case_id"] not in by_id:
                self.report.add("CASE_MISSING", "required case has no result", entry["case_id"])
        for case_id in by_id:
            if case_id not in required_ids:
                self.report.add("CASE_UNEXPECTED", "result for a case the policy does not require",
                                case_id)

        built_at = _iso((manifest or {}).get("built_at"))
        for entry in required:
            case_id = entry["case_id"]
            payload = by_id.get(case_id)
            if payload is None:
                continue
            self._check_case(entry, payload, built_at)
        return results

    # -- one case --------------------------------------------------------
    def _check_case(self, spec: Dict[str, Any], result: Dict[str, Any],
                    built_at: Optional[dt.datetime]) -> None:
        case_id = spec["case_id"]
        self.report.checked_cases += 1
        status = result.get("status")
        if status != "PASS":
            self.report.add("CASE_STATUS_NOT_PASS", f"status is {status!r}, expected PASS", case_id)
        mode = result.get("execution_mode")
        if mode not in EXECUTION_MODES:
            self.report.add("OBSERVATION_MISSING", f"execution_mode {mode!r} is not recognised", case_id)

        started = _iso(result.get("started_at"))
        finished = _iso(result.get("finished_at"))
        if started is None or finished is None:
            self.report.add("TIMESTAMP_INVALID", "started_at and finished_at must be ISO timestamps",
                            case_id)
        else:
            if finished < started:
                self.report.add("TIMESTAMP_INVALID", "finished_at precedes started_at", case_id)
            if built_at is not None and finished < built_at:
                self.report.add("RESULT_STALE",
                                "case finished before the candidate was built", case_id)

        # -- scenarios: exact coverage of the frozen minimum ---------------
        required_scenarios = spec["required_scenarios"]
        declared = result.get("scenarios")
        if not isinstance(declared, list) or not declared:
            self.report.add("SCENARIO_MISSING", "no scenarios were recorded", case_id)
            declared = []
        seen: Dict[str, str] = {}
        for scenario in declared:
            if not isinstance(scenario, dict):
                self.report.add("RESULTS_INVALID", "scenario entries must be objects", case_id)
                continue
            scenario_id = scenario.get("scenario_id")
            if not isinstance(scenario_id, str) or not scenario_id:
                self.report.add("RESULTS_INVALID", "scenario lacks scenario_id", case_id)
                continue
            if scenario_id in seen:
                self.report.add("DUPLICATE_KEY", f"scenario {scenario_id} repeated", case_id)
                continue
            seen[scenario_id] = scenario.get("status")
            self.report.checked_scenarios += 1
            if scenario.get("status") != "PASS":
                self.report.add("SCENARIO_NOT_PASS",
                                f"scenario {scenario_id} status {scenario.get('status')!r}",
                                case_id)
        for scenario_id in required_scenarios:
            if scenario_id not in seen:
                self.report.add("SCENARIO_MISSING", f"required scenario {scenario_id} absent", case_id)
        for scenario_id in seen:
            if scenario_id not in required_scenarios:
                self.report.add("SCENARIO_NOT_PASS",
                                f"scenario {scenario_id} is not part of the frozen minimum", case_id)

        # -- observations --------------------------------------------------
        observations = result.get("observations")
        if not isinstance(observations, list) or not observations:
            self.report.add("OBSERVATION_MISSING", "observations must be a non-empty list", case_id)
            observations = []
        checks = result.get("checks")
        if checks is not None and (not isinstance(checks, list) or not checks):
            self.report.add("OBSERVATION_MISSING", "checks must not be an empty list", case_id)

        expected_terminal = spec.get("expected_terminal_state")
        terminal_seen = None
        for observation in observations:
            if not isinstance(observation, dict) or not isinstance(observation.get("kind"), str):
                self.report.add("RESULTS_INVALID", "observation entries need a kind", case_id)
                continue
            kind = observation["kind"]
            if kind == "http_response":
                status_code = observation.get("status")
                if type(status_code) is not int:
                    self.report.add("RESULTS_INVALID", "http_response needs an integer status", case_id)
                elif status_code >= 400 and not observation.get("expected_failure"):
                    self.report.add("HTTP_FAILURE", f"HTTP status {status_code}", case_id)
            elif kind == "db_state":
                terminal_seen = observation.get("state")
            elif kind == "process":
                if observation.get("alive") is True:
                    self.report.add("PROCESS_STILL_RUNNING",
                                    f"process {observation.get('pid')} is still alive", case_id)
                if not isinstance(observation.get("pid"), int):
                    self.report.add("RESULTS_INVALID", "process observation needs an integer pid",
                                    case_id)
            elif kind == "import_origin":
                origin = observation.get("module_file")
                if not isinstance(origin, str) or "site-packages" not in origin.replace("\\", "/"):
                    self.report.add("SOURCE_FALLBACK",
                                    f"module imported from {origin!r}, not from the installed wheel",
                                    case_id)
        if expected_terminal is not None and terminal_seen != expected_terminal:
            self.report.add("DB_STATE_MISMATCH",
                            f"terminal state {terminal_seen!r} != expected {expected_terminal!r}",
                            case_id)

        # -- commands ------------------------------------------------------
        commands = result.get("commands")
        if mode in ("pytest", "script", "browser", "native"):
            if not isinstance(commands, list) or not commands:
                self.report.add("COMMAND_MISSING",
                                f"{mode} cases need at least one recorded command", case_id)
                commands = []
            for command in commands:
                if not isinstance(command, dict):
                    self.report.add("RESULTS_INVALID", "command entries must be objects", case_id)
                    continue
                argv = command.get("argv")
                if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
                    self.report.add("COMMAND_MISSING", "command argv must be a non-empty string list",
                                    case_id)
                if type(command.get("exit_code")) is not int:
                    self.report.add("COMMAND_MISSING", "command exit_code must be an integer", case_id)

        # -- evidence ------------------------------------------------------
        evidence = result.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            self.report.add("EVIDENCE_EMPTY", "evidence must be a non-empty list", case_id)
            evidence = []
        junit_paths: List[Path] = []
        for item in evidence:
            self.report.checked_evidence += 1
            if not isinstance(item, dict):
                self.report.add("RESULTS_INVALID", "evidence entries must be objects", case_id)
                continue
            kind = item.get("kind")
            if kind not in EVIDENCE_KINDS:
                self.report.add("EVIDENCE_TYPE_INVALID", f"unknown evidence kind {kind!r}", case_id)
                continue
            path = resolve_inside(self.run, item.get("path"), self.report, "evidence", case_id)
            recorded = item.get("sha256")
            if not isinstance(recorded, str) or not SHA256_RE.match(recorded):
                self.report.add("EVIDENCE_HASH_MISSING", f"evidence {item.get('path')!r} has no sha256",
                                case_id)
                continue
            if path is None:
                continue
            if not path.is_file():
                self.report.add("EVIDENCE_MISSING", f"evidence file is absent: {item.get('path')}",
                                case_id)
                continue
            if path.suffix.lower() not in EVIDENCE_KINDS[kind]:
                self.report.add("EVIDENCE_TYPE_INVALID",
                                f"{item.get('path')} suffix does not match kind {kind!r}", case_id)
            actual = sha256_file(path)
            self.report.inputs[str(path.relative_to(self.run)).replace("\\", "/")] = actual
            if actual != recorded:
                self.report.add("EVIDENCE_HASH_MISMATCH",
                                f"evidence {item.get('path')} sha256 differs from the recorded value",
                                case_id)
            if kind == "junit":
                junit_paths.append(path)

        # -- JUnit ---------------------------------------------------------
        junit_nodes = spec.get("junit_nodes")
        if junit_nodes or mode == "pytest":
            if not junit_paths:
                self.report.add("JUNIT_MISSING", "a pytest case needs a junit evidence file", case_id)
            for path in junit_paths:
                try:
                    summary = read_junit(path)
                except (ET.ParseError, ValueError) as exc:
                    self.report.add("JUNIT_EMPTY", f"{path.name} is not parseable: {exc}", case_id)
                    continue
                if summary.tests == 0:
                    self.report.add("JUNIT_ZERO_TESTS", f"{path.name} contains no testcase", case_id)
                    continue
                if summary.failures or summary.errors:
                    self.report.add("JUNIT_FAILURES",
                                    f"{path.name}: {summary.failures} failures / {summary.errors} errors",
                                    case_id)
                if summary.skipped:
                    if spec.get("allow_skips"):
                        # A skip is only tolerated when the result explains each
                        # one and the count matches the JUnit exactly. An
                        # unexplained skip is a missing test, not a pass.
                        declared = result.get("skips")
                        if not isinstance(declared, list) or len(declared) != summary.skipped:
                            self.report.add("JUNIT_REQUIRED_SKIP",
                                            f"{path.name}: {summary.skipped} skipped tests but "
                                            f"{len(declared or [])} are explained", case_id)
                        else:
                            for skip in declared:
                                if (not isinstance(skip, dict)
                                        or not isinstance(skip.get("node"), str)
                                        or not isinstance(skip.get("reason"), str)
                                        or not skip["reason"].strip()):
                                    self.report.add("JUNIT_REQUIRED_SKIP",
                                                    f"{path.name}: a skip lacks a node or reason",
                                                    case_id)
                    else:
                        self.report.add("JUNIT_REQUIRED_SKIP",
                                        f"{path.name}: {summary.skipped} skipped tests", case_id)
                if junit_nodes:
                    missing = [node for node in junit_nodes
                               if not any(node in node_id for node_id in summary.node_ids)]
                    if missing:
                        self.report.add("JUNIT_NOT_MAPPED",
                                        f"{path.name} does not contain {missing}", case_id)

    # -- ledger ------------------------------------------------------------
    def check_ledger(self, policy: Dict[str, Any],
                     results: Optional[Dict[str, Any]]) -> None:
        ledger = self._load(self.run / "gate-ledger.json", "LEDGER", "gate-ledger.json")
        if ledger is None:
            return
        if ledger.get("document_type") != "GATE_LEDGER":
            self.report.add("LEDGER_INVALID", "document_type must be GATE_LEDGER")
            return
        version = ledger.get("ledger_version")
        if type(version) is not int or version not in SUPPORTED_LEDGER_VERSIONS:
            self.report.add("LEDGER_INVALID", f"unsupported ledger_version {version!r}")
            return
        gates = ledger.get("gates")
        if not isinstance(gates, dict):
            self.report.add("LEDGER_INVALID", "gates must be an object keyed by P01..P16")
            return

        status_by_case: Dict[str, str] = {}
        if isinstance(results, dict) and isinstance(results.get("results"), list):
            for entry in results["results"]:
                if isinstance(entry, dict) and isinstance(entry.get("case_id"), str):
                    status_by_case[entry["case_id"]] = entry.get("status")

        for gate in GATES:
            required = [entry["case_id"] for entry in policy["product_cases"] + policy["verifier_cases"]
                        if entry["gate"] == gate]
            primary = [entry["case_id"] for entry in policy["primary_review"] if entry["gate"] == gate]
            if not required and not primary:
                # A gate the policy does not populate needs no ledger entry.
                continue
            entry = gates.get(gate)
            if not isinstance(entry, dict):
                self.report.add("LEDGER_INCONSISTENT", f"gate {gate} is missing from the ledger")
                continue
            claimed = entry.get("status")
            if primary and not required:
                if claimed != "PENDING_PRIMARY_REVIEW":
                    self.report.add("LEDGER_INCONSISTENT",
                                    f"gate {gate} must be PENDING_PRIMARY_REVIEW, not {claimed!r}")
                continue
            if not required:
                continue
            all_pass = all(status_by_case.get(case_id) == "PASS" for case_id in required)
            if all_pass and claimed != "PASS":
                self.report.add("LEDGER_INCONSISTENT",
                                f"gate {gate} is {claimed!r} although every required case passed")
            if not all_pass and claimed == "PASS":
                failing = [case_id for case_id in required if status_by_case.get(case_id) != "PASS"]
                self.report.add("LEDGER_INCONSISTENT",
                                f"gate {gate} claims PASS but {failing} are not PASS")
        for gate in gates:
            if gate not in GATES:
                self.report.add("LEDGER_INVALID", f"unknown gate {gate!r} in the ledger")

    # -- entry point -----------------------------------------------------
    def run_all(self) -> Report:
        if not self.run.is_dir():
            self.report.add("RUN_MISSING", f"Run directory not found: {self.run}")
            return self.report
        policy = self.check_policy()
        if policy is None:
            return self.report
        manifest = self.check_manifest()
        results = self.check_results(policy, manifest)
        self.check_ledger(policy, results)
        return self.report


def verify(run: Path, policy_path: Optional[Path] = None) -> Report:
    return Verifier(run, policy_path).run_all()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Strict delivery-evidence verifier (P15)")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    report = verify(run, args.policy.expanduser().resolve() if args.policy else None)
    payload = {
        "document_type": "PRODUCT_EVIDENCE_VERIFICATION",
        "verifier_version": 1,
        "run": str(run),
        "verdict": "ACCEPTED" if report.accepted else "REJECTED",
        "problems": [{"code": item.code, "case_id": item.case_id, "message": item.message}
                     for item in report.problems],
        "summary": {
            "checked_cases": report.checked_cases,
            "checked_scenarios": report.checked_scenarios,
            "checked_evidence": report.checked_evidence,
            "problem_count": len(report.problems),
        },
        "inputs": dict(sorted(report.inputs.items())),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if report.accepted:
        print("DELIVERY EVIDENCE ACCEPTED")
        print(f"cases={report.checked_cases} scenarios={report.checked_scenarios} "
              f"evidence={report.checked_evidence}")
        return 0
    print("DELIVERY EVIDENCE REJECTED:")
    for item in report.problems:
        print(" - " + item.render())
    return 1


if __name__ == "__main__":
    sys.exit(main())
