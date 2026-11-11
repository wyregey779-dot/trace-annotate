"""The 20-check UI runner must bind installed bytes and retain stalled evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_workspace_vision_editor.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("vision_editor_runner_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(output, *arguments, env=None):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output), *arguments],
                          capture_output=True, timeout=30, env=env)


def test_existing_evidence_is_preserved_before_importing_product(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    original = b'{"ok":false,"historical_failure":true}\n'
    (output / "summary.json").write_bytes(original)
    result = invoke(output)
    assert result.returncode == 2
    assert (output / "summary.json").read_bytes() == original
    assert list(output.iterdir()) == [output / "summary.json"]


def test_installed_mode_requires_isolation_and_keeps_bootstrap_failure(tmp_path):
    output = tmp_path / "not-isolated"
    result = invoke(output, "--installed-wheel", str(tmp_path / "candidate.whl"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert summary["status"] == "FAIL" and summary["ok"] is False
    assert "with -I" in summary["error"] and summary["failed_phase"] == "runtime.verify"
    assert summary["installed_identity"] is None and summary["checks"] == []
    assert summary["worker_exit_code"] == 1 and summary["finished_utc"]
    assert (output / "failure.log").is_file() and not (output / "workspace").exists()


def test_missing_media_tools_fails_before_fixtures_or_browser(tmp_path):
    output = tmp_path / "no-media-tools"
    result = invoke(output, env=dict(os.environ, PATH=str(tmp_path)))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert summary["status"] == "FAIL" and summary["ok"] is False
    assert "FFmpeg and ffprobe are required" in summary["error"]
    assert not (output / "workspace").exists() and not (output / "vision-a.mp4").exists()
    assert (output / "worker.stderr.log").is_file() and (output / "failure.log").is_file()


@pytest.mark.parametrize("flag", ["--visible-evidence", "--ready-marker"])
def test_headless_cannot_be_labelled_as_visible_evidence(tmp_path, flag):
    output = tmp_path / "invalid-visible"
    result = invoke(output, "--headless", flag, str(output / "capture.png"))
    assert result.returncode == 2 and b"non-headless" in result.stderr
    assert not output.exists()


def test_evidence_path_must_stay_inside_new_run(tmp_path):
    output = tmp_path / "run"
    result = invoke(output, "--ready-marker", str(tmp_path / "outside.json"))
    assert result.returncode == 2 and b"inside --output" in result.stderr
    assert not output.exists() and not (tmp_path / "outside.json").exists()


@pytest.mark.parametrize("drift", ["changed", "missing", "extra", "empty", "duplicate", "traversal"])
def test_installed_verifier_rejects_all_resource_drift(tmp_path, runner, drift):
    package = tmp_path / "trace_annotate"
    (package / "workspace/fonts").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# package\n")
    resource = package / "workspace/fonts/OFL.txt"
    resource.write_bytes(b"licence bytes")
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        if drift != "empty":
            archive.writestr("trace_annotate/__init__.py", b"# package\n")
            archive.writestr("trace_annotate/workspace/fonts/OFL.txt", b"licence bytes")
        if drift == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("trace_annotate/__init__.py", b"# package\n")
        if drift == "traversal":
            archive.writestr("trace_annotate/../outside.txt", b"outside")
    if drift == "changed":
        resource.write_bytes(b"changed licence")
    elif drift == "missing":
        resource.unlink()
    elif drift == "extra":
        (package / "unexpected.dat").write_bytes(b"unshipped resource")
    with pytest.raises(RuntimeError, match="differs from wheel|Invalid or duplicated"):
        runner.verify_package_resources(package, wheel)


def test_installed_verifier_accepts_exact_resources_ignoring_bytecode(tmp_path, runner):
    package = tmp_path / "trace_annotate"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"# package\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__/__init__.cpython-313.pyc").write_bytes(b"runtime generated")
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trace_annotate/__init__.py", b"# package\n")
    assert list(runner.verify_package_resources(package, wheel)) == ["trace_annotate/__init__.py"]


def test_watchdog_bounds_worker_stalled_before_first_browser_assertion(tmp_path, runner):
    output = tmp_path / "stalled"
    output.mkdir()
    summary = {"status": "RUNNING", "ok": False, "phase": "playwright.start", "checks": []}
    runner.write_json(output / "summary.json", summary)
    args = SimpleNamespace(output=output, timeout_seconds=4, stall_timeout_seconds=0.5)
    code = "import time; print('startup reached', flush=True); time.sleep(60)"
    assert runner.supervise([sys.executable, "-u", "-c", code], args, summary) == 1
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "INCOMPLETE" and result["ok"] is False
    assert result["failed_phase"] == "playwright.start" and "Worker timeout" in result["error"]
    assert result["worker_exit_code"] != 0 and result["checks"] == [] and result["finished_utc"]
    assert b"startup reached" in (output / "worker.stdout.log").read_bytes()
    assert (output / "supervisor-failure.log").is_file()


def test_zero_exit_without_twenty_checks_is_a_failure(tmp_path, runner):
    output = tmp_path / "partial"
    output.mkdir()
    summary = {"status": "PASS", "ok": True, "checks": ["only-one-check"]}
    runner.write_json(output / "summary.json", summary)
    args = SimpleNamespace(output=output, timeout_seconds=4, stall_timeout_seconds=2)
    assert runner.supervise([sys.executable, "-c", "pass"], args, summary) == 1
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL" and result["worker_exit_code"] == 0
    assert "without completing the 20 checks" in result["error"]


def test_worker_start_failure_has_a_final_record(tmp_path, runner):
    output = tmp_path / "cannot-start"
    output.mkdir()
    summary = {"status": "RUNNING", "ok": False, "phase": "worker.starting", "checks": []}
    runner.write_json(output / "summary.json", summary)
    args = SimpleNamespace(output=output, timeout_seconds=4, stall_timeout_seconds=2)
    assert runner.supervise([str(tmp_path / "nonexistent-python")], args, summary) == 1
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL" and result["finished_utc"]
    assert result["failed_phase"] == "worker.starting" and (output / "failure.log").is_file()
