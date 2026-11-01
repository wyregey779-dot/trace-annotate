"""Acceptance infrastructure must preserve evidence and reject unsafe runners."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_workspace_analysis.py"


def test_analysis_runner_preserves_existing_output_without_starting_product(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    original = b'{"status":"FAIL","keep":true}\n'
    (output / "result.json").write_bytes(original)
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output)], capture_output=True, timeout=30)
    assert result.returncode == 2
    assert (output / "result.json").read_bytes() == original
    assert list(output.iterdir()) == [output / "result.json"]


def test_installed_analysis_runner_refuses_checkout_fallback_and_records_failure(tmp_path):
    output = tmp_path / "not-isolated"
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--installed-wheel", str(tmp_path / "candidate.whl"),
                             "--output", str(output)], capture_output=True, timeout=30)
    assert result.returncode == 1
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == "FAIL" and "with -I" in record["error"]
    assert record["finished_utc"] and record["installed_identity"] is None
    assert (output / "failure.log").is_file()
    assert not (output / "workspace").exists()


def test_analysis_runner_missing_media_tool_is_not_a_skipped_pass(tmp_path):
    output = tmp_path / "missing-ffmpeg"
    environment = dict(os.environ, PATH=str(tmp_path))
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output)],
                            env=environment, capture_output=True, timeout=30)
    assert result.returncode == 1
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == "FAIL" and "FFmpeg and ffprobe are required" in record["error"]
    assert record["finished_utc"] and (output / "failure.log").is_file()
    assert not (output / "workspace").exists()


@pytest.mark.parametrize("error", [RuntimeError("bootstrap fault"), KeyboardInterrupt()])
def test_analysis_runner_early_fault_never_reports_completed_work(tmp_path, monkeypatch, error):
    spec = importlib.util.spec_from_file_location("analysis_acceptance_runner_under_test", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "new-output"
    old_environment = {key: os.environ.get(key) for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME")}
    old_bytecode = sys.dont_write_bytecode
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    def fail(_wheel):
        raise error
    monkeypatch.setattr(runner, "load_harness", fail)
    assert runner.main() == 1
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == ("INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "FAIL")
    assert record["finished_utc"] and (output / "failure.log").is_file()
    assert record["cancellation_tested"] is False
    assert not (output / "workspace").exists()
    assert {key: os.environ.get(key) for key in old_environment} == old_environment
    assert sys.dont_write_bytecode == old_bytecode
