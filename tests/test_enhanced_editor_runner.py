"""Faults in acceptance infrastructure must not erase evidence or report PASS."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_enhanced_editor.py"


def test_existing_enhancement_evidence_is_not_replaced(tmp_path):
    output = tmp_path / "existing-evidence"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_bytes(b'{"status":"FAIL","original":true}\n')
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output)],
                            capture_output=True, timeout=30)
    assert result.returncode == 2
    assert sentinel.read_bytes() == b'{"status":"FAIL","original":true}\n'
    assert list(output.iterdir()) == [sentinel]


def test_installed_enhancement_refuses_nonisolated_python_and_records_failure(tmp_path):
    output = tmp_path / "wrong-environment"
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output),
                             "--installed-wheel", str(tmp_path / "candidate.whl")],
                            capture_output=True, timeout=30)
    assert result.returncode == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert "with -I" in summary["error"]
    assert summary["finished_utc"]
    assert "RuntimeError" in (output / "failure.log").read_text(encoding="utf-8")
    assert not (output / "workspace").exists()


def test_missing_media_prerequisite_is_failed_with_persistent_evidence(tmp_path):
    output = tmp_path / "missing-ffmpeg"
    environment = dict(os.environ, PATH=str(tmp_path))
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output)],
                            env=environment, capture_output=True, timeout=30)
    assert result.returncode == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert "FFmpeg and ffprobe are required" in summary["error"]
    assert summary["finished_utc"]
    assert (output / "failure.log").is_file()
    assert not (output / "workspace").exists()


@pytest.mark.parametrize("interruption", [RuntimeError("fixture startup failed"), KeyboardInterrupt()])
def test_early_bootstrap_failure_or_interruption_never_reports_pass(tmp_path, monkeypatch, interruption):
    spec = importlib.util.spec_from_file_location("enhancement_runner_under_test", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "new-evidence"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])

    def fail_before_browser(_wheel):
        raise interruption

    monkeypatch.setattr(runner, "load_harness", fail_before_browser)
    assert runner.main() == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    expected = "INCOMPLETE" if isinstance(interruption, KeyboardInterrupt) else "FAIL"
    assert summary["status"] == expected
    assert summary["finished_utc"]
    assert (output / "failure.log").is_file()
