"""Render acceptance runner must not overwrite evidence or fake early success."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_workspace_render.py"


def test_render_runner_preserves_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    original = b'{"status":"FAIL","keep":true}\n'
    (output / "result.json").write_bytes(original)
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output)], capture_output=True, timeout=30)
    assert result.returncode == 2
    assert (output / "result.json").read_bytes() == original
    assert list(output.iterdir()) == [output / "result.json"]


def test_installed_render_runner_requires_isolation_before_product_start(tmp_path):
    output = tmp_path / "not-isolated"
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--installed-wheel", str(tmp_path / "candidate.whl"),
                             "--output", str(output)], capture_output=True, timeout=30)
    assert result.returncode == 1
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == "FAIL" and "with -I" in record["error"]
    assert record["finished_utc"] and record["installed_identity"] is None
    assert (output / "failure.log").is_file() and not (output / "workspace").exists()


def test_render_runner_missing_media_tools_fails_without_skip(tmp_path):
    output = tmp_path / "missing-ffmpeg"
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output)],
                            env=dict(os.environ, PATH=str(tmp_path)), capture_output=True, timeout=30)
    assert result.returncode == 1
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == "FAIL" and "FFmpeg and ffprobe are required" in record["error"]
    assert record["finished_utc"] and (output / "failure.log").is_file()
    assert not (output / "workspace").exists()


@pytest.mark.parametrize("error", [RuntimeError("bootstrap fault"), KeyboardInterrupt()])
def test_render_runner_early_fault_preserves_failure_and_environment(tmp_path, monkeypatch, error):
    spec = importlib.util.spec_from_file_location("render_acceptance_runner_under_test", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "new-output"
    keys = ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME")
    before = {key: os.environ.get(key) for key in keys}
    bytecode = sys.dont_write_bytecode
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    def fail(_wheel):
        raise error
    monkeypatch.setattr(runner, "load_harness", fail)
    assert runner.main() == 1
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == ("INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "FAIL")
    assert record["finished_utc"] and (output / "failure.log").is_file()
    assert record["cancellation_tested"] is False and not (output / "workspace").exists()
    assert {key: os.environ.get(key) for key in keys} == before and sys.dont_write_bytecode == bytecode


@pytest.mark.parametrize("tamper", ["changed", "extra", "missing", "empty-wheel"])
def test_render_runner_rejects_installed_font_and_licence_drift(tmp_path, tamper):
    spec = importlib.util.spec_from_file_location("render_font_verification_under_test", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    package = tmp_path / "trace_annotate"
    folder = package / "workspace/fonts"
    folder.mkdir(parents=True)
    (folder / "OFL.txt").write_bytes(b"actual licence bytes")
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        if tamper != "empty-wheel":
            archive.writestr("trace_annotate/workspace/fonts/OFL.txt", b"actual licence bytes")
    if tamper == "changed":
        (folder / "OFL.txt").write_bytes(b"changed licence")
    elif tamper == "extra":
        (folder / "other.otf").write_bytes(b"unverified extra font")
    elif tamper == "missing":
        (folder / "OFL.txt").unlink()
    with pytest.raises(RuntimeError, match="differs from wheel"):
        runner.verify_font_resources(package, wheel)
