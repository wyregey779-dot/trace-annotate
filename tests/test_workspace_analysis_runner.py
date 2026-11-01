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
    old_environment = {key: os.environ.get(key) for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME", "PLAYWRIGHT_BROWSERS_PATH")}
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


@pytest.mark.parametrize("script_name", ["check_workspace_analysis.py", "check_workspace_render.py"])
@pytest.mark.parametrize("platform,cache_variable,cache_folder", [
    ("linux", "XDG_CACHE_HOME", "original-xdg"),
    ("win32", "LOCALAPPDATA", "original-localappdata"),
    ("darwin", None, "home/Library/Caches"),
    ("linux", None, "home/.cache"),
])
def test_browser_registry_keeps_original_installation_when_runtime_cache_is_isolated(
        tmp_path, monkeypatch, script_name, platform, cache_variable, cache_folder):
    script = SCRIPT.with_name(script_name)
    spec = importlib.util.spec_from_file_location("browser_cache_runner_under_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for key in ("PLAYWRIGHT_BROWSERS_PATH", "npm_config_playwright_browsers_path",
                "npm_package_config_playwright_browsers_path", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runner.sys, "platform", platform)
    monkeypatch.setattr(runner.Path, "home", lambda: tmp_path / "home")
    if cache_variable:
        monkeypatch.setenv(cache_variable, str(tmp_path / cache_folder))
    original = str(tmp_path / cache_folder / "ms-playwright")
    record = runner.preserve_browser_installation()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "isolated-run-cache"))
    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == original
    assert record == {"source": "original_platform_cache", "path": original, "read_only_dependency": True}
    assert not (tmp_path / "isolated-run-cache").exists(), "Resolving dependencies must not install or create browsers"


@pytest.mark.parametrize("script_name", ["check_workspace_analysis.py", "check_workspace_render.py"])
@pytest.mark.parametrize("key,value", [
    ("PLAYWRIGHT_BROWSERS_PATH", "explicit-registry"),
    ("PLAYWRIGHT_BROWSERS_PATH", "0"),
    ("npm_config_playwright_browsers_path", "npm-registry"),
    ("npm_package_config_playwright_browsers_path", "package-registry"),
])
def test_browser_registry_preserves_explicit_and_hermetic_installation(tmp_path, monkeypatch, script_name, key, value):
    spec = importlib.util.spec_from_file_location("explicit_browser_cache_runner_under_test", SCRIPT.with_name(script_name))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for name in ("PLAYWRIGHT_BROWSERS_PATH", "npm_config_playwright_browsers_path",
                 "npm_package_config_playwright_browsers_path"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, value)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "original-cache"))
    record = runner.preserve_browser_installation()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "isolated-run-cache"))
    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == value
    assert record["source"] == key and record["path"] == value
