"""Execute Windows launchers against controlled input and failure conditions."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Real Windows CMD launchers")


def _run(script, *args, report_root=None):
    return subprocess.run(["cmd.exe", "/d", "/c", str(script), *map(str, args)],
                          cwd=str(ROOT), capture_output=True, timeout=30,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                               **({"TRACE_QC_REPORT_ROOT": str(report_root)} if report_root else {})})


def _fixture(path, *, broken=False):
    doc = {"clip": "synthetic.mp4", "duration": 10, "text_description": "合成流程测试",
           "clip_status": "正常", "working": [{"start": 1, "end": 3,
           "step_number": 1, "verdict": "对"}], "idle": []}
    if broken:
        doc["working"][0]["start"] = -1
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path.read_bytes()


def test_cmd_launchers_have_valid_line_endings_and_do_not_claim_browser_readiness():
    for name in ("windows_qc.cmd", "windows_workbench.cmd"):
        raw = (ROOT / "scripts" / name).read_bytes()
        raw.decode("gbk", errors="strict")
        assert b"\r\r\n" not in raw
        assert raw.count(b"\r\n") == raw.count(b"\n")
    result = _run(ROOT / "scripts/windows_workbench.cmd", "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"Browser execution has not been checked" in result.stdout


def test_batch_launcher_propagates_quality_failure_and_keeps_original_input(tmp_path):
    target = tmp_path / "bad annotation.json"
    original = _fixture(target, broken=True)
    report_root = tmp_path / "reports"
    result = _run(ROOT / "scripts/windows_qc.cmd", target, "1", "--no-open", report_root=report_root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert b"[FAIL]" in result.stdout and b"[OK] Automated gates" not in result.stdout
    assert target.read_bytes() == original
    reports = list(report_root.glob("run-*/qc_report.html"))
    assert len(reports) == 1
    assert "INVALID_TIME" in reports[0].read_text(encoding="utf-8")


def test_batch_launcher_runs_valid_batch_and_missing_input_without_opening_a_browser(tmp_path):
    target = tmp_path / "valid annotation.json"
    original = _fixture(target)
    report_root = tmp_path / "reports"
    result = _run(ROOT / "scripts/windows_qc.cmd", target, "1", "--no-open", report_root=report_root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"[OK] Automated gates passed" in result.stdout
    assert target.read_bytes() == original
    reports = list(report_root.glob("run-*/qc_report.html"))
    assert len(reports) == 1
    assert "PASS_SCOPED" in reports[0].read_text(encoding="utf-8")
    missing = _run(ROOT / "scripts/windows_qc.cmd", tmp_path / "missing.json", "--no-open")
    assert missing.returncode == 2
    assert b"Input does not exist" in missing.stdout
