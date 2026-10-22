"""Exercise the published getting-started workflow through its CLI processes."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "run_quickstart.py"


def run_example(destination):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(destination)],
        cwd=ROOT, capture_output=True, encoding="utf-8", env=environment,
    )


def test_documented_workflow_produces_outputs_without_changing_fixture(tmp_path):
    fixture = ROOT / "examples" / "annotations" / "assembly.json"
    original = fixture.read_bytes()
    destination = tmp_path / "example"
    result = run_example(destination)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Violations: 0" in result.stdout
    assert "5 subtitle entries" in result.stdout
    assert (destination / "qc-report.html").is_file()
    assert fixture.read_bytes() == original


def test_rerunning_workflow_refuses_to_overwrite_existing_output(tmp_path):
    destination = tmp_path / "example"
    destination.mkdir()
    sentinel = destination / "assembly.json"
    sentinel.write_text("my existing annotation", encoding="utf-8")
    result = run_example(destination)
    assert result.returncode != 0
    assert "destination already exists" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "my existing annotation"
    assert list(destination.iterdir()) == [sentinel]
