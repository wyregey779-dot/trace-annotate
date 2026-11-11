"""OCR extras select a runtime supported by each declared Python version."""
import ast
from pathlib import Path
import re

from packaging.requirements import Requirement
import pytest


@pytest.mark.parametrize("extra", ["ocr", "full"])
@pytest.mark.parametrize("python_version,expected", [
    ("3.9", "rapidocr-onnxruntime"),
    ("3.11", "rapidocr-onnxruntime"),
    ("3.13", "rapidocr"),
])
def test_ocr_extra_selects_supported_runtime(extra, python_version, expected):
    # These dependency arrays contain only string literals and Python comments.
    # Reading their literal values keeps this test usable on Python 3.9 too.
    metadata = (Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8")
    match = re.search(r"^" + extra + r"\s*=\s*(\[[\s\S]*?\])", metadata, re.MULTILINE)
    assert match is not None
    requirements = [Requirement(value) for value in ast.literal_eval(match.group(1))]
    active = [requirement for requirement in requirements
              if requirement.name.startswith("rapidocr")
              and (requirement.marker is None or requirement.marker.evaluate(
                  {"python_version": python_version,
                   "python_full_version": python_version + ".0"}))]
    assert len(active) == 1
    assert active[0].name == expected
    if expected == "rapidocr":
        assert "3.10.0" in active[0].specifier
        assert "4.0.0" not in active[0].specifier
