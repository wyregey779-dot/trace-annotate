"""Strict Unicode admission and output creation must preserve user files."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trace_annotate.embodied.quality_io import (
    QualityInputError, _decode_json, collect_annotations,
    replace_annotation_explicitly, write_new_text,
)


@pytest.mark.parametrize("payload", [
    r'{"text_description":"\ud800"}',
    r'{"text_description":"\udfff"}',
    r'{"metadata":[{"note":"prefix\ud800suffix"}]}',
    r'{"\ud800":"invalid key"}',
    r'{"metadata":{"\udfff":"invalid nested key"}}',
    r'{"metadata":"\ud800\ud800"}',
])
def test_json_boundary_rejects_unpaired_surrogates_in_values_and_keys(payload):
    with pytest.raises(ValueError, match="unpaired Unicode surrogate") as error:
        _decode_json(payload)
    # The diagnostic must itself remain safe for UTF-8 CLI/report output.
    str(error.value).encode("utf-8")


def test_json_boundary_accepts_valid_surrogate_pairs_and_normal_unicode():
    parsed = _decode_json(r'{"\ud83d\ude80":"标注 \ud83d\ude80 café"}')
    assert parsed == {"🚀": "标注 🚀 café"}
    assert _decode_json(json.dumps(parsed, ensure_ascii=False)) == parsed


def test_bad_unicode_jsonl_invalidates_whole_file_and_retains_physical_line(tmp_path):
    source = tmp_path / "input.jsonl"
    original = b'\n{"clip":"valid"}\n{"clip":"\\ud800"}\n'
    source.write_bytes(original)
    batch = collect_annotations(str(source))
    assert batch.records == []
    assert any(":3:" in error and "unpaired Unicode surrogate" in error for error in batch.errors)
    assert source.read_bytes() == original


@pytest.mark.parametrize("command,option", [
    ("embodied-pipeline", "--report"), ("embodied-qc", "--output"),
])
def test_cli_reports_unicode_input_error_without_mutating_source(tmp_path, command, option):
    source = tmp_path / "input.json"
    source.write_bytes(b'{"clip":"sample.mp4","text_description":"\\ud800"}')
    original = source.read_bytes()
    output = tmp_path / "report.html"
    result = subprocess.run(
        [sys.executable, "-B", "-c", "from trace_annotate.cli import main; main()",
         command, str(source), option, str(output)],
        cwd=str(Path(__file__).parents[1]), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "INPUT_ERROR" and summary["passed"] is False
    html = output.read_text(encoding="utf-8")
    assert "INPUT_ERROR" in html and "unpaired Unicode surrogate" in html
    assert "Traceback" not in result.stderr
    assert source.read_bytes() == original


def test_unencodable_output_never_creates_a_file_or_parent_directory(tmp_path):
    output = tmp_path / "new-parent" / "report.html"
    with pytest.raises(QualityInputError, match="unpaired Unicode surrogate"):
        write_new_text(str(output), "report \ud800")
    assert not output.exists() and not output.parent.exists()
    write_new_text(str(output), "valid replacement request 🚀\n")
    assert output.read_bytes() == "valid replacement request 🚀\n".encode("utf-8")


def test_output_preencoding_keeps_existing_and_input_protection(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"user source")
    with pytest.raises(QualityInputError, match="already exists"):
        write_new_text(str(source), "bad \ud800", [str(source)])
    assert source.read_bytes() == b"user source"
    output = tmp_path / "output.html"
    write_new_text(str(output), "中文 🚀\r\nnext\n")
    assert output.read_bytes() == "中文 🚀\r\nnext\n".encode("utf-8")
    with pytest.raises(QualityInputError, match="already exists"):
        write_new_text(str(output), "replacement")
    assert output.read_bytes() == "中文 🚀\r\nnext\n".encode("utf-8")


def test_unencodable_inplace_output_preserves_source_without_backup_or_temp(tmp_path):
    source = tmp_path / "input.json"
    original = b'{"clip":"sample"}'
    source.write_bytes(original)
    with pytest.raises(QualityInputError, match="unpaired Unicode surrogate"):
        replace_annotation_explicitly(str(source), original, "bad \ud800")
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]
