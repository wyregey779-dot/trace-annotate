from __future__ import annotations

import pytest

from scripts.check_visual_pipeline_real import (
    _artifact_summary,
    _execution_summary,
    _model_run_summary,
    _write_json,
)


def test_artifact_summary_matches_professional_exchange_artifact_shape() -> None:
    document = {
        "artifacts": [
            {
                "id": "art-b",
                "relative_path": "objects/b",
                "sha256": "b" * 64,
                "size_bytes": "2",
            },
            {
                "id": "art-a",
                "relative_path": "objects/a",
                "sha256": "a" * 64,
                "size_bytes": "1",
            },
        ]
    }

    assert _artifact_summary(document) == [
        {"id": "art-a", "sha256": "a" * 64, "size_bytes": "1"},
        {"id": "art-b", "sha256": "b" * 64, "size_bytes": "2"},
    ]


def test_public_model_run_summary_redacts_local_paths() -> None:
    summary = _model_run_summary({
        "id": "run-a",
        "provider": "unavailable",
        "model_name": r"D:\private\weights",
        "model_version": "unavailable",
        "weights_sha256": None,
        "license": "not declared",
        "input_sha256": "a" * 64,
        "output_sha256": None,
        "status": "not_run",
        "failure_reason": "weights_unavailable",
        "score_kind": "none",
    })

    assert summary["model_name"] == "<redacted-local-path>"


def test_execution_summary_records_exit_and_sanitized_command() -> None:
    summary = _execution_summary(
        window_seconds=8, exit_code=0, duration_seconds=12.3456789)

    assert summary["exit_code"] == 0
    assert summary["duration_seconds"] == 12.345679
    assert summary["command"][-2:] == ["--window-seconds", "8"]
    assert not any(":" in item or "/home/" in item
                   for item in summary["command"])


@pytest.mark.parametrize("private_path", [
    r"D:\private\weights.bin",
    "D:/private/weights.bin",
    r"\\server\share\weights.bin",
    "//server/share/weights.bin",
])
def test_public_report_rejects_windows_and_unc_paths(tmp_path, private_path) -> None:
    with pytest.raises(ValueError, match="public_report_contains_absolute_path"):
        _write_json(tmp_path / "public.json", {"value": private_path})


def test_public_report_path_check_does_not_reject_https_url(tmp_path) -> None:
    output = tmp_path / "public.json"
    _write_json(output, {"value": "https://example.org/models/card"})
    assert output.is_file()
