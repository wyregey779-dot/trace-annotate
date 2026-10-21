"""Regressions for file formats, timeline duration, and filesystem boundaries."""
import json
import os
import subprocess
from argparse import Namespace

import pytest

from trace_annotate.cli import cmd_embodied_idle, main
from trace_annotate.config import StagingConfig
from trace_annotate.core.staging import StagingManager, StagingSecurityError
from trace_annotate.embodied.exporters.action_dataset import export_to_action_segmentation
from trace_annotate.embodied.validator import validate_embodied_doc, validate_embodied_export


def _doc():
    return {
        "clip": "example.mp4",
        "duration": 10.0,
        "clip_status": "正常",
        "working": [{"start": 1.0, "end": 2.0, "step_number": 1, "verdict": "对"}],
    }


@pytest.mark.parametrize("as_list", [False, True])
def test_validator_reads_pretty_json_documents(tmp_path, as_list):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([_doc()] if as_list else _doc(), indent=2), encoding="utf-8")
    report = validate_embodied_export(str(path))
    assert report["passed"] is True
    assert report["records_checked"] == 1


def test_validator_reports_bad_jsonl_line_and_checks_remaining_records(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(_doc()) + "\ninvalid\n" + json.dumps(_doc()) + "\n", encoding="utf-8")
    report = validate_embodied_export(str(path))
    assert report["passed"] is False
    assert report["records_checked"] == 2
    assert report["violations_count"] == 1
    assert ":2: JSON decode error" in report["violations"][0]


def test_validator_rejects_non_object_records_without_crashing(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text('[42, null, "not an annotation"]', encoding="utf-8")
    report = validate_embodied_export(str(path))
    assert report["passed"] is False
    assert report["violations_count"] == 3


@pytest.mark.parametrize("document", [
    {},
    {"clip": "example.mp4"},
    {"clip": "", "working": []},
    {"clip": 123, "working": []},
    {"clip": "example.mp4", "working": None},
    {"clip": "example.mp4", "working": {}},
    {"clip": "example.mp4", "working": [None]},
    {"clip": "example.mp4", "clip_status": "佩戴问题", "working": [], "report": 123},
])
def test_validator_reports_invalid_minimum_schema_without_crashing(document):
    assert validate_embodied_doc(document)


@pytest.mark.parametrize("step", [True, False, 0, -1])
def test_validator_requires_positive_integer_step_excluding_boolean(step):
    document = _doc()
    document["working"][0]["step_number"] = step
    assert any("step_number must be a positive integer" in error for error in validate_embodied_doc(document))


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 1.5])
def test_validator_rejects_invalid_or_truncated_document_duration(duration):
    document = _doc()
    document["duration"] = duration
    assert any("duration" in error for error in validate_embodied_doc(document))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validator_rejects_nonfinite_times(value):
    doc = _doc()
    doc["working"][0]["end"] = value
    assert any("finite" in message for message in validate_embodied_doc(doc))


@pytest.mark.parametrize("override, expected_end", [(None, 10.0), (12.0, 12.0)])
def test_idle_cli_preserves_tail_using_document_duration(tmp_path, capsys, override, expected_end):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    cmd_embodied_idle(Namespace(file=str(path), duration=override, inplace=False))
    result = json.loads(capsys.readouterr().out)
    assert result["idle"] == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": expected_end}]


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 1.5])
@pytest.mark.parametrize("source", ["argument", "document"])
def test_idle_rejects_invalid_duration_before_inplace_write(tmp_path, duration, source):
    document = _doc()
    if source == "document":
        document["duration"] = duration
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    args = Namespace(file=str(path), duration=duration if source == "argument" else None, inplace=True)
    with pytest.raises(ValueError, match="duration"):
        cmd_embodied_idle(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize("duration", ["0", "nan", "inf", "-1"])
def test_idle_cli_parser_reports_invalid_duration(tmp_path, monkeypatch, capsys, duration):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["trace-annotate", "embodied-idle", str(path), "--duration", duration])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert "duration must be a finite positive number" in capsys.readouterr().err


@pytest.mark.parametrize("clip", ["../escaped.mp4", "sub/clip.mp4", r"sub\clip.mp4", r"C:\escaped.mp4", "/escaped.mp4", "..", "a:b.mp4", ""])
def test_action_export_rejects_unsafe_clip_before_any_write(tmp_path, clip):
    output = tmp_path / "dataset"
    doc = _doc()
    doc["clip"] = clip
    with pytest.raises(ValueError):
        export_to_action_segmentation([doc], str(output))
    assert not output.exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_action_export_rejects_colliding_output_names_before_write(tmp_path):
    output = tmp_path / "dataset"
    doc_a, doc_b = _doc(), _doc()
    doc_b["clip"] = "example.mov"
    with pytest.raises(ValueError, match="unique"):
        export_to_action_segmentation([doc_a, doc_b], str(output))
    assert not output.exists()


def _directory_link(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"Directory links unavailable: {exc}")
        # Windows junctions require no Developer Mode or elevated token.
        quoted_link = str(link).replace("'", "''")
        quoted_target = str(target).replace("'", "''")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             f"New-Item -ItemType Junction -Path '{quoted_link}' -Target '{quoted_target}' | Out-Null"],
            capture_output=True, text=True,
        )
        if result.returncode:
            pytest.skip("Unable to create test-only Windows junction")


def test_staging_rejects_directory_link_escape_without_modifying_outside_file(tmp_path):
    source, staging, outside = (tmp_path / name for name in ("source", "staging", "outside"))
    for folder in (source, staging, outside):
        folder.mkdir()
    (source / "redirect").mkdir()
    (source / "redirect" / "asset.txt").write_text("source", encoding="utf-8")
    sentinel = outside / "asset.txt"
    sentinel.write_text("keep outside unchanged", encoding="utf-8")
    link = staging / "redirect"
    _directory_link(link, outside)
    manager = StagingManager(str(source), str(staging), StagingConfig(free_disk_guard_bytes=0))
    try:
        with pytest.raises(StagingSecurityError):
            list(manager.stage_batch(["redirect/asset.txt"]))
        assert sentinel.read_text(encoding="utf-8") == "keep outside unchanged"
    finally:
        if os.path.islink(link):
            link.unlink()
        elif os.path.lexists(link):
            os.rmdir(link)


def test_staging_atomic_replacement_does_not_write_through_existing_hardlink(tmp_path):
    source, staging = (tmp_path / name for name in ("source", "staging"))
    source.mkdir()
    staging.mkdir()
    (source / "asset.txt").write_text("source", encoding="utf-8")
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("keep outside unchanged", encoding="utf-8")
    try:
        os.link(sentinel, staging / "asset.txt")
    except OSError as exc:
        pytest.skip(f"Hard links unavailable: {exc}")
    manager = StagingManager(str(source), str(staging), StagingConfig(free_disk_guard_bytes=0))
    for batch in manager.stage_batch(["asset.txt"]):
        with open(batch["asset.txt"], encoding="utf-8") as stream:
            assert stream.read() == "source"
        assert sentinel.read_text(encoding="utf-8") == "keep outside unchanged"
    assert sentinel.read_text(encoding="utf-8") == "keep outside unchanged"
    assert not (staging / "asset.txt").exists()


def test_staging_cleanup_rechecks_directory_links_inserted_during_processing(tmp_path):
    source, staging, outside = (tmp_path / name for name in ("source", "staging", "outside"))
    for folder in (source, staging, outside):
        folder.mkdir()
    (source / "work").mkdir()
    (source / "work" / "asset.txt").write_text("source", encoding="utf-8")
    sentinel = outside / "asset.txt"
    sentinel.write_text("keep outside unchanged", encoding="utf-8")
    manager = StagingManager(str(source), str(staging), StagingConfig(free_disk_guard_bytes=0))
    batches = manager.stage_batch(["work/asset.txt"])
    next(batches)
    (staging / "work").rename(staging / "original-work")
    link = staging / "work"
    _directory_link(link, outside)
    try:
        with pytest.raises(StagingSecurityError):
            next(batches)
        assert sentinel.read_text(encoding="utf-8") == "keep outside unchanged"
    finally:
        if os.path.islink(link):
            link.unlink()
        elif os.path.lexists(link):
            os.rmdir(link)
