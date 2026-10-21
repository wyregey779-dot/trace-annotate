"""Regressions for file formats, timeline duration, and filesystem boundaries."""
import json
import os
import posixpath
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


def test_validator_checks_all_segment_fields_despite_nonfinite_times():
    document = _doc()
    document["working"] = [
        {"start": 0, "end": float("nan"), "step_number": True, "verdict": "invalid"},
        {"start": float("inf"), "end": 4, "step_number": None,
         "verdict": "错", "error_type": "invalid", "evidence": ""},
    ]
    errors = validate_embodied_doc(document)
    for index in (0, 1):
        segment_errors = [error for error in errors if f"working[{index}]" in error]
        assert any("finite" in error for error in segment_errors)
        assert any("step_number" in error for error in segment_errors)
    assert any("Invalid verdict" in error and "working[0]" in error for error in errors)
    assert any("error_type" in error and "working[1]" in error for error in errors)
    assert any("evidence" in error and "working[1]" in error for error in errors)
    assert not any("overlap in time" in error for error in errors)


@pytest.mark.parametrize("override, expected_end", [(None, 10.0), (12.0, 12.0)])
def test_idle_cli_preserves_tail_using_document_duration(tmp_path, capsys, override, expected_end):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    cmd_embodied_idle(Namespace(file=str(path), duration=override, inplace=False))
    result = json.loads(capsys.readouterr().out)
    assert result["duration"] == expected_end
    assert result["idle"] == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": expected_end}]


@pytest.mark.parametrize("inplace", [False, True])
def test_idle_duration_override_survives_action_dataset_export(tmp_path, capsys, inplace):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    cmd_embodied_idle(Namespace(file=str(path), duration=12.0, inplace=inplace))
    printed = capsys.readouterr().out
    document = json.loads(path.read_text(encoding="utf-8") if inplace else printed)
    assert document["duration"] == 12.0
    assert document["idle"][-1]["end"] == 12.0
    _, ground_truth = export_to_action_segmentation([document], str(tmp_path / "dataset"), fps=1.0)
    with open(ground_truth[0], encoding="utf-8") as stream:
        labels = stream.read().splitlines()
    assert len(labels) == 12
    assert labels == ["idle", "step_1"] + ["idle"] * 10


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


@pytest.mark.parametrize("first, second", [
    ("Task.mp4", "task.mov"),
    ("Caf\u00e9.mp4", "Cafe\u0301.mov"),
    ("Stra\u00dfe.mp4", "STRASSE.mov"),
])
def test_action_export_rejects_portable_name_collisions_before_writes(tmp_path, monkeypatch, first, second):
    output = tmp_path / "dataset"
    doc_a, doc_b = _doc(), _doc()
    doc_a["clip"], doc_b["clip"] = first, second
    # POSIX normcase does not fold case, including on macOS volumes whose
    # filesystem itself treats case aliases as a single output file.
    with monkeypatch.context() as context:
        context.setattr("trace_annotate.embodied.exporters.action_dataset.os.path.normcase", posixpath.normcase)
        with pytest.raises(ValueError, match="unique"):
            export_to_action_segmentation([doc_a, doc_b], str(output))
    assert not output.exists()


@pytest.mark.parametrize("device", ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"] + [
    prefix + digit for prefix in ("COM", "LPT") for digit in "123456789¹²³"
])
def test_action_export_rejects_windows_devices_with_or_without_extensions(tmp_path, device):
    for index, extension in enumerate(("", ".mp4", ".backup.mp4")):
        output = tmp_path / f"dataset-{index}"
        document = _doc()
        document["clip"] = device.lower() + extension
        with pytest.raises(ValueError, match="reserved Windows device"):
            export_to_action_segmentation([document], str(output))
        assert not output.exists()


@pytest.mark.parametrize("clip", [
    "clip.mp4 ", "clip.mp4.", "clip .mp4", "clip..mp4",
    "NUL .mp4", "NUL .backup.mp4", "clip\t.mp4", "clip?.mp4",
])
def test_action_export_rejects_windows_name_aliases_before_writes(tmp_path, clip):
    output = tmp_path / "dataset"
    document = _doc()
    document["clip"] = clip
    with pytest.raises(ValueError):
        export_to_action_segmentation([document], str(output))
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
