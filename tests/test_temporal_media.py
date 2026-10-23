"""Media identity validation, explicit migration and real CLI file workflows."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from trace_annotate.temporal.formats import export_csv, export_subtitles, load_document
from trace_annotate.temporal.commands import _run
from trace_annotate.temporal.media import (
    MAX_CHUNK_SIZE, MediaIdentity, bind_document, identify_media, stream_sha256,
)
from trace_annotate.temporal.schema import MAX_SAFE_INTEGER, validate_document


def document(**updates):
    result = {
        "kind": "temporal", "schema_version": 1, "clip": "media.bin",
        "duration": 12, "media_size": 10,
        "segments": [{"id": "stable-😀", "start": 1.25, "end": 3.5,
                      "label": "第一行\n第二行", "text": "a\r\nb",
                      "tags": ["换\n行", "😀"], "track": "A\nB", "review_status": "reviewed"}],
        "metadata": {"unchanged": {"unicode": "中文 😀", "numbers": [None, True, 0.5]}},
    }
    result.update(updates)
    return result


def identity(**updates):
    values = {"clip": "media.bin", "duration": 12, "media_size": 10,
              "media_sha256": hashlib.sha256(b"0123456789").hexdigest()}
    values.update(updates)
    return MediaIdentity(**values)


def strong_document(**updates):
    result = document(schema_version=2, media_sha256=identity().media_sha256)
    result.update(updates)
    return result


def test_v1_schema_is_unchanged_and_does_not_accept_a_hash_extension():
    weak = document()
    assert validate_document(weak) == []
    del weak["media_size"]
    assert validate_document(weak) == []
    weak["media_sha256"] = identity().media_sha256
    assert any("unknown field 'media_sha256'" in error for error in validate_document(weak))


def test_v2_native_round_trip_keeps_all_values_and_remains_exportable(tmp_path):
    original = strong_document()
    path = tmp_path / "strong.json"
    path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()
    assert validate_document(original) == []
    assert load_document(path) == original
    assert path.read_bytes() == before
    assert "stable-😀" in export_csv(original)
    # Subtitle is explicitly lossy and does not change the identity or review.
    assert "a\nb" in export_subtitles(original, "vtt")
    assert original == strong_document()


@pytest.mark.parametrize("missing", ["media_size", "media_sha256"])
def test_v2_requires_both_identity_fields(missing):
    doc = strong_document()
    del doc[missing]
    assert any(missing in error for error in validate_document(doc))


@pytest.mark.parametrize("value", [None, True, 123, [], "", "0" * 63, "0" * 65,
                                    "G" * 64, "A" * 64, "0" * 63 + "\n", "é" * 64])
def test_v2_rejects_noncanonical_hash(value):
    assert any("media_sha256" in error for error in validate_document(strong_document(media_sha256=value)))


@pytest.mark.parametrize("value", [None, True, -1, 0.0, float("nan"), float("inf"),
                                    MAX_SAFE_INTEGER + 1, "10"])
def test_v2_rejects_invalid_size(value):
    assert any("media_size" in error for error in validate_document(strong_document(media_size=value)))


@pytest.mark.parametrize("value", [True, 1.0, 2.0, 0, 3, [], {}, None])
def test_schema_versions_are_exact_integers(value):
    assert any("schema_version" in error for error in validate_document(strong_document(schema_version=value)))


def test_v2_rejects_unknown_document_and_segment_fields():
    doc = strong_document(digest="must not disappear")
    doc["segments"][0]["confidence"] = 1
    errors = "\n".join(validate_document(doc))
    assert "digest" in errors and "confidence" in errors


def test_v1_migration_is_explicit_resets_reviews_and_preserves_every_other_value():
    original = document()
    original["segments"].append(dict(original["segments"][0], id="second", review_status="unreviewed"))
    before = copy.deepcopy(original)
    bound = bind_document(original, identity(duration=12.0009))
    expected = copy.deepcopy(before)
    expected.update(schema_version=2, media_sha256=identity().media_sha256)
    for segment in expected["segments"]:
        segment["review_status"] = "unreviewed"
    assert bound == expected
    assert original == before
    bound["metadata"]["unchanged"]["numbers"].append("independent copy")
    bound["segments"][0]["tags"].append("independent copy")
    assert original == before


def test_v1_without_optional_size_acquires_the_actual_size():
    weak = document()
    del weak["media_size"]
    assert bind_document(weak, identity())["media_size"] == 10
    assert "media_size" not in weak


@pytest.mark.parametrize("allow_rename", [False, True])
def test_v1_weak_identity_cannot_authorize_a_rename(allow_rename):
    with pytest.raises(ValueError, match="Version 1 filename must match"):
        bind_document(document(), identity(clip="renamed.bin"), allow_rename=allow_rename)


@pytest.mark.parametrize("version", [1, 2])
def test_duration_mismatch_fails_without_mutation(version):
    doc = document() if version == 1 else strong_document()
    before = copy.deepcopy(doc)
    with pytest.raises(ValueError, match="duration.*1 millisecond"):
        bind_document(doc, identity(duration=12.0011))
    assert doc == before


@pytest.mark.parametrize("version", [1, 2])
def test_within_duration_tolerance_still_rejects_segment_past_actual_end(version):
    doc = document() if version == 1 else strong_document()
    doc["segments"][0]["end"] = 12
    before = copy.deepcopy(doc)
    with pytest.raises(ValueError, match="segment.end.*no timestamps were clipped"):
        bind_document(doc, identity(duration=11.9995))
    assert doc == before


@pytest.mark.parametrize("version", [1, 2])
def test_size_mismatch_fails_even_when_name_and_duration_match(version):
    with pytest.raises(ValueError, match="media_size"):
        bind_document(document() if version == 1 else strong_document(), identity(media_size=11))


def test_v2_checks_full_hash_even_for_same_size_name_and_duration():
    doc = strong_document()
    before = copy.deepcopy(doc)
    with pytest.raises(ValueError, match="full-file SHA256"):
        bind_document(doc, identity(media_sha256=hashlib.sha256(b"9876543210").hexdigest()))
    assert doc == before


def test_v2_revalidation_preserves_reviews_and_all_original_values():
    doc = strong_document()
    result = bind_document(doc, identity(duration=12.0009))
    assert result == doc
    assert result is not doc and result["segments"] is not doc["segments"]


@pytest.mark.parametrize("version", [1, 2])
def test_binding_preserves_valid_deep_metadata_without_deepcopy_recursion(version):
    nested = {"value": "中文 😀", "negative_zero": -0.0, "array": [None, True, 0.5]}
    for _ in range(600):
        nested = {"nested": nested}
    doc = document(metadata=nested) if version == 1 else strong_document(metadata=nested)
    before = json.dumps(doc, ensure_ascii=False, allow_nan=False)
    assert validate_document(doc) == []
    result = bind_document(doc, identity())
    assert json.dumps(doc, ensure_ascii=False, allow_nan=False) == before
    assert json.dumps(result["metadata"], ensure_ascii=False, allow_nan=False) == json.dumps(nested, ensure_ascii=False, allow_nan=False)
    assert result["metadata"] is not nested
    assert result["segments"][0]["review_status"] == ("unreviewed" if version == 1 else "reviewed")


@pytest.mark.parametrize("error", [ValueError("serialization failed"), RecursionError("too deep")])
def test_binding_serialization_failure_is_clear_and_does_not_mutate(monkeypatch, error):
    doc = document()
    before = copy.deepcopy(doc)

    def fail_copy(*args, **kwargs):
        raise error

    monkeypatch.setattr("trace_annotate.temporal.media.json.dumps", fail_copy)
    with pytest.raises(ValueError, match="JSON could not be copied safely"):
        bind_document(doc, identity())
    assert doc == before


def test_v2_rename_requires_explicit_permission_and_preserves_reviews():
    original = strong_document()
    with pytest.raises(ValueError, match="filename differs"):
        bind_document(original, identity(clip="renamed.bin"))
    result = bind_document(original, identity(clip="renamed.bin"), allow_rename=True)
    assert result == dict(original, clip="renamed.bin")
    assert result["segments"][0]["review_status"] == "reviewed"
    assert original["clip"] == "media.bin"


@pytest.mark.parametrize("invalid", [identity(media_size=True), identity(media_sha256="bad"),
                                      identity(duration=True), identity(clip="../bad"), None, {}])
def test_binding_rejects_invalid_supplied_identity(invalid):
    with pytest.raises(ValueError):
        bind_document(document(), invalid)


def test_streaming_hash_matches_known_vector_and_identity_retains_declared_duration(tmp_path):
    media = tmp_path / "media.bin"
    media.write_bytes(b"0123456789")
    assert stream_sha256(media, chunk_size=3) == identity().media_sha256
    assert identify_media(media, 12, chunk_size=3) == identity()
    assert media.read_bytes() == b"0123456789"


def test_empty_regular_file_hash_is_supported_without_claiming_it_is_decodable(tmp_path):
    media = tmp_path / "empty.bin"
    media.write_bytes(b"")
    result = identify_media(media, 1)
    assert result.media_size == 0
    assert result.media_sha256 == hashlib.sha256(b"").hexdigest()


def test_hash_compares_ctime_within_stat_api_not_between_windows_stat_and_fstat(tmp_path, monkeypatch):
    media = tmp_path / "ctime.bin"
    media.write_bytes(b"unchanged bytes")
    original_fstat = os.fstat

    def different_ctime_api(fd):
        value = original_fstat(fd)
        return SimpleNamespace(st_mode=value.st_mode, st_dev=value.st_dev,
                               st_ino=value.st_ino, st_size=value.st_size,
                               st_mtime_ns=value.st_mtime_ns,
                               st_ctime_ns=value.st_ctime_ns + 100)

    monkeypatch.setattr(os, "fstat", different_ctime_api)
    assert stream_sha256(media) == hashlib.sha256(b"unchanged bytes").hexdigest()


@pytest.mark.parametrize("size", [True, 0, -1, 1.0, MAX_CHUNK_SIZE + 1, None])
def test_hash_rejects_unbounded_or_invalid_chunk_sizes_before_reading(tmp_path, size):
    with pytest.raises(ValueError, match="chunk_size"):
        stream_sha256(tmp_path / "absent.bin", chunk_size=size)


def test_streaming_reader_never_requests_more_than_its_bound(tmp_path, monkeypatch):
    path = tmp_path / "bounded.bin"
    payload = bytes(range(256)) * 100
    path.write_bytes(payload)
    original_open = Path.open
    requested = []

    class Reader:
        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            self.raw.__enter__()
            return self

        def __exit__(self, *args):
            return self.raw.__exit__(*args)

        def fileno(self):
            return self.raw.fileno()

        def read(self, count):
            requested.append(count)
            return self.raw.read(count)

    monkeypatch.setattr(Path, "open", lambda source, *args, **kwargs: Reader(original_open(source, *args, **kwargs)))
    assert stream_sha256(path, chunk_size=257) == hashlib.sha256(payload).hexdigest()
    assert len(requested) > 1 and requested[0] == 257
    assert all(1 <= count <= 257 for count in requested)


def test_hash_rejects_nonregular_and_missing_sources(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        stream_sha256(tmp_path)
    with pytest.raises(FileNotFoundError):
        stream_sha256(tmp_path / "absent.bin")


def test_hash_cancellation_stops_reading_and_does_not_change_the_source(tmp_path):
    media = tmp_path / "cancel.bin"
    original = b"some data" * 100
    media.write_bytes(original)
    calls = 0

    def cancel_after_first_chunk():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(InterruptedError, match="cancelled"):
        stream_sha256(media, chunk_size=4, cancelled=cancel_after_first_chunk)
    assert calls == 3
    assert media.read_bytes() == original
    with pytest.raises(InterruptedError):
        stream_sha256(tmp_path / "absent.bin", cancelled=lambda: True)


def test_hash_rejects_observed_file_mutation_during_read(tmp_path):
    media = tmp_path / "changing.bin"
    media.write_bytes(b"0123456789")
    calls = 0

    def mutate_after_first_chunk():
        nonlocal calls
        calls += 1
        if calls == 3:
            with media.open("ab") as stream:
                stream.write(b"changed")
        return False

    with pytest.raises(OSError, match="changed during hashing"):
        identify_media(media, 12, chunk_size=4, cancelled=mutate_after_first_chunk)


def test_hash_stops_at_initial_boundary_while_a_writer_keeps_appending(tmp_path):
    media = tmp_path / "recording.bin"
    media.write_bytes(b"01234567")
    calls = 0

    def keep_recording_after_first_read():
        nonlocal calls
        calls += 1
        if calls >= 3:
            with media.open("ab") as writer:
                writer.write(b"more")
        if calls > 4:
            pytest.fail("Hash reader kept following a growing file instead of failing at its original boundary")
        return False

    with pytest.raises(OSError, match="grew beyond its initial size"):
        stream_sha256(media, chunk_size=4, cancelled=keep_recording_after_first_read)
    assert calls == 4
    assert media.read_bytes() == b"01234567moremore"


def test_hash_detects_growth_of_an_initially_empty_file_before_returning_a_digest(tmp_path):
    media = tmp_path / "new-recording.bin"
    media.write_bytes(b"")
    calls = 0

    def start_recording_after_open():
        nonlocal calls
        calls += 1
        if calls == 2:
            with media.open("ab") as writer:
                writer.write(b"x")
        return False

    with pytest.raises(OSError, match="grew beyond its initial size"):
        stream_sha256(media, cancelled=start_recording_after_open)
    assert calls == 2


def test_hash_rejects_observed_replacement_before_open(tmp_path, monkeypatch):
    media = tmp_path / "changing.bin"
    media.write_bytes(b"0123456789")
    original_open = Path.open

    def alter_then_open(source, *args, **kwargs):
        with original_open(source, "ab") as stream:
            stream.write(b"changed")
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", alter_then_open)
    with pytest.raises(OSError, match="changed before hashing"):
        stream_sha256(media)


def run_cli(*args):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "trace_annotate.cli", *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", env=environment)


def test_cli_hash_bind_validate_and_recheck_workflow(tmp_path):
    media = tmp_path / "media.bin"
    media.write_bytes(b"0123456789")
    source = tmp_path / "input.json"
    original = json.dumps(document(), ensure_ascii=False)
    source.write_text(original, encoding="utf-8")
    hashed = run_cli("video-hash", media)
    assert hashed.returncode == 0, hashed.stderr
    assert hashed.stdout.strip() == hashlib.sha256(media.read_bytes()).hexdigest()
    output = tmp_path / "bound.json"
    bound = run_cli("video-bind", source, "--media", media, "--duration", "12", "--output", output)
    assert bound.returncode == 0, bound.stderr
    assert "caller-declared" in bound.stderr and "no media decoding" in bound.stderr
    migrated = load_document(output)
    assert migrated["schema_version"] == 2
    assert migrated["media_sha256"] == hashed.stdout.strip()
    assert migrated["segments"][0]["review_status"] == "unreviewed"
    assert source.read_text(encoding="utf-8") == original
    assert run_cli("video-validate", output).returncode == 0
    recheck = tmp_path / "rechecked.json"
    assert run_cli("video-bind", output, "--media", media, "--duration", "12", "-o", recheck).returncode == 0
    assert load_document(recheck) == migrated
    assert media.read_bytes() == b"0123456789"


def test_cli_binds_valid_deep_metadata_and_reports_excessive_nesting_without_traceback(tmp_path):
    media = tmp_path / "media.bin"
    media.write_bytes(b"0123456789")
    nested = {}
    for _ in range(600):
        nested = {"nested": nested}
    valid = tmp_path / "deep-valid.json"
    valid.write_text(json.dumps(strong_document(metadata=nested)), encoding="utf-8")
    output = tmp_path / "deep-bound.json"
    result = run_cli("video-bind", valid, "--media", media, "--duration", "12", "-o", output)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert json.dumps(load_document(output)["metadata"]) == json.dumps(nested)
    excessive = tmp_path / "excessive.json"
    base = json.dumps(strong_document(metadata={}))
    excessive.write_text(base.replace('"metadata": {}', '"metadata": ' + '{"nested":' * 1500 + '{}' + '}' * 1500), encoding="utf-8")
    refused = tmp_path / "not-written.json"
    result = run_cli("video-bind", excessive, "--media", media, "--duration", "12", "-o", refused)
    assert result.returncode == 1 and "Error:" in result.stderr
    assert "deep" in result.stderr and "Traceback" not in result.stderr
    assert not refused.exists()


def test_cli_final_serialization_recursion_is_an_error_without_partial_output(tmp_path, monkeypatch, capsys):
    source = tmp_path / "valid.json"
    source.write_text(json.dumps(strong_document()), encoding="utf-8")
    output = tmp_path / "not-written.json"

    def exceed_nesting(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("trace_annotate.temporal.commands.json.dumps", exceed_nesting)
    args = argparse.Namespace(command="video-export", file=source, format="json", track=None, output=output)
    with pytest.raises(SystemExit) as result:
        _run(args)
    assert result.value.code == 1
    captured = capsys.readouterr()
    assert "nesting is too deep" in captured.err and "Traceback" not in captured.err
    assert not output.exists()


def test_cli_does_not_overwrite_and_does_not_publish_a_mismatch(tmp_path):
    media = tmp_path / "media.bin"
    media.write_bytes(b"9876543210")
    source = tmp_path / "strong.json"
    source.write_text(json.dumps(strong_document()), encoding="utf-8")
    output = tmp_path / "absent.json"
    mismatch = run_cli("video-bind", source, "--media", media, "--duration", "12", "-o", output)
    assert mismatch.returncode == 1 and "media_sha256" in mismatch.stderr
    assert not output.exists()
    media.write_bytes(b"0123456789")
    output.write_bytes(b"keep existing data")
    existing = run_cli("video-bind", source, "--media", media, "--duration", "12", "-o", output)
    assert existing.returncode == 1 and "already exists" in existing.stderr
    assert output.read_bytes() == b"keep existing data"
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_explicit_v2_rename_and_v1_rejection(tmp_path):
    media = tmp_path / "renamed.bin"
    media.write_bytes(b"0123456789")
    source = tmp_path / "strong.json"
    source.write_text(json.dumps(strong_document()), encoding="utf-8")
    output = tmp_path / "renamed.json"
    refused = run_cli("video-bind", source, "--media", media, "--duration", "12", "-o", output)
    assert refused.returncode == 1 and not output.exists()
    allowed = run_cli("video-bind", source, "--media", media, "--duration", "12", "--allow-rename", "-o", output)
    assert allowed.returncode == 0, allowed.stderr
    assert load_document(output) == strong_document(clip="renamed.bin")
    source.write_text(json.dumps(document()), encoding="utf-8")
    weak_output = tmp_path / "weak-rename.json"
    weak = run_cli("video-bind", source, "--media", media, "--duration", "12", "--allow-rename", "-o", weak_output)
    assert weak.returncode == 1 and "weak identity" in weak.stderr
    assert not weak_output.exists()


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "-1", "9007199254740992"])
def test_cli_invalid_declared_duration_does_not_create_output(tmp_path, duration):
    media = tmp_path / "media.bin"
    media.write_bytes(b"0123456789")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(document()), encoding="utf-8")
    output = tmp_path / "absent.json"
    result = run_cli("video-bind", source, "--media", media, "--duration", duration, "-o", output)
    assert result.returncode != 0 and not output.exists()
