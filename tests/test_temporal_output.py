"""Fault injection for complete, non-overwriting temporal CLI outputs."""
import errno
import json
from pathlib import Path
import sys

import pytest

from trace_annotate.cli import main
from trace_annotate.temporal import commands


def _install_writer(monkeypatch, write, flush=None):
    """Keep a real same-directory file while injecting its stream behavior."""
    create = commands.tempfile.NamedTemporaryFile

    class Writer:
        def __init__(self, *args, **kwargs):
            self.stream = create(*args, **kwargs)
            self.name = self.stream.name

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, data):
            return write(self.stream, data)

        def flush(self):
            return flush(self.stream) if flush is not None else self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(commands.tempfile, "NamedTemporaryFile", Writer)


@pytest.mark.parametrize("content", ["", "中文标签\n第二行 🦊\n", "\ufeff列名,说明\n片段,中文\n"])
def test_success_writes_exact_utf8_and_removes_temporary(tmp_path, content):
    parent = tmp_path / "中文目录"
    parent.mkdir()
    output = parent / "自定义输出.json"
    commands._write_new(str(output), content)
    assert output.read_bytes() == content.encode("utf-8")
    assert list(parent.iterdir()) == [output]


def test_short_writes_are_completed_before_publication(tmp_path, monkeypatch):
    content = "多字节文本\n" * 10
    _install_writer(monkeypatch, lambda stream, data: stream.write(data[:3]))
    output = tmp_path / "annotations.json"
    commands._write_new(str(output), content)
    assert output.read_bytes() == content.encode("utf-8")
    assert list(tmp_path.iterdir()) == [output]


def test_publication_happens_only_after_flush_and_sync(tmp_path, monkeypatch):
    stages = []
    fsync, link = commands.os.fsync, commands.os.link

    def flush(stream):
        stream.flush()
        stages.append("flush")

    def sync(descriptor):
        assert stages == ["flush"]
        fsync(descriptor)
        stages.append("fsync")

    def publish(source, destination):
        assert stages == ["flush", "fsync"]
        assert Path(source).read_bytes() == b"complete text"
        link(source, destination)
        stages.append("publish")

    _install_writer(monkeypatch, lambda stream, data: stream.write(data), flush=flush)
    monkeypatch.setattr(commands.os, "fsync", sync)
    monkeypatch.setattr(commands.os, "link", publish)
    commands._write_new(str(tmp_path / "annotations.json"), "complete text")
    assert stages == ["flush", "fsync", "publish"]


@pytest.mark.parametrize("result", [None, 0, -1, 100000])
def test_invalid_write_progress_has_no_output(tmp_path, monkeypatch, result):
    _install_writer(monkeypatch, lambda stream, data: result)
    output = tmp_path / "annotations.json"
    with pytest.raises(OSError, match="valid progress"):
        commands._write_new(str(output), "payload")
    assert list(tmp_path.iterdir()) == []


def test_exception_after_partial_write_has_no_output(tmp_path, monkeypatch):
    failure = OSError(errno.ENOSPC, "simulated full disk")
    calls = 0

    def write(stream, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return stream.write(data[:3])
        raise failure

    _install_writer(monkeypatch, write)
    with pytest.raises(OSError) as raised:
        commands._write_new(str(tmp_path / "annotations.json"), "not a complete document")
    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []


def test_keyboard_interrupt_after_partial_write_has_no_output(tmp_path, monkeypatch):
    def write(stream, data):
        stream.write(data[:3])
        raise KeyboardInterrupt()

    _install_writer(monkeypatch, write)
    with pytest.raises(KeyboardInterrupt):
        commands._write_new(str(tmp_path / "annotations.json"), "not a complete document")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", ["flush", "fsync"])
def test_sync_failure_has_no_output(tmp_path, monkeypatch, stage):
    failure = OSError(errno.EIO, "simulated sync failure")

    def fail(*args):
        raise failure

    if stage == "flush":
        _install_writer(monkeypatch, lambda stream, data: stream.write(data), flush=fail)
    else:
        monkeypatch.setattr(commands.os, "fsync", fail)
    with pytest.raises(OSError) as raised:
        commands._write_new(str(tmp_path / "annotations.json"), "complete text")
    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling-symlink"])
def test_existing_entry_is_never_replaced(tmp_path, kind):
    output = tmp_path / "annotations.json"
    original = tmp_path / "original"
    original.write_bytes(b"original bytes")
    if kind == "file":
        output.write_bytes(b"existing output")
    elif kind == "directory":
        output.mkdir()
    else:
        try:
            output.symlink_to(original if kind == "symlink" else tmp_path / "missing")
        except OSError as exc:
            pytest.skip(f"Symbolic link creation unavailable: {exc}")
    before = set(tmp_path.iterdir())
    with pytest.raises(FileExistsError):
        commands._write_new(str(output), "new data")
    assert set(tmp_path.iterdir()) == before
    assert original.read_bytes() == b"original bytes"
    if kind == "file":
        assert output.read_bytes() == b"existing output"
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()


def test_destination_created_during_publication_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "annotations.json"
    link = commands.os.link

    def publish(source, destination):
        assert Path(source).read_bytes() == "完整内容".encode("utf-8")
        Path(destination).write_bytes(b"concurrent writer")
        link(source, destination)

    monkeypatch.setattr(commands.os, "link", publish)
    with pytest.raises(FileExistsError):
        commands._write_new(str(output), "完整内容")
    assert output.read_bytes() == b"concurrent writer"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("failure", [OSError(errno.ENOTSUP, "no hard links"), NotImplementedError("no hard links")])
def test_unsupported_hard_link_does_not_fall_back(tmp_path, monkeypatch, failure):
    def fail(*args):
        raise failure

    monkeypatch.setattr(commands.os, "link", fail)
    with pytest.raises(OSError):
        commands._write_new(str(tmp_path / "annotations.json"), "complete text")
    assert list(tmp_path.iterdir()) == []


def test_cleanup_failure_does_not_hide_publication_failure(tmp_path, monkeypatch):
    failure = OSError(errno.ENOTSUP, "primary publication failure")
    unlink = commands.os.unlink
    temporary = []

    def fail_link(source, destination):
        temporary.append(Path(source))
        raise failure

    def fail_unlink(path):
        raise PermissionError(errno.EACCES, "secondary cleanup failure")

    monkeypatch.setattr(commands.os, "link", fail_link)
    monkeypatch.setattr(commands.os, "unlink", fail_unlink)
    with pytest.raises(OSError) as raised:
        commands._write_new(str(tmp_path / "annotations.json"), "complete text")
    assert raised.value is failure
    assert not (tmp_path / "annotations.json").exists()
    assert temporary[0].read_bytes() == b"complete text"
    unlink(temporary[0])


def test_cleanup_failure_after_publication_reports_existing_complete_output(tmp_path, monkeypatch):
    unlink = commands.os.unlink
    temporary = []

    def fail(path):
        temporary.append(Path(path))
        raise PermissionError(errno.EACCES, "cleanup denied")

    monkeypatch.setattr(commands.os, "unlink", fail)
    output = tmp_path / "annotations.json"
    with pytest.raises(OSError, match="Output was written, but temporary cleanup failed"):
        commands._write_new(str(output), "complete text")
    assert output.read_bytes() == b"complete text"
    assert temporary[0].read_bytes() == b"complete text"
    unlink(temporary[0])


def test_cleanup_does_not_remove_a_replaced_temporary(tmp_path, monkeypatch):
    replacement = tmp_path / "other-file"
    replacement.write_bytes(b"belongs to another writer")
    temporary = []
    failure = OSError(errno.EIO, "publication failed")

    def fail(source, destination):
        temporary.append(Path(source))
        commands.os.replace(replacement, source)
        raise failure

    monkeypatch.setattr(commands.os, "link", fail)
    with pytest.raises(OSError) as raised:
        commands._write_new(str(tmp_path / "annotations.json"), "complete text")
    assert raised.value is failure
    assert not (tmp_path / "annotations.json").exists()
    assert temporary[0].read_bytes() == b"belongs to another writer"


def test_invalid_unicode_does_not_create_any_file(tmp_path):
    with pytest.raises(UnicodeEncodeError):
        commands._write_new(str(tmp_path / "annotations.json"), "bad\ud800text")
    assert list(tmp_path.iterdir()) == []


def test_cli_preserves_source_and_reports_atomic_publication_error(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"kind": "temporal", "schema_version": 1,
                                 "clip": "sample.mp4", "duration": 1, "segments": []}), encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / "export.json"

    def fail(*args):
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    monkeypatch.setattr(commands.os, "link", fail)
    monkeypatch.setattr(sys, "argv", ["trace-annotate", "video-export", str(source),
                                      "--format", "json", "-o", str(output)])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    assert "Error:" in capsys.readouterr().err
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [source]
