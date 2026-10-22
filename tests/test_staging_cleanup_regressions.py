"""Filesystem regressions for staging cleanup and in-use protection."""
import os
import subprocess
from pathlib import Path

import pytest

from trace_annotate.config import StagingConfig
from trace_annotate.core.staging import (
    ColdStorageStagingError,
    StagingManager,
    StagingSecurityError,
)


@pytest.fixture
def staging_manager(tmp_path):
    source, staging = tmp_path / "source", tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    return StagingManager(str(source), str(staging), StagingConfig(free_disk_guard_bytes=0))


def file_link(link, target):
    try:
        os.symlink(target, link)
    except OSError as exc:
        if exc.errno in (1, 13, 38, 95) or getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"Filesystem symlinks unavailable: {exc}")
        raise


def directory_link(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
            if exc.errno in (1, 13, 38, 95):
                pytest.skip(f"Directory links unavailable: {exc}")
            raise
        # Test-only junctions work on Windows without symlink privileges.
        link_text, target_text = str(link).replace("'", "''"), str(target).replace("'", "''")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             f"New-Item -ItemType Junction -Path '{link_text}' -Target '{target_text}' | Out-Null"],
            capture_output=True, text=True,
        )
        if result.returncode:
            pytest.fail(f"Could not create test junction: {result.stderr}")


def remove_directory_link(link):
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.lexists(link):
        os.rmdir(link)


def test_cleanup_unlinks_alias_without_deleting_locked_target(staging_manager):
    manager = staging_manager
    target = os.path.join(manager.staging_root, "active.txt")
    alias = os.path.join(manager.staging_root, "alias.txt")
    with open(target, "w", encoding="utf-8") as stream:
        stream.write("in use")
    file_link(alias, target)
    manager.lock_in_use("active.txt")

    cleaned = manager.cleanup_completed()

    assert os.path.isfile(target), "cleanup must preserve the locked symlink target"
    with open(target, encoding="utf-8") as stream:
        assert stream.read() == "in use"
    assert not os.path.lexists(alias)
    assert cleaned == ["alias.txt"]


def test_cleanup_honors_normalized_lock_path(staging_manager):
    manager = staging_manager
    target = os.path.join(manager.staging_root, "active.txt")
    with open(target, "w", encoding="utf-8") as stream:
        stream.write("in use")
    manager.lock_in_use("./active.txt")

    assert manager.cleanup_completed() == []
    assert os.path.isfile(target)
    manager.unlock_in_use("active.txt")
    assert manager.cleanup_completed() == ["active.txt"]


def test_cleanup_does_not_traverse_directory_alias_to_locked_file(staging_manager):
    manager = staging_manager
    actual = os.path.join(manager.staging_root, "actual")
    alias = os.path.join(manager.staging_root, "alias")
    os.mkdir(actual)
    target = os.path.join(actual, "active.txt")
    with open(target, "w", encoding="utf-8") as stream:
        stream.write("in use")
    directory_link(alias, actual)
    manager.lock_in_use("actual/active.txt")
    try:
        cleaned = manager.cleanup_completed()
        assert os.path.isfile(target), "cleanup must not follow a directory alias into locked data"
        assert cleaned == []
    finally:
        remove_directory_link(alias)


def test_lock_through_directory_alias_protects_target(staging_manager):
    manager = staging_manager
    actual = Path(manager.staging_root) / "actual"
    alias = Path(manager.staging_root) / "alias"
    actual.mkdir()
    target = actual / "active.txt"
    target.write_text("keep", encoding="utf-8")
    directory_link(alias, actual)
    try:
        manager.lock_in_use("alias/active.txt")
        assert manager.cleanup_completed() == []
        assert target.read_text(encoding="utf-8") == "keep"
        manager.unlock_in_use("./alias/active.txt")
        assert manager.cleanup_completed() == ["actual/active.txt"]
        assert not target.exists()
    finally:
        remove_directory_link(alias)


def test_cleanup_skips_outside_directory_link(staging_manager, tmp_path):
    manager = staging_manager
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("outside", encoding="utf-8")
    link = Path(manager.staging_root) / "outside"
    directory_link(link, outside)
    try:
        assert manager.cleanup_completed() == []
        assert sentinel.read_text(encoding="utf-8") == "outside"
        assert os.path.lexists(link)
    finally:
        remove_directory_link(link)


def test_cleanup_removes_broken_in_tree_file_link(staging_manager):
    root = Path(staging_manager.staging_root)
    alias = root / "broken.txt"
    file_link(alias, root / "missing.txt")
    assert staging_manager.cleanup_completed() == ["broken.txt"]
    assert not os.path.lexists(alias)


def test_cleanup_rejects_outside_file_link(staging_manager, tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("outside", encoding="utf-8")
    alias = Path(staging_manager.staging_root) / "alias.txt"
    file_link(alias, sentinel)
    with pytest.raises(StagingSecurityError):
        staging_manager.cleanup_completed()
    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert os.path.lexists(alias)


def test_lock_via_file_link_preserves_both_entries(staging_manager):
    root = Path(staging_manager.staging_root)
    target, alias = root / "active.txt", root / "alias.txt"
    target.write_text("keep", encoding="utf-8")
    file_link(alias, target)
    staging_manager.lock_in_use("alias.txt")
    assert staging_manager.cleanup_completed() == []
    assert target.read_text(encoding="utf-8") == "keep"
    assert os.path.lexists(alias)


def test_cleanup_dry_run_preserves_entries(staging_manager):
    target = Path(staging_manager.staging_root) / "done.txt"
    target.write_text("ready", encoding="utf-8")
    assert staging_manager.cleanup_completed(dry_run=True) == ["done.txt"]
    assert target.read_text(encoding="utf-8") == "ready"
    assert staging_manager.cleanup_completed() == ["done.txt"]
    assert not target.exists()


def test_cleanup_reports_failed_unlink(staging_manager, monkeypatch):
    target = Path(staging_manager.staging_root) / "done.txt"
    target.write_text("ready", encoding="utf-8")

    def denied(path):
        raise PermissionError("test: file is busy")

    monkeypatch.setattr("trace_annotate.core.staging.os.remove", denied)
    with pytest.raises(ColdStorageStagingError, match="Could not remove staged file"):
        staging_manager.cleanup_completed()
    assert target.read_text(encoding="utf-8") == "ready"


def test_cleanup_does_not_report_already_missing_entry(staging_manager, monkeypatch):
    target = Path(staging_manager.staging_root) / "done.txt"
    target.write_text("ready", encoding="utf-8")
    real_remove = os.remove

    def removed_elsewhere(path):
        real_remove(path)
        raise FileNotFoundError("test: another cleaner removed the file")

    monkeypatch.setattr("trace_annotate.core.staging.os.remove", removed_elsewhere)
    assert staging_manager.cleanup_completed() == []
    assert not target.exists()


@pytest.mark.parametrize("path", ["../outside.txt", "/absolute.txt", "C:\\outside.txt", "."])
def test_lock_rejects_invalid_paths(staging_manager, path):
    with pytest.raises(StagingSecurityError):
        staging_manager.lock_in_use(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows path case semantics")
def test_cleanup_honors_case_insensitive_lock(staging_manager):
    target = Path(staging_manager.staging_root) / "Active.txt"
    target.write_text("keep", encoding="utf-8")
    staging_manager.lock_in_use("ACTIVE.TXT")
    assert staging_manager.cleanup_completed() == []
    staging_manager.unlock_in_use("active.txt")
    assert staging_manager.cleanup_completed() == ["Active.txt"]


def test_batch_cleanup_honors_normalized_lock(staging_manager):
    manager = staging_manager
    source = Path(manager.source_root) / "active.txt"
    source.write_text("source", encoding="utf-8")
    target = Path(manager.staging_root) / "active.txt"
    for batch in manager.stage_batch(["active.txt"]):
        assert Path(batch["active.txt"]).read_text(encoding="utf-8") == "source"
        manager.lock_in_use("./active.txt")
    assert target.read_text(encoding="utf-8") == "source"
    manager.unlock_in_use("active.txt")
    assert manager.cleanup_completed() == ["active.txt"]


def test_batch_cleanup_preserves_replaced_link_to_locked_file(staging_manager):
    manager = staging_manager
    source = Path(manager.source_root) / "asset.txt"
    source.write_text("source", encoding="utf-8")
    sentinel = Path(manager.staging_root) / "locked.txt"
    sentinel.write_text("keep", encoding="utf-8")
    manager.lock_in_use("locked.txt")
    # Establish capability before starting the generator, so a skip cannot leave
    # a generator awaiting cleanup of a partly modified fixture.
    probe = Path(manager.staging_root) / "probe.txt"
    file_link(probe, sentinel)
    probe.unlink()
    batches = manager.stage_batch(["asset.txt"])
    next(batches)
    target = Path(manager.staging_root) / "asset.txt"
    target.unlink()
    file_link(target, sentinel)
    # Lock resolution recognizes the target, so closing keeps the link and the
    # active file instead of deleting either through the changed path.
    with pytest.raises(StopIteration):
        next(batches)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert os.path.lexists(target)


def test_batch_rejects_existing_file_link_before_copy(staging_manager):
    manager = staging_manager
    source = Path(manager.source_root) / "asset.txt"
    source.write_text("new", encoding="utf-8")
    sentinel = Path(manager.staging_root) / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    target = Path(manager.staging_root) / "asset.txt"
    file_link(target, sentinel)
    with pytest.raises(StagingSecurityError, match="redirected"):
        list(manager.stage_batch(["asset.txt"]))
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert os.path.lexists(target)


def test_batch_rejects_in_tree_directory_alias_before_copy(staging_manager):
    manager = staging_manager
    source_dir = Path(manager.source_root) / "alias"
    source_dir.mkdir()
    (source_dir / "asset.txt").write_text("new", encoding="utf-8")
    actual = Path(manager.staging_root) / "actual"
    actual.mkdir()
    sentinel = actual / "asset.txt"
    sentinel.write_text("keep", encoding="utf-8")
    alias = Path(manager.staging_root) / "alias"
    directory_link(alias, actual)
    try:
        with pytest.raises(StagingSecurityError, match="redirected"):
            list(manager.stage_batch(["alias/asset.txt"]))
        assert sentinel.read_text(encoding="utf-8") == "keep"
    finally:
        remove_directory_link(alias)
