"""P01: a backup carries failed uploads, and only a safe closure may restore.

The invariants under test:

* the retryable-source closure is compared as an order-independent set, so a
  backup restores identically with its entries in the original, reversed or
  shuffled order, and a legacy manifest without the key still restores the
  failed intakes it does contain;
* duplicates, missing or extra entries and any per-entry identity change are
  still refused, never reduced to a count comparison;
* a real transient preparation failure survives backup -> restore into a new
  directory, can be retried to success there, then edited, reopened and
  exported; a genuinely corrupt source stays failed instead;
* every unsafe or incomplete declaration (traversal, absolute, drive, UNC,
  backslash, a different legal path, duplicate/missing/extra entry, missing
  source, wrong hash or size, a linked path, a corrupt database, an unknown
  version, an existing target, an interrupted copy) is refused without writing
  outside this run's own sandbox and without replacing anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trace_annotate.workspace import backup as backup_module
from trace_annotate.workspace import media as media_module
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.backup import (
    BACKUP_MANIFEST, RESTORE_MANIFEST, SNAPSHOT_DB, backup_workspace, restore_workspace)
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.reporting import export_task
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools


def _ffmpeg() -> str:
    return require_media_tools()[0]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("p01-clips") / "clip.mp4"
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=10",
                    "-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, timeout=300)
    return out


@pytest.fixture()
def broken(tmp_path) -> Path:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not media at all" * 32)
    return path


@pytest.fixture()
def store(tmp_path) -> WorkspaceStore:
    value = WorkspaceStore(tmp_path / "proj")
    value.create()
    yield value
    try:
        value.close()
    except Exception:
        pass


def _open(root: Path) -> WorkspaceStore:
    value = WorkspaceStore(root)
    value.open()
    return value


def _failed_async(store: WorkspaceStore, tmp_path: Path, name: str, payload: bytes) -> str:
    """Create one really failed async intake with its own staged source."""
    source = tmp_path / f"{name}.bin"
    source.write_bytes(payload)
    upload = store.tmp_dir / f"{name}.tmp"
    shutil.copy2(source, upload)
    intakes = AsyncIntakes(store)
    accepted = intakes.enqueue(upload, display_name=name)
    intakes.run_pending()
    row = intakes.get(accepted["id"])
    assert row["state"] == "failed", row
    assert row["staged_present"] is True, row
    return accepted["id"]


def _failed_sync(store: WorkspaceStore, broken: Path, name: str) -> str:
    """Create one failed legacy (synchronous) intake: no workspace-owned source."""
    with pytest.raises(WorkspaceError):
        store.import_source(broken, display_name=name)
    rows = [row for row in store.list_intakes() if row["display_name"] == name]
    assert len(rows) == 1 and rows[0]["state"] == "failed", rows
    assert rows[0]["staged_path"] is None
    return rows[0]["id"]


def _manifest(backup: Path) -> dict:
    return json.loads((backup / BACKUP_MANIFEST).read_text(encoding="utf-8"))


def _rewrite(backup: Path, data: dict) -> None:
    (backup / BACKUP_MANIFEST).write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def _copy_backup(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _sentinel(tmp_path: Path, name: str = "outside.txt") -> Path:
    path = tmp_path / name
    path.write_text("outside the sandbox", encoding="utf-8")
    return path


def _closure(intake_ids, store):
    rows = {row["id"]: row for row in store.list_intakes()}
    return {ident: (rows[ident]["source_sha256"], rows[ident]["source_size"]) for ident in intake_ids}


# ---------------------------------------------------------------------------
# BACKUP_ORDER_INDEPENDENT
# ---------------------------------------------------------------------------
def test_a_backup_restores_identically_in_original_reversed_and_shuffled_order(
        store, tmp_path, broken):
    first = _failed_async(store, tmp_path, "order-a", b"alpha" * 100)
    second = _failed_async(store, tmp_path, "order-b", b"beta" * 200)
    third = _failed_async(store, tmp_path, "order-c", b"gamma" * 300)
    ids = [first, second, third]
    expected = _closure(ids, store)
    staged = {ident: (store.root / f"tmp/intakes/{ident}/source").read_bytes() for ident in ids}

    manifest = backup_workspace(store, tmp_path / "backup")
    assert {entry["intake_id"] for entry in manifest["retryable_sources"]} == set(ids)
    store.close()

    import random
    shuffled = list(ids)
    random.Random(20261030).shuffle(shuffled)
    assert shuffled != ids, "the shuffle must actually change the order"
    orderings = {"original": ids, "reversed": list(reversed(ids)), "shuffled": shuffled}

    for label, order in orderings.items():
        backup = _copy_backup(tmp_path / "backup", tmp_path / f"backup-{label}")
        data = _manifest(backup)
        by_id = {entry["intake_id"]: entry for entry in data["retryable_sources"]}
        data["retryable_sources"] = [by_id[ident] for ident in order]
        _rewrite(backup, data)

        target = tmp_path / f"restored-{label}"
        receipt = restore_workspace(backup, target)
        assert receipt["state"] == "complete", receipt
        assert (target / RESTORE_MANIFEST).is_file()

        reopened = _open(target)
        try:
            assert _closure(ids, reopened) == expected, label
            intakes = AsyncIntakes(reopened)
            for ident in ids:
                row = intakes.get(ident)
                assert row["state"] == "failed" and row["staged_present"] is True, (label, row)
                assert (target / f"tmp/intakes/{ident}/source").read_bytes() == staged[ident], label
        finally:
            reopened.close()


def test_a_legacy_backup_without_a_retryable_closure_still_restores(store, tmp_path, broken):
    """A pre-closure manifest keeps its legacy failed intakes and claims nothing."""
    first = _failed_sync(store, broken, "legacy-a.mp4")
    second = _failed_sync(store, broken, "legacy-b.mp4")
    manifest = backup_workspace(store, tmp_path / "backup")
    assert manifest["retryable_sources"] == [], manifest["retryable_sources"]
    store.close()

    legacy = _copy_backup(tmp_path / "backup", tmp_path / "legacy-backup")
    data = _manifest(legacy)
    del data["retryable_sources"]  # the older manifest shape
    _rewrite(legacy, data)

    target = tmp_path / "restored-legacy"
    receipt = restore_workspace(legacy, target)
    assert receipt["state"] == "complete"
    reopened = _open(target)
    try:
        intakes = AsyncIntakes(reopened)
        for ident in (first, second):
            row = intakes.get(ident)
            assert row["state"] == "failed"
            assert row["staged_present"] is False
            with pytest.raises(WorkspaceError) as exc:
                intakes.retry_failed(ident)
            assert exc.value.code == "SOURCE_MISSING"
        with reopened.transaction():
            assert reopened.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("damage", [
    "duplicate", "missing", "extra", "hash", "size", "other_path"])
def test_the_closure_still_refuses_duplicates_missing_extra_and_identity_changes(
        store, tmp_path, damage):
    ids = [_failed_async(store, tmp_path, f"closure-{index}", bytes([index]) * 512)
           for index in (1, 2)]
    backup_workspace(store, tmp_path / "backup")
    store.close()
    sentinel = _sentinel(tmp_path)

    forged = _copy_backup(tmp_path / "backup", tmp_path / f"forged-{damage}")
    data = _manifest(forged)
    entries = data["retryable_sources"]
    if damage == "duplicate":
        entries.append(dict(entries[0]))
    elif damage == "missing":
        entries.pop()
    elif damage == "extra":
        entries.append({"intake_id": "int-extra000000",
                        "relative_path": "tmp/intakes/int-extra000000/source",
                        "sha256": "0" * 64, "size_bytes": "1"})
    elif damage == "hash":
        entries[0]["sha256"] = "1" * 64
    elif damage == "size":
        entries[0]["size_bytes"] = str(int(entries[0]["size_bytes"]) + 1)
    else:
        entries[0]["relative_path"] = entries[0]["relative_path"].replace("/source", "/other")
    _rewrite(forged, data)
    before = hashlib.sha256((forged / BACKUP_MANIFEST).read_bytes()).hexdigest()

    target = tmp_path / f"restored-{damage}"
    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(forged, target)
    assert exc.value.code == "BACKUP_INTEGRITY_FAILED", (damage, exc.value.code)
    assert not target.exists(), "a refused manifest must not create a target"
    assert sentinel.read_text(encoding="utf-8") == "outside the sandbox"
    assert hashlib.sha256((forged / BACKUP_MANIFEST).read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# BACKUP_RETRY_AND_EDIT
# ---------------------------------------------------------------------------
def test_a_transient_failure_can_be_restored_retried_edited_and_exported(
        store, tmp_path, clip):
    available = {"value": False}
    original = media_module.prepare_workspace_media

    def flaky(*args, **kwargs):
        if not available["value"]:
            raise WorkspaceError("PREPARE_UNAVAILABLE", "decoder is temporarily unavailable",
                                 retryable=True)
        return original(*args, **kwargs)

    media_module.prepare_workspace_media = flaky
    try:
        upload = store.tmp_dir / "transient.mp4"
        shutil.copy2(clip, upload)
        intakes = AsyncIntakes(store)
        accepted = intakes.enqueue(upload, display_name="transient.mp4")
        intakes.run_pending()
        failed = intakes.get(accepted["id"])
        assert failed["state"] == "failed" and failed["error_code"] == "PREPARE_UNAVAILABLE", failed
        assert failed["staged_present"] is True
    finally:
        media_module.prepare_workspace_media = original

    backup_workspace(store, tmp_path / "backup")
    store.close()

    restored_root = tmp_path / "restored-elsewhere"
    receipt = restore_workspace(tmp_path / "backup", restored_root)
    assert receipt["state"] == "complete"
    reopened = _open(restored_root)
    try:
        intakes = AsyncIntakes(reopened)
        row = intakes.get(accepted["id"])
        assert row["state"] == "failed" and row["error_code"] == "PREPARE_UNAVAILABLE", row

        # The temporary error is gone in the restored workspace: retry for real.
        retried = intakes.retry_failed(accepted["id"])
        assert retried["state"] == "queued", retried
        intakes.run_pending()
        done = intakes.get(accepted["id"])
        assert done["state"] == "succeeded", done
        assert done["task_id"]
        with reopened.transaction():
            assert reopened.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        task = reopened.get_task(done["task_id"])
        assert task["revision"] == 0 and task["annotations"] == []

        # A human edit is saved as a real revisioned command.
        response = reopened.apply_task_command(
            command_id="cmd-edit-after-restore", task_id=task["id"], expected_revision=0,
            expected_sha256=task["content_sha256"], operation="add_annotation",
            payload={"annotation": {
                "id": "ann-restored", "type": "segment", "label": "working", "text": "restored",
                "tags": [], "track": "main", "attributes": {},
                "provenance": {"origin": "manual", "actor_id": "user", "job_id": None},
                "evidence_ids": [],
                "start": {"seconds": S.rational(0.0)}, "end": {"seconds": S.rational(1.0)},
            }}, request_body=b"{}")
        assert response["response"]["revision"] == 1
    finally:
        reopened.close()

    # Close and reopen: the edit is durable and exportable.
    again = _open(restored_root)
    try:
        task = again.get_task(done["task_id"])
        assert task["revision"] == 1
        assert [a["label"] for a in task["annotations"]] == ["working"]
        exported = export_task(again, task["id"], "json")
        payload = json.loads(exported["data"])
        assert payload["task"]["annotations"][0]["id"] == "ann-restored"
        assert payload["binding"]["revision"] == 1
        temporal = json.loads(export_task(again, task["id"], "temporal")["data"])
        assert [segment["id"] for segment in temporal.get("segments", [])] == ["ann-restored"], temporal
    finally:
        again.close()


def test_corrupt_media_stays_failed_after_restore_and_retry(store, tmp_path, broken):
    intake_id = _failed_async(store, tmp_path, "corrupt", broken.read_bytes())
    backup_workspace(store, tmp_path / "backup")
    store.close()

    target = tmp_path / "restored-corrupt"
    assert restore_workspace(tmp_path / "backup", target)["state"] == "complete"
    reopened = _open(target)
    try:
        intakes = AsyncIntakes(reopened)
        before = intakes.get(intake_id)
        assert before["state"] == "failed"
        intakes.retry_failed(intake_id)
        intakes.run_pending()
        after = intakes.get(intake_id)
        assert after["state"] == "failed", after
        assert after["task_id"] is None
        assert after["attempt"] == before["attempt"] + 1
        assert after["staged_present"] is True
        with reopened.transaction():
            assert reopened.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# BACKUP_UNSAFE_OR_INCOMPLETE_REJECT
# ---------------------------------------------------------------------------
def _two_source_backup(store, tmp_path) -> Path:
    for index in (1, 2):
        _failed_async(store, tmp_path, f"reject-{index}", bytes([index]) * 256)
    backup_workspace(store, tmp_path / "backup")
    store.close()
    return tmp_path / "backup"


@pytest.mark.parametrize("relative", [
    "../escape/source",
    "/etc/passwd",
    "C:/Windows/win.ini",
    "//server/share/source",
    "tmp\\intakes\\{id}\\source",
    "tmp/intakes/{id}/source.bak",
])
def test_an_unsafe_or_noncanonical_source_path_is_refused(store, tmp_path, relative):
    backup = _two_source_backup(store, tmp_path)
    sentinel = _sentinel(tmp_path)
    ident = _manifest(backup)["retryable_sources"][0]["intake_id"]
    forged = _copy_backup(backup, tmp_path / "forged-path")
    data = _manifest(forged)
    data["retryable_sources"][0]["relative_path"] = relative.replace("{id}", ident)
    _rewrite(forged, data)

    target = tmp_path / "restored-path"
    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(forged, target)
    assert exc.value.code in {"BACKUP_INTEGRITY_FAILED", "UNSAFE_PATH"}, exc.value.code
    assert not target.exists()
    assert sentinel.read_text(encoding="utf-8") == "outside the sandbox"
    assert not (tmp_path / "escape").exists()


def test_a_missing_source_file_is_an_incomplete_backup(store, tmp_path):
    backup = _two_source_backup(store, tmp_path)
    sentinel = _sentinel(tmp_path)
    forged = _copy_backup(backup, tmp_path / "forged-missing")
    entry = _manifest(forged)["retryable_sources"][0]
    (forged / entry["relative_path"]).unlink()

    target = tmp_path / "restored-missing"
    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(forged, target)
    assert exc.value.code == "BACKUP_INCOMPLETE", exc.value.code
    assert not target.exists()
    assert sentinel.read_text(encoding="utf-8") == "outside the sandbox"


def _make_junction(link: Path, target: Path) -> bool:
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True)
        if result.returncode == 0:
            return True
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


def test_a_linked_source_directory_is_refused(store, tmp_path):
    backup = _two_source_backup(store, tmp_path)
    sentinel_dir = tmp_path / "outside-sources"
    sentinel_dir.mkdir()
    entry = _manifest(backup)["retryable_sources"][0]
    staged = backup / "tmp" / "intakes"
    outside = sentinel_dir / "intakes"
    shutil.move(str(staged), str(outside))
    if not _make_junction(staged, outside):
        pytest.skip("this account cannot create a junction or directory symlink")
    sentinel = _sentinel(tmp_path)

    target = tmp_path / "restored-link"
    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(backup, target)
    assert exc.value.code == "UNSAFE_PATH", exc.value.code
    assert not target.exists()
    assert (outside / entry["intake_id"] / "source").is_file(), "the linked source was removed"
    assert sentinel.read_text(encoding="utf-8") == "outside the sandbox"


def test_a_corrupt_snapshot_database_is_refused(store, tmp_path):
    backup = _two_source_backup(store, tmp_path)
    sentinel = _sentinel(tmp_path)
    forged = _copy_backup(backup, tmp_path / "forged-db")
    database = forged / SNAPSHOT_DB
    raw = bytearray(database.read_bytes())
    raw[len(raw) // 2: len(raw) // 2 + 64] = b"\xff" * 64
    database.write_bytes(bytes(raw))

    target = tmp_path / "restored-db"
    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(forged, target)
    assert exc.value.code in {"BACKUP_INTEGRITY_FAILED", "BACKUP_INCOMPLETE"}, exc.value.code
    assert not target.exists()
    assert sentinel.read_text(encoding="utf-8") == "outside the sandbox"


@pytest.mark.parametrize("version", [2, "1", 0, None])
def test_an_unknown_backup_version_is_refused(store, tmp_path, version):
    backup = _two_source_backup(store, tmp_path)
    forged = _copy_backup(backup, tmp_path / "forged-version")
    data = _manifest(forged)
    data["backup_version"] = version
    _rewrite(forged, data)

    target = tmp_path / "restored-version"
    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(forged, target)
    assert exc.value.code == "UNSUPPORTED_BACKUP_VERSION", exc.value.code
    assert not target.exists()


def test_an_existing_target_is_never_replaced(store, tmp_path):
    backup = _two_source_backup(store, tmp_path)
    target = tmp_path / "existing-target"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(backup, target)
    assert exc.value.code == "BACKUP_TARGET_EXISTS", exc.value.code
    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert sorted(path.name for path in target.iterdir()) == ["keep.txt"]


def test_an_interrupted_copy_leaves_no_complete_backup(store, tmp_path, monkeypatch):
    _failed_async(store, tmp_path, "interrupted", b"payload" * 64)
    original = backup_module._copy_verified
    calls = {"count": 0}

    def interrupt(source, target, expected):
        calls["count"] += 1
        if calls["count"] == 1:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes()[:16])
            raise OSError("simulated storage failure")
        return original(source, target, expected)

    monkeypatch.setattr(backup_module, "_copy_verified", interrupt)
    target = tmp_path / "interrupted-backup"
    with pytest.raises(OSError):
        backup_workspace(store, target)
    assert not (target / BACKUP_MANIFEST).exists(), "an interrupted copy wrote a manifest"
    partial = [path for path in target.rglob("*") if path.is_file() and path.name != SNAPSHOT_DB]
    assert partial, "the partial copy was silently deleted instead of kept as evidence"

    with pytest.raises(WorkspaceError) as exc:
        restore_workspace(target, tmp_path / "restored-interrupted")
    assert exc.value.code in {"BACKUP_INCOMPLETE", "UNSUPPORTED_BACKUP_VERSION"}, exc.value.code
    assert not (tmp_path / "restored-interrupted").exists()

    # The source project is untouched by the failed backup.
    assert len(store.list_intakes()) == 1
    assert store.list_intakes()[0]["state"] == "failed"
