"""Backup is proven by fresh-path recovery, not by a copied SQLite filename."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading

import pytest

from trace_annotate.workspace import backup as B
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture
def store(tmp_path):
    value = WorkspaceStore(tmp_path / "original")
    value.create()
    value._put_artifact(b"a durable registered object")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("Real recovery media fixture requires FFmpeg and ffprobe")
    path = tmp_path_factory.mktemp("backup-media") / "clip.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    "testsrc2=duration=2:size=160x120:rate=10", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=30)
    return path


def _command(store, ident, task_id, operation="add_annotation", payload=None):
    task = store.get_task(task_id)
    return store.apply_task_command(command_id=ident, task_id=task_id,
        expected_revision=task["revision"], expected_sha256=task["content_sha256"],
        operation=operation,
        payload=payload or {"annotation": {"id": "ann-preserved", "type": "segment", "label": "原文 -42",
                    "text": "中文\n1,234", "tags": ["same", "same"],
                    "start": {"seconds": S.rational("0.1")}, "end": {"seconds": S.rational("1.9")}}},
        request_body=ident.encode("ascii"))


def test_wal_backup_recovers_real_media_revisions_and_original_receipt(store, video, tmp_path):
    admission = store.import_source(video)
    task_id = admission["task_id"]
    original = _command(store, "cmd-before-backup", task_id)
    task_before = store.get_task(task_id)
    receipt_before = store.get_command_receipt("cmd-before-backup")
    before_db_hash = hashlib.sha256((store.root / "workspace.sqlite3").read_bytes()).hexdigest()
    # The source can still have uncheckpointed WAL. The snapshot must include it.
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    target = tmp_path / "backup"
    manifest = B.backup_workspace(store, target)
    assert manifest["state"] == "complete"
    assert manifest["scope"] == "quiescent-sqlite-and-registered-objects"
    assert (target / B.SNAPSHOT_DB).is_file()
    assert not (target / "workspace.sqlite3").exists()
    # No raw SQLite source-file copying assumption: a recovered fresh store is the oracle.
    restored = tmp_path / "recovered"
    outcome = B.restore_workspace(target, restored)
    assert outcome["jobs_restarted"] is False and outcome["commands_replayed"] is False
    recovered = WorkspaceStore(restored)
    recovered.open()
    try:
        assert recovered.get_task(task_id) == task_before
        assert recovered.get_command_receipt("cmd-before-backup") == receipt_before
        replay = recovered.apply_task_command(command_id="cmd-before-backup", task_id=task_id,
            expected_revision=0,
            expected_sha256=store.conn.execute("SELECT content_sha256 FROM task_revisions WHERE task_id=? AND revision=0", (task_id,)).fetchone()[0],
            operation="add_annotation", payload={"annotation": {"id": "ann-preserved", "type": "segment", "label": "原文 -42",
                    "text": "中文\n1,234", "tags": ["same", "same"],
                    "start": {"seconds": S.rational("0.1")}, "end": {"seconds": S.rational("1.9")}}},
            request_body=b"cmd-before-backup")
        assert replay["idempotent_replay"] and replay["response"] == original["response"]
        _command(recovered, "cmd-after-restore", task_id, "update_annotation",
                 {"annotation_id": "ann-preserved", "changes": {"label": "continued"}})
    finally:
        recovered.close()
    recovered.open()
    try:
        assert recovered.get_task(task_id)["annotations"][0]["label"] == "continued"
        assert recovered.get_task(task_id)["revision"] == task_before["revision"] + 1
    finally:
        recovered.close()
    assert store.get_task(task_id) == task_before
    assert before_db_hash == hashlib.sha256((store.root / "workspace.sqlite3").read_bytes()).hexdigest()


@pytest.mark.parametrize("operation", ["backup", "restore"])
def test_existing_target_is_never_replaced(store, tmp_path, operation):
    backup = tmp_path / "backup"
    B.backup_workspace(store, backup)
    existing = tmp_path / "user-data"
    existing.mkdir()
    sentinel = existing / "preserve.bin"
    sentinel.write_bytes(b"user data")
    with pytest.raises(WorkspaceError) as error:
        if operation == "backup":
            B.backup_workspace(store, existing)
        else:
            B.restore_workspace(backup, existing)
    assert error.value.code == "BACKUP_TARGET_EXISTS"
    assert sentinel.read_bytes() == b"user data"
    assert sorted(path.name for path in existing.iterdir()) == ["preserve.bin"]


def test_backup_refuses_an_active_source_transaction_without_hanging(store, tmp_path):
    target = tmp_path / "backup"
    with store.transaction():
        with pytest.raises(WorkspaceError) as error:
            B.backup_workspace(store, target)
        assert error.value.code == "BACKUP_IN_TRANSACTION"
    assert not target.exists()


def test_backup_refuses_queued_jobs_without_starting_them(store, video, tmp_path):
    task_id = store.import_source(video)["task_id"]
    queue = DurableQueue(store)
    queued = queue.enqueue(task_id, operation="verify-media", processor_version="1")
    with pytest.raises(WorkspaceError) as error:
        B.backup_workspace(store, tmp_path / "backup")
    assert error.value.code == "BACKUP_ACTIVE_WORK"
    assert queue.get_job(queued["id"])["state"] == "queued"
    assert queue.get_job(queued["id"])["attempt"] == 0


def test_backup_refuses_running_intake_without_changing_it(store, tmp_path):
    store.conn.execute("INSERT INTO intakes(id,group_id,state,created_at) VALUES('int-active','grp-active','running','now')")
    with pytest.raises(WorkspaceError) as error:
        B.backup_workspace(store, tmp_path / "backup")
    assert error.value.code == "BACKUP_ACTIVE_WORK"
    assert store.conn.execute("SELECT state FROM intakes WHERE id='int-active'").fetchone()[0] == "running"


def test_object_tampering_prevents_restore_before_creating_target(store, tmp_path):
    backup = tmp_path / "backup"
    manifest = B.backup_workspace(store, backup)
    artifact = backup / manifest["artifacts"][0]["relative_path"]
    artifact.write_bytes(b"tampered")
    target = tmp_path / "restore"
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, target)
    assert error.value.code == "BACKUP_INTEGRITY_FAILED"
    assert not target.exists()


def test_unknown_schema_rejected_even_with_updated_database_fingerprint(store, tmp_path):
    backup = tmp_path / "backup"
    manifest = B.backup_workspace(store, backup)
    db = backup / B.SNAPSHOT_DB
    connection = sqlite3.connect(str(db), isolation_level=None)
    connection.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    connection.close()
    manifest["database"].update(B._fingerprint(db))
    (backup / B.BACKUP_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, tmp_path / "restore")
    assert error.value.code == "UNSUPPORTED_BACKUP_VERSION"
    assert not (tmp_path / "restore").exists()


def test_manifest_cannot_change_the_database_artifact_closure(store, tmp_path):
    backup = tmp_path / "backup"
    manifest = B.backup_workspace(store, backup)
    manifest["artifacts"][0]["relative_path"] = "../outside"
    (backup / B.BACKUP_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, tmp_path / "restore")
    assert error.value.code == "BACKUP_INTEGRITY_FAILED"
    assert not (tmp_path / "restore").exists()


def test_undeclared_wal_is_refused_even_when_database_hash_is_valid(store, tmp_path):
    backup = tmp_path / "backup"
    B.backup_workspace(store, backup)
    (backup / (B.SNAPSHOT_DB + "-wal")).write_bytes(b"untracked sidecar")
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, tmp_path / "restore")
    assert error.value.code == "BACKUP_INCOMPLETE"
    assert not (tmp_path / "restore").exists()


def test_manifest_changed_during_recovery_is_not_reported_complete(store, tmp_path, monkeypatch):
    backup = tmp_path / "backup"
    B.backup_workspace(store, backup)
    original_copy = B._copy_verified
    def change_manifest(source, target, expected):
        result = original_copy(source, target, expected)
        if target.name == "workspace.sqlite3":
            with (backup / B.BACKUP_MANIFEST).open("ab") as manifest:
                manifest.write(b"\n")
        return result
    monkeypatch.setattr(B, "_copy_verified", change_manifest)
    target = tmp_path / "restore"
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, target)
    assert error.value.code == "BACKUP_INTEGRITY_FAILED"
    assert not (target / B.RESTORE_MANIFEST).exists()


def test_closed_store_backup_does_not_create_a_target(store, tmp_path):
    store.close()
    with pytest.raises(WorkspaceError) as error:
        B.backup_workspace(store, tmp_path / "backup")
    assert error.value.code == "PROJECT_CLOSED"
    assert not (tmp_path / "backup").exists()


def test_interrupted_backup_has_no_complete_manifest_and_is_not_restorable(store, tmp_path, monkeypatch):
    backup = tmp_path / "backup"
    def interruption(*args):
        raise OSError("injected disk failure")
    monkeypatch.setattr(B, "_copy_verified", interruption)
    with pytest.raises(OSError):
        B.backup_workspace(store, backup)
    assert backup.is_dir()
    assert (backup / B.SNAPSHOT_DB).is_file()
    assert not (backup / B.BACKUP_MANIFEST).exists()
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, tmp_path / "restore")
    assert error.value.code == "BACKUP_INCOMPLETE"


def test_interrupted_restore_retains_evidence_without_completion_marker(store, tmp_path, monkeypatch):
    backup = tmp_path / "backup"
    B.backup_workspace(store, backup)
    original_copy = B._copy_verified
    def interruption(source, target, expected):
        if target.name == "workspace.sqlite3":
            raise OSError("injected restore interruption")
        return original_copy(source, target, expected)
    monkeypatch.setattr(B, "_copy_verified", interruption)
    target = tmp_path / "restore"
    with pytest.raises(OSError):
        B.restore_workspace(backup, target)
    assert target.is_dir() and any((target / "objects").iterdir())
    assert not (target / B.RESTORE_MANIFEST).exists()
    assert not (target / "workspace.sqlite3").exists()
    assert (backup / B.BACKUP_MANIFEST).is_file()


def test_backup_and_restore_reject_nested_targets(store, tmp_path):
    with pytest.raises(WorkspaceError) as error:
        B.backup_workspace(store, store.root / "backup")
    assert error.value.code == "UNSAFE_PATH"
    backup = tmp_path / "backup"
    B.backup_workspace(store, backup)
    with pytest.raises(WorkspaceError) as error:
        B.restore_workspace(backup, backup / "restore")
    assert error.value.code == "UNSAFE_PATH"


def test_writes_wait_until_database_and_object_closure_are_complete(store, tmp_path, monkeypatch):
    entered, release, writer_attempted, writer_finished = (threading.Event() for _ in range(4))
    problems = []
    original_copy = B._copy_verified
    def barrier(source, target, expected):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release copy barrier")
        return original_copy(source, target, expected)
    monkeypatch.setattr(B, "_copy_verified", barrier)
    def snapshot():
        try:
            B.backup_workspace(store, tmp_path / "backup")
        except BaseException as exc:
            problems.append(exc)
    def write():
        writer_attempted.set()
        try:
            with store.transaction():
                store._put_artifact(b"new after frozen snapshot")
            writer_finished.set()
        except BaseException as exc:
            problems.append(exc)
    backup_thread = threading.Thread(target=snapshot)
    writer_thread = threading.Thread(target=write)
    backup_thread.start()
    try:
        assert entered.wait(10)
        writer_thread.start()
        assert writer_attempted.wait(10)
        assert not writer_finished.is_set()
    finally:
        release.set()
        backup_thread.join(10)
        if writer_thread.ident is not None:
            writer_thread.join(10)
    assert not backup_thread.is_alive() and not writer_thread.is_alive()
    assert not problems
    assert writer_finished.is_set()
    B.restore_workspace(tmp_path / "backup", tmp_path / "restore")
    snapshot = WorkspaceStore(tmp_path / "restore")
    snapshot.open()
    try:
        assert snapshot.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
    finally:
        snapshot.close()
    assert store.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 2
