"""Committed candidates/diagnostics must survive GC, including restart barriers."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from trace_annotate.workspace.backup import backup_workspace, restore_workspace
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.store import WorkspaceStore
from trace_annotate.workspace import schema as S


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Real artifact lifecycle requires FFmpeg")
    path = tmp_path_factory.mktemp("lifecycle-media") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    "color=blue:size=32x32:rate=5:duration=0.4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], capture_output=True, check=True, timeout=30)
    return path


@pytest.fixture
def store(tmp_path, clip):
    store = WorkspaceStore(tmp_path / "project")
    store.create()
    store.import_source(clip)
    try:
        yield store
    finally:
        store.close()


def queue_result(store, *, fail=False):
    queue = DurableQueue(store)
    task_id = store.list_tasks()[0]["task_id"]
    queue.enqueue(task_id, operation="lifecycle-probe", processor_version="v1")
    job = queue.claim("worker-probe")
    if fail:
        queue.fail(job["id"], job["lease_generation"], "worker-probe", diagnostic={"message": "retained failure"})
        ident = queue.get_job(job["id"])["attempts"][0]["diagnostic_artifact_id"]
    else:
        ident = queue.finish(job["id"], job["lease_generation"], "worker-probe", {"candidate": "preserve me"})["artifact"]["id"]
    return ident


@pytest.mark.parametrize("fail", [False, True], ids=["candidate", "diagnostic"])
def test_committed_queue_objects_are_never_unbound(store, tmp_path, fail):
    ident = queue_result(store, fail=fail)
    assert ident not in {item["id"] for item in store.unbound_artifacts()["artifacts"]}
    outcome = store.sweep_unbound_artifacts(apply=True)
    assert outcome["removed"] == 0
    backup_workspace(store, tmp_path / "backup")
    restore_workspace(tmp_path / "backup", tmp_path / "recovered")
    reopened = WorkspaceStore(tmp_path / "recovered")
    reopened.open()
    try:
        assert ident not in {item["id"] for item in reopened.unbound_artifacts()["artifacts"]}
        assert reopened.conn.execute("SELECT 1 FROM artifacts WHERE id=?", (ident,)).fetchone()
    finally:
        reopened.close()


def test_independent_owners_and_historical_task_pin_survive_reopen(store, tmp_path):
    artifact = store._put_artifact(b"one object owned twice")
    task_id = store.list_tasks()[0]["task_id"]
    revision_id = store.conn.execute("SELECT id FROM task_revisions WHERE task_id=? AND revision=0", (task_id,)).fetchone()[0]
    task = store.get_task(task_id)
    store.apply_task_command(command_id="cmd-make-history", task_id=task_id, expected_revision=0,
        expected_sha256=task["content_sha256"], operation="add_annotation", payload={"annotation": {
            "id": "ann-present", "type": "whole_clip", "label": "current revision differs from pinned revision"}}, request_body=b"history")
    with store.transaction():
        store.bind_artifact_reference(artifact, owner_kind="task_revision", owner_id=revision_id, owner_revision="0", role="analysis")
        store.bind_artifact_reference(artifact, owner_kind="export_pin", owner_id="export-frozen", role="render")
    store.release_artifact_reference(artifact["id"], owner_kind="export_pin", owner_id="export-frozen", role="render")
    assert store.sweep_unbound_artifacts(apply=True)["removed"] == 0
    backup = backup_workspace(store, tmp_path / "backup-pinned")
    assert any(r["artifact_id"] == artifact["id"] for r in backup["artifact_references"])
    restore_workspace(tmp_path / "backup-pinned", tmp_path / "restored-pinned")
    store.close()
    store.open()
    assert store.unbound_artifacts()["count"] == 0
    store.release_artifact_reference(artifact["id"], owner_kind="task_revision", owner_id=revision_id, owner_revision="0", role="analysis")
    assert store.sweep_unbound_artifacts(apply=True)["removed"] == 1


@pytest.mark.parametrize("corruption", ["media_json", "missing_diagnostic", "commit_hash", "unknown_owner", "dangling_task_owner"])
def test_corrupt_closure_fails_closed_without_deleting_orphan(store, corruption):
    orphan = store._put_artifact(b"must not be deleted with corrupt owners")
    if corruption == "media_json":
        store.conn.execute("UPDATE meta SET value='{' WHERE key LIKE 'media_objects:%'")
    elif corruption == "missing_diagnostic":
        queue_result(store, fail=True)
        store.conn.execute("UPDATE attempts SET diagnostic_artifact_id='art-missing'")
    elif corruption == "commit_hash":
        queue_result(store)
        row = store.conn.execute("SELECT job_id,body FROM job_commits").fetchone()
        body = json.loads(row[1]); body["artifact"]["sha256"] = "0" * 64
        store.conn.execute("UPDATE job_commits SET body=? WHERE job_id=?", (json.dumps(body), row[0]))
    else:
        store.conn.execute("INSERT INTO artifact_references VALUES(?,?,?,?,?)", (orphan["id"], "unknown" if corruption == "unknown_owner" else "task_revision", "rev-missing", "0", "analysis"))
    with pytest.raises(WorkspaceError) as error:
        store.sweep_unbound_artifacts(apply=True)
    assert error.value.code == "RECOVERY_REQUIRED"
    assert (store.root / orphan["relative_path"]).read_bytes() == b"must not be deleted with corrupt owners"
    assert store.conn.execute("SELECT 1 FROM artifacts WHERE id=?", (orphan["id"],)).fetchone()


def test_missing_or_forged_pin_is_rejected(store):
    artifact = store._put_artifact(b"identity")
    forged = dict(artifact, sha256="0" * 64)
    with pytest.raises(WorkspaceError):
        store.bind_artifact_reference(forged, owner_kind="analysis_report", owner_id="report-one", role="result")
    assert store.conn.execute("SELECT COUNT(*) FROM artifact_references").fetchone()[0] == 0


def test_legacy_snapshot_without_pin_table_recovers_inferred_queue_closure(store, tmp_path):
    ident = queue_result(store)
    backup = tmp_path / "legacy-backup"
    backup_workspace(store, backup)
    connection = sqlite3.connect(str(backup / "snapshot.sqlite3"))
    connection.execute("DROP TABLE artifact_references")
    connection.execute("DROP TABLE artifact_gc_pending")
    connection.execute("UPDATE meta SET value='1' WHERE key='storage_version'")
    connection.commit(); connection.close()
    path = backup / "backup-manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest.pop("artifact_references")
    manifest["metadata"]["storage_version"] = "1"
    db = backup / "snapshot.sqlite3"
    manifest["database"].update(sha256=hashlib.sha256(db.read_bytes()).hexdigest(), size_bytes=str(db.stat().st_size))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = restore_workspace(backup, tmp_path / "restored-legacy")
    assert receipt["storage_migration"] == {"source_version": "1", "restored_version": "2", "source_backup_modified": False}
    reopened = WorkspaceStore(tmp_path / "restored-legacy")
    reopened.open()
    try:
        assert ident not in {a["id"] for a in reopened.unbound_artifacts()["artifacts"]}
    finally:
        reopened.close()
    assert "artifact_references" not in json.loads(path.read_text("utf-8")), "source backup was rewritten"


def test_failed_legacy_migration_rolls_back_version_and_additive_tables(store):
    root = store.root
    artifact = store._put_artifact(b"tamper before version1 migration")
    store.close()
    connection = sqlite3.connect(str(root / "workspace.sqlite3"))
    connection.execute("DROP TABLE artifact_references")
    connection.execute("DROP TABLE artifact_gc_pending")
    connection.execute("UPDATE meta SET value='1' WHERE key='storage_version'")
    connection.commit(); connection.close()
    (root / artifact["relative_path"]).write_bytes(b"changed")
    with pytest.raises(WorkspaceError) as error:
        store.open()
    assert error.value.code == "RECOVERY_REQUIRED"
    connection = sqlite3.connect(str(root / "workspace.sqlite3"))
    try:
        assert connection.execute("SELECT value FROM meta WHERE key='storage_version'").fetchone()[0] == "1"
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name IN ('artifact_references','artifact_gc_pending')").fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize("table", ["artifact_references", "artifact_gc_pending"])
def test_version2_missing_ownership_table_is_not_recreated_as_empty(store, table):
    artifact = store._put_artifact(b"must preserve lost-reference evidence")
    store.bind_artifact_reference(artifact, owner_kind="export_pin", owner_id="export-before-corruption", role="saved")
    root = store.root
    store.close()
    connection = sqlite3.connect(str(root / "workspace.sqlite3"))
    connection.execute("DROP TABLE " + table)
    connection.commit(); connection.close()
    with pytest.raises(WorkspaceError) as error:
        store.open()
    assert error.value.code == "RECOVERY_REQUIRED"
    assert (root / artifact["relative_path"]).read_bytes() == b"must preserve lost-reference evidence"
    connection = sqlite3.connect(str(root / "workspace.sqlite3"))
    try:
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone()
    finally:
        connection.close()


def test_new_storage_version_rejects_old_writer_before_it_can_run_old_gc(store, tmp_path):
    ident = queue_result(store)
    root = store.root
    store.close()
    # Pin the actual pre-migration writer. HEAD stops being an old writer after
    # this fix is committed, and an sdist/source snapshot has no .git at all.
    # The fixture is MIT-licensed project code from commit209d0ccc7940e3b6ff630c2ba0897c28cc7c8aef.
    old_source = (Path(__file__).parent / "fixtures/storage_v1_store.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(old_source).hexdigest() == "603d2bb85fd8b599a344176534362b4c7f0477887d65a09bfacc25f20501810e"
    script = tmp_path / "old_store.py"
    script.write_bytes(old_source)
    child = r'''
import sys,types
from pathlib import Path
module=types.ModuleType('trace_annotate.workspace.old_storage_probe')
module.__package__='trace_annotate.workspace'
sys.modules[module.__name__]=module
exec(compile(Path(sys.argv[1]).read_bytes(),sys.argv[1],'exec'),module.__dict__)
s=module.WorkspaceStore(Path(sys.argv[2]))
try:s.open()
except Exception as exc:
    assert getattr(exc,'code',None)=='WORKSPACE_MIGRATION_REQUIRED',repr(exc)
    raise SystemExit(0)
else:raise SystemExit(91)
'''
    result = subprocess.run([sys.executable, "-B", "-c", child, str(script), str(root)], capture_output=True, timeout=30)
    (tmp_path / "legacy-writer.stdout").write_bytes(result.stdout)
    (tmp_path / "legacy-writer.stderr").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    store.open()
    assert store.conn.execute("SELECT value FROM meta WHERE key='storage_version'").fetchone()[0] == "2"
    assert ident not in {a["id"] for a in store.unbound_artifacts()["artifacts"]}


def test_new_snapshot_reference_manifest_cannot_be_omitted_or_tampered(store, tmp_path):
    artifact = store._put_artifact(b"pinned backup bytes")
    store.bind_artifact_reference(artifact, owner_kind="export_pin", owner_id="export-immutable", role="original")
    target = tmp_path / "backup"
    backup_workspace(store, target)
    path = target / "backup-manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    original = manifest["artifact_references"]
    for index, alteration in enumerate((None, [], [dict(original[0], owner_id="forged")])):
        changed = dict(manifest)
        if alteration is None:
            changed.pop("artifact_references")
        else:
            changed["artifact_references"] = alteration
        path.write_text(json.dumps(changed), encoding="utf-8")
        destination = tmp_path / ("bad-restore-" + str(index))
        with pytest.raises(WorkspaceError) as error:
            restore_workspace(target, destination)
        assert error.value.code == "BACKUP_INTEGRITY_FAILED"
        assert not destination.exists()


def test_durable_commit_remains_an_owner_without_queue_projection(store):
    ident = queue_result(store)
    store.conn.execute("DELETE FROM job_commits")
    assert ident not in {a["id"] for a in store.unbound_artifacts()["artifacts"]}


def test_corrupt_cleanup_journal_cannot_touch_external_path(store, tmp_path):
    outside = tmp_path / "user-file"
    outside.write_bytes(b"untouched")
    store.conn.execute("INSERT INTO artifact_gc_pending VALUES(?,?,?,?)", ("art-invalid", "../user-file", hashlib.sha256(b"untouched").hexdigest(), 9))
    with pytest.raises(WorkspaceError) as error:
        store.sweep_unbound_artifacts(apply=True)
    assert error.value.code == "RECOVERY_REQUIRED"
    assert outside.read_bytes() == b"untouched"


def test_cleanup_cannot_unlink_inside_outer_rollback(store):
    artifact = store._put_artifact(b"outer rollback")
    with store.transaction():
        with pytest.raises(WorkspaceError) as error:
            store.sweep_unbound_artifacts(apply=True)
        assert error.value.code == "GC_IN_TRANSACTION"
    assert (store.root / artifact["relative_path"]).read_bytes() == b"outer rollback"


def test_uncommitted_media_publication_is_protected_until_live_decoder_exits(store):
    artifact = store._put_artifact(b"in-flight admission object")
    store.conn.execute("INSERT INTO intakes(id,group_id,state,created_at) VALUES('int-inflight','grp-live','preparing','now')")
    token = object()
    store._active_media_publications = {token}
    with pytest.raises(WorkspaceError) as error:
        store.sweep_unbound_artifacts(apply=True)
    assert error.value.code == "GC_ACTIVE_WORK"
    assert (store.root / artifact["relative_path"]).is_file()
    store._active_media_publications.remove(token)
    # A stale persisted preparing row after a real process exit owns no live
    # decoder in this exclusive-open process and cannot block cleanup forever.
    assert store.sweep_unbound_artifacts(apply=True)["removed"] == 1


def test_live_import_entry_guard_blocks_gc_and_is_released_on_exit(store, clip):
    entered, release = threading.Event(), threading.Event()
    calls, problems = [], []
    store._put_artifact(b"unrelated orphan must wait for admission")
    def cancellation_check():
        if not calls:
            calls.append(True)
            entered.set()
            if not release.wait(10):
                raise TimeoutError("import entry guard not released")
        return False
    def admit():
        try:
            store.import_source(clip, cancelled=cancellation_check)
        except BaseException as exc:
            problems.append(exc)
    worker = threading.Thread(target=admit)
    worker.start()
    try:
        assert entered.wait(10)
        with pytest.raises(WorkspaceError) as error:
            store.sweep_unbound_artifacts(apply=True)
        assert error.value.code == "GC_ACTIVE_WORK"
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not problems
    assert not store._active_media_publications
    assert store.sweep_unbound_artifacts(apply=True)["removed"] == 1


def test_plain_annotation_command_cannot_forge_imported_provenance(store):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    with pytest.raises(WorkspaceError) as error:
        store.apply_task_command(command_id="cmd-forged-import", task_id=task["id"],
            expected_revision=task["revision"], expected_sha256=task["content_sha256"],
            operation="add_annotation", payload={"annotation": {"id": "ann-forged", "type": "whole_clip", "label": "fake import",
                "provenance": {"origin": "imported", "actor_id": "local-user", "job_id": None}}}, request_body=b"fake")
    assert error.value.code == "SCHEMA_INVALID"
    assert store.get_task(task["id"]) == task


@pytest.mark.parametrize("changes", [
    {"origin": "manual", "actor_id": "local-user", "job_id": None},
    {"origin": "imported", "actor_id": "other-actor", "job_id": None},
])
def test_plain_edit_cannot_launder_previously_imported_origin(store, clip, changes):
    from trace_annotate.workspace.professional_store import export_professional
    source = store.get_task(store.list_tasks()[0]["task_id"])
    store.apply_task_command(command_id="cmd-seed-import", task_id=source["id"], expected_revision=0,
        expected_sha256=source["content_sha256"], operation="add_annotation", payload={"annotation": {
            "id": "ann-source", "type": "whole_clip", "label": "source"}}, request_body=b"source")
    document = json.loads(export_professional(store, source["id"])["data"])
    destination_id = store.import_source(clip)["task_id"]
    destination = store.get_task(destination_id)
    store.apply_task_command(command_id="cmd-trusted-import", task_id=destination_id, expected_revision=0,
        expected_sha256=destination["content_sha256"], operation="import_professional",
        payload={"document": document}, request_body=b"trusted-import")
    imported = store.get_task(destination_id)
    annotation = imported["annotations"][0]
    assert annotation["provenance"]["origin"] == "imported"
    with pytest.raises(WorkspaceError) as error:
        store.apply_task_command(command_id="cmd-launder", task_id=destination_id, expected_revision=1,
            expected_sha256=imported["content_sha256"], operation="update_annotation",
            payload={"annotation_id": annotation["id"], "changes": {"provenance": changes}}, request_body=b"launder")
    assert error.value.code == "SCHEMA_INVALID"
    assert store.get_task(destination_id) == imported
    # Editing actual content remains usable without erasing imported provenance.
    store.apply_task_command(command_id="cmd-allowed-edit", task_id=destination_id, expected_revision=1,
        expected_sha256=imported["content_sha256"], operation="update_annotation",
        payload={"annotation_id": annotation["id"], "changes": {"label": "edited locally"}}, request_body=b"content-edit")
    edited = store.get_task(destination_id)["annotations"][0]
    assert edited["provenance"] == annotation["provenance"]


def test_malformed_historical_reference_array_fails_closed(store):
    task_id = store.list_tasks()[0]["task_id"]
    row = store.conn.execute("SELECT id,body FROM task_revisions WHERE task_id=? AND revision=0", (task_id,)).fetchone()
    body = json.loads(row[1])
    body["extensions"] = {"professional": {"analysis_reports": [{"artifact_ids": ""}]}}
    store.conn.execute("UPDATE task_revisions SET body=? WHERE id=?", (json.dumps(body), row[0]))
    with pytest.raises(WorkspaceError) as error:
        store.unbound_artifacts()
    assert error.value.code == "RECOVERY_REQUIRED"


def test_tombstone_transaction_failure_keeps_object_and_registration(store, monkeypatch):
    artifact = store._put_artifact(b"sql failure")
    original = store.transaction
    @contextmanager
    def failed_commit():
        with original() as conn:
            yield conn
            if conn.execute("SELECT COUNT(*) FROM artifact_gc_pending").fetchone()[0]:
                raise RuntimeError("injected before SQL commit")
    monkeypatch.setattr(store, "transaction", failed_commit)
    with pytest.raises(RuntimeError, match="before SQL commit"):
        store.sweep_unbound_artifacts(apply=True)
    assert (store.root / artifact["relative_path"]).is_file()
    assert store.conn.execute("SELECT 1 FROM artifacts WHERE id=?", (artifact["id"],)).fetchone()
    assert store.conn.execute("SELECT COUNT(*) FROM artifact_gc_pending").fetchone()[0] == 0


def test_unlink_failure_keeps_durable_cleanup_and_restart_recovers(store, monkeypatch, tmp_path):
    artifact = store._put_artifact(b"unlink temporarily denied")
    path = store.root / artifact["relative_path"]
    unlink = Path.unlink
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("injected sharing violation")
        return unlink(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        with pytest.raises(PermissionError):
            store.sweep_unbound_artifacts(apply=True)
    assert path.is_file()
    assert store.conn.execute("SELECT COUNT(*) FROM artifact_gc_pending").fetchone()[0] == 1
    assert store.conn.execute("SELECT 1 FROM artifacts WHERE id=?", (artifact["id"],)).fetchone() is None
    with pytest.raises(WorkspaceError) as error:
        backup_workspace(store, tmp_path / "incomplete-gc-backup")
    assert error.value.code == "GC_RECOVERY_REQUIRED"
    store.close()
    store.open()
    assert not path.exists()
    assert store.conn.execute("SELECT COUNT(*) FROM artifact_gc_pending").fetchone()[0] == 0
    # The exact same bytes may be republished after recovery.
    assert store._put_artifact(b"unlink temporarily denied") == artifact


@pytest.mark.parametrize("barrier,exit_code", [("committed", 73), ("unlinked", 74)])
def test_real_process_exit_at_gc_barriers_recovers(store, tmp_path, barrier, exit_code):
    artifact = store._put_artifact(b"crash barrier object")
    root = store.root
    store.close()
    script = r'''
import os,sys
from pathlib import Path
from trace_annotate.workspace.store import WorkspaceStore
s=WorkspaceStore(Path(sys.argv[1]));s.open()
if sys.argv[2]=='committed':
    original=s._recover_pending_gc
    def barrier():
        if s.conn.execute('SELECT COUNT(*) FROM artifact_gc_pending').fetchone()[0]:os._exit(73)
        return original()
    s._recover_pending_gc=barrier
else:
    unlink=Path.unlink
    def barrier(self,*args,**kwargs):
        result=unlink(self,*args,**kwargs)
        if self.name==sys.argv[3]:os._exit(74)
        return result
    Path.unlink=barrier
s.sweep_unbound_artifacts(apply=True)
raise SystemExit(99)
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(root), barrier, artifact["sha256"]], capture_output=True, timeout=30)
    (tmp_path / (barrier + "-stdout.log")).write_bytes(result.stdout)
    (tmp_path / (barrier + "-stderr.log")).write_bytes(result.stderr)
    assert result.returncode == exit_code, result.stderr.decode(errors="replace")
    store.open()
    assert not (root / artifact["relative_path"]).exists()
    assert store.conn.execute("SELECT COUNT(*) FROM artifact_gc_pending").fetchone()[0] == 0
    assert store.unbound_artifacts()["count"] == 0


def test_publication_and_new_pin_serialize_against_gc(store, monkeypatch):
    linked, release, sweep_entered, swept = (threading.Event() for _ in range(4))
    data = b"publish concurrently with cleanup"
    digest = hashlib.sha256(data).hexdigest()
    link = os.link
    errors = []
    def gated_link(source, target, *args, **kwargs):
        result = link(source, target, *args, **kwargs)
        if Path(target).name == digest:
            linked.set()
            if not release.wait(10):
                raise TimeoutError("publication barrier not released")
        return result
    monkeypatch.setattr(os, "link", gated_link)
    def publish():
        try:
            with store.transaction():
                artifact = store._put_artifact(data)
                store.bind_artifact_reference(artifact, owner_kind="analysis_report", owner_id="analysis-published", role="report")
        except BaseException as exc:
            errors.append(exc)
    def sweep():
        sweep_entered.set()
        try:
            store.sweep_unbound_artifacts(apply=True)
            swept.set()
        except BaseException as exc:
            errors.append(exc)
    writer = threading.Thread(target=publish)
    collector = threading.Thread(target=sweep)
    writer.start()
    try:
        assert linked.wait(10)
        collector.start()
        assert sweep_entered.wait(10)
        assert not swept.is_set()
    finally:
        release.set()
        writer.join(10)
        if collector.ident is not None:
            collector.join(10)
    assert not writer.is_alive() and not collector.is_alive() and not errors
    assert (store.objects_dir / digest).read_bytes() == data
    assert store.unbound_artifacts()["count"] == 0
