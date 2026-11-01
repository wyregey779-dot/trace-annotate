"""Consistent, quiescent workspace backups and verified recovery to a new path.

The snapshot contains SQLite and every registered immutable object, not arbitrary
exports/logs/tmp files or external original media. Active intakes and nonterminal
queue jobs are refused: this API never restarts a job, lease or pending command.
It never overwrites a target or changes a source snapshot. Opening a recovered
legacy v1 copy atomically upgrades only its additive storage ownership schema to
v2; task schema/digests and business history remain unchanged.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Tuple
import uuid

from . import schema as S
from .errors import Conflict, WorkspaceError
from .artifact_refs import collect_artifact_references
from .store import (DB_FILE, STORAGE_VERSION, SCHEMA_VERSION, WorkspaceStore,
                    _safe, _sync_directory)

BACKUP_MANIFEST = "backup-manifest.json"
RESTORE_MANIFEST = "restore-manifest.json"
SNAPSHOT_DB = "snapshot.sqlite3"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_META_KEYS = ("project_id", "schema_version", "storage_version", "digest_version", "workspace_revision")
_TERMINAL = ("succeeded", "failed", "cancelled")


def _fail(code: str, message: str) -> None:
    raise WorkspaceError(code, message)


def _regular(path: Path) -> None:
    _safe(path)
    if not path.is_file():
        _fail("BACKUP_INCOMPLETE", "A required snapshot file is missing")


def _fingerprint(path: Path) -> Dict[str, Any]:
    _regular(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            data = source.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
            size += len(data)
    return {"sha256": digest.hexdigest(), "size_bytes": str(size)}


def _copy_verified(source: Path, target: Path, expected: Dict[str, Any]) -> None:
    _regular(source)
    _safe(target)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        while True:
            data = incoming.read(1024 * 1024)
            if not data:
                break
            outgoing.write(data)
            digest.update(data)
            size += len(data)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if digest.hexdigest() != expected["sha256"] or str(size) != expected["size_bytes"]:
        _fail("BACKUP_INTEGRITY_FAILED", "Snapshot file is missing, changed or corrupt")


def _destination(path: Path, source: Path) -> Path:
    path, source = Path(path).absolute(), Path(source).absolute()
    _safe(path)
    if str(path).startswith(("\\\\", "//")):
        _fail("UNSUPPORTED_STORAGE", "Network recovery targets are unsupported")
    if path == source or path in source.parents or source in path.parents:
        _fail("UNSAFE_PATH", "Snapshot and project directories must not contain one another")
    if path.exists():
        raise Conflict("BACKUP_TARGET_EXISTS", "Choose a new, nonexistent snapshot directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe(path.parent)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise Conflict("BACKUP_TARGET_EXISTS", "Snapshot target already exists") from exc
    for name in ("objects", "tmp", "exports", "logs"):
        (path / name).mkdir()
    return path


def _quiescent(connection: sqlite3.Connection) -> None:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"meta", "artifacts", "tasks", "task_revisions", "command_receipts", "intakes"} <= tables:
        _fail("BACKUP_INCOMPLETE", "Workspace tables are missing")
    if "artifact_gc_pending" in tables and connection.execute("SELECT 1 FROM artifact_gc_pending LIMIT 1").fetchone():
        raise Conflict("GC_RECOVERY_REQUIRED", "Complete interrupted artifact cleanup before backup")
    if connection.execute("SELECT 1 FROM intakes WHERE state NOT IN ('succeeded','failed','cancelled') LIMIT 1").fetchone():
        raise Conflict("BACKUP_ACTIVE_WORK", "Finish or cancel media admission before taking a snapshot")
    queue_tables = {"batches", "jobs", "batch_jobs", "attempts", "job_commits"}
    if tables & queue_tables:
        if not queue_tables <= tables:
            _fail("BACKUP_INCOMPLETE", "Queue tables are incomplete")
        for table in ("jobs", "attempts"):
            if connection.execute("SELECT 1 FROM " + table +
                                  " WHERE state NOT IN ('succeeded','failed','cancelled') LIMIT 1").fetchone():
                raise Conflict("BACKUP_ACTIVE_WORK", "Resolve nonterminal queue work before backup or restore")
    if connection.execute("SELECT 1 FROM command_receipts WHERE state!='committed' LIMIT 1").fetchone():
        raise Conflict("BACKUP_ACTIVE_WORK", "Snapshot has an unresolved command state")


def _retryable_sources(connection: sqlite3.Connection) -> list:
    """Describe the exact failed-upload closure using database identities only.

    Legacy synchronous failures have no workspace-owned upload. They remain
    nonretryable external-source records; an async failure may not silently
    lose its staged source and still be called a complete backup.
    """
    columns = {row[1] for row in connection.execute("PRAGMA table_info(intakes)")}
    if "staged_path" not in columns:
        return []  # Supported pre-async snapshots contain only legacy intakes.
    request_column = "request_sha256" if "request_sha256" in columns else "NULL"
    result = []
    for ident, relative, digest, size, request_sha in connection.execute(
            "SELECT id,staged_path,source_sha256,source_size," + request_column +
            " FROM intakes WHERE state='failed' ORDER BY id"):
        if relative is None and request_sha is None:
            continue
        try:
            S.require_id(ident, "intake_id")
            S.require_hash(digest)
        except WorkspaceError as exc:
            raise WorkspaceError("BACKUP_INTEGRITY_FAILED", "Retryable intake identity is invalid") from exc
        if relative != f"tmp/intakes/{ident}/source" or type(size) is not int or size < 0:
            _fail("BACKUP_INTEGRITY_FAILED", "Retryable intake path or size is invalid")
        result.append({"intake_id": ident, "relative_path": relative,
                       "sha256": digest, "size_bytes": str(size)})
    return result


def _source_path(root: Path, declaration: Dict[str, Any]) -> Path:
    """Bind a retry source to its sole supported lexical path before any I/O."""
    ident = declaration.get("intake_id")
    try:
        S.require_id(ident, "intake_id")
    except WorkspaceError as exc:
        raise WorkspaceError("BACKUP_INTEGRITY_FAILED", "Retryable source ID is invalid") from exc
    relative = declaration.get("relative_path")
    if relative != f"tmp/intakes/{ident}/source":
        _fail("BACKUP_INTEGRITY_FAILED", "Retryable source path is invalid")
    root = Path(root).absolute()
    target = root / relative
    _safe(root)
    _safe(target)
    try:
        target.relative_to(root)
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise WorkspaceError("UNSAFE_PATH", "Retryable source escapes its workspace") from exc
    return target


def _compare_retryable_closure(declared, db_records) -> None:
    """Compare manifest closure to the DB closure as an order-independent set.

    Order of entries is not part of the semantics, but duplicates, missing or
    extra IDs, and any per-entry identity mismatch must be rejected.
    """
    if not isinstance(declared, list) or not isinstance(db_records, list):
        _fail("BACKUP_INTEGRITY_FAILED", "Retryable source closure has wrong shape")
    by_id: Dict[str, Dict[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            _fail("BACKUP_INTEGRITY_FAILED", "Retryable source entry has wrong shape")
        ident = entry.get("intake_id")
        try:
            S.require_id(ident, "intake_id")
        except WorkspaceError as exc:
            raise WorkspaceError("BACKUP_INTEGRITY_FAILED", "Retryable source ID is invalid") from exc
        if ident in by_id:
            _fail("BACKUP_INTEGRITY_FAILED", "Duplicate retryable source intake_id in manifest")
        # Re-run the lexical path guard so a forged relative_path cannot slip.
        if entry.get("relative_path") != f"tmp/intakes/{ident}/source":
            _fail("BACKUP_INTEGRITY_FAILED", "Retryable source path is invalid")
        by_id[ident] = {key: entry.get(key) for key in ("intake_id", "relative_path", "sha256", "size_bytes")}
    db_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in db_records:
        ident = entry["intake_id"]
        db_by_id[ident] = {key: entry.get(key) for key in ("intake_id", "relative_path", "sha256", "size_bytes")}
    if set(by_id) != set(db_by_id):
        _fail("BACKUP_INTEGRITY_FAILED", "Manifest retryable source set differs from database closure")
    for ident, declared_row in by_id.items():
        if declared_row != db_by_id[ident]:
            _fail("BACKUP_INTEGRITY_FAILED", "Manifest retryable source identity differs from database")


def _inspect(connection: sqlite3.Connection) -> Dict[str, Any]:
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        _fail("BACKUP_INTEGRITY_FAILED", "Snapshot database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        _fail("BACKUP_INTEGRITY_FAILED", "Snapshot database references are inconsistent")
    meta = dict(connection.execute("SELECT key,value FROM meta"))
    if (meta.get("storage_version") not in {"1", STORAGE_VERSION}
            or meta.get("schema_version") != str(SCHEMA_VERSION)
            or meta.get("digest_version") != "workspace-task-v1"):
        _fail("UNSUPPORTED_BACKUP_VERSION", "Snapshot needs an explicit supported migration")
    S.require_id(meta.get("project_id"), "project_id")
    revision = meta.get("workspace_revision")
    if not isinstance(revision, str) or not revision.isdecimal():
        _fail("BACKUP_INTEGRITY_FAILED", "Snapshot revision metadata is invalid")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if meta["storage_version"] == STORAGE_VERSION and not {"artifact_references", "artifact_gc_pending"} <= tables:
        _fail("BACKUP_INTEGRITY_FAILED", "Version2 snapshot is missing artifact ownership tables")
    _quiescent(connection)
    artifacts = []
    for ident, relative, digest, size in connection.execute(
            "SELECT id,rel_path,sha256,size_bytes FROM artifacts ORDER BY id"):
        S.require_id(ident, "artifact_id")
        S.require_hash(digest)
        if relative != "objects/" + digest or type(size) is not int or size < 0:
            _fail("BACKUP_INTEGRITY_FAILED", "Snapshot artifact identity is invalid")
        artifacts.append({"id": ident, "relative_path": relative,
                          "sha256": digest, "size_bytes": str(size)})
    try:
        references = collect_artifact_references(connection)
    except WorkspaceError as exc:
        raise WorkspaceError("BACKUP_INTEGRITY_FAILED", "Snapshot artifact reference closure is invalid") from exc
    return {"metadata": {key: meta[key] for key in _META_KEYS}, "artifacts": artifacts,
            "artifact_references": references, "retryable_sources": _retryable_sources(connection)}


def _write_manifest(path: Path, value: Dict[str, Any]) -> None:
    data = S.canonical_json(value)
    if len(data) > MAX_MANIFEST_BYTES:
        _fail("BACKUP_LIMIT_EXCEEDED", "Snapshot manifest exceeds its supported size")
    temporary = path.parent / ("." + path.name + ".pending-" + uuid.uuid4().hex)
    # A complete manifest appears only after all its files have been synced.
    # Keep any interrupted temporary file as evidence; never erase user data.
    with temporary.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    _sync_directory(path.parent)
    temporary.unlink()


def backup_workspace(store: WorkspaceStore, target: Path) -> Dict[str, Any]:
    """Pause store writes and snapshot a quiescent opened workspace.

    Call outside any store transaction. SQLite.backup from a connection already
    in BEGIN IMMEDIATE can deadlock; holding the mutex preserves the snapshot
    while permitting the SQLite backup API to finish without a source write tx.
    """
    with store._mutex:
        connection = store.conn
        if store._depth or connection.in_transaction:
            raise Conflict("BACKUP_IN_TRANSACTION", "Backup must run outside a workspace transaction")
        _quiescent(connection)
        if getattr(store, "_queue_active_processors", set()):
            raise Conflict("BACKUP_ACTIVE_WORK", "A processor has not finished exiting")
        store._check_artifacts()
        # _check_artifacts has completed its own transaction before backup.
        if store._depth or connection.in_transaction:
            raise Conflict("BACKUP_IN_TRANSACTION", "Workspace validation left an active transaction")
        # Reject missing/corrupt staged inputs before creating a destination.
        retryable_sources = _retryable_sources(connection)
        for declaration in retryable_sources:
            _check_file(_source_path(store.root, declaration), declaration)
        output = _destination(target, store.root)
        database = output / SNAPSHOT_DB
        snapshot = sqlite3.connect(str(database), isolation_level=None)
        try:
            connection.backup(snapshot, pages=256, sleep=0.01)
            snapshot.execute("PRAGMA journal_mode=DELETE")
            description = _inspect(snapshot)
        finally:
            snapshot.close()
        # Windows _commit/fsync requires a writable descriptor, even though
        # SQLite already flushed the bytes and this call does not change them.
        with database.open("r+b") as incoming:
            os.fsync(incoming.fileno())
        copied = set()
        for artifact in description["artifacts"]:
            relative = artifact["relative_path"]
            if relative not in copied:
                _copy_verified(store.root / relative, output / relative, artifact)
                copied.add(relative)
        _sync_directory(output / "objects")
        # Retryable (failed) staged uploads are part of the closure: without
        # them restore can never re-run the intake. Verified against the DB.
        for declaration in description["retryable_sources"]:
            source_path = _source_path(store.root, declaration)
            destination = _source_path(output, declaration)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_verified(source_path, destination, declaration)
            _sync_directory(destination.parent)
        manifest = {"kind": "traceannotate-workspace-backup", "backup_version": 1,
                    "state": "complete", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "scope": "quiescent-sqlite-and-registered-objects",
                    "excluded": ["unregistered-files", "tmp-other", "exports", "logs", "external-source-media"],
                    "database": {"relative_path": SNAPSHOT_DB, **_fingerprint(database)}, **description}
        _write_manifest(output / BACKUP_MANIFEST, manifest)
        return manifest


def _load_backup(root: Path) -> Tuple[Dict[str, Any], str]:
    _safe(root)
    path = root / BACKUP_MANIFEST
    _regular(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        _fail("BACKUP_LIMIT_EXCEEDED", "Snapshot manifest is too large")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate manifest key")
            result[key] = value
        return result
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=unique,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise WorkspaceError("BACKUP_INCOMPLETE", "Snapshot manifest is invalid") from exc
    if (not isinstance(manifest, dict) or manifest.get("kind") != "traceannotate-workspace-backup"
            or type(manifest.get("backup_version")) is not int or manifest["backup_version"] != 1
            or manifest.get("state") != "complete"
            or manifest.get("scope") != "quiescent-sqlite-and-registered-objects"):
        _fail("UNSUPPORTED_BACKUP_VERSION", "Snapshot manifest version or scope is unsupported")
    database = manifest.get("database")
    if not isinstance(database, dict) or database.get("relative_path") != SNAPSHOT_DB:
        _fail("BACKUP_INCOMPLETE", "Snapshot database declaration is invalid")
    _check_file(root / SNAPSHOT_DB, database)
    for suffix in ("-wal", "-shm", "-journal"):
        if (root / (SNAPSHOT_DB + suffix)).exists():
            _fail("BACKUP_INCOMPLETE", "Snapshot must not depend on undeclared SQLite sidecars")
    uri = (root / SNAPSHOT_DB).as_uri() + "?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        try:
            description = _inspect(connection)
            legacy_references = not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_references'").fetchone()
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise WorkspaceError("BACKUP_INTEGRITY_FAILED", "Snapshot database cannot be read") from exc
    if description["metadata"] != manifest.get("metadata") or description["artifacts"] != manifest.get("artifacts"):
        _fail("BACKUP_INTEGRITY_FAILED", "Manifest differs from its database object closure")
    if ("artifact_references" not in manifest and legacy_references
            and description["metadata"]["storage_version"] == "1"):
        # Supported older snapshots had authoritative media/queue owners but
        # no explicit pin table or closure field. Infer, do not mutate them.
        manifest["artifact_references"] = description["artifact_references"]
    if manifest.get("artifact_references") != description["artifact_references"]:
        _fail("BACKUP_INTEGRITY_FAILED", "Manifest differs from its artifact reference closure")
    declarations = manifest.get("retryable_sources", [])
    _compare_retryable_closure(declarations, description["retryable_sources"])
    for source in declarations:
        _check_file(_source_path(root, source), source)
    for artifact in description["artifacts"]:
        _check_file(root / artifact["relative_path"], artifact)
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _check_file(path: Path, declaration: Dict[str, Any]) -> None:
    if _fingerprint(path) != {key: declaration.get(key) for key in ("sha256", "size_bytes")}:
        _fail("BACKUP_INTEGRITY_FAILED", "Snapshot file hash or size does not match")


def restore_workspace(backup: Path, target: Path) -> Dict[str, Any]:
    """Validate a complete supported backup, then restore to a nonexistent path.

    The original snapshot and project are never opened for writing. No queue
    wrapper is constructed, and no pending command or external process runs.
    An interrupted/failed destination is retained without a completion marker.
    """
    backup = Path(backup).absolute()
    manifest, manifest_sha256 = _load_backup(backup)
    output = _destination(target, backup)
    copied = set()
    for artifact in manifest["artifacts"]:
        relative = artifact["relative_path"]
        if relative not in copied:
            _copy_verified(backup / relative, output / relative, artifact)
            copied.add(relative)
    _sync_directory(output / "objects")
    # Objects are complete before the workspace database becomes available.
    _copy_verified(backup / SNAPSHOT_DB, output / DB_FILE, manifest["database"])
    # Restore retryable staged sources so failed intakes can be retried.
    for source in manifest.get("retryable_sources", []):
        destination = _source_path(output, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_verified(_source_path(backup, source), destination, source)
        _sync_directory(destination.parent)
    recovered = WorkspaceStore(output)
    try:
        recovered.open()
        with recovered._mutex:
            description = _inspect(recovered.conn)
        expected_metadata = dict(manifest["metadata"])
        if expected_metadata["storage_version"] == "1":
            expected_metadata["storage_version"] = STORAGE_VERSION
        if (description["metadata"] != expected_metadata or description["artifacts"] != manifest["artifacts"]
                or description["artifact_references"] != manifest["artifact_references"]):
            _fail("BACKUP_INTEGRITY_FAILED", "Opening the recovered project changed its snapshot identity")
        try:
            _compare_retryable_closure(description["retryable_sources"],
                                       manifest.get("retryable_sources", []))
        except WorkspaceError:
            _fail("BACKUP_INTEGRITY_FAILED", "Opening the recovered project changed its snapshot identity")
    finally:
        recovered.close()
    if _fingerprint(backup / BACKUP_MANIFEST)["sha256"] != manifest_sha256:
        _fail("BACKUP_INTEGRITY_FAILED", "Snapshot manifest changed during recovery")
    receipt = {"kind": "traceannotate-workspace-restore", "restore_version": 1, "state": "complete",
               "project_id": manifest["metadata"]["project_id"],
               "workspace_revision": manifest["metadata"]["workspace_revision"],
               "source_backup_manifest_sha256": manifest_sha256,
               "source_database_sha256": manifest["database"]["sha256"],
               "restored_database": _fingerprint(output / DB_FILE),
               "storage_migration": {"source_version": manifest["metadata"]["storage_version"],
                                     "restored_version": STORAGE_VERSION,
                                     "source_backup_modified": False},
               "jobs_restarted": False, "commands_replayed": False}
    _write_manifest(output / RESTORE_MANIFEST, receipt)
    return receipt
