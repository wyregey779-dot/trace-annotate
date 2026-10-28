"""Persistent workspace storage: SQLite + immutable object files.

Design contract (contracts/first-batch-contract.md section 6):
* workspace.sqlite3 is the only authoritative mutable state; workspace-v3 JSON
  is an exchange snapshot.
* Objects are immutable files under objects/<sha256>.
* Every successful user command writes a task_revision, a durable_commit and a
  command_receipt in one SQLite transaction.
* Second process cannot open the project for writing (OS file lock).
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps
import io
import stat
import tempfile
import threading
import uuid
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from ..temporal.media import stream_sha256
from typing import Any, Dict, List, Optional

from . import schema as S
from .errors import Conflict, NotFound, SchemaInvalid, WorkspaceError, revision_conflict

SCHEMA_VERSION = 3
STORAGE_VERSION = "1"
DB_FILE = "workspace.sqlite3"
LOCK_FILE = "write.lock"
OBJECTS_DIR = "objects"
TMP_DIR = "tmp"
EXPORTS_DIR = "exports"
LOGS_DIR = "logs"

DEFAULT_TEMPLATE = {
    "id": "generic-segment",
    "version": 1,
    "name": "General video segments",
    "annotation_types": ["whole_clip", "event", "segment"],
    "labels": [],
    "attributes": [],
    "coverage_policy": "sparse",
}

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS templates(id TEXT NOT NULL, version INTEGER NOT NULL,
    body TEXT NOT NULL, PRIMARY KEY(id, version));
CREATE TABLE IF NOT EXISTS media(id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, media_id TEXT NOT NULL,
    template_id TEXT NOT NULL, template_version INTEGER NOT NULL,
    revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS task_revisions(id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
    revision INTEGER NOT NULL, body TEXT NOT NULL, content_sha256 TEXT NOT NULL,
    commit_id TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS durable_commits(id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS command_receipts(command_id TEXT PRIMARY KEY, task_id TEXT,
    request_sha256 TEXT NOT NULL, state TEXT NOT NULL, response TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, rel_path TEXT NOT NULL,
    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS intakes(id TEXT PRIMARY KEY, group_id TEXT NOT NULL,
    state TEXT NOT NULL, display_name TEXT, source_sha256 TEXT, source_size INTEGER,
    media_id TEXT, task_id TEXT, error_code TEXT, created_at TEXT NOT NULL, job_id TEXT, template_id TEXT, template_version INTEGER);
CREATE UNIQUE INDEX IF NOT EXISTS idx_revision ON task_revisions(task_id, revision);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex}"


def _safe(path):
    for entry in (Path(path).absolute(), *Path(path).absolute().parents):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise WorkspaceError("UNSAFE_PATH", "Workspace paths cannot contain links or reparse points")


def _sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _serialized(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with self.transaction():
            return function(self, *args, **kwargs)
    return invoke


@dataclass
class ProjectLock:
    """OS single-writer lock held for the lifetime of an open store."""
    path: Path
    _handle: Optional[int] = None

    def acquire(self):
        if self._handle is not None:
            raise Conflict("PROJECT_LOCKED", "Workspace is already open")
        _safe(self.path)
        handle = os.open(str(self.path), os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(handle)
            raise Conflict("PROJECT_LOCKED", "Another process holds the workspace lock") from exc
        self._handle = handle

    def release(self):
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self._handle, 0, os.SEEK_SET)
                msvcrt.locking(self._handle, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._handle, fcntl.LOCK_UN)
        finally:
            os.close(self._handle)
            self._handle = None


class WorkspaceStore:
    """Owns the on-disk project directory and the SQLite connection."""

    def __init__(self, root: Path):
        self.root = Path(root).absolute()
        if str(self.root).startswith(("\\\\", "//")):
            raise WorkspaceError("UNSUPPORTED_STORAGE", "Network workspaces are unsupported")
        self._mutex = threading.RLock()
        self._conn = None
        self._depth = 0
        self.lock_path = self.root / LOCK_FILE
        self.objects_dir = self.root / OBJECTS_DIR
        self.tmp_dir = self.root / TMP_DIR
        self.exports_dir = self.root / EXPORTS_DIR
        self.logs_dir = self.root / LOGS_DIR
        self.lock = ProjectLock(self.lock_path)

    # -- lifecycle ---------------------------------------------------------
    def create(self):
        _safe(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock.acquire()
        try:
            if (self.root / DB_FILE).exists():
                raise Conflict("PROJECT_EXISTS", "Workspace already exists")
            for directory in (self.objects_dir, self.tmp_dir, self.exports_dir, self.logs_dir):
                _safe(directory)
                directory.mkdir(exist_ok=True)
            self._init_db(new=True)
            self._upsert_template(DEFAULT_TEMPLATE)
        except BaseException:
            self.close()
            raise

    def open(self):
        _safe(self.root / DB_FILE)
        if not (self.root / DB_FILE).is_file():
            raise NotFound("PROJECT_NOT_FOUND", "Workspace database is missing")
        self.lock.acquire()
        try:
            for directory in (self.objects_dir, self.tmp_dir, self.exports_dir, self.logs_dir):
                _safe(directory)
                if not directory.is_dir():
                    raise WorkspaceError("RECOVERY_REQUIRED", "Workspace directory is missing")
            self._init_db(new=False)
            self._check_artifacts()
            with self.transaction():
                self.conn.execute("UPDATE intakes SET state='failed',error_code='INTERRUPTED' "
                                  "WHERE state='running' AND task_id IS NULL")
        except BaseException:
            self.close()
            raise

    def close(self):
        with self._mutex:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self.lock.release()

    @property
    def conn(self):
        if self.lock._handle is None:
            raise WorkspaceError("PROJECT_CLOSED", "Open the workspace before accessing its database")
        if self._conn is None:
            _safe(self.root / DB_FILE)
            self._conn = sqlite3.connect(str(self.root / DB_FILE), isolation_level=None,
                                         timeout=30, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    @contextmanager
    def transaction(self):
        with self._mutex:
            connection = self.conn
            depth = self._depth
            marker = f"nested_{depth}"
            connection.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT {marker}")
            self._depth += 1
            try:
                yield connection
                connection.execute("COMMIT" if depth == 0 else f"RELEASE SAVEPOINT {marker}")
            except BaseException:
                if connection.in_transaction:
                    if depth == 0:
                        connection.execute("ROLLBACK")
                    else:
                        connection.execute(f"ROLLBACK TO SAVEPOINT {marker}")
                        connection.execute(f"RELEASE SAVEPOINT {marker}")
                raise
            finally:
                self._depth = depth

    def _init_db(self, *, new):
        if not new:
            try:
                values = dict(self.conn.execute("SELECT key,value FROM meta"))
            except sqlite3.DatabaseError as exc:
                raise WorkspaceError("RECOVERY_REQUIRED", "Workspace database cannot be read") from exc
            if values.get("storage_version") != STORAGE_VERSION:
                raise WorkspaceError("WORKSPACE_MIGRATION_REQUIRED", "Experimental workspace requires explicit migration; original data preserved")
            if values.get("schema_version") != str(SCHEMA_VERSION):
                raise WorkspaceError("UNSUPPORTED_SCHEMA", "Workspace schema is unsupported")
            if (values.get("digest_version") != "workspace-task-v1"
                    or not isinstance(values.get("project_id"), str)
                    or S.ID_RE.fullmatch(values["project_id"]) is None):
                raise WorkspaceError("RECOVERY_REQUIRED", "Workspace identity or digest metadata is missing")
        with self.transaction():
            for statement in DDL.split(";"):
                if statement.strip():
                    self.conn.execute(statement)
            if new:
                self.conn.executemany("INSERT INTO meta VALUES(?,?)", [
                    ("schema_version", str(SCHEMA_VERSION)), ("storage_version", STORAGE_VERSION),
                    ("project_id", _new_id("proj")), ("digest_version", "workspace-task-v1"),
                    ("workspace_revision", "0")])
            self.conn.execute("INSERT OR IGNORE INTO meta VALUES('workspace_revision','0')")
            self._install_revision_triggers(("tasks", "intakes", "artifacts", "templates", "durable_commits"))
        if self.conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise WorkspaceError("RECOVERY_REQUIRED", "Database integrity check failed")

    def _install_revision_triggers(self, tables):
        allowed = {"tasks", "intakes", "artifacts", "templates", "durable_commits",
                   "batches", "jobs", "attempts", "batch_jobs"}
        for table in tables:
            if table not in allowed:
                raise ValueError("Unexpected revision table")
            for operation in ("INSERT", "UPDATE", "DELETE"):
                self.conn.execute(f"CREATE TRIGGER IF NOT EXISTS revision_{table}_{operation} "
                                  f"AFTER {operation} ON {table} BEGIN UPDATE meta "
                                  "SET value=CAST(value AS INTEGER)+1 WHERE key='workspace_revision'; END")

    @_serialized
    def _check_artifacts(self):
        for rel, digest, size in self.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts"):
            if rel != f"objects/{digest}":
                raise WorkspaceError("RECOVERY_REQUIRED", "Artifact path is invalid")
            path = self.root / rel
            _safe(path)
            if not path.is_file() or path.stat().st_size != size or stream_sha256(path) != digest:
                raise WorkspaceError("RECOVERY_REQUIRED", "Artifact missing or corrupt; writes refused",
                                     details={"artifact_sha256": digest})
        for (body,) in self.conn.execute("SELECT body FROM media"):
            media = json.loads(body)
            frame_id = media.get("frame_map_artifact")
            if not frame_id or not self.conn.execute("SELECT 1 FROM artifacts WHERE id=?", (frame_id,)).fetchone():
                raise WorkspaceError("RECOVERY_REQUIRED", "Media frame-map reference is missing")
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (f"media_objects:{media['id']}",)).fetchone()
            if row is None:
                raise WorkspaceError("RECOVERY_REQUIRED", "Media object bindings are missing")
            bindings = json.loads(row[0])
            for key in ("playback_path", "frame_map_path", "manifest_path", "preparation_log_path"):
                reference = bindings.get(key, {})
                registered = self.conn.execute("SELECT sha256,size_bytes FROM artifacts WHERE id=?", (reference.get("id"),)).fetchone()
                if registered is None or (registered[0], str(registered[1])) != (reference.get("sha256"), reference.get("size_bytes")):
                    raise WorkspaceError("RECOVERY_REQUIRED", "Media object binding is missing or inconsistent")
            if (bindings["frame_map_path"]["id"] != frame_id
                    or bindings["playback_path"]["sha256"] != media["playback"]["sha256"]
                    or bindings["playback_path"]["size_bytes"] != media["playback"]["size_bytes"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Media identity does not match its objects")
        for task_id, revision in self.conn.execute("SELECT id,revision FROM tasks"):
            task = self.get_task(task_id)
            if task["revision"] != revision or self.task_digest(task) != task["content_sha256"]:
                raise WorkspaceError("RECOVERY_REQUIRED", "Current task revision or digest is inconsistent")

    @_serialized
    def _upsert_template(self, body):
        S.require_id(body.get("id"))
        if type(body.get("version")) is not int or body["version"] < 1:
            raise SchemaInvalid("SCHEMA_INVALID", "Template version must be a positive integer")
        encoded = S.canonical_json(body).decode("utf-8")
        previous = self.conn.execute("SELECT body FROM templates WHERE id=? AND version=?", (body["id"], body["version"])).fetchone()
        if previous:
            if S.canonical_json(json.loads(previous[0])).decode("utf-8") != encoded:
                raise Conflict("TEMPLATE_IMMUTABLE", "Publish a new template version")
            return
        self.conn.execute("INSERT INTO templates VALUES(?,?,?)", (body["id"], body["version"], encoded))

    def _publish_stream(self, stream):
        _safe(self.tmp_dir)
        _safe(self.objects_dir)
        handle, name = tempfile.mkstemp(dir=self.tmp_dir, suffix=".part")
        temporary = Path(name)
        try:
            digest, size = hashlib.sha256(), 0
            with os.fdopen(handle, "wb") as output:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha = digest.hexdigest()
            target = self.objects_dir / sha
            _safe(target)
            try:
                os.link(temporary, target)
                _sync_directory(self.objects_dir)
            except FileExistsError:
                if target.stat().st_size != size or stream_sha256(target) != sha:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Existing immutable object is corrupt")
            ident = S.make_id("art", sha)
            with self.transaction():
                self.conn.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)", (ident, f"objects/{sha}", sha, size))
            return {"id": ident, "relative_path": f"objects/{sha}", "sha256": sha, "size_bytes": str(size)}
        finally:
            temporary.unlink(missing_ok=True)

    def _put_artifact(self, data, rel_prefix="objects"):
        if not isinstance(data, bytes) or rel_prefix != OBJECTS_DIR:
            raise SchemaInvalid("SCHEMA_INVALID", "Artifact must be bytes in objects")
        return self._publish_stream(io.BytesIO(data))

    def _put_file(self, path):
        with Path(path).open("rb") as source:
            return self._publish_stream(source)

    # -- templates ----------------------------------------------------------
    @_serialized
    def list_templates(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT body FROM templates").fetchall()
        return [json.loads(r[0]) for r in rows]

    # -- intake / admission -------------------------------------------------
    def import_source(self, source, *, display_name=None, template_id="generic-segment", template_version=1):
        from .media import prepare_workspace_media
        source = Path(source)
        display_name = source.name if display_name is None else display_name
        if not isinstance(display_name, str) or not display_name:
            raise SchemaInvalid("SCHEMA_INVALID", "Display name must be a nonempty string")
        S.canonical_json(display_name)
        S.require_id(template_id)
        if type(template_version) is not int or template_version < 1:
            raise SchemaInvalid("SCHEMA_INVALID", "Template version must be a positive integer")
        intake_id, group_id, job_id = _new_id("int"), _new_id("grp"), _new_id("job")
        with self.transaction():
            if not self.conn.execute("SELECT 1 FROM templates WHERE id=? AND version=?", (template_id, template_version)).fetchone():
                raise NotFound("TEMPLATE_NOT_FOUND", "Requested template does not exist")
            self.conn.execute("INSERT INTO intakes(id,group_id,state,display_name,created_at,job_id,template_id,template_version) VALUES(?,?,?,?,?,?,?,?)",
                              (intake_id, group_id, "running", display_name, _now(), job_id, template_id, template_version))
        try:
            if not source.is_file():
                raise NotFound("SOURCE_MISSING", "Source file does not exist")
            digest = stream_sha256(source)
            size = source.stat().st_size
            with self.transaction():
                self.conn.execute("UPDATE intakes SET source_sha256=?,source_size=? WHERE id=?", (digest, size, intake_id))
                candidates = [json.loads(row[0]) for row in self.conn.execute("SELECT body FROM media")]
                media = next((value for value in candidates if value["source"]["sha256"] == digest), None)
            reused = media is not None
            objects = None
            if reused:
                self._check_artifacts()
            if media is None:
                prepared = prepare_workspace_media(source, self.tmp_dir / job_id)
                if prepared["source_sha256"] != digest or prepared["source_size"] != size:
                    raise Conflict("MEDIA_CHANGED", "Source changed during preparation")
                objects = {key: self._put_file(prepared[key]) for key in
                           ("playback_path", "frame_map_path", "manifest_path", "preparation_log_path")}
                media = dict(prepared["media_fields"])
                media["source"]["display_name"] = display_name
                media["id"] = S.make_id("med", digest)
                media["frame_map_artifact"] = objects["frame_map_path"]["id"]
                if objects["playback_path"]["sha256"] != media["playback"]["sha256"]:
                    raise Conflict("MEDIA_CHANGED", "Playback changed during admission")
            if stream_sha256(source) != digest or source.stat().st_size != size:
                raise Conflict("MEDIA_CHANGED", "Source changed before admission commit")
            task_id = _new_id("tsk")
            task = {"id": task_id, "media_id": media["id"], "template_id": template_id,
                    "template_version": template_version, "revision": 0, "annotations": [], "extensions": {}}
            with self.transaction():
                self.conn.execute("INSERT OR IGNORE INTO media VALUES(?,?)", (media["id"], S.canonical_json(media).decode("utf-8")))
                if objects:
                    self.conn.execute("INSERT OR IGNORE INTO meta VALUES(?,?)", (f"media_objects:{media['id']}", S.canonical_json(objects).decode("utf-8")))
                task["content_sha256"] = self.task_digest(task)
                self.conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?)", (task_id, media["id"], template_id, template_version, 0))
                self.conn.execute("INSERT INTO task_revisions VALUES(?,?,?,?,?,?,?)", (_new_id("rev"), task_id, 0,
                                  S.canonical_json(task).decode("utf-8"), task["content_sha256"], None, _now()))
                self.conn.execute("UPDATE intakes SET state='succeeded',media_id=?,task_id=? WHERE id=?", (media["id"], task_id, intake_id))
            return {"intake_id": intake_id, "media_id": media["id"], "task_id": task_id,
                    "duration_seconds": float(S.rational_value(media["duration"])), "frames": media["frame_count"], "reused_existing": reused}
        except BaseException as exc:
            with self.transaction():
                self.conn.execute("UPDATE intakes SET state='failed',error_code=? WHERE id=?", (getattr(exc, "code", "IMPORT_FAILED"), intake_id))
            raise

    @_serialized
    def list_intakes(self):
        cursor = self.conn.execute("SELECT * FROM intakes ORDER BY rowid")
        names = [value[0] for value in cursor.description]
        return [dict(zip(names, row)) for row in cursor]

    @_serialized
    def task_digest(self, task):
        media = self.get_media(task["media_id"])
        for key in ("source", "playback"):
            media.get(key, {}).pop("display_name", None)
        template = self.conn.execute("SELECT body FROM templates WHERE id=? AND version=?", (task["template_id"], task["template_version"])).fetchone()
        if template is None:
            raise NotFound("TEMPLATE_NOT_FOUND", "Task template does not exist")
        artifacts = []
        if media.get("frame_map_artifact"):
            row = self.conn.execute("SELECT id,sha256,size_bytes FROM artifacts WHERE id=?", (media["frame_map_artifact"],)).fetchone()
            if row is None:
                raise WorkspaceError("RECOVERY_REQUIRED", "Task frame map is missing")
            artifacts.append({"id": row[0], "sha256": row[1], "size_bytes": str(row[2])})
        return S.sha256_text({"digest_version": "workspace-task-v1", "task": S.task_subject(task),
                              "media": media, "template": json.loads(template[0]), "artifacts": artifacts})

    # -- reads ---------------------------------------------------------------
    @_serialized
    def get_task(self, task_id: str) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT r.body FROM tasks t JOIN task_revisions r ON r.task_id=t.id AND r.revision=t.revision WHERE t.id=?",
            (task_id,)).fetchone()
        if row is None:
            raise NotFound("TASK_NOT_FOUND", f"No such task {task_id}")
        return json.loads(row[0])

    @_serialized
    def list_tasks(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT id, media_id, revision FROM tasks").fetchall()
        return [{"task_id": t, "media_id": m, "revision": r} for (t, m, r) in rows]

    @_serialized
    def get_media(self, media_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT body FROM media WHERE id=?", (media_id,)).fetchone()
        if row is None:
            raise NotFound("MEDIA_NOT_FOUND", f"No such media {media_id}")
        return json.loads(row[0])

    @_serialized
    def get_snapshot(self) -> Dict[str, Any]:
        media = [json.loads(r[0]) for r in self.conn.execute("SELECT body FROM media")]
        templates = [json.loads(r[0]) for r in self.conn.execute("SELECT body FROM templates")]
        tasks = []
        for (tid,) in self.conn.execute("SELECT id FROM tasks ORDER BY rowid"):
            tasks.append(self.get_task(tid))
        revs = int(self.conn.execute("SELECT value FROM meta WHERE key='workspace_revision'").fetchone()[0])
        from .queue import read_queue_snapshot
        queue_snapshot = read_queue_snapshot(self)
        return {
            "kind": "traceannotate-workspace",
            "schema_version": SCHEMA_VERSION,
            "project_id": self._project_id(),
            "workspace_revision": revs,
            "media": media,
            "templates": templates,
            "tasks": tasks,
            "evidence": [],
            "reviews": [],
            "batches": queue_snapshot["batches"],
            "commits": [json.loads(r[0]) for r in self.conn.execute("SELECT body FROM durable_commits ORDER BY rowid")],
            "artifacts": [{"id": r[0], "relative_path": r[1], "sha256": r[2], "size_bytes": str(r[3])}
                          for r in self.conn.execute(
                              "SELECT id,rel_path,sha256,size_bytes FROM artifacts")],
        }

    @_serialized
    def _project_id(self) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key='project_id'").fetchone()
        if row:
            return row[0]
        pid = S.make_id("proj", self.root.resolve().__str__() + "v3")
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('project_id',?)", (pid,))
        return pid

    @_serialized
    def get_command_receipt(self, command_id: str) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT task_id,state,response,request_sha256 FROM command_receipts WHERE command_id=?",
            (command_id,)).fetchone()
        if row is None:
            raise NotFound("COMMAND_UNKNOWN", f"No command {command_id}")
        return {"command_id": command_id, "task_id": row[0], "state": row[1],
                "response": json.loads(row[2]) if row[2] else None,
                "request_sha256": row[3]}

    # -- writes ---------------------------------------------------------------
    @_serialized
    def apply_task_command(self, *, command_id: str, task_id: str, expected_revision: int,
                           expected_sha256: str, operation: str, payload: Dict[str, Any],
                           request_body: bytes) -> Dict[str, Any]:
        S.require_id(command_id)
        S.require_id(task_id)
        S.require_hash(expected_sha256)
        if type(expected_revision) is not int or not 0 <= expected_revision <= S.MAX_SAFE_INTEGER:
            raise SchemaInvalid("SCHEMA_INVALID", "Expected revision must be a nonnegative safe integer")
        if not isinstance(payload, dict) or not isinstance(request_body, bytes):
            raise SchemaInvalid("SCHEMA_INVALID", "Invalid command payload")
        request_sha = S.sha256_text({"task_id": task_id, "expected_revision": expected_revision,
                                    "expected_sha256": expected_sha256, "operation": operation,
                                    "payload": payload, "body_sha256": S.sha256_bytes(request_body)})
        existing = self.conn.execute(
            "SELECT state,response,request_sha256 FROM command_receipts WHERE command_id=?",
            (command_id,)).fetchone()
        if existing is not None:
            if existing[2] != request_sha:
                raise Conflict("IDEMPOTENCY_CONFLICT", "Same command_id with different request body")
            resp = json.loads(existing[1]) if existing[1] else None
            return {"applied": existing[0] == "committed", "idempotent_replay": True,
                    "response": resp}
        current = self.get_task(task_id)
        current_rev = current["revision"]
        if current_rev != expected_revision:
            raise revision_conflict(expected_revision, current_rev)
        if current["content_sha256"] != expected_sha256 or self.task_digest(current) != expected_sha256:
            raise Conflict("MEDIA_CHANGED", "Task content hash mismatch",
                           details={"expected_sha256": expected_sha256,
                                    "current_sha256": current["content_sha256"]})
        annotations = [dict(a) for a in current["annotations"]]
        annotations = self._apply_operation(annotations, operation, payload)
        new_task = dict(current)
        new_task["annotations"] = annotations
        new_task["revision"] = current_rev + 1
        template = next(value for value in self.list_templates() if value["id"] == current["template_id"] and value["version"] == current["template_version"])
        for annotation in new_task["annotations"]:
            self._validate_annotation(annotation, self.get_media(current["media_id"]), template)
        new_task["content_sha256"] = self.task_digest(new_task)
        new_rev_id = S.make_id("rev", task_id + str(new_task["revision"]))
        commit_id = S.make_id("cmt", command_id)
        response = {
            "command_id": command_id, "applied": True, "task_id": task_id,
            "previous_revision": current_rev, "revision": new_task["revision"],
            "content_sha256": new_task["content_sha256"], "commit_id": commit_id,
        }
        ts = _now()
        with self.transaction():
            self.conn.execute("UPDATE tasks SET revision=? WHERE id=?",
                              (new_task["revision"], task_id))
            self.conn.execute(
                "INSERT INTO task_revisions(id,task_id,revision,body,content_sha256,commit_id,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (new_rev_id, task_id, new_task["revision"],
                 json.dumps(new_task, ensure_ascii=False), new_task["content_sha256"], commit_id, ts))
            self.conn.execute(
                "INSERT INTO durable_commits(id,body) VALUES(?,?)",
                (commit_id, json.dumps({"id": commit_id, "task_id": task_id,
                                        "revision": new_task["revision"]}, ensure_ascii=False)))
            self.conn.execute(
                "INSERT INTO command_receipts(command_id,task_id,request_sha256,state,response,created_at)"
                " VALUES(?,?,?,?,?,?)",
                (command_id, task_id, request_sha, "committed",
                 json.dumps(response, ensure_ascii=False), ts))
        return {"applied": True, "idempotent_replay": False, "response": response}

    def _apply_operation(self, annotations: List[Dict[str, Any]], operation: str,
                         payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        required = {"add_annotation": {"annotation"}, "update_annotation": {"annotation_id", "changes"}, "delete_annotation": {"annotation_id"}}
        if operation not in required:
            raise SchemaInvalid("UNSUPPORTED_OPERATION", "Unsupported annotation operation")
        if set(payload) != required[operation]:
            raise SchemaInvalid("SCHEMA_INVALID", "Operation payload has missing or unknown fields")
        if operation == "add_annotation":
            if not isinstance(payload["annotation"], dict):
                raise SchemaInvalid("SCHEMA_INVALID", "Annotation must be an object")
            ann = dict(payload["annotation"])
            ann["id"] = S.require_id(ann.get("id"), "annotation.id")
            if any(a["id"] == ann["id"] for a in annotations):
                raise Conflict("ID_EXISTS", f"Annotation {ann['id']} already exists")
            annotations.append(ann)
        elif operation == "update_annotation":
            aid = payload["annotation_id"]
            changes = payload["changes"]
            if not isinstance(changes, dict):
                raise SchemaInvalid("SCHEMA_INVALID", "Changes must be an object")
            found = False
            for a in annotations:
                if a["id"] == aid:
                    found = True
                    for k, v in changes.items():
                        if k in ("id", "type"):
                            raise SchemaInvalid("INVALID_CHANGE", f"Cannot change {k}")
                        a[k] = v
            if not found:
                raise NotFound("ANNOTATION_NOT_FOUND", f"No annotation {aid}")
        elif operation == "delete_annotation":
            aid = payload["annotation_id"]
            before = len(annotations)
            annotations = [a for a in annotations if a["id"] != aid]
            if len(annotations) == before:
                raise NotFound("ANNOTATION_NOT_FOUND", f"No annotation {aid}")
        else:
            raise SchemaInvalid("UNSUPPORTED_OPERATION", f"Unknown operation {operation}")
        return annotations

    def _validate_annotation(self, ann, media, template):
        allowed = {"id", "type", "label", "text", "tags", "track", "attributes", "evidence_ids", "provenance", "start", "end", "at", "legacy_id"}
        if set(ann) - allowed:
            raise SchemaInvalid("SCHEMA_INVALID", "Annotation contains unknown fields")
        S.require_id(ann.get("id"))
        kind = ann.get("type")
        if kind not in template["annotation_types"] or kind not in ("whole_clip", "event", "segment"):
            raise SchemaInvalid("SCHEMA_INVALID", "Unsupported annotation type")
        if not isinstance(ann.get("label"), str) or not ann["label"]:
            raise SchemaInvalid("SCHEMA_INVALID", "Annotation requires a nonempty label")
        if template["labels"] and ann["label"] not in template["labels"]:
            raise SchemaInvalid("SCHEMA_INVALID", "Label is not in this template")
        ann.setdefault("text", "")
        ann.setdefault("tags", [])
        ann.setdefault("track", "main")
        ann.setdefault("attributes", {})
        ann.setdefault("evidence_ids", [])
        ann.setdefault("provenance", {"origin": "manual", "actor_id": "user", "job_id": None})
        if not isinstance(ann["text"], str) or not isinstance(ann["track"], str) or not ann["track"]:
            raise SchemaInvalid("SCHEMA_INVALID", "Invalid text or track")
        if not isinstance(ann["tags"], list) or any(not isinstance(tag, str) for tag in ann["tags"]):
            raise SchemaInvalid("SCHEMA_INVALID", "Tags must be strings")
        if ann["evidence_ids"] != [] or ann["attributes"] != {}:
            raise SchemaInvalid("UNSUPPORTED_OPERATION", "Evidence and custom attributes need their own admission workflow")
        provenance = ann["provenance"]
        if (not isinstance(provenance, dict) or set(provenance) != {"origin", "actor_id", "job_id"}
                or provenance["origin"] != "manual" or provenance["job_id"] is not None):
            raise SchemaInvalid("SCHEMA_INVALID", "Manual commands require manual provenance")
        S.require_id(provenance.get("actor_id"))
        times = {"whole_clip": set(), "event": {"at"}, "segment": {"start", "end"}}[kind]
        if {"start", "end", "at"}.intersection(ann) != times:
            raise SchemaInvalid("SCHEMA_INVALID", "Invalid time fields for this annotation type")
        values = {}
        duration = S.rational_value(media["duration"])
        for key in times:
            value = ann[key]
            if not isinstance(value, dict) or set(value) != {"seconds"}:
                raise SchemaInvalid("UNSUPPORTED_OPERATION", "Frame anchors require mapped-frame validation, not yet exposed here")
            if not isinstance(value["seconds"], dict):
                raise SchemaInvalid("INVALID_RATIONAL", "Time must be an n/d object with decimal strings")
            reduced = S.rational(value["seconds"])
            if reduced != value["seconds"]:
                raise SchemaInvalid("INVALID_RATIONAL", "Time must be reduced")
            values[key] = S.rational_value(reduced)
            if values[key] < 0 or values[key] > duration or (key != "end" and values[key] == duration):
                raise SchemaInvalid("SCHEMA_INVALID", "Annotation time is outside media duration")
        if kind == "segment" and values["start"] >= values["end"]:
            raise SchemaInvalid("SCHEMA_INVALID", "Segment start must precede end")

    def export_snapshot(self, export_name=None):
        name = _new_id("exp") if export_name is None else S.require_id(export_name, "export_name")
        _safe(self.exports_dir)
        _safe(self.tmp_dir)
        target = self.exports_dir / f"{name}.json"
        _safe(target)
        data = S.canonical_json(self.get_snapshot())
        handle, temporary = tempfile.mkstemp(dir=self.tmp_dir, suffix=".export.part")
        temporary = Path(temporary)
        try:
            with os.fdopen(handle, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise Conflict("EXPORT_EXISTS", "Export already exists; choose a new name") from exc
            _sync_directory(self.exports_dir)
            return target
        finally:
            temporary.unlink(missing_ok=True)
