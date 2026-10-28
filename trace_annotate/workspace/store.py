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
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Dict, List, Optional

from . import schema as S
from .errors import Conflict, NotFound, SchemaInvalid, WorkspaceError, revision_conflict

SCHEMA_VERSION = 3
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
    media_id TEXT, task_id TEXT, error_code TEXT, created_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS idx_revision ON task_revisions(task_id, revision);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _tool(name: str) -> str:
    """Resolve ffmpeg/ffprobe by PATH, then the known winget install location."""
    from shutil import which
    found = which(name)
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) /
        "Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "ffmpeg-9.0.1-full_build/bin" / f"{name}.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise WorkspaceError("DEPENDENCY_UNAVAILABLE", f"{name} not found on PATH")


def _probe_media(path: Path) -> Dict[str, Any]:
    """Use ffprobe for duration and video stream facts. Never fakes values."""
    cmd = [
        _tool("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceError("DEPENDENCY_UNAVAILABLE", f"ffprobe unavailable: {exc}") from exc
    data = json.loads(out.stdout or "{}")
    vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise WorkspaceError("UNSUPPORTED_MEDIA", "No video stream found", details={"path": path.name})
    duration = float(data.get("format", {}).get("duration") or vstream.get("duration") or 0)
    if duration <= 0:
        raise WorkspaceError("UNSUPPORTED_MEDIA", "Media duration is zero or unreadable")
    rotation = 0
    side = vstream.get("side_data_list") or []
    if side:
        rotation = int(side[0].get("rotation", 0) or 0) % 360
    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    if rotation in (90, 270):
        width, height = height, width
    fps_num, fps_den = 0, 1
    r_frame = vstream.get("r_frame_rate", "0/1")
    if "/" in r_frame:
        a, b = r_frame.split("/", 1)
        fps_num, fps_den = int(a), int(b)
    nb_frames = vstream.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else max(1, int(duration * fps_num / max(fps_den, 1)))
    return {
        "duration": S.rational(duration),
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "rotation": rotation,
        "fps_num": fps_num,
        "fps_den": max(fps_den, 1),
        "frame_count": str(frame_count),
        "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


def _stream_sha256(path: Path) -> Dict[str, Any]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "size_bytes": size}


@dataclass
class ProjectLock:
    """Windows single-writer lock held for the process lifetime."""
    path: Path
    _handle: Optional[int] = None

    def acquire(self) -> None:
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = os.open(str(self.path), os.O_CREAT | os.O_RDWR | os.O_BINARY)
        try:
            msvcrt.locking(self._handle, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            os.close(self._handle)
            self._handle = None
            raise WorkspaceError("PROJECT_LOCKED",
                                 "Another process has this project open for writing") from exc

    def release(self) -> None:
        if self._handle is not None:
            import msvcrt
            try:
                msvcrt.locking(self._handle, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            os.close(self._handle)
            self._handle = None


class WorkspaceStore:
    """Owns the on-disk project directory and the SQLite connection."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock_path = self.root / LOCK_FILE
        self.objects_dir = self.root / OBJECTS_DIR
        self.tmp_dir = self.root / TMP_DIR
        self.exports_dir = self.root / EXPORTS_DIR
        self.logs_dir = self.root / LOGS_DIR
        self.lock = ProjectLock(self.lock_path)

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> None:
        if (self.root / DB_FILE).exists():
            raise WorkspaceError("PROJECT_EXISTS", "Workspace already exists; open it instead")
        self.root.mkdir(parents=True, exist_ok=True)
        for d in (self.objects_dir, self.tmp_dir, self.exports_dir, self.logs_dir):
            d.mkdir(exist_ok=True)
        self.lock.acquire()
        self._init_db()
        self._upsert_template(DEFAULT_TEMPLATE)

    def open(self) -> None:
        if not (self.root / DB_FILE).exists():
            raise NotFound("PROJECT_NOT_FOUND", f"No workspace at {self.root}")
        self.lock.acquire()
        self._init_db()
        # startup integrity: verify referenced objects exist (basic recovery)
        self._check_artifacts()

    def close(self) -> None:
        if hasattr(self, "_conn") and self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
        self.lock.release()

    # -- internals ----------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self, "_conn") or self._conn is None:
            self._conn = sqlite3.connect(str(self.root / DB_FILE), isolation_level=None, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        self.conn.executescript(DDL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),)
        )

    def _check_artifacts(self) -> None:
        rows = self.conn.execute("SELECT rel_path, sha256 FROM artifacts").fetchall()
        missing = []
        for rel, digest in rows:
            p = self.root / rel
            if not p.exists():
                missing.append(rel)
        if missing:
            raise WorkspaceError("RECOVERY_REQUIRED",
                                 f"{len(missing)} artifact(s) missing; project opened read-only",
                                 details={"missing": missing[:10]})

    def _upsert_template(self, body: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO templates(id,version,body) VALUES(?,?,?)",
            (body["id"], body["version"], json.dumps(body, ensure_ascii=False)),
        )

    def _put_artifact(self, data: bytes, rel_prefix: str = "objects") -> Dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        rel = f"{OBJECTS_DIR}/{digest}"
        path = self.root / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.tmp_dir / f"{digest}.part"
            tmp.write_bytes(data)
            os.replace(tmp, path)
        ident = S.make_id("art", digest)
        self.conn.execute(
            "INSERT OR REPLACE INTO artifacts(id,rel_path,sha256,size_bytes) VALUES(?,?,?,?)",
            (ident, rel, digest, len(data)),
        )
        return {"id": ident, "relative_path": rel, "sha256": digest, "size_bytes": str(len(data))}

    # -- templates ----------------------------------------------------------
    def list_templates(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT body FROM templates").fetchall()
        return [json.loads(r[0]) for r in rows]

    # -- intake / admission -------------------------------------------------
    def import_source(self, source: Path, *, display_name: Optional[str] = None,
                      template_id: str = "generic-segment", template_version: int = 1) -> Dict[str, Any]:
        source = Path(source)
        if not source.is_file():
            raise NotFound("SOURCE_MISSING", f"Source not found: {source}")
        display_name = display_name or source.name
        group_id = S.make_id("grp", str(source) + str(_now()))
        intake_id = S.make_id("int", str(source) + str(_now()))
        job_id = S.make_id("job", intake_id + "prepare")
        tpl = self.conn.execute(
            "SELECT body FROM templates WHERE id=? AND version=?", (template_id, template_version)
        ).fetchone()
        if tpl is None:
            raise NotFound("TEMPLATE_NOT_FOUND", f"Template {template_id} v{template_version} not found")
        identity = _stream_sha256(source)
        probe = _probe_media(source)
        media_id = S.make_id("med", identity["sha256"])
        existing = self.conn.execute("SELECT id FROM media WHERE id=?", (media_id,)).fetchone()
        if existing is not None:
            # Same bytes already admitted: record the intake but reuse media/task.
            row = self.conn.execute(
                "SELECT id FROM tasks WHERE media_id=? ORDER BY rowid LIMIT 1",
                (media_id,)).fetchone()
            ts = _now()
            with self.conn:
                self.conn.execute("INSERT INTO intakes(id,group_id,state,display_name,source_sha256,"
                                 "source_size,media_id,task_id,error_code,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                 (intake_id, group_id, "succeeded", display_name, identity["sha256"],
                                  identity["size_bytes"], media_id, row[0], None, ts))
            return {"intake_id": intake_id, "media_id": media_id, "task_id": row[0],
                    "duration_seconds": probe["duration_seconds"], "frames": probe["frame_count"],
                    "reused_existing": True}
        task_id = S.make_id("tsk", media_id + template_id)
        duration = probe["duration"]
        media_body = {
            "id": media_id,
            "source": {"sha256": identity["sha256"], "size_bytes": str(identity["size_bytes"]),
                       "display_name": display_name},
            "playback": {"sha256": identity["sha256"], "size_bytes": str(identity["size_bytes"]),
                         "display_name": display_name},
            "duration": duration,
            "video_start": S.rational(0),
            "video_end": duration,
            "common_origin": S.rational(0),
            "source_time_base": S.rational((probe["fps_den"], probe["fps_num"] or 1)),
            "proxy_time_base": S.rational((probe["fps_den"], probe["fps_num"] or 1)),
            "frame_count": probe["frame_count"],
            "frame_map_artifact": self._put_artifact(
                json.dumps({"note": "approximate until frame-accurate index lands in W04",
                            "fps": f'{probe["fps_num"]}/{probe["fps_den"]}',
                            "frame_count": probe["frame_count"]}).encode("utf-8")
            )["id"],
            "source_video_stream": 0,
            "coordinate_policy": "source_decoded_pixels_with_explicit_transform",
            "source_width": probe["width"],
            "source_height": probe["height"],
            "rotation_degrees": probe["rotation"],
            "pixel_aspect_ratio": S.rational(1),
        }
        task_body = {
            "id": task_id, "media_id": media_id,
            "template_id": template_id, "template_version": template_version,
            "revision": 0, "annotations": [], "extensions": {},
        }
        task_body["content_sha256"] = S.task_content_sha256(task_body)
        ts = _now()
        with self.conn:
            self.conn.execute("INSERT INTO intakes(id,group_id,state,display_name,source_sha256,"
                             "source_size,media_id,task_id,error_code,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                             (intake_id, group_id, "succeeded", display_name, identity["sha256"],
                              identity["size_bytes"], media_id, task_id, None, ts))
            self.conn.execute("INSERT OR REPLACE INTO media(id,body) VALUES(?,?)",
                              (media_id, json.dumps(media_body, ensure_ascii=False)))
            self.conn.execute("INSERT OR REPLACE INTO tasks(id,media_id,template_id,template_version,revision)"
                              " VALUES(?,?,?,?,0)",
                              (task_id, media_id, template_id, template_version))
            rev_id = S.make_id("rev", task_id + "0")
            self.conn.execute(
                "INSERT INTO task_revisions(id,task_id,revision,body,content_sha256,commit_id,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (rev_id, task_id, 0, json.dumps(task_body, ensure_ascii=False),
                 task_body["content_sha256"], None, ts))
        return {"intake_id": intake_id, "media_id": media_id, "task_id": task_id,
                "duration_seconds": probe["duration_seconds"], "frames": probe["frame_count"]}

    # -- reads ---------------------------------------------------------------
    def get_task(self, task_id: str) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT body FROM task_revisions WHERE task_id=? ORDER BY revision DESC LIMIT 1",
            (task_id,)).fetchone()
        if row is None:
            raise NotFound("TASK_NOT_FOUND", f"No such task {task_id}")
        return json.loads(row[0])

    def list_tasks(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT id, media_id, revision FROM tasks").fetchall()
        return [{"task_id": t, "media_id": m, "revision": r} for (t, m, r) in rows]

    def get_media(self, media_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT body FROM media WHERE id=?", (media_id,)).fetchone()
        if row is None:
            raise NotFound("MEDIA_NOT_FOUND", f"No such media {media_id}")
        return json.loads(row[0])

    def get_snapshot(self) -> Dict[str, Any]:
        media = [json.loads(r[0]) for r in self.conn.execute("SELECT body FROM media")]
        templates = [json.loads(r[0]) for r in self.conn.execute("SELECT body FROM templates")]
        tasks = []
        for (tid,) in self.conn.execute("SELECT id FROM tasks ORDER BY rowid"):
            tasks.append(self.get_task(tid))
        revs = self.conn.execute(
            "SELECT COUNT(*) FROM task_revisions WHERE revision>0").fetchone()[0]
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
            "batches": [],
            "commits": [],
            "artifacts": [{"id": r[0], "relative_path": r[1], "sha256": r[2], "size_bytes": str(r[3])}
                          for r in self.conn.execute(
                              "SELECT id,rel_path,sha256,size_bytes FROM artifacts")],
        }

    def _project_id(self) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key='project_id'").fetchone()
        if row:
            return row[0]
        pid = S.make_id("proj", self.root.resolve().__str__() + "v3")
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('project_id',?)", (pid,))
        return pid

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
    def apply_task_command(self, *, command_id: str, task_id: str, expected_revision: int,
                           expected_sha256: str, operation: str, payload: Dict[str, Any],
                           request_body: bytes) -> Dict[str, Any]:
        request_sha = hashlib.sha256(request_body).hexdigest()
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
        if current["content_sha256"] != expected_sha256:
            raise Conflict("MEDIA_CHANGED", "Task content hash mismatch",
                           details={"expected_sha256": expected_sha256,
                                    "current_sha256": current["content_sha256"]})
        annotations = [dict(a) for a in current["annotations"]]
        annotations = self._apply_operation(annotations, operation, payload)
        new_task = dict(current)
        new_task["annotations"] = annotations
        new_task["revision"] = current_rev + 1
        new_task["content_sha256"] = S.task_content_sha256(new_task)
        new_rev_id = S.make_id("rev", task_id + str(new_task["revision"]))
        commit_id = S.make_id("cmt", command_id)
        response = {
            "command_id": command_id, "applied": True, "task_id": task_id,
            "previous_revision": current_rev, "revision": new_task["revision"],
            "content_sha256": new_task["content_sha256"], "commit_id": commit_id,
        }
        ts = _now()
        with self.conn:
            self.conn.execute("UPDATE tasks SET revision=? WHERE id=?",
                              (new_task["revision"], task_id))
            self.conn.execute(
                "INSERT INTO task_revisions(id,task_id,revision,body,content_sha256,commit_id,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (new_rev_id, task_id, new_task["revision"],
                 json.dumps(new_task, ensure_ascii=False), new_task["content_sha256"], commit_id, ts))
            self.conn.execute(
                "INSERT OR REPLACE INTO durable_commits(id,body) VALUES(?,?)",
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
        if operation == "add_annotation":
            ann = dict(payload["annotation"])
            ann["id"] = S.require_id(ann["id"], "annotation.id")
            if any(a["id"] == ann["id"] for a in annotations):
                raise Conflict("ID_EXISTS", f"Annotation {ann['id']} already exists")
            self._validate_annotation(ann)
            annotations.append(ann)
        elif operation == "update_annotation":
            aid = payload["annotation_id"]
            changes = payload.get("changes", {})
            found = False
            for a in annotations:
                if a["id"] == aid:
                    found = True
                    for k, v in changes.items():
                        if k in ("id", "type"):
                            raise SchemaInvalid("INVALID_CHANGE", f"Cannot change {k}")
                        a[k] = v
                    self._validate_annotation(a)
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

    def _validate_annotation(self, ann: Dict[str, Any]) -> None:
        atype = ann.get("type")
        if atype not in ("whole_clip", "event", "segment"):
            raise SchemaInvalid("SCHEMA_INVALID", f"Bad annotation type {atype}")
        if atype == "segment":
            s, e = ann.get("start"), ann.get("end")
            if not s or not e:
                raise SchemaInvalid("SCHEMA_INVALID", "segment requires start and end")
            if S.rational_value(s["seconds"]) >= S.rational_value(e["seconds"]):
                raise SchemaInvalid("SCHEMA_INVALID", "segment start must be before end")
        if atype == "event" and "at" not in ann:
            raise SchemaInvalid("SCHEMA_INVALID", "event requires at")
        ann.setdefault("text", "")
        ann.setdefault("tags", [])
        ann.setdefault("track", "main")
        ann.setdefault("attributes", {})
        ann.setdefault("evidence_ids", [])
        ann.setdefault("provenance", {"origin": "manual", "actor_id": "user", "job_id": None})

    def export_snapshot(self, export_name: Optional[str] = None) -> Path:
        snap = self.get_snapshot()
        name = export_name or S.make_id("exp", str(_now()))
        out = self.exports_dir / f"{name}.json"
        tmp = self.tmp_dir / f"{name}.export.part"
        tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, out)
        return out
