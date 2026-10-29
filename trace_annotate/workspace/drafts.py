"""Revisioned rescue documents; saving a draft never applies a task command.

Draft revisions belong to an editor session, not the task's annotation revision.
An old, valid task base is retained for explicit conflict resolution after an
offline save. Pending commands are opaque recovery data and are never executed.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict

from . import schema as S
from .errors import Conflict, LimitExceeded, NotFound, SchemaInvalid, WorkspaceError


MAX_DRAFT_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 200000
KIND = "traceannotate-draft-rescue"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def _json_document(body):
    if not isinstance(body, dict):
        raise SchemaInvalid("DRAFT_INVALID", "A rescue body must be a JSON object")
    pending = [(body, 0, False)]
    containers = set()
    count = 0
    while pending:
        value, depth, leaving = pending.pop()
        if leaving:
            containers.remove(id(value))
            continue
        count += 1
        if depth > MAX_JSON_DEPTH or count > MAX_JSON_NODES:
            raise LimitExceeded("DRAFT_LIMIT", "Rescue JSON exceeds depth or item limits")
        if isinstance(value, (dict, list)):
            # Reject cycles, but allow aliases that JSON serializes as copies.
            if id(value) in containers:
                raise SchemaInvalid("DRAFT_INVALID", "Rescue JSON cannot contain cyclic containers")
            containers.add(id(value))
            pending.append((value, depth, True))
            values = value.values() if isinstance(value, dict) else value
            pending.extend((item, depth + 1, False) for item in values)
    encoded = S.canonical_json(body)
    if len(encoded) > MAX_DRAFT_BYTES:
        raise LimitExceeded("DRAFT_LIMIT", "Rescue JSON exceeds the four MiB limit")
    return encoded


class WorkspaceDrafts:
    """Draft-only table using the open store's single-writer transactions."""

    def __init__(self, store):
        self.store = store
        with store.transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS drafts(
                draft_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                editor_session_id TEXT NOT NULL, task_id TEXT NOT NULL,
                media_id TEXT NOT NULL, draft_revision INTEGER NOT NULL,
                body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(id),
                FOREIGN KEY(media_id) REFERENCES media(id))""")
            columns = {item[1] for item in connection.execute("PRAGMA table_info(drafts)")}
            required = {"draft_id", "project_id", "editor_session_id", "task_id", "media_id",
                        "draft_revision", "body", "body_sha256", "created_at", "updated_at"}
            if not required <= columns:
                raise WorkspaceError("RECOVERY_REQUIRED", "Draft storage schema is not readable")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_drafts_task ON drafts(task_id)")

    def _validate_body(self, body, *, draft_id, editor_session_id, task_id, media_id):
        encoded = _json_document(body)
        if body.get("kind") != KIND or type(body.get("schema_version")) is not int or body["schema_version"] != 1:
            raise SchemaInvalid("DRAFT_INVALID", "Unsupported rescue kind or schema version")
        expected = {"project_id": self.store._project_id(), "task_id": task_id, "media_id": media_id}
        for field, identity in expected.items():
            if body.get(field) != identity:
                raise Conflict("DRAFT_IDENTITY_MISMATCH", "Rescue identity does not match this project/task/media",
                               details={"field": field})
        for field, identity in (("draft_id", draft_id), ("editor_session_id", editor_session_id)):
            if field in body and body[field] != identity:
                raise Conflict("DRAFT_IDENTITY_MISMATCH", "Rescue editor identity differs from its request",
                               details={"field": field})
        for field in ("token", "access_token", "authorization", "cookie", "headers", "api_key", "password"):
            if field in body:
                raise SchemaInvalid("DRAFT_INVALID", "Rescue envelopes cannot carry transport credentials")
        if (not isinstance(body.get("current_form"), dict) or not isinstance(body.get("anchors"), dict)
                or type(body.get("dirty")) is not bool):
            raise SchemaInvalid("DRAFT_INVALID", "Rescue requires form, anchors and a boolean dirty state")
        revision = body.get("base_revision")
        if type(revision) is not int or not 0 <= revision <= S.MAX_SAFE_INTEGER:
            raise SchemaInvalid("DRAFT_INVALID", "Rescue base revision must be a nonnegative safe integer")
        S.require_hash(body.get("base_sha256"), "base_sha256")
        for field in ("draft_revision",):
            if field in body and (type(body[field]) is not int or not 0 <= body[field] <= S.MAX_SAFE_INTEGER):
                raise SchemaInvalid("DRAFT_INVALID", "Embedded draft revision is invalid")
        selected = body.get("selected_id")
        if selected is not None:
            S.require_id(selected, "selected_id")
        task = self.store.get_task(task_id)
        if task["media_id"] != media_id:
            raise Conflict("DRAFT_IDENTITY_MISMATCH", "Draft media does not belong to its task")
        media = self.store.get_media(media_id)
        for field, media_key in (("source_sha256", "source"), ("playback_sha256", "playback")):
            S.require_hash(body.get(field), field)
            if body[field] != media[media_key]["sha256"]:
                raise Conflict("DRAFT_IDENTITY_MISMATCH", "Rescue source or playback hash does not match",
                               details={"field": field})
        base = self.store.conn.execute("SELECT content_sha256,body FROM task_revisions WHERE task_id=? AND revision=?",
                                       (task_id, revision)).fetchone()
        if base is None or base[0] != body["base_sha256"]:
            raise Conflict("DRAFT_BASE_MISMATCH", "Rescue base is not a known revision of this task")
        snapshot = body.get("task_snapshot")
        if snapshot is not None:
            if not isinstance(snapshot, dict) or any(snapshot.get(field) != expected_value for field, expected_value in (
                    ("id", task_id), ("media_id", media_id), ("revision", revision),
                    ("content_sha256", body["base_sha256"]))):
                raise Conflict("DRAFT_IDENTITY_MISMATCH", "Rescue task snapshot does not match its base")
            if S.canonical_json(snapshot) != S.canonical_json(json.loads(base[1])):
                raise Conflict("DRAFT_BASE_MISMATCH", "Rescue task snapshot differs from the stored base revision")
        pending = body.get("pending_command")
        if pending is not None:
            if not isinstance(pending, dict):
                raise SchemaInvalid("DRAFT_INVALID", "Pending command must be recovery data or null")
            command = pending.get("command", pending)
            if not isinstance(command, dict):
                raise SchemaInvalid("DRAFT_INVALID", "Pending command envelope is invalid")
            for field, identity in expected.items():
                if field in command and command[field] != identity:
                    raise Conflict("DRAFT_IDENTITY_MISMATCH", "Pending command belongs to another project/task/media")
        return encoded, task

    def save(self, *, draft_id, editor_session_id, task_id, media_id, expected_draft_revision, body):
        for field, value in (("draft_id", draft_id), ("editor_session_id", editor_session_id),
                             ("task_id", task_id), ("media_id", media_id)):
            S.require_id(value, field)
        if type(expected_draft_revision) is not int or not 0 <= expected_draft_revision < S.MAX_SAFE_INTEGER:
            raise SchemaInvalid("DRAFT_INVALID", "Expected draft revision must be a nonnegative safe integer")
        with self.store.transaction() as connection:
            encoded, _ = self._validate_body(body, draft_id=draft_id, editor_session_id=editor_session_id,
                                             task_id=task_id, media_id=media_id)
            current = connection.execute("SELECT editor_session_id,task_id,media_id,draft_revision FROM drafts WHERE draft_id=?",
                                          (draft_id,)).fetchone()
            if current is not None and current[:3] != (editor_session_id, task_id, media_id):
                raise Conflict("DRAFT_OWNER_CONFLICT", "An existing draft belongs to another editor/task/media")
            actual = current[3] if current else 0
            if actual != expected_draft_revision:
                raise Conflict("DRAFT_REVISION_CONFLICT", "Draft changed after this editor's saved revision",
                               details={"expected_draft_revision": expected_draft_revision,
                                        "current_draft_revision": actual})
            now = _now()
            if current is None:
                connection.execute("INSERT INTO drafts VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (draft_id, self.store._project_id(), editor_session_id, task_id, media_id, 1,
                     encoded.decode("utf-8"), S.sha256_bytes(encoded), now, now))
            else:
                connection.execute("UPDATE drafts SET draft_revision=?,body=?,body_sha256=?,updated_at=? WHERE draft_id=?",
                    (actual + 1, encoded.decode("utf-8"), S.sha256_bytes(encoded), now, draft_id))
            return self.get(draft_id)

    def _record(self, row):
        fields = ("draft_id", "project_id", "editor_session_id", "task_id", "media_id", "draft_revision",
                  "body", "body_sha256", "created_at", "updated_at")
        record = dict(zip(fields, row))
        try:
            if type(record["draft_revision"]) is not int or not 1 <= record["draft_revision"] <= S.MAX_SAFE_INTEGER:
                raise ValueError("Invalid stored draft revision")
            for field in ("draft_id", "editor_session_id", "task_id", "media_id"):
                S.require_id(record[field], field)
            record["body"] = json.loads(record["body"])
            encoded, task = self._validate_body(record["body"], draft_id=record["draft_id"],
                editor_session_id=record["editor_session_id"], task_id=record["task_id"], media_id=record["media_id"])
            if record["project_id"] != self.store._project_id() or S.sha256_bytes(encoded) != record["body_sha256"]:
                raise ValueError("Draft identity or content hash changed")
        except (ValueError, TypeError, WorkspaceError) as error:
            raise WorkspaceError("RECOVERY_REQUIRED", "Saved draft is corrupt; annotations were not changed",
                                 details={"draft_id": record["draft_id"]}) from error
        record["base_conflict"] = (task["revision"] != record["body"]["base_revision"] or
                                    task["content_sha256"] != record["body"]["base_sha256"])
        record["current_task_revision"] = task["revision"]
        return record

    def get(self, draft_id):
        S.require_id(draft_id, "draft_id")
        with self.store.transaction() as connection:
            row = connection.execute("SELECT draft_id,project_id,editor_session_id,task_id,media_id,draft_revision,"
                                     "body,body_sha256,created_at,updated_at FROM drafts WHERE draft_id=?", (draft_id,)).fetchone()
            if row is None:
                raise NotFound("DRAFT_NOT_FOUND", "No saved rescue document has this draft id")
            return self._record(row)

    def list(self, task_id=None, *, include_clean=False):
        if task_id is not None:
            S.require_id(task_id, "task_id")
        if type(include_clean) is not bool:
            raise SchemaInvalid("DRAFT_INVALID", "include_clean must be a boolean")
        with self.store.transaction() as connection:
            sql = "SELECT draft_id FROM drafts"
            values = ()
            if task_id is not None:
                sql += " WHERE task_id=?"
                values = (task_id,)
            records = [self.get(row[0]) for row in connection.execute(sql + " ORDER BY updated_at,draft_id", values).fetchall()]
            return [record for record in records if include_clean or record["body"]["dirty"]]
