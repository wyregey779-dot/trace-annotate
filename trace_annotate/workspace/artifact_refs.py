"""One fail-closed artifact reference closure for GC and backup validation.

Legacy owners remain authoritative; no backfill rewrites their histories. Typed
pins are additive and support immutable historical analysis/render ownership.
"""
from __future__ import annotations

import json
import sqlite3

from . import schema as S
from .errors import WorkspaceError

OWNER_KINDS = frozenset({"task_revision", "analysis_report", "evidence", "render_manifest", "export_pin"})
REFERENCE_DDL = (
    "CREATE TABLE IF NOT EXISTS artifact_references (artifact_id TEXT NOT NULL REFERENCES artifacts(id), "
    "owner_kind TEXT NOT NULL, owner_id TEXT NOT NULL, owner_revision TEXT NOT NULL, role TEXT NOT NULL, "
    "PRIMARY KEY(artifact_id,owner_kind,owner_id,owner_revision,role))",
    "CREATE TABLE IF NOT EXISTS artifact_gc_pending (artifact_id TEXT PRIMARY KEY, "
    "rel_path TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL)",
)


def _fail(message):
    raise WorkspaceError("RECOVERY_REQUIRED", message)


def _object(text):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate key")
            out[key] = value
        return out
    try:
        value = json.loads(text, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise WorkspaceError("RECOVERY_REQUIRED", "Artifact owner JSON is corrupt") from exc


def validate_owner(connection, owner_kind, owner_id, owner_revision, role):
    if not isinstance(owner_kind, str) or owner_kind not in OWNER_KINDS:
        _fail("Artifact owner kind is unsupported")
    try:
        S.require_id(owner_id, "owner_id")
        S.require_id(role, "artifact role")
    except WorkspaceError as exc:
        raise WorkspaceError("RECOVERY_REQUIRED", "Artifact owner identity is invalid") from exc
    if (not isinstance(owner_revision, str) or not owner_revision.isascii()
            or not owner_revision.isdecimal() or len(owner_revision) > 16
            or str(int(owner_revision)) != owner_revision or int(owner_revision) > S.MAX_SAFE_INTEGER):
        _fail("Artifact owner revision is invalid")
    if owner_kind == "task_revision":
        row = connection.execute("SELECT revision FROM task_revisions WHERE id=?", (owner_id,)).fetchone()
        if row is None or str(row[0]) != owner_revision:
            _fail("Artifact historical task revision is missing or inconsistent")


def collect_artifact_references(connection):
    """Validate all known owner formats, returning stable typed reference rows.

    This is a DB-only routine usable on immutable backup snapshots. File hashes
    are checked by the caller. Registered-but-unowned artifacts are permitted.
    """
    try:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        registered = {}
        for ident, rel, digest, size in connection.execute("SELECT id,rel_path,sha256,size_bytes FROM artifacts"):
            S.require_id(ident, "artifact_id")
            S.require_hash(digest)
            if (rel != "objects/" + digest or ident != S.make_id("art", digest)
                    or type(size) is not int or size < 0):
                _fail("Artifact registration is invalid")
            registered[ident] = (rel, digest, size)
        refs = set()

        def add(ident, kind, owner, revision, role, descriptor=None):
            if ident not in registered:
                _fail("Artifact owner points to a missing registration")
            if descriptor is not None:
                if not isinstance(descriptor, dict):
                    _fail("Artifact descriptor is invalid")
                rel, digest, size = registered[ident]
                for key, expected in (("relative_path", rel), ("sha256", digest), ("size_bytes", str(size))):
                    if key in descriptor and descriptor[key] != expected:
                        _fail("Artifact descriptor differs from registered identity")
            refs.add((ident, kind, str(owner), str(revision), str(role)))

        # ID-only legacy orphan-media pins are retained conservatively; active
        # media objects are additionally checked by WorkspaceStore.
        for key, text in connection.execute("SELECT key,value FROM meta WHERE key LIKE 'media_objects:%'"):
            bindings = _object(text)
            if not bindings:
                _fail("Media artifact bindings are empty")
            for role, reference in bindings.items():
                if not isinstance(reference, dict) or not isinstance(reference.get("id"), str):
                    _fail("Media artifact binding is invalid")
                add(reference["id"], "media_binding", key.split(":", 1)[1], "0", role, reference)

        queue_tables = {"batches", "jobs", "attempts", "batch_jobs", "job_commits"}
        if tables & queue_tables:
            if not queue_tables <= tables:
                _fail("Queue artifact owners are incomplete")
            for job_id, ident, text in connection.execute("SELECT job_id,id,body FROM job_commits"):
                commit = _object(text)
                if commit.get("id") != ident or commit.get("job_id") != job_id:
                    _fail("Queue commit identity is inconsistent")
                artifact = commit.get("artifact")
                if not isinstance(artifact, dict) or set(artifact) != {"id", "relative_path", "sha256", "size_bytes"}:
                    _fail("Queue commit artifact is invalid")
                add(artifact["id"], "job_commit", ident, commit.get("lease_generation", "0"), "candidate", artifact)
            for job_id, attempt, ident in connection.execute(
                    "SELECT job_id,attempt,diagnostic_artifact_id FROM attempts WHERE diagnostic_artifact_id IS NOT NULL"):
                add(ident, "job_diagnostic", job_id, attempt, "diagnostic")

        for ident, text in connection.execute("SELECT id,body FROM durable_commits"):
            commit = _object(text)
            if "artifact" in commit:
                artifact = commit["artifact"]
                if (not isinstance(artifact, dict) or set(artifact) != {"id", "relative_path", "sha256", "size_bytes"}
                        or commit.get("id") != ident):
                    _fail("Durable commit artifact is invalid")
                add(artifact["id"], "durable_commit", ident, commit.get("lease_generation", "0"), "candidate", artifact)

        # Every historical task body is scanned, not just the current revision.
        # Reserved fields are explicit: arbitrary strings resembling artifact
        # IDs are never treated as references or followed as filesystem paths.
        for ident, task_id, revision, text in connection.execute("SELECT id,task_id,revision,body FROM task_revisions"):
            body = _object(text)
            if body.get("id") != task_id or body.get("revision") != revision:
                _fail("Historical artifact owner identity is inconsistent")
            professional = body.get("extensions", {}).get("professional", {})
            if not isinstance(professional, dict):
                _fail("Professional artifact owner is invalid")
            source = professional.get("import_source", {})
            if not isinstance(source, dict):
                _fail("Professional source archive is invalid")
            if "document_artifact" in source:
                artifact = source["document_artifact"]
                if not isinstance(artifact, dict) or set(artifact) != {"id", "relative_path", "sha256", "size_bytes"}:
                    _fail("Professional source archive descriptor is invalid")
                add(artifact["id"], "task_revision", ident, revision, "professional_source", artifact)
            reports = professional.get("analysis_reports", [])
            if not isinstance(reports, list):
                _fail("Analysis report references are invalid")
            for report in reports:
                if not isinstance(report, dict):
                    _fail("Analysis report owner is invalid")
                artifact_ids, artifacts = report.get("artifact_ids", []), report.get("artifacts", [])
                if not isinstance(artifact_ids, list) or not isinstance(artifacts, list):
                    _fail("Analysis artifact references must be arrays")
                for artifact_id in artifact_ids:
                    add(artifact_id, "task_revision", ident, revision, "analysis")
                for artifact in artifacts:
                    if not isinstance(artifact, dict):
                        _fail("Analysis artifact descriptor is invalid")
                    add(artifact.get("id"), "task_revision", ident, revision, "analysis", artifact)
        if "artifact_references" in tables:
            for artifact_id, kind, owner, revision, role in connection.execute(
                    "SELECT artifact_id,owner_kind,owner_id,owner_revision,role FROM artifact_references"):
                validate_owner(connection, kind, owner, revision, role)
                add(artifact_id, kind, owner, revision, role)
        return [{"artifact_id": a, "owner_kind": k, "owner_id": o, "owner_revision": v, "role": r}
                for a, k, o, v, r in sorted(refs)]
    except (sqlite3.DatabaseError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise WorkspaceError("RECOVERY_REQUIRED", "Artifact reference closure cannot be validated") from exc
