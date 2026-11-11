"""Portable professional evidence must obey local object safety and budgets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_workspace_backup_recovery import _make_junction
from tests.test_workspace_professional_codec import resign
from tests.test_workspace_professional_http import (
    _commit, _export, _import_body, _seed, _task, _url,
)
from tests.test_workspace_server import api, http_video, request, running
from trace_annotate.workspace import professional_schema as P
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.professional_store import EXPORT_DOCUMENT_LIMIT, export_professional
from trace_annotate.workspace.store import WorkspaceStore


def _attach_evidence(server, store, data=b"portable evidence sample"):
    _seed(server, store)
    descriptor = store._put_artifact(data)
    result = _commit(server, store, "cmd-attach-portable-evidence", "update_annotation",
                     {"annotation_id": "ann-exact", "changes": {
                         "evidence_ids": [descriptor["id"]]}})
    assert result[0] == 200, result[2]
    return descriptor


def test_professional_export_refuses_a_linked_evidence_directory(api, tmp_path):
    server, store = api
    descriptor = _attach_evidence(server, store)
    outside = tmp_path / "outside-object-closure"
    store.objects_dir.rename(outside)
    if not _make_junction(store.objects_dir, outside):
        outside.rename(store.objects_dir)
        pytest.skip("this account cannot create a junction or directory symlink")
    before = store.get_snapshot()
    with pytest.raises(WorkspaceError) as error:
        export_professional(store, _task(store)["id"])
    assert error.value.code == "UNSAFE_PATH"
    assert store.get_snapshot() == before
    assert (outside / descriptor["sha256"]).read_bytes() == b"portable evidence sample"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_professional_export_refuses_unavailable_source_archive(api, http_video, damage):
    server, store = api
    _seed(server, store)
    document, _ = _export(server, store)
    destination = store.import_source(http_video)["task_id"]
    result = request(server, "POST", _url(store, "professional-import", destination),
                     _import_body(store, document, task_id=destination))
    assert result[0] == 200, result[2]
    archived = _task(store, destination)["extensions"]["professional"]["import_source"]["document_artifact"]
    path = store.root / archived["relative_path"]
    if damage == "missing":
        path.unlink()
    else:
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
    before = store.get_snapshot()
    with pytest.raises(WorkspaceError) as error:
        export_professional(store, destination)
    assert error.value.code == "RECOVERY_REQUIRED"
    assert store.get_snapshot() == before


@pytest.mark.parametrize("damage", ["declared_large", "actual_large"])
def test_professional_evidence_budget_is_checked_before_opening_the_sample(api, monkeypatch, damage):
    server, store = api
    data = b"x" * (512 * 1024 + 1) if damage == "declared_large" else b"small sample"
    descriptor = _attach_evidence(server, store, data)
    path = store.root / descriptor["relative_path"]
    if damage == "actual_large":
        path.write_bytes(b"x" * (512 * 1024 + 1))
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if self == path:
            raise AssertionError("oversized evidence was opened before its budget/size check")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    before = store.get_snapshot()
    with pytest.raises(WorkspaceError) as error:
        export_professional(store, _task(store)["id"])
    assert error.value.code == ("PROFESSIONAL_LIMIT_EXCEEDED" if damage == "declared_large"
                                else "RECOVERY_REQUIRED")
    assert store.get_snapshot() == before


def test_professional_http_import_accepts_a_real_document_at_the_export_byte_budget(api, http_video):
    server, store = api
    _seed(server, store)
    document, _ = _export(server, store)
    document["history"]["source_envelopes"].append({"padding": ""})
    document["content_sha256"] = P.document_digest(document)
    remaining = EXPORT_DOCUMENT_LIMIT - len(S.canonical_json(document))
    document["history"]["source_envelopes"][-1]["padding"] = (
        "中" * (remaining // 3) + "x" * (remaining % 3))
    document["content_sha256"] = P.document_digest(document)
    data = S.canonical_json(document)
    assert len(data) == EXPORT_DOCUMENT_LIMIT
    assert P.decode_document(data) == document
    destination = store.import_source(http_video)["task_id"]
    body = _import_body(store, document, command_id="cmd-full-document-budget", task_id=destination)
    assert len(S.canonical_json(body)) <= server.max_command_bytes
    result = request(server, "POST", _url(store, "professional-import", destination), S.canonical_json(body))
    assert result[0] == 200, result[2]
    imported = _task(store, destination)
    archive = imported["extensions"]["professional"]["import_source"]["document_artifact"]
    assert (store.root / archive["relative_path"]).read_bytes() == data
    assert len(S.canonical_json(imported)) < 32 * 1024
    reexported, _ = _export(server, store, destination)
    assert len(reexported["task"]["annotations"]) == len(document["task"]["annotations"])
    assert all(actual["text"] == original["text"] for actual, original in
               zip(reexported["task"]["annotations"], document["task"]["annotations"]))


@pytest.mark.parametrize("damage", ["task_budget", "missing_evidence", "wrong_local_anchor", "whitespace_legacy"])
def test_professional_import_preflights_rejection_before_publishing_objects(api, http_video, damage):
    server, store = api
    _seed(server, store)
    document, _ = _export(server, store)
    if damage == "task_budget":
        document["task"]["annotations"][0]["text"] = "x" * (800 * 1024)
    elif damage == "missing_evidence":
        absent = b"not present in the receiving workspace"
        digest = hashlib.sha256(absent).hexdigest()
        evidence = {"id": S.make_id("art", digest), "relative_path": "objects/" + digest,
                    "sha256": digest, "size_bytes": str(len(absent))}
        document["artifacts"].append(evidence)
        document["task"]["annotations"][0]["evidence_ids"] = [evidence["id"]]
    elif damage == "wrong_local_anchor":
        # The rational/PTS values remain internally consistent, but this index
        # belongs to a different row of the authoritative admitted frame map.
        document["task"]["annotations"][0]["start"]["anchor"]["frame_index"] = "3"
    else:
        document["task"]["annotations"][0]["legacy_id"] = " \t "
    resign(document, task=True)
    if damage != "whitespace_legacy":
        P.validate_document(document)
    destination = store.import_source(http_video)["task_id"]
    body = _import_body(store, document, command_id="cmd-preflight-" + damage.replace("_", "-"),
                        task_id=destination)
    before = store.get_snapshot()
    artifact_rows = store.conn.execute("SELECT * FROM artifacts ORDER BY id").fetchall()
    object_files = {path.name for path in store.objects_dir.iterdir()}
    result = request(server, "POST", _url(store, "professional-import", destination), body)
    assert result[0] == 422, result[2]
    assert store.get_snapshot() == before
    assert store.conn.execute("SELECT * FROM artifacts ORDER BY id").fetchall() == artifact_rows
    assert {path.name for path in store.objects_dir.iterdir()} == object_files
    assert request(server, "GET", "/api/v1/projects/current/commands/" + body["command_id"])[0] == 404


@pytest.mark.parametrize("damage", ["missing", "corrupt", "linked_directory"])
def test_professional_import_checks_unembedded_local_evidence_bytes_before_publication(
        api, http_video, tmp_path, damage):
    server, store = api
    descriptor = _attach_evidence(server, store)
    document, _ = _export(server, store)
    next(a for a in document["artifacts"] if a["id"] == descriptor["id"]).pop("data_base64")
    resign(document)
    destination = store.import_source(http_video)["task_id"]
    path = store.root / descriptor["relative_path"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"x" * path.stat().st_size)
    else:
        outside = tmp_path / "outside-import-objects"
        store.objects_dir.rename(outside)
        if not _make_junction(store.objects_dir, outside):
            outside.rename(store.objects_dir)
            pytest.skip("this account cannot create a junction or directory symlink")
    body = _import_body(store, document, command_id="cmd-check-local-evidence", task_id=destination)
    before = store.get_snapshot()
    artifact_rows = store.conn.execute("SELECT * FROM artifacts ORDER BY id").fetchall()
    object_files = {item.name for item in store.objects_dir.iterdir()}
    result = request(server, "POST", _url(store, "professional-import", destination), body)
    assert result[0] == 500, result[2]
    assert result[2]["error"]["code"] == ("UNSAFE_PATH" if damage == "linked_directory" else "RECOVERY_REQUIRED")
    assert store.get_snapshot() == before
    assert store.conn.execute("SELECT * FROM artifacts ORDER BY id").fetchall() == artifact_rows
    assert {item.name for item in store.objects_dir.iterdir()} == object_files
    assert request(server, "GET", "/api/v1/projects/current/commands/" + body["command_id"])[0] == 404


def test_exact_512kib_portable_evidence_survives_independent_workspace_roundtrip(api, http_video, tmp_path):
    server, source = api
    data = bytes(range(256)) * 2048
    assert len(data) == 512 * 1024
    descriptor = _attach_evidence(server, source, data)
    document, _ = _export(server, source)
    target = WorkspaceStore(tmp_path / "independent-target")
    target.create()
    try:
        target.import_source(http_video)
        with running(target) as target_server:
            body = _import_body(target, document, command_id="cmd-max-portable-evidence")
            result = request(target_server, "POST", _url(target, "professional-import"), S.canonical_json(body))
            assert result[0] == 200, result[2]
            imported = _task(target)
            assert all(a["review_status"] == "unreviewed" for a in imported["annotations"])
            assert (target.root / descriptor["relative_path"]).read_bytes() == data
            reexported, _ = _export(target_server, target)
            copied = next(a for a in reexported["artifacts"] if a["id"] == descriptor["id"])
            original = next(a for a in document["artifacts"] if a["id"] == descriptor["id"])
            assert copied == original
            assert json.loads(export_professional(target, imported["id"])["data"])["task"] == imported
    finally:
        target.close()
