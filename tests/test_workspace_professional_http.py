"""Professional data must survive actual HTTP, new identities and recovery.

These are boundary and persistence tests, not approval of annotation meaning or
audio timbre. The media fixture is encoded and admitted through the real product.
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from tests.test_workspace_server import api, assert_error, command, http_video, request, running
from trace_annotate.workspace import backup as B
from trace_annotate.workspace import schema as S
from trace_annotate.workspace import professional_schema as P
from trace_annotate.workspace.professional_store import EXPORT_DOCUMENT_LIMIT
from trace_annotate.workspace.server import MAX_COMMAND_BYTES
from trace_annotate.workspace.frames import FrameService
from trace_annotate.workspace.store import WorkspaceStore


def _task(store, ident=None):
    return store.get_task(ident or store.list_tasks()[0]["task_id"])


def _url(store, suffix, ident=None):
    return f"/api/v1/projects/current/tasks/{_task(store, ident)['id']}/{suffix}"


def _commit(server, store, command_id, operation, payload, task_id=None):
    task = _task(store, task_id)
    return request(server, "POST", _url(store, "commands", task_id), {
        "command_id": command_id, "expected_revision": task["revision"],
        "expected_sha256": task["content_sha256"], "operation": operation, "payload": payload})


def _seed(server, store, *, reviewed=False):
    task = _task(store)
    rows = FrameService(store).page(task["media_id"])["frames"]
    def anchor(row):
        return {"seconds": row["seconds"], "anchor": {
            key: row[key] for key in ("frame_index", "source_pts", "proxy_pts")}}
    common = {"text": "完整中文\n-42 / 1,234 / 😀，大括号{原文}与\\反斜杠", "tags": ["same", "same", "中文"],
              "track": "镜头\t轨道", "attributes": {}, "evidence_ids": [],
              "provenance": {"origin": "manual", "actor_id": "user", "job_id": None}}
    annotations = [
        {**copy.deepcopy(common), "id": "ann-exact", "type": "segment", "label": "完整镜头",
         "start": anchor(rows[2]), "end": anchor(rows[7])},
        {**copy.deepcopy(common), "id": "ann-event", "type": "event", "label": "切点事件",
         "at": anchor(rows[9])},
        {**copy.deepcopy(common), "id": "ann-whole", "type": "whole_clip", "label": "整片说明"},
    ]
    for index, ann in enumerate(annotations):
        response = _commit(server, store, f"cmd-seed-{index}", "add_annotation", {"annotation": ann})
        assert response[0] == 200, response[2]
    if reviewed:
        response = _commit(server, store, "cmd-review-source", "review_annotation",
                           {"annotation_id": "ann-exact", "reviewed": True})
        assert response[0] == 200, response[2]
    return _task(store)


def _export(server, store, task_id=None):
    status, headers, document = request(server, "GET", _url(store, "export?format=professional", task_id))
    assert status == 200, document
    assert document["kind"] == "traceannotate.professional-task"
    assert document["schema_version"] == 1
    return document, headers


def _import_body(store, document, *, command_id="cmd-professional-import", task_id=None):
    task = _task(store, task_id)
    return {"command_id": command_id, "expected_revision": task["revision"],
            "expected_sha256": task["content_sha256"], "document": copy.deepcopy(document)}


def _report(store, *, scope="file", subject=None, ident="report-declared-width"):
    task = _task(store)
    media = store.get_media(task["media_id"])
    report = {"id": ident, "kind": "technical_declared", "scope": scope,
              "file_sha256": media["source"]["sha256"], "stream_index": None,
              "time_domain": "admitted_source_seconds", "window": None,
              "subject_annotation_id": subject,
              "method": {"id": "fixture-ffprobe", "version": "1", "parameters": {}},
              "input_digest": "0" * 64, "status": "available", "evidence_ids": [],
              "values": [{"name": "source_width", "value": media["source_width"], "unit": "px",
                          "basis": "declared", "status": "available", "reason": None}]}
    if scope == "interval":
        annotation = next(a for a in task["annotations"] if a["id"] == subject)
        report["window"] = {"start": annotation["start"]["seconds"], "end": annotation["end"]["seconds"]}
    report["input_digest"] = P.analysis_input_digest(report)
    return report


def _report_body(store, report, *, command_id="cmd-save-report"):
    task = _task(store)
    return {"command_id": command_id, "expected_revision": task["revision"],
            "expected_sha256": task["content_sha256"], "report": copy.deepcopy(report)}


def _archived_document(store, task):
    """Verify the complete passive source bytes, not only an archive label."""
    record = task["extensions"]["professional"]["import_source"]
    assert "document" not in record  # No repeated full document in task snapshots.
    artifact = record["document_artifact"]
    assert artifact["relative_path"] == "objects/" + artifact["sha256"]
    data = (store.root / artifact["relative_path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    assert len(data) == int(artifact["size_bytes"])
    return json.loads(data)


def _reject_without_mutation(server, store, url, body, *, status=None, code=None, **kwargs):
    before = store.get_snapshot()
    response = request(server, "POST", url, body, **kwargs)
    if status is not None and code is not None:
        assert_error(response, status, code)
    else:
        assert response[0] in {400, 409, 413, 422}, response[2]
        assert isinstance(response[2]["error"]["code"], str)
    assert store.get_snapshot() == before
    return response


def test_professional_export_preserves_all_annotation_kinds_exact_anchors_and_revision(api):
    server, store = api
    original = _seed(server, store, reviewed=True)
    before = store.get_snapshot()
    doc, headers = _export(server, store)
    assert doc["task"] == original
    assert doc["media"] == store.get_media(original["media_id"])
    assert [a["type"] for a in doc["task"]["annotations"]] == ["segment", "event", "whole_clip"]
    assert doc["task"]["annotations"][0]["start"]["seconds"] == S.rational("0.2")
    assert doc["task"]["annotations"][0]["start"]["anchor"]["frame_index"] == "2"
    assert headers["X-TraceAnnotate-Task-Revision"] == str(original["revision"])
    assert headers["X-TraceAnnotate-Content-SHA256"] == original["content_sha256"]
    assert headers["Content-Type"].startswith("application/json")
    assert "professional" in headers["Content-Disposition"]
    assert "media" in headers["X-TraceAnnotate-Export-Losses"].lower()
    assert doc["history"]["scope"] == "available_local_audit_index"
    assert all("annotations" not in row for row in doc["history"]["revisions"])
    assert store.get_snapshot() == before


def test_professional_import_is_durable_idempotent_and_preserves_passive_source_history(api, http_video):
    server, store = api
    source = _seed(server, store, reviewed=True)
    doc, _ = _export(server, store)
    destination = store.import_source(http_video)["task_id"]
    original_destination = _task(store, destination)
    body = _import_body(store, doc, task_id=destination)
    path = _url(store, "professional-import", destination)
    response = request(server, "POST", path, body)
    assert response[0] == 200, response[2]
    receipt = response[2]
    assert receipt["revision"] == original_destination["revision"] + 1
    assert receipt["commit_id"]
    replay = request(server, "POST", path, body)
    assert replay[0] == response[0] and replay[2] == response[2]
    imported = _task(store, destination)
    assert imported["revision"] == 1
    assert len(imported["annotations"]) == len(source["annotations"])
    assert {a["id"] for a in imported["annotations"]}.isdisjoint({a["id"] for a in source["annotations"]})
    for original, actual in zip(source["annotations"], imported["annotations"]):
        for key in ("type", "label", "text", "track", "tags", "start", "end", "at"):
            if key in original:
                assert actual[key] == original[key]
        assert actual["review_status"] == "unreviewed"
        assert "review_binding" not in actual
        assert actual["provenance"]["origin"] == "imported"
    archive_record = imported["extensions"]["professional"]["import_source"]
    archived = _archived_document(store, imported)
    assert archived == doc
    assert archive_record["scope"] == "historical_source_not_current_review_or_local_receipt"
    assert archived["task"]["annotations"][0]["review_status"] == "reviewed"
    assert _task(store, source["id"]) == source
    status, _, saved_receipt = request(server, "GET", "/api/v1/projects/current/commands/" + body["command_id"])
    assert status == 200 and saved_receipt["state"] == "committed"
    assert saved_receipt["response"] == receipt


def test_professional_import_conflicts_do_not_overwrite_later_edits(api, http_video):
    server, store = api
    _seed(server, store)
    doc, _ = _export(server, store)
    destination = store.import_source(http_video)["task_id"]
    body = _import_body(store, doc, task_id=destination)
    path = _url(store, "professional-import", destination)
    assert request(server, "POST", path, body)[0] == 200
    before = _task(store, destination)
    changed = copy.deepcopy(body)
    changed["document"]["task"]["annotations"][0]["text"] = "Changed input cannot reuse command ID"
    _reject_without_mutation(server, store, path, changed, status=409, code="IDEMPOTENCY_CONFLICT")
    stale = copy.deepcopy(body)
    stale["command_id"] = "cmd-professional-stale"
    _reject_without_mutation(server, store, path, stale, status=409, code="REVISION_CONFLICT")
    assert _task(store, destination) == before


@pytest.mark.parametrize("headers,authorized,code", [
    ({}, False, "INVALID_TOKEN"),
    ({"Origin": "https://foreign.invalid"}, True, "FORBIDDEN_ORIGIN"),
    ({"Host": "foreign.invalid"}, True, "FORBIDDEN_ORIGIN"),
    ({"Sec-Fetch-Site": "cross-site"}, True, "FORBIDDEN_ORIGIN"),
])
def test_professional_routes_preserve_existing_origin_and_token_boundaries(api, headers, authorized, code):
    server, store = api
    doc, _ = _export(server, store)
    before = store.get_snapshot()
    assert_error(request(server, "GET", _url(store, "export?format=professional"),
                         headers=headers, authorized=authorized), 403, code)
    body = _import_body(store, doc)
    _reject_without_mutation(server, store, _url(store, "professional-import"), body,
                            status=403, code=code, headers=headers, authorized=authorized)
    report_body = _report_body(store, _report(store))
    _reject_without_mutation(server, store, _url(store, "analysis-reports"), report_body,
                            status=403, code=code, headers=headers, authorized=authorized)
    assert store.get_snapshot() == before


@pytest.mark.parametrize("suffix", ["professional-import", "analysis-reports"])
@pytest.mark.parametrize("raw", [b'{"command_id":"cmd-invalid", "document": NaN}',
                                  b'{"command_id":"cmd-a","command_id":"cmd-b"}',
                                  b'{"document":"\\ud800"}', b"[]", b"null"])
def test_professional_write_routes_reject_malformed_json_without_receipt(api, suffix, raw):
    server, store = api
    _reject_without_mutation(server, store, _url(store, suffix), raw)


def test_professional_import_rejects_wrong_media_and_unknown_version_without_mutation(api, http_video):
    server, store = api
    _seed(server, store)
    doc, _ = _export(server, store)
    destination = store.import_source(http_video)["task_id"]
    path = _url(store, "professional-import", destination)
    version = copy.deepcopy(doc)
    version["schema_version"] = 987654
    _reject_without_mutation(server, store, path, _import_body(store, version, task_id=destination))
    wrong_media = copy.deepcopy(doc)
    wrong_media["media"]["source"]["sha256"] = "f" * 64
    _reject_without_mutation(server, store, path, _import_body(store, wrong_media, task_id=destination))
    # A fully valid professional document for a *different* admitted source is
    # stronger than a merely tampered self-digest.
    other_path = Path(http_video).with_name("different-professional-source.mp4")
    original_bytes = Path(http_video).read_bytes()
    if not other_path.exists():
        other_path.write_bytes(original_bytes + b"different-source-identity")
    other = store.import_source(other_path)["task_id"]
    other_doc, _ = _export(server, store, other)
    assert other_doc["media"]["source"]["sha256"] != doc["media"]["source"]["sha256"]
    _reject_without_mutation(server, store, path,
                            _import_body(store, other_doc, command_id="cmd-wrong-valid-media", task_id=destination))
    assert _task(store, destination)["revision"] == 0


def test_professional_export_rejects_track_options_without_changing_state(api):
    server, store = api
    before = store.get_snapshot()
    assert_error(request(server, "GET", _url(store, "export?format=professional&track=main")),
                 422, "UNSUPPORTED_EXPORT_OPTION")
    assert store.get_snapshot() == before


def test_analysis_report_save_is_durable_idempotent_and_does_not_self_invalidate(api):
    server, store = api
    report = _report(store)
    body = _report_body(store, report)
    path = _url(store, "analysis-reports")
    first = request(server, "POST", path, body)
    assert first[0] == 200, first[2]
    assert first[2]["analysis_id"] == report["id"]
    replay = request(server, "POST", path, body)
    assert replay[0] == 200 and replay[2] == first[2]
    assert _task(store)["revision"] == 1
    doc, _ = _export(server, store)
    assert doc["analyses"] == [report]
    assert doc["analysis_assessment"] == [{"id": report["id"], "status": "fresh", "reason": None}]
    # A regular label does not change the decoded file on which this file
    # report depends. Nor should saving the report invalidate itself.
    path, add_body = command(store, "cmd-unrelated-annotation")
    assert request(server, "POST", path, add_body)[0] == 200
    after, _ = _export(server, store)
    assert after["analyses"] == [report]
    assert after["analysis_assessment"][0]["status"] == "fresh"


def test_interval_analysis_becomes_stale_only_when_its_subject_window_changes(api):
    server, store = api
    _seed(server, store)
    report = _report(store, scope="interval", subject="ann-exact", ident="report-interval")
    result = request(server, "POST", _url(store, "analysis-reports"), _report_body(store, report))
    assert result[0] == 200, result[2]
    exported, _ = _export(server, store)
    assert exported["analysis_assessment"][0]["status"] == "fresh"
    assert _commit(server, store, "cmd-label-only", "update_annotation",
                   {"annotation_id": "ann-exact", "changes": {"label": "新标签不改时间"}})[0] == 200
    labelled, _ = _export(server, store)
    assert labelled["analysis_assessment"][0]["status"] == "fresh"
    assert _commit(server, store, "cmd-boundary-change", "update_annotation",
                   {"annotation_id": "ann-exact", "changes": {"end": {"seconds": S.rational("0.8")}}})[0] == 200
    moved, _ = _export(server, store)
    assert moved["analyses"] == [report]  # Historical raw measurement retained.
    assert moved["analysis_assessment"][0]["status"] == "stale"
    assert moved["analysis_assessment"][0]["reason"]


@pytest.mark.parametrize("case", ["wrong_file", "unavailable_with_value", "inference_as_fact", "bad_unit", "unknown_field"])
def test_analysis_rejections_never_change_task_or_create_receipt(api, case):
    server, store = api
    report = _report(store)
    if case == "wrong_file":
        report["file_sha256"] = "f" * 64
    elif case == "unavailable_with_value":
        report["status"] = "unavailable"
        report["values"][0].update(status="unavailable", reason="no_input")
    elif case == "inference_as_fact":
        report["values"][0]["basis"] = "inferred"
    elif case == "bad_unit":
        report["values"][0]["unit"] = "system_volume_percent"
    else:
        report["unvalidated_extension"] = {"pretend_reviewed": True}
    report["input_digest"] = P.analysis_input_digest(report)
    body = _report_body(store, report, command_id="cmd-reject-" + case.replace("_", "-"))
    _reject_without_mutation(server, store, _url(store, "analysis-reports"), body)
    assert_error(request(server, "GET", "/api/v1/projects/current/commands/" + body["command_id"]),
                 404, "COMMAND_UNKNOWN")


def test_professional_import_history_survives_close_reopen_and_new_path_backup(tmp_path, http_video):
    root = tmp_path / "original"
    store = WorkspaceStore(root)
    store.create()
    store.import_source(http_video)
    try:
        with running(store) as server:
            _seed(server, store, reviewed=True)
            report = _report(store)
            analysis_result = request(server, "POST", _url(store, "analysis-reports"), _report_body(store, report))
            assert analysis_result[0] == 200, analysis_result[2]
            doc, _ = _export(server, store)
            destination = store.import_source(http_video)["task_id"]
            body = _import_body(store, doc, task_id=destination)
            result = request(server, "POST", _url(store, "professional-import", destination), body)
            assert result[0] == 200, result[2]
            task_before = _task(store, destination)
            receipt_before = store.get_command_receipt(body["command_id"])
        backup = tmp_path / "backup"
        B.backup_workspace(store, backup)
    finally:
        store.close()
    reopened = WorkspaceStore(root)
    reopened.open()
    try:
        assert _task(reopened, destination) == task_before
        assert reopened.get_command_receipt(body["command_id"]) == receipt_before
    finally:
        reopened.close()
    restored_path = tmp_path / "restored"
    restore = B.restore_workspace(backup, restored_path)
    assert restore["jobs_restarted"] is False and restore["commands_replayed"] is False
    restored = WorkspaceStore(restored_path)
    restored.open()
    try:
        assert _task(restored, destination) == task_before
        assert restored.get_command_receipt(body["command_id"]) == receipt_before
        with running(restored) as server:
            exported, _ = _export(server, restored, destination)
            assert _archived_document(restored, exported["task"]) == doc
            assert exported["analyses"] == [report]
            assert exported["analysis_assessment"][0]["status"] == "fresh"
    finally:
        restored.close()


def test_professional_import_maps_media_and_subjects_between_independent_workspaces(tmp_path, http_video):
    source = WorkspaceStore(tmp_path / "source")
    target = WorkspaceStore(tmp_path / "target")
    source.create()
    target.create()
    try:
        source.import_source(http_video)
        target.import_source(http_video)
        with running(source) as server:
            source_task = _seed(server, source, reviewed=True)
            interval = _report(source, scope="interval", subject="ann-exact", ident="report-source-interval")
            result = request(server, "POST", _url(source, "analysis-reports"), _report_body(source, interval))
            assert result[0] == 200, result[2]
            document, _ = _export(server, source)
        target_task = _task(target)
        assert target_task["id"] != source_task["id"]
        # Media IDs are deterministic content identities in the current
        # product. Independent projects may legitimately reuse the same ID;
        # the receiving project's admitted media remains authoritative.
        assert target_task["media_id"] == source_task["media_id"]
        with running(target) as server:
            body = _import_body(target, document, command_id="cmd-cross-workspace")
            status, _, receipt = request(server, "POST", _url(target, "professional-import"), body)
            assert status == 200, receipt
            imported = _task(target)
            exported, _ = _export(server, target)
            assert exported["media"]["id"] == target_task["media_id"]
            assert imported["media_id"] == target_task["media_id"]
            assert imported["annotations"][0]["start"] == source_task["annotations"][0]["start"]
            # Report dependency follows its mapped current local subject, while
            # its historical original remains byte-for-byte JSON in the source envelope.
            imported_report = exported["analyses"][0]
            assert imported_report["subject_annotation_id"] == imported["annotations"][0]["id"]
            assert imported_report["subject_annotation_id"] != "ann-exact"
            assert imported_report["input_digest"] != interval["input_digest"]
            assert imported_report["input_digest"] == P.analysis_input_digest(imported_report)
            assert exported["analysis_assessment"] == [{"id": interval["id"], "status": "fresh", "reason": None}]
            assert _archived_document(target, imported) == document
            assert all(a["review_status"] == "unreviewed" for a in imported["annotations"])
    finally:
        source.close()
        target.close()


def test_report_conflicts_and_immutable_ids_preserve_the_last_committed_values(api):
    server, store = api
    report = _report(store)
    body = _report_body(store, report)
    path = _url(store, "analysis-reports")
    first = request(server, "POST", path, body)
    assert first[0] == 200, first[2]
    assert _task(store)["revision"] == 1
    stale = copy.deepcopy(body)
    stale["command_id"] = "cmd-report-stale"
    _reject_without_mutation(server, store, path, stale, status=409, code="REVISION_CONFLICT")
    changed = copy.deepcopy(body)
    changed["report"]["values"][0]["value"] += 1
    _reject_without_mutation(server, store, path, changed, status=409, code="IDEMPOTENCY_CONFLICT")
    # A different command and current revision still cannot rewrite the raw
    # measured report under its old identity. A new report requires a new ID.
    immutable_id = _report_body(store, changed["report"], command_id="cmd-report-identity")
    _reject_without_mutation(server, store, path, immutable_id, status=422, code="ANALYSIS_ID_EXISTS")
    exported, _ = _export(server, store)
    assert exported["analyses"] == [report]


@pytest.mark.parametrize("suffix", ["professional-import", "analysis-reports"])
def test_professional_http_budget_is_enforced_before_any_commit(api, suffix):
    server, store = api
    raw = b'{"padding":"' + b"x" * server.max_command_bytes + b'"}'
    _reject_without_mutation(server, store, _url(store, suffix), raw, status=413, code="LIMIT_EXCEEDED")


@pytest.mark.parametrize("suffix", ["professional-import", "analysis-reports"])
@pytest.mark.parametrize("near_limit", [True, False])
def test_budget_rejection_does_not_wait_for_a_slow_or_huge_body(api, suffix, near_limit):
    server, store = api
    before = store.get_snapshot()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.putrequest("POST", _url(store, suffix))
        connection.putheader("Origin", server.origin)
        connection.putheader("X-TraceAnnotate-Token", server.token)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(server.max_command_bytes + 1 if near_limit else 9999999999))
        # A just-over-limit partial request exercises the bounded drain's
        # *total deadline*. A huge announcement exercises the no-drain cap.
        connection.endheaders(b"{" if near_limit else None)
        started = time.monotonic()
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 413 and payload["error"]["code"] == "LIMIT_EXCEEDED"
        assert time.monotonic() - started < 2
    finally:
        connection.close()
    assert store.get_snapshot() == before
    # Drain timeout/abort must not poison a following authenticated request.
    status, _, after = request(server, "GET", "/api/v1/projects/current/snapshot")
    assert status == 200 and after == before


def test_maximum_export_document_leaves_room_for_the_full_import_command_envelope():
    # The document budget and full HTTP command budget are different. Check
    # the largest declared export body with maximum-size legal identity fields,
    # avoiding the former case where a valid export could not fit its wrapper.
    assert 0 < EXPORT_DOCUMENT_LIMIT < P.MAX_DOCUMENT_BYTES <= MAX_COMMAND_BYTES
    document = {"payload": "x" * (EXPORT_DOCUMENT_LIMIT - len(S.canonical_json({"payload": ""})))}
    assert len(S.canonical_json(document)) == EXPORT_DOCUMENT_LIMIT
    envelope = {"command_id": "c" * 96, "expected_revision": S.MAX_SAFE_INTEGER,
                "expected_sha256": "f" * 64, "document": document}
    assert len(S.canonical_json(envelope)) <= MAX_COMMAND_BYTES


def test_ordinary_annotation_api_cannot_manufacture_imported_provenance(api):
    server, store = api
    path, body = command(store, "cmd-forged-import-source")
    body["payload"]["annotation"]["provenance"] = {
        "origin": "imported", "actor_id": "local-user", "job_id": None}
    _reject_without_mutation(server, store, path, body)
    assert_error(request(server, "GET", "/api/v1/projects/current/commands/cmd-forged-import-source"),
                 404, "COMMAND_UNKNOWN")


@pytest.mark.parametrize("launder", ["origin", "actor"])
def test_imported_annotations_remain_editable_without_laundering_their_source(api, http_video, launder):
    server, store = api
    _seed(server, store)
    doc, _ = _export(server, store)
    destination = store.import_source(http_video)["task_id"]
    path = _url(store, "professional-import", destination)
    imported = request(server, "POST", path, _import_body(store, doc, task_id=destination))
    assert imported[0] == 200, imported[2]
    annotation = _task(store, destination)["annotations"][0]
    provenance = copy.deepcopy(annotation["provenance"])
    label_result = _commit(server, store, "cmd-edit-imported-label", "update_annotation",
                           {"annotation_id": annotation["id"], "changes": {"label": "正常修改导入标签"}},
                           destination)
    assert label_result[0] == 200, label_result[2]
    assert _task(store, destination)["annotations"][0]["provenance"] == provenance
    changed = copy.deepcopy(provenance)
    if launder == "origin":
        changed["origin"] = "manual"
    else:
        changed["actor_id"] = "someone-else"
    task = _task(store, destination)
    body = {"command_id": "cmd-launder-" + launder, "expected_revision": task["revision"],
            "expected_sha256": task["content_sha256"], "operation": "update_annotation",
            "payload": {"annotation_id": annotation["id"], "changes": {"provenance": changed}}}
    _reject_without_mutation(server, store, _url(store, "commands", destination), body)
    assert _task(store, destination)["annotations"][0]["provenance"] == provenance


def _node_import_body(command_body, document_text, mode):
    """Run real Node JSON parsing, using the shipped helper for fixed mode."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Real Node is required for the cross-runtime numeric-token regression")
    source = (Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py").read_text("utf-8")
    start = source.index("function professionalImportBody(")
    end = source.index("async function saveCommand(", start)
    helper = source[start:end]
    program = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
if (data.mode === 'old-parse-stringify') {
  process.stdout.write(JSON.stringify({...data.command, document: JSON.parse(data.documentText)}));
} else {
  const helper = vm.runInNewContext('(' + data.helper + ')');
  process.stdout.write(helper(data.command, data.documentText));
}
"""
    value = {"mode": mode, "helper": helper, "command": command_body, "documentText": document_text}
    result = subprocess.run([node, "-e", program], input=json.dumps(value).encode("utf-8"),
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


def test_real_node_import_preserves_python_float_tokens_and_exact_retry_body(api, http_video, tmp_path):
    server, store = api
    _seed(server, store)
    media = store.get_media(_task(store)["media_id"])
    report = _report(store, ident="report-numeric-transport")
    report["kind"] = "frame_timing"
    report["method"] = {"id": "fixture-numeric-transport", "version": "1",
                        "parameters": {"serialization_only_parameter": 0.125}}
    # Actual declarations of this admitted two-second/10fps test video.
    # Negative zero represents the same zero clock origin; it is retained as
    # a lexical numeric stress case, not claimed to be an acoustic finding.
    assert S.rational_value(media["common_origin"]) == 0
    report["values"] = [
        {"name": "admitted_duration_seconds", "value": float(S.rational_value(media["duration"])),
         "unit": "s", "basis": "declared", "status": "available", "reason": None},
        {"name": "declared_fixture_fps", "value": 10.0, "unit": "fps", "basis": "declared", "status": "available", "reason": None},
        {"name": "common_origin_seconds", "value": -0.0, "unit": "s", "basis": "declared", "status": "available", "reason": None},
    ]
    report["input_digest"] = P.analysis_input_digest(report)
    analysis = request(server, "POST", _url(store, "analysis-reports"), _report_body(store, report))
    assert analysis[0] == 200, analysis[2]
    document, _ = _export(server, store)
    # Use the canonical file the product actually exports. Python's numeric
    # tokens participate in both the frozen task and outer document hashes.
    document_text = S.canonical_json(document).decode("utf-8")
    assert '"value":2.0' in document_text and '"value":-0.0' in document_text
    (tmp_path / "original.professional.json").write_text(document_text, encoding="utf-8")
    destination = store.import_source(http_video)["task_id"]
    target = _task(store, destination)
    command_body = {"command_id": "cmd-node-raw-import", "expected_revision": target["revision"],
                    "expected_sha256": target["content_sha256"]}
    old_raw = _node_import_body(command_body, document_text, "old-parse-stringify")
    (tmp_path / "old-node-request.json").write_bytes(old_raw)
    assert '"value":2.0' not in old_raw.decode("utf-8")
    assert '"value":-0.0' not in old_raw.decode("utf-8")
    path = _url(store, "professional-import", destination)
    _reject_without_mutation(server, store, path, old_raw, status=422, code="PROFESSIONAL_DIGEST_MISMATCH")
    assert_error(request(server, "GET", "/api/v1/projects/current/commands/cmd-node-raw-import"),
                 404, "COMMAND_UNKNOWN")
    new_raw = _node_import_body(command_body, document_text, "shipped-raw-document-helper")
    (tmp_path / "new-node-request.json").write_bytes(new_raw)
    assert document_text.encode("utf-8") in new_raw
    assert '"value":2.0' in new_raw.decode("utf-8") and '"value":-0.0' in new_raw.decode("utf-8")
    saved = request(server, "POST", path, new_raw)
    assert saved[0] == 200, saved[2]
    replay = request(server, "POST", path, new_raw)
    assert replay[0] == 200 and replay[2] == saved[2]
    assert _task(store, destination)["revision"] == 1
    exported, _ = _export(server, store, destination)
    assert S.canonical_json(exported["analyses"]) == S.canonical_json([report])
    assert _archived_document(store, exported["task"]) == document
    status, _, receipt = request(server, "GET", "/api/v1/projects/current/commands/cmd-node-raw-import")
    assert status == 200 and receipt["response"] == saved[2]


def test_shipped_pending_recovery_preserves_the_original_numeric_document_in_real_node(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Real Node is required for the pending-body recovery regression")
    source = (Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py").read_text("utf-8")
    def between(start, end):
        begin = source.index(start)
        return source[begin:source.index(end, begin + len(start))]
    implementations = "\n".join([
        between("function stashPending(", "async function acceptReceipt("),
        between("async function recoverPending(", "function professionalImportBody("),
        between("function professionalImportBody(", "async function saveCommand("),
        between("async function saveCommand(", "const taskUndoStacks="),
    ])
    document_text = '{"numeric_regression_only":true,"values":[2.0,-0.0,0.125],"text":"中文\\n原文"}'
    program = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const storage = new Map(), sends = [], observations = [];
const context = {
  state: {busy:false,pending:null,conflict:false,loadingTask:false,
          task:{id:'tsk-target',revision:0,content_sha256:'f'.repeat(64)},
          media:{id:'med-target'},snapshot:{project_id:'proj-target'},
          selected:null,formBase:{text:'draft'},anchors:{},anchorBase:{},original:null},
  clone: value => JSON.parse(JSON.stringify(value)), currentForm: () => ({text:'draft'}),
  sessionStorage: {setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
  commandId: () => 'cmd-should-not-be-generated', renderSaveState(){}, notice(){}, describeError:e=>e.message,
  json: async (path, options) => {
    if (options?.method==='POST') {
      sends.push(options.body);
      if (sends.length===1) throw Object.assign(new Error('simulated lost transport'), {status:0});
      return {command_id:'cmd-raw-pending',task_id:'tsk-target',applied:true,commit_id:'cmt-target',revision:1};
    }
    throw Object.assign(new Error('no durable receipt yet'), {code:'COMMAND_UNKNOWN',status:404});
  },
  acceptReceipt: async receipt => {
    observations.push(context.state.pending.body);
    assert.equal(receipt.command_id,context.state.pending.command.command_id);
    context.state.pending=null;context.stashPending();return receipt;
  }
};
vm.createContext(context);vm.runInContext(input.implementations,context);
(async()=>{
  const first=await context.saveCommand('import_professional',{}, {
    command_id:'cmd-raw-pending',endpoint:'/tasks/tsk-target/professional-import',
    document:JSON.parse(input.documentText),documentText:input.documentText});
  assert.equal(first,null);assert.ok(context.state.pending);
  const persisted=storage.get('traceannotate-pending:proj-target');assert.ok(persisted);
  const firstBody=context.state.pending.body;
  context.state.pending=JSON.parse(persisted);
  assert.equal(context.state.pending.body,firstBody);
  const recovered=await context.recoverPending({resend:true});
  assert.equal(recovered.revision,1);assert.equal(sends.length,2);
  assert.equal(sends[0],firstBody);assert.equal(sends[1],firstBody);
  assert.equal(observations[0],firstBody);assert.equal(context.state.pending,null);
  assert.equal(storage.size,0);
  process.stdout.write(JSON.stringify({status:'PASS',sendBodies:sends,persistedBody:JSON.parse(persisted).body,
    generatedExtraCommand:false,scope:'shipped functions with explicit transport/storage mocks, not browser interaction'}));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    data = {"implementations": implementations, "documentText": document_text}
    result = subprocess.run([node, "-e", program], input=json.dumps(data).encode("utf-8"),
                            capture_output=True, timeout=10)
    (tmp_path / "pending-node.stdout").write_bytes(result.stdout)
    (tmp_path / "pending-node.stderr").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    outcome = json.loads(result.stdout)
    assert outcome["status"] == "PASS"
    assert outcome["sendBodies"][0] == outcome["sendBodies"][1] == outcome["persistedBody"]
    assert document_text in outcome["persistedBody"]
    assert '2.0,-0.0,0.125' in outcome["persistedBody"]


def test_professional_export_scales_to_102_sequential_annotations_with_audit_index(api):
    server, store = api
    text = ("这是规模回归的测试正文，仅验证保存和导出，不声称描述合成视频。"
            "保留中文、数字-42与1,234、标点和完整说明，历史记录不能重复嵌入整张标注表。")
    for index in range(102):
        # 102 exact, contiguous intervals on a real two-second test clip.
        annotation = {"id": f"ann-scale-{index}", "type": "segment", "label": f"规模片段{index+1}",
                      "text": text, "track": "规模测试", "tags": ["测试数据"],
                      "start": {"seconds": S.rational((index, 51))},
                      "end": {"seconds": S.rational((index+1, 51))}}
        response = _commit(server, store, f"cmd-scale-{index}", "add_annotation", {"annotation": annotation})
        assert response[0] == 200, response[2]
        if index in (57, 101):
            document, _ = _export(server, store)
            assert len(document["task"]["annotations"]) == index+1
            assert document["history"]["scope"] == "available_local_audit_index"
            assert len(document["history"]["revisions"]) == index+2
            assert all("annotations" not in revision for revision in document["history"]["revisions"])
            assert len(S.canonical_json(document)) < EXPORT_DOCUMENT_LIMIT
    assert document["task"] == _task(store)
    assert all(a["text"] == text for a in document["task"]["annotations"])


def test_repeated_eight_roundtrips_keep_source_archives_gc_safe_and_backup_recoverable(api, http_video, tmp_path):
    server, store = api
    _seed(server, store, reviewed=True)
    report = _report(store, scope="interval", subject="ann-exact", ident="report-repeated-interval")
    response = request(server, "POST", _url(store, "analysis-reports"), _report_body(store, report))
    assert response[0] == 200, response[2]
    origin = _task(store)["id"]
    archive_ids = []
    expected_archives = {}
    task_sizes = []
    for index in range(8):
        document, _ = _export(server, store, origin)
        destination = store.import_source(http_video)["task_id"]
        body = _import_body(store, document, command_id=f"cmd-cycle-{index}", task_id=destination)
        response = request(server, "POST", _url(store, "professional-import", destination), body)
        assert response[0] == 200, response[2]
        current = _task(store, destination)
        assert _archived_document(store, current) == document
        descriptor = current["extensions"]["professional"]["import_source"]["document_artifact"]
        archive_ids.append(descriptor["id"])
        expected_archives[descriptor["id"]] = descriptor
        task_sizes.append(len(S.canonical_json(current)))
        assert len(current["annotations"]) == 3
        assert all(a["review_status"] == "unreviewed" for a in current["annotations"])
        assert len(S.canonical_json(current)) < 32 * 1024
        next_doc, _ = _export(server, store, destination)
        assert next_doc["analysis_assessment"][0]["status"] == "fresh"
        origin = destination
    # Archive ownership and source bytes remain available even though the
    # latest current task does not contain recursively duplicated documents.
    unbound_ids = {row["id"] for row in store.sweep_unbound_artifacts(apply=False)["artifacts"]}
    assert set(archive_ids).isdisjoint(unbound_ids)
    store.sweep_unbound_artifacts(apply=True)
    for descriptor in expected_archives.values():
        data = (store.root / descriptor["relative_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == descriptor["sha256"]
    # The parent's server is quiescent apart from read-only requests here.
    backup = tmp_path / "eight-cycle-backup"
    B.backup_workspace(store, backup)
    restored_path = tmp_path / "eight-cycle-restored"
    receipt = B.restore_workspace(backup, restored_path)
    assert receipt["jobs_restarted"] is False and receipt["commands_replayed"] is False
    restored = WorkspaceStore(restored_path)
    restored.open()
    try:
        assert _task(restored, origin) == _task(store, origin)
        for descriptor in expected_archives.values():
            copied = (restored.root / descriptor["relative_path"]).read_bytes()
            assert hashlib.sha256(copied).hexdigest() == descriptor["sha256"]
        with running(restored) as recovered_server:
            recovered_doc, _ = _export(recovered_server, restored, origin)
            assert recovered_doc["task"] == _task(store, origin)
    finally:
        restored.close()


def test_unanchored_import_rejects_report_for_a_different_actual_playback(api, http_video, tmp_path, monkeypatch):
    server, target = api
    # Two real preparations of identical source bytes. Varying the encoder
    # setting models another legitimate proxy generation/version; no database
    # or immutable media object is edited after admission.
    from trace_annotate.temporal import preparation
    original_run = preparation._run
    def different_proxy(command_line, *args, **kwargs):
        changed = list(command_line)
        if "-crf" in changed:
            changed[changed.index("-crf")+1] = "24"
        return original_run(changed, *args, **kwargs)
    source = WorkspaceStore(tmp_path / "different-proxy-source")
    source.create()
    try:
        with monkeypatch.context() as patched:
            patched.setattr(preparation, "_run", different_proxy)
            source.import_source(http_video)
        with running(source) as source_server:
            command_url, command_body = command(source)
            assert request(source_server, "POST", command_url, command_body)[0] == 200
            source_media = source.get_media(_task(source)["media_id"])
            report = _report(source, ident="report-proxy-specific")
            report["time_domain"] = "playback_seconds"
            report["file_sha256"] = source_media["playback"]["sha256"]
            report["input_digest"] = P.analysis_input_digest(report)
            saved = request(source_server, "POST", _url(source, "analysis-reports"), _report_body(source, report))
            assert saved[0] == 200, saved[2]
            document, _ = _export(source_server, source)
        target_media = target.get_media(_task(target)["media_id"])
        assert source_media["source"]["sha256"] == target_media["source"]["sha256"]
        assert source_media["playback"]["sha256"] != target_media["playback"]["sha256"]
        assert not any("anchor" in annotation.get(key, {}) for annotation in document["task"]["annotations"]
                       for key in ("start", "end", "at"))
        body = _import_body(target, document, command_id="cmd-reject-other-proxy-analysis")
        _reject_without_mutation(server, target, _url(target, "professional-import"), body,
                                 status=422, code="MEDIA_IDENTITY_MISMATCH")
        assert _task(target)["revision"] == 0
        assert_error(request(server, "GET", "/api/v1/projects/current/commands/" + body["command_id"]),
                     404, "COMMAND_UNKNOWN")
    finally:
        source.close()
