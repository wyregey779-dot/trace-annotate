"""Semantic candidates through the real store write path: origin, authorship, evidence.

A model may store observations and measurements with registered evidence; it
may never store an interpretation, and accepting it stays a separate command.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from trace_annotate.workspace import semantic_schema as SS
from trace_annotate.workspace import professional_schema as PS
from trace_annotate.workspace import vision_schema as VS
from trace_annotate.workspace.errors import SchemaInvalid, WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for workspace fixtures")
    source = tmp_path_factory.mktemp("semantic-source") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
        "-i", "color=navy:s=64x64:r=10:d=1.0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(source),
    ], check=True, capture_output=True, timeout=60)
    return source


@pytest.fixture
def store(tmp_path, clip):
    project = WorkspaceStore(tmp_path / "workspace")
    project.create()
    try:
        project.import_source(clip)
        yield project
    finally:
        project.close()


def _current(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


def _command(store, operation, payload, command_id):
    task = _current(store)
    return store.apply_task_command(command_id=command_id, task_id=task["id"],
                                    expected_revision=task["revision"],
                                    expected_sha256=store.task_digest(task),
                                    operation=operation, payload=payload,
                                    request_body=b"{}")["response"]


def _observable(evidence_id):
    return VS.claim(claim_type="observable", statement="画面中出现蓝色画面块",
                    method_id="scene-stat", method_version="1",
                    parameters={"stat": "luma_median"}, evidence_ids=[evidence_id],
                    start={"n": "0", "d": "1"}, end={"n": "1", "d": "1"},
                    score=0.9, score_kind="rule_score")


def _candidate(annotation_id, claims, evidence_id, job_id="job-semantic-1"):
    return VS.build_annotation(annotation_id=annotation_id, kind="segment",
                               label="镜头候选", text="", track="镜头", claims=claims,
                               tags=["vision"], evidence_ids=[evidence_id],
                               start={"n": "0", "d": "1"}, end={"n": "1", "d": "1"},
                               actor_id="scene-stat", job_id=job_id)


def test_public_candidate_builders_require_and_propagate_declared_model_runs():
    with pytest.raises(SchemaInvalid, match="not declared"):
        VS.claim(claim_type="observable", statement="模型观察", method_id="detector",
                 method_version="1", evidence_ids=["art-e1"], model_run_id="run-one",
                 score=0.8, score_kind="rule_score")
    claim = VS.claim(claim_type="observable", statement="模型观察", method_id="detector",
                     method_version="1", evidence_ids=["art-e1"], model_run_id="run-one",
                     model_run_ids={"run-one"}, score=0.8, score_kind="rule_score")
    annotation = VS.build_annotation(
        annotation_id="ann-with-run", kind="whole_clip", label="模型观察", text="",
        track="vision", claims=[claim], evidence_ids=["art-e1"], job_id="job-one",
        model_run_ids={"run-one"})
    assert VS.validate_candidate(annotation, model_run_ids={"run-one"}) == annotation
    with pytest.raises(SchemaInvalid, match="not declared"):
        VS.validate_candidate(annotation)


def test_model_candidate_stores_observable_with_registered_evidence(store):
    artifact = store._put_artifact(b"frame-bytes-evidence")
    annotation = _candidate("ann-candidate-1", [_observable(artifact["id"])], artifact["id"])
    _command(store, "add_annotation", {"annotation": annotation}, "cmd-candidate-1")
    saved = [a for a in _current(store)["annotations"] if a["id"] == "ann-candidate-1"][0]
    assert saved["provenance"]["origin"] == "model_candidate"
    assert saved["review_status"] == "unreviewed"
    assert saved["evidence_ids"] == [artifact["id"]]
    assert SS.semantic_summary(saved["attributes"]["semantic"])["observable"] == 1
    with pytest.raises(WorkspaceError) as exc:
        _command(store, "review_annotation",
                 {"annotation_id": "ann-candidate-1", "reviewed": True}, "cmd-review-1")
    assert exc.value.code == "SCHEMA_INVALID"


def test_model_candidate_may_not_store_an_interpretation(store):
    artifact = store._put_artifact(b"frame-bytes-evidence-2")
    with pytest.raises(SchemaInvalid) as exc:
        VS.build_annotation(annotation_id="ann-candidate-2", kind="segment",
                            label="镜头候选", text="", track="镜头",
                            claims=[VS.interpretation(statement="人物显得紧张",
                                                      method_id="human-note", method_version="1",
                                                      evidence_ids=[artifact["id"]],
                                                      start={"n": "0", "d": "1"},
                                                      end={"n": "1", "d": "1"})],
                            evidence_ids=[artifact["id"]], start={"n": "0", "d": "1"},
                            end={"n": "1", "d": "1"}, job_id="job-semantic-2")
    assert exc.value.code == "INTERPRETATION_REQUIRES_HUMAN"


def test_professional_document_rejects_model_interpretation_even_if_actor_is_declared_human(store):
    artifact = store._put_artifact(b"frame-bytes-professional-forged-actor")
    annotation = _candidate("ann-forged-actor", [_observable(artifact["id"])], artifact["id"])
    annotation["attributes"]["semantic"]["claims"] = [VS.interpretation(
        statement="Person seems distressed", method_id="human-note", method_version="1",
        evidence_ids=[artifact["id"]], start={"n": "0", "d": "1"},
        end={"n": "1", "d": "1"})]
    task = _current(store)
    media = store.get_media(task["media_id"])
    template = next(row for row in store.list_templates() if row["id"] == task["template_id"])
    with pytest.raises(WorkspaceError) as exc:
        PS.validate_annotation(annotation, media, template, artifact_ids={artifact["id"]},
                               actor_types={annotation["provenance"]["actor_id"]: "human"})
    assert exc.value.code == "INTERPRETATION_REQUIRES_HUMAN"


def test_reviewed_status_without_review_binding_is_rejected(store):
    artifact = store._put_artifact(b"frame-bytes-forged-review")
    annotation = _candidate(
        "ann-forged-review", [_observable(artifact["id"])], artifact["id"])
    annotation["provenance"] = {
        "origin": "manual", "actor_id": "local-user", "job_id": None}
    annotation["review_status"] = "reviewed"
    annotation.pop("review_binding", None)
    with pytest.raises(WorkspaceError) as exc:
        _command(store, "add_annotation", {"annotation": annotation},
                 "cmd-forged-review")
    assert exc.value.code == "SCHEMA_INVALID"


def test_unregistered_evidence_is_refused(store):
    annotation = _candidate("ann-candidate-3",
                            [_observable("art-not-registered")], "art-not-registered")
    with pytest.raises(WorkspaceError) as exc:
        _command(store, "add_annotation", {"annotation": annotation}, "cmd-candidate-3")
    assert exc.value.code == "UNSUPPORTED_PROFESSIONAL_EVIDENCE"


def test_human_annotation_carries_interpretation_and_review_binds_it(store):
    artifact = store._put_artifact(b"frame-bytes-evidence-3")
    annotation = {"id": "ann-human-1", "type": "segment", "label": "manual",
                  "text": "标注", "tags": [], "track": "main",
                  "start": {"seconds": {"n": "0", "d": "1"}},
                  "end": {"seconds": {"n": "1", "d": "1"}},
                  "attributes": {"semantic": {
                      "version": SS.SEMANTIC_VERSION,
                      "claims": [dict(_observable(artifact["id"]),
                                      claim_type="interpretation",
                                      statement="人物显得紧张",
                                      method={"id": "human-note", "version": "1", "parameters": {}},
                                      subject_ref=None, model_run_id=None, score=None,
                                      score_kind="none")]}},
                  "evidence_ids": [artifact["id"]],
                  "provenance": {"origin": "manual", "actor_id": "user", "job_id": None}}
    _command(store, "add_annotation", {"annotation": annotation}, "cmd-human-1")
    _command(store, "review_annotation",
             {"annotation_id": "ann-human-1", "reviewed": True}, "cmd-human-review")
    saved = [a for a in _current(store)["annotations"] if a["id"] == "ann-human-1"][0]
    assert saved["review_status"] == "reviewed"
    assert saved["review_binding"]["content_sha256"] == store.annotation_review_digest(saved, _current(store))
    _command(store, "update_annotation",
             {"annotation_id": "ann-human-1", "changes": {"text": "改写后的说明"}}, "cmd-human-edit")
    edited = [a for a in _current(store)["annotations"] if a["id"] == "ann-human-1"][0]
    assert edited["review_status"] == "unreviewed"
    assert "review_binding" not in edited


def test_v3_object_and_recipient_edits_invalidate_review_binding(store):
    artifact = store._put_artifact(b"frame-bytes-evidence-v3-refs")
    claim = dict(_observable(artifact["id"]), object_ref="object-1",
                 recipient_ref="recipient-1")
    annotation = _candidate("ann-v3-refs", [claim], artifact["id"])
    annotation["provenance"] = {"origin": "manual", "actor_id": "local-user", "job_id": None}
    _command(store, "add_annotation", {"annotation": annotation}, "cmd-v3-add")
    _command(store, "review_annotation", {"annotation_id": "ann-v3-refs", "reviewed": True},
             "cmd-v3-review")
    reviewed = next(a for a in _current(store)["annotations"] if a["id"] == "ann-v3-refs")
    before = reviewed["review_binding"]["content_sha256"]
    changed = dict(reviewed["attributes"]["semantic"])
    changed["claims"] = [dict(reviewed["attributes"]["semantic"]["claims"][0],
                              recipient_ref="recipient-2")]
    _command(store, "update_annotation", {"annotation_id": "ann-v3-refs",
             "changes": {"attributes": {"semantic": changed}}}, "cmd-v3-edit-ref")
    edited = next(a for a in _current(store)["annotations"] if a["id"] == "ann-v3-refs")
    assert edited["review_status"] == "unreviewed" and "review_binding" not in edited
    assert store.annotation_review_digest(edited, _current(store)) != before


def test_generic_delete_cannot_bypass_candidate_decision(store):
    artifact = store._put_artifact(b"frame-bytes-evidence-v3-reject")
    annotation = _candidate("ann-v3-reject", [_observable(artifact["id"])], artifact["id"])
    _command(store, "add_annotation", {"annotation": annotation}, "cmd-v3-reject-add")
    with pytest.raises(WorkspaceError) as exc:
        _command(store, "delete_annotation", {"annotation_id": "ann-v3-reject"},
                 "cmd-v3-reject-delete")
    assert exc.value.code == "CANDIDATE_DECISION_REQUIRED"
    assert any(a["id"] == "ann-v3-reject" for a in _current(store)["annotations"])
