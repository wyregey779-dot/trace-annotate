"""Semantic candidates through the real store write path: origin, authorship, evidence.

A model may store observations and measurements with registered evidence; it
may never store an interpretation, and accepting it stays a separate command.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from trace_annotate.workspace import semantic_schema as SS
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
