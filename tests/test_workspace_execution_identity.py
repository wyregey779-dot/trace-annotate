"""Frozen implementation and local-resource identity for vision jobs."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from trace_annotate.workspace import model_worker as M, vision_jobs as V
from trace_annotate.workspace.errors import WorkspaceError


def test_implementation_manifest_detects_same_size_replacement_with_restored_mtime(tmp_path):
    source = tmp_path / "extractors" / "actions.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"A" * 64)
    before_stat = source.stat()
    before = V._implementation_manifest(tmp_path, ("extractors/actions.py",))

    source.write_bytes(b"B" * 64)
    os.utime(source, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    after = V._implementation_manifest(tmp_path, ("extractors/actions.py",))

    assert before["sha256"] != after["sha256"]
    assert before["files"][0]["size_bytes"] == after["files"][0]["size_bytes"]


@pytest.mark.parametrize("relative", ("temporal/preparation.py", "core/hasher.py",
                                       "workspace/slots.py", "config.py",
                                       "extractors/base.py", "extractors/hands_worker.py"))
def test_default_execution_manifest_binds_transitive_runtime_dependencies(tmp_path, relative):
    """A helper used below vision_jobs must change the frozen job identity."""
    assert relative in V.EXECUTION_SOURCE_FILES
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"A" * 64)
    before = V._implementation_manifest(tmp_path)
    source.write_bytes(b"B" * 64)
    after = V._implementation_manifest(tmp_path)
    assert before["sha256"] != after["sha256"]
    before_row = next(row for row in before["files"] if row["path"] == relative)
    after_row = next(row for row in after["files"] if row["path"] == relative)
    assert before_row["sha256"] != after_row["sha256"]


def test_job_identity_changes_with_execution_deployment_bytes(tmp_path):
    source = tmp_path / "workspace" / "vision_jobs.py"
    source.parent.mkdir(parents=True)
    source.write_text("first", encoding="utf-8")
    first = V._execution_deployment(
        package_root=tmp_path, ffmpeg_path=str(source), ffprobe_path=str(source))
    source.write_text("other", encoding="utf-8")
    second = V._execution_deployment(
        package_root=tmp_path, ffmpeg_path=str(source), ffprobe_path=str(source))

    first_identity = V._job_identity(
        include_optional=False, include_model_resources=False,
        include_transition_discovery=False, execution_deployment=first)
    second_identity = V._job_identity(
        include_optional=False, include_model_resources=False,
        include_transition_discovery=False, execution_deployment=second)
    assert first_identity["config_sha256"] != second_identity["config_sha256"]


@pytest.mark.parametrize("relative", ("temporal/preparation.py", "core/hasher.py"))
def test_transitive_dependency_bytes_change_job_config_sha256(tmp_path, relative):
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"A" * 64)
    first = V._execution_deployment(
        package_root=tmp_path, ffmpeg_path=sys.executable, ffprobe_path=sys.executable)
    source.write_bytes(b"B" * 64)
    second = V._execution_deployment(
        package_root=tmp_path, ffmpeg_path=sys.executable, ffprobe_path=sys.executable)
    first_identity = V._job_identity(
        include_optional=False, include_model_resources=False,
        include_transition_discovery=False, execution_deployment=first)
    second_identity = V._job_identity(
        include_optional=False, include_model_resources=False,
        include_transition_discovery=False, execution_deployment=second)
    assert first_identity["config_sha256"] != second_identity["config_sha256"]


@pytest.mark.parametrize("relative", ("core/hasher.py", "config.py",
                                       "extractors/base.py", "workspace/slots.py"))
def test_model_worker_identity_binds_transitive_runtime_dependencies(tmp_path, relative):
    assert relative in M.MODEL_IMPLEMENTATION_SOURCE_FILES
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"A" * 64)
    before = M._implementation_identity(tmp_path, (relative,))
    source.write_bytes(b"B" * 64)
    after = M._implementation_identity(tmp_path, (relative,))
    assert before != after


def test_loaded_face_must_match_the_frozen_bytes():
    extractors = {
        "face": SimpleNamespace(code_identity={
            "detector_kind": "yunet", "weights_sha256": "b" * 64}),
        "ocr": SimpleNamespace(code_identity={"weights_files": []}),
    }
    deployment = {"face": {"status": "available", "sha256": "a" * 64},
                  "ocr": {"files": []}}
    with pytest.raises(WorkspaceError) as failure:
        V._validate_loaded_deployment(extractors, deployment)
    assert failure.value.code == "VISION_RESUBMIT_REQUIRED"


def test_loaded_ocr_must_match_every_frozen_role():
    extractors = {
        "face": SimpleNamespace(code_identity={"detector_kind": None}),
        "ocr": SimpleNamespace(code_identity={
            "weights_files": [{"role": "det", "sha256": "b" * 64}]}),
    }
    deployment = {"face": {"status": "unavailable"},
                  "ocr": {"files": [{"role": "det", "status": "available",
                                      "sha256": "a" * 64}]}}
    with pytest.raises(WorkspaceError) as failure:
        V._validate_loaded_deployment(extractors, deployment)
    assert failure.value.code == "VISION_RESUBMIT_REQUIRED"


def test_exact_loaded_resource_identity_is_accepted():
    digest = "a" * 64
    extractors = {
        "face": SimpleNamespace(code_identity={
            "detector_kind": "yunet", "weights_sha256": digest}),
        "ocr": SimpleNamespace(code_identity={
            "weights_files": [{"role": role, "sha256": digest}
                              for role in ("det", "cls", "rec")]}),
    }
    deployment = {"face": {"status": "available", "sha256": digest},
                  "ocr": {"files": [{"role": role, "status": "available",
                                      "sha256": digest}
                                     for role in ("det", "cls", "rec")]}}
    V._validate_loaded_deployment(extractors, deployment)
