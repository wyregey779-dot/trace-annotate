"""Optional providers keep configuration, geometry and candidate evidence honest."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from trace_annotate.workspace import optional_vision as O
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.errors import WorkspaceError
from tests.test_workspace_vision_jobs import clip, store, task, submit


def frame(index, seconds, boxes, scene=0):
    return {"scene_index": scene, "frame_index": str(index),
            "timestamp": {"n": str(seconds), "d": "1"},
            "person": {"status": "success", "persons_detected": [
                {"box": box, "confidence": .9} for box in boxes]}}


def test_model_bytes_and_worker_identity_are_frozen(tmp_path):
    model = tmp_path / "person.onnx"
    model.write_bytes(b"first")
    first = O.provider_config(person_model=str(model))
    assert first["person"]["weights_sha256"]
    assert first["implementation_sha256"]
    model.write_bytes(b"second")
    second = O.provider_config(person_model=str(model))
    assert first != second
    assert V._job_identity(optional_providers=first) != V._job_identity(optional_providers=second)
    O.validate_config(first)
    invalid = copy.deepcopy(first)
    invalid["person"]["weights_sha256"] = "fake"
    with pytest.raises(WorkspaceError):
        O.validate_config(invalid)


def test_person_geometry_preferred_to_face_and_preserves_xyxy():
    image = np.zeros((100, 200, 3), np.uint8)
    result = V._shot_framing(image, {"status": "success", "faces_detected": [
        {"box": [90, 30, 20, 20]}]}, {"status": "success", "persons_detected": [
        {"box": [60, 10, 140, 90]}]})
    assert result["subject_kind"] == "person"
    assert result["measurements"][0]["box"] == [60, 10, 140, 90]
    assert result["measurements"][0]["height_ratio"] == .8


def test_multiple_people_associate_once_each_and_no_cross_scene_or_long_gap():
    samples = [frame(0, 0, [[0, 0, 20, 50], [50, 0, 70, 50]]),
               frame(1, 1, [[51, 0, 71, 50], [1, 0, 21, 50]]),
               frame(2, 2, [[52, 0, 72, 50], [2, 0, 22, 50]]),
               frame(90, 90, [[2, 0, 22, 50]]),
               frame(91, 91, [[2, 0, 22, 50]], scene=1)]
    tracks = V._subject_tracks(samples)
    assert [len(t["points"]) for t in tracks] == [3, 3, 1, 1]
    assert all(t["kind"] == "person" for t in tracks)
    assert tracks[0]["points"][1]["box"][0] == 1
    assert tracks[1]["points"][1]["box"][0] == 51


def test_missing_landmarks_are_not_a_successful_empty_prediction(tmp_path):
    result = O.run_landmarks(O.provider_config(), [], tmp_path, cancelled=lambda: False)
    assert result["status"] == "not_run"
    assert result["reason"] == "landmark_weights_unavailable"


@pytest.mark.parametrize("faces", [[], [{"box": [1, 1, 10, 10], "landmarks": {"mouth": []}}]])
def test_landmark_success_does_not_upgrade_unusable_expression(faces):
    measured = {"status": "success", "faces_detected": faces}
    result = V._expressive_result(measured, {"frame_index": "7", "evidence_sha256": "a" * 64})
    assert measured["status"] == "success"
    assert result["status"] == "not_run" and result["reason"] and result["cues"] == []


def test_unverified_onnx_cannot_inherit_official_model_license(tmp_path):
    path = tmp_path / "local.onnx"
    path.write_bytes(b"not-the-official-model")
    config = O.provider_config(person_model=str(path))
    job = {"id": "job-provider-license", "parameters": {
        "job_identity": V._job_identity(optional_providers=config)}}
    runs = V._build_model_runs(job, {}, [], {"status": "success", "scenes": []}, None, [], {}, "0" * 64, None)
    person = next(run for run in runs if run["id"] == V._module_run_id(job["id"], "person"))
    assert person["license"] == "unverified local weights"
    assert person["model_name"] == "unverified_local_onnx"


def test_worker_cancellation_terminates_owned_process(tmp_path):
    marker = tmp_path / "started"
    script = tmp_path / "wait.py"
    script.write_text("import pathlib,sys,time\npathlib.Path(sys.argv[1]).write_text('ready')\ntime.sleep(60)\n")
    start = time.monotonic()
    with pytest.raises(WorkspaceError, match="cancelled"):
        O.run_worker([sys.executable, "-I", "-B", str(script), str(marker)],
                     tmp_path / "out.log", cancelled=lambda: marker.exists(), timeout=5)
    assert time.monotonic() - start < 5


def test_worker_timeout_is_bounded(tmp_path):
    with pytest.raises(TimeoutError):
        O.run_worker([sys.executable, "-I", "-B", "-c", "import time;time.sleep(60)"],
                     tmp_path / "timeout.log", cancelled=lambda: False, timeout=.15)


def test_provider_pipeline_links_real_decoded_evidence_and_never_reviews(store, monkeypatch):
    """Deterministic adapters test wiring, not natural-image model accuracy."""
    class Person:
        def detect_people(self, image):
            return {"status": "success", "person_count": 1,
                    "persons_detected": [{"box": [10, 5, 50, 59], "confidence": .9}]}

    def landmarks(config, images, work, *, cancelled):
        return {"status": "success", "results": [
            {"status": "success", "frame_index": item["frame_index"],
             "faces_detected": [{"box": [4, 3, 60, 62], "frame_size": [64, 64],
                                 "landmarks": {"mouth": [[20, 30], [40, 30], [30, 25], [30, 35]]}}]}
            for item in images]}

    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    monkeypatch.setattr(O, "run_landmarks", landmarks)
    def load(thresholds):
        jobs._extractors = {"face": None, "ocr": None, "clip": None, "vlm": None, "person": Person()}
    monkeypatch.setattr(jobs, "_load_extractors", load)
    before = task(store)
    try:
        queued = submit(jobs, command_id="optional-wiring")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        result = jobs.result(queued["id"])
        assert all(result["module_status"][name] == "success" for name in ("person", "landmarks", "expressive"))
        assert result["module_status"]["actions_unsupported"] == ["拿起", "放下", "打开", "关闭", "递交"]
        assert all(item["framing"]["subject_kind"] == "person" for item in result["keyframes"])
        for module in ("person", "landmarks", "expressive"):
            run = next(item for item in result["model_runs"] if item["id"] == V._module_run_id(queued["id"], module))
            assert run["status"] == "success" and run["output_sha256"]
        assert task(store) == before
        jobs.apply(queued["id"], command_id="optional-apply", expected_revision=before["revision"],
                   expected_sha256=before["content_sha256"], request_body=b"{}")
        annotations = task(store)["annotations"]
        assert annotations and all(a["review_status"] == "unreviewed" for a in annotations)
        claims = [c for a in annotations for c in a["attributes"]["semantic"]["claims"]]
        for method in ("person-box-local", "expressive-cues-v1", "shot-framing-local"):
            assert any(c["method"]["id"] == method and c["evidence_ids"] and c["model_run_id"] for c in claims)
    finally:
        jobs.close()


def test_pre_optional_frozen_job_is_valid_but_requires_resubmit(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    current = task(store)
    parameters = dict(V._parameters(), job_identity=V._job_identity(include_optional=False))
    try:
        batch = jobs.queue.create_batch([{"task_id": current["id"], "expected_revision": current["revision"],
                                         "expected_sha256": current["content_sha256"]}], operation=V.OPERATION,
                                       processor_version=V.VERSION, parameters=parameters, command_id="pre-optional")
        assert V._validate_job(batch["jobs"][0]) == parameters
        failed = jobs.run_pending(batch_id=batch["id"])[0]
        assert failed["state"] == "failed" and failed["error_code"] == "VISION_RESUBMIT_REQUIRED"
    finally:
        jobs.close()
