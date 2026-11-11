"""Durable vision candidate jobs: scene cuts, keyframes, honest extractor status,
candidate-only imports and the human upgrade path.

The actual extractors (scene/face/ocr/clip) follow their own real behavior; this
suite exercises the durable queue boundary, the candidate contract and the
review discipline with a real ffmpeg fixture so scene detection runs for real.
"""
from __future__ import annotations

import json
import copy
import base64
import shutil
import subprocess
import threading

import pytest
import numpy as np

from trace_annotate.workspace import professional_store as PStore
from trace_annotate.workspace import schema as S
from trace_annotate.workspace import semantic_schema as SS
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.errors import Conflict, SchemaInvalid, WorkspaceError
from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for vision fixtures")
    source = tmp_path_factory.mktemp("vision-jobs-source") / "source.mp4"
    # Two distinct scene blocks (blue, then red) so scene detection has a real cut.
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=blue:s=64x64:r=10:d=0.5",
        "-f", "lavfi", "-i", "color=red:s=64x64:r=10:d=0.5",
        "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=1.0",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
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


@pytest.fixture(scope="module")
def content_clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for vision fixtures")
    source = tmp_path_factory.mktemp("vision-content-source") / "content.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=green:s=64x64:r=10:d=1.0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True, capture_output=True, timeout=60)
    return source


@pytest.fixture(scope="module")
def long_content_clip(tmp_path_factory):
    """One 1000-frame shot: the fixed 26-frame content budget must not apply."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for vision fixtures")
    source = tmp_path_factory.mktemp("vision-long-content-source") / "long-content.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=green:s=64x64:r=10:d=100.0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True, capture_output=True, timeout=120)
    return source


@pytest.fixture(scope="module")
def eight_frame_clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for vision fixtures")
    source = tmp_path_factory.mktemp("vision-eight-source") / "eight.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=blue:s=64x64:r=10:d=0.8",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True, capture_output=True, timeout=60)
    return source


@pytest.fixture
def content_store(tmp_path, content_clip):
    project = WorkspaceStore(tmp_path / "content-workspace")
    project.create()
    try:
        project.import_source(content_clip)
        yield project
    finally:
        project.close()


@pytest.fixture(scope="module")
def continuous_content_clip(tmp_path_factory):
    """Controlled fixture: a textured panel enters one continuous shot, no cut.

    This exercises real video decoding/detection, not natural-material T06
    acceptance. The underlying scene is retained outside the moving panel.
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for vision fixtures")
    source = tmp_path_factory.mktemp("vision-continuous-source") / "continuous.mp4"
    y, x = np.indices((64, 64))
    checker = np.where((x // 16 + y // 16) % 2 == 0, 200, 60).astype(np.uint8)
    base = np.repeat(checker[..., None], 3, axis=2)
    stripes = np.where((x // 2 % 2)[..., None] == 0,
                       np.array([230, 180, 20]), np.array([40, 30, 10])).astype(np.uint8)
    frames = []
    for index in range(40):
        frame = base.copy()
        width = min(64, max(0, (index - 7) * 4))
        frame[:, :width] = stripes[:, :width]
        frames.append(frame.tobytes())
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "64x64", "-r", "10", "-i", "pipe:0", "-an",
        "-vf", "setsar=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], input=b"".join(frames), check=True, capture_output=True, timeout=60)
    return source


def task(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


def submit(jobs, *, command_id="vision-first", **kwargs):
    current = task(jobs.store)
    return jobs.submit(current["id"], command_id=command_id, expected_revision=current["revision"],
                       expected_sha256=jobs.store.task_digest(current), **kwargs)


def fail_code(code, call):
    with pytest.raises(WorkspaceError) as exc:
        call()
    assert exc.value.code == code


def test_content_span_handles_single_boundary_index():
    assert V._content_span([42]) == 1
    assert V._content_span([2, 10]) == 9


def test_content_overlap_rejects_empty_or_clipped_window():
    assert V._content_overlap({"n": "5", "d": "1"}, {"n": "5", "d": "1"}) is False
    assert V._content_overlap({"n": "6", "d": "1"}, {"n": "5", "d": "1"}) is False
    assert V._content_overlap({"n": "5", "d": "1"}, {"n": "6", "d": "1"}) is True


def test_content_clip_keeps_scene_inside_playback_duration():
    start, end = V._content_clip({"n": "121", "d": "1"},
                                 {"n": "130", "d": "1"},
                                 {"n": "120", "d": "1"})
    assert S.rational_value(start) == 120
    assert S.rational_value(end) == 120


def test_transition_claim_uses_event_interval_not_whole_scene_window():
    candidate = {"job_id": "job-time", "processor_version": "1", "keyframes": [],
                 "boundary_frames": [{"frame_index": 99, "seconds": 9.9},
                                     {"frame_index": 100, "seconds": 10.0}]}
    scene = {"start": "0", "end": "20"}
    event = {"kind": "cut", "anchor_frame": 100, "previous_frame": 99,
             "confidence": 0.9, "metrics": {}}
    body = V._candidate_claims(candidate, 0, scene, {"n": "0", "d": "1"},
                               {"n": "20", "d": "1"},
                               {"frame_index": 90, "source_pts": "9.0",
                                "boundary_events": [event]})
    transition = next(claim for claim in body["claims"]
                      if claim["method"]["id"] == "shot-boundaries-v1")
    assert transition["claim_type"] == "observable"
    assert transition["time_window"]["start"]["seconds"] == {"n": "99", "d": "10"}
    assert transition["time_window"]["end"]["seconds"] == {"n": "10", "d": "1"}
    assert transition["measurement"] is None


def test_vision_job_is_hash_bound_candidate_and_does_not_edit_task(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    initial = task(store)
    queued = submit(jobs)["jobs"][0]
    assert queued["parameters"]["time_domain"] == "playback_seconds"
    identity = queued["parameters"]["job_identity"]
    assert identity["pipeline"] == "vision-assist-v1"
    assert len(identity["config_sha256"]) == 64
    assert identity["config"]["sampling"]["max_frames"] == V.MAX_KEYFRAMES
    fail_code("VISION_NOT_READY", lambda: jobs.result(queued["id"]))
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded"
    result = jobs.result(queued["id"])
    assert result["file_sha256"] == store.get_media(initial["media_id"])["playback"]["sha256"]
    assert result["scope"] == "registered_playback_only; original_source_not_measured; candidate_not_applied"
    assert result["semantic_listening"] == "not_performed"
    assert result["extractor_status"]["scene"] in {"success", "failed"}
    assert result["module_status"]["actions"] in {"success", "not_run", "failed"}
    assert result["module_status"]["acoustics"] in {"success", "not_run", "failed"}
    assert result["module_status"]["boundary"] in {"success", "failed", "not_run"}
    assert "content_reason" in result["module_status"]
    content = result["content_sampling"]
    assert content["status"] == "insufficient_samples"
    assert content["required_samples"] == 9
    assert content["insufficient_scenes"] == len(content["scene_results"])
    assert all(item["provided_samples"] < 9 for item in content["scene_results"])
    assert all(item["message"] and "9" in item["message"]
               for item in content["scene_results"])
    runs = result["model_runs"]
    modules = {"face", "ocr", "clip", "vlm", "scene", "motion", "content",
               "actions", "acoustics", "boundary", "person", "landmarks", "expressive"}
    run_ids = {run["id"] for run in runs}
    assert len(run_ids) == len(runs) == len(modules)
    assert run_ids == {V._module_run_id(queued["id"], name) for name in modules}
    assert all(claim["model_run_id"] is None or claim["model_run_id"] in run_ids
               for annotation in V._candidate_annotations(result)
               for claim in annotation["attributes"]["semantic"]["claims"])
    content_run = next(run for run in runs if run["id"] == V._module_run_id(queued["id"], "content"))
    assert (content_run["provider"], content_run["model_name"]) == ("trace-annotate", "scene-change-v1")
    assert content_run["status"] == "failed"
    assert "insufficient_samples" in content_run["failure_reason"]
    for run in runs:
        assert SS.validate_model_run(run) == run
    for module in ("person", "landmarks", "expressive"):
        optional_run = next(run for run in runs if run["id"] == V._module_run_id(queued["id"], module))
        assert optional_run["status"] == "not_run"
        assert optional_run["failure_reason"]
        assert optional_run["output_sha256"] is None
    assert any(run["status"] == "not_run" and run["failure_reason"] for run in runs)
    assert all(run["status"] == "success" or run["failure_reason"] for run in runs)
    assert len(result["scenes"]) >= 1
    assert all({"start", "end", "duration"} <= set(scene) for scene in result["scenes"])
    assert result["evidence"] and all(item["relative_path"].startswith("tmp/vision/jobs/") for item in result["evidence"])
    assert str(store.root) not in json.dumps(result)
    assert len(S.canonical_json(result)) < V.MAX_RESULT_BYTES
    assert len(result["content_frames"]) <= V.MAX_CONTENT_FRAMES
    assert task(store) == initial
    assert jobs.close()["stopped"]


def test_boundary_budget_is_independent_of_keyframe_budget():
    assert V.BOUNDARY_MAX_FRAMES == V.MAX_KEYFRAMES * 4


def test_content_sampling_returns_at_least_nine_contiguous_frames_for_a_ten_frame_scene(
        content_store):
    jobs = V.VisionJobs(content_store,
                        clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    queued = submit(jobs, command_id="vision-content")["jobs"][0]
    assert jobs.run_pending()[0]["state"] == "succeeded"
    result = jobs.result(queued["id"])
    content = result["content_sampling"]
    assert content["status"] == "success"
    assert content["required_samples"] == 9
    assert content["insufficient_scenes"] == 0
    assert content["scene_results"]
    assert all(item["provided_samples"] >= 9 for item in content["scene_results"])
    assert all(item["max_gap_frames"] <= 1 for item in content["scene_results"])
    assert result["keyframes"]
    assert all(keyframe["content_sampling"]["provided_samples"] >= 9
               for keyframe in result["keyframes"])
    jobs.close()


def test_long_shot_content_sampling_scales_beyond_the_short_scene_budget(
        tmp_path, long_content_clip):
    project = WorkspaceStore(tmp_path / "long-content-workspace")
    project.create()
    try:
        project.import_source(long_content_clip)
        jobs = V.VisionJobs(project,
                            clip_model_path="nonexistent-clip-weights",
                            vlm_model_path="nonexistent-vlm-weights")
        queued = submit(jobs, command_id="vision-content-long")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        result = jobs.result(queued["id"])
        content = result["content_sampling"]
        assert content["status"] == "success"
        assert content["scene_results"]
        assert content["returned"] > 26
        span = max(1, int(round((result["scenes"][0]["end"]
                                 - result["scenes"][0]["start"]) * 10)))
        expected_gap = max(1, -(-span // 32))
        assert content["max_gap_frames"] <= max(2, expected_gap + 1)
        jobs.close()
    finally:
        project.close()


def test_job_status_is_read_only_and_reports_progress(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    queued = submit(jobs)["jobs"][0]
    before = task(store)
    status = jobs.job_status(queued["id"])
    assert status["state"] == "queued"
    assert status["task_id"] == before["id"]
    assert status["attempt"] == 0
    assert status["job_identity"]["pipeline"] == "vision-assist-v1"
    assert status["has_candidate"] is False
    done = jobs.run_pending()[0]
    status = jobs.job_status(queued["id"])
    assert status["state"] == "succeeded"
    assert status["attempt"] == 1
    assert status["has_candidate"] is True
    assert status["error_code"] is None
    assert task(store) == before
    fail_code("JOB_NOT_FOUND", lambda: jobs.job_status("job-does-not-exist"))
    jobs.close()


def test_configuration_change_creates_a_new_frozen_job_identity(store, monkeypatch):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    first = submit(jobs, command_id="identity-one")["jobs"][0]
    old_sha = first["parameters"]["job_identity"]["config_sha256"]
    monkeypatch.setattr(V, "BOUNDARY_RADIUS", 9)
    second = submit(jobs, command_id="identity-two")["jobs"][0]
    new_sha = second["parameters"]["job_identity"]["config_sha256"]
    assert new_sha != old_sha
    assert first["idempotency_key"] != second["idempotency_key"]
    assert second["parameters"]["job_identity"]["config"]["sampling"]["boundary_radius"] == 9
    jobs.close()


def test_vision_result_requires_success_and_correct_operation(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    other = DurableQueue(store)
    batch = other.create_batch(items=[{"task_id": task(store)["id"],
                                       "expected_revision": task(store)["revision"],
                                       "expected_sha256": store.task_digest(task(store))}],
                               operation="not-vision", processor_version="1", parameters={},
                               command_id="other-op")
    fail_code("UNSUPPORTED_PROCESSOR", lambda: jobs.result(batch["jobs"][0]["id"]))
    fail_code("JOB_NOT_FOUND", lambda: jobs.result("job-does-not-exist"))
    jobs.close()


def test_submit_requires_reduced_interval_and_known_subject(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    current = task(store)
    base = dict(command_id="receipt", expected_revision=current["revision"],
                expected_sha256=store.task_digest(current))
    fail_code("SCHEMA_INVALID", lambda: jobs.submit(current["id"], **base, window={"start": {"n": "0", "d": "1"}, "end": {"n": "2", "d": "2"}}))
    fail_code("SCHEMA_INVALID", lambda: jobs.submit(current["id"], **base, window={"start": {"n": "1", "d": "2"}, "end": {"n": "1", "d": "2"}}))
    fail_code("SCHEMA_INVALID", lambda: jobs.submit(current["id"], **base, window={"start": {"n": "0", "d": "1"}, "end": {"n": "1", "d": "1"}}, subject_annotation_id="ann-missing"))
    jobs.close()


def test_submit_is_idempotent_after_task_edit_and_shutdown(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    current = task(store)
    arguments = dict(command_id="receipt", expected_revision=current["revision"], expected_sha256=store.task_digest(current))
    first = jobs.submit(current["id"], **arguments)
    with store.transaction():
        result = store.apply_task_command(
            command_id="edit-after-submit", task_id=current["id"], expected_revision=current["revision"],
            expected_sha256=store.task_digest(current), operation="add_annotation",
            payload={"annotation": {"id": "ann-touch", "type": "segment", "label": "touched",
                                    "start": {"seconds": {"n": "0", "d": "1"}}, "end": {"seconds": {"n": "4", "d": "5"}}}},
            request_body=b'{}')["response"]
    jobs.close()
    assert jobs.submit(current["id"], **arguments) == first
    fail_code("WORKSPACE_CLOSING", lambda: jobs.submit(current["id"], command_id="new-after-close",
                                                       expected_revision=current["revision"],
                                                       expected_sha256=store.task_digest(current)))
    fail_code("IDEMPOTENCY_CONFLICT", lambda: jobs.submit(current["id"], command_id="receipt",
                                                          expected_revision=current["revision"],
                                                          expected_sha256=store.task_digest(current),
                                                          window={"start": {"n": "0", "d": "1"}, "end": {"n": "1", "d": "1"}}))
    fail_code("IDEMPOTENCY_CONFLICT", lambda: submit(jobs, command_id="receipt"))
    jobs.close()


def test_apply_imports_candidates_as_model_candidate_and_refuses_reviewed(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    current = task(store)
    queued = submit(jobs, command_id="vision-apply")["jobs"][0]
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded"
    result = jobs.result(queued["id"])
    annotations = V._candidate_annotations(result)
    assert annotations and all(a["provenance"]["origin"] == "model_candidate" for a in annotations)
    assert all(a["provenance"]["job_id"] == queued["id"] for a in annotations)
    assert all(a["review_status"] == "unreviewed" for a in annotations)
    body = json.dumps({"candidates": annotations, "source_job_id": queued["id"]}).encode("utf-8")
    applied = jobs.apply(queued["id"], command_id="apply-cmd",
                         expected_revision=current["revision"], expected_sha256=store.task_digest(current),
                         request_body=body)
    assert applied["candidate_count"] == len(annotations)
    after = task(store)
    assert len(after["annotations"]) == len(annotations)
    assert all(a["review_status"] == "unreviewed" for a in after["annotations"])
    for annotation in after["annotations"]:
        semantic = annotation["attributes"]["semantic"]
        assert semantic["version"] == "visual-semantics-v2"
        assert semantic["claims"]
        assert all(claim["claim_type"] in {"observable", "measured"}
                   for claim in semantic["claims"])
        assert annotation["evidence_ids"]
        for claim in semantic["claims"]:
            if claim["method"]["id"] == "keyframe-sampling-local":
                assert claim["evidence_ids"]
            elif claim["method"]["id"] == "shot-boundaries-v1":
                assert claim["score"] is not None
                assert len(claim["evidence_ids"]) >= 2
            elif claim["method"]["id"] == "shot-boundary-local":
                assert claim["evidence_ids"] == []
    # Review attempt on a model candidate is refused by the store path.
    for annotation in after["annotations"]:
        fail_code("SCHEMA_INVALID", lambda a=annotation: store.apply_task_command(
            command_id="review-candidate", task_id=after["id"], expected_revision=after["revision"],
            expected_sha256=store.task_digest(after), operation="review_annotation",
            payload={"annotation_id": a["id"], "reviewed": True}, request_body=b'{}'))
    jobs.close()


def test_candidate_semantics_survive_professional_export_import(store, clip, tmp_path):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    current = task(store)
    queued = submit(jobs, command_id="vision-roundtrip")["jobs"][0]
    jobs.run_pending()
    body = json.dumps({"candidates": V._candidate_annotations(jobs.result(queued["id"])),
                       "source_job_id": queued["id"]}).encode("utf-8")
    jobs.apply(queued["id"], command_id="apply-roundtrip", expected_revision=current["revision"],
               expected_sha256=store.task_digest(current), request_body=body)
    source_task = task(store)
    document = json.loads(PStore.export_professional(store, source_task["id"])["data"])
    destination = WorkspaceStore(tmp_path / "roundtrip-workspace")
    destination.create()
    try:
        destination.import_source(clip)
        target = destination.get_task(destination.list_tasks()[0]["task_id"])
        imported = PStore.apply_import(
            destination, target["id"], command_id="import-roundtrip",
            expected_revision=target["revision"], expected_sha256=destination.task_digest(target),
            document=document, request_body=b"{}")
        after = destination.get_task(target["id"])
        assert imported["imported_count"] == len(source_task["annotations"])
        assert len(after["annotations"]) == len(source_task["annotations"])
        for original, restored in zip(source_task["annotations"], after["annotations"]):
            assert restored["legacy_id"] == original["id"]
            assert restored["attributes"] == original["attributes"]
            assert restored["tags"] == original["tags"]
            assert restored["review_status"] == "unreviewed"
            assert restored["provenance"]["origin"] == "imported"
    finally:
        destination.close()
        jobs.close()


def test_detected_content_candidate_evidence_review_and_roundtrip(tmp_path, continuous_content_clip):
    """Real normal-entry pipeline on a controlled continuous-shot fixture."""
    project = WorkspaceStore(tmp_path / "content-chain")
    project.create()
    jobs = None
    destination = None
    try:
        admitted = project.import_source(continuous_content_clip)
        task_id = admitted["task_id"]
        jobs = V.VisionJobs(project, clip_model_path="nonexistent-clip-weights",
                            vlm_model_path="nonexistent-vlm-weights")
        queued = submit(jobs, command_id="continuous-content")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        result = jobs.result(queued["id"])
        assert len(result["scenes"]) == 1, "Controlled panel entry must not create a hard cut"
        assert result["content_sampling"]["changes"] > 0
        run = next(item for item in result["model_runs"]
                   if item["id"] == V._module_run_id(queued["id"], "content"))
        assert run["status"] == "success"
        inputs = V._content_input(result["file_sha256"], result["window"],
                                  result["content_sampling"], result["content_frames"])
        assert run["input_sha256"] == S.sha256_text(inputs)
        changed_inputs = copy.deepcopy(inputs)
        changed_inputs["frames"][-1]["evidence_sha256"] = "0" * 64
        assert S.sha256_text(changed_inputs) != run["input_sha256"]
        assert len(S.canonical_json(result)) < V.MAX_RESULT_BYTES

        # A new-contract artifact cannot shed its required provenance fields.
        # Mapping a historical shape still discloses its missing provenance;
        # the separate legacy-job test verifies its read/apply compatibility.
        legacy = copy.deepcopy(result)
        legacy.pop("content_frames")
        legacy["model_runs"] = [item for item in legacy["model_runs"] if item["id"] != run["id"]]
        fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(legacy, jobs.queue.get_job(queued["id"])))
        missing_run = dict(result, model_runs=legacy["model_runs"])
        fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(missing_run, jobs.queue.get_job(queued["id"])))
        old_claim = next(claim for annotation in V._candidate_annotations(legacy)
                         for claim in annotation["attributes"]["semantic"]["claims"]
                         if claim["method"]["id"] == "scene-change-v1")
        assert old_claim["model_run_id"] is None
        assert old_claim["method"]["parameters"]["sample_frames"] == []
        assert "旧候选" in old_claim["uncertainty_reason"]
        invalid = copy.deepcopy(result)
        invalid["content_frames"][0]["evidence_sha256"] = "0" * 64
        fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(invalid, jobs.queue.get_job(queued["id"])))
        invalid_time = copy.deepcopy(result)
        invalid_time["content_frames"][0]["seconds"] = S.rational(1)
        fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(invalid_time, jobs.queue.get_job(queued["id"])))
        excessive = dict(result, content_frames=[result["content_frames"][0]] * (V.MAX_CONTENT_FRAMES + 1))
        fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(excessive, jobs.queue.get_job(queued["id"])))

        current = project.get_task(task_id)
        jobs.apply(queued["id"], command_id="apply-content", expected_revision=current["revision"],
                   expected_sha256=current["content_sha256"], request_body=b"{}")
        after = project.get_task(task_id)
        annotation = next(item for item in after["annotations"] if "vision-content-change" in item["tags"])
        claims = [claim for claim in annotation["attributes"]["semantic"]["claims"]
                  if claim["method"]["id"] == "scene-change-v1"]
        frame_map = {item["frame_index"]: item for item in result["content_frames"]}
        for claim in claims:
            params = claim["method"]["parameters"]
            assert claim["model_run_id"] == run["id"]
            assert claim["time_window"] == {
                "start": {"seconds": frame_map[params["start_frame"]]["seconds"]},
                "end": {"seconds": frame_map[params["end_frame"]]["seconds"]}}
            assert claim["time_window"] != {"start": annotation["start"], "end": annotation["end"]}
            assert len(params["sample_frames"]) == 3
            assert int(params["sample_frames"][0]["frame_index"]) < int(params["start_frame"])
            assert len(claim["evidence_ids"]) >= 2
            assert set(claim["evidence_ids"]) <= set(annotation["evidence_ids"])
            assert all(sample["evidence_sha256"] == frame_map[sample["frame_index"]]["evidence_sha256"]
                       for sample in params["sample_frames"])
        assert annotation["review_status"] == "unreviewed"
        fail_code("SCHEMA_INVALID", lambda: project.apply_task_command(
            command_id="refuse-model-review", task_id=task_id, expected_revision=after["revision"],
            expected_sha256=after["content_sha256"], operation="review_annotation",
            payload={"annotation_id": annotation["id"], "reviewed": True}, request_body=b"{}"))
        jobs.close()
        jobs = None
        project.close()
        project.open()
        reopened = project.get_task(task_id)
        assert reopened == after

        project.apply_task_command(
            command_id="content-human-edit", task_id=task_id, expected_revision=reopened["revision"],
            expected_sha256=reopened["content_sha256"], operation="update_annotation",
            payload={"annotation_id": annotation["id"], "changes": {
                "label": "受控测试人工复核流程",
                "provenance": {"origin": "manual", "actor_id": "local-user", "job_id": None}}},
            request_body=b"{}")
        edited = project.get_task(task_id)
        project.apply_task_command(
            command_id="content-human-review", task_id=task_id, expected_revision=edited["revision"],
            expected_sha256=edited["content_sha256"], operation="review_annotation",
            payload={"annotation_id": annotation["id"], "reviewed": True}, request_body=b"{}")
        reviewed = project.get_task(task_id)
        source_annotation = next(item for item in reviewed["annotations"] if item["id"] == annotation["id"])
        assert source_annotation["review_status"] == "reviewed"
        assert source_annotation["attributes"] == annotation["attributes"]
        document = json.loads(PStore.export_professional(project, task_id)["data"])
        evidence_ids = {ref for claim in claims for ref in claim["evidence_ids"]}
        exported = {item["id"]: item for item in document["artifacts"]}
        assert evidence_ids <= exported.keys()
        for reference in evidence_ids:
            artifact = exported[reference]
            assert S.sha256_bytes(base64.b64decode(artifact["data_base64"])) == artifact["sha256"]

        destination = WorkspaceStore(tmp_path / "content-import")
        destination.create()
        target_id = destination.import_source(continuous_content_clip)["task_id"]
        target = destination.get_task(target_id)
        PStore.apply_import(destination, target_id, command_id="import-content",
                            expected_revision=target["revision"], expected_sha256=target["content_sha256"],
                            document=document, request_body=b"{}")
        destination.close()
        destination.open()
        imported = destination.get_task(target_id)
        restored = next(item for item in imported["annotations"] if item["legacy_id"] == annotation["id"])
        assert restored["review_status"] == "unreviewed"
        assert restored["attributes"] == source_annotation["attributes"]
        assert restored["evidence_ids"] == source_annotation["evidence_ids"]
        assert run in imported["extensions"]["professional"]["model_runs"]
        reexported = json.loads(PStore.export_professional(destination, target_id)["data"])
        assert evidence_ids <= {item["id"] for item in reexported["artifacts"]}
    finally:
        if jobs is not None:
            jobs.close()
        if destination is not None:
            destination.close()
        project.close()


def test_content_evidence_does_not_borrow_another_decode_of_the_same_frame(tmp_path):
    from PIL import Image

    project = WorkspaceStore(tmp_path / "same-frame-evidence")
    project.create()
    try:
        sources = []
        for name, color in (("keyframe", "red"), ("content", "blue")):
            path = project.tmp_dir / (name + ".png")
            Image.new("RGB", (16, 16), color).save(path)
            sources.append({"relative_path": path.relative_to(project.root).as_posix(),
                            "sha256": V.stream_sha256(path), "size_bytes": str(path.stat().st_size)})
        keyframe = {"scene_index": 0, "frame_index": "4", "evidence_sha256": sources[0]["sha256"]}
        content = {"scene_index": 0, "frame_index": "4", "evidence_sha256": sources[1]["sha256"]}
        candidate = {"keyframes": [keyframe], "content_frames": [content],
                     "boundary_frames": [], "evidence": sources}
        annotation = {"id": "ann-same-frame", "evidence_ids": [], "attributes": {"semantic": {"claims": [
            {"method": {"id": "keyframe-sampling-local", "parameters": {"frame_index": "4"}},
             "evidence_ids": []},
            {"method": {"id": "scene-change-v1", "parameters": {"sample_frames": [content]}},
             "evidence_ids": []},
        ]}}}
        attached = copy.deepcopy(annotation)
        V._attach_candidate_evidence(project, candidate, [attached])
        ordinary, change = attached["attributes"]["semantic"]["claims"]
        assert ordinary["evidence_ids"] and change["evidence_ids"]
        assert ordinary["evidence_ids"] != change["evidence_ids"]
        assert set(attached["evidence_ids"]) == set(ordinary["evidence_ids"] + change["evidence_ids"])
        missing_content = dict(candidate, evidence=sources[:1])
        fail_code("VISION_EVIDENCE_UNAVAILABLE", lambda: V._attach_candidate_evidence(
            project, missing_content, [copy.deepcopy(annotation)]))
        # A descriptor for the content bytes pointing at the other file must
        # also fail its real byte/hash check, even though a keyframe is present.
        wrong_file = {**sources[1], "relative_path": sources[0]["relative_path"]}
        changed_content = dict(candidate, evidence=[sources[0], wrong_file])
        fail_code("VISION_EVIDENCE_UNAVAILABLE", lambda: V._attach_candidate_evidence(
            project, changed_content, [copy.deepcopy(annotation)]))
    finally:
        project.close()


def test_content_sampling_and_claims_stay_inside_requested_middle_window(tmp_path, continuous_content_clip):
    padded = tmp_path / "long-continuous.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(continuous_content_clip),
        "-vf", "tpad=start_duration=4:stop_duration=4:start_mode=clone:stop_mode=clone",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(padded),
    ], check=True, capture_output=True, timeout=60)
    project = WorkspaceStore(tmp_path / "middle-window")
    project.create()
    jobs = None
    try:
        project.import_source(padded)
        jobs = V.VisionJobs(project, clip_model_path="nonexistent-clip-weights",
                            vlm_model_path="nonexistent-vlm-weights")
        window = {"start": S.rational(4.25), "end": S.rational(7.75)}
        queued = submit(jobs, command_id="content-middle", window=window)["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        result = jobs.result(queued["id"])
        assert len(result["scenes"]) == 1 and result["scenes"][0]["duration"] == 12
        assert result["content_sampling"]["scene_results"][0]["window"] == window
        left, right = S.rational_value(window["start"]), S.rational_value(window["end"])
        assert result["content_frames"]
        assert all(left <= S.rational_value(frame["seconds"]) < right for frame in result["content_frames"])
        claims = [claim for annotation in V._candidate_annotations(result)
                  for claim in annotation["attributes"]["semantic"]["claims"]
                  if claim["method"]["id"] == "scene-change-v1"]
        assert claims, "Middle-window fixture must produce a real content candidate"
        for claim in claims:
            assert left <= S.rational_value(claim["time_window"]["start"]["seconds"])
            assert S.rational_value(claim["time_window"]["end"]["seconds"]) < right
            assert all(left <= S.rational_value(frame["seconds"]) < right
                       for frame in claim["method"]["parameters"]["sample_frames"])
        # Even an internally rehashed descriptor cannot claim a right-edge
        # (excluded) sample belongs to this half-open request.
        invalid = copy.deepcopy(result)
        invalid["content_frames"][-1]["seconds"] = window["end"]
        run = next(item for item in invalid["model_runs"] if item["id"] == V._module_run_id(queued["id"], "content"))
        run["input_sha256"] = S.sha256_text(V._content_input(
            invalid["file_sha256"], window, invalid["content_sampling"], invalid["content_frames"]))
        fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(invalid, jobs.queue.get_job(queued["id"])))
    finally:
        if jobs is not None:
            jobs.close()
        project.close()


def test_content_contract_invalidates_reuse_but_reads_and_applies_legacy_candidate(store, monkeypatch):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        legacy_parameters = dict(V._parameters(),
                                 job_identity=V._job_identity(content_evidence_contract=None))
        old = jobs.queue.create_batch(
            [{"task_id": current["id"], "expected_revision": current["revision"],
              "expected_sha256": current["content_sha256"]}],
            operation=V.OPERATION, processor_version=V.VERSION,
            parameters=legacy_parameters, command_id="legacy-content-contract")["jobs"][0]
        processor = jobs._processor

        def old_shape(job, context):
            # Execute real detection, then serialize the pre-evidence artifact
            # shape. No detection outputs or successful job state are invented.
            result = processor(job, context)
            result.pop("content_frames")
            result["model_runs"] = [run for run in result["model_runs"]
                                    if run["id"] != V._module_run_id(job["id"], "content")]
            return result

        with monkeypatch.context() as patch:
            # Simulate the old installed identity while producing the old
            # artifact fixture. The current worker below must reject old jobs.
            identity = V._job_identity
            patch.setattr(V, "_job_identity", lambda **kwargs: identity(content_evidence_contract=None))
            patch.setattr(jobs, "_processor", old_shape)
            assert jobs.run_pending()[0]["state"] == "succeeded"
        legacy_result = jobs.result(old["id"])
        assert "content_frames" not in legacy_result
        pending_parameters = dict(legacy_parameters, window={"start": S.rational(.1), "end": S.rational(.9)})
        pending = jobs.queue.create_batch(
            [{"task_id": current["id"], "expected_revision": current["revision"],
              "expected_sha256": current["content_sha256"]}],
            operation=V.OPERATION, processor_version=V.VERSION,
            parameters=pending_parameters, command_id="legacy-queued-content")
        rejected = jobs.run_pending(batch_id=pending["id"])[0]
        assert rejected["state"] == "failed"
        assert rejected["error_code"] == "VISION_RESUBMIT_REQUIRED"
        assert "resubmit" in rejected["error_message"]
        fresh = submit(jobs, command_id="new-content-contract")["jobs"][0]
        assert fresh["id"] != old["id"]
        assert fresh["input_revision"] == old["input_revision"]
        assert fresh["state"] == "queued"
        assert fresh["parameters"]["job_identity"]["config"]["content_sampling"]["evidence_contract"] == V.CONTENT_EVIDENCE_CONTRACT
        fail_code("VISION_NOT_READY", lambda: jobs.result(fresh["id"]))
        assert jobs.run_pending()[0]["state"] == "succeeded"
        assert jobs.result(fresh["id"])["content_frames"]
        assert jobs.result(old["id"]) == legacy_result
        # Preserve the legacy immutable artifact's normal apply path as well.
        jobs.apply(old["id"], command_id="apply-legacy-content", expected_revision=current["revision"],
                   expected_sha256=current["content_sha256"], request_body=b"{}")
        assert all(item["review_status"] == "unreviewed" for item in task(store)["annotations"])
        altered = copy.deepcopy(fresh)
        altered["parameters"]["job_identity"]["config"]["content_sampling"]["evidence_contract"] = "unknown"
        fail_code("SCHEMA_INVALID", lambda: V._validate_job(altered))
    finally:
        jobs.close()


def test_human_upgrade_rewrites_origin_then_reviews(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    current = task(store)
    queued = submit(jobs, command_id="vision-upgrade")["jobs"][0]
    jobs.run_pending()
    result = jobs.result(queued["id"])
    annotations = V._candidate_annotations(result)
    body = json.dumps({"candidates": annotations, "source_job_id": queued["id"]}).encode("utf-8")
    jobs.apply(queued["id"], command_id="apply-upgrade",
               expected_revision=current["revision"], expected_sha256=store.task_digest(current), request_body=body)
    after = task(store)
    candidate = after["annotations"][0]
    upgraded = store.apply_task_command(
        command_id="human-edit", task_id=after["id"], expected_revision=after["revision"],
        expected_sha256=store.task_digest(after), operation="update_annotation",
        payload={"annotation_id": candidate["id"],
                 "changes": {"label": "人工确认的镜头", "provenance": {"origin": "manual", "actor_id": "local-user", "job_id": None}}},
        request_body=b'{}')["response"]
    after = store.get_task(after["id"])
    reviewed = store.apply_task_command(
        command_id="human-review", task_id=after["id"], expected_revision=after["revision"],
        expected_sha256=store.task_digest(after), operation="review_annotation",
        payload={"annotation_id": candidate["id"], "reviewed": True}, request_body=b'{}')["response"]
    final = store.get_task(after["id"])
    upgraded_ann = next(a for a in final["annotations"] if a["id"] == candidate["id"])
    assert upgraded_ann["label"] == "人工确认的镜头"
    assert upgraded_ann["provenance"]["origin"] == "manual"
    assert upgraded_ann["review_status"] == "reviewed"
    assert isinstance(upgraded_ann.get("review_binding"), dict)
    assert upgraded_ann["review_binding"]["actor_id"] == "local-user"
    jobs.close()


def test_close_stops_runner_and_cancels_active(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    submit(jobs)
    started = threading.Event()

    def run():
        jobs.run_pending()
    thread = threading.Thread(target=run)
    thread.start()
    started.set()
    report = jobs.close(timeout=5)
    thread.join(timeout=5)
    assert report["closing"] is True
    assert report["stopped"] is True


def test_http_vision_job_reports_eight_frames_need_nine(tmp_path, eight_frame_clip):
    """The workbench vision route must state 8<9 instead of silently reporting none."""
    import threading
    import time
    import urllib.request
    from trace_annotate.workspace.server import TOKEN_HEADER, make_server

    project = WorkspaceStore(tmp_path / "eight-workspace")
    project.create()
    server = None
    thread = None
    try:
        project.import_source(eight_frame_clip)
        current = task(project)
        server = make_server(project)
        server.vision_jobs.close(timeout=5)
        server.vision_jobs = V.VisionJobs(project, queue=server.queue,
                                          closing=server.closing,
                                          clip_model_path="nonexistent-clip-weights",
                                          vlm_model_path="nonexistent-vlm-weights")
        server.vision_jobs.start()
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}/api/v1/projects/current/"
        headers = {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}",
                   TOKEN_HEADER: server.token}

        def get(path):
            request = urllib.request.Request(base + path, headers=headers)
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read())

        def post(path, body):
            request = urllib.request.Request(
                base + path, data=json.dumps(body).encode("utf-8"),
                headers={**headers, "Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read())

        submission = {"command_id": "http-vision-eight", "expected_revision": current["revision"],
                      "expected_sha256": current["content_sha256"], "window": None,
                      "subject_annotation_id": None}
        job_id = post(f"tasks/{current['id']}/vision-jobs", submission)["jobs"][0]["id"]
        result = None
        for _ in range(100):
            try:
                result = get(f"jobs/{job_id}/vision-result")
                break
            except Exception:
                time.sleep(0.1)
        assert result is not None
        content = result["content_sampling"]
        assert content["status"] == "insufficient_samples"
        assert content["required_samples"] == 9
        assert content["insufficient_scenes"] == len(content["scene_results"])
        assert any(item["provided_samples"] == 8 for item in content["scene_results"])
        first = next(item for item in content["scene_results"]
                     if item["status"] == "insufficient_samples")
        assert "9" in first["message"] and "8" in first["message"]
    finally:
        if server is not None:
            server.shutdown()
            if thread is not None:
                thread.join(timeout=10)
            server.server_close()
        project.close()
def test_http_vision_chain_submit_result_apply_status(store, clip):
    """Exercise the loopback HTTP vision routes end to end on a real import."""
    import threading
    import time
    import urllib.request
    from trace_annotate.workspace.server import TOKEN_HEADER, make_server

    server = make_server(store)
    # Replace the auto-started worker with one that degrades CLIP/VLM to
    # not_run: this test exercises the HTTP contract, not real model weights.
    server.vision_jobs.close(timeout=5)
    server.vision_jobs = V.VisionJobs(store, queue=server.queue, closing=server.closing,
                                      clip_model_path="nonexistent-clip-weights",
                                      vlm_model_path="nonexistent-vlm-weights")
    server.vision_jobs.start()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}/api/v1/projects/current/"
    headers = {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}", TOKEN_HEADER: server.token}

    def get(path):
        request = urllib.request.Request(base + path, headers=headers)
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    def post(path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                         headers={**headers, "Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    try:
        options = get("vision-options")
        assert options["capabilities"]["scene"] is True
        raw = clip.read_bytes()
        upload = urllib.request.Request(base + "imports", data=raw, headers={**headers,
            "Content-Type": "application/octet-stream", "Content-Length": str(len(raw)),
            "X-TraceAnnotate-Filename": "vision-source.mp4"})
        with urllib.request.urlopen(upload) as response:
            admitted = json.loads(response.read())
        task_id = admitted["task_id"]
        current = get(f"tasks/{task_id}")
        submission = {"command_id": "http-vision-submit", "expected_revision": current["revision"],
                      "expected_sha256": current["content_sha256"], "window": None, "subject_annotation_id": None}
        batch = post(f"tasks/{task_id}/vision-jobs", submission)
        job_id = batch["jobs"][0]["id"]
        result = None
        for _ in range(80):
            try:
                result = get(f"jobs/{job_id}/vision-result")
                break
            except Exception:
                time.sleep(0.1)
        assert result is not None
        assert result["kind"] == "traceannotate.vision-candidate"
        assert result["scenes"]
        applied = post(f"jobs/{job_id}/apply-vision", {"command_id": "http-vision-apply",
            "expected_revision": current["revision"], "expected_sha256": current["content_sha256"],
            "operation": "apply_vision_job", "payload": {}})
        assert applied["candidate_count"] == len(result["scenes"])
        status = get(f"tasks/{task_id}/vision-status")
        assert status["candidate_count"] == len(result["scenes"])
        assert all(c["provenance"]["origin"] == "model_candidate" for c in status["candidates"])
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_vision_sampling_uses_pts_anchored_multi_frame_plan(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        queued = submit(jobs, command_id="vision-samples")["jobs"][0]
        jobs.run_pending()
        result = jobs.result(queued["id"])
    finally:
        jobs.close()
    sampling = result["sampling"]
    assert sampling["frame_map_sha256"]
    assert sampling["fingerprint"]
    assert sampling["returned"] >= 2
    assert all(isinstance(frame["frame_index"], str) for frame in result["keyframes"])
    assert all(frame["source_pts"] for frame in result["keyframes"])
    assert all("first" in frame["roles"] or "last" in frame["roles"]
               or "uniform" in frame["roles"] or "boundary" in frame["roles"]
               or "boundary_before" in frame["roles"] or "boundary_after" in frame["roles"]
               for frame in result["keyframes"])
    covered = {frame["scene_index"] for frame in result["keyframes"]}
    assert covered == set(range(len(result["scenes"])))
    assert all(item["relative_path"].endswith((".png", "audio/acoustic.m4a"))
               for item in result["evidence"])
    assert any(item["relative_path"].endswith("audio/acoustic.m4a") for item in result["evidence"])


def test_submit_many_freezes_each_task_and_replays_the_same_receipt(store):
    jobs = V.VisionJobs(store, face_model_path="nonexistent-face-weights",
                        clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        items = [{"task_id": current["id"], "expected_revision": current["revision"],
                  "expected_sha256": store.task_digest(current)}]
        batch = jobs.submit_many(items, command_id="vision-many")
        assert len(batch["jobs"]) == 1
        job = batch["jobs"][0]
        assert job["task_id"] == current["id"]
        assert job["input_revision"] == current["revision"]
        replay = jobs.submit_many(items, command_id="vision-many")
        assert replay["id"] == batch["id"]
        assert [item["id"] for item in replay["jobs"]] == [job["id"]]
    finally:
        jobs.close(timeout=5)


def test_vision_options_never_loads_models(store, monkeypatch):
    """A status probe must stay cheap: opening the editor cannot block on torch."""
    jobs = V.VisionJobs(store)

    def forbidden(*args, **kwargs):
        raise AssertionError("options() must not build extractors")

    monkeypatch.setattr(jobs, "_load_extractors", forbidden)
    try:
        options = jobs.options()
    finally:
        jobs.close(timeout=5)
    assert options["loaded"] is False
    assert options["capabilities"]["scene"] is True
    assert options["capabilities"]["clip"] == "not_loaded"
    assert options["extractor_error"] is None


def test_submit_many_rejects_duplicate_tasks(store):
    jobs = V.VisionJobs(store, face_model_path="nonexistent-face-weights",
                        clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        item = {"task_id": current["id"], "expected_revision": current["revision"],
                "expected_sha256": store.task_digest(current)}
        with pytest.raises(SchemaInvalid, match="Duplicate task"):
            jobs.submit_many([item, dict(item)], command_id="vision-many-dup")
    finally:
        jobs.close(timeout=5)
