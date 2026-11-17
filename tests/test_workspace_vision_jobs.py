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
import hashlib
import shutil
import subprocess
import threading
from pathlib import PurePosixPath
from types import SimpleNamespace

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


def decide(store, candidate, *, command_id, decision, changes=None, actor_id="test-reviewer",
           actor_type="human", diff_summary="", expected_task=None):
    current = expected_task or task(store)
    return store.apply_task_command(
        command_id=command_id, task_id=current["id"], expected_revision=current["revision"],
        expected_sha256=current["content_sha256"], operation="decide_candidate",
        payload={"annotation_id": candidate["id"], "decision": decision, "actor_id": actor_id,
                 "actor_type": actor_type,
                 "expected_candidate_sha256": store.candidate_digest(candidate),
                 "changes": changes or {}, "diff_summary": diff_summary}, request_body=b"{}")


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


def test_candidate_projection_clips_rounded_tail_to_exact_media_duration():
    candidate = {
        "job_id": "job-rounded-tail", "processor_version": "1", "model_runs": [],
        "scenes": [{"start": 145.933333, "end": 147.666667}],
        "keyframes": [{"scene_index": 0, "frame_index": "4378", "source_pts": "145.933333",
                       "person": {"status": "success", "persons_detected": [{}]},
                       "motion": {"status": "success", "summary": {
                           "class": "static", "mean_speed_px_s": 0.0}, "pairs": []}}],
    }
    original = copy.deepcopy(candidate)
    duration = {"n": "443", "d": "3"}
    annotation = V._candidate_annotations(candidate, duration=duration)[0]
    assert annotation["end"]["seconds"] == duration
    claims = annotation["attributes"]["semantic"]["claims"]
    assert {claim["method"]["id"] for claim in claims} == {
        "shot-boundary-local", "keyframe-sampling-local", "person-box-local", "camera-motion-local"}
    for claim in claims:
        assert claim["time_window"]["end"]["seconds"] == duration
        if claim["measurement"] is not None:
            assert claim["measurement"]["analysis_window"] == claim["time_window"]
    assert candidate == original, "Projection must preserve the immutable detector output"
    candidate["scenes"][0] = {"start": 1.25, "end": 2.5}
    assert V._candidate_annotations(candidate, duration=duration) == V._candidate_annotations(candidate)
    candidate["scenes"][0] = {"start": -0.25, "end": 0.5}
    assert V._candidate_annotations(candidate, duration=duration)[0]["start"]["seconds"] == S.rational(0)
    candidate["scenes"][0] = {"start": 147.666667, "end": 148.0}
    fail_code("VISION_CANDIDATE_OUTSIDE_MEDIA", lambda: V._candidate_annotations(candidate, duration=duration))


def test_action_projection_requires_positive_half_open_scene_overlap():
    candidate = {"job_id": "job-action-window", "processor_version": "1",
                 "scenes": [{"start": 1.0, "end": 2.0}],
                 "keyframes": [{"scene_index": 0, "frame_index": "15", "source_pts": "1.5"}],
                 "actions_sampling": {"candidates": [
                     {"action": "touch-left", "window_seconds": {"start": 0.0, "end": 1.0}},
                     {"action": "touch-right", "window_seconds": {"start": 2.0, "end": 3.0}},
                     {"action": "empty", "window_seconds": {"start": 1.5, "end": 1.5}},
                     {"action": "走", "window_seconds": {"start": 1.25, "end": 1.75},
                      "window_pts": {"start": S.rational(1.25),
                                     "end": S.rational(1.75)},
                      "subject_ref": "subject-1", "object_ref": None, "recipient_ref": None,
                      "provider": "subject-translation-v1", "method": "subject-translation-v1",
                      "model_run_id": V._module_run_id("job-action-window", "actions"),
                      "sample_frames": [], "metrics": {"speed_heights_per_s": .5},
                      "score": .5, "uncertainty": "requires review"},
                 ]}}
    annotation = V._candidate_annotations(candidate)[0]
    assert [tag for tag in annotation["tags"] if tag.startswith("vision-action-")] == ["vision-action-走"]
    actions = [claim for claim in annotation["attributes"]["semantic"]["claims"]
               if claim["method"]["id"] == "subject-translation-v1"]
    assert len(actions) == 1
    assert actions[0]["time_window"] == {"start": {"seconds": S.rational(1.25)},
                                         "end": {"seconds": S.rational(1.75)}}


@pytest.mark.parametrize("prompt", [None, "<CAPTION>"])
def test_vlm_model_run_records_the_actual_adapter_task_prompt(prompt):
    extractors = {} if prompt is None else {"vlm": SimpleNamespace(task_prompt=prompt, code_identity={})}
    runs = V._build_model_runs(
        {"id": "job-prompt", "parameters": {"job_identity": {"config": {}}}},
        extractors, [], {"status": "not_run", "scenes": []}, None, [], {}, "0" * 64, None)
    vlm = next(run for run in runs if run["id"] == V._module_run_id("job-prompt", "vlm"))
    assert vlm["parameters"]["task_prompt"] == (prompt or "<MORE_DETAILED_CAPTION>")


@pytest.mark.parametrize("reported", [
    "success", "partial", "insufficient_samples", "insufficient_evidence", "not_applicable"])
def test_completed_model_outcomes_are_not_misreported_as_execution_failures(reported):
    assert V._run_status(reported) == "success"


@pytest.mark.parametrize("reported,expected", [
    ("not_run", "not_run"), ("skipped", "not_run"),
    ("failed", "failed"), ("unknown", "failed")])
def test_nonexecution_and_real_failures_keep_distinct_execution_status(reported, expected):
    assert V._run_status(reported) == expected


@pytest.mark.parametrize("status", ["failed", "not_run", None])
@pytest.mark.parametrize("engine_loaded", [True, False])
def test_run_ocr_preserves_explicit_adapter_status(status, engine_loaded):
    result = {"text": "", "boxes": [], "reason": "runtime test"}
    if status is not None:
        result["status"] = status
    adapter = SimpleNamespace(engine=object() if engine_loaded else None,
                              extract_text=lambda frame: dict(result))
    measured = V._run_ocr(np.zeros((4, 4, 3), dtype=np.uint8), adapter)
    assert measured["status"] == (status or ("success" if engine_loaded else "not_run"))
    assert measured["text"] == "" and measured["boxes"] == []
    assert measured["reason"] == "runtime test"


def test_model_runs_use_loaded_weight_identity_and_separate_licenses():
    identity = {"weights_sha256": "a" * 64, "selected_revision": "actual-snapshot",
                "revision": "requested-main", "runtime_license": "runtime-license",
                "weights_license": "weights-license", "weights_license_source": "https://example.org/license",
                "weights_identity_status": "verified", "model_code_sha256": "b" * 64,
                "model_code_license": "code-license",
                "weights_files": [{"role": "det", "path": "C:/private/model.onnx",
                                   "sha256": "c" * 64, "size_bytes": 123}]}
    extractors = {name: SimpleNamespace(code_identity=dict(identity), task_prompt="<CAPTION>")
                  for name in ("clip", "vlm", "ocr")}
    runs = V._build_model_runs(
        {"id": "job-identity", "parameters": {"job_identity": {"config": {}}}},
        extractors, [], {"status": "not_run", "scenes": []}, None, [], {}, "0" * 64, None)
    for module in ("clip", "vlm"):
        run = next(item for item in runs if item["id"] == V._module_run_id("job-identity", module))
        assert run["weights_sha256"] == "a" * 64
        assert run["model_version"] == "actual-snapshot"
        assert run["license"] == "weights-license"
        assert run["preprocessing"]["runtime_license"] == "runtime-license"
        assert run["preprocessing"]["weights_license_source"] == "https://example.org/license"
    vlm = next(item for item in runs if item["id"] == V._module_run_id("job-identity", "vlm"))
    assert vlm["preprocessing"]["model_code_sha256"] == "b" * 64
    assert vlm["preprocessing"]["model_code_license"] == "code-license"
    assert vlm["parameters"]["requested_revision"] == "requested-main"
    ocr = next(item for item in runs if item["id"] == V._module_run_id("job-identity", "ocr"))
    assert ocr["weights_sha256"] == "a" * 64
    assert ocr["license"] == "weights-license"
    assert ocr["preprocessing"]["weights_files"] == [{"role": "det", "sha256": "c" * 64, "size_bytes": 123}]


def test_object_pipeline_associates_sparse_samples_without_asserting_actions():
    keyframes = []
    for index, (frame, seconds) in enumerate(((0, 0), (90, 3), (180, 6))):
        digest = f"{index + 1:064x}"
        x = index * 8
        keyframes.append({
            "scene_index": 0, "frame_index": str(frame),
            "timestamp": S.rational(seconds), "evidence_sha256": digest,
            "object": {"status": "success", "reason": None, "provider": V.OBJECT_PROVIDER,
                       "objects_detected": [{"class_id": 39, "class_label": "bottle",
                                             "box": [x, 0, x + 10, 10], "score": .8,
                                             "score_kind": "model_score_uncalibrated"}]},
            "person": {"status": "success", "persons_detected": [
                {"box": [x + 9, 0, x + 49, 20], "confidence": .9}]},
            "face": {"status": "not_run"},
        })
    sampling, relations = V._object_analysis(keyframes, "job-objects")
    assert sampling["status"] == "success"
    assert sampling["tracking"]["parameters"]["max_gap_frames"] == 89
    assert sampling["tracking"]["parameters"]["max_gap_seconds"] == 5.0
    assert len(sampling["tracking"]["tracks"]) == 1
    assert sampling["tracking"]["scope"].startswith("sparse_observation_")
    assert relations["status"] == "success"
    assert {item["relation"] for item in relations["relations"]} >= {
        "near", "contact_candidate", "co_moving", "held_by"}
    assert all("action" not in item for item in relations["relations"])
    assert relations["unsupported_action_assertions"] == [
        "pick_up", "put_down", "open", "close", "hand_over"]


def test_object_pipeline_caps_each_sample_and_discloses_truncation():
    detections = [{"class_id": 39, "class_label": "bottle", "box": [i, 0, i + 1, 1],
                   "score": .8, "score_kind": "model_score_uncalibrated"}
                  for i in range(V.MAX_OBJECTS_PER_KEYFRAME + 3)]
    extractor = SimpleNamespace(detect_objects=lambda frame: {
        "status": "success", "reason": None, "provider": V.OBJECT_PROVIDER,
        "object_count": len(detections), "objects_detected": detections,
        "excluded_class_ids": [0]})
    result = V._run_object(np.zeros((4, 4, 3), dtype=np.uint8), extractor)
    assert result["object_count"] == V.MAX_OBJECTS_PER_KEYFRAME
    assert result["detector_object_count"] == len(detections)
    assert result["truncated"] is True
    assert len(result["objects_detected"]) == V.MAX_OBJECTS_PER_KEYFRAME


def test_hand_sampling_is_bounded_uniform_and_covers_both_ends():
    decoded = [{"frame_index": str(index), "sha256": f"{index:064x}"}
               for index in range(100)]
    selected = V._uniform_hand_sample_keys(decoded)
    indexes = sorted(int(frame) for frame, _digest in selected)
    assert len(selected) == V.MAX_HAND_KEYFRAMES
    assert indexes[0] == 0 and indexes[-1] == 99
    assert max(right - left for left, right in zip(indexes, indexes[1:])) <= 7
    assert V._uniform_hand_sample_keys([]) == set()
    assert V._uniform_hand_sample_keys(decoded[:1]) == {("0", f"{0:064x}")}


def test_hand_observations_bind_frame_evidence_run_and_reject_tampering():
    config = V._validate_hand_configuration({
        "contract": V.HAND_EVIDENCE_CONTRACT, "provider": V.HAND_PROVIDER,
        "model": {"status": "available", "name": "hand_landmarker.task",
                  "path_identity_sha256": "1" * 64,
                  "sha256": V.HAND_WEIGHTS_SHA256,
                  "size_bytes": str(V.HAND_MODEL_SIZE)},
        "expected_size_bytes": str(V.HAND_MODEL_SIZE),
        "expected_sha256": V.HAND_WEIGHTS_SHA256,
        "runtime_version": "0.test", "offline": True,
        "authority": "hand_keypoint_observations_only",
        "cancellation_boundary": "owned_subprocess_polled_during_inference",
        "sampling_policy": V.HAND_SAMPLING_POLICY,
        "max_frames_per_job": V.MAX_HAND_KEYFRAMES,
        "phase_timeout_seconds": V.HAND_PHASE_TIMEOUT_SECONDS,
    })
    job_id, run_id = "job-hand-binding", V._module_run_id("job-hand-binding", "hands")
    pts, evidence, pixel = {"n": "1", "d": "2"}, "2" * 64, "3" * 64
    observation_identity = {
        "scene_index": 2, "frame_index": "7", "pts": pts,
        "evidence_sha256": evidence, "pixel_sha256": pixel,
        "model_run_id": run_id, "hand_index": 0,
    }
    encoded = json.dumps(observation_identity, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    observation = {
        "contract": V.HAND_EVIDENCE_CONTRACT,
        "observation_id": "handobs-" + hashlib.sha256(encoded).hexdigest()[:40],
        **observation_identity, "subject_ref": None,
        "landmark_names": list(V.HAND_LANDMARK_NAMES),
        "landmarks_normalized": [[.1 + index / 1000, .2 + index / 1000, 0.0]
                                  for index in range(21)],
        "landmarks_pixel": [[2.0 + index / 10, 2.0 + index / 10, 0.0]
                            for index in range(21)],
        "world_landmarks_m": [[index / 1000, index / 2000, 0.0]
                              for index in range(21)],
        "box_xyxy": [1.0, 1.0, 10.0, 8.0], "outside_frame": False,
        "handedness_raw": "Left", "handedness_score": .8,
        "handedness_category_index": 1,
        "handedness_score_kind": "mediapipe_category_score_uncalibrated",
        "handedness_mirrored_input_assumption_verified": False,
        "review_status": "unreviewed",
        "uncertainty": "Unreviewed keypoints do not establish touch, grasp or action.",
    }
    raw_identity = {
        "provider": "mediapipe-tasks-cpu", "model_name": V.HAND_PROVIDER,
        "model_version": "float16/1", "model_source": V.HAND_MODEL_URL,
        "model_card": V.HAND_MODEL_CARD, "weights_path": r"D:\private\hand.task",
        "weights_size": V.HAND_MODEL_SIZE, "weights_sha256": V.HAND_WEIGHTS_SHA256,
        "license": V.HAND_MODEL_LICENSE, "license_source": V.HAND_LICENSE_SOURCE,
        "runtime_license_source": V.HAND_RUNTIME_LICENSE_SOURCE,
        "runtime_version": "0.test", "code_version": V.HAND_CODE_VERSION,
        "running_mode": "IMAGE", "max_hands": 4, "offline": True,
        "downloads_allowed": False, "authority": "hand_keypoint_observations_only",
    }
    raw_result = {
        "status": "success", "hands": [observation], "provider": "hand_landmarks",
        "model_identity": raw_identity, "fingerprint": "ignored",
        "review_status": "unreviewed", "authority": "hand_keypoint_observations_only",
        "unsupported_assertions": list(V.HAND_UNSUPPORTED_ASSERTIONS),
        "cancellation_boundary": "owned_subprocess_polled_during_inference",
        "frame_identity": {"scene_index": 2, "frame_index": "7", "pts": pts,
                           "evidence_sha256": evidence, "pixel_sha256": pixel,
                           "model_run_id": run_id, "shape": [10, 20, 3]},
    }
    extractor = SimpleNamespace(
        code_identity=raw_identity,
        analyze_frame=lambda *_args, **_kwargs: copy.deepcopy(raw_result))
    result = V._run_hands(
        np.zeros((10, 20, 3), dtype=np.uint8), extractor,
        scene_index=2, frame_index="7", pts=pts, evidence_sha256=evidence,
        model_run_id=run_id, configuration=config, cancelled=lambda: False)
    assert "weights_path" not in result["model_identity"]
    assert result["model_identity"]["path_identity_sha256"] == "1" * 64
    frame = {"scene_index": 2, "frame_index": "7", "timestamp": pts,
             "evidence_sha256": evidence, "hands": result}
    run = V._model_run_record(
        job_id, "hands", provider="mediapipe-tasks-cpu", model_name=V.HAND_PROVIDER,
        model_version="float16/1", weights_sha256=V.HAND_WEIGHTS_SHA256,
        license=V.HAND_MODEL_LICENSE,
        preprocessing={"path_identity_sha256": "1" * 64},
        parameters={"max_hands": 4, "offline": True, "downloads_allowed": False,
                    "authority": "hand_keypoint_observations_only",
                    "cancellation_boundary": "owned_subprocess_polled_during_inference",
                    "sampling_policy": V.HAND_SAMPLING_POLICY,
                    "max_frames_per_job": V.MAX_HAND_KEYFRAMES,
                    "phase_timeout_seconds": V.HAND_PHASE_TIMEOUT_SECONDS,
                    "coverage": V._hand_coverage([frame]),
                    "unsupported_assertions": list(V.HAND_UNSUPPORTED_ASSERTIONS)},
        input_value=V._keyframe_input([frame]), output_value=V._frame_output([frame], "hands"),
        status="success", failure_reason=None,
        code_version=f"vision_jobs={V.VERSION}; hands={V.HAND_CODE_VERSION}")
    V._validate_hand_analysis([frame], job_id=job_id, model_runs=[run],
                              evidence_hashes={evidence}, configuration=config)
    for mutate in (
            lambda value: value["hands"][0].update(model_run_id="run-other"),
            lambda value: value["hands"][0].update(action="grasp"),
            lambda value: value["hands"][0].update(review_status="accepted"),
            lambda value: value["frame_identity"].update(pts={"n": "2", "d": "3"})):
        changed = copy.deepcopy(result)
        mutate(changed)
        altered = dict(frame, hands=changed)
        fail_code("RECOVERY_REQUIRED", lambda value=altered: V._validate_hand_analysis(
            [value], job_id=job_id, model_runs=[run], evidence_hashes={evidence},
            configuration=config))


def test_vision_job_is_hash_bound_candidate_and_does_not_edit_task(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights", vlm_model_path="nonexistent-vlm-weights")
    initial = task(store)
    queued = submit(jobs)["jobs"][0]
    assert queued["parameters"]["time_domain"] == "playback_seconds"
    identity = queued["parameters"]["job_identity"]
    assert identity["pipeline"] == "vision-assist-v1"
    assert len(identity["config_sha256"]) == 64
    assert identity["config"]["sampling"]["max_frames"] == V.MAX_KEYFRAMES
    assert identity["config"]["hand_landmarks"]["model"]["status"] == "unavailable"
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
               "actions", "object_relations", "acoustics", "boundary", "person", "object",
                   "landmarks", "expressive", "hands", "action_video"}
    run_ids = {run["id"] for run in runs}
    expected_run_ids = {V._module_run_id(queued["id"], name) for name in modules}
    assert len(run_ids) == len(runs), "duplicate model-run ids must be rejected"
    assert not (expected_run_ids - run_ids), "required model-run ids are missing"
    assert not (run_ids - expected_run_ids), "unexpected model-run ids are present"
    duplicate_runs = copy.deepcopy(result)
    duplicate_runs["model_runs"].append(copy.deepcopy(duplicate_runs["model_runs"][0]))
    fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(
        duplicate_runs, jobs.queue.get_job(queued["id"])))
    missing_actions_run = copy.deepcopy(result)
    missing_actions_run["model_runs"] = [run for run in missing_actions_run["model_runs"]
                                          if run["id"] != V._module_run_id(queued["id"], "actions")]
    fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(
        missing_actions_run, jobs.queue.get_job(queued["id"])))
    for module in ("object", "object_relations", "hands"):
        missing_object_run = copy.deepcopy(result)
        missing_object_run["model_runs"] = [run for run in missing_object_run["model_runs"]
                                             if run["id"] != V._module_run_id(queued["id"], module)]
        fail_code("RECOVERY_REQUIRED", lambda value=missing_object_run: jobs._validate_candidate(
            value, jobs.queue.get_job(queued["id"])))
    tampered_relations = copy.deepcopy(result)
    tampered_relations["object_relations"]["scope"] = "action_recognition"
    fail_code("RECOVERY_REQUIRED", lambda: jobs._validate_candidate(
        tampered_relations, jobs.queue.get_job(queued["id"])))
    assert all(claim["model_run_id"] is None or claim["model_run_id"] in run_ids
               for annotation in V._candidate_annotations(result)
               for claim in annotation["attributes"]["semantic"]["claims"])
    content_run = next(run for run in runs if run["id"] == V._module_run_id(queued["id"], "content"))
    assert (content_run["provider"], content_run["model_name"]) == ("trace-annotate", "scene-change-v1")
    assert content_run["status"] == "success"
    assert content_run["failure_reason"] is None
    assert content_run["output_sha256"]
    assert content_run["parameters"]["reported_outcome_status"] == "insufficient_samples"
    for run in runs:
        assert SS.validate_model_run(run) == run
    for module in ("person", "object", "landmarks", "expressive", "hands"):
        optional_run = next(run for run in runs if run["id"] == V._module_run_id(queued["id"], module))
        assert optional_run["status"] == "not_run"
        assert optional_run["failure_reason"]
        assert optional_run["output_sha256"] is None
    assert result["extractor_status"]["hands"] == "not_run"
    assert all(frame["hands"]["status"] == "not_run"
               and frame["hands"]["reason"] == "hand_landmark_weights_missing"
               and frame["hands"]["hands"] == [] for frame in result["keyframes"])
    outcomes = result["actions_sampling"]["class_results"]
    assert set(outcomes) == set(V.ACTION_VOCAB)
    for action in ("拿起", "放下", "打开", "关闭", "递交"):
        assert outcomes[action]["status"] == "not_run"
        assert outcomes[action]["reason"] == "provider_unavailable"
    action_run = next(run for run in runs if run["id"] == V._module_run_id(queued["id"], "actions"))
    assert action_run["provider"] == "subject-translation-v1"
    assert result["object_sampling"]["status"] == "not_run"
    assert result["object_sampling"]["tracking"]["status"] == "not_run"
    assert result["object_relations"]["status"] == "not_run"
    assert result["object_relations"]["relations"] == []
    relation_run = next(run for run in runs
                        if run["id"] == V._module_run_id(queued["id"], "object_relations"))
    assert relation_run["status"] == "not_run"
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
        assert semantic["version"] == "visual-semantics-v3"
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


def test_apply_projection_preflight_preserves_receipts_and_immutable_objects(store, monkeypatch):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        queued = submit(jobs, command_id="preflight-submit")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(queued["id"])
        before_objects = {path.name for path in store.objects_dir.iterdir()}
        before_receipts = list(store.conn.execute("SELECT command_id,request_sha256 FROM command_receipts"))
        invalid = copy.deepcopy(candidate)
        invalid["scenes"][0] = {"start": 1.5, "end": 2.0, "duration": 0.5}
        _, result_snapshot = jobs._read_result_with_snapshot(queued["id"])
        monkeypatch.setattr(jobs, "_read_result_with_snapshot",
                            lambda job_id: (invalid, result_snapshot))
        fail_code("VISION_CANDIDATE_OUTSIDE_MEDIA", lambda: jobs.apply(
            queued["id"], command_id="outside-media", expected_revision=current["revision"],
            expected_sha256=current["content_sha256"], request_body=b"{}"))
        monkeypatch.setattr(jobs, "_read_result_with_snapshot",
                            lambda job_id: (candidate, result_snapshot))
        fail_code("REVISION_CONFLICT", lambda: jobs.apply(
            queued["id"], command_id="stale-projection", expected_revision=current["revision"] + 1,
            expected_sha256=current["content_sha256"], request_body=b"{}"))
        assert task(store) == current
        assert list(store.conn.execute("SELECT command_id,request_sha256 FROM command_receipts")) == before_receipts
        assert {path.name for path in store.objects_dir.iterdir()} == before_objects
    finally:
        jobs.close()


def test_apply_replays_committed_projection_after_mapper_changes(store, monkeypatch):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        queued = submit(jobs, command_id="replay-submit")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        mapper = V._candidate_annotations

        def old_mapper(candidate, **kwargs):
            annotations = mapper(candidate, **kwargs)
            annotations[0]["tags"].append("legacy-projection-preserved")
            return annotations

        arguments = dict(command_id="replay-apply", expected_revision=current["revision"],
                         expected_sha256=current["content_sha256"], request_body=b"original-body")
        with monkeypatch.context() as patch:
            patch.setattr(V, "_candidate_annotations", old_mapper)
            original_receipt = jobs.apply(queued["id"], **arguments)
        after = task(store)
        objects = {path.name for path in store.objects_dir.iterdir()}
        # A subsequent edit must not change which historical payload is replayed.
        decide(store, after["annotations"][0], command_id="later-edit", decision="edited",
               changes={"text": "later edit"}, diff_summary="Test revision after candidate apply")
        edited = task(store)
        # A committed command is replayed from its immutable task revision,
        # not from the disposable candidate object.
        descriptor = jobs.queue.get_job(queued["id"])["commit"]["artifact"]
        candidate_path = store.root / descriptor["relative_path"]
        candidate_path.write_bytes(b"x" * int(descriptor["size_bytes"]))
        assert jobs.apply(queued["id"], **arguments) == original_receipt
        assert task(store) == edited
        assert {path.name for path in store.objects_dir.iterdir()} == objects
        fail_code("IDEMPOTENCY_CONFLICT", lambda: jobs.apply(
            queued["id"], **dict(arguments, request_body=b"different-body")))
    finally:
        jobs.close()


def test_concurrent_identical_apply_is_one_commit_and_one_replay(store, monkeypatch):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        queued = submit(jobs, command_id="concurrent-submit")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        barrier = threading.Barrier(2)
        original = V._prepublish_candidate_evidence

        def aligned(project, publication):
            original(project, publication)
            barrier.wait(timeout=20)

        monkeypatch.setattr(V, "_prepublish_candidate_evidence", aligned)
        arguments = dict(command_id="concurrent-apply",
                         expected_revision=current["revision"],
                         expected_sha256=current["content_sha256"],
                         request_body=b"same-request")
        results = []
        errors = []

        def apply_one():
            try:
                results.append(jobs.apply(queued["id"], **arguments))
            except BaseException as exc:
                errors.append(exc)

        workers = [threading.Thread(target=apply_one) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
        assert all(not worker.is_alive() for worker in workers)
        assert not errors
        assert len(results) == 2 and results[0] == results[1]
        assert store.conn.execute(
            "SELECT COUNT(*) FROM command_receipts WHERE command_id='concurrent-apply'").fetchone()[0] == 1
        assert store.conn.execute(
            "SELECT COUNT(*) FROM durable_commits WHERE id=?",
            (results[0]["commit_id"],)).fetchone()[0] == 1
        assert task(store)["revision"] == current["revision"] + 1
    finally:
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

        decide(project, annotation, command_id="content-human-edit", decision="edited",
               actor_id="test-reviewer", changes={"label": "受控测试人工复核流程"},
               diff_summary="Reviewed the visible content in the controlled clip")
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


def test_prepared_evidence_publication_rolls_back_database_bindings(tmp_path):
    project = WorkspaceStore(tmp_path / "evidence-transaction")
    project.create()
    try:
        data = b"prepared-vision-evidence"
        digest = S.sha256_bytes(data)
        artifact = {"id": S.make_id("art", digest),
                    "relative_path": "objects/" + digest,
                    "sha256": digest, "size_bytes": str(len(data))}
        publication = [{"artifact": artifact, "data": data, "path": None,
                        "owner_id": "ann-transaction", "role": "vision-sample-preview"}]
        with project._evidence_publication_scope() as publication_token:
            V._prepublish_candidate_evidence(project, publication)
            with pytest.raises(RuntimeError, match="late apply failure"):
                with project.transaction():
                    V._bind_candidate_evidence(project, publication, publication_token)
                    raise RuntimeError("late apply failure")
        assert project.conn.execute(
            "SELECT 1 FROM artifacts WHERE id=?", (artifact["id"],)).fetchone() is not None
        assert project.conn.execute(
            "SELECT 1 FROM artifact_references WHERE artifact_id=?", (artifact["id"],)).fetchone() is None
        assert (project.root / artifact["relative_path"]).read_bytes() == data
        cleaned = project.sweep_unbound_artifacts(apply=True)
        assert cleaned["removed"] == 1
        assert not (project.root / artifact["relative_path"]).exists()
    finally:
        project.close()


def test_prepublish_waits_for_another_thread_transaction(store):
    data = b"serialized-prepublication"
    digest = S.sha256_bytes(data)
    artifact = {"id": S.make_id("art", digest),
                "relative_path": "objects/" + digest,
                "sha256": digest, "size_bytes": str(len(data))}
    publication = [{"artifact": artifact, "data": data,
                    "owner_id": "ann-serialized", "role": "vision-sample-preview"}]
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    errors = []

    def hold_transaction():
        try:
            with store.transaction():
                entered.set()
                if not release.wait(15):
                    raise TimeoutError("transaction release timed out")
        except BaseException as exc:
            errors.append(exc)

    def publish():
        try:
            V._prepublish_candidate_evidence(store, publication)
            completed.set()
        except BaseException as exc:
            errors.append(exc)

    owner = threading.Thread(target=hold_transaction)
    publisher = threading.Thread(target=publish)
    owner.start()
    assert entered.wait(10)
    publisher.start()
    assert not completed.wait(.2)
    release.set()
    owner.join(10)
    publisher.join(10)
    assert not owner.is_alive() and not publisher.is_alive() and not errors
    assert completed.is_set()
    assert (store.root / artifact["relative_path"]).read_bytes() == data


def test_prepublish_refuses_same_thread_transaction(store):
    data = b"nested-prepublication"
    digest = S.sha256_bytes(data)
    publication = [{"artifact": {"id": S.make_id("art", digest),
                    "relative_path": "objects/" + digest,
                    "sha256": digest, "size_bytes": str(len(data))},
                    "data": data, "owner_id": "ann-nested",
                    "role": "vision-sample-preview"}]
    with store.transaction():
        fail_code("EVIDENCE_PUBLICATION_IN_TRANSACTION",
                  lambda: V._prepublish_candidate_evidence(store, publication))


def test_prepublished_evidence_tamper_is_rejected_before_binding(store):
    data = b"prepublished-object-identity"
    digest = S.sha256_bytes(data)
    artifact = {"id": S.make_id("art", digest),
                "relative_path": "objects/" + digest,
                "sha256": digest, "size_bytes": str(len(data))}
    publication = [{"artifact": artifact, "data": data,
                    "owner_id": "ann-tampered", "role": "vision-sample-preview"}]
    with store._evidence_publication_scope():
        V._prepublish_candidate_evidence(store, publication)
        (store.root / artifact["relative_path"]).write_bytes(b"x" * len(data))
        fail_code("RECOVERY_REQUIRED",
                  lambda: V._verify_prepublished_candidate_evidence(store, publication))
    assert store.conn.execute(
        "SELECT 1 FROM artifact_references WHERE artifact_id=?", (artifact["id"],)).fetchone() is None


def test_evidence_preparation_enforces_cumulative_input_budget(store, monkeypatch):
    first = "1" * 64
    second = "2" * 64
    candidate = {
        "evidence": [
            {"relative_path": "tmp/first.png", "sha256": first, "size_bytes": "1"},
            {"relative_path": "tmp/second.png", "sha256": second, "size_bytes": "1"},
        ],
        "keyframes": [
            {"scene_index": 0, "frame_index": "0", "evidence_sha256": first},
            {"scene_index": 1, "frame_index": "1", "evidence_sha256": second},
        ],
        "boundary_frames": [], "content_frames": [],
    }
    annotations = [
        {"id": "ann-budget-one", "attributes": {"semantic": {"claims": []}}},
        {"id": "ann-budget-two", "attributes": {"semantic": {"claims": []}}},
    ]
    reads = []
    monkeypatch.setattr(V, "MAX_EVIDENCE_PREPARATION_INPUT_BYTES", 1)
    monkeypatch.setattr(V, "_read_stable_evidence_bytes",
                        lambda project, descriptor: reads.append(descriptor["sha256"]) or b"image")
    monkeypatch.setattr(V, "_evidence_preview", lambda source: b"preview-" + source)
    fail_code("ANALYSIS_LIMIT_EXCEEDED",
              lambda: V._prepare_candidate_evidence(store, candidate, annotations))
    assert reads == [first]


def test_result_snapshot_corrupt_durable_body_is_recovery_error(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        queued = submit(jobs, command_id="corrupt-snapshot-submit")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        snapshot = jobs._capture_result_snapshot(queued["id"])
        with store.transaction():
            store.conn.execute("UPDATE durable_commits SET body=? WHERE id=?",
                               ("{not-json", snapshot["commit"]["id"]))
        fail_code("RECOVERY_REQUIRED", lambda: jobs._verify_result_snapshot(snapshot))
    finally:
        jobs.close()


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


def test_new_evidence_contracts_invalidate_reuse_but_legacy_candidate_remains_readable(store, monkeypatch):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        current = task(store)
        legacy_parameters = dict(V._parameters(),
                                 job_identity=V._job_identity(
                                     content_evidence_contract=None,
                                     object_evidence_contract=None,
                                     include_hand_configuration=False))
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
            result.pop("object_sampling")
            result.pop("object_relations")
            result["model_runs"] = [run for run in result["model_runs"]
                                    if run["id"] not in {
                                        V._module_run_id(job["id"], "content"),
                                        V._module_run_id(job["id"], "object"),
                                        V._module_run_id(job["id"], "object_relations")}]
            return result

        with monkeypatch.context() as patch:
            # Simulate the old installed identity while producing the old
            # artifact fixture. The current worker below must reject old jobs.
            identity = V._job_identity
            patch.setattr(V, "_job_identity", lambda **kwargs: identity(
                content_evidence_contract=None, object_evidence_contract=None,
                include_hand_configuration=False))
            patch.setattr(jobs, "_processor", old_shape)
            assert jobs.run_pending()[0]["state"] == "succeeded"
        legacy_result = jobs.result(old["id"])
        assert "content_frames" not in legacy_result
        assert "object_sampling" not in legacy_result
        assert "object_relations" not in legacy_result
        assert all("hands" not in frame for frame in legacy_result["keyframes"])
        assert V._module_run_id(old["id"], "hands") not in {
            run["id"] for run in legacy_result["model_runs"]}
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
        assert fresh["parameters"]["job_identity"]["config"]["object_sampling"]["evidence_contract"] == V.OBJECT_EVIDENCE_CONTRACT
        assert fresh["parameters"]["job_identity"]["config"]["hand_landmarks"]["contract"] == V.HAND_EVIDENCE_CONTRACT
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
    upgraded = decide(store, candidate, command_id="human-edit", decision="edited",
                     actor_id="test-reviewer", changes={"label": "人工确认的镜头"},
                     diff_summary="Reviewed the shot in the controlled fixture")["response"]
    final = store.get_task(after["id"])
    upgraded_ann = next(a for a in final["annotations"] if a["id"] == candidate["id"])
    assert upgraded_ann["label"] == "人工确认的镜头"
    assert upgraded_ann["provenance"]["origin"] == "manual"
    assert upgraded_ann["review_status"] == "reviewed"
    assert isinstance(upgraded_ann.get("review_binding"), dict)
    assert upgraded_ann["review_binding"]["actor_id"] == "test-reviewer"
    jobs.close()


@pytest.mark.parametrize(
    "decision,actor_type,changes,summary,expected_status",
    [
        ("accepted", "assistant", {}, "", "unreviewed"),
        ("edited", "human", {"label": "Reviewed test shot"}, "Visible test shot checked", "reviewed"),
        ("rejected", "human", {}, "", "removed"),
    ],
)
def test_candidate_decision_preserves_source_and_rejects_stale_or_generic_edits(
        store, clip, tmp_path, decision, actor_type, changes, summary, expected_status):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        before = task(store)
        queued = submit(jobs, command_id="candidate-decision-submit")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        jobs.apply(queued["id"], command_id="candidate-decision-apply",
                   expected_revision=before["revision"], expected_sha256=before["content_sha256"],
                   request_body=b"{}")
        applied = task(store)
        candidate = applied["annotations"][0]
        candidate_sha = store.candidate_digest(candidate)
        fail_code("CANDIDATE_DECISION_REQUIRED", lambda: store.apply_task_command(
            command_id="generic-candidate-delete", task_id=applied["id"],
            expected_revision=applied["revision"], expected_sha256=applied["content_sha256"],
            operation="delete_annotation", payload={"annotation_id": candidate["id"]},
            request_body=b"{}"))
        fail_code("CANDIDATE_STALE", lambda: store.apply_task_command(
            command_id="stale-candidate-decision", task_id=applied["id"],
            expected_revision=applied["revision"], expected_sha256=applied["content_sha256"],
            operation="decide_candidate", payload={
                "annotation_id": candidate["id"], "decision": decision, "actor_id": "test-reviewer",
                "actor_type": actor_type, "expected_candidate_sha256": "0" * 64,
                "changes": changes, "diff_summary": summary}, request_body=b"{}"))
        fail_code("ACTOR_TYPE_CONFLICT", lambda: store.apply_task_command(
            command_id="forged-human-generator", task_id=applied["id"],
            expected_revision=applied["revision"], expected_sha256=applied["content_sha256"],
            operation="decide_candidate", payload={
                "annotation_id": candidate["id"], "decision": "accepted",
                "actor_id": candidate["provenance"]["actor_id"], "actor_type": "human",
                "expected_candidate_sha256": candidate_sha,
                "changes": {}, "diff_summary": ""}, request_body=b"{}"))
        assert task(store) == applied
        receipt = decide(store, candidate, command_id="candidate-decision", decision=decision,
                         actor_type=actor_type, changes=changes, diff_summary=summary)
        assert receipt["response"]["candidate_decision"]["review_status"] == expected_status
        assert receipt["response"]["candidate_decision"]["identity_verification"] == "self_declared"
        assert decide(store, candidate, command_id="candidate-decision", decision=decision,
                      actor_type=actor_type, changes=changes, diff_summary=summary,
                      expected_task=applied)["idempotent_replay"]
        after = task(store)
        records = after["extensions"]["professional"]["review_decisions"]
        assert len(records) == 1
        record = records[0]
        assert record["source_annotation"] == candidate
        assert record["source_annotation_sha256"] == candidate_sha
        assert record["generation_lineage"]["job_id"] == queued["id"]
        assert record["generation_lineage"]["result_sha256"] == jobs.queue.get_job(queued["id"])["commit"]["artifact"]["sha256"]
        if decision == "rejected":
            assert all(item["id"] != candidate["id"] for item in after["annotations"])
        else:
            active = next(item for item in after["annotations"] if item["id"] == candidate["id"])
            assert active["provenance"]["actor_id"] == "test-reviewer"
            assert active["review_status"] == expected_status
            if actor_type == "human":
                assert active["review_binding"]["identity_verification"] == "self_declared"
            else:
                assert "review_binding" not in active
        document = json.loads(PStore.export_professional(store, after["id"])["data"])
        assert document["task"]["extensions"]["professional"]["review_decisions"][0] == record
        declared_actors = {actor["id"]: actor["type"] for actor in document["actors"]}
        assert declared_actors[candidate["provenance"]["actor_id"]] == "tool"
        assert declared_actors["test-reviewer"] == actor_type
        destination = WorkspaceStore(tmp_path / "candidate-decision-import")
        destination.create()
        try:
            destination.import_source(clip)
            target = task(destination)
            PStore.apply_import(destination, target["id"], command_id="decision-import",
                                expected_revision=target["revision"],
                                expected_sha256=target["content_sha256"],
                                document=document, request_body=b"{}")
            destination.close()
            destination.open()
            imported = task(destination)
            assert imported["extensions"]["professional"]["review_decisions"] == records
        finally:
            destination.close()
    finally:
        jobs.close()


def test_professional_import_cannot_silently_replace_active_model_candidate(store, clip):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        before = task(store)
        queued = submit(jobs, command_id="protected-import-submit")["jobs"][0]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        jobs.apply(queued["id"], command_id="protected-import-apply",
                   expected_revision=before["revision"], expected_sha256=before["content_sha256"],
                   request_body=b"{}")
        current = task(store)
        candidate = current["annotations"][0]
        document = json.loads(PStore.export_professional(store, current["id"])["data"])
        fail_code("CANDIDATE_DECISION_REQUIRED", lambda: PStore.apply_import(
            store, current["id"], command_id="protected-import-overwrite",
            expected_revision=current["revision"], expected_sha256=current["content_sha256"],
            document=document, request_body=b"{}"))
        unchanged = task(store)
        assert unchanged == current
        assert unchanged["annotations"][0]["id"] == candidate["id"]
    finally:
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
def test_http_vision_chain_submit_result_apply_status(store, clip, monkeypatch):
    """Exercise the loopback HTTP vision routes end to end on a real import."""
    import threading
    import time
    import urllib.request
    from trace_annotate.workspace.server import TOKEN_HEADER, make_server

    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(store.root / "missing-clip-cache"))
    monkeypatch.setenv("TRACE_ANNOTATE_VLM_CACHE", str(store.root / "missing-vlm-cache"))
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
        candidate = status["candidates"][0]
        assert status["candidate_sha256"][candidate["id"]] == store.candidate_digest(candidate)
        decision = post(f"tasks/{task_id}/commands", {
            "command_id": "http-candidate-accept", "expected_revision": status["revision"],
            "expected_sha256": status["content_sha256"], "operation": "decide_candidate",
            "payload": {"annotation_id": candidate["id"], "decision": "accepted",
                        "actor_id": "test-assistant", "actor_type": "assistant",
                        "expected_candidate_sha256": status["candidate_sha256"][candidate["id"]],
                        "changes": {}, "diff_summary": ""}})
        assert decision["candidate_decision"]["review_status"] == "unreviewed"
        decided = get(f"tasks/{task_id}")
        assert next(a for a in decided["annotations"] if a["id"] == candidate["id"])["review_status"] == "unreviewed"
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
    for item in result["evidence"]:
        relative = PurePosixPath(item["relative_path"])
        is_diagnostic = (relative.name in {"commands.jsonl", "decode.stdout", "decode.stderr",
                                          "select.filter", "decode-tool.stdout", "decode-tool.stderr"}
                         and relative.parent.name.startswith(".decode-"))
        is_global_scan = (relative.parent.name == "transitions-global" and relative.name in {
            "commands.jsonl", "discovery.json", "frames.gray", "scan.stderr", "scan.stdout"})
        is_model_run = ("models" in relative.parts and relative.name in {
            "request.json", "result.json", "progress.json", "process-receipt.json", "worker.log"})
        assert relative.suffix == ".png" or str(relative).endswith("audio/acoustic.m4a") or is_diagnostic or is_global_scan or is_model_run
        evidence = store.root / item["relative_path"]
        assert evidence.stat().st_size == int(item["size_bytes"])
        assert V.stream_sha256(evidence) == item["sha256"]
    assert any(item["relative_path"].endswith(".png") for item in result["evidence"])
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


def test_extractor_initialization_failure_never_publishes_local_path(
        store, monkeypatch):
    private_path = r"D:\private\models\face.onnx"

    def explode(*_args, **_kwargs):
        raise RuntimeError("cannot open " + private_path)

    monkeypatch.setattr(V, "FaceExtractor", explode)
    jobs = V.VisionJobs(store)
    jobs._load_extractors(V.MetricThresholds())
    options = jobs.options()
    assert options["extractor_error"] == "extractor_initialization_failed"
    assert "private" not in json.dumps(options).lower()


def test_new_visual_wrappers_never_publish_provider_exception_paths():
    private_path = r"D:\private\models\provider.bin"

    class ExplodingDetector:
        def detect_people(self, _frame):
            raise RuntimeError("cannot open " + private_path)

        def detect_objects(self, _frame):
            raise RuntimeError("cannot open " + private_path)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    person = V._run_person(frame, ExplodingDetector())
    objects = V._run_object(frame, ExplodingDetector())
    assert person["reason"] == "person_inference_failed"
    assert objects["reason"] == "object_inference_failed"
    assert "private" not in json.dumps([person, objects]).lower()


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
