"""Durable vision candidate jobs: scene cuts, keyframes, honest extractor status,
candidate-only imports and the human upgrade path.

The actual extractors (scene/face/ocr/clip) follow their own real behavior; this
suite exercises the durable queue boundary, the candidate contract and the
review discipline with a real ffmpeg fixture so scene detection runs for real.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading

import pytest

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
    runs = result["model_runs"]
    assert len(runs) == 9
    assert {run["id"] for run in runs} == {V._module_run_id(queued["id"], name) for name in
                                           ("face", "ocr", "clip", "vlm", "scene", "motion",
                                            "actions", "acoustics", "boundary")}
    for run in runs:
        assert SS.validate_model_run(run) == run
    assert any(run["status"] == "not_run" and run["failure_reason"] for run in runs)
    assert all(run["status"] == "success" or run["failure_reason"] for run in runs)
    assert len(result["scenes"]) >= 1
    assert all({"start", "end", "duration"} <= set(scene) for scene in result["scenes"])
    assert result["evidence"] and all(item["relative_path"].startswith("tmp/vision/jobs/") for item in result["evidence"])
    assert str(store.root) not in json.dumps(result)
    assert task(store) == initial
    assert jobs.close()["stopped"]


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
