"""Read-only hand/object geometry is bound to a verified vision result."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.server import TOKEN_HEADER, make_server
from trace_annotate.workspace.store import WorkspaceStore
from trace_annotate.workspace.vision_jobs import VisionJobs


def _completed_job(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the real vision fixture")
    clip = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=blue:s=64x64:r=10:d=1", "-c:v", "libx264", "-pix_fmt",
        "yuv420p", str(clip),
    ], capture_output=True, check=True, timeout=60)
    store = WorkspaceStore(tmp_path / "workspace")
    store.create()
    store.import_source(clip)
    task = store.get_task(store.list_tasks()[0]["task_id"])
    jobs = VisionJobs(
        store, face_model_path="missing-face-model",
        clip_model_path="missing-clip-model", vlm_model_path="missing-vlm-model",
        person_model_path="missing-person-model",
        landmark_model_path="missing-landmark-model",
        hand_model_path="missing-hand-model",
        action_video_model_path="missing-action-model")
    batch = jobs.submit(task["id"], command_id="hand-object-geometry-submit",
                        expected_revision=task["revision"],
                        expected_sha256=task["content_sha256"])
    job_id = batch["jobs"][0]["id"]
    queued_status = jobs.job_status(job_id)
    assert queued_status["state"] == "queued"
    assert queued_status["result_sha256"] is None
    jobs.run_pending()
    return store, jobs, task, job_id


def test_geometry_service_and_authenticated_http_are_read_only_and_hash_bound(tmp_path):
    store, jobs, original_task, job_id = _completed_job(tmp_path)
    server = None
    thread = None
    try:
        result = jobs.result(job_id)
        derived = jobs.hand_object_geometry(job_id)
        assert derived["kind"] == "traceannotate.hand-object-geometry"
        assert derived["version"] == 1
        assert derived["source"]["job_id"] == job_id
        assert derived["source"]["task_id"] == original_task["id"]
        assert derived["source"]["input_revision"] == original_task["revision"]
        assert derived["source"]["vision_result_sha256"] == S.sha256_text(result)
        assert derived["source"]["vision_result_sha256"] == jobs.queue.get_job(job_id)["commit"]["artifact"]["sha256"]
        completed_status = jobs.job_status(job_id)
        assert completed_status["state"] == "succeeded"
        assert completed_status["has_candidate"] is True
        assert completed_status["result_sha256"] == derived["source"]["vision_result_sha256"]
        assert derived["geometry"]["status"] == "not_run"
        assert derived["geometry"]["pairs"] == []
        assert derived["omission"]["sampled_module_coverage_complete"] is False
        assert "not_persisted" in derived["scope"]
        assert store.get_task(original_task["id"]) == original_task

        server = make_server(store)
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        port = server.server_address[1]
        url = (f"http://127.0.0.1:{port}/api/v1/projects/current/jobs/"
               f"{job_id}/hand-object-geometry")
        headers = {"Host": f"127.0.0.1:{port}",
                   "Origin": f"http://127.0.0.1:{port}", TOKEN_HEADER: server.token}
        status_url = (f"http://127.0.0.1:{port}/api/v1/projects/current/jobs/"
                      f"{job_id}/vision-status")
        with urllib.request.urlopen(urllib.request.Request(
                status_url, headers=headers), timeout=30) as response:
            assert json.loads(response.read())["result_sha256"] == completed_status["result_sha256"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert json.loads(response.read()) == derived
        with pytest.raises(urllib.error.HTTPError) as unauthenticated:
            urllib.request.urlopen(urllib.request.Request(
                url, headers={"Host": f"127.0.0.1:{port}",
                              "Origin": f"http://127.0.0.1:{port}"}), timeout=30)
        assert unauthenticated.value.code == 403
        assert json.loads(unauthenticated.value.read())["error"]["code"] == "INVALID_TOKEN"
        assert store.get_task(original_task["id"]) == original_task

        descriptor = jobs.queue.get_job(job_id)["commit"]["artifact"]
        commit_id = jobs.queue.get_job(job_id)["commit"]["id"]
        with store.transaction():
            original_commit = store.conn.execute(
                "SELECT body FROM durable_commits WHERE id=?", (commit_id,)).fetchone()[0]
            store.conn.execute("UPDATE durable_commits SET body=? WHERE id=?",
                               ("{malformed-json", commit_id))
        with pytest.raises(WorkspaceError) as malformed:
            jobs.job_status(job_id)
        assert malformed.value.code == "RECOVERY_REQUIRED"
        with pytest.raises(urllib.error.HTTPError) as malformed_http:
            urllib.request.urlopen(urllib.request.Request(status_url, headers=headers), timeout=30)
        assert malformed_http.value.code == 500
        assert json.loads(malformed_http.value.read())["error"]["code"] == "RECOVERY_REQUIRED"
        with store.transaction():
            store.conn.execute("UPDATE durable_commits SET body=? WHERE id=?",
                               (original_commit, commit_id))
        assert jobs.job_status(job_id)["result_sha256"] == descriptor["sha256"]

        path = store.root / descriptor["relative_path"]
        original_bytes = path.read_bytes()
        path.write_bytes(b"x" + original_bytes[1:])
        with pytest.raises(WorkspaceError) as corrupt:
            jobs.hand_object_geometry(job_id)
        assert corrupt.value.code == "RECOVERY_REQUIRED"
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)
        assert rejected.value.code == 500
        assert json.loads(rejected.value.read())["error"]["code"] == "RECOVERY_REQUIRED"
        with store.transaction():
            store.conn.execute("UPDATE artifacts SET sha256=? WHERE id=?",
                               ("b" * 64, descriptor["id"]))
        with pytest.raises(WorkspaceError) as invalid_commit:
            jobs.job_status(job_id)
        assert invalid_commit.value.code == "RECOVERY_REQUIRED"
        assert store.get_task(original_task["id"]) == original_task
    finally:
        if server is not None:
            server.shutdown()
            if thread is not None:
                thread.join(timeout=10)
            server.server_close()
        jobs.close()
        store.close()


def test_legacy_result_without_hand_fields_has_explicit_not_run(monkeypatch):
    legacy = {"job_id": "job-legacy", "task_id": "tsk-legacy",
              "input_revision": 0, "file_sha256": "a" * 64,
              "keyframes": [{"scene_index": 0, "frame_index": "0",
                             "timestamp": {"n": "0", "d": "1"},
                             "evidence_sha256": "b" * 64}]}
    jobs = object.__new__(VisionJobs)
    monkeypatch.setattr(jobs, "result", lambda job_id: legacy)
    derived = jobs.hand_object_geometry("job-legacy")
    assert derived["geometry"]["status"] == "not_run"
    assert derived["geometry"]["reason"] == "hand_landmarks_unavailable"
    assert derived["source"]["vision_result_sha256"] == S.sha256_text(legacy)
    assert derived["geometry"]["pairs"] == []
