"""Persistent per-video history and stale apply regression tests."""
from __future__ import annotations

import json

from tests.test_workspace_server import assert_error, command, http_video, request, running
from tests.test_workspace_vision_jobs import clip, fail_code, store, task
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.store import WorkspaceStore


def test_history_is_persisted_task_scoped_and_stably_paginated(tmp_path, http_video):
    project = WorkspaceStore(tmp_path / "history")
    project.create()
    try:
        first_task = project.import_source(http_video)["task_id"]
        second_task = project.import_source(http_video)["task_id"]
        queue = DurableQueue(project, clock=lambda: 1_700_000_000.0)
        windows = [
            None,
            {"start": {"n": "0", "d": "1"}, "end": {"n": "1", "d": "1"}},
            {"start": {"n": "1", "d": "1"}, "end": {"n": "2", "d": "1"}},
        ]
        jobs = [queue.enqueue(first_task, operation=V.OPERATION, processor_version=V.VERSION,
                              parameters={"window": window, "private": "not in history"})
                for window in windows]
        duplicate = queue.enqueue(first_task, operation=V.OPERATION, processor_version=V.VERSION,
                                  parameters=jobs[0]["parameters"])
        assert duplicate["id"] == jobs[0]["id"]
        other = queue.enqueue(second_task, operation=V.OPERATION, processor_version=V.VERSION)
        unrelated = queue.enqueue(first_task, operation="index-media", processor_version="1")
        for job in (*jobs, other, unrelated):
            queue.cancel(job["id"])
    finally:
        project.close()

    reopened = WorkspaceStore(project.root)
    reopened.open()
    try:
        with running(reopened) as server:
            route = f"/api/v1/projects/current/tasks/{first_task}/vision-history"
            expected = sorted(jobs, key=lambda job: (job["created_at"], job["id"]), reverse=True)
            before = reopened.get_task(first_task)
            page = request(server, "GET", route + "?limit=2")
            assert page[0] == 200
            assert page[1]["Cache-Control"] == "no-store"
            assert page[2] == {
                "task_id": first_task,
                "jobs": [
                    {**{key: job[key] for key in ("id", "created_at", "processor_version",
                                                 "input_revision", "input_sha256")},
                     "state": "cancelled", "error_code": "CANCELLED",
                     "window": job["parameters"]["window"]} for job in expected[:2]
                ],
                "next_cursor": expected[1]["id"],
            }

            # A newer submission between pages must not duplicate a previous row.
            with reopened.transaction():
                newest = server.queue.enqueue(first_task, operation=V.OPERATION, processor_version=V.VERSION,
                                              parameters={"submission": "newer"})
                server.queue.cancel(newest["id"])
            assert request(server, "GET", route)[2]["jobs"][0]["id"] == newest["id"]
            tail = request(server, "GET", route + "?limit=2&cursor=" + page[2]["next_cursor"])
            assert tail[0] == 200
            assert [job["id"] for job in tail[2]["jobs"]] == [expected[2]["id"]]
            assert tail[2]["next_cursor"] is None
            for cursor in (other["id"], unrelated["id"], "missing-job", ""):
                assert_error(request(server, "GET", route + "?cursor=" + cursor), 400, "INVALID_QUERY")
            for query in ("limit=0", "limit=101", "limit=01", "limit=1&limit=2",
                          "limit=1&offset=0", "cursor=%ZZ"):
                assert_error(request(server, "GET", route + "?" + query), 400, "INVALID_QUERY")
            assert_error(request(server, "GET", route, authorized=False), 403, "INVALID_TOKEN")
            assert_error(request(server, "GET", route.replace(first_task, "task-missing")),
                         404, "TASK_NOT_FOUND")
            assert reopened.get_task(first_task) == before
    finally:
        reopened.close()


def test_older_vision_result_remains_readable_but_cannot_be_applied_to_new_revision(store):
    jobs = V.VisionJobs(store, clip_model_path="nonexistent-clip-weights",
                        vlm_model_path="nonexistent-vlm-weights")
    try:
        original = task(store)
        batch = jobs.submit(original["id"], command_id="freeze-old-result",
                            expected_revision=original["revision"], expected_sha256=original["content_sha256"])
        job_id = batch["jobs"][0]["id"]
        assert jobs.run_pending()[0]["state"] == "succeeded"
        original_result = jobs.result(job_id)
        _, edit = command(store, "edit-after-vision")
        annotation = edit["payload"]["annotation"]
        annotation["type"] = "whole_clip"
        annotation.pop("start")
        annotation.pop("end")
        store.apply_task_command(task_id=original["id"], request_body=json.dumps(edit).encode(), **edit)
        edited = task(store)
        objects = set(store.objects_dir.iterdir())
        fail_code("VISION_STALE", lambda: jobs.apply(
            job_id, command_id="stale-vision-apply", expected_revision=edited["revision"],
            expected_sha256=edited["content_sha256"], request_body=b"{}"))
        fail_code("COMMAND_UNKNOWN", lambda: store.get_command_receipt("stale-vision-apply"))
        assert set(store.objects_dir.iterdir()) == objects
        assert task(store) == edited
        history = jobs.history(original["id"])
        assert history["jobs"][0]["input_revision"] == original["revision"]
        assert jobs.result(job_id) == original_result
    finally:
        jobs.close()
