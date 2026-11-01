"""Actual HTTP previews, real encoded renders, frozen pairing and recovery.

No fake renderer supplies success bytes. Controlled failures/cancellation act
at real FFmpeg boundaries; small recipes declare exactly which fields display.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading

from PIL import Image, ImageChops
import pytest

from tests.test_workspace_server import request, assert_error
from trace_annotate.temporal import preparation as P
from trace_annotate.workspace import backup as B, professional_schema as PS
from trace_annotate.workspace import render_jobs as J, rendering as R, schema as S
from trace_annotate.workspace.analysis_jobs import AnalysisJobs
from trace_annotate.workspace.server import make_server, TOKEN_HEADER
from trace_annotate.workspace.store import WorkspaceStore

API = "/api/v1/projects/current"


@pytest.fixture(scope="module")
def render_clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Actual render HTTP acceptance requires FFmpeg and ffprobe")
    source = tmp_path_factory.mktemp("render-http-source") / "two-second-stereo.mov"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "testsrc2=duration=2:size=160x120:rate=10", "-f", "lavfi", "-i",
        "aevalsrc=0.5*sin(2*PI*997*t)|0.25*sin(2*PI*499*t):s=48000:d=2:c=stereo",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s24le", str(source)], check=True, capture_output=True, timeout=30)
    return source


def task(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


@pytest.fixture()
def render_store(tmp_path, render_clip):
    store = WorkspaceStore(tmp_path / "workspace")
    store.create()
    try:
        store.import_source(render_clip)
        before = task(store)
        store.apply_task_command(command_id="cmd-synthetic-annotation", task_id=before["id"], expected_revision=before["revision"],
            expected_sha256=before["content_sha256"], operation="add_annotation", payload={"annotation": {
                "id": "ann-render", "type": "segment", "label": "合成素材验证", "text": "完整正文 -42，数量1,234。",
                "track": "camera", "tags": ["synthetic"], "start": {"seconds": S.rational(0)},
                "end": {"seconds": S.rational("1.5")}}}, request_body=b"actual-synthetic-only-fixture")
        yield store
    finally:
        store.close()


@contextmanager
def serving(store, monkeypatch):
    # Expose deterministic queued states; execution below still calls the real
    # registered-playback processor, renderer and independent decode checks.
    with monkeypatch.context() as patch:
        patch.setattr(AnalysisJobs, "start", lambda self: self)
        patch.setattr(J.RenderJobs, "start", lambda self: self)
        server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        assert not thread.is_alive()


def recipe(server, *, position="bottom", full=True):
    response = request(server, "GET", API + "/render-options")
    assert response[0] == 200 and response[2]["available"] is True, response
    value = copy.deepcopy(response[2]["default_recipe"])
    value["position"] = position
    if full:
        # Keep the product's22/26px typography.512px fits the declared full
        # text on this small160px-wide synthetic video; nothing is truncated.
        value["panel_extent_px"] = 512
    else:
        value.update(panel_extent_px=128, body_font_px=12, title_font_px=14, padding_px=8, line_spacing_px=0,
                     fields=["label"], tracks=["camera"])
    return value


def preview_body(store, chosen, frame="3", **changes):
    current = task(store)
    body = {"expected_revision": current["revision"], "expected_sha256": current["content_sha256"],
            "recipe": chosen, "frame_index": frame}
    body.update(changes)
    return body


def submission(server, store, *, command_id="cmd-render", chosen=None, **changes):
    current = task(store)
    body = {"command_id": command_id, "expected_revision": current["revision"],
            "expected_sha256": current["content_sha256"], "recipe": chosen or recipe(server)}
    body.update(changes)
    path = API + "/tasks/" + current["id"] + "/render-jobs"
    return path, body


def submit(server, store, **kwargs):
    path, body = submission(server, store, **kwargs)
    response = request(server, "POST", path, body)
    assert response[0] == 202, response
    batch = response[2]
    assert len(batch["jobs"]) == 1 and batch["jobs"][0]["state"] == "queued"
    return batch["jobs"][0], batch, path, body


def complete(server, store, **kwargs):
    job, batch, path, body = submit(server, store, **kwargs)
    runs = server.render_jobs.run_pending(batch_id=batch["id"])
    assert len(runs) == 1 and runs[0]["state"] == "succeeded", runs
    response = request(server, "GET", API + "/jobs/" + job["id"] + "/render-result")
    assert response[0] == 200, response
    return job, response[2], path, body


def test_render_options_use_the_bundled_hash_and_license_without_task_mutation(render_store, monkeypatch):
    store = render_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        response = request(server, "GET", API + "/render-options")
        assert response[0] == 200 and response[2]["available"]
        options = response[2]
        assert options["font"]["sha256"] == hashlib.sha256(J.DEFAULT_FONT.read_bytes()).hexdigest()
        assert "Open Font License" in options["font"]["license"]
        assert options["default_recipe"]["font_sha256"] == options["font"]["sha256"]
        assert options["default_recipe"]["font_license"] == options["font"]["license"]
        assert int(options["limits"]["download_max_bytes"]) == 512 * 1024 * 1024
        assert task(store) == before and server.queue.list_batches() == []


@pytest.mark.parametrize("position", ["top", "bottom", "left", "right"])
def test_actual_png_preview_uses_requested_frame_and_full_source_pixels_at_custom_position(render_store, monkeypatch, position):
    store = render_store
    before = task(store)
    media = store.get_media(before["media_id"])
    with serving(store, monkeypatch) as server:
        chosen = recipe(server, position=position, full=False)
        response = request(server, "POST", API + "/tasks/" + before["id"] + "/render-preview", preview_body(store, chosen))
        assert response[0] == 200 and response[1]["Content-Type"] == "image/png", response
        assert response[1]["X-TraceAnnotate-Task-Revision"] == str(before["revision"])
        assert response[1]["X-TraceAnnotate-Content-SHA256"] == before["content_sha256"]
        assert response[1]["X-TraceAnnotate-Frame-Index"] == "3"
        assert re.fullmatch(r"[0-9a-f]{64}", response[1]["X-TraceAnnotate-Render-Plan-SHA256"])
        original = server.image_service.image(media["id"], 3)
        with Image.open(io.BytesIO(response[2])) as composite, Image.open(io.BytesIO(original["data"])) as frame:
            assert composite.size == ((288, 120) if position in {"left", "right"} else (160, 248))
            x, y = (128 if position == "left" else 0), (128 if position == "top" else 0)
            actual_source = composite.crop((x, y, x + 160, y + 120)).convert("RGB")
            assert ImageChops.difference(actual_source, frame.convert("RGB")).getbbox() is None
            box = {"top": (0, 0, 160, 128), "bottom": (0, 120, 160, 248),
                   "left": (0, 0, 128, 120), "right": (160, 0, 288, 120)}[position]
            panel = composite.crop(box).convert("RGB")
            red, green, blue = panel.split()
            assert ImageChops.difference(red, green).getbbox() is None
            assert ImageChops.difference(red, blue).getbbox() is None
            assert panel.getextrema()[0] == (0, 255)
        assert task(store) == before and server.queue.list_batches() == []


@pytest.mark.parametrize("frame", [3, True, "03", "-1", "1.0", "", None])
def test_preview_requires_canonical_string_frame_without_saving_anything(render_store, monkeypatch, frame):
    store = render_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        body = preview_body(store, recipe(server, full=False), frame=frame)
        assert_error(request(server, "POST", API + "/tasks/" + before["id"] + "/render-preview", body), 422, "SCHEMA_INVALID")
        assert task(store) == before and server.queue.list_batches() == []


def test_preview_cas_bounds_overflow_and_busy_are_explicit_not_scaled_or_saved(render_store, monkeypatch):
    store = render_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        url = API + "/tasks/" + before["id"] + "/render-preview"
        chosen = recipe(server)
        assert_error(request(server, "POST", url, preview_body(store, chosen, expected_revision=99)), 409, "REVISION_CONFLICT")
        assert_error(request(server, "POST", url, preview_body(store, chosen, expected_sha256="0" * 64)), 409, "MEDIA_CHANGED")
        assert_error(request(server, "POST", url, preview_body(store, chosen, frame="20")), 422, "FRAME_OUT_OF_RANGE")
        overflow = copy.deepcopy(chosen)
        overflow.update(panel_extent_px=64, body_font_px=96, title_font_px=128)
        assert_error(request(server, "POST", url, preview_body(store, overflow)), 422, "RENDER_TEXT_OVERFLOW")
        assert server.render_preview_lock.acquire(blocking=False)
        try:
            assert_error(request(server, "POST", url, preview_body(store, chosen)), 409, "PREVIEW_BUSY")
        finally:
            server.render_preview_lock.release()
        assert task(store) == before and server.queue.list_batches() == []


def test_real_validated_files_are_not_downloadable_before_atomic_queue_commit(render_store, monkeypatch):
    store = render_store
    before = task(store)
    reached, release = threading.Event(), threading.Event()
    render = R.render_video
    def after_actual_validation(*args, **kwargs):
        value = render(*args, **kwargs)
        reached.set()
        assert release.wait(20)
        return value
    monkeypatch.setattr(R, "render_video", after_actual_validation)
    with serving(store, monkeypatch) as server:
        job, batch, _, _ = submit(server, store)
        errors = []
        results = []
        def run():
            try:
                results.extend(server.render_jobs.run_pending(batch_id=batch["id"]))
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert reached.wait(30)
            actual = list((store.tmp_dir / "render/jobs" / job["id"]).glob("*/render/validated/annotation.mp4"))
            assert len(actual) == 1 and actual[0].stat().st_size > 0
            for suffix in ("render-result", "render-video", "render-json"):
                assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/" + suffix), 409, "RENDER_NOT_READY")
            assert task(store) == before
        finally:
            release.set()
            worker.join(timeout=30)
        assert not worker.is_alive() and not errors
        assert results[0]["state"] == "succeeded"
        assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-video")[0] == 200


def test_real_frozen_video_and_json_remain_paired_after_later_edits_and_support_exact_range(render_store, monkeypatch):
    store = render_store
    frozen = task(store)
    with serving(store, monkeypatch) as server:
        job, result, path, submitted_body = complete(server, store)
        assert task(store) == frozen
        manifest = result["manifest"]
        assert result["input_binding"] == result["current_binding"]
        assert manifest["renderer"]["validation"]["all_frame_pts_exact"] is True
        assert manifest["renderer"]["validation"]["full_decode_exit"] == 0
        assert set(manifest["files"]) == set(J.FILES)
        assert store.conn.execute("SELECT COUNT(*) FROM artifact_references WHERE owner_kind='render_manifest' AND owner_id=?", (job["id"],)).fetchone()[0] == 5
        store.apply_task_command(command_id="cmd-later-edit", task_id=frozen["id"], expected_revision=frozen["revision"],
            expected_sha256=frozen["content_sha256"], operation="update_annotation", payload={"annotation_id": "ann-render", "changes": {"label": "稍后修改"}}, request_body=b"later edit")
        current = task(store)
        result = request(server, "GET", API + "/jobs/" + job["id"] + "/render-result")[2]
        assert result["input_binding"]["revision"] == frozen["revision"] and result["current_binding"]["revision"] == current["revision"]
        exported = request(server, "GET", API + "/jobs/" + job["id"] + "/render-json")
        assert exported[0] == 200
        document = PS.validate_document(exported[2])
        assert document["task"] == frozen and document["task"] != current
        assert document["history"]["scope"] == "not_included" and document["history"]["revisions"] == []
        assert "history not included" in exported[1]["X-TraceAnnotate-Export-Losses"]
        video = request(server, "GET", API + "/jobs/" + job["id"] + "/render-video")
        assert video[0] == 200 and video[1]["Content-Type"] == "video/mp4"
        descriptor = manifest["files"]["annotation.mp4"]
        assert hashlib.sha256(video[2]).hexdigest() == descriptor["sha256"] == video[1]["X-TraceAnnotate-File-SHA256"]
        assert len(video[2]) == int(descriptor["size_bytes"])
        for delivery in (video, exported):
            assert delivery[1]["X-TraceAnnotate-Task-Revision"] == str(frozen["revision"])
            assert delivery[1]["X-TraceAnnotate-Content-SHA256"] == frozen["content_sha256"]
            assert "-r" + str(frozen["revision"]) in delivery[1]["Content-Disposition"]
        partial = request(server, "GET", API + "/jobs/" + job["id"] + "/render-video", headers={"Range": "bytes=17-127"})
        assert partial[0] == 206 and partial[2] == video[2][17:128]
        assert partial[1]["Content-Range"] == f"bytes 17-127/{len(video[2])}" and partial[1]["Accept-Ranges"] == "bytes"
        invalid = request(server, "GET", API + "/jobs/" + job["id"] + "/render-video", headers={"Range": "bytes=999999999-"})
        assert invalid[0] == 416 and invalid[2] == b""
        assert request(server, "POST", path, submitted_body)[2]["jobs"][0]["id"] == job["id"]
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert task(store) == current


def test_real_render_http_cancel_stops_owned_encoder_and_cancelled_job_can_resume(render_store, monkeypatch):
    store = render_store
    before = task(store)
    run = P._run
    register = P._register_child
    running_encoder = {}
    cancelled = []
    def remember_child(process, name, command):
        record = register(process, name, command)
        if name == "encode":
            running_encoder["process"] = process
        return record
    monkeypatch.setattr(P, "_register_child", remember_child)
    with serving(store, monkeypatch) as server:
        job, batch, _, _ = submit(server, store)
        def with_http_cancel(command, directory, name, control, **kwargs):
            if name != "encode" or cancelled:
                return run(command, directory, name, control, **kwargs)
            def gate():
                process = running_encoder.get("process")
                if process is not None and process.poll() is None and not cancelled:
                    response = request(server, "POST", API + "/jobs/" + job["id"] + "/control", {"command_id": "cmd-real-cancel", "action": "cancel"})
                    assert response[0] == 200 and response[2]["state"] == "cancel_requested"
                    cancelled.append({"response": response[2], "observed_live_pid": process.pid})
                return control() if control is not None else False
            return run(command, directory, name, gate, **kwargs)
        monkeypatch.setattr(P, "_run", with_http_cancel)
        first = server.render_jobs.run_pending(batch_id=batch["id"])
        assert cancelled and first[0]["state"] == "cancelled" and first[0]["commit"] is None
        logs = list((store.tmp_dir / "render/jobs" / job["id"]).glob("*/render/work/commands.jsonl"))
        assert len(logs) == 1
        records = [json.loads(line) for line in logs[0].read_text("utf-8").splitlines()]
        encoded = next(record for record in records if record["name"] == "encode")
        assert encoded["pid"] == cancelled[0]["observed_live_pid"]
        assert encoded["cancelled_or_interrupted"] is True and encoded["exit_code"] is not None
        assert encoded["pid"] not in {p["pid"] for p in P.live_media_processes()}
        assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/render-video"), 409, "RENDER_NOT_READY")
        response = request(server, "POST", API + "/jobs/" + job["id"] + "/control", {"command_id": "cmd-resume-render", "action": "retry_cancelled"})
        assert response[0] == 200 and response[2]["state"] == "queued"
        second = server.render_jobs.run_pending(batch_id=batch["id"])
        assert second[0]["state"] == "succeeded" and second[0]["attempt"] == 2
        assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-video")[0] == 200
        assert task(store) == before and not P.live_media_processes()


def test_real_ffmpeg_failure_has_no_download_then_explicit_retry_succeeds(render_store, monkeypatch):
    store = render_store
    before = task(store)
    run = P._run
    first = True
    def invalid_first_encode(command, directory, name, control, **kwargs):
        nonlocal first
        if name == "encode" and first:
            first = False
            command = command[:-1] + ["-traceannotate_invalid_option", command[-1]]
        return run(command, directory, name, control, **kwargs)
    monkeypatch.setattr(P, "_run", invalid_first_encode)
    with serving(store, monkeypatch) as server:
        job, batch, _, _ = submit(server, store)
        failure = server.render_jobs.run_pending(batch_id=batch["id"])
        assert failure[0]["state"] == "failed" and failure[0]["error_code"] == "RENDER_FAILED"
        for suffix in ("render-result", "render-video", "render-json"):
            assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/" + suffix), 409, "RENDER_NOT_READY")
        control = {"command_id": "cmd-retry-render", "action": "retry_failed"}
        response = request(server, "POST", API + "/jobs/" + job["id"] + "/control", control)
        assert response[0] == 200 and response[2]["state"] == "queued"
        assert request(server, "POST", API + "/jobs/" + job["id"] + "/control", control)[2] == response[2]
        success = server.render_jobs.run_pending(batch_id=batch["id"])
        assert success[0]["state"] == "succeeded" and success[0]["attempt"] == 2
        assert task(store) == before
        assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-json")[0] == 200


def test_real_render_pins_survive_gc_store_reopen_and_verified_backup_restore(render_store, monkeypatch, tmp_path):
    store = render_store
    with serving(store, monkeypatch) as server:
        job, initial, _, _ = complete(server, store)
        video_before = request(server, "GET", API + "/jobs/" + job["id"] + "/render-video")[2]
        json_before = request(server, "GET", API + "/jobs/" + job["id"] + "/render-json")[2]
    garbage = store._put_artifact(b"actual-unbound-render-http-control")
    assert store.sweep_unbound_artifacts(apply=True)["applied"]
    assert not (store.root / garbage["relative_path"]).exists()
    for descriptor in initial["manifest"]["files"].values():
        assert (store.root / descriptor["relative_path"]).is_file()
    backup = tmp_path / "backup"
    B.backup_workspace(store, backup)
    store.close()
    store.open()
    with serving(store, monkeypatch) as server:
        assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-result")[2] == initial
        assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-video")[2] == video_before
    restored_root = tmp_path / "restored"
    report = B.restore_workspace(backup, restored_root)
    assert report["jobs_restarted"] is False and report["commands_replayed"] is False
    restored = WorkspaceStore(restored_root)
    restored.open()
    try:
        with serving(restored, monkeypatch) as server:
            assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-result")[2] == initial
            assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-video")[2] == video_before
            assert request(server, "GET", API + "/jobs/" + job["id"] + "/render-json")[2] == json_before
    finally:
        restored.close()


def test_actual_render_artifact_tampering_revokes_all_three_download_views(render_store, monkeypatch):
    store = render_store
    with serving(store, monkeypatch) as server:
        job, result, _, _ = complete(server, store)
        path = store.root / result["manifest"]["files"]["annotation.mp4"]["relative_path"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tampered")
            for suffix in ("render-result", "render-video", "render-json"):
                assert_error(request(server, "GET", API + "/jobs/" + job["id"] + "/" + suffix), 500, "RECOVERY_REQUIRED")
        finally:
            path.write_bytes(original)


@pytest.mark.parametrize("route", ["options", "preview", "submit", "result", "video", "json"])
def test_render_routes_require_token_and_exact_origin_before_resource_lookup(render_store, monkeypatch, route):
    store = render_store
    with serving(store, monkeypatch) as server:
        base = task(store)
        method, body = "GET", None
        if route == "options":
            path = API + "/render-options"
        elif route in {"preview", "submit"}:
            method = "POST"
            suffix = "render-preview" if route == "preview" else "render-jobs"
            path = API + "/tasks/" + base["id"] + "/" + suffix
            body = {}
        else:
            path = API + "/jobs/job-absent/render-" + route
        assert_error(request(server, method, path, body, authorized=False), 403, "INVALID_TOKEN")
        assert_error(request(server, method, path, body, headers={TOKEN_HEADER: "wrong"}), 403, "INVALID_TOKEN")
        assert_error(request(server, method, path, body, headers={"Origin": "https://attacker.example"}), 403, "FORBIDDEN_ORIGIN")
        assert_error(request(server, method, path, body, headers={"Host": "attacker.example"}), 403, "FORBIDDEN_ORIGIN")
        assert task(store) == base and server.queue.list_batches() == []


@pytest.mark.parametrize("change", ["path", "font", "license", "encoding", "duplicate_fields"])
def test_render_submit_refuses_untrusted_paths_font_encoding_and_invalid_recipe(render_store, monkeypatch, change):
    store = render_store
    before = task(store)
    with serving(store, monkeypatch) as server:
        chosen = recipe(server)
        path, body = submission(server, store, chosen=chosen)
        if change == "path":
            body["source_path"] = "C:/Users/private/input.mp4"
        elif change == "font":
            body["recipe"]["font_sha256"] = "0" * 64
        elif change == "license":
            body["recipe"]["font_license"] = "unknown alternate declaration"
        elif change == "encoding":
            body["recipe"]["encoding"]["video_codec"] = "untrusted-shell-command"
        else:
            body["recipe"]["fields"] = ["text", "text"]
        result = request(server, "POST", path, body)
        assert result[0] == 422, result
        assert result[2]["error"]["code"] in {"SCHEMA_INVALID", "RENDER_FONT_CHANGED", "UNSUPPORTED_RENDER_ENCODING", "INVALID_RENDER_RECIPE"}
        assert task(store) == before and server.queue.list_batches() == []


@pytest.mark.parametrize("raw", [b'{"x":NaN}', b'{"x":1,"x":2}', b'null', b'[]'])
def test_render_write_routes_parse_strict_json_and_reject_nonobject_bodies(render_store, monkeypatch, raw):
    store = render_store
    with serving(store, monkeypatch) as server:
        for route in ("render-preview", "render-jobs"):
            response = request(server, "POST", API + "/tasks/" + task(store)["id"] + "/" + route, raw)
            assert response[0] == (400 if raw.startswith(b'{') else 422)
            assert response[2]["error"]["code"] == ("INVALID_JSON" if raw.startswith(b'{') else "SCHEMA_INVALID")
        assert server.queue.list_batches() == []


def test_render_paths_queries_and_oversize_inputs_do_not_read_local_files_or_enqueue(render_store, monkeypatch):
    store = render_store
    with serving(store, monkeypatch) as server:
        path = API + "/tasks/" + task(store)["id"] + "/render-jobs"
        assert_error(request(server, "POST", path, b'{"padding":"' + b"x" * (1024 * 1024) + b'"}'), 413, "LIMIT_EXCEEDED")
        assert_error(request(server, "GET", API + "/jobs/job-absent/render-video?path=../../secret"), 404, "NOT_FOUND")
        response = request(server, "GET", API + "/jobs/../render-video")
        assert response[0] in {404, 422} and response[0] != 500
        assert server.queue.list_batches() == []
