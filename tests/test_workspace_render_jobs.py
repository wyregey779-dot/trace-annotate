"""Queue/CAS/pin boundary tests; controlled renderer cases are not media QA."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.temporal.media import stream_sha256
from trace_annotate.workspace import backup as B, render_jobs as J, rendering as R, schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.slots import get_prepare_slot
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Real admission and one real rendering require FFmpeg/ffprobe")
    path = tmp_path_factory.mktemp("render-jobs-input") / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=160x120:r=10:d=0.6",
        "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=0.6",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True, timeout=30)
    return path


@pytest.fixture
def store(tmp_path, clip):
    value = WorkspaceStore(tmp_path / "workspace"); value.create()
    try:
        value.import_source(clip)
        yield value
    finally:
        value.close()


def task(store):
    return store.get_task(store.list_tasks()[0]["task_id"])


def manager(store, **kwargs):
    return J.RenderJobs(store, max_output_bytes=8 * 1024 * 1024, **kwargs)


def submit(jobs, command="render-one", recipe=None):
    current = task(jobs.store)
    return jobs.submit(current["id"], command_id=command, expected_revision=current["revision"],
        expected_sha256=current["content_sha256"], recipe=recipe or jobs.options()["default_recipe"])


def error(code, call):
    with pytest.raises(WorkspaceError) as exc:
        call()
    assert exc.value.code == code


def edit(store, command="edit-label"):
    current = task(store)
    return store.apply_task_command(command_id=command, task_id=current["id"], expected_revision=current["revision"],
        expected_sha256=current["content_sha256"], operation="add_annotation", payload={"annotation": {
            "id": "ann-after-render", "type": "whole_clip", "label": "Later label -42", "text": "1,234",
            "track": "main", "tags": [], "attributes": {}, "evidence_ids": []}}, request_body=command.encode())


def pins(store, job_id):
    return store.conn.execute("SELECT artifact_id,owner_revision,role FROM artifact_references "
                              "WHERE owner_kind='render_manifest' AND owner_id=? ORDER BY role", (job_id,)).fetchall()


def fake_renderer(source, plan, output, **kwargs):
    """Synthetic bytes only: verifies atomic publication plumbing, not encoding."""
    output.mkdir(parents=True)
    validated = output / "validated"; validated.mkdir()
    def write(name, value):
        (validated / name).write_bytes(S.canonical_json(value) + b"\n")
    (validated / "annotation.mp4").write_bytes(b"Controlled renderer output, not a real MP4")
    write("render-plan.json", plan)
    write("verification.json", {"kind": "traceannotate.render-verification", "version": 1,
        "frame_count": plan["frame_count"], "full_decode_exit": 0, "all_frame_pts_exact": True, "final_hold_exact": True})
    write("audit.json", {"kind": "controlled-test", "version": 1})
    files = [{"file": name, "sha256": stream_sha256(validated / name), "size_bytes": str((validated / name).stat().st_size)}
             for name in sorted(set(J.FILES) - {"project.json"})]
    manifest = {"kind": "traceannotate.rendered-video", "version": 1, "status": "verified_not_workspace_published",
        "renderer_version": R.VERSION,
        "binding": plan["binding"], "layout_sha256": plan["content_sha256"], "encoding": plan["recipe"]["encoding"],
        "files": files, "validation": {"full_decode_exit": 0, "all_frame_pts_exact": True},
        "scope": "Controlled test only; no media validation claim"}
    manifest["content_sha256"] = S.sha256_text(manifest)
    write("project.json", manifest)
    return manifest


def test_actual_renderer_publishes_five_pinned_files_and_paired_json(store):
    jobs = manager(store)
    edit(store)
    initial = task(store)
    queued = submit(jobs)["jobs"][0]
    error("RENDER_NOT_READY", lambda: jobs.result(queued["id"]))
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded", done
    result = jobs.result(queued["id"])
    assert result["kind"] == "traceannotate.render-result" and result["task_id"] == initial["id"]
    assert result["input_binding"] == result["current_binding"]
    assert len(pins(store, queued["id"])) == 5 and set(result["manifest"]["files"]) == set(J.FILES)
    path, digest, size = jobs.video(queued["id"])
    assert path.is_file() and stream_sha256(path) == digest and path.stat().st_size == size
    exported = jobs.frozen_document(queued["id"])
    document = json.loads(exported["data"])
    assert document["task"] == initial and document["history"]["scope"] == "not_included"
    assert task(store) == initial and jobs.close()["stopped"]


def test_missing_trusted_font_only_disables_render(store, tmp_path):
    jobs = J.RenderJobs(store, font_path=tmp_path / "missing.otf", font_license="Test")
    assert jobs.options()["available"] is False and jobs.options()["reason"]
    assert jobs.options()["layout_version"] == J.L.VERSION
    assert jobs.options()["renderer_version"] == R.VERSION
    assert store.list_tasks() and jobs.queue.list_batches() == []
    jobs.close()


@pytest.mark.parametrize("change", ["path", "hash", "license", "position"])
def test_client_recipe_cannot_select_font_paths_or_untrusted_font(store, change):
    jobs = manager(store)
    recipe = jobs.options()["default_recipe"]
    if change == "path": recipe["font_path"] = "C:/outside.otf"
    elif change == "hash": recipe["font_sha256"] = "0" * 64
    elif change == "license": recipe["font_license"] = "Unverified client declaration"
    else: recipe["position"] = "overlay"
    with pytest.raises(WorkspaceError):
        submit(jobs, recipe=recipe)
    assert jobs.queue.list_batches() == []


def test_submit_receipt_replay_after_edit_and_close_and_other_operation_exclusion(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store)
    current = task(store); recipe = jobs.options()["default_recipe"]
    args = dict(command_id="one", expected_revision=current["revision"], expected_sha256=current["content_sha256"], recipe=recipe)
    first = jobs.submit(current["id"], **args)
    other = jobs.queue.enqueue(current["id"], operation="analyze-media", processor_version="1")
    assert jobs.run_pending()[0]["state"] == "succeeded"
    assert jobs.queue.get_job(other["id"])["state"] == "queued"
    edit(store)
    jobs.close()
    assert jobs.submit(current["id"], **args) == first
    error("WORKSPACE_CLOSING", lambda: submit(jobs, command="new-after-close"))


def test_snapshot_and_paired_json_never_mix_later_task_revision(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store)
    before = task(store)
    own = submit(jobs)["jobs"][0]
    edit(store)
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded" and done["commit"]["input_still_current"] is False
    result = jobs.result(own["id"])
    assert result["input_binding"]["revision"] == before["revision"]
    assert result["current_binding"]["revision"] == before["revision"] + 1
    document = json.loads(jobs.frozen_document(own["id"])["data"])
    assert document["task"] == before and document["binding"]["revision"] == before["revision"]


def test_gc_backup_restore_preserve_render_pin_closure(store, monkeypatch, tmp_path):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store)
    own = submit(jobs)["jobs"][0]
    jobs.run_pending(); initial = jobs.result(own["id"])
    jobs.close()
    store.sweep_unbound_artifacts(apply=True)
    backup_path = tmp_path / "backup"; restored_path = tmp_path / "restored"
    B.backup_workspace(store, backup_path); B.restore_workspace(backup_path, restored_path)
    restored = WorkspaceStore(restored_path); restored.open()
    try:
        current = manager(restored)
        assert current.result(own["id"]) == initial and len(pins(restored, own["id"])) == 5
        current.close()
    finally:
        restored.close()


@pytest.mark.parametrize("fault", ["pin", "video", "project", "candidate", "path", "oversize"])
def test_result_and_video_refuse_missing_pin_corrupt_bytes_or_registration(store, monkeypatch, fault):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store); own = submit(jobs)["jobs"][0]
    done = jobs.run_pending()[0]; candidate = jobs.result(own["id"])["manifest"]
    descriptor = candidate["files"]["annotation.mp4" if fault != "project" else "project.json"]
    if fault == "pin":
        store.conn.execute("DELETE FROM artifact_references WHERE owner_id=? AND role='video'", (own["id"],))
    elif fault in ("video", "project"):
        (store.root / descriptor["relative_path"]).write_bytes(b"tampered")
    elif fault == "candidate":
        (store.root / done["commit"]["artifact"]["relative_path"]).write_bytes(b"{}")
    elif fault == "path":
        store.conn.execute("UPDATE artifacts SET rel_path='../outside.mp4' WHERE id=?", (descriptor["id"],))
    else:
        artifact = done["commit"]["artifact"]
        store.conn.execute("UPDATE artifacts SET size_bytes=? WHERE id=?", (J.MAX_RESULT_BYTES + 1, artifact["id"]))
    error("RECOVERY_REQUIRED", lambda: jobs.result(own["id"]))
    error("RECOVERY_REQUIRED", lambda: jobs.video(own["id"]))


def test_cancel_during_cas_stream_keeps_gc_out_and_has_no_pins_or_download(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store); own = submit(jobs)["jobs"][0]
    entered, release = threading.Event(), threading.Event()
    publish = store._publish_stream
    calls = []
    def blocked(stream):
        value = publish(stream)
        calls.append(value)
        if len(calls) == 1:
            entered.set(); assert release.wait(timeout=5)
        return value
    monkeypatch.setattr(store, "_publish_stream", blocked)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(jobs.run_pending)
        assert entered.wait(timeout=5)
        try:
            error("GC_ACTIVE_WORK", lambda: store.sweep_unbound_artifacts(apply=True))
            jobs.queue.cancel(own["id"])
        finally:
            release.set()
        done = future.result(timeout=5)[0]
    assert done["state"] == "cancelled" and done["commit"] is None and pins(store, own["id"]) == []
    error("RENDER_NOT_READY", lambda: jobs.video(own["id"]))
    store.sweep_unbound_artifacts(apply=True)
    assert len(store.list_tasks()) == 1
    jobs.queue.control_job(own["id"], "retry_cancelled", command_id="restart-render")
    resumed = jobs.run_pending()[0]
    assert resumed["state"] == "succeeded" and resumed["attempt"] == 2
    assert jobs.video(own["id"])[0].is_file() and len(pins(store, own["id"])) == 5


def test_finish_failure_rolls_back_all_pins_then_retry_succeeds(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store); batch = submit(jobs); own = batch["jobs"][0]
    finish = jobs.queue.finish
    def fail(*a, **kwargs): raise WorkspaceError("CONTROLLED_FINISH_FAILURE", "commit rollback")
    monkeypatch.setattr(jobs.queue, "finish", fail)
    failed = jobs.run_pending()[0]
    assert failed["state"] == "failed" and failed["commit"] is None and pins(store, own["id"]) == []
    monkeypatch.setattr(jobs.queue, "finish", finish)
    jobs.queue.retry_failed(own["id"])
    done = jobs.run_pending()[0]
    assert done["state"] == "succeeded" and done["attempt"] == 2 and len(pins(store, own["id"])) == 5


def controlled_encoder(source, plan, output, **kwargs):
    output.mkdir(parents=True)
    P._run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-re", "-f", "lavfi", "-i",
            "anullsrc=r=48000:cl=stereo", "-t", "30", "-f", "null", "-"], output,
           "controlled-render-job-encoder", kwargs["cancelled"])
    pytest.fail("Controlled child should be interrupted")


@pytest.mark.parametrize("action", ["cancel", "close", "deadline"])
def test_actual_owned_child_stops_for_job_controls_and_deadline(store, monkeypatch, action):
    monkeypatch.setattr(R, "render_video", controlled_encoder)
    jobs = manager(store, deadline_seconds=.8 if action == "deadline" else 10)
    own = submit(jobs)["jobs"][0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(jobs.run_pending)
        until = time.monotonic() + 5; owned = []
        while time.monotonic() < until:
            owned = [p for p in P.live_media_processes() if p["name"] == "controlled-render-job-encoder"]
            if owned: break
            time.sleep(.01)
        assert owned
        if action == "cancel": jobs.queue.cancel(own["id"])
        elif action == "close": assert jobs.close(timeout=5)["stopped"]
        done = future.result(timeout=6)[0]
    assert done["state"] == ("failed" if action == "deadline" else "cancelled")
    assert done["error_code"] == ("RENDER_TIMEOUT" if action == "deadline" else "CANCELLED")
    assert done["commit"] is None and pins(store, own["id"]) == []
    assert not ({p["pid"] for p in owned} & {p["pid"] for p in P.live_media_processes()})
    jobs.close()


def test_one_second_lease_renews_during_render(store, monkeypatch):
    def slow(*args, **kwargs):
        until = time.monotonic() + 1.5
        while time.monotonic() < until:
            assert not kwargs["cancelled"]()
            time.sleep(.01)
        return fake_renderer(*args, **kwargs)
    monkeypatch.setattr(R, "render_video", slow)
    queue = DurableQueue(store, lease_seconds=1)
    jobs = manager(store, queue=queue)
    own = submit(jobs)["jobs"][0]
    assert jobs.run_pending()[0]["state"] == "succeeded" and jobs.result(own["id"])


def test_shared_prepare_slot_does_not_claim_queued_render(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store); own = submit(jobs)["jobs"][0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        with get_prepare_slot(store.root).hold():
            future = executor.submit(jobs.run_pending)
            time.sleep(.1)
            assert jobs.queue.get_job(own["id"])["state"] == "queued"
            assert jobs.close(timeout=2)["stopped"]
            assert future.result(timeout=2) == []


@pytest.mark.parametrize("window", ["before_finish", "after_commit"])
def test_real_process_crash_never_separates_pins_from_success_commit(store, tmp_path, window):
    jobs = manager(store); own = submit(jobs)["jobs"][0]
    root = store.root; jobs.close(); store.close()
    script = r'''
from pathlib import Path
from contextlib import contextmanager
import importlib.util, os, sys
from trace_annotate.workspace import render_jobs as J, rendering as R
from trace_annotate.workspace.store import WorkspaceStore
spec=importlib.util.spec_from_file_location("controlled_render_fixture",sys.argv[3])
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
R.render_video=fixture.fake_renderer
store=WorkspaceStore(Path(sys.argv[1]));store.open()
jobs=J.RenderJobs(store,max_output_bytes=8*1024*1024)
if sys.argv[2]=="before_finish":
    jobs.queue.finish=lambda *a,**kw:os._exit(74)
else:
    original=store.transaction
    @contextmanager
    def barrier():
        with original() as value:
            yield value
        if store._depth==0 and store.conn.execute("SELECT 1 FROM job_commits").fetchone():
            os._exit(75)
    store.transaction=barrier
jobs.run_pending()
os._exit(76)
'''
    child = subprocess.run([sys.executable, "-B", "-c", script, str(root), window, str(Path(__file__).absolute())],
                           cwd=str(Path(__file__).parents[1]), capture_output=True, text=True, timeout=30)
    assert child.returncode == (74 if window == "before_finish" else 75), child.stderr
    reopened = WorkspaceStore(root); reopened.open()
    try:
        recovered = manager(reopened)
        job = recovered.queue.get_job(own["id"])
        if window == "before_finish":
            assert job["state"] == "queued" and job["commit"] is None and pins(reopened, own["id"]) == []
            reopened.sweep_unbound_artifacts(apply=True)
        else:
            assert job["state"] == "succeeded" and len(pins(reopened, own["id"])) == 5
            assert recovered.result(own["id"])
        recovered.close()
    finally:
        reopened.close()


def test_close_keeps_store_open_until_post_dispatch_pin_cleanup_finishes(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store); submit(jobs)
    entered, release = threading.Event(), threading.Event()
    original = jobs._recover_pins
    def blocked_cleanup():
        entered.set()
        assert release.wait(timeout=5)
        return original()
    monkeypatch.setattr(jobs, "_recover_pins", blocked_cleanup)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(jobs.run_pending)
        assert entered.wait(timeout=5)
        try:
            assert jobs.close(timeout=.02)["stopped"] is False
        finally:
            release.set()
        assert future.result(timeout=5)[0]["state"] == "succeeded"
    assert jobs.close(timeout=2)["stopped"]



def test_preview_has_exact_plan_and_store_stays_open_until_owned_planner_stops(store, monkeypatch):
    jobs = manager(store)
    initial = task(store); options = jobs.options()
    plan = jobs.plan(initial["id"], expected_revision=initial["revision"], expected_sha256=initial["content_sha256"],
                     recipe=options["default_recipe"])
    assert plan["binding"]["task_content_sha256"] == initial["content_sha256"]
    assert plan["geometry"]["source_crop"] is False and plan["geometry"]["source_scale"] is False
    entered, release = threading.Event(), threading.Event()
    prepare = jobs._prepare
    def blocked(*args, **kwargs):
        entered.set(); assert release.wait(timeout=5)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(jobs, "_prepare", blocked)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(jobs.plan, initial["id"], expected_revision=initial["revision"],
                                 expected_sha256=initial["content_sha256"], recipe=options["default_recipe"])
        assert entered.wait(timeout=3)
        try:
            assert jobs.close(timeout=.02)["stopped"] is False
        finally:
            release.set()
        with pytest.raises(WorkspaceError): future.result(timeout=5)
    assert jobs.close(timeout=2)["stopped"]


def test_enqueue_receipt_survives_professional_import_changing_source_archive(store):
    from trace_annotate.workspace.professional_store import export_professional
    jobs = manager(store)
    initial = task(store); recipe = jobs.options()["default_recipe"]
    arguments = dict(command_id="archive-stable-enqueue", expected_revision=initial["revision"],
                     expected_sha256=initial["content_sha256"], recipe=recipe)
    accepted = jobs.submit(initial["id"], **arguments)
    document = json.loads(export_professional(store, initial["id"])["data"])
    store.apply_task_command(command_id="new-professional-archive", task_id=initial["id"],
        expected_revision=initial["revision"], expected_sha256=initial["content_sha256"],
        operation="import_professional", payload={"document": document}, request_body=b"real-professional-import")
    assert task(store)["extensions"]["professional"]["import_source"]["document_artifact"]
    assert jobs.submit(initial["id"], **arguments) == accepted
    changed = json.loads(S.canonical_json(recipe)); changed["body_font_px"] += 1
    error("IDEMPOTENCY_CONFLICT", lambda: jobs.submit(initial["id"], **dict(arguments, recipe=changed)))
    error("REVISION_CONFLICT", lambda: jobs.submit(initial["id"], **dict(arguments, command_id="new-stale-base")))


def test_restart_releases_only_obsolete_uncommitted_render_pins(store, monkeypatch):
    monkeypatch.setattr(R, "render_video", fake_renderer)
    jobs = manager(store); own = submit(jobs)["jobs"][0]
    claimed = jobs.queue.claim("lost-render-worker")
    orphan = store._put_artifact(b"Controlled stale uncommitted render artifact")
    store.bind_artifact_reference(orphan, owner_kind="render_manifest", owner_id=own["id"],
                                  owner_revision=str(claimed["lease_generation"]), role="video")
    path = store.root / orphan["relative_path"]
    root = store.root; store.close()
    reopened = WorkspaceStore(root); reopened.open()
    try:
        resumed = manager(reopened)
        assert resumed.queue.get_job(own["id"])["state"] == "queued"
        assert pins(reopened, own["id"]) == [] and path.is_file()
        assert orphan["id"] in {a["id"] for a in reopened.unbound_artifacts()["artifacts"]}
        done = resumed.run_pending()[0]
        assert done["state"] == "succeeded" and len(pins(reopened, own["id"])) == 5
        resumed.close()
    finally:
        reopened.close()






@pytest.mark.parametrize('action',['close','deadline','job_cancel'])
def test_layout_phase_uses_preview_or_worker_control_before_painting_text(store,monkeypatch,action):
    jobs=manager(store)
    edit(store)
    initial=task(store)
    recipe=jobs.options()['default_recipe']
    own=submit(jobs)['jobs'][0] if action=='job_cancel' else None
    original=J.L._panel_entries
    reached=[]
    def interrupt(*args,**kwargs):
        assert callable(kwargs.get('cancelled'))
        reached.append(True)
        if action=='close':jobs._closing.set()
        elif action=='deadline':jobs.deadline_seconds=1e-12
        else:jobs.queue.cancel(own['id'])
        return original(*args,**kwargs)
    def unwanted(*args):
        pytest.fail('Cancelled layout continued to measure text')
    monkeypatch.setattr(J.L,'_panel_entries',interrupt)
    monkeypatch.setattr(J.L,'_width',unwanted)
    if action=='job_cancel':
        done=jobs.run_pending()[0]
        assert done['state']=='cancelled' and done['commit'] is None
        assert pins(store,own['id'])==[]
    else:
        error('CANCELLED' if action=='close' else 'RENDER_TIMEOUT',lambda:jobs.plan(initial['id'],
            expected_revision=initial['revision'],expected_sha256=initial['content_sha256'],recipe=recipe))
    assert reached
    assert jobs.close()['stopped']


def test_layout_upgrade_reads_real_committed_v1_bytes_and_replays_original_receipt(store, monkeypatch):
    """Real short MP4, not synthetic renderer bytes; no old-plan reflow."""
    current_layout = J.L.VERSION
    assert current_layout != "annotation-panel-v1"
    jobs = manager(store)
    edit(store)
    initial = task(store)
    recipe = jobs.options()["default_recipe"]
    assert jobs.options()["layout_version"] == current_layout
    assert jobs.options()["renderer_version"] == R.VERSION
    arguments = dict(command_id="before-layout-upgrade", expected_revision=initial["revision"],
                     expected_sha256=initial["content_sha256"], recipe=recipe)
    with monkeypatch.context() as old:
        old.setattr(J.L, "VERSION", "annotation-panel-v1")
        accepted = jobs.submit(initial["id"], **arguments)
        legacy_id = accepted["jobs"][0]["id"]
        assert jobs.run_pending(batch_id=accepted["id"])[0]["state"] == "succeeded"
        before = jobs.result(legacy_id)
        old_video, old_hash, old_size = jobs.video(legacy_id)
        old_bytes = old_video.read_bytes()
        old_document = jobs.frozen_document(legacy_id)
        old_plan = json.loads((store.root / before["manifest"]["files"]["render-plan.json"]["relative_path"]).read_bytes())
        assert old_plan["renderer_version"] == "annotation-panel-v1"
    assert J.L.VERSION == current_layout
    # A reader may validate stored layout/hash, but must never recompute it.
    def no_reflow(*args, **kwargs):
        pytest.fail("Reading an immutable legacy candidate attempted a new layout")
    monkeypatch.setattr(J.L, "prepare_plan", no_reflow)
    assert jobs.result(legacy_id) == before
    assert jobs.video(legacy_id) == (old_video, old_hash, old_size)
    assert old_video.read_bytes() == old_bytes and stream_sha256(old_video) == old_hash
    assert jobs.frozen_document(legacy_id) == old_document
    assert jobs.submit(initial["id"], **arguments) == accepted
    current = jobs.submit(initial["id"], **dict(arguments, command_id="after-layout-upgrade"))
    assert current["jobs"][0]["id"] != legacy_id
    assert current["jobs"][0]["parameters"]["layout_version"] == current_layout
    assert current["jobs"][0]["state"] == "queued"
    changed = copy.deepcopy(recipe)
    changed["body_font_px"] += 1
    error("IDEMPOTENCY_CONFLICT", lambda: jobs.submit(initial["id"], **dict(arguments, recipe=changed)))
    # The historical receipt must not excuse different base revisions either.
    store.apply_task_command(command_id="after-upgrade-edit", task_id=initial["id"],
        expected_revision=initial["revision"], expected_sha256=initial["content_sha256"], operation="update_annotation",
        payload={"annotation_id": "ann-after-render", "changes": {"label": "Changed after upgrade"}}, request_body=b"after-upgrade-edit")
    changed_base = task(store)
    error("IDEMPOTENCY_CONFLICT", lambda: jobs.submit(initial["id"], **dict(arguments,
        expected_revision=changed_base["revision"], expected_sha256=changed_base["content_sha256"])))
    assert jobs.submit(initial["id"], **arguments) == accepted
    assert len(pins(store, legacy_id)) == 5
    assert jobs.close()["stopped"]


@pytest.mark.parametrize("layout_version", ["annotation-panel-v1", "annotation-panel-future-999"])
def test_unavailable_queued_layout_fails_before_probe_layout_or_encoder(store, monkeypatch, layout_version):
    jobs = manager(store)
    current = task(store)
    recipe = jobs.options()["default_recipe"]
    parameters = jobs._request_parameters(current, recipe)
    parameters["layout_version"] = layout_version
    accepted = jobs.queue.create_batch([{"task_id": current["id"], "expected_revision": current["revision"],
        "expected_sha256": current["content_sha256"]}], operation=J.OPERATION, processor_version=J.VERSION,
        parameters=parameters, command_id="unsupported-layout-command")
    own = accepted["jobs"][0]
    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported layout reached media or planning work")
    monkeypatch.setattr(jobs, "_prepare", unexpected)
    monkeypatch.setattr(R, "render_video", unexpected)
    done = jobs.run_pending(batch_id=accepted["id"])[0]
    assert done["state"] == "failed" and done["error_code"] == "UNSUPPORTED_RENDER_LAYOUT"
    assert done["commit"] is None and pins(store, own["id"]) == []
    assert not (store.tmp_dir / "render/jobs" / own["id"]).exists()
    # Read compatibility is never authorization to execute a queued old plan.
    error("UNSUPPORTED_RENDER_LAYOUT", lambda: jobs._job(jobs.queue.get_job(own["id"]), allow_legacy_read=True))
    assert jobs.close()["stopped"]


def test_committed_unknown_future_layout_is_not_accepted_as_legacy(store, monkeypatch):
    jobs = manager(store)
    with monkeypatch.context() as future:
        future.setattr(J.L, "VERSION", "annotation-panel-future-999")
        batch = submit(jobs)
        own = batch["jobs"][0]
        assert jobs.run_pending(batch_id=batch["id"])[0]["state"] == "succeeded"
        assert jobs.result(own["id"])
    error("UNSUPPORTED_RENDER_LAYOUT", lambda: jobs.result(own["id"]))
    error("UNSUPPORTED_RENDER_LAYOUT", lambda: jobs.video(own["id"]))
    error("UNSUPPORTED_RENDER_LAYOUT", lambda: jobs.frozen_document(own["id"]))
    assert jobs.close()["stopped"]


@pytest.mark.parametrize("relabel", ["layout", "renderer"])
def test_candidate_rejects_internally_rehashed_layout_or_renderer_version_relabelling(store, monkeypatch, relabel):
    def altered(source, plan, output, **kwargs):
        changed_plan = copy.deepcopy(plan)
        if relabel == "layout":
            changed_plan["renderer_version"] = "annotation-panel-v1"
            changed_plan["content_sha256"] = S.sha256_text({k: v for k, v in changed_plan.items() if k != "content_sha256"})
        manifest = fake_renderer(source, changed_plan, output, **kwargs)
        if relabel == "renderer":
            manifest["renderer_version"] = "a-different-renderer"
            manifest["content_sha256"] = S.sha256_text({k: v for k, v in manifest.items() if k != "content_sha256"})
            (output / "validated/project.json").write_bytes(S.canonical_json(manifest) + b"\n")
        return manifest
    monkeypatch.setattr(R, "render_video", altered)
    jobs = manager(store)
    own = submit(jobs)["jobs"][0]
    done = jobs.run_pending()[0]
    assert done["state"] == "failed" and done["error_code"] == "RECOVERY_REQUIRED"
    assert done["commit"] is None and pins(store, own["id"]) == []
    error("RENDER_NOT_READY", lambda: jobs.video(own["id"]))
    assert jobs.close()["stopped"]
