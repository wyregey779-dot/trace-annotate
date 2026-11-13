"""Full-range proposals use real frame identity, finite budgets and cancellation."""
import copy
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


def discovery():
    from trace_annotate.workspace import transition_discovery
    return transition_discovery


def samples(values, offset=0):
    for position, value in enumerate(values):
        index = offset + position
        yield {"frame_index": str(index), "seconds": index / 17,
               "source_pts": str(index * 1001), "proxy_pts": str(index * 17),
               "frame": np.full((16, 24), value, dtype=np.uint8)}


def test_global_proposal_finds_flash_without_a_scene_boundary_and_keeps_identity():
    result = discovery().discover_events(samples([120] * 15 + [255] + [120] * 20, 300))
    assert [(event["kind"], event["anchor_frame"]) for event in result["events"]] == [("flash", "315")]
    assert result["events"][0]["split_point"] is None
    assert result["events"][0]["needs_review"] is True


def test_overlap_produces_one_flash_and_reports_bounded_event_output():
    values = [120] * 330
    values[193] = values[257] = 255
    result = discovery().discover_events(samples(values), max_events=1)
    assert len(result["events"]) == 1
    assert result["events"][0]["anchor_frame"] == "193"
    assert result["events_truncated"] is True
    assert result["dropped_events"] >= 1


def test_global_scan_cancellation_is_observed_in_feature_windows():
    calls = []
    def cancelled():
        calls.append(1)
        return len(calls) > 5
    with pytest.raises(WorkspaceError) as error:
        discovery().discover_events(samples([120] * 500), cancelled=cancelled)
    assert error.value.code == "CANCELLED"


def test_global_scan_discovers_slow_crossfade_below_local_step_threshold():
    y, x = np.indices((48, 64))
    first = (((x // 4 + y // 6) % 2) * 180 + 35).astype(float)
    last = (((x // 9 + y // 3) % 2) * 150 + 70).astype(float)
    values = [0.0] * 8 + list(np.linspace(0.0, 1.0, 100)) + [1.0] * 8
    frames = [{"frame_index": str(i), "seconds": i / 30,
               "frame": np.round((1 - a) * first + a * last).astype(np.uint8)}
              for i, a in enumerate(values)]
    result = discovery().discover_events(frames)
    assert any(event["kind"] == "dissolve" and int(event["interval"]["end_frame"]) - int(event["interval"]["start_frame"]) >= 50
               for event in result["events"])


def test_cross_window_merge_identifies_the_interval_behind_fit_metrics():
    y, x = np.indices((32, 48))
    first = (((x // 4 + y // 4) % 2) * 255).astype(float)
    last = 255 - first
    weights = [0.0] * 100 + list(np.linspace(0, 1, 200)) + [1.0] * 100
    frames = [{"frame_index": str(i), "seconds": i / 30,
               "frame": np.round((1 - a) * first + a * last).astype(np.uint8)}
              for i, a in enumerate(weights)]
    event = next(event for event in discovery().discover_events(frames)["events"] if event["kind"] == "dissolve")
    assert event["metrics"]["merged_interval"] is True
    assert int(event["interval"]["end_frame"]) > int(event["metrics"]["fit_interval"]["end_frame"])
    assert event["metrics"]["endpoint_distance"] < 1.0


@pytest.fixture
def admitted(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for decoded transition discovery")
    source = tmp_path / "gradual.mp4"
    values = [0] * 4 + list(range(0, 181, 9)) + [180] * 20
    data = b"".join(np.full((64, 64, 3), value, dtype=np.uint8).tobytes() for value in values)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "64x64", "-r", "10", "-i", "pipe:0", "-an",
                    "-vf", "setsar=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                   input=data, check=True, capture_output=True, timeout=60)
    store = WorkspaceStore(tmp_path / "project")
    store.create()
    item = store.import_source(source)
    try:
        yield store, item
    finally:
        store.close()


def test_decoded_scan_finds_fade_and_binds_every_frame_to_admitted_map(admitted, tmp_path):
    store, item = admitted
    result = discovery().scan_transitions(store, item["media_id"], tmp_path / "scan")
    assert result["status"] == "success"
    assert result["coverage"]["complete"] is True
    assert result["decoded_frames"] == int(store.get_media(item["media_id"])["frame_count"])
    assert result["frame_map_sha256"]
    event = next(e for e in result["events"] if e["kind"] == "fade_in")
    assert event["interval"]["end_frame"] != event["interval"]["start_frame"]
    assert Path(result["measurement_path"]).is_file()


def test_byte_budget_explicitly_limits_coverage_and_honours_half_open_window(admitted, tmp_path):
    store, item = admitted
    result = discovery().scan_transitions(store, item["media_id"], tmp_path / "scan-budget",
        window={"start": S.rational(1), "end": S.rational(3)},
        max_bytes=discovery().FRAME_BYTES * 5)
    assert result["decoded_frames"] == 5
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["first_frame"] == "10"
    assert result["coverage"]["last_frame"] == "14"
    assert result["coverage"]["requested_frames"] == 20
    assert result["coverage"]["reason"] == "raw_frame_byte_budget"


def test_byte_cap_marks_an_active_fade_interval_as_clipped(admitted, tmp_path):
    store, item = admitted
    result = discovery().scan_transitions(
        store, item["media_id"], tmp_path / "active-fade-cap",
        max_bytes=discovery().FRAME_BYTES * 15)
    assert result["coverage"]["complete"] is False
    event = next(event for event in result["events"] if event["kind"] == "fade_in")
    assert event["interval"]["end_frame"] == "14"
    assert event["metrics"]["window_clipped"] is True
    assert "字节预算" in event["uncertainty"]


def test_full_global_interval_is_retained_beside_a_short_local_candidate(admitted, tmp_path):
    from trace_annotate.workspace import vision_jobs as V
    store, item = admitted
    partial = discovery().scan_transitions(
        store, item["media_id"], tmp_path / "partial-local-measurement",
        window={"start": S.rational(0), "end": S.rational("3/2")})
    local = next(event for event in partial["events"] if event["kind"] == "fade_in")
    local = copy.deepcopy(local)
    local["metrics"]["analysis_scope"] = "scene_start_neighbourhood"
    original = copy.deepcopy(local)
    first = {"scene_index": 0, "frame_index": "0", "boundary_events": [local]}
    frames = []
    result = V._attach_global_transitions(store, item["media_id"], store.tmp_dir,
        [first], [{"start": 0.0, "end": 4.5}], frames, None, shutil.which("ffmpeg"), None)
    assert result["projected_events"] == 1
    assert first["boundary_events"][0] == original
    full = first["boundary_events"][1]
    assert int(full["interval"]["end_frame"]) > int(local["interval"]["end_frame"])
    assert full["needs_review"] is True
    assert full["metrics"]["overlapping_local_anchors"] == [local["anchor_frame"]]
    assert any(frame["frame_index"] == full["interval"]["end_frame"] for frame in frames)


def test_normal_vision_entry_finds_fade_in_single_scene_and_exports_unreviewed(admitted, monkeypatch):
    from trace_annotate.workspace import vision_jobs as V
    store, item = admitted
    current = store.get_task(item["task_id"])
    jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="", landmark_python="",
                        face_model_path="missing", clip_model_path="missing", vlm_model_path="missing")
    def unavailable(thresholds):
        jobs._extractors = {name: None for name in ("face", "person", "ocr", "clip", "vlm")}
    monkeypatch.setattr(jobs, "_load_extractors", unavailable)
    try:
        batch = jobs.submit(current["id"], command_id="global-gradual", expected_revision=0,
                            expected_sha256=current["content_sha256"])
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(batch["jobs"][0]["id"])
        assert len(candidate["scenes"]) == 1, "fixture must bypass the previous scene-start neighbourhood path"
        assert candidate["boundary_sampling"]["global_discovery"]["coverage"]["complete"] is True
        events = [e for f in candidate["keyframes"] for e in f.get("boundary_events", [])]
        assert any(e["kind"] == "fade_in" and e["metrics"]["analysis_scope"] == "full_range" for e in events)
        assert store.get_task(current["id"]) == current
        receipt = jobs.apply(batch["jobs"][0]["id"], command_id="global-gradual-save",
                             expected_revision=0, expected_sha256=current["content_sha256"], request_body=b"{}")
        assert receipt["applied"] is True
        saved = store.get_task(current["id"])
        assert all(a["review_status"] == "unreviewed" for a in saved["annotations"])
        claims = [c for a in saved["annotations"] for c in a["attributes"]["semantic"]["claims"]]
        transition = next(c for c in claims if c["method"]["parameters"].get("kind") == "fade_in")
        assert transition["evidence_ids"]
        assert len(candidate["model_runs"]) == 13
    finally:
        jobs.close()


def test_completed_old_transition_jobs_remain_readable_while_queued_jobs_require_resubmit(admitted, monkeypatch):
    from trace_annotate.workspace import vision_jobs as V
    store, item = admitted
    current = store.get_task(item["task_id"])
    configuration = discovery().configuration
    legacy = configuration()
    legacy.pop("long_dissolve", None)
    assert legacy != configuration()
    jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="", landmark_python="",
                        face_model_path="missing", clip_model_path="missing", vlm_model_path="missing")
    def unavailable(thresholds):
        jobs._extractors = {name: None for name in ("face", "person", "ocr", "clip", "vlm")}
    monkeypatch.setattr(jobs, "_load_extractors", unavailable)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(discovery(), "configuration", lambda: copy.deepcopy(legacy))
            old = jobs.submit(current["id"], command_id="old-transition-completed", expected_revision=0,
                              expected_sha256=current["content_sha256"])
            assert jobs.run_pending()[0]["state"] == "succeeded"
        old_job = old["jobs"][0]
        saved_result = jobs.result(old_job["id"])
        old_parameters = V._parameters(window={"start": S.rational(0), "end": S.rational(1)},
            optional_providers=old_job["parameters"]["job_identity"]["config"]["optional_providers"],
            model_resources=old_job["parameters"]["job_identity"]["config"]["model_resources"],
            transition_configuration=legacy)
        pending = jobs.queue.create_batch([{"task_id": current["id"], "expected_revision": 0,
            "expected_sha256": current["content_sha256"]}], operation=V.OPERATION,
            processor_version=V.VERSION, parameters=old_parameters, command_id="old-transition-pending")
        rejected = jobs.run_pending(batch_id=pending["id"])[0]
        assert rejected["state"] == "failed"
        assert rejected["error_code"] == "VISION_RESUBMIT_REQUIRED"
        fresh = jobs.submit(current["id"], command_id="new-transition-configuration", expected_revision=0,
                            expected_sha256=current["content_sha256"])
        assert fresh["jobs"][0]["id"] != old_job["id"]
        assert jobs.result(old_job["id"]) == saved_result
        jobs.apply(old_job["id"], command_id="old-transition-apply", expected_revision=0,
                   expected_sha256=current["content_sha256"], request_body=b"{}")
        assert all(a["review_status"] == "unreviewed" for a in store.get_task(current["id"])["annotations"])
    finally:
        jobs.close()


@pytest.mark.parametrize("transition_frames,selected_window", [
    (100, None), (400, None), (1000, None),
    (1000, {"start": S.rational(5), "end": S.rational(30)})])
def test_normal_entry_decodes_slow_textured_crossfade_and_saves_bound_evidence(
        tmp_path, monkeypatch, transition_frames, selected_window):
    from trace_annotate.workspace import vision_jobs as V
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for an encoded crossfade fixture")
    y, x = np.indices((48, 64))
    first = (((x // 4 + y // 6) % 2) * 180 + 35).astype(float)
    last = (((x // 9 + y // 3) % 2) * 150 + 70).astype(float)
    weights = [0.0] * 20 + list(np.linspace(0, 1, transition_frames)) + [1.0] * 20
    data = b"".join(np.repeat(np.round((1 - a) * first + a * last).astype(np.uint8)[..., None],
                               3, axis=2).tobytes() for a in weights)
    source = tmp_path / "slow-textured-crossfade.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "64x48", "-r", "30", "-i", "pipe:0", "-an", "-vf", "setsar=1",
                    "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(source)],
                   input=data, check=True, capture_output=True, timeout=60)
    store = WorkspaceStore(tmp_path / "crossfade-project")
    store.create()
    jobs = None
    try:
        item = store.import_source(source)
        current = store.get_task(item["task_id"])
        jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="", landmark_python="",
                            face_model_path="missing", clip_model_path="missing", vlm_model_path="missing")
        def unavailable(thresholds):
            jobs._extractors = {name: None for name in ("face", "person", "ocr", "clip", "vlm")}
        monkeypatch.setattr(jobs, "_load_extractors", unavailable)
        batch = jobs.submit(current["id"], command_id="encoded-slow-crossfade", expected_revision=0,
                            expected_sha256=current["content_sha256"], window=selected_window)
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(batch["jobs"][0]["id"])
        assert len(candidate["scenes"]) == 1
        global_result = candidate["boundary_sampling"]["global_discovery"]
        assert global_result["coverage"]["complete"] is True
        event = next(e for e in global_result["events"] if e["kind"] == "dissolve")
        expected_frames = (transition_frames if selected_window is None else
            int(30 * (S.rational_value(selected_window["end"]) - S.rational_value(selected_window["start"]))))
        assert int(event["interval"]["end_frame"]) - int(event["interval"]["start_frame"]) >= expected_frames * .75
        if selected_window is not None:
            assert event["previous_frame"] is None
            assert event["metrics"]["window_clipped"] is True
            start_row = store._frame_service.row(item["media_id"], int(event["interval"]["start_frame"]))
            assert event["time_window"] == {"start": {"seconds": start_row["seconds"]},
                                            "end": {"seconds": selected_window["end"]}}
            assert S.rational_value(selected_window["start"]) <= S.rational_value(start_row["seconds"])
            assert S.rational_value(start_row["seconds"]) <= S.rational_value(selected_window["start"]) + S.rational_value(S.rational("16/30"))
        assert event["time_window"]["start"]["seconds"]["d"]
        assert store.get_task(current["id"]) == current
        jobs.apply(batch["jobs"][0]["id"], command_id="encoded-slow-crossfade-save",
                   expected_revision=0, expected_sha256=current["content_sha256"], request_body=b"{}")
        saved = store.get_task(current["id"])
        claim = next(c for a in saved["annotations"] for c in a["attributes"]["semantic"]["claims"]
                     if c["method"]["parameters"].get("kind") == "dissolve")
        assert claim["time_window"] == event["time_window"]
        assert claim["evidence_ids"]
        assert all(a["review_status"] == "unreviewed" for a in saved["annotations"])
        for evidence_id in claim["evidence_ids"]:
            artifact = store.conn.execute("SELECT sha256, rel_path FROM artifacts WHERE id=?",
                                          (evidence_id,)).fetchone()
            assert artifact is not None
            assert V.stream_sha256(store.root / artifact[1]) == artifact[0]
    finally:
        if jobs is not None:
            jobs.close()
        store.close()


def test_full_existing_semantic_claim_budget_omits_new_global_projection_and_still_applies(
        admitted, monkeypatch):
    """A controlled full semantic record retains its claims and raw discovery.

    The video scan and job/apply path run for real. The mapper stub supplies an
    existing valid 32-claim load without invoking unrelated semantic models.
    """
    from trace_annotate.workspace import semantic_schema as SS
    from trace_annotate.workspace import vision_jobs as V
    store, item = admitted
    current = store.get_task(item["task_id"])
    jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="", landmark_python="",
                        face_model_path="missing", clip_model_path="missing", vlm_model_path="missing")
    original_claims = V._candidate_claims

    def unavailable(thresholds):
        jobs._extractors = {name: None for name in ("face", "person", "ocr", "clip", "vlm")}

    def full_existing_claims(*args, **kwargs):
        semantic = original_claims(*args, **kwargs)
        global_claims = [claim for claim in semantic["claims"]
                         if claim["method"]["id"] == "shot-boundaries-v1"
                         and claim["method"]["parameters"].get("metrics", {}).get("analysis_scope") == "full_range"]
        existing = [claim for claim in semantic["claims"] if claim not in global_claims]
        while len(existing) < SS.MAX_CLAIMS:
            claim = copy.deepcopy(existing[0])
            claim["id"] = S.make_id("clm", "controlled-existing-claim-" + str(len(existing)))
            claim["statement"] = "受控回归测试既有语义项 " + str(len(existing))
            claim["method"] = {"id": "controlled-existing-claim", "version": "1", "parameters": {}}
            claim["model_run_id"] = None
            existing.append(claim)
        semantic["claims"] = existing + global_claims
        return semantic

    monkeypatch.setattr(jobs, "_load_extractors", unavailable)
    monkeypatch.setattr(V, "_candidate_claims", full_existing_claims)
    try:
        batch = jobs.submit(current["id"], command_id="full-semantic-global", expected_revision=0,
                            expected_sha256=current["content_sha256"])
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(batch["jobs"][0]["id"])
        global_result = candidate["boundary_sampling"]["global_discovery"]
        assert any(event["kind"] == "fade_in" for event in global_result["events"])
        assert global_result["projected_events"] == 0
        assert any(item["reason"] == "semantic_claim_budget"
                   for item in global_result["projection_omitted"])
        projected = V._candidate_annotations(candidate)
        assert len(projected[0]["attributes"]["semantic"]["claims"]) == SS.MAX_CLAIMS
        receipt = jobs.apply(batch["jobs"][0]["id"], command_id="full-semantic-global-save",
                             expected_revision=0, expected_sha256=current["content_sha256"], request_body=b"{}")
        assert receipt["applied"] is True
        saved = store.get_task(current["id"])
        assert len(saved["annotations"][0]["attributes"]["semantic"]["claims"]) == SS.MAX_CLAIMS
        assert all(annotation["review_status"] == "unreviewed" for annotation in saved["annotations"])
    finally:
        jobs.close()


def test_event_window_preserves_high_denominator_frame_map_rationals():
    from trace_annotate.workspace import vision_jobs as V
    exact = {"start": {"seconds": {"n": "3000000022", "d": "1000000007"}},
             "end": {"seconds": {"n": "3000000023", "d": "1000000007"}}}
    event = {"kind": "fade_in", "time_window": exact,
             "interval": {"start_seconds": 3.0, "end_seconds": 3.000001}}
    fallback = {"start": {"seconds": S.rational(0)}, "end": {"seconds": S.rational(9)}}
    actual, precise = V._event_time_window({}, event, fallback)
    assert precise is True
    assert actual == exact


def test_global_flash_claim_owns_flash_frame_and_attaches_its_recovery_frame(tmp_path, monkeypatch):
    from trace_annotate.workspace import vision_jobs as V
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for a decoded flash fixture")
    source = tmp_path / "single-flash.mp4"
    values = [120] * 20 + [255] + [124] * 24
    data = b"".join(np.full((64, 64, 3), value, dtype=np.uint8).tobytes() for value in values)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "64x64", "-r", "10", "-i", "pipe:0", "-an",
                    "-vf", "setsar=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                   input=data, check=True, capture_output=True, timeout=60)
    store = WorkspaceStore(tmp_path / "flash-project")
    store.create()
    jobs = None
    preview_sources = []
    preview_hashes_by_sample = {}
    original_preview = V._evidence_preview

    def unavailable(thresholds):
        jobs._extractors = {name: None for name in ("face", "person", "ocr", "clip", "vlm")}

    def capture_preview(path):
        preview_sources.append(path)
        preview = original_preview(path)
        preview_hashes_by_sample.setdefault(V.stream_sha256(path), set()).add(V.stream_sha256(preview))
        return preview

    try:
        admitted = store.import_source(source)
        current = store.get_task(admitted["task_id"])
        jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="", landmark_python="",
                            face_model_path="missing", clip_model_path="missing", vlm_model_path="missing")
        monkeypatch.setattr(jobs, "_load_extractors", unavailable)
        monkeypatch.setattr(V, "_evidence_preview", capture_preview)
        # An empty hard-cut proposal set isolates the real global scan from the
        # prior local scene-start detector. The encoded flash remains real.
        monkeypatch.setattr(V, "_run_scenes", lambda *args: {
            "status": "success", "error": None,
            "scenes": [{"start": 0.0, "end": 4.5, "duration": 4.5}]})
        batch = jobs.submit(current["id"], command_id="global-flash-recovery", expected_revision=0,
                            expected_sha256=current["content_sha256"])
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(batch["jobs"][0]["id"])
        flash = next(event for frame in candidate["keyframes"] for event in frame.get("boundary_events", [])
                     if event["kind"] == "flash" and event["metrics"]["analysis_scope"] == "full_range")
        assert flash["anchor_frame"] == "20"
        assert flash["recovery_frame"] == "21"
        frame_map = store._frame_service
        expected_window = {"start": {"seconds": frame_map.row(admitted["media_id"], 20)["seconds"]},
                           "end": {"seconds": frame_map.row(admitted["media_id"], 21)["seconds"]}}
        assert flash["time_window"] == expected_window
        recovery = next(frame for frame in candidate["boundary_frames"] if frame["frame_index"] == "21")
        descriptor = next(item for item in candidate["evidence"] if item["sha256"] == recovery["evidence_sha256"])
        recovery_path = store.root / descriptor["relative_path"]
        assert V.stream_sha256(recovery_path) == recovery["evidence_sha256"]
        jobs.apply(batch["jobs"][0]["id"], command_id="global-flash-recovery-save", expected_revision=0,
                   expected_sha256=current["content_sha256"], request_body=b"{}")
        saved = store.get_task(current["id"])
        claim = next(claim for annotation in saved["annotations"]
                     for claim in annotation["attributes"]["semantic"]["claims"]
                     if claim["method"]["parameters"].get("kind") == "flash")
        assert claim["time_window"] == expected_window
        assert claim["method"]["parameters"]["recovery_frame"] == "21"
        previous = next(frame for frame in candidate["boundary_frames"] if frame["frame_index"] == "19")
        anchor = next(frame for frame in candidate["boundary_frames"] if frame["frame_index"] == "20")
        assert recovery["evidence_sha256"] not in {previous["evidence_sha256"], anchor["evidence_sha256"]}
        claim_hashes = {store.conn.execute("SELECT sha256 FROM artifacts WHERE id=?", (evidence_id,)).fetchone()[0]
                        for evidence_id in claim["evidence_ids"]}
        assert preview_hashes_by_sample[recovery["evidence_sha256"]] <= claim_hashes
        # Identical decoded images can share a content-addressed evidence file.
        assert any(V.stream_sha256(path) == recovery["evidence_sha256"] for path in preview_sources)
        assert claim["evidence_ids"]
        assert all(annotation["review_status"] == "unreviewed" for annotation in saved["annotations"])
    finally:
        if jobs is not None:
            jobs.close()
        store.close()


def test_scene_failure_retains_its_status_and_explicitly_skips_global_scan(admitted, monkeypatch):
    from trace_annotate.workspace import vision_jobs as V
    store, item = admitted
    current = store.get_task(item["task_id"])
    jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="", landmark_python="",
                        face_model_path="missing", clip_model_path="missing", vlm_model_path="missing")
    def unavailable(thresholds):
        jobs._extractors = {name: None for name in ("face", "person", "ocr", "clip", "vlm")}
    monkeypatch.setattr(jobs, "_load_extractors", unavailable)
    monkeypatch.setattr(V, "_run_scenes", lambda *args: {
        "status": "failed", "error": "controlled_scene_decoder_failure", "scenes": []})
    try:
        batch = jobs.submit(current["id"], command_id="global-scene-failure", expected_revision=0,
                            expected_sha256=current["content_sha256"])
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(batch["jobs"][0]["id"])
        assert candidate["extractor_status"]["scene"] == "failed"
        assert candidate["boundary_sampling"]["global_discovery"]["status"] == "not_run"
        scene_run = next(run for run in candidate["model_runs"] if run["model_name"] == "scene-cut-ffmpeg")
        assert scene_run["failure_reason"] == "controlled_scene_decoder_failure"
        assert store.get_task(current["id"]) == current
    finally:
        jobs.close()
