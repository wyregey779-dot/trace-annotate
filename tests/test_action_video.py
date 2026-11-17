"""VideoMAE SSV2 remains an offline, non-authoritative window helper."""
from __future__ import annotations

from contextlib import contextmanager
import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
from fractions import Fraction

import numpy as np
import pytest

from trace_annotate.extractors import action_video as A
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


def labels():
    values = {index: f"class-{index}" for index in range(A.NUM_CLASSES)}
    values.update(A.EXPECTED_TARGET_IDS)
    values.update(A.EXPECTED_REJECTION_IDS)
    return values


def logits(index, score=10.0):
    values = np.full(A.NUM_CLASSES, -10.0, dtype=np.float64)
    values[index] = score
    return values


def samples():
    values = []
    for index in range(A.NUM_FRAMES):
        pts = Fraction(index, 10)
        values.append({"frame_index": str(index),
                       "pts": {"n": str(pts.numerator), "d": str(pts.denominator)},
                       "scene_index": 0,
                       "evidence_sha256": f"{index:064x}",
                       "image": np.zeros((8, 12, 3), dtype=np.uint8)})
    return values


def long_scene_samples(scene_index=0, *, count=96, step=Fraction(1, 4)):
    return [{"scene_index": scene_index, "frame_index": str(index),
             "pts": S.rational(Fraction(index) * step),
             "seconds": S.rational(Fraction(index) * step),
             "evidence_sha256": f"{scene_index * 1000 + index:064x}",
             "path": f"unused-{scene_index}-{index}.npy"}
            for index in range(count)]


def test_workspace_model_deployment_distinguishes_missing_from_tampered(tmp_path):
    missing_root = tmp_path / "missing-model"
    missing = V._action_video_configuration(str(missing_root))
    assert missing["status"] == "unavailable"
    assert missing["reason"] == "video_action_weights_unavailable"
    assert len(missing["model_root_identity_sha256"]) == 64
    assert V._validate_action_video_configuration(missing) == missing

    invalid_root = tmp_path / "invalid-model"
    invalid_root.mkdir()
    (invalid_root / A.WEIGHTS_FILE).write_bytes(b"wrong")
    invalid = V._action_video_configuration(str(invalid_root))
    assert invalid["status"] == "invalid"
    assert invalid["reason"] == "unverified_video_action_model_identity"
    assert V._validate_action_video_configuration(invalid) == invalid


def test_local_window_policy_is_explicit_opt_in_and_versioned(monkeypatch, tmp_path):
    monkeypatch.delenv("TRACE_ANNOTATE_ACTION_VIDEO_WINDOW_POLICY", raising=False)
    root = str(tmp_path / "missing")
    default = V._action_video_configuration(root)
    assert default["contract"] == V.ACTION_VIDEO_EVIDENCE_CONTRACT
    assert default["window_policy"] == V.ACTION_VIDEO_WINDOW_POLICY_V1
    monkeypatch.setenv("TRACE_ANNOTATE_ACTION_VIDEO_WINDOW_POLICY",
                       V.ACTION_VIDEO_WINDOW_POLICY_V2)
    v2 = V._action_video_configuration(root)
    assert v2["contract"] == V.ACTION_VIDEO_EVIDENCE_CONTRACT_V2
    assert v2["window_policy"] == V.ACTION_VIDEO_WINDOW_POLICY_V2
    assert V._validate_action_video_configuration(v2) == v2
    incompatible = dict(v2, contract=V.ACTION_VIDEO_EVIDENCE_CONTRACT)
    with pytest.raises(Exception):
        V._validate_action_video_configuration(incompatible)
    monkeypatch.setenv("TRACE_ANNOTATE_ACTION_VIDEO_WINDOW_POLICY", "unsupported")
    with pytest.raises(Exception):
        V._action_video_configuration(root)


def test_local_windows_find_sampled_short_interval_that_uniform_v1_skips():
    scene = {"start": 0.0, "end": 24.0, "duration": 24.0}
    frames = long_scene_samples()
    old_frames = {int(item["frame_index"])
                  for item in V._select_action_video_frames(frames)}
    assert {17, 18}.isdisjoint(old_frames)
    plan = V._action_video_v2_plan({0: frames}, [scene], None)
    selected_frames = {int(item["frame_index"])
                       for _, selected, _ in plan["planned"] for item in selected}
    assert {17, 18} <= selected_frames
    assert len(plan["planned"]) == 6
    windows = [{"scene_index": index, "input": descriptor,
                "status": "success", "result": {}}
               for index, _, descriptor in plan["planned"]]
    coverage = V._action_video_v2_coverage(plan, windows, {0: frames})[0]
    assert coverage["successful_windows"] == 6
    assert coverage["observation_scope"].endswith("envelopes_not_continuous")
    assert coverage["uncovered_intervals"][-1]["window_pts"]["end"] == S.rational(24)


def test_sparse_frames_are_an_explicit_unknown_not_an_action_absence():
    scene = {"start": 0.0, "end": 480.0, "duration": 480.0}
    frames = long_scene_samples(step=5)
    plan = V._action_video_v2_plan({0: frames}, [scene], None)
    assert plan["planned"] == []
    assert plan["ineligible_scenes"] == [0]
    assert {item["reason"] for item in plan["omitted"][0]} == {
        "decoded_sample_gap_exceeds_limit"}
    coverage = V._action_video_v2_coverage(plan, [], {0: frames})[0]
    assert coverage["successful_windows"] == 0
    assert coverage["uncovered_intervals"] == [{
        "window_pts": {"start": S.rational(0), "end": S.rational(480)},
        "reason": "outside_successful_window_envelopes"}]


def test_two_minute_single_shot_is_not_claimed_from_ninety_six_sparse_frames():
    scene = {"start": 0.0, "end": 120.0, "duration": 120.0}
    frames = long_scene_samples(step=Fraction(5, 4))
    plan = V._action_video_v2_plan({0: frames}, [scene], None)
    assert plan["planned"] == []
    assert plan["ineligible_scenes"] == [0]
    assert all(item["reason"] == "decoded_sample_gap_exceeds_limit"
               for item in plan["omitted"][0])


def test_global_window_budget_spreads_work_and_discloses_omitted_intervals():
    scenes = [{"start": float(index * 24), "end": float((index + 1) * 24),
               "duration": 24.0} for index in range(3)]
    inputs = {index: [{**item,
                      "pts": S.rational(Fraction(index * 24) + Fraction(int(item["frame_index"]), 4)),
                      "seconds": S.rational(Fraction(index * 24) + Fraction(int(item["frame_index"]), 4))}
                      for item in long_scene_samples(index)] for index in range(3)}
    plan = V._action_video_v2_plan(inputs, scenes, None)
    assert len(plan["planned"]) == V.ACTION_VIDEO_V2_MAX_WINDOWS
    assert {index for index, _, _ in plan["planned"]} == {0, 1, 2}
    assert sum(len(items) for items in plan["omitted"].values()) == 10
    assert {item["reason"] for items in plan["omitted"].values()
            for item in items} == {"global_window_budget"}


def test_v2_unavailable_and_v1_legacy_model_runs_are_bound_and_tamper_rejected(tmp_path):
    scenes = [{"start": 0.0, "end": 100.0, "duration": 100.0}]
    frames = long_scene_samples()
    digest = "a" * 64
    for policy, expected_contract in ((V.ACTION_VIDEO_WINDOW_POLICY_V1,
                                       V.ACTION_VIDEO_EVIDENCE_CONTRACT),
                                      (V.ACTION_VIDEO_WINDOW_POLICY_V2,
                                       V.ACTION_VIDEO_EVIDENCE_CONTRACT_V2)):
        config = V._action_video_configuration(
            str(tmp_path / "missing"), window_policy=policy)
        sampling = V._run_action_video_sampling(
            config, None, {0: frames}, tmp_path / policy,
            scenes=scenes, requested=None, cancelled=lambda: False)
        run = V._action_video_model_run("job-action-test", digest, None, sampling, config)
        assert run["parameters"]["contract"] == expected_contract
        assert sampling["contract"] == expected_contract
        V._validate_action_video_sampling(
            sampling, configuration=config, scenes=scenes, content_frames=frames,
            evidence_hashes={item["evidence_sha256"] for item in frames},
            model_runs=[run], job_id="job-action-test", digest=digest, window=None)
        if policy == V.ACTION_VIDEO_WINDOW_POLICY_V2:
            changed = json.loads(json.dumps(sampling))
            changed["coverage"][0]["uncovered_intervals"] = []
            with pytest.raises(WorkspaceError) as exc:
                V._validate_action_video_sampling(
                    changed, configuration=config, scenes=scenes, content_frames=frames,
                    evidence_hashes={item["evidence_sha256"] for item in frames},
                    model_runs=[run], job_id="job-action-test", digest=digest,
                    window=None)
            assert exc.value.code == "RECOVERY_REQUIRED"


def test_v2_export_is_only_an_unreviewed_unspecified_window_hint():
    frames = long_scene_samples(count=16)
    descriptor = V._action_video_input(frames)
    candidate = {
        "job_id": "job-action-export", "processor_version": V.VERSION,
        "action_video_sampling": {
            "contract": V.ACTION_VIDEO_EVIDENCE_CONTRACT_V2,
            "supported_hints": ["close", "open", "pick_up", "put_down_surface"],
            "unsupported_actions": list(A.UNSUPPORTED_ACTIONS),
            "windows": [{"scene_index": 0, "input": descriptor,
                         "status": "success", "result": {
                             "decision": "auxiliary_candidate",
                             "authority": "auxiliary_window_semantics_only",
                             "subject_ref": None, "object_ref": None,
                             "recipient_ref": None,
                             "semantic_hints": ["pick_up"],
                             "ranked": [{"score": 0.7, "class_id": 47,
                                         "label": "Picking something up"}],
                             "score_kind": "softmax_score_uncalibrated",
                             "review_status": "unreviewed",
                             "uncertainty": "Only a sampled window hint."}}],
        },
    }
    scene = {"start": 0.0, "end": 16.0, "duration": 16.0}
    keyframe = {"frame_index": "0", "source_pts": S.rational(0), "roles": []}
    exported = V._candidate_claims(candidate, 0, scene, S.rational(0),
                                   S.rational(16), keyframe)
    hint = next(claim for claim in exported["claims"]
                if claim["method"]["id"] == A.PROVIDER)
    params = hint["method"]["parameters"]
    assert hint["claim_type"] == "unspecified"
    assert hint["subject_ref"] is None and hint["object_ref"] is None
    assert hint["recipient_ref"] is None
    assert "action_contract" not in params
    assert params["sampling_contract"] == V.ACTION_VIDEO_EVIDENCE_CONTRACT_V2
    assert params["window_policy"] == V.ACTION_VIDEO_WINDOW_POLICY_V2
    assert hint["model_run_id"] == V._module_run_id(candidate["job_id"], "action_video")


def test_v2_cancel_after_one_window_does_not_publish_partial_result(
        monkeypatch, tmp_path):
    from trace_annotate.workspace import resource_budget
    config = V._action_video_configuration(
        str(tmp_path / "missing"), window_policy=V.ACTION_VIDEO_WINDOW_POLICY_V2)
    config["status"], config["reason"] = "available", None
    monkeypatch.setattr(V, "_validate_action_video_configuration", lambda value: value)
    monkeypatch.setattr(resource_budget, "policy", lambda: {"phase_seconds": 30.0})
    monkeypatch.setattr(V, "_load_frame", lambda path: np.zeros((8, 12, 3), dtype=np.uint8))
    calls = {"count": 0}

    def one_window(*args, **kwargs):
        calls["count"] += 1
        return {"status": "success", "reason": "top_class_outside_supported_auxiliary_mapping",
                "decision": "unknown", "semantic_hints": [], "ranked": [],
                "review_status": "unreviewed", "authority": "auxiliary_window_semantics_only"}

    monkeypatch.setattr(A, "analyze_window_isolated", one_window)
    with pytest.raises(WorkspaceError) as exc:
        V._run_action_video_sampling(
            config, str(tmp_path / "fake-model"), {0: long_scene_samples()},
            tmp_path / "cancel", scenes=[{"start": 0.0, "end": 100.0,
                                           "duration": 100.0}], requested=None,
            cancelled=lambda: calls["count"] > 0)
    assert exc.value.code == "CANCELLED"
    assert calls["count"] == 1


def test_v2_phase_budget_skips_remaining_windows_without_running_a_model(
        monkeypatch, tmp_path):
    from trace_annotate.workspace import resource_budget
    config = V._action_video_configuration(
        str(tmp_path / "missing"), window_policy=V.ACTION_VIDEO_WINDOW_POLICY_V2)
    config["status"], config["reason"] = "available", None
    monkeypatch.setattr(V, "_validate_action_video_configuration", lambda value: value)
    monkeypatch.setattr(resource_budget, "policy", lambda: {"phase_seconds": 30.0})
    clock = {"first": True}

    def now():
        if clock["first"]:
            clock["first"] = False
            return 0.0
        return 301.0

    monkeypatch.setattr(V, "time", types.SimpleNamespace(monotonic=now))
    monkeypatch.setattr(V, "_load_frame", lambda path: pytest.fail("frame loaded after deadline"))
    monkeypatch.setattr(A, "analyze_window_isolated",
                        lambda *args, **kwargs: pytest.fail("model run after deadline"))
    result = V._run_action_video_sampling(
        config, str(tmp_path / "fake-model"), {0: long_scene_samples()},
        tmp_path / "phase-budget", scenes=[{"start": 0.0, "end": 24.0,
                                             "duration": 24.0}], requested=None,
        cancelled=lambda: False)
    assert result["status"] == "not_run"
    assert result["reason"] == "video_action_phase_budget_exhausted"
    assert result["processed_windows"] == 0
    assert all(item["status"] == "not_run" and item["result"] is None
               for item in result["windows"])
    assert result["coverage"][0]["uncovered_intervals"]


def test_v2_phase_deadline_reaches_owned_worker_cancellation(monkeypatch, tmp_path):
    from trace_annotate.workspace import resource_budget
    config = V._action_video_configuration(
        str(tmp_path / "missing"), window_policy=V.ACTION_VIDEO_WINDOW_POLICY_V2)
    config["status"], config["reason"] = "available", None
    monkeypatch.setattr(V, "_validate_action_video_configuration", lambda value: value)
    monkeypatch.setattr(resource_budget, "policy", lambda: {"phase_seconds": 30.0})
    monkeypatch.setattr(V, "_load_frame", lambda path: np.zeros((8, 12, 3), dtype=np.uint8))
    clock = {"now": 0.0, "calls": 0}
    monkeypatch.setattr(V, "time", types.SimpleNamespace(monotonic=lambda: clock["now"]))

    def fake_owned(*args, **kwargs):
        clock["calls"] += 1
        clock["now"] = 301.0
        assert kwargs["cancelled"]() is True
        raise WorkspaceError("CANCELLED", "phase deadline reached")

    monkeypatch.setattr(A, "analyze_window_isolated", fake_owned)
    result = V._run_action_video_sampling(
        config, str(tmp_path / "fake-model"), {0: long_scene_samples()},
        tmp_path / "deadline-inside-worker", scenes=[{"start": 0.0, "end": 24.0,
                                                        "duration": 24.0}], requested=None,
        cancelled=lambda: False)
    assert clock["calls"] == 1
    assert result["status"] == "not_run"
    assert all(item["reason"] == "video_action_phase_budget_exhausted"
               for item in result["windows"])


def test_v2_slow_frame_load_stops_before_reading_another_frame(monkeypatch, tmp_path):
    from trace_annotate.workspace import resource_budget
    config = V._action_video_configuration(
        str(tmp_path / "missing"), window_policy=V.ACTION_VIDEO_WINDOW_POLICY_V2)
    config["status"], config["reason"] = "available", None
    monkeypatch.setattr(V, "_validate_action_video_configuration", lambda value: value)
    monkeypatch.setattr(resource_budget, "policy", lambda: {"phase_seconds": 30.0})
    clock = {"now": 0.0, "reads": 0}
    monkeypatch.setattr(V, "time", types.SimpleNamespace(monotonic=lambda: clock["now"]))

    def slow_load(path):
        clock["reads"] += 1
        clock["now"] = 301.0
        return np.zeros((8, 12, 3), dtype=np.uint8)

    monkeypatch.setattr(V, "_load_frame", slow_load)
    monkeypatch.setattr(A, "analyze_window_isolated",
                        lambda *args, **kwargs: pytest.fail("expired load reached model"))
    result = V._run_action_video_sampling(
        config, str(tmp_path / "fake-model"), {0: long_scene_samples()},
        tmp_path / "slow-frame", scenes=[{"start": 0.0, "end": 24.0,
                                           "duration": 24.0}], requested=None,
        cancelled=lambda: False)
    assert clock["reads"] == 1
    assert result["status"] == "not_run"
    assert all(item["reason"] == "video_action_phase_budget_exhausted"
               for item in result["windows"])


@pytest.mark.parametrize("limit", ["window_bytes", "disk_reserve"])
def test_v2_resource_gate_skips_without_staging_or_model_execution(
        monkeypatch, tmp_path, limit):
    from trace_annotate.workspace import resource_budget
    config = V._action_video_configuration(
        str(tmp_path / "missing"), window_policy=V.ACTION_VIDEO_WINDOW_POLICY_V2)
    config["status"], config["reason"] = "available", None
    monkeypatch.setattr(V, "_validate_action_video_configuration", lambda value: value)
    monkeypatch.setattr(resource_budget, "policy", lambda: {"phase_seconds": 30.0})
    if limit == "window_bytes":
        monkeypatch.setattr(V, "_load_frame", lambda path:
                            types.SimpleNamespace(nbytes=V.ACTION_VIDEO_V2_MAX_WINDOW_INPUT_BYTES + 1))
        reason = "video_action_window_byte_budget_exhausted"
    else:
        monkeypatch.setattr(V, "_load_frame", lambda path: np.zeros((8, 12, 3), dtype=np.uint8))
        monkeypatch.setattr(V.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=0))
        reason = "video_action_disk_reserve_unavailable"
    monkeypatch.setattr(A, "analyze_window_isolated",
                        lambda *args, **kwargs: pytest.fail("resource gate reached model"))
    result = V._run_action_video_sampling(
        config, str(tmp_path / "fake-model"), {0: long_scene_samples()},
        tmp_path / limit, scenes=[{"start": 0.0, "end": 24.0,
                                   "duration": 24.0}], requested=None,
        cancelled=lambda: False)
    assert result["status"] == "not_run" and result["reason"] == reason
    assert result["verified_success_input_bytes"] == 0
    assert all(item["reason"] == reason and item["result"] is None
               for item in result["windows"])


@pytest.mark.parametrize("mutation", ["global_bytes", "worker_bytes", "channels",
                                      "window_limit"])
def test_v2_recomputes_success_byte_receipts_before_accepting_model_run(
        monkeypatch, tmp_path, mutation):
    from trace_annotate.workspace import resource_budget
    config = V._action_video_configuration(
        str(tmp_path / "missing"), window_policy=V.ACTION_VIDEO_WINDOW_POLICY_V2)
    config["status"], config["reason"] = "available", None
    monkeypatch.setattr(V, "_validate_action_video_configuration", lambda value: value)
    monkeypatch.setattr(resource_budget, "policy", lambda: {"phase_seconds": 30.0})
    monkeypatch.setattr(V, "_load_frame", lambda path: np.zeros((8, 12, 3), dtype=np.uint8))

    def fake_owned(model_path, worker_samples, work, **kwargs):
        _, identity = A._window(worker_samples, cancelled=None)
        request = {"window": {key: identity[key] for key in (
                       "start_frame", "end_frame", "window_pts", "scene_index",
                       "total_input_bytes")}, "frames": identity["frames"]}
        result = _worker_success(request, actual_pid=222, actual_parent_pid=111)["result"]
        result["execution"] = {
            "contract": A.ISOLATED_CONTRACT, "status": "completed",
            "request_sha256": "b" * 64, "parent_pid": 111, "wrapper_pid": 222,
            "actual_pid": 222, "actual_parent_pid": 111,
            "peak_working_set_bytes": 1024, "actual_exit_code": 0,
            "elapsed_seconds": 0.1, "error_code": None,
        }
        return result

    monkeypatch.setattr(A, "analyze_window_isolated", fake_owned)
    frames = long_scene_samples()
    scenes = [{"start": 0.0, "end": 24.0, "duration": 24.0}]
    sampling = V._run_action_video_sampling(
        config, str(tmp_path / "fake-model"), {0: frames}, tmp_path / "success",
        scenes=scenes, requested=None, cancelled=lambda: False)
    assert sampling["status"] == "success"
    assert sampling["verified_success_input_bytes"] == 6 * 16 * 8 * 12 * 3
    digest = "a" * 64
    run = V._action_video_model_run("job-byte-proof", digest, None, sampling, config)

    def validate(value):
        current_run = V._action_video_model_run(
            "job-byte-proof", digest, None, value, config)
        V._validate_action_video_sampling(
            value, configuration=config, scenes=scenes, content_frames=frames,
            evidence_hashes={item["evidence_sha256"] for item in frames},
            model_runs=[current_run], job_id="job-byte-proof", digest=digest,
            window=None)

    assert run["parameters"]["contract"] == V.ACTION_VIDEO_EVIDENCE_CONTRACT_V2
    validate(sampling)
    changed = json.loads(json.dumps(sampling))
    if mutation == "global_bytes":
        changed["verified_success_input_bytes"] += 1
    elif mutation == "worker_bytes":
        changed["windows"][0]["result"]["window"]["total_input_bytes"] += 1
    elif mutation == "channels":
        changed["windows"][0]["result"]["window"]["frames"][0]["shape"][2] = 4
    else:
        for frame in changed["windows"][0]["result"]["window"]["frames"]:
            frame["shape"] = [2000, 2000, 3]
        changed["windows"][0]["result"]["window"]["total_input_bytes"] = 16 * 2000 * 2000 * 3
        changed["verified_success_input_bytes"] += (16 * 2000 * 2000 * 3) - (16 * 8 * 12 * 3)
    with pytest.raises(WorkspaceError) as exc:
        validate(changed)
    assert exc.value.code == "RECOVERY_REQUIRED"


def test_v2_real_video_job_preserves_unreviewed_contract_without_model(
        monkeypatch, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the real media boundary")
    source = tmp_path / "dense-single-shot.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "color=blue:s=64x64:r=30:d=6", "-an", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(source)],
                   check=True, capture_output=True, timeout=60)
    store = WorkspaceStore(tmp_path / "workspace")
    store.create()
    store.import_source(source)
    monkeypatch.setenv("TRACE_ANNOTATE_ACTION_VIDEO_WINDOW_POLICY",
                       V.ACTION_VIDEO_WINDOW_POLICY_V2)
    jobs = V.VisionJobs(store, person_model_path="", landmark_model_path="",
                        landmark_python="", face_model_path="missing",
                        clip_model_path="missing", vlm_model_path="missing",
                        action_video_model_path=str(tmp_path / "missing-videomae"))

    def unavailable(thresholds):
        jobs._extractors = {name: None for name in
                            ("face", "person", "ocr", "clip", "vlm")}

    monkeypatch.setattr(jobs, "_load_extractors", unavailable)
    try:
        task = store.get_task(store.list_tasks()[0]["task_id"])
        batch = jobs.submit(task["id"], command_id="v2-real-media",
                            expected_revision=task["revision"],
                            expected_sha256=task["content_sha256"])
        assert jobs.run_pending()[0]["state"] == "succeeded"
        candidate = jobs.result(batch["jobs"][0]["id"])
        sampling = candidate["action_video_sampling"]
        assert sampling["contract"] == V.ACTION_VIDEO_EVIDENCE_CONTRACT_V2
        assert sampling["status"] == "not_run"
        assert sampling["windows"]
        assert sampling["processed_windows"] == 0
        assert sampling["coverage"][0]["uncovered_intervals"]
        run = next(item for item in candidate["model_runs"]
                   if item["id"] == V._module_run_id(candidate["job_id"], "action_video"))
        assert run["parameters"]["contract"] == sampling["contract"]
        saved = jobs.apply(batch["jobs"][0]["id"], command_id="v2-apply",
                           expected_revision=task["revision"],
                           expected_sha256=task["content_sha256"], request_body=b"{}")
        assert saved["applied"] is True
        assert all(item["review_status"] == "unreviewed"
                   for item in store.get_task(task["id"])["annotations"])
    finally:
        jobs.close()
        store.close()


def test_workspace_window_input_accepts_persisted_content_pts_name():
    persisted = [{"frame_index": item["frame_index"],
                  "seconds": item["pts"],
                  "evidence_sha256": item["evidence_sha256"]}
                 for item in samples()]
    assert V._action_video_input(persisted) == V._action_video_input(samples())


def test_parent_module_import_does_not_import_heavy_model_runtimes():
    package_root = Path(A.__file__).resolve().parents[2]
    script = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "import trace_annotate.extractors.action_video;"
        "print(int('torch' in sys.modules),int('transformers' in sys.modules))"
    )
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(package_root)],
        capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "0 0"


@pytest.mark.parametrize("index,hint", [
    (5, "close"), (46, "open"), (47, "pick_up"),
    (102, "put_down_surface"), (109, "put_down_surface"),
    (112, "put_down_surface"), (119, "put_down_surface"),
])
def test_only_exact_pinned_labels_become_auxiliary_hints(index, hint):
    result = A.interpret_logits(logits(index), labels())
    assert result["decision"] == "auxiliary_candidate"
    assert result["semantic_hints"] == [hint]
    assert result["review_status"] == "unreviewed"
    assert result["authority"] == "auxiliary_window_semantics_only"
    assert result["ranked"][0]["score_kind"] == "softmax_score_uncalibrated"
    assert set(result["unsupported_actions"]) == {"walk", "run", "hand_over"}


@pytest.mark.parametrize("index", [13, 66, 67, 68, 74, 75])
def test_pretending_or_failed_top_class_is_an_explicit_rejection(index):
    result = A.interpret_logits(logits(index), labels())
    assert result["decision"] == "reject"
    assert result["reason"] == "pretending_or_failed_action_top_class"
    assert result["semantic_hints"] == []
    assert result["ranked"][0]["rejection_signal"] is True


def test_other_ssv2_classes_are_unknown_not_renamed_to_project_actions():
    result = A.interpret_logits(logits(16), labels())  # Holding is not pick-up.
    assert result["decision"] == "unknown"
    assert result["semantic_hints"] == []


@pytest.mark.parametrize("bad", [
    np.zeros(173), np.r_[np.zeros(173), np.nan], np.r_[np.zeros(173), np.inf],
])
def test_logits_shape_and_finiteness_fail_closed(bad):
    with pytest.raises(ValueError):
        A.interpret_logits(bad, labels())


def test_label_contract_mismatch_fails_closed():
    wrong = labels()
    wrong[47] = "Taking [something]"
    with pytest.raises(ValueError, match="label_contract"):
        A.interpret_logits(logits(47), wrong)


def test_window_requires_sixteen_strictly_ordered_evidence_bound_frames():
    with pytest.raises(ValueError, match="exactly_16"):
        A._window(samples()[:-1], cancelled=None)
    duplicate = samples()
    duplicate[8]["frame_index"] = duplicate[7]["frame_index"]
    with pytest.raises(ValueError, match="strictly_increasing"):
        A._window(duplicate, cancelled=None)
    bad_pts = samples()
    bad_pts[4]["pts"] = {"n": "2", "d": "4"}
    with pytest.raises(ValueError, match="reduced_rational"):
        A._window(bad_pts, cancelled=None)
    bad_hash = samples()
    bad_hash[3]["evidence_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="sha256"):
        A._window(bad_hash, cancelled=None)
    crossed = samples()
    crossed[-1]["scene_index"] = 1
    with pytest.raises(ValueError, match="scene_boundary"):
        A._window(crossed, cancelled=None)


def test_window_converts_bgr_to_rgb_and_records_exact_endpoints():
    values = samples()
    values[0]["image"][0, 0] = [1, 2, 3]
    images, identity = A._window(values, cancelled=None)
    assert list(np.asarray(images[0])[0, 0]) == [3, 2, 1]
    assert identity["start_frame"] == "0" and identity["end_frame"] == "15"
    assert identity["window_pts"] == {
        "start": {"n": "0", "d": "1"}, "end": {"n": "3", "d": "2"}}
    assert identity["scene_index"] == 0
    assert len(identity["frames"]) == A.NUM_FRAMES
    assert identity["frames"][0]["shape"] == [8, 12, 3]
    assert len(identity["frames"][0]["pixel_sha256"]) == 64


def test_window_rejects_mixed_frame_geometry():
    values = samples()
    values[-1]["image"] = np.zeros((9, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="one_frame_geometry"):
        A._window(values, cancelled=None)


def test_cancellation_is_checked_inside_frame_validation():
    calls = {"count": 0}

    def cancelled():
        calls["count"] += 1
        return calls["count"] == 5

    with pytest.raises(InterruptedError, match="cancelled"):
        A._window(samples(), cancelled=cancelled)


def test_missing_or_unverified_local_weights_never_inherit_upstream_identity(tmp_path):
    missing = A.VideoActionAuxiliary()
    assert missing.code_identity["model_name"] == "unverified_local_model"
    result = missing.analyze_window(samples())
    assert result["status"] == "not_run" and result["decision"] == "not_run"

    root = tmp_path / "model"
    root.mkdir()
    (root / A.WEIGHTS_FILE).write_bytes(b"wrong")
    unverified = A.VideoActionAuxiliary(str(root))
    assert unverified.code_identity["model_name"] == "unverified_local_model"
    assert unverified.code_identity["weights_license"] is None
    assert unverified.analyze_window(samples())["status"] == "failed"


def test_verified_files_with_missing_runtime_are_honest_not_run(monkeypatch, tmp_path):
    root = tmp_path / "exact-model"
    root.mkdir()
    files = {
        A.WEIGHTS_FILE: b"weights",
        "config.json": b"{}",
        "preprocessor_config.json": b"{}",
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    monkeypatch.setattr(A, "WEIGHTS_SIZE", len(files[A.WEIGHTS_FILE]))
    monkeypatch.setattr(A, "WEIGHTS_SHA256", hashlib.sha256(files[A.WEIGHTS_FILE]).hexdigest())
    monkeypatch.setattr(A, "CONFIG_SHA256", hashlib.sha256(files["config.json"]).hexdigest())
    monkeypatch.setattr(A, "PROCESSOR_SHA256",
                        hashlib.sha256(files["preprocessor_config.json"]).hexdigest())
    real_import = builtins.__import__

    def without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch unavailable for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    extractor = A.VideoActionAuxiliary(str(root))
    result = extractor.analyze_window(samples())
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["reason"].startswith("video_action_runtime_unavailable")


class FakeProcessor:
    def __call__(self, *, images, return_tensors):
        assert len(images) == A.NUM_FRAMES and return_tensors == "pt"
        return {"pixel_values": object()}


class FakeTensor:
    ndim = 2
    shape = (1, A.NUM_CLASSES)

    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        assert index == 0
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


class FakeModel:
    def __init__(self, index):
        self.index = index
        self.config = types.SimpleNamespace(id2label=labels())

    def __call__(self, **inputs):
        assert "pixel_values" in inputs
        return types.SimpleNamespace(logits=FakeTensor(logits(self.index)))


def test_forward_result_keeps_participant_references_null(monkeypatch):
    extractor = A.VideoActionAuxiliary()
    extractor.processor = FakeProcessor()
    extractor.model = FakeModel(46)
    extractor._verified = True
    extractor._status, extractor._reason = "success", None
    result = extractor.analyze_window(samples())
    assert result["status"] == "success"
    assert result["decision"] == "auxiliary_candidate"
    assert result["semantic_hints"] == ["open"]
    assert result["subject_ref"] is None
    assert result["object_ref"] is None
    assert result["recipient_ref"] is None
    assert "does not identify" in result["uncertainty"]
    assert "weights_path" not in result["model_identity"]


def test_forward_failure_never_publishes_third_party_exception_paths():
    class ExplodingModel(FakeModel):
        def __call__(self, **inputs):
            raise RuntimeError(r"failed to open D:\private\models\secret.bin")

    extractor = A.VideoActionAuxiliary()
    extractor.processor = FakeProcessor()
    extractor.model = ExplodingModel(46)
    extractor._verified = True
    extractor._status, extractor._reason = "success", None
    result = extractor.analyze_window(samples())
    assert result["status"] == "failed"
    assert result["reason"] == "video_action_inference_failed"
    assert "private" not in json.dumps(result).lower()


def test_extract_never_treats_a_single_staged_image_as_video_evidence():
    result = A.VideoActionAuxiliary().extract({}, "frame.png")
    assert result["status"] == "not_run"
    assert result["reason"] == "video_window_samples_required"


def _worker_success(request, *, actual_pid, actual_parent_pid):
    frames = [{key: frame[key] for key in (
        "frame_index", "pts", "scene_index", "evidence_sha256",
        "pixel_sha256", "shape")}
        for frame in request["frames"]]
    return {
        "contract": A.ISOLATED_CONTRACT,
        "provider": A.PROVIDER,
        "request_sha256": None,
        "result": {
            "status": "success", "reason": "top_class_outside_supported_auxiliary_mapping",
            "decision": "unknown", "semantic_hints": [],
            "ranked": [{"class_id": 0, "label": "class-0", "score": 0.5,
                        "score_kind": "softmax_score_uncalibrated",
                        "semantic_hint": None, "rejection_signal": False}],
            "score_kind": "softmax_score_uncalibrated",
            "review_status": "unreviewed",
            "authority": "auxiliary_window_semantics_only",
            "unsupported_actions": list(A.UNSUPPORTED_ACTIONS),
            "provider": A.PROVIDER,
            "model_identity": {
                "provider": "transformers",
                "model_name": A.PROVIDER,
                "upstream_model": A.UPSTREAM_MODEL,
                "runtime_version": "test-transformers",
                "torch_version": "test-torch",
                "weights_sha256": A.WEIGHTS_SHA256,
                "config_sha256": A.CONFIG_SHA256,
                "preprocessor_config_sha256": A.PROCESSOR_SHA256,
                "upstream_revision": A.UPSTREAM_REVISION,
                "weights_license": A.WEIGHTS_LICENSE,
                "weights_license_source": A.WEIGHTS_LICENSE_SOURCE,
                "runtime_license": "Apache-2.0 (Transformers)",
                "local_files_only": True,
                "authority": "auxiliary_window_semantics_only",
                "num_frames": A.NUM_FRAMES,
                "supported_hints": ["close", "open", "pick_up", "put_down_surface"],
                "unsupported_actions": list(A.UNSUPPORTED_ACTIONS),
            },
            "window": {**request["window"], "frames": frames},
            "subject_ref": None, "object_ref": None, "recipient_ref": None,
            "uncertainty": "Window semantics only.",
        },
        "actual_pid": actual_pid,
        "actual_parent_pid": actual_parent_pid,
        "peak_working_set_bytes": 123456,
    }


def test_isolated_parent_binds_request_result_and_owned_pid(monkeypatch, tmp_path):
    parent_pid, worker_pid = os.getpid(), 43210

    def run_owned(command, directory, **kwargs):
        directory.mkdir(parents=True)
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        raw = request_path.read_bytes()
        value = _worker_success(json.loads(raw), actual_pid=worker_pid,
                                actual_parent_pid=parent_pid)
        value["request_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return {"status": "completed", "wrapper_pid": worker_pid,
                "parent_pid": parent_pid, "worker_group": "test-owned-group"}

    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "run_owned", run_owned)
    result = A.analyze_window_isolated(
        str(tmp_path / "model"), samples(), tmp_path / "isolated", cancelled=lambda: False)
    assert result["status"] == "success" and result["decision"] == "unknown"
    assert result["execution"]["actual_pid"] == worker_pid
    request = json.loads((tmp_path / "isolated" / "request.json").read_text("utf-8"))
    assert len(request["frames"]) == A.NUM_FRAMES
    assert all((tmp_path / "isolated" / item["file"]).is_file()
               for item in request["frames"])


def test_isolated_parent_rejects_pid_substitution_and_removes_result(monkeypatch, tmp_path):
    def run_owned(command, directory, **kwargs):
        directory.mkdir(parents=True)
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        raw = request_path.read_bytes()
        value = _worker_success(json.loads(raw), actual_pid=99999,
                                actual_parent_pid=os.getpid())
        value["request_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return {"status": "completed", "wrapper_pid": 11111,
                "parent_pid": os.getpid(), "worker_group": "test-owned-group"}

    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "run_owned", run_owned)
    work = tmp_path / "pid-mismatch"
    with pytest.raises(WorkspaceError) as exc:
        A.analyze_window_isolated(str(tmp_path / "model"), samples(), work,
                                  cancelled=lambda: False)
    assert exc.value.code == "MODEL_WORKER_PROTOCOL"
    assert not (work / "result.json").exists()


@pytest.mark.parametrize("mutation", [
    "ranking", "decision", "peak_memory", "identity_path", "reason_path"])
def test_isolated_parent_rejects_tampered_semantics_and_resource_receipt(
        monkeypatch, tmp_path, mutation):
    parent_pid, worker_pid = os.getpid(), 43210

    def run_owned(command, directory, **kwargs):
        directory.mkdir(parents=True)
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        raw = request_path.read_bytes()
        value = _worker_success(json.loads(raw), actual_pid=worker_pid,
                                actual_parent_pid=parent_pid)
        value["request_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
        if mutation == "ranking":
            value["result"]["ranked"][0]["semantic_hint"] = "pick_up"
        elif mutation == "decision":
            value["result"]["decision"] = "auxiliary_candidate"
            value["result"]["semantic_hints"] = ["open"]
        elif mutation == "peak_memory":
            value["peak_working_set_bytes"] = -1
        elif mutation == "reason_path":
            value["result"] = {
                "status": "failed", "decision": "not_run", "semantic_hints": [],
                "review_status": "unreviewed",
                "authority": "auxiliary_window_semantics_only",
                "reason": str((tmp_path / "private-model.bin").resolve()),
            }
        else:
            value["result"]["model_identity"]["weights_path"] = str(
                (tmp_path / "private-model.bin").resolve())
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return {"status": "completed", "wrapper_pid": worker_pid,
                "parent_pid": parent_pid, "worker_group": "test-owned-group"}

    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "run_owned", run_owned)
    work = tmp_path / mutation
    with pytest.raises(WorkspaceError) as exc:
        A.analyze_window_isolated(str(tmp_path / "model"), samples(), work,
                                  cancelled=lambda: False)
    assert exc.value.code == "MODEL_WORKER_PROTOCOL"
    assert not (work / "result.json").exists()


def test_isolated_parent_accepts_a_worker_owned_by_the_venv_launcher(monkeypatch, tmp_path):
    parent_pid, wrapper_pid, worker_pid = os.getpid(), 11111, 22222

    def run_owned(command, directory, **kwargs):
        directory.mkdir(parents=True)
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        raw = request_path.read_bytes()
        value = _worker_success(json.loads(raw), actual_pid=worker_pid,
                                actual_parent_pid=wrapper_pid)
        value["request_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return {"status": "completed", "wrapper_pid": wrapper_pid,
                "parent_pid": parent_pid, "worker_group": "test-owned-group"}

    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "run_owned", run_owned)
    result = A.analyze_window_isolated(
        str(tmp_path / "model"), samples(), tmp_path / "launcher-child",
        cancelled=lambda: False)
    assert result["execution"]["actual_pid"] == worker_pid
    assert result["execution"]["actual_parent_pid"] == wrapper_pid


def test_isolated_worker_rejects_changed_frame_bytes_before_model_load(tmp_path):
    from trace_annotate.extractors import action_video_worker as worker
    work = tmp_path / "tamper"
    request, _ = A._write_isolated_request(
        str(tmp_path / "model"), samples(), work, top_k=5, cancelled=lambda: False)
    (work / "frames" / "00.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed|missing or exceeds"):
        worker.execute(request, work / "result.json", work / "progress.json")
    assert not (work / "result.json").exists()


@pytest.mark.parametrize("mutation", ["window", "pts", "duplicate_file", "relative_model"])
def test_isolated_worker_rejects_internally_inconsistent_request_even_without_model(
        tmp_path, mutation):
    from trace_annotate.extractors import action_video_worker as worker
    work = tmp_path / ("request-" + mutation)
    request_path, _ = A._write_isolated_request(
        str((tmp_path / "missing-model").resolve()), samples(), work,
        top_k=5, cancelled=lambda: False)
    request = json.loads(request_path.read_text("utf-8"))
    if mutation == "window":
        request["window"]["end_frame"] = "14"
    elif mutation == "pts":
        request["frames"][4]["pts"] = {"n": "2", "d": "4"}
    elif mutation == "duplicate_file":
        request["frames"][1]["file"] = request["frames"][0]["file"]
        request["frames"][1]["file_sha256"] = request["frames"][0]["file_sha256"]
        request["frames"][1]["pixel_sha256"] = request["frames"][0]["pixel_sha256"]
    else:
        request["model_path"] = "relative-model"
    request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError):
        worker.execute(request_path, work / "result.json", work / "progress.json")
    assert not (work / "result.json").exists()


def test_isolated_worker_writes_only_a_complete_atomic_result(monkeypatch, tmp_path):
    from trace_annotate.extractors import action_video_worker as worker
    work = tmp_path / "worker"
    request_path, request_sha = A._write_isolated_request(
        str(tmp_path / "model"), samples(), work, top_k=5, cancelled=lambda: False)
    request = json.loads(request_path.read_text("utf-8"))

    class FakeAuxiliary:
        def __init__(self, model_path):
            assert model_path == request["model_path"]

        def analyze_window(self, values, *, top_k, cancelled):
            assert len(values) == A.NUM_FRAMES and top_k == 5 and cancelled is None
            return _worker_success(request, actual_pid=0,
                                   actual_parent_pid=0)["result"]

    monkeypatch.setattr(worker, "VideoActionAuxiliary", FakeAuxiliary)
    output, progress = work / "result.json", work / "progress.json"
    worker.execute(request_path, output, progress)
    value = json.loads(output.read_text("utf-8"))
    assert value["request_sha256"] == request_sha
    assert value["result"]["authority"] == "auxiliary_window_semantics_only"
    assert not (work / "result.json.partial").exists()
    assert json.loads(progress.read_text("utf-8"))["phase"] == "complete"


def _small_resource_policy():
    return {"contract": "isolated-model-memory-v1", "model_bytes": 1,
            "reserve_bytes": 1, "wait_seconds": 1.0, "frame_seconds": 5.0,
            "phase_seconds": 10.0,
            "host_lock": {"contract": "traceannotate-host-model-slot-v1",
                          "kind": "file", "identity": str(Path.cwd() / "model-resource-v1.lock")}}


@contextmanager
def _admitted(settings, *, cancelled):
    yield {"available_at_admission": 10 ** 12, "required_bytes": 2,
           "wait_seconds": 0.0, "host_lock": {}}


def test_isolated_cancellation_terminates_owned_worker_and_keeps_no_result(monkeypatch, tmp_path):
    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "policy", _small_resource_policy)
    monkeypatch.setattr(resources, "admit", _admitted)
    monkeypatch.setattr(resources, "_available", lambda: 10 ** 12)
    monkeypatch.setattr(A, "_isolated_worker_command", lambda *args: [
        sys.executable, "-c", "import time; time.sleep(30)"])
    started = time.monotonic()
    work = tmp_path / "cancelled"
    with pytest.raises(WorkspaceError) as exc:
        A.analyze_window_isolated(
            str(tmp_path / "model"), samples(), work,
            cancelled=lambda: time.monotonic() - started > 0.2,
            timeout_seconds=5.0)
    assert exc.value.code == "CANCELLED"
    receipt = json.loads((work / "execution" / "process-receipt.json").read_text("utf-8"))
    assert receipt["status"] == "failed" and receipt["actual_exit_code"] is not None
    assert not (work / "result.json").exists()


def test_isolated_timeout_terminates_owned_worker_and_keeps_no_result(monkeypatch, tmp_path):
    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "policy", _small_resource_policy)
    monkeypatch.setattr(resources, "admit", _admitted)
    monkeypatch.setattr(resources, "_available", lambda: 10 ** 12)
    monkeypatch.setattr(A, "_isolated_worker_command", lambda *args: [
        sys.executable, "-c", "import time; time.sleep(30)"])
    work = tmp_path / "timed-out"
    with pytest.raises(WorkspaceError) as exc:
        A.analyze_window_isolated(
            str(tmp_path / "model"), samples(), work,
            cancelled=lambda: False, timeout_seconds=0.2)
    assert exc.value.code == "MODEL_TIMEOUT"
    receipt = json.loads((work / "execution" / "process-receipt.json").read_text("utf-8"))
    assert receipt["status"] == "failed" and receipt["actual_exit_code"] is not None
    assert not (work / "result.json").exists()


def test_real_isolated_worker_protocol_returns_honest_failure_for_missing_model(
        monkeypatch, tmp_path):
    """Exercise the real argv/bootstrap/Job-or-session path without model weights."""
    from trace_annotate.workspace import resource_budget as resources
    monkeypatch.setattr(resources, "policy", _small_resource_policy)
    monkeypatch.setattr(resources, "admit", _admitted)
    monkeypatch.setattr(resources, "_available", lambda: 10 ** 12)
    work = (tmp_path / "real-worker").resolve()
    result = A.analyze_window_isolated(
        str((tmp_path / "missing-model").resolve()), samples(), work,
        cancelled=lambda: False, timeout_seconds=5.0)
    assert result["status"] == "failed"
    assert result["decision"] == "not_run" and result["semantic_hints"] == []
    assert result["execution"]["status"] == "completed"
    receipt = json.loads((work / "execution" / "process-receipt.json").read_text("utf-8"))
    assert receipt["status"] == "completed" and receipt["actual_exit_code"] == 0
