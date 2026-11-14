"""VideoMAE SSV2 remains an offline, non-authoritative window helper."""
from __future__ import annotations

from contextlib import contextmanager
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from fractions import Fraction

import numpy as np
import pytest

from trace_annotate.extractors import action_video as A
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.errors import WorkspaceError


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
