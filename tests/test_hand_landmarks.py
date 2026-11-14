"""Exact offline Hand Landmarker identity and observation-only contract."""
from __future__ import annotations

import hashlib
import json
import ctypes
import stat
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trace_annotate.extractors import hands as H


class _ImageFormat:
    SRGB = "srgb"


class _MP:
    ImageFormat = _ImageFormat

    @staticmethod
    def Image(*, image_format, data):
        assert image_format == "srgb"
        return data


def _points(offset=0.0):
    return [SimpleNamespace(x=0.2 + offset + index * 0.01,
                            y=0.3 + index * 0.005,
                            z=-index * 0.001) for index in range(21)]


def _world():
    return [SimpleNamespace(x=index * 0.001, y=index * 0.002,
                            z=index * -0.001) for index in range(21)]


def _output():
    return SimpleNamespace(
        hand_landmarks=[_points()], hand_world_landmarks=[_world()],
        handedness=[[SimpleNamespace(category_name="Left", score=0.82, index=1)]],
    )


class _Runtime:
    def __init__(self, output=None):
        self.output = _output() if output is None else output
        self.calls = 0
        self.closed = False

    def detect(self, image):
        self.calls += 1
        assert image.shape == (100, 200, 3)
        return self.output

    def close(self):
        self.closed = True


def _extractor(monkeypatch, tmp_path, output=None):
    data = b"exact-test-task"
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(data)
    monkeypatch.setattr(H, "OFFICIAL_MODEL_SIZE", len(data))
    monkeypatch.setattr(H, "OFFICIAL_WEIGHTS_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(H, "_mediapipe_version", lambda: "test-runtime")
    runtime = _Runtime(output)
    extractor = H.HandLandmarkExtractor(str(path), max_hands=2)

    def invoke(frame_bytes, header, cancelled):
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            header["height"], header["width"], header["channels"])
        output_value = runtime.detect(frame)
        if cancelled is not None and cancelled():
            raise InterruptedError("hand_landmark_analysis_cancelled")
        if not (len(output_value.hand_landmarks)
                == len(output_value.hand_world_landmarks)
                == len(output_value.handedness)):
            raise ValueError("parallel output mismatch")
        hands_raw = []
        for image_points, world_points, categories in zip(
                output_value.hand_landmarks, output_value.hand_world_landmarks,
                output_value.handedness):
            hands_raw.append({
                "landmarks": [[point.x, point.y, point.z] for point in image_points],
                "world_landmarks": [[point.x, point.y, point.z] for point in world_points],
                "handedness": {
                    "category_name": categories[0].category_name,
                    "score": categories[0].score,
                    "index": categories[0].index,
                },
            })
        return {
            "kind": "traceannotate.hand-landmarks-worker-result", "version": 1,
            "status": "success", "reason": None, "error": None,
            "runtime_version": "test-runtime",
            "pixel_sha256": hashlib.sha256(frame_bytes).hexdigest(),
            "hands_raw": hands_raw,
        }

    monkeypatch.setattr(extractor, "_invoke_worker", invoke)
    return extractor, runtime


def _analyze(extractor, **overrides):
    values = {
        "scene_index": 3, "frame_index": "42", "pts": {"n": "21", "d": "25"},
        "evidence_sha256": "a" * 64, "model_run_id": "run-hands",
    }
    values.update(overrides)
    return extractor.analyze_frame(np.zeros((100, 200, 3), dtype=np.uint8), **values)


def test_missing_weights_are_not_run_and_do_not_download():
    extractor = H.HandLandmarkExtractor()
    result = _analyze(extractor)
    assert result["status"] == "not_run"
    assert result["reason"] == "hand_landmark_weights_missing"
    assert result["hands"] == []
    assert result["model_identity"]["downloads_allowed"] is False
    assert result["model_identity"]["license"] == "unverified"


def test_supplied_non_regular_model_path_fails_closed(tmp_path):
    result = _analyze(H.HandLandmarkExtractor(str(tmp_path)))
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_model_must_be_a_regular_file"


def test_windows_reparse_model_path_is_rejected_before_open():
    class ReparsePath:
        @staticmethod
        def lstat():
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o600,
                                   st_file_attributes=0x400)

    with pytest.raises(ValueError, match="regular_file"):
        H._read_exact_model(ReparsePath())


def test_reparse_parent_directory_is_rejected_before_model_open(monkeypatch, tmp_path):
    path = tmp_path / "hand.task"
    path.write_bytes(b"model")
    original = Path.lstat
    target_parent = path.parent.absolute()

    def lstat(candidate):
        info = original(candidate)
        if candidate.absolute() == target_parent:
            return SimpleNamespace(
                st_mode=info.st_mode, st_file_attributes=0x400,
                st_size=info.st_size, st_dev=info.st_dev, st_ino=info.st_ino,
                st_mtime_ns=info.st_mtime_ns)
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="path_contains_reparse_point"):
        H._read_exact_model(path)


def test_changed_or_wrong_size_weights_fail_closed(monkeypatch, tmp_path):
    path = tmp_path / "wrong.task"
    path.write_bytes(b"wrong")
    monkeypatch.setattr(H, "OFFICIAL_MODEL_SIZE", 6)
    result = H.HandLandmarkExtractor(str(path))
    assert _analyze(result)["reason"] == "hand_landmark_weights_size_mismatch"
    monkeypatch.setattr(H, "OFFICIAL_MODEL_SIZE", 5)
    monkeypatch.setattr(H, "OFFICIAL_WEIGHTS_SHA256", "0" * 64)
    assert _analyze(H.HandLandmarkExtractor(str(path)))["reason"] == "hand_landmark_weights_changed"


def test_missing_runtime_is_explicit_not_run(monkeypatch, tmp_path):
    data = b"model"
    path = tmp_path / "hand.task"
    path.write_bytes(data)
    monkeypatch.setattr(H, "OFFICIAL_MODEL_SIZE", len(data))
    monkeypatch.setattr(H, "OFFICIAL_WEIGHTS_SHA256", hashlib.sha256(data).hexdigest())

    monkeypatch.setattr(H, "_mediapipe_version", lambda: "unavailable")
    result = _analyze(H.HandLandmarkExtractor(str(path)))
    assert result["status"] == "not_run"
    assert result["reason"] == "hand_landmark_runtime_unavailable"


def test_model_load_failure_never_publishes_runtime_path(monkeypatch, tmp_path):
    path = tmp_path / "hand.task"
    path.write_bytes(b"model")

    def explode(_path):
        raise RuntimeError(r"cannot read D:\private\models\hand.task")

    monkeypatch.setattr(H, "_read_exact_model", explode)
    extractor = H.HandLandmarkExtractor(str(path))
    result = _analyze(extractor)
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_model_load_failed"
    assert "private" not in json.dumps(result).lower()


def test_success_binds_21_points_frame_pts_evidence_and_model_run(monkeypatch, tmp_path):
    extractor, runtime = _extractor(monkeypatch, tmp_path)
    result = _analyze(extractor)
    assert result["status"] == "success" and runtime.calls == 1
    assert result["authority"] == "hand_keypoint_observations_only"
    hand = result["hands"][0]
    assert hand["contract"] == "hand-landmark-observation-v1"
    assert hand["scene_index"] == 3 and hand["frame_index"] == "42"
    assert hand["pts"] == {"n": "21", "d": "25"}
    assert hand["evidence_sha256"] == "a" * 64
    assert hand["model_run_id"] == "run-hands"
    assert len(hand["landmarks_normalized"]) == len(hand["world_landmarks_m"]) == 21
    assert hand["landmark_names"][8] == "index_finger_tip"
    assert hand["landmarks_pixel"][0][:2] == pytest.approx([40.0, 30.0])
    assert hand["subject_ref"] is None
    assert hand["review_status"] == "unreviewed"
    assert hand["handedness_raw"] == "Left" and hand["handedness_score"] == 0.82
    assert hand["handedness_score_kind"] == "mediapipe_category_score_uncalibrated"
    assert hand["handedness_mirrored_input_assumption_verified"] is False
    assert "grasp" in hand["uncertainty"]
    assert set(result["unsupported_assertions"]) >= {"grasp", "pick_up", "hand_over"}
    assert result["model_identity"]["license"] == "Apache-2.0"
    assert result["model_identity"]["license_source"] == H.OFFICIAL_MODEL_CARD
    assert result["cancellation_boundary"] == H.CANCELLATION_BOUNDARY
    assert result["frame_identity"]["pixel_sha256"] == hashlib.sha256(
        np.zeros((100, 200, 3), dtype=np.uint8).tobytes()).hexdigest()


@pytest.mark.parametrize("changes", [
    {"scene_index": -1}, {"frame_index": "01"},
    {"pts": {"n": "2", "d": "4"}}, {"evidence_sha256": "A" * 64},
    {"model_run_id": "bad run"},
])
def test_invalid_frame_identity_fails_without_inference(monkeypatch, tmp_path, changes):
    extractor, runtime = _extractor(monkeypatch, tmp_path)
    result = _analyze(extractor, **changes)
    assert result["status"] == "failed" and runtime.calls == 0


def test_invalid_image_and_callback_are_rejected(monkeypatch, tmp_path):
    extractor, runtime = _extractor(monkeypatch, tmp_path)
    result = extractor.analyze_frame(
        np.zeros((10, 10), dtype=np.uint8), scene_index=0, frame_index="0",
        pts={"n": "0", "d": "1"}, evidence_sha256="a" * 64,
        model_run_id="run-hands")
    assert result["status"] == "failed" and result["reason"] == "invalid_image"
    assert runtime.calls == 0
    with pytest.raises(ValueError, match="cancelled must be callable"):
        _analyze(extractor, cancelled=True)


def test_cancellation_before_or_after_forward_never_returns_partial_output(monkeypatch, tmp_path):
    extractor, runtime = _extractor(monkeypatch, tmp_path)
    with pytest.raises(InterruptedError, match="cancelled"):
        _analyze(extractor, cancelled=lambda: True)
    assert runtime.calls == 0
    states = iter([False, True])
    with pytest.raises(InterruptedError, match="cancelled"):
        _analyze(extractor, cancelled=lambda: next(states))
    assert runtime.calls == 1


@pytest.mark.parametrize("output", [
    SimpleNamespace(hand_landmarks=[_points()], hand_world_landmarks=[],
                    handedness=[[SimpleNamespace(category_name="Left", score=0.9, index=1)]]),
    SimpleNamespace(hand_landmarks=[_points()[:20]], hand_world_landmarks=[_world()],
                    handedness=[[SimpleNamespace(category_name="Left", score=0.9, index=1)]]),
    SimpleNamespace(hand_landmarks=[_points()], hand_world_landmarks=[_world()],
                    handedness=[[SimpleNamespace(category_name="Unknown", score=0.9, index=1)]]),
    SimpleNamespace(hand_landmarks=[_points(), _points(.1), _points(.2)],
                    hand_world_landmarks=[_world(), _world(), _world()],
                    handedness=[[SimpleNamespace(category_name="Left", score=0.9, index=1)]] * 3),
    SimpleNamespace(hand_landmarks=[_points(H.MAX_ABS_LANDMARK_COORDINATE + 1)],
                    hand_world_landmarks=[_world()],
                    handedness=[[SimpleNamespace(category_name="Left", score=0.9, index=1)]]),
])
def test_malformed_runtime_outputs_fail_closed(monkeypatch, tmp_path, output):
    extractor, _runtime = _extractor(monkeypatch, tmp_path, output)
    result = _analyze(extractor)
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_failed"
    assert result["hands"] == []


def test_close_releases_runtime_and_prevents_reuse(monkeypatch, tmp_path):
    extractor, runtime = _extractor(monkeypatch, tmp_path)
    extractor.close()
    assert runtime.calls == 0
    result = _analyze(extractor)
    assert result["status"] == "not_run"
    assert result["reason"] == "hand_landmark_runtime_closed"


def test_extract_does_not_invent_missing_frame_identity(monkeypatch, tmp_path):
    extractor, _runtime = _extractor(monkeypatch, tmp_path)
    result = extractor.extract({}, "ignored")
    assert result["status"] == "not_run"
    assert result["reason"] == "decoded_frame_identity_required"


class _FakeJobApi:
    def __init__(self, *, create=101, set_result=True, assign_result=True,
                 terminate_result=True, close_result=True, error=5):
        self.handle = create
        self.set_result = set_result
        self.assign_result = assign_result
        self.terminate_result = terminate_result
        self.close_result = close_result
        self.error = error
        self.calls = []

    def create(self):
        self.calls.append(("create",))
        return self.handle

    def set_kill_on_close(self, handle):
        self.calls.append(("set_kill_on_close", handle))
        return self.set_result

    def assign(self, handle, process_handle):
        self.calls.append(("assign", handle, process_handle))
        return self.assign_result

    def terminate(self, handle, exit_code):
        self.calls.append(("terminate", handle, exit_code))
        return self.terminate_result

    def close(self, handle):
        self.calls.append(("close", handle))
        return self.close_result

    def last_error(self):
        return self.error


def test_windows_job_object_is_configured_assigned_and_closed(monkeypatch):
    api = _FakeJobApi()
    monkeypatch.setattr(H, "_WindowsJobApi", lambda: api)
    job = H._WindowsKillJob.attach(SimpleNamespace(_handle=202))
    assert api.calls[:3] == [
        ("create",), ("set_kill_on_close", 101), ("assign", 101, 202)]
    assert job.terminate_and_close(exit_code=73) is True
    assert api.calls[-2:] == [("terminate", 101, 73), ("close", 101)]
    assert job.close() is True
    assert api.calls.count(("close", 101)) == 1


@pytest.mark.parametrize(("failure", "message"), [
    ("create", "job_create"), ("set", "job_limit"), ("assign", "job_assign")])
def test_windows_job_object_setup_failure_is_explicit_and_closes_handle(
        monkeypatch, failure, message):
    api = _FakeJobApi(
        create=0 if failure == "create" else 101,
        set_result=failure != "set", assign_result=failure != "assign")
    monkeypatch.setattr(H, "_WindowsJobApi", lambda: api)
    with pytest.raises(H._WorkerIsolationUnavailable, match=message):
        H._WindowsKillJob.attach(SimpleNamespace(_handle=202))
    if failure == "create":
        assert not [call for call in api.calls if call[0] == "close"]
    else:
        assert api.calls[-1] == ("close", 101)


def test_job_setup_failure_is_reported_not_silently_downgraded(monkeypatch, tmp_path):
    extractor, _runtime = _extractor(monkeypatch, tmp_path)

    def unavailable(*_args, **_kwargs):
        raise H._WorkerIsolationUnavailable("nested_job_denied")

    monkeypatch.setattr(extractor, "_invoke_worker", unavailable)
    result = _analyze(extractor)
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_isolation_unavailable"
    assert result["hands"] == []


_HELPER = r'''
import hashlib,json,os,subprocess,sys,time
mode=sys.argv[1]
line=sys.stdin.buffer.readline()
header=json.loads(line)
raw=sys.stdin.buffer.read(header['frame_bytes'])
if mode=='sleep':time.sleep(30)
if mode=='tree_sleep':
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    with open(sys.argv[2],'w',encoding='ascii') as stream:
        stream.write(str(child.pid));stream.flush();os.fsync(stream.fileno())
    time.sleep(30)
if mode=='crash':os._exit(7)
if mode=='flood':
    sys.stdout.buffer.write(b'x'*4096);sys.stdout.buffer.flush();time.sleep(30)
if mode=='path_leak':
    value={'kind':'traceannotate.hand-landmarks-worker-result','version':1,
    'status':'failed','reason':'hand_landmark_worker_failed','error':'C:/private/model.task',
    'runtime_version':'fake-subprocess','pixel_sha256':None,'hands_raw':[]}
    data=json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
    sys.stdout.buffer.write(data);sys.stdout.buffer.flush();raise SystemExit(0)
points=[[0.20+i*0.01,0.30+i*0.005,-i*0.001] for i in range(21)]
world=[[i*0.001,i*0.002,-i*0.001] for i in range(21)]
value={'kind':'traceannotate.hand-landmarks-worker-result','version':1,
'status':'success','reason':None,'error':None,'runtime_version':'fake-subprocess',
'pixel_sha256':hashlib.sha256(raw).hexdigest(),'hands_raw':[{
'landmarks':points,'world_landmarks':world,
'handedness':{'category_name':'Left','score':0.82,'index':1}}]}
data=json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''


def _subprocess_extractor(monkeypatch, tmp_path, mode, *, timeout=2.0,
                          extra_args=()):
    data = b"exact-worker-task"
    model = tmp_path / "worker-hand.task"
    model.write_bytes(data)
    script = tmp_path / "fake_hand_worker.py"
    script.write_text(textwrap.dedent(_HELPER), encoding="utf-8")
    monkeypatch.setattr(H, "OFFICIAL_MODEL_SIZE", len(data))
    monkeypatch.setattr(H, "OFFICIAL_WEIGHTS_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(H, "_mediapipe_version", lambda: "fake-subprocess")
    return H.HandLandmarkExtractor(
        str(model), max_hands=2,
        worker_command=[sys.executable, str(script), mode, *extra_args],
        worker_timeout=timeout)


def test_real_owned_subprocess_protocol_success_and_no_path_leak(monkeypatch, tmp_path):
    extractor = _subprocess_extractor(monkeypatch, tmp_path, "success")
    result = _analyze(extractor)
    assert result["status"] == "success"
    assert len(result["hands"]) == 1
    assert result["frame_identity"]["pixel_sha256"] == hashlib.sha256(
        np.zeros((100, 200, 3), dtype=np.uint8).tobytes()).hexdigest()
    encoded = json.dumps(result, ensure_ascii=False)
    assert str(tmp_path) not in encoded
    assert "weights_path" not in result["model_identity"]


def test_owned_subprocess_timeout_is_bounded(monkeypatch, tmp_path):
    extractor = _subprocess_extractor(monkeypatch, tmp_path, "sleep", timeout=.15)
    started = time.monotonic()
    result = _analyze(extractor)
    assert time.monotonic() - started < 5
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_timeout"
    assert not extractor._active_processes


def _windows_process_alive(pid):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object integration")
def test_windows_job_timeout_kills_worker_grandchild(monkeypatch, tmp_path):
    child_pid_file = tmp_path / "child.pid"
    extractor = _subprocess_extractor(
        monkeypatch, tmp_path, "tree_sleep", timeout=.75,
        extra_args=(str(child_pid_file),))
    result = _analyze(extractor)
    if result["reason"] == "hand_landmark_worker_isolation_unavailable":
        pytest.skip("host Job Object policy forbids a nested owned Job")
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_timeout"
    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text(encoding="ascii"))
    deadline = time.monotonic() + 5
    while _windows_process_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not _windows_process_alive(child_pid)


def test_owned_subprocess_cancellation_terminates_worker(monkeypatch, tmp_path):
    extractor = _subprocess_extractor(monkeypatch, tmp_path, "sleep", timeout=10)
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancelled"):
        _analyze(extractor, cancelled=lambda: time.monotonic() - started > .15)
    assert time.monotonic() - started < 5
    assert not extractor._active_processes


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_cancel_callback_exception_is_always_a_failed_result(
        monkeypatch, tmp_path, fail_on_call):
    extractor = _subprocess_extractor(monkeypatch, tmp_path, "sleep", timeout=10)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise RuntimeError("callback failure must not escape")
        return False

    result = _analyze(extractor, cancelled=cancelled)
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_failed"
    assert result["error"] == "RuntimeError"
    assert calls == fail_on_call
    assert not extractor._active_processes


@pytest.mark.parametrize("fail_on_start", [1, 2, 3])
def test_thread_start_failure_terminates_and_unregisters_worker(
        monkeypatch, tmp_path, fail_on_start):
    extractor = _subprocess_extractor(monkeypatch, tmp_path, "sleep", timeout=10)
    original_start = threading.Thread.start
    original_terminate = H._terminate_process_tree
    starts = 0
    terminated = []

    def start(thread):
        nonlocal starts
        starts += 1
        if starts == fail_on_start:
            raise RuntimeError("injected thread start failure")
        return original_start(thread)

    def terminate(process, windows_job=None):
        terminated.append(process)
        return original_terminate(process, windows_job)

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(H, "_terminate_process_tree", terminate)
    result = _analyze(extractor)
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_failed"
    assert result["error"] == "RuntimeError"
    assert not extractor._active_processes
    assert terminated and all(process.poll() is not None for process in terminated)


@pytest.mark.parametrize("mode", ["crash", "flood", "path_leak"])
def test_owned_subprocess_crash_or_oversized_output_fails_closed(
        monkeypatch, tmp_path, mode):
    if mode == "flood":
        monkeypatch.setattr(H, "MAX_WORKER_RESPONSE_BYTES", 512)
    extractor = _subprocess_extractor(monkeypatch, tmp_path, mode, timeout=2)
    result = _analyze(extractor)
    assert result["status"] == "failed"
    assert result["reason"] == "hand_landmark_worker_failed"
    assert result["hands"] == []
    assert not extractor._active_processes
    assert str(tmp_path) not in json.dumps(result)


def test_close_terminates_all_concurrent_owned_workers(monkeypatch, tmp_path):
    extractor = _subprocess_extractor(monkeypatch, tmp_path, "sleep", timeout=10)
    outcomes = []

    def run():
        try:
            outcomes.append(_analyze(extractor))
        except BaseException as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with extractor._process_lock:
            if len(extractor._active_processes) == 2:
                break
        time.sleep(.01)
    with extractor._process_lock:
        assert len(extractor._active_processes) == 2
    extractor.close()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert not extractor._active_processes
    assert all(isinstance(value, InterruptedError) for value in outcomes)
