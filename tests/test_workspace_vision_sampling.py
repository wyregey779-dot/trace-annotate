"""PTS-aware sampling: half-open windows, edge/boundary roles, budgets, real decode."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time

import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.workspace import schema as S
from trace_annotate.workspace import vision_sampling as VSmp
from trace_annotate.workspace.errors import DependencyUnavailable, SchemaInvalid, WorkspaceError
from trace_annotate.workspace.frames import FrameService
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for sampling fixtures")
    source = tmp_path_factory.mktemp("sampling-source") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
        "-i", "testsrc2=s=64x64:r=10:d=1.0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-g", "10", str(source),
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


def _media(store):
    return store.get_task(store.list_tasks()[0]["task_id"])["media_id"]


def _rows(store, media_id):
    return FrameService(store).page(media_id, 0, 1000)["frames"]


def test_window_is_left_closed_right_open(store):
    media_id = _media(store)
    rows = _rows(store, media_id)
    plan = VSmp.plan_samples(store, media_id,
                             window={"start": {"n": "0", "d": "1"},
                                     "end": {"n": "1", "d": "2"}},
                             uniform=2)
    assert plan["samples"]
    for sample in plan["samples"]:
        seconds = S.rational_value(sample["seconds"])
        assert 0 <= seconds < 0.5
    assert plan["samples"][0]["frame_index"] == rows[0]["frame_index"]
    assert plan["samples"][-1]["frame_index"] == rows[4]["frame_index"]


def test_roles_cover_edges_boundary_and_dedupe(store):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=4, boundary_frames=[5], budget=VSmp.SamplingBudget(boundary_radius=1))
    roles = {sample["frame_index"]: sample["roles"] for sample in plan["samples"]}
    first = plan["samples"][0]["frame_index"]
    last = plan["samples"][-1]["frame_index"]
    assert "first" in roles[first]
    assert "last" in roles[last]
    assert {"boundary_before", "boundary", "boundary_after"} <= {
        role for values in roles.values() for role in values}


def test_boundary_neighbourhood_is_materialised(store):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=0, boundary_frames=[5],
                             include_edges=False,
                             budget=VSmp.SamplingBudget(boundary_radius=1, max_frames=8))
    indices = [sample["frame_index"] for sample in plan["samples"]]
    assert indices == ["4", "5", "6"]
    assert plan["samples"][0]["roles"] == ["boundary_before"]
    assert plan["samples"][1]["roles"] == ["boundary"]
    assert plan["samples"][2]["roles"] == ["boundary_after"]


def test_budget_truncation_is_reported(store):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=6,
                             budget=VSmp.SamplingBudget(max_frames=3))
    assert plan["truncated"] is True
    assert len(plan["samples"]) == 3
    assert plan["coverage"]["complete"] is False


def test_invalid_windows_are_refused(store):
    media_id = _media(store)
    with pytest.raises(SchemaInvalid):
        VSmp.plan_samples(store, media_id, window={"start": {"n": "1", "d": "1"},
                                                   "end": {"n": "1", "d": "2"}})
    with pytest.raises(SchemaInvalid):
        VSmp.plan_samples(store, media_id, window={"start": {"n": "0", "d": "1"},
                                                   "end": {"n": "3", "d": "1"}})
    with pytest.raises(SchemaInvalid):
        VSmp.plan_samples(store, media_id, uniform=999)


class _LargeFrameService:
    """Large-map double that fails if the planner tries to scan every row."""

    def __init__(self, total=5001, rate=10):
        self.total = total
        self.rate = rate
        self.digest = "f" * 64
        self.pages = []
        self.lookups = []

    def _row(self, index):
        return {"frame_index": str(index), "source_pts": str(index),
                "proxy_pts": str(index),
                "seconds": {"n": str(index), "d": str(self.rate)}}

    def page(self, _media_id, offset=0, limit=100):
        self.pages.append((offset, limit))
        assert limit <= 1000
        return {"total": str(self.total), "frame_map_sha256": self.digest,
                "frames": [self._row(index)
                           for index in range(offset, min(self.total, offset + limit))]}

    def lookup(self, _media_id, target, mode):
        assert mode == "ceil"
        self.lookups.append(target)
        position = int(math.ceil(float(S.rational_value(target)) * self.rate))
        position = min(max(0, position), self.total)
        return {"frame_index": str(position), "frame": None,
                "total": str(self.total), "frame_map_sha256": self.digest}


class _LargeStore:
    def __init__(self, service):
        self._frame_service = service

    def get_media(self, _media_id):
        return {"duration": {"n": str(self._frame_service.total),
                             "d": str(self._frame_service.rate)}}


def test_large_map_window_reads_only_selected_rows_and_clips_boundary_hints():
    service = _LargeFrameService()
    store = _LargeStore(service)
    plan = VSmp.plan_samples(
        store, "med-large",
        window={"start": {"n": "1", "d": "1"},
                "end": {"n": "2", "d": "1"}},
        uniform=3, boundary_frames=[1, 15, 30],
        budget=VSmp.SamplingBudget(max_frames=16, boundary_radius=1))
    indices = [int(sample["frame_index"]) for sample in plan["samples"]]
    assert indices == [10, 14, 15, 16, 19]
    assert all(10 <= index < 20 for index in indices)
    assert sum(limit for _offset, limit in service.pages) == 6
    assert len(service.lookups) == 2


def test_large_map_boundary_lookup_does_not_materialise_frame_table():
    service = _LargeFrameService(total=100_001)
    store = _LargeStore(service)
    assert VSmp.boundary_indices(
        store, "med-large", [{"n": "1", "d": "1"},
                             {"n": "99999", "d": "10"}]) == [10, 99_999]
    assert service.pages == [(0, 1)]
    assert len(service.lookups) == 2


def test_boundary_lookup_rejects_unbounded_requests_and_identity_changes():
    service = _LargeFrameService()
    store = _LargeStore(service)
    with pytest.raises(SchemaInvalid, match="planner bound"):
        VSmp.boundary_indices(
            store, "med-large",
            [{"n": str(index), "d": "10"}
             for index in range(VSmp.MAX_PLAN_FRAMES + 1)])
    original_lookup = service.lookup

    def changed_lookup(media_id, target, mode):
        result = original_lookup(media_id, target, mode)
        result["frame_map_sha256"] = "0" * 64
        return result

    service.lookup = changed_lookup
    with pytest.raises(WorkspaceError, match="identity changed"):
        VSmp.boundary_indices(store, "med-large", [{"n": "1", "d": "1"}])


def test_boundary_only_plan_refuses_hints_outside_window():
    service = _LargeFrameService()
    store = _LargeStore(service)
    with pytest.raises(SchemaInvalid, match="selected no frames"):
        VSmp.plan_samples(
            store, "med-large",
            window={"start": {"n": "1", "d": "1"},
                    "end": {"n": "2", "d": "1"}},
            uniform=0, boundary_frames=[100], include_edges=False)


def test_large_boundary_set_is_rejected_at_the_planner_bound():
    service = _LargeFrameService(total=10_000)
    store = _LargeStore(service)
    with pytest.raises(SchemaInvalid, match="planner bound"):
        VSmp.plan_samples(
            store, "med-large", uniform=0,
            boundary_frames=range(VSmp.MAX_PLAN_FRAMES + 1), include_edges=False,
            budget=VSmp.SamplingBudget(max_frames=16, boundary_radius=0))


def test_decode_verifies_count_identity_and_bytes(store, tmp_path):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=3,
                             window={"start": {"n": "0", "d": "1"},
                                     "end": {"n": "7", "d": "10"}})
    evidence = VSmp.decode_samples(store, media_id, plan, tmp_path / "frames",
                                   timeout=120)
    assert evidence["count_verified"] is True
    assert len(evidence["decoded"]) == len(plan["samples"])
    assert evidence["playback_sha256"] == store.get_media(media_id)["playback"]["sha256"]
    for item, planned in zip(evidence["decoded"], plan["samples"]):
        assert item["frame_index"] == planned["frame_index"]
        assert item["source_pts"] == planned["source_pts"]
        assert item["proxy_pts"] == planned["proxy_pts"]
        assert item["seconds"] == planned["seconds"]
        assert item["sha256"] == S.sha256_bytes(open(item["path"], "rb").read())


def test_plan_fingerprint_is_stable_and_time_independent(store):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=3)
    assert VSmp.samples_fingerprint(plan) == VSmp.samples_fingerprint(plan)
    widened = VSmp.plan_samples(store, media_id, uniform=5)
    assert VSmp.samples_fingerprint(plan) != VSmp.samples_fingerprint(widened)


def test_decode_many_frames_uses_one_absolute_filter_script(tmp_path):
    source = tmp_path / "long.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
        "-i", "testsrc2=s=64x64:r=10:d=6.0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-g", "10", str(source),
    ], check=True, capture_output=True, timeout=60)
    project = WorkspaceStore(tmp_path / "long-workspace")
    project.create()
    try:
        project.import_source(source)
        media_id = project.get_task(project.list_tasks()[0]["task_id"])["media_id"]
        plan = VSmp.plan_samples(project, media_id, uniform=60,
                                 budget=VSmp.SamplingBudget(max_frames=60, max_uniform=60))
        evidence = VSmp.decode_samples(project, media_id, plan, tmp_path / "long-frames",
                                       timeout=120)
        assert len(plan["samples"]) == 60
        assert len(evidence["decoded"]) == 60
        assert len(evidence["commands"]) == 1
        command = evidence["commands"][0]
        assert command[command.index("-frames:v") + 1] == "60"
        script = tmp_path / "long-frames" / ".decode-0000" / "select.filter"
        script_option = "-/filter:v" if "-/filter:v" in command else "-filter_script:v"
        assert command[command.index(script_option) + 1] == str(script)
        assert script.read_text(encoding="ascii") == "select=between(n\\,0\\,59)"
        assert all("-ss" not in command for command in evidence["commands"])
        assert [item["frame_index"] for item in evidence["decoded"]] == [
            item["frame_index"] for item in plan["samples"]]
    finally:
        project.close()


def _fake_binding(monkeypatch, tmp_path, count=1):
    source = tmp_path / "playback.mp4"
    source.write_bytes(b"immutable playback fixture")
    monkeypatch.setattr("trace_annotate.workspace.frames._binding",
                        lambda *args: ({}, source, S.sha256_bytes(source.read_bytes())))
    monkeypatch.setattr(VSmp, "_filter_script_option", lambda *args: "-/filter:v")
    samples = [{"frame_index": str(index), "source_pts": str(index),
                "proxy_pts": str(index), "seconds": {"n": str(index), "d": "1"},
                "roles": ["uniform"]} for index in range(count)]
    return {"samples": samples, "frame_map_sha256": "a" * 64,
            "time_domain": "playback_seconds"}


@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_decode_stops_owned_process_and_keeps_diagnostics(monkeypatch, tmp_path, reason):
    plan = _fake_binding(monkeypatch, tmp_path)
    original_run = P._run
    original_popen = P.subprocess.Popen
    processes = []
    owned = []

    def launch(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    def sleeping_decoder(command, directory, name, cancelled):
        return original_run([sys.executable, "-c",
                             "import sys,time; sys.stderr.write('decoder started\\n'); "
                             "sys.stderr.flush(); time.sleep(60)"],
                            directory, name, cancelled)

    def cancel():
        active = [child for child in P.live_media_processes()
                  if child["name"] == "decode"]
        owned.extend(active)
        return reason == "cancel" and bool(active)

    monkeypatch.setattr(P.subprocess, "Popen", launch)
    monkeypatch.setattr(P, "_run", sleeping_decoder)
    started = time.monotonic()
    output = tmp_path / "frames"
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, output,
                            timeout=0.4 if reason == "timeout" else 10,
                            cancelled=cancel, ffmpeg=sys.executable)
    assert failure.value.code == ("CANCELLED" if reason == "cancel" else "FRAME_DECODE_TIMEOUT")
    assert time.monotonic() - started < 5
    assert len(processes) == 1 and processes[0].poll() is not None
    assert owned and all(record["pid"] == processes[0].pid for record in owned)
    assert processes[0].pid not in {record["pid"] for record in P.live_media_processes()}
    entry = json.loads((output / ".decode-0000" / "commands.jsonl").read_text())
    assert entry["pid"] == processes[0].pid and entry["exit_code"] is not None
    assert entry["cancelled_or_interrupted"] is True
    assert (output / ".decode-0000" / "decode.stderr").is_file()
    assert not list(output.glob("sample-*.png"))


def test_decode_keeps_one_deadline_during_single_process(monkeypatch, tmp_path):
    plan = _fake_binding(monkeypatch, tmp_path, count=41)
    clock = [0.0]
    calls = []
    monkeypatch.setattr(VSmp.time, "monotonic", lambda: clock[0])

    def run(command, directory, name, cancelled):
        cancelled()
        calls.append(command)
        clock[0] += 6
        cancelled()
        clock[0] += 5
        cancelled()

    monkeypatch.setattr(P, "_run", run)
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, tmp_path / "frames",
                            timeout=10, ffmpeg=sys.executable)
    assert failure.value.code == "FRAME_DECODE_TIMEOUT"
    assert len(calls) == 1


def test_decode_byte_budget_stops_output_before_next_frame(monkeypatch, tmp_path):
    plan = _fake_binding(monkeypatch, tmp_path, count=41)
    calls = []

    def run(command, directory, name, cancelled):
        calls.append(command)
        (directory / "frame-0001.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        (directory / "decode.stderr").write_text("retained decoder diagnostic", encoding="utf-8")
        (directory / "commands.jsonl").write_text(json.dumps({"exit_code": 0}) + "\n")
        cancelled()
        pytest.fail("byte budget did not stop the decoder")

    monkeypatch.setattr(P, "_run", run)
    output = tmp_path / "frames"
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, output,
                            max_bytes=1, ffmpeg=sys.executable)
    assert failure.value.code == "FRAME_TOO_LARGE"
    assert len(calls) == 1
    assert not (output / ".decode-0001").exists()
    assert not (output / ".decode-0000" / "frame-0002.png").exists()
    assert (output / ".decode-0000" / "decode.stderr").read_text() == "retained decoder diagnostic"
    assert json.loads((output / ".decode-0000" / "commands.jsonl").read_text())["exit_code"] == 0
    assert not list(output.glob("sample-*.png"))


def test_decode_byte_budget_terminates_owned_process(monkeypatch, tmp_path):
    plan = _fake_binding(monkeypatch, tmp_path, count=41)
    original_run = P._run
    original_popen = P.subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    def oversized_decoder(command, directory, name, cancelled):
        writer = ("from pathlib import Path; import sys,time; "
                  "Path(sys.argv[1]).write_bytes(b'\\x89PNG\\r\\n\\x1a\\n' + b'x'*4096); "
                  "time.sleep(60)")
        return original_run([sys.executable, "-c", writer,
                             str(directory / "frame-0001.png")],
                            directory, name, cancelled)

    monkeypatch.setattr(P.subprocess, "Popen", launch)
    monkeypatch.setattr(P, "_run", oversized_decoder)
    output = tmp_path / "frames"
    started = time.monotonic()
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, output,
                            timeout=10, max_bytes=1024, ffmpeg=sys.executable)
    assert failure.value.code == "FRAME_TOO_LARGE"
    assert time.monotonic() - started < 5
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].pid not in {record["pid"] for record in P.live_media_processes()}
    entry = json.loads((output / ".decode-0000" / "commands.jsonl").read_text())
    assert entry["pid"] == processes[0].pid
    assert entry["exit_code"] is not None and entry["cancelled_or_interrupted"] is True
    assert not (output / ".decode-0000" / "frame-0002.png").exists()
    assert not list(output.glob("sample-*.png"))


@pytest.mark.parametrize("indices", [[], [1, 1], [3, 2], [-1]])
def test_select_expression_rejects_nonascending_indices(indices):
    with pytest.raises(SchemaInvalid):
        VSmp._select_expression(indices)


def test_sparse_filter_script_decodes_beyond_expression_depth_limit(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for sparse filter parser regression")
    indices = list(range(0, 600, 2))
    expression = VSmp._select_expression(indices)
    # The script contains only controlled integer selectors and has logarithmic
    # parser depth; a left-associated sum of this size fails on FFmpeg 9.
    script = tmp_path / "sparse.filter"
    script.write_text(expression, encoding="ascii")
    output = tmp_path / "sparse-%04d.png"
    script_option = VSmp._filter_script_option(shutil.which("ffmpeg"), tmp_path, lambda: False)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
               "-filter_threads", "1", "-f", "lavfi", "-i",
               "testsrc2=s=16x16:r=100:d=6", script_option, str(script),
               "-fps_mode", "passthrough", "-c:v", "png", "-threads:v", "1",
               "-frames:v", str(len(indices)), str(output)]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    assert len(list(tmp_path.glob("sparse-*.png"))) == len(indices)


@pytest.mark.parametrize("version,option", [("6.0", "-filter_script:v"),
                                            ("n8.0", "-filter_script:v"),
                                            ("9.0.1-full_build", "-/filter:v")])
def test_filter_script_option_matches_released_toolchain(monkeypatch, tmp_path, version, option):
    def run(command, directory, name, cancelled):
        assert command == ["ffmpeg", "-version"]
        cancelled()
        receipt = directory / f"{name}.stdout"
        receipt.write_text(f"ffmpeg version {version} Copyright\n", encoding="ascii")
        return receipt

    monkeypatch.setattr(P, "_run", run)
    assert VSmp._filter_script_option("ffmpeg", tmp_path, lambda: False) == option


@pytest.mark.parametrize("version", ["N-1234-abcdef", "5.1", "custom build"])
def test_filter_script_option_rejects_unknown_or_unsupported_toolchain(monkeypatch, tmp_path, version):
    def run(command, directory, name, cancelled):
        receipt = directory / f"{name}.stdout"
        receipt.write_text(f"ffmpeg version {version} Copyright\n", encoding="ascii")
        return receipt

    monkeypatch.setattr(P, "_run", run)
    with pytest.raises(DependencyUnavailable) as failure:
        VSmp._filter_script_option("ffmpeg", tmp_path, lambda: False)
    assert failure.value.code == "UNSUPPORTED_TOOLCHAIN"


@pytest.mark.parametrize("max_bytes,timeout", [(0, 1), (True, 1), (1, 0),
                                               (1, float("nan")), (1, True)])
def test_decode_rejects_invalid_resource_limits(monkeypatch, tmp_path, max_bytes, timeout):
    plan = _fake_binding(monkeypatch, tmp_path)
    with pytest.raises(SchemaInvalid):
        VSmp.decode_samples(None, "med-test", plan, tmp_path / "frames",
                            max_bytes=max_bytes, timeout=timeout, ffmpeg=sys.executable)


def test_explicit_decoder_still_verifies_playback_when_path_has_no_ffmpeg(monkeypatch, tmp_path):
    plan = _fake_binding(monkeypatch, tmp_path)
    monkeypatch.setattr(VSmp.shutil, "which", lambda name: None)

    def run(command, directory, name, cancelled):
        (directory / "frame-0001.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        (tmp_path / "playback.mp4").write_bytes(b"changed playback")

    monkeypatch.setattr(P, "_run", run)
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, tmp_path / "frames",
                            ffmpeg=sys.executable)
    assert failure.value.code == "RECOVERY_REQUIRED"


@pytest.mark.parametrize("mode", ["failed", "missing", "invalid_png", "byte_budget"])
def test_decode_preserves_failure_codes_and_byte_cap(monkeypatch, tmp_path, mode):
    plan = _fake_binding(monkeypatch, tmp_path)

    def run(command, directory, name, cancelled):
        if mode == "failed":
            raise P.MediaPreparationError("ffmpeg diagnostic " + "x" * 2000 + " tail marker")
        if mode != "missing":
            (directory / "frame-0001.png").write_bytes(
                b"invalid image" if mode == "invalid_png" else b"\x89PNG\r\n\x1a\nfixture")

    monkeypatch.setattr(P, "_run", run)
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, tmp_path / "frames",
                            max_bytes=1 if mode == "byte_budget" else 1024,
                            ffmpeg=sys.executable)
    assert failure.value.code == ("FRAME_TOO_LARGE" if mode == "byte_budget" else "FRAME_DECODE_FAILED")
    if mode == "failed":
        assert failure.value.message.endswith("tail marker")
        assert len(failure.value.message) < 700


@pytest.mark.parametrize("variable", [False, True], ids=["cfr", "vfr"])
def test_bounded_single_thread_decode_matches_original_png_bytes(tmp_path, variable):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for sampling fixtures")
    source = tmp_path / "source.mp4"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
               "testsrc2=s=64x64:r=10:d=1.0"]
    if variable:
        command += ["-vf", "select=not(eq(n\\,3)+eq(n\\,7))", "-fps_mode", "vfr"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", str(source)]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    project = WorkspaceStore(tmp_path / "workspace")
    project.create()
    try:
        project.import_source(source)
        media_id = _media(project)
        rows = _rows(project, media_id)
        deltas = {S.rational_value(b["seconds"]) - S.rational_value(a["seconds"])
                  for a, b in zip(rows, rows[1:])}
        assert (len(deltas) > 1) == variable
        plan = VSmp.plan_samples(project, media_id, uniform=3)
        evidence = VSmp.decode_samples(project, media_id, plan, tmp_path / "new", timeout=120)
        from trace_annotate.workspace.frames import _binding
        _, playback, _ = _binding(project, media_id, "playback_path")
        legacy = tmp_path / "original"
        legacy.mkdir()
        expression = "select=" + "+".join(f"eq(n\\,{item['frame_index']})" for item in plan["samples"])
        command = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-n", "-xerror",
                   "-protocol_whitelist", "file,pipe", "-err_detect", "explode",
                   "-i", str(playback), "-map", "0:v:0", "-vf", expression,
                   "-fps_mode", "passthrough", "-c:v", "png", "-pix_fmt", "rgb24",
                   "-f", "image2", str(legacy / "frame-%04d.png")]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        original = sorted(legacy.glob("frame-*.png"))
        assert len(original) == len(evidence["decoded"])
        assert [S.sha256_bytes(path.read_bytes()) for path in original] == [
            item["sha256"] for item in evidence["decoded"]]
    finally:
        project.close()


@pytest.mark.parametrize("variable", [False, True], ids=["cfr", "vfr"])
def test_exact_decode_rechecks_map_and_actual_output_pts(tmp_path, variable):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for exact PTS verification")
    source = tmp_path / "source.mp4"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
               "testsrc2=s=64x64:r=10:d=2"]
    if variable:
        command += ["-vf", "select=not(eq(n\\,3)+eq(n\\,7)+eq(n\\,12))",
                    "-fps_mode", "vfr"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    project = WorkspaceStore(tmp_path / "workspace")
    project.create()
    try:
        project.import_source(source)
        media_id = _media(project)
        plan = VSmp.plan_samples(project, media_id, uniform=8,
                                 budget=VSmp.SamplingBudget(max_frames=8))
        evidence = VSmp.decode_samples(
            project, media_id, plan, tmp_path / "exact", verify_output_pts=True,
            max_bytes=16 * 1024 * 1024, timeout=120)
        assert evidence["output_pts_verified"] is True
        base = S.rational_value(project.get_media(media_id)["proxy_time_base"])
        assert evidence["output_pts"] == [S.rational(int(item["proxy_pts"]) * base)
                                          for item in plan["samples"]]
        assert len(evidence["pts_log_sha256"]) == 64
        assert len(evidence["output_pixel_checksums"]) == len(plan["samples"])
        changed = json.loads(json.dumps(plan))
        changed["samples"][0]["proxy_pts"] = S.rational(99)
        with pytest.raises(WorkspaceError, match="identity changed"):
            VSmp.decode_samples(project, media_id, changed, tmp_path / "forged",
                                verify_output_pts=True)
        assert not (tmp_path / "forged" / ".decode-0000").exists()
    finally:
        project.close()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reversed", "wrong_pts",
                                       "error", "warning", "invalid"])
def test_exact_output_pts_parser_fails_closed(tmp_path, mutation):
    path = tmp_path / "decode.stderr"
    lines = ["[Parsed_showinfo_2 @ 0x1] [info] config in time_base: 1/1000, frame_rate: 10/1",
             "[Parsed_showinfo_2 @ 0x1] [info] n:   0 pts:      0 pts_time:0 fmt:rgb24 checksum:00000000",
             "[Parsed_showinfo_2 @ 0x1] [info] n:   1 pts:    100 pts_time:0.1 fmt:rgb24 checksum:00000001"]
    if mutation == "missing":
        lines.pop()
    elif mutation == "duplicate":
        lines.append(lines[-1])
    elif mutation == "reversed":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "wrong_pts":
        lines[-1] = lines[-1].replace("100 pts_time", "101 pts_time")
    elif mutation == "error":
        lines.append("[error] Decoder reported corrupt input")
    elif mutation == "warning":
        lines.append("[warning] Decoder concealed timing discontinuity")
    else:
        lines.append("[info] Invalid frame metadata")
    path.write_text("\n".join(lines), encoding="utf-8")
    if mutation == "wrong_pts":
        assert [item[0] for item in VSmp._observed_output_pts(path, 2)] != [
            S.rational_value(0), S.rational_value({"n": "1", "d": "10"})]
    else:
        with pytest.raises(WorkspaceError) as exc:
            VSmp._observed_output_pts(path, 2)
        assert exc.value.code == "FRAME_DECODE_FAILED"


def test_exact_decode_rejects_swapped_png_after_ffmpeg(store, monkeypatch, tmp_path):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=3,
                             budget=VSmp.SamplingBudget(max_frames=3))
    original = P._run

    def swap(command, directory, name, cancelled, **kwargs):
        result = original(command, directory, name, cancelled, **kwargs)
        if name == "decode":
            first, second = directory / "frame-0001.png", directory / "frame-0002.png"
            hold = directory / "hold.png"
            first.replace(hold)
            second.replace(first)
            hold.replace(second)
        return result

    monkeypatch.setattr(P, "_run", swap)
    with pytest.raises(WorkspaceError, match="pixels differ") as exc:
        VSmp.decode_samples(store, media_id, plan, tmp_path / "swapped",
                            verify_output_pts=True, max_bytes=16 * 1024 * 1024)
    assert exc.value.code == "FRAME_DECODE_FAILED"


def test_exact_decode_uses_proxy_clock_for_nonzero_source_origin_and_vfr(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required")
    source = tmp_path / "offset-vfr.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    "testsrc2=s=64x64:r=10:d=2", "-vf",
                    "select=not(eq(n\\,3)+eq(n\\,7)),setpts=PTS+10/TB",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(source)], check=True, capture_output=True, timeout=60)
    project = WorkspaceStore(tmp_path / "workspace")
    project.create()
    try:
        project.import_source(source)
        media_id = _media(project)
        media = project.get_media(media_id)
        assert S.rational_value(media["common_origin"]) == 10
        plan = VSmp.plan_samples(project, media_id, uniform=8,
                                 budget=VSmp.SamplingBudget(max_frames=8))
        evidence = VSmp.decode_samples(
            project, media_id, plan, tmp_path / "exact-offset",
            verify_output_pts=True, max_bytes=16 * 1024 * 1024)
        base = S.rational_value(media["proxy_time_base"])
        assert evidence["output_pts"] == [S.rational(int(item["proxy_pts"]) * base)
                                          for item in plan["samples"]]
        assert plan["samples"][0]["source_pts"] != plan["samples"][0]["proxy_pts"]
    finally:
        project.close()


def test_exact_decode_rejects_nonzero_ffmpeg_exit_even_with_info_stderr(
        store, monkeypatch, tmp_path):
    media_id = _media(store)
    plan = VSmp.plan_samples(store, media_id, uniform=3,
                             budget=VSmp.SamplingBudget(max_frames=3))
    # -vf treats the filter-script path as an expression and FFmpeg exits nonzero.
    monkeypatch.setattr(VSmp, "_filter_script_option", lambda *_args: "-vf")
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(store, media_id, plan, tmp_path / "bad-filter",
                            verify_output_pts=True, max_bytes=16 * 1024 * 1024)
    assert failure.value.code == "FRAME_DECODE_FAILED"
    records = [json.loads(line) for line in
               (tmp_path / "bad-filter" / ".decode-0000" / "commands.jsonl").read_text().splitlines()]
    assert records[-1]["exit_code"] != 0
