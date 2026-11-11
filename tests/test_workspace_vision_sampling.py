"""PTS-aware sampling: half-open windows, edge/boundary roles, budgets, real decode."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

import pytest

from trace_annotate.temporal import preparation as P
from trace_annotate.workspace import schema as S
from trace_annotate.workspace import vision_sampling as VSmp
from trace_annotate.workspace.errors import SchemaInvalid, WorkspaceError
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


def test_decode_splits_many_select_terms_into_bounded_ffmpeg_calls(tmp_path):
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
        assert len(evidence["commands"]) > 1
        assert [command[command.index("-frames:v") + 1]
                for command in evidence["commands"]] == ["40", "20"]
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


def test_decode_uses_one_deadline_across_chunks(monkeypatch, tmp_path):
    plan = _fake_binding(monkeypatch, tmp_path, count=41)
    clock = [0.0]
    calls = []
    monkeypatch.setattr(VSmp.time, "monotonic", lambda: clock[0])

    def run(command, directory, name, cancelled):
        cancelled()
        calls.append(command)
        count = int(command[command.index("-frames:v") + 1])
        for index in range(1, count + 1):
            (directory / f"frame-{index:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        clock[0] += 6

    monkeypatch.setattr(P, "_run", run)
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, tmp_path / "frames",
                            timeout=10, ffmpeg=sys.executable)
    assert failure.value.code == "FRAME_DECODE_TIMEOUT"
    assert len(calls) == 2


def test_decode_byte_budget_prevents_later_chunks_and_retains_logs(monkeypatch, tmp_path):
    plan = _fake_binding(monkeypatch, tmp_path, count=41)
    calls = []

    def run(command, directory, name, cancelled):
        calls.append(command)
        count = int(command[command.index("-frames:v") + 1])
        for index in range(1, count + 1):
            (directory / f"frame-{index:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        (directory / "decode.stderr").write_text("retained decoder diagnostic", encoding="utf-8")
        (directory / "commands.jsonl").write_text(json.dumps({"exit_code": 0}) + "\n")

    monkeypatch.setattr(P, "_run", run)
    output = tmp_path / "frames"
    with pytest.raises(WorkspaceError) as failure:
        VSmp.decode_samples(None, "med-test", plan, output,
                            max_bytes=1, ffmpeg=sys.executable)
    assert failure.value.code == "FRAME_TOO_LARGE"
    assert len(calls) == 1
    assert not (output / ".decode-0001").exists()
    assert (output / ".decode-0000" / "decode.stderr").read_text() == "retained decoder diagnostic"
    assert json.loads((output / ".decode-0000" / "commands.jsonl").read_text())["exit_code"] == 0
    assert not list(output.glob("sample-*.png"))


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
