"""PTS-aware sampling: half-open windows, edge/boundary roles, budgets, real decode."""
from __future__ import annotations

import shutil
import subprocess

import pytest

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
