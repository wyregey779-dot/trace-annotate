"""P05: a real attempt may outlive the 120-second lease and still commit.

Two separate observations, labelled so neither is read as a performance claim:

* ``owned_worker_lifetime_over_120_seconds``: a real admission of a real clip,
  with a test-only delay of ``LEASE_DELAY_SECONDS`` inserted *inside* the
  attempt after preparation. The delay is artificial and is declared here; what
  it proves is that the real heartbeat renewed the lease across the original
  120-second boundary, because without renewal the attempt would have failed
  with ``LEASE_LOST`` instead of committing.
* ``real_long_media_prepare``: a real 45-minute source prepared end to end,
  timed, with its renewal trace. It stays inside one lease on this machine; the
  timing is a property of this fixture and hardware, not a benchmark.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from trace_annotate.workspace import media as media_module
from trace_annotate.workspace.intake import DEFAULT_LEASE_SECONDS, AsyncIntakes
from trace_annotate.workspace.store import WorkspaceStore
from tests.support.media_tools import require_media_tools

LEASE_DELAY_SECONDS = 135.0  # > DEFAULT_LEASE_SECONDS on purpose, artificial


def _video(path: Path, *, seconds: float, size: str) -> Path:
    ffmpeg = require_media_tools()[0]
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size={size}:rate=10",
                    "-c:v", "mpeg4", "-q:v", "8", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=600)
    return path


@pytest.fixture(scope="module")
def fast_clip(tmp_path_factory) -> Path:
    return _video(tmp_path_factory.mktemp("p05-long-fast") / "fast.mp4", seconds=1, size="160x120")


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory) -> Path:
    # 2700 s at 640x360: measured at roughly a minute of real preparation on the
    # reference machine. The exact duration is not asserted as a performance
    # result; the test only requires that the real prepare completes.
    return _video(tmp_path_factory.mktemp("p05-long-media") / "long.mp4",
                  seconds=2700, size="640x360")


@pytest.fixture()
def store(tmp_path) -> WorkspaceStore:
    value = WorkspaceStore(tmp_path / "proj")
    value.create()
    yield value
    try:
        value.close()
    except Exception:
        pass


def _upload(store: WorkspaceStore, source: Path, name: str) -> str:
    target = store.tmp_dir / name
    shutil.copy2(source, target)
    return AsyncIntakes(store).enqueue(target, display_name=name)["id"]


def test_owned_worker_lifetime_over_120_seconds(store, fast_clip):
    """The lease is renewed across the original boundary, then the task commits."""
    assert LEASE_DELAY_SECONDS > DEFAULT_LEASE_SECONDS
    original = media_module.prepare_workspace_media
    applied = {"delay": None}

    def slow_prepare(*args, **kwargs):
        prepared = original(*args, **kwargs)
        # Artificial delay owned by the test: it happens after a real
        # preparation and before the admission transaction, so the only way the
        # attempt can survive is a real, fenced lease renewal.
        applied["delay"] = LEASE_DELAY_SECONDS
        time.sleep(LEASE_DELAY_SECONDS)
        return prepared

    media_module.prepare_workspace_media = slow_prepare
    try:
        intake_id = _upload(store, fast_clip, "long-lived.mp4")
        intakes = AsyncIntakes(store)  # default 120 s lease
        started = time.monotonic()
        results = intakes.run_pending()
        elapsed = time.monotonic() - started
    finally:
        media_module.prepare_workspace_media = original

    assert applied["delay"] == LEASE_DELAY_SECONDS
    assert elapsed > DEFAULT_LEASE_SECONDS, elapsed
    assert results and results[0]["state"] == "succeeded", results
    row = intakes.get(intake_id)
    assert row["state"] == "succeeded" and row["task_id"], row
    trace = intakes.last_heartbeat()
    assert trace["intake_id"] == intake_id
    assert trace["stop_reason"] is None, trace
    assert trace["lease_lost"] is False and trace["deadline_exceeded"] is False, trace
    assert trace["elapsed_seconds"] > DEFAULT_LEASE_SECONDS, trace
    # 120 s lease, renewal every lease/3 => at least two real renewals.
    assert trace["renewals"] >= 2, trace
    with store.transaction():
        assert store.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert store.conn.execute("SELECT COUNT(*) FROM admission_completions").fetchone()[0] == 1


def test_real_long_media_prepare(store, long_clip):
    """A real long source prepares end to end and commits exactly one task."""
    intake_id = _upload(store, long_clip, "long.mp4")
    intakes = AsyncIntakes(store)
    started = time.monotonic()
    results = intakes.run_pending()
    elapsed = time.monotonic() - started
    assert results and results[0]["state"] == "succeeded", results
    row = intakes.get(intake_id)
    assert row["state"] == "succeeded" and row["task_id"]
    media = store.get_media(row["media_id"])
    assert int(media["frame_count"]) == 27000, media["frame_count"]
    trace = intakes.last_heartbeat()
    assert trace["stop_reason"] is None and trace["lease_lost"] is False
    assert trace["renewals"] >= 0
    assert trace["elapsed_seconds"] > 0
    # Record the measured preparation time as fixture metadata, not a claim.
    print(f"P05 real_long_media_prepare: {elapsed:.1f}s "
          f"(renewals={trace['renewals']}, frames={media['frame_count']})")
