"""Regression tests for the official vision benchmark evaluator.

These tests pin the two defects that made the historical n06 scripts unusable:
FFmpeg score/time row pairing and the 1.5/fps boundary tolerance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vision_benchmark as vb  # noqa: E402


def _metadata(*rows: tuple[float, float]) -> str:
    lines = []
    for index, (time_value, score) in enumerate(rows):
        lines.append(f"frame:{index} pts:{index * 1000} pts_time:{time_value:.6f}")
        lines.append(f"lavfi.scene_score={score:.6f}")
    return "\n".join(lines)


def test_parse_scene_metadata_pairs_each_time_with_the_next_score() -> None:
    parsed = vb.parse_scene_metadata(_metadata((0.0, 0.0), (0.033333, 0.1), (0.066667, 0.2)))
    assert parsed["complete"] is True
    assert parsed["frame_count"] == 3
    assert [row["pts_time"] for row in parsed["frames"]] == pytest.approx([0.0, 0.033333, 0.066667])
    assert [row["score"] for row in parsed["frames"]] == pytest.approx([0.0, 0.1, 0.2])


def test_parse_scene_metadata_does_not_shift_scores_to_the_next_frame() -> None:
    parsed = vb.parse_scene_metadata(_metadata((0.0, 0.1), (1.0, 0.9)))
    assert parsed["frames"][0]["score"] == pytest.approx(0.1)
    assert parsed["frames"][1]["score"] == pytest.approx(0.9)


def test_last_frame_with_a_score_is_not_dropped() -> None:
    parsed = vb.parse_scene_metadata(_metadata((0.0, 0.1), (10.0, 0.7)))
    assert parsed["complete"] is True
    assert parsed["frames"][-1]["pts_time"] == pytest.approx(10.0)
    assert parsed["frames"][-1]["score"] == pytest.approx(0.7)


def test_tail_frame_without_a_score_is_reported_as_incomplete() -> None:
    text = _metadata((0.0, 0.1), (1.0, 0.2)) + "\nframe:2 pts:2000 pts_time:2.000000"
    parsed = vb.parse_scene_metadata(text)
    assert parsed["complete"] is False
    assert parsed["frame_count"] == 2
    assert parsed["incomplete_count"] == 1
    assert parsed["incomplete"][0]["frame_index"] == 2
    assert any(issue["kind"] == "missing_score" for issue in parsed["issues"])


def test_middle_frame_without_a_score_keeps_the_following_frame_aligned() -> None:
    text = (
        "frame:0 pts:0 pts_time:0.000000\n"
        "lavfi.scene_score=0.000000\n"
        "frame:1 pts:1000 pts_time:1.000000\n"
        "frame:2 pts:2000 pts_time:2.000000\n"
        "lavfi.scene_score=0.500000\n"
    )
    parsed = vb.parse_scene_metadata(text)
    assert parsed["frame_count"] == 2
    assert [row["pts_time"] for row in parsed["frames"]] == [0.0, 2.0]
    assert [row["score"] for row in parsed["frames"]] == [0.0, 0.5]
    assert parsed["incomplete"][0]["pts_time"] == 1.0


def test_orphan_score_is_an_issue() -> None:
    parsed = vb.parse_scene_metadata("lavfi.scene_score=0.300000\n")
    assert parsed["complete"] is False
    assert any(issue["kind"] == "orphan_score" for issue in parsed["issues"])


def test_duplicate_score_is_an_issue() -> None:
    text = (
        "frame:0 pts:0 pts_time:0.000000\n"
        "lavfi.scene_score=0.100000\n"
        "lavfi.scene_score=0.200000\n"
    )
    parsed = vb.parse_scene_metadata(text)
    assert parsed["complete"] is False
    assert any(issue["kind"] == "duplicate_score" for issue in parsed["issues"])


def test_unparsable_frame_time_is_an_issue() -> None:
    parsed = vb.parse_scene_metadata("frame:0 pts:0 pts_time:N/A\nlavfi.scene_score=0.1\n")
    assert parsed["complete"] is False
    assert any(issue["kind"] == "unparsable_frame_line" for issue in parsed["issues"])


def test_non_numeric_and_out_of_range_scores_are_issues() -> None:
    parsed = vb.parse_scene_metadata("frame:0 pts:0 pts_time:0.000000\nlavfi.scene_score=not-a-number\n")
    assert parsed["complete"] is False
    assert any(issue["kind"] == "unparsable_score" for issue in parsed["issues"])
    parsed = vb.parse_scene_metadata("frame:0 pts:0 pts_time:0.000000\nlavfi.scene_score=1.5\n")
    assert parsed["complete"] is False


def test_frame_table_accepts_ffprobe_csv_with_trailing_comma(tmp_path: Path) -> None:
    table = tmp_path / "frames.csv"
    table.write_text("0.000000,\n0.033333,\n0.066667,\n", encoding="utf-8")
    rows = vb.read_frame_table(table)
    assert [row["pts_time"] for row in rows] == pytest.approx([0.0, 0.033333, 0.066667])
    assert [row["frame_index"] for row in rows] == [0, 1, 2]


def test_frame_table_accepts_vfr_non_uniform_times(tmp_path: Path) -> None:
    table = tmp_path / "vfr.csv"
    table.write_text("0.000000,\n0.033333,\n0.100000,\n0.133333,\n0.200000,\n", encoding="utf-8")
    rows = vb.read_frame_table(table)
    assert [row["pts_time"] for row in rows] == pytest.approx([0.0, 0.033333, 0.1, 0.133333, 0.2])
    times = [row["pts_time"] for row in rows]
    assert vb.nearest_frame_index(times, 0.116) == 2
    assert vb.nearest_frame_index(times, 0.117) == 3


def test_frame_table_rejects_out_of_order_pts(tmp_path: Path) -> None:
    table = tmp_path / "bad.csv"
    table.write_text("1.000000,\n0.000000,\n", encoding="utf-8")
    with pytest.raises(vb.BenchmarkError):
        vb.read_frame_table(table)


def test_nearest_frame_uses_actual_times() -> None:
    frames = [{"frame_index": i, "pts_time": i / 10.0} for i in range(11)]
    assert vb.nearest_frame_index([row["pts_time"] for row in frames], 0.14) == 1
    assert vb.nearest_frame_index([row["pts_time"] for row in frames], 0.16) == 2
    assert vb.nearest_frame_index([row["pts_time"] for row in frames], -5.0) == 0
    assert vb.nearest_frame_index([row["pts_time"] for row in frames], 99.0) == 10


def test_match_boundaries_one_to_one_and_tolerance() -> None:
    frames = [{"frame_index": i, "pts_time": i / 10.0} for i in range(11)]
    result = vb.match_boundaries([0.6], [0.5], frames, tolerance_frames=1)
    assert result["true_positives"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["matches"][0]["distance_frames"] == 1
    result = vb.match_boundaries([0.6], [0.5], frames, tolerance_frames=0)
    assert result["true_positives"] == 0
    assert result["false_positives"] and result["missed"]


def test_match_boundaries_does_not_double_count_duplicate_predictions() -> None:
    frames = [{"frame_index": i, "pts_time": i / 10.0} for i in range(11)]
    result = vb.match_boundaries([0.49, 0.51], [0.5], frames, tolerance_frames=1)
    assert result["true_positives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == 1.0
    assert len(result["false_positives"]) == 1


def test_match_boundaries_perfect_and_known_error() -> None:
    frames = [{"frame_index": i, "pts_time": i / 10.0} for i in range(20)]
    perfect = vb.match_boundaries([0.5, 1.5], [0.5, 1.5], frames, tolerance_frames=1)
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0 and perfect["f1"] == 1.0
    missed = vb.match_boundaries([0.5], [0.5, 1.5], frames, tolerance_frames=1)
    assert missed["precision"] == 1.0 and missed["recall"] == pytest.approx(0.5)


def test_evaluate_thresholds_keeps_confidence_independent() -> None:
    frames = [{"frame_index": i, "pts_time": i / 10.0, "score": 0.0 if i < 5 else 0.8}
              for i in range(10)]
    scores = {"frames": frames}
    reference = {"hard_cut": [{"start": 0.5}, {"start": 1.5}]}
    table = [{"frame_index": i, "pts_time": i / 10.0} for i in range(10)]
    result = vb.evaluate_thresholds(scores, reference, table, [0.5, 0.95])
    assert result["thresholds"][0]["prediction_count"] == 5
    assert result["thresholds"][1]["prediction_count"] == 0
    assert result["best"]["threshold"] == 0.5


def test_freeze_reference_excludes_gradual_from_hard_cut(tmp_path: Path) -> None:
    shots = {
        "seedCuts": [1.0, 2.5],
        "manualCuts": [3.0],
        "shots": [
            {"id": 0, "start": 0.0, "end": 1.0, "transitionIn": "cut"},
            {"id": 1, "start": 1.0, "end": 2.5, "transitionIn": "cut"},
            {"id": 2, "start": 2.5, "end": 3.0, "transitionIn": "fade-in"},
            {"id": 3, "start": 3.0, "end": 4.0, "transitionIn": "cut"},
        ],
    }
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(shots), encoding="utf-8")
    frames = [{"frame_index": i, "pts_time": float(i)} for i in range(5)]
    frozen = vb.freeze_reference(path, frame_table=frames, source_id="synthetic")
    assert frozen["all_boundary_count"] == 3
    assert frozen["hard_cut_count"] == 2
    assert frozen["gradual_count"] == 1
    assert frozen["gradual"][0]["transition_in"] == "fade-in"
    assert frozen["gradual"][0]["interval_basis"] == "shot_span"
    assert frozen["status"] == "FROZEN_HARD_CUT_GRADUAL_INTERVALS_DISPUTED"


def test_freeze_reference_keeps_seed_noise_visible(tmp_path: Path) -> None:
    shots = {
        "seedCuts": [0.5, 1.0, 9.0],
        "manualCuts": [1.0],
        "shots": [
            {"id": 0, "start": 0.0, "end": 1.0, "transitionIn": "cut"},
            {"id": 1, "start": 1.0, "end": 2.0, "transitionIn": "cut"},
        ],
    }
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(shots), encoding="utf-8")
    frozen = vb.freeze_reference(path, source_id="synthetic")
    assert frozen["reconciliation"]["seed_cuts_not_in_shots"] == [0.5, 9.0]
    assert frozen["reconciliation"]["manual_cuts_not_in_shots"] == []
    assert [row["time"] for row in frozen["disputed"]] == [0.5, 9.0]
    assert all(row["source"] == "seedCuts" for row in frozen["disputed"])


def test_build_scene_command_requests_all_frames() -> None:
    command = vb.build_scene_command(Path("demo.mp4"), "ffmpeg.exe")
    assert "metadata=print:file=-" in command[command.index("-vf") + 1]
    assert "gte(scene,0)" in command[command.index("-vf") + 1]
    assert command[-1] == "-"


def test_cli_boundary_metrics_writes_input_hashes(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps({"frames": [
        {"frame_index": 0, "pts_time": 0.0, "score": 0.0},
        {"frame_index": 1, "pts_time": 0.1, "score": 0.9},
    ]}), encoding="utf-8")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({"hard_cut": [{"start": 0.1}]}), encoding="utf-8")
    frame_path = tmp_path / "frames.csv"
    frame_path.write_text("0.000000,\n0.100000,\n", encoding="utf-8")
    out = tmp_path / "metrics.json"
    exit_code = vb.main([
        "boundary-metrics", "--scores", str(score_path),
        "--reference-freeze", str(reference_path),
        "--frame-table", str(frame_path), "--thresholds", "0.5",
        "--tolerance-frames", "1", "--out", str(out),
    ])
    assert exit_code == 0
    assert out.read_bytes().endswith(b"\n") and b"\r\n" not in out.read_bytes()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["best"]["f1"] == 1.0
    assert payload["inputs"]["scores_sha256"]
