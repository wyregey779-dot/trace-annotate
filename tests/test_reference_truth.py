"""Typed reference import must not promote shot starts to benchmark truth."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_reference_truth as reference  # noqa: E402


def write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    video = tmp_path / "media.bin"
    video.write_bytes(b"fixture media identity; no codec decoding performed")
    digest = reference.sha256(video)
    frame_map = write(tmp_path / "frames.json", {
        "streams": [{"index": 0, "codec_type": "video", "time_base": "1/1000", "nb_frames": "4"}],
        # Deliberately VFR and a non-zero first PTS: FPS multiplication is invalid.
        "frames": [{"pts": value} for value in (7, 27, 87, 127)],
    })
    payload = {
        "media": {"source": {"sha256": digest}, "frame_count": "4",
                  "common_origin": {"n": "0", "d": "1"},
                  "source_time_base": {"n": "1", "d": "1000"},
                  "proxy_time_base": {"n": "1", "d": "1000"},
                  "duration": {"n": "100", "d": "1"}},
        "task": {"annotations": [{
            "id": "shot-002", "track": "Shots", "label": "Ingredients", "tags": [],
            "start": {"seconds": {"n": "87", "d": "1000"},
                      "anchor": {"frame_index": "2", "source_pts": "87", "proxy_pts": "87"}},
            "end": {"seconds": {"n": "127", "d": "1000"}},
            "review_status": "unreviewed", "provenance": {"origin": "manual", "actor_id": "codex-assistant"},
        }]},
    }
    annotation = write(tmp_path / "annotations.json", payload)
    return annotation, video, frame_map, digest, payload


def build(inputs):
    annotation, video, frame_map, digest, payload = inputs
    write(annotation, payload)
    return reference.build_reference(annotation, video, frame_map, digest, "fixture")


def decision(inputs):
    annotation, video, frame_map, digest, payload = inputs
    evidence = []
    for index in (1, 2):
        path = annotation.parent / f"frame-{index}.bin"
        path.write_bytes(f"frame fixture {index}".encode())
        evidence.append({"path": path.name, "frame_index": index, "sha256": reference.sha256(path),
                         "source_sha256": digest, "frame_map_sha256": reference.sha256(frame_map)})
    payload["task"]["annotations"][0]["transition_decision"] = {
        "kind": "hard_cut", "status": "reviewed", "actor": {"id": "review-agent", "type": "agent"},
        "evidence": evidence,
    }
    return payload["task"]["annotations"][0]


def test_historical_manual_actor_does_not_become_human_or_hard_cut(inputs):
    output = build(inputs)
    event = output["events"][0]
    assert (event["kind"], event["review_status"], event["adjudication_status"]) == ("unknown", "unreviewed", "pending")
    assert event["source_annotation"]["provenance"]["actor_id"] == "codex-assistant"
    assert output["evaluation_eligible_count"] == 0
    assert output["human_review"] is False and output["holdout_sealed"] is False
    assert output["previously_observed"] is True


def test_real_pts_and_decoded_count_ignore_audio_tail_and_fps(inputs):
    output = build(inputs)
    assert output["frame_map"]["decoded_frame_count"] == 4
    event = output["events"][0]
    assert (event["frame_index"], event["pts"], event["seconds"]) == (2, 87, {"n": "87", "d": "1000"})
    assert event["frame_identity_basis"] == "native_anchor_verified"


def test_professional_source_pts_subtract_common_origin_once(inputs):
    payload = inputs[-1]
    payload["media"]["common_origin"] = {"n": "7", "d": "1000"}
    item = payload["task"]["annotations"][0]
    item["start"] = {"seconds": {"n": "2", "d": "25"},
                     "anchor": {"frame_index": "2", "source_pts": "87", "proxy_pts": "80"}}
    item["end"] = {"seconds": {"n": "3", "d": "25"}}
    event = build(inputs)["events"][0]
    assert (event["pts"], event["seconds"]) == (87, {"n": "2", "d": "25"})


def test_temporal_proxy_pts_do_not_subtract_source_origin_again(inputs):
    annotation, video, frame_map, digest, _ = inputs
    write(frame_map, {"streams": [{"index": 0, "codec_type": "video", "time_base": "1/1000"}],
                      "frames": [{"pts": pts} for pts in (0, 20, 80, 120)]})
    payload = {"media_sha256": digest,
               "metadata": {"prepared_media": {"timeline": {
                   "origin_time_fraction": {"n": "7", "d": "1000"}, "frame_count": "4"}},
                   "assignment_review": {"shot_frame_index": [{"id": "shot-2", "start_frame": 2}]}},
               "segments": [{"id": "shot-2", "track": "镜头内容", "label": "Card",
                             "start": 0.08, "end": 0.12}]}
    write(annotation, payload)
    event = reference.build_reference(annotation, video, frame_map, digest, "proxy")["events"][0]
    assert (event["frame_index"], event["pts"], event["seconds"]) == (2, 80, {"n": "2", "d": "25"})


def test_dissolve_label_and_approximate_interval_remain_pending(inputs):
    item = inputs[-1]["task"]["annotations"][0]
    item["label"] = "Dissolve to ingredients"
    item["tags"] = ["start-approximate", "end-approximate"]
    item["start"] = {"seconds": {"n": "8", "d": "100"}}
    event = build(inputs)["events"][0]
    assert event["kind"] == "dissolve"
    assert event["approximate"] is True
    assert event["frame_index"] is None
    assert event["evaluation_eligible"] is False
    assert event["declared_interval"]["start"] == item["start"]


def test_explicit_hard_cut_without_boundary_evidence_is_isolated(inputs):
    inputs[-1]["task"]["annotations"][0]["transition_type"] = "hard_cut"
    event = build(inputs)["events"][0]
    assert event["kind"] == "hard_cut"
    assert event["evaluation_eligible"] is False
    assert event["adjudication_status"] == "pending"


def test_evidence_bound_agent_decision_can_be_evaluated_without_human_claim(inputs):
    decision(inputs)
    output = build(inputs)
    event = output["events"][0]
    assert output["evaluation_eligible_count"] == 1
    assert event["adjudication_status"] == "confirmed"
    assert event["review_status"] == "agent_reviewed"
    assert event["human_review"] is False


def test_reviewed_decision_must_itself_confirm_the_transition_type(inputs):
    item = decision(inputs)
    item["transition_type"] = "hard_cut"
    del item["transition_decision"]["kind"]
    event = build(inputs)["events"][0]
    assert event["kind"] == "hard_cut"
    assert event["evaluation_eligible"] is False
    assert "reviewed_decision_does_not_confirm_transition_type" in event["ineligible_reasons"]


@pytest.mark.parametrize("damage", ["hash", "source", "map", "missing_previous"])
def test_hard_cut_evidence_integrity_and_frame_pair_are_required(inputs, damage):
    item = decision(inputs)
    evidence = item["transition_decision"]["evidence"]
    if damage == "missing_previous":
        del evidence[0]
        assert build(inputs)["events"][0]["evaluation_eligible"] is False
    else:
        field = {"hash": "sha256", "source": "source_sha256", "map": "frame_map_sha256"}[damage]
        evidence[0][field] = "0" * 64
        with pytest.raises(ValueError, match="evidence"):
            build(inputs)


def test_anchor_mismatch_rejected_instead_of_silently_recomputed(inputs):
    inputs[-1]["task"]["annotations"][0]["start"]["anchor"]["source_pts"] = "88"
    with pytest.raises(ValueError, match="anchor PTS"):
        build(inputs)


@pytest.mark.parametrize("damage", ["nonmonotonic", "count", "wrong_source"])
def test_frame_map_identity_failures_rejected(inputs, damage):
    annotation, video, frame_map, digest, payload = inputs
    data = json.loads(frame_map.read_text())
    if damage == "nonmonotonic":
        data["frames"][2]["pts"] = 27
    elif damage == "count":
        data["streams"][0]["nb_frames"] = "5"
    else:
        with pytest.raises(ValueError, match="source declaration"):
            reference.build_reference(annotation, video, frame_map, "0" * 64, "fixture")
        return
    write(frame_map, data)
    with pytest.raises(ValueError):
        build(inputs)


def test_legacy_rounded_seconds_use_explicit_index_not_nominal_fps(inputs):
    annotation, video, frame_map, digest, payload = inputs
    write(frame_map, {"streams": [{"index": 0, "codec_type": "video", "time_base": "1/30"}],
                      "frames": [{"pts": 0}, {"pts": 1}, {"pts": 2}]})
    legacy = {"media_sha256": digest, "duration": 999,
              "segments": [{"id": "shot-2", "track": "镜头内容", "start": 0.033333, "end": 0.066667,
                            "label": "Card", "review_status": "unreviewed"}],
              "metadata": {"assignment_review": {"shot_frame_index": [{"id": "shot-2", "start_frame": 1}]}}}
    write(annotation, legacy)
    output = reference.build_reference(annotation, video, frame_map, digest, "changsha")
    assert output["frame_map"]["decoded_frame_count"] == 3
    assert output["events"][0]["seconds"] == {"n": "1", "d": "30"}
    assert output["events"][0]["kind"] == "unknown"


def test_existing_native_jsonl_frame_map_is_supported(inputs):
    frame_map = inputs[2]
    frame_map.write_text("\n".join(json.dumps({"frame_index": str(index), "source_pts": str(pts),
                        "proxy_pts": str(pts), "seconds": {"n": str(pts), "d": "1000"}})
                        for index, pts in enumerate((7, 27, 87, 127))), encoding="utf-8")
    event = build(inputs)["events"][0]
    assert event["pts"] == 87
    assert event["frame_identity_basis"] == "native_anchor_verified"


def test_cli_preserves_existing_output_and_rejects_sealed_holdout(inputs, tmp_path):
    annotation, video, frame_map, digest, payload = inputs
    target = tmp_path / "reference.json"
    args = ["--annotation", str(annotation), "--video", str(video), "--frame-map", str(frame_map),
            "--frame-map-source-sha256", digest, "--source-id", "fixture", "--out", str(target)]
    assert reference.main(args) == 0
    original = target.read_bytes()
    with pytest.raises(SystemExit) as exc:
        reference.main(args)
    assert exc.value.code == 2 and target.read_bytes() == original
    with pytest.raises(SystemExit) as exc:
        reference.main(args + ["--split", "sealed_holdout"])
    assert exc.value.code == 2
