"""Portable object, relation and hand projections remain bounded and honest."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
import subprocess

import pytest
from PIL import Image

from trace_annotate.workspace import professional_store as PStore
from trace_annotate.workspace import schema as S
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for professional round-trip")
    source = tmp_path_factory.mktemp("portable-visual-source") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=blue:s=64x64:r=10:d=1", "-c:v", "libx264", "-pix_fmt",
        "yuv420p", str(source),
    ], check=True, capture_output=True, timeout=60)
    return source


def _run(job_id: str, module: str):
    digest = hashlib.sha256((job_id + ":" + module).encode()).hexdigest()
    return {
        "id": V._module_run_id(job_id, module), "provider": "test-provider",
        "model_name": module, "model_version": "1", "weights_sha256": None,
        "license": "test-only", "preprocessing": {}, "parameters": {},
        "input_sha256": digest, "output_sha256": digest, "code_version": "test",
        "status": "success", "failure_reason": None, "score_kind": "none",
    }


def _hand(scene: int, frame: str, pts, evidence: str, job_id: str, index: int = 0):
    points = [[round(index * 0.01 + number * 0.001, 6), 0.25, 0.0]
              for number in range(21)]
    return {
        "contract": V.HAND_EVIDENCE_CONTRACT,
        "observation_id": f"handobs-{scene}-{frame}-{index}",
        "scene_index": scene, "frame_index": frame, "pts": pts,
        "evidence_sha256": evidence, "pixel_sha256": "b" * 64,
        "model_run_id": V._module_run_id(job_id, "hands"), "hand_index": index,
        "subject_ref": None, "landmark_names": list(V.HAND_LANDMARK_NAMES),
        "landmarks_normalized": points, "landmarks_pixel": points,
        "world_landmarks_m": points, "box_xyxy": [0.0, 0.0, 10.0, 10.0],
        "outside_frame": False, "handedness_raw": "Left",
        "handedness_score": 0.8, "handedness_category_index": 0,
        "handedness_score_kind": "mediapipe_category_score_uncalibrated",
        "handedness_mirrored_input_assumption_verified": False,
        "review_status": "unreviewed",
        "uncertainty": "Unreviewed landmarks; mirror state is unknown.",
    }


def _candidate(evidence: str, *, relative_path: str = "tmp/visual/key.png",
               scenes: int = 1, observations_per_scene: int = 1,
               relations: tuple[str, ...] = ("held_by", "control_transfer"),
               hands_per_scene: int = 1):
    job_id = "job-portable-visual"
    tracks = []
    keyframes = []
    source_relations = []
    for scene in range(scenes):
        frame = str(scene * 10 + 1)
        pts = {"n": str(scene * 10 + 1), "d": "10"}
        hand_values = [_hand(scene, frame, pts, evidence, job_id, number)
                       for number in range(hands_per_scene)]
        keyframes.append({
            "scene_index": scene, "frame_index": frame, "source_pts": pts,
            "timestamp": pts, "seconds": float(scene) + 0.1,
            "evidence_sha256": evidence, "roles": ["middle"],
            "hands": {"status": "success" if hand_values else "not_run",
                      "hands": hand_values},
        })
        for number in range(observations_per_scene):
            observation = {
                "observation_id": f"objobs-{scene}-{number}",
                "scene_index": scene, "frame_index": frame, "pts": pts,
                "class_id": 39, "class_label": "bottle",
                "box_xyxy": [1.0 + number, 2.0, 10.0 + number, 12.0],
                "score": 0.75, "score_kind": "model_score_uncalibrated",
                "evidence_id": evidence,
                "model_run_id": V._module_run_id(job_id, "object"),
                "track_id": f"objtrk-{scene}-{number}",
            }
            tracks.append({
                "contract": "object-tracks-v1", "track_id": observation["track_id"],
                "scene_index": scene, "class_id": 39, "class_label": "bottle",
                "model_run_id": V._module_run_id(job_id, "object"),
                "observations": [observation], "ambiguity": False,
                "ambiguity_reason": None, "termination_reason": "end_of_scene",
            })
        if observations_per_scene:
            for number, relation in enumerate(relations):
                source_relations.append({
                    "contract": "object-relations-v1", "relation": relation,
                    "subject_ref": f"subject-{scene}",
                    "object_ref": f"objtrk-{scene}-0",
                    "recipient_ref": (f"subject-{scene}-recipient"
                                      if relation == "control_transfer" else None),
                    "start_frame": frame, "end_frame": frame,
                    "window_pts": {"start": pts, "end": pts},
                    "evidence_ids": [evidence],
                    "samples": [{"frame_index": frame, "pts": pts,
                                 "object_box": [1.0, 2.0, 10.0, 12.0],
                                 "subject_box": [0.0, 0.0, 20.0, 30.0],
                                 "edge_gap": 0.0, "evidence_ids": [evidence]}],
                    "score": 0.5 + number * 0.1, "score_kind": "rule_score",
                    "review_status": "unreviewed",
                    "uncertainty": "Geometry only; no hand or state evidence.",
                })
    model_runs = [_run(job_id, name) for name in
                  ("scene", "object", "object_relations", "hands")]
    return {
        "job_id": job_id, "processor_version": "1.1",
        "scenes": [{"start": str(index), "end": str(index + 1)}
                   for index in range(scenes)],
        "keyframes": keyframes, "boundary_frames": [], "content_frames": [],
        "object_sampling": {"contract": "object-observations-v1",
                            "tracking": {"contract": "object-tracks-v1",
                                         "tracks": tracks}},
        "object_relations": {"contract": "object-relations-v1",
                             "relations": source_relations},
        "model_runs": model_runs,
        "evidence": [{"relative_path": relative_path, "sha256": evidence,
                      "size_bytes": "0"}],
    }


def test_projection_is_deterministic_bounded_and_uses_safe_relation_names():
    candidate = _candidate("a" * 64, scenes=40, observations_per_scene=3,
                           hands_per_scene=2)
    first = V._build_portable_visual_projection(candidate)
    second = V._build_portable_visual_projection(copy.deepcopy(candidate))
    assert first == second
    manifest = first["manifest"]
    assert manifest["projected"]["object_observations"] <= V.PORTABLE_OBJECT_OBSERVATIONS
    assert manifest["projected"]["object_tracks"] <= V.PORTABLE_OBJECT_TRACKS
    assert manifest["projected"]["relation_candidates"] <= V.PORTABLE_RELATION_CANDIDATES
    assert manifest["projected"]["hand_observations"] <= V.PORTABLE_HAND_OBSERVATIONS
    assert manifest["limits"]["used_parameter_bytes"] <= V.PORTABLE_VISUAL_PROJECTION_BYTES
    assert manifest["limits"]["used_evidence_frames"] <= V.PORTABLE_VISUAL_EVIDENCE_FRAMES
    assert manifest["projection_complete"] is False
    assert any(manifest["omitted"].values())
    names = {item["relation_candidate"] for scene in first["scenes"].values()
             for item in (scene.get("relations") or {}).get("relation_candidates", [])}
    assert "held_candidate" in names
    assert "control_transfer_candidate" in names
    assert "held_by" not in names and "control_transfer" not in names
    assert b'"weights_path"' not in S.canonical_json(first)
    annotations = V._candidate_annotations(candidate)
    assert all(len(item["attributes"]["semantic"]["claims"]) <= V.SS.MAX_CLAIMS
               for item in annotations)
    assert len(S.canonical_json(annotations)) < 768 * 1024


def test_not_run_candidate_does_not_invent_portable_visual_claims():
    candidate = _candidate("a" * 64, observations_per_scene=0,
                           relations=(), hands_per_scene=0)
    projection = V._build_portable_visual_projection(candidate)
    assert projection == {"manifest": None, "scenes": {}}
    annotations = V._candidate_annotations(candidate)
    methods = {claim["method"]["id"] for claim in
               annotations[0]["attributes"]["semantic"]["claims"]}
    assert not methods.intersection(V.PORTABLE_PROJECTION_METHODS)
    assert "visual-candidate-projection-manifest-v1" not in methods


@pytest.mark.parametrize("field,value", [
    ("evidence_sha256", "f" * 64),
    ("pts", {"n": "2", "d": "10"}),
])
def test_portable_projection_requires_exact_frame_digest_and_pts(tmp_path, clip, field, value):
    store = WorkspaceStore(tmp_path / "tamper-workspace")
    store.create()
    try:
        store.import_source(clip)
        path = store.root / "tmp" / "visual" / "key.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "blue").save(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        candidate = _candidate(digest)
        candidate["evidence"][0]["size_bytes"] = str(len(data))
        annotations = V._candidate_annotations(candidate)
        projection_claim = next(claim for claim in annotations[0]["attributes"]["semantic"]["claims"]
                                if claim["method"]["id"] == "object-observation-projection-v1")
        projection_claim["method"]["parameters"]["sample_frames"][0][field] = value
        with pytest.raises(WorkspaceError) as exc:
            V._prepare_candidate_evidence(store, candidate, annotations)
        assert exc.value.code == "VISION_EVIDENCE_UNAVAILABLE"
    finally:
        store.close()


def test_projection_apply_and_professional_export_import_roundtrip(tmp_path, clip):
    source = WorkspaceStore(tmp_path / "projection-source")
    destination = None
    source.create()
    try:
        task_id = source.import_source(clip)["task_id"]
        path = source.root / "tmp" / "visual" / "key.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "blue").save(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        candidate = _candidate(digest)
        candidate["evidence"][0]["size_bytes"] = str(len(data))
        annotations = V._candidate_annotations(candidate)
        publication = V._prepare_candidate_evidence(source, candidate, annotations)
        V._publish_candidate_evidence(source, publication)
        current = source.get_task(task_id)
        PStore.apply_vision_candidates(
            source, task_id, command_id="apply-portable-visual",
            expected_revision=current["revision"],
            expected_sha256=current["content_sha256"],
            candidates=annotations, source_job_id=candidate["job_id"],
            model_runs=candidate["model_runs"], request_body=b"{}")
        saved = source.get_task(task_id)
        semantic = saved["annotations"][0]["attributes"]["semantic"]
        by_method = {claim["method"]["id"]: claim for claim in semantic["claims"]}
        assert V.PORTABLE_PROJECTION_METHODS <= set(by_method)
        assert len(by_method["hand-landmark-observation-projection-v1"]
                   ["method"]["parameters"]["observations"][0]
                   ["landmarks_normalized"]) == 21
        for method in V.PORTABLE_PROJECTION_METHODS:
            assert by_method[method]["evidence_ids"]
            assert set(by_method[method]["evidence_ids"]) <= set(saved["annotations"][0]["evidence_ids"])
        document = json.loads(PStore.export_professional(source, task_id)["data"])
        assert len(S.canonical_json(document)) < 1024 * 1024
        exported = {item["id"]: item for item in document["artifacts"]}
        for method in V.PORTABLE_PROJECTION_METHODS:
            for reference in by_method[method]["evidence_ids"]:
                assert reference in exported and "data_base64" in exported[reference]
                assert hashlib.sha256(base64.b64decode(exported[reference]["data_base64"])).hexdigest() == exported[reference]["sha256"]

        destination = WorkspaceStore(tmp_path / "projection-destination")
        destination.create()
        target_id = destination.import_source(clip)["task_id"]
        target = destination.get_task(target_id)
        PStore.apply_import(destination, target_id, command_id="import-portable-visual",
                            expected_revision=target["revision"],
                            expected_sha256=target["content_sha256"],
                            document=document, request_body=b"{}")
        restored = destination.get_task(target_id)["annotations"][0]
        assert restored["attributes"] == saved["annotations"][0]["attributes"]
        assert restored["evidence_ids"] == saved["annotations"][0]["evidence_ids"]
        reexported = json.loads(PStore.export_professional(destination, target_id)["data"])
        assert restored["attributes"] == reexported["task"]["annotations"][0]["attributes"]
    finally:
        if destination is not None:
            destination.close()
        source.close()
