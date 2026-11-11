"""Offline landmark loading, geometry contract and measured failure paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from trace_annotate.extractors.face_landmarks import FaceLandmarkExtractor, mesh_to_face
from trace_annotate.extractors import expressive_cues as EC


def _mesh():
    points = np.zeros((478, 3), dtype=float)
    points[:, :2] = [0.5, 0.5]
    points[10, :2], points[152, :2] = [0.5, 0.1], [0.5, 0.9]
    points[234, :2], points[454, :2] = [0.2, 0.5], [0.8, 0.5]
    for indices, coordinates in (
        ([33, 160, 158, 133, 153, 144], [[.3,.4],[.35,.39],[.4,.39],[.45,.4],[.4,.41],[.35,.41]]),
        ([362, 385, 387, 263, 373, 380], [[.55,.4],[.6,.39],[.65,.39],[.7,.4],[.65,.41],[.6,.41]]),
        ([61, 291, 13, 14], [[.35,.67],[.65,.67],[.5,.65],[.5,.82]]),
        ([107, 336, 1], [[.46,.3],[.54,.3],[.5,.53]]),
    ):
        points[indices, :2] = coordinates
    return points


def _fake_runtime(monkeypatch, tmp_path, *, fail=False):
    model = tmp_path / "face.task"
    model.write_bytes(b"fixture model, no real inference")
    output = SimpleNamespace(face_landmarks=[
        [SimpleNamespace(x=x, y=y, z=z) for x,y,z in _mesh()]
    ])
    class Runtime:
        closed = False
        def detect(self, image):
            if fail:
                raise RuntimeError("forward fixture failure")
            return output
        def close(self):
            self.closed = True
    runtime = Runtime()
    mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda **kw: kw)
    monkeypatch.setattr(FaceLandmarkExtractor, "_create_runtime", lambda self: (mp, runtime))
    return FaceLandmarkExtractor(str(model)), runtime


def test_missing_weights_are_not_run_and_never_attempt_runtime(monkeypatch):
    def unexpected(self):
        raise AssertionError("must not import or download")
    monkeypatch.setattr(FaceLandmarkExtractor, "_create_runtime", unexpected)
    result = FaceLandmarkExtractor().detect_landmarks(np.zeros((200,200,3),np.uint8))
    assert result["status"] == "not_run"
    assert result["reason"] == "landmark_weights_missing"


def test_present_bad_weights_are_failed(monkeypatch, tmp_path):
    path = tmp_path / "broken.task"
    path.write_bytes(b"broken")
    def broken(self):
        raise ValueError("invalid task model")
    monkeypatch.setattr(FaceLandmarkExtractor, "_create_runtime", broken)
    result = FaceLandmarkExtractor(str(path)).detect_landmarks(np.zeros((2,2,3),np.uint8))
    assert result["status"] == "failed"
    assert "invalid task model" in result["error"]


def test_absent_dependency_is_not_run(monkeypatch, tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"model")
    def missing(self):
        raise ImportError("mediapipe missing")
    monkeypatch.setattr(FaceLandmarkExtractor, "_create_runtime", missing)
    result = FaceLandmarkExtractor(str(path)).detect_landmarks(np.zeros((2,2,3),np.uint8))
    assert result["status"] == "not_run"
    assert result["reason"] == "landmark_runtime_unavailable"


def test_forward_maps_xyxy_and_real_landmark_pixels(monkeypatch, tmp_path):
    extractor, runtime = _fake_runtime(monkeypatch, tmp_path)
    result = extractor.detect_landmarks(np.zeros((200,300,3),np.uint8))
    assert result["status"] == "success"
    face = result["faces_detected"][0]
    assert face["box"] == pytest.approx([60,20,240,180])
    assert face["frame_size"] == [300,200]
    assert face["landmarks"]["mouth"][0] == pytest.approx([105,134])
    assert face["confidence"] is None
    assert extractor.code_identity["weights_sha256"]
    extractor.close()
    assert runtime.closed


def test_forward_exception_is_failed(monkeypatch, tmp_path):
    extractor, _ = _fake_runtime(monkeypatch, tmp_path, fail=True)
    result = extractor.detect_landmarks(np.zeros((200,200,3),np.uint8))
    assert result["status"] == "failed"
    assert "forward fixture failure" in result["error"]


@pytest.mark.parametrize("image", [None, np.zeros((2,2),np.uint8), np.zeros((0,2,3),np.uint8),
                                   np.zeros((2,2,3),float)])
def test_invalid_images_are_failed(monkeypatch, tmp_path, image):
    extractor, _ = _fake_runtime(monkeypatch, tmp_path)
    assert extractor.detect_landmarks(image)["status"] == "failed"


def test_mesh_refuses_nan_and_wrong_shape():
    with pytest.raises(ValueError):
        mesh_to_face(np.zeros((5,3)), 200, 200)
    points = _mesh()
    points[13,0] = np.nan
    with pytest.raises(ValueError):
        mesh_to_face(points, 200, 200)


def test_bilateral_eye_closure_is_observable_with_evidence():
    face = mesh_to_face(_mesh(), 300, 200)
    result = EC.face_cue_candidates(face, frame_index="21", evidence_id="ev-21")
    cues = {cue["kind"]: cue for cue in result["cues"]}
    assert "eyes_closed" in cues
    assert "mouth_open" in cues
    assert cues["eyes_closed"]["evidence_ids"] == ["ev-21"]
    claims = EC.cues_to_claims(list(cues.values()))
    assert all(item["claim_type"] == "observable" for item in claims)
    assert any("双眼接近闭合" in item["statement"] for item in claims)


def test_side_view_and_known_occlusion_refuse_cues():
    face = mesh_to_face(_mesh(), 300, 200)
    face["quality"]["side_facing"] = True
    assert EC.face_cue_candidates(face,frame_index="1",evidence_id="ev")["status"] == "not_applicable"
    face["quality"]["side_facing"] = False
    face["occluded"] = True
    assert EC.face_cue_candidates(face,frame_index="1",evidence_id="ev")["status"] == "not_applicable"


def test_narrowed_lids_do_not_claim_closed_eyes():
    points = _mesh()
    points[[160,158,385,387],1] = .384
    points[[153,144,373,380],1] = .416
    result = EC.face_cue_candidates(mesh_to_face(points,300,200),frame_index="1",evidence_id="ev")
    kinds = {item["kind"] for item in result["cues"]}
    assert "eyes_narrowed" in kinds
    assert "eyes_closed" not in kinds


def test_new_provider_does_not_call_anatomical_brow_gap_furrowing():
    points = _mesh()
    points[[107,336],0] = [.49,.51]
    result = EC.face_cue_candidates(mesh_to_face(points,300,200),frame_index="1",evidence_id="ev")
    assert "brow_furrow" not in {item["kind"] for item in result["cues"]}


def test_expected_digest_mismatch_fails_before_runtime(monkeypatch,tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"changed")
    monkeypatch.setattr(FaceLandmarkExtractor,"_create_runtime",lambda self: pytest.fail("must not load changed weights"))
    extractor = FaceLandmarkExtractor(str(path),expected_weights_sha256="a"*64)
    result = extractor.detect_landmarks(np.zeros((200,200,3),np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "landmark_weights_changed"


def test_worker_protocol_rejects_untyped_paths_and_wrong_schema():
    from trace_annotate.extractors.face_landmarks import run_landmark_request,PROTOCOL
    with pytest.raises(ValueError):
        run_landmark_request({"schema":"wrong","images":[]})
    with pytest.raises(ValueError):
        run_landmark_request({"schema":PROTOCOL,"images":[{"path":1,"frame_index":"2"}]})
