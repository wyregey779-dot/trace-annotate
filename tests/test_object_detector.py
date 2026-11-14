"""NanoDet COCO observations keep labels, geometry and capability boundaries."""
from __future__ import annotations

import hashlib
import json
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from trace_annotate.extractors import objects as O


def outputs():
    scores_all, boxes_all = [], []
    for stride in (8, 16, 32):
        count = (416 // stride) ** 2
        scores = np.zeros((1, count, 80), dtype=np.float32)
        boxes = np.full((1, count, 32), -30, dtype=np.float32)
        boxes.reshape(1, count, 4, 8)[:, :, :, 2] = 30
        scores_all.append(scores)
        boxes_all.append(boxes)
    return scores_all + boxes_all


def test_coco_label_contract_is_frozen_to_upstream_revision():
    encoded = json.dumps(O.COCO80_LABELS, separators=(",", ":")).encode("ascii")
    assert len(O.COCO80_LABELS) == 80
    assert O.COCO80_LABELS[0] == "person"
    assert O.COCO80_LABELS[-1] == "toothbrush"
    assert hashlib.sha256(encoded).hexdigest() == O.COCO80_LABELS_SHA256
    assert O.OFFICIAL_UPSTREAM_REVISION in O.LABEL_SOURCE


def test_non_person_class_and_letterbox_are_recovered_without_claiming_probability():
    values = outputs()
    index = 20 * 52 + 20
    values[0][0, index, 2] = 0.9
    values[0][0, index + 4, 0] = 0.98  # person stays in the subject provider
    result = O.decode_object_outputs(values, (832, 416), (0.5, 0.5, 0.0, 104.0))
    assert len(result) == 1
    assert result[0]["box"] == pytest.approx([295.0, 87.0, 359.0, 151.0])
    assert result[0]["class_id"] == 2 and result[0]["class_label"] == "car"
    assert result[0]["score"] == pytest.approx(0.9)
    assert result[0]["score_kind"] == "model_score_uncalibrated"
    assert result[0]["coordinate_space"] == "decoded_frame_xyxy"


def test_person_projection_requires_explicit_opt_in():
    values = outputs()
    values[0][0, 100, 0] = 0.9
    assert O.decode_object_outputs(values, (416, 416), (1.0, 1.0, 0.0, 0.0)) == []
    included = O.decode_object_outputs(
        values, (416, 416), (1.0, 1.0, 0.0, 0.0), include_person=True)
    assert [(item["class_id"], item["class_label"]) for item in included] == [(0, "person")]


def test_nms_is_per_class_and_preserves_overlapping_different_classes():
    values = outputs()
    anchor = 20 * 52 + 20
    values[0][0, anchor, 2] = 0.9
    values[0][0, anchor + 1, 2] = 0.8  # overlapping car: suppressed
    values[0][0, anchor + 2, 7] = 0.85  # overlapping truck: retained
    result = O.decode_object_outputs(values, (416, 416), (1.0, 1.0, 0.0, 0.0))
    assert [item["class_label"] for item in result] == ["car", "truck"]


def test_output_budgets_are_enforced_deterministically():
    values = outputs()
    for offset in range(6):
        values[0][0, (4 + offset * 7) * 52 + 4, 2] = 0.9 - offset * 0.01
    limited = O.decode_object_outputs(
        values, (416, 416), (1.0, 1.0, 0.0, 0.0), max_objects=3,
        max_per_class=3)
    assert len(limited) == 3
    assert [item["score"] for item in limited] == sorted(
        (item["score"] for item in limited), reverse=True)


@pytest.mark.parametrize("bad", ["count", "class_shape", "box_shape", "nan", "inf", "score_range"])
def test_unexpected_model_output_is_rejected(bad):
    values = outputs()
    if bad == "count":
        values.pop()
    elif bad == "class_shape":
        values[0] = np.zeros((1, 3, 80))
    elif bad == "box_shape":
        values[3] = np.zeros((1, 2704, 31))
    elif bad == "nan":
        values[0][0, 0, 0] = np.nan
    elif bad == "inf":
        values[3][0, 0, 0] = np.inf
    else:
        values[0][0, 0, 2] = 2.0
    with pytest.raises(ValueError):
        O.decode_object_outputs(values, (416, 416), (1.0, 1.0, 0.0, 0.0))


@pytest.mark.parametrize("kwargs", [
    {"threshold": float("nan")}, {"threshold": -0.1}, {"threshold": 1.1},
    {"max_objects": 0}, {"max_per_class": O.MAX_OBJECTS_PER_CLASS + 1},
])
def test_invalid_decoding_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        O.decode_object_outputs(
            outputs(), (416, 416), (1.0, 1.0, 0.0, 0.0), **kwargs)


def test_empty_success_is_distinct_from_no_provider(tmp_path):
    assert O.decode_object_outputs(
        outputs(), (416, 416), (1.0, 1.0, 0.0, 0.0)) == []
    missing = O.ObjectExtractor(str(tmp_path / "missing.onnx"))
    result = missing.detect_objects(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "not_run"
    assert result["object_count"] is None
    assert result["objects_detected"] == []


def test_unknown_weights_are_rejected_before_runtime_and_inherit_no_coco_identity(tmp_path):
    path = tmp_path / "unknown.onnx"
    path.write_bytes(b"not the pinned OpenCV Zoo checkpoint")
    extractor = O.ObjectExtractor(str(path))
    result = extractor.detect_objects(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "unverified_object_model_identity"
    assert extractor.code_identity["model_name"] == "unverified_local_onnx"
    assert extractor.code_identity["license"] == "unverified"
    assert extractor.code_identity["label_set"] is None
    assert extractor.code_identity["label_set_sha256"] is None


def test_expected_weight_digest_mismatch_is_distinct(tmp_path):
    path = tmp_path / "changed.onnx"
    path.write_bytes(b"changed")
    result = O.ObjectExtractor(str(path), expected_sha256="0" * 64).detect_objects(
        np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "weights_hash_mismatch"


def test_official_attribution_requires_exact_verified_digest():
    extractor = O.ObjectExtractor()
    extractor.weights_sha256 = O.OFFICIAL_WEIGHTS_SHA256
    identity = extractor.code_identity
    assert identity["model_name"] == O.PROVIDER
    assert identity["license"] == "Apache-2.0 (OpenCV Zoo)"
    assert identity["label_set"] == "COCO-80"
    assert identity["label_set_sha256"] == O.COCO80_LABELS_SHA256
    assert identity["upstream_revision"] == O.OFFICIAL_UPSTREAM_REVISION


def test_model_byte_budget_is_checked_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "too-large.onnx"
    path.write_bytes(b"1234")
    monkeypatch.setattr(O, "MAX_MODEL_BYTES", 3, raising=False)

    def must_not_open(*args, **kwargs):
        raise AssertionError("oversized object model was opened")

    monkeypatch.setattr(type(path), "open", must_not_open)
    result = O.ObjectExtractor(str(path)).detect_objects(
        np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "object_model_exceeds_byte_budget"


def test_model_loader_rejects_non_regular_path_before_open(tmp_path, monkeypatch):
    path = tmp_path / "linked.onnx"
    path.write_bytes(b"placeholder")
    original = type(path).lstat

    def linked(candidate):
        value = original(candidate)
        if candidate == path:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_size=value.st_size,
                                   st_dev=value.st_dev, st_ino=value.st_ino,
                                   st_mtime_ns=value.st_mtime_ns)
        return value

    monkeypatch.setattr(type(path), "lstat", linked)
    monkeypatch.setattr(type(path), "open", lambda *_a, **_kw:
                        (_ for _ in ()).throw(AssertionError("link was opened")))
    result = O.ObjectExtractor(str(path)).detect_objects(
        np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "object_model_must_be_regular_file"


def test_model_loader_rejects_windows_reparse_point(tmp_path, monkeypatch):
    path = tmp_path / "reparse.onnx"
    path.write_bytes(b"placeholder")
    original = type(path).lstat

    def reparse(candidate):
        value = original(candidate)
        if candidate == path:
            return SimpleNamespace(st_mode=value.st_mode, st_size=value.st_size,
                                   st_dev=value.st_dev, st_ino=value.st_ino,
                                   st_mtime_ns=value.st_mtime_ns,
                                   st_file_attributes=0x400)
        return value

    monkeypatch.setattr(type(path), "lstat", reparse)
    result = O.ObjectExtractor(str(path)).detect_objects(
        np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "object_model_must_be_regular_file"


def test_model_loader_rejects_path_replacement_during_read(tmp_path, monkeypatch):
    path = tmp_path / "replaced.onnx"
    path.write_bytes(b"model bytes")
    original = type(path).lstat
    calls = 0

    def changed(candidate):
        nonlocal calls
        value = original(candidate)
        if candidate != path:
            return value
        calls += 1
        if calls < 2:
            return value
        return SimpleNamespace(st_mode=value.st_mode, st_size=value.st_size,
                               st_dev=value.st_dev, st_ino=value.st_ino + 1,
                               st_mtime_ns=value.st_mtime_ns)

    monkeypatch.setattr(type(path), "lstat", changed)
    result = O.ObjectExtractor(str(path)).detect_objects(
        np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "object_model_changed_during_read"


def test_object_inference_failure_never_publishes_runtime_paths():
    class ExplodingModel:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, *_args, **_kwargs):
            raise RuntimeError(r"cannot read D:\private\models\object.onnx")

    extractor = O.ObjectExtractor()
    extractor.model = ExplodingModel()
    extractor._status, extractor._reason = "success", None
    result = extractor.detect_objects(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["reason"] == "object_inference_failed"
    assert "private" not in json.dumps(result).lower()
