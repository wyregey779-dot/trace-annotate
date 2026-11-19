"""OCR execution status and legacy fields, with explicit fake-runtime contracts."""
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trace_annotate.config import MetricThresholds
from trace_annotate.extractors import ocr as module
from trace_annotate.extractors.ocr import OCRExtractor


def extractor(engine, backend="rapidocr_onnxruntime"):
    result = OCRExtractor.__new__(OCRExtractor)
    result.thresholds = MetricThresholds()
    result.engine = engine
    result.runtime_backend = backend
    result.load_error = None
    result._load_status = "not_run"
    return result


def assert_legacy_empty(result, status):
    assert result["status"] == status
    assert result["ocr_text"] == result["ocr_readable"] == result["ocr_clean"] == ""
    assert result["ocr_boxes"] == []


def test_missing_runtime_is_not_run_and_has_load_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    def unexpected_distribution(name):
        pytest.fail(f"Missing OCR runtime must not look up unrelated metadata: {name!r}")
    monkeypatch.setattr(module.metadata, "distribution", unexpected_distribution)
    value = OCRExtractor()
    assert value.load_error
    assert value.code_identity["weights_sha256"] is None
    assert value.code_identity["license"] == "unverified"
    assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "not_run")


@pytest.mark.parametrize("raw", [(None, None), ([], [])])
def test_valid_empty_result_is_success(raw):
    value = extractor(lambda frame: raw)
    assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "success")


def test_engine_failure_is_failed_and_does_not_become_empty_success():
    def crash(frame):
        raise RuntimeError("synthetic OCR inference error")
    result = extractor(crash).extract_text(np.zeros((8, 8, 3), dtype=np.uint8))
    assert_legacy_empty(result, "failed")
    assert "synthetic OCR inference error" in result["error"]


@pytest.mark.parametrize("frame", [None, "bad", np.array([]), np.zeros((0, 4, 3)),
    np.zeros((4, 0, 3)), np.zeros((2, 2, 2)), np.full((2, 2, 3), np.nan),
    np.full((2, 2, 3), "text", dtype=object)])
def test_empty_or_invalid_frame_is_failed_without_invoking_engine(frame):
    def unexpected(frame):
        raise AssertionError("invalid frame reached runtime")
    assert_legacy_empty(extractor(unexpected).extract_text(frame), "failed")


def test_bad_file_is_failed_with_legacy_fields(tmp_path):
    assert_legacy_empty(extractor(None).extract({}, str(tmp_path / "missing.png")), "failed")


def test_numpy_boxes_and_legacy_text_fields_are_preserved():
    box = np.array([[0, 0], [12, 0], [12, 6], [0, 6]], dtype=np.float32)
    result = extractor(lambda frame: ([[box, "中文 Text!", .9876]], [0, 0, 0])).extract_text(
        np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["status"] == "success"
    assert result["ocr_text"] == result["ocr_readable"] == "中文 Text!"
    assert result["ocr_clean"] == "中文Text"
    assert result["ocr_boxes"] == [{"text": "中文 Text!", "score": .988,
        "is_readable": True, "box": box.tolist()}]


def test_modern_object_result_has_equivalent_legacy_fields():
    box = np.array([[[0, 0], [12, 0], [12, 6], [0, 6]]])
    output = SimpleNamespace(boxes=box, txts=("Camera",), scores=(.99,))
    result = extractor(lambda frame: output, "rapidocr").extract_text(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["status"] == "success" and result["ocr_text"] == "Camera"
    assert result["ocr_boxes"][0]["box"] == box[0].tolist()


def test_modern_empty_object_is_success():
    output = SimpleNamespace(boxes=None, txts=None, scores=None)
    value = extractor(lambda frame: output, "rapidocr")
    assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "success")


@pytest.mark.parametrize("raw", [([[[], "text", float("nan")]], None), ([[[1], "text", .8]], None),
                                  ([[]], None), ("malformed", None)])
def test_malformed_engine_result_is_failed(raw):
    assert_legacy_empty(extractor(lambda frame: raw).extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "failed")


def legacy_package(monkeypatch, tmp_path, constructor):
    root = tmp_path / "legacy"
    models = root / "models"
    models.mkdir(parents=True)
    for name in ["ch_PP-OCRv4_det_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                 "ch_PP-OCRv4_rec_infer.onnx"]:
        (models / name).write_bytes(name.encode())
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime",
                        SimpleNamespace(__file__=str(root / "__init__.py"), RapidOCR=constructor))
    return models


def test_initialization_failure_is_reported_as_failed(monkeypatch, tmp_path):
    def broken(**kwargs):
        raise RuntimeError("synthetic session initialization failure")
    legacy_package(monkeypatch, tmp_path, broken)
    value = OCRExtractor()
    assert "initialization failure" in value.load_error
    assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "failed")


def test_legacy_constructor_uses_explicit_local_paths_and_identity(monkeypatch, tmp_path):
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return lambda frame: (None, None)
    models = legacy_package(monkeypatch, tmp_path, construct)
    value = OCRExtractor()
    assert len(calls) == 1
    assert all(str(models) in calls[0][role + "_model_path"] for role in ("det", "cls", "rec"))
    identity = value.code_identity
    assert len(identity["weights_files"]) == 3 and len(identity["weights_sha256"]) == 64
    for item in identity["weights_files"]:
        assert hashlib.sha256(open(item["path"], "rb").read()).hexdigest() == item["sha256"]


def test_modern_missing_local_weights_does_not_initialize_or_download(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    calls = []
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(
        __file__=str(tmp_path / "modern" / "__init__.py"), RapidOCR=lambda **kwargs: calls.append(kwargs)))
    value = OCRExtractor()
    assert not calls
    assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "not_run")


@pytest.mark.parametrize("corrupt", [False, True])
def test_modern_initialization_is_local_explicit_and_checksum_guarded(monkeypatch, tmp_path, corrupt):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    models = tmp_path / "models"
    models.mkdir()
    expected = {}
    for role in ("det", "cls", "rec"):
        data = ("fixture " + role).encode()
        name = role + ".onnx"
        (models / name).write_bytes(data)
        expected[role] = (name, hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(module, "MODERN_MODELS", expected)
    if corrupt:
        (models / "rec.onnx").write_bytes(b"unexpected different weights")
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return lambda frame: SimpleNamespace(boxes=None, txts=None, scores=None)
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(
        __file__=str(tmp_path / "__init__.py"), RapidOCR=construct))
    value = OCRExtractor(model_dir=str(models))
    if corrupt:
        assert not calls and "differs" in value.load_error
        assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "failed")
    else:
        assert len(calls) == 1
        params = calls[0]["params"]
        assert all(params[role.capitalize() + ".model_path"] == str(models / (role + ".onnx"))
                   for role in ("det", "cls", "rec"))
        assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 1
        assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
        assert_legacy_empty(value.extract_text(np.zeros((8, 8, 3), dtype=np.uint8)), "success")


def test_numpy_result_container_does_not_use_ambiguous_truthiness():
    rows = np.empty((1, 3), dtype=object)
    rows[0] = [np.array([[0, 0], [12, 0], [12, 6], [0, 6]]), "Text", .99]
    value = extractor(lambda frame: (rows, None))
    assert value.extract_text(np.zeros((100, 100, 3), dtype=np.uint8))["ocr_text"] == "Text"


def test_inconsistent_modern_result_fields_are_failed():
    output = SimpleNamespace(boxes=np.zeros((1, 4, 2)), txts=("one", "two"), scores=(.9,))
    value = extractor(lambda frame: output, "rapidocr")
    assert_legacy_empty(value.extract_text(np.zeros((100, 100, 3), dtype=np.uint8)), "failed")
