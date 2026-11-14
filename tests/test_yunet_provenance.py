"""YuNet input geometry and weight-specific provenance regressions."""
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trace_annotate.extractors.faces import FaceExtractor
from trace_annotate.workspace import semantic_schema as SS
from trace_annotate.workspace import vision_jobs as V

MODEL = Path(str(resources.files("trace_annotate").joinpath("models", "face_detection_yunet_2023mar.onnx")))
WEIGHT_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
LICENSE_URL = "https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE"


def face_record(identity):
    runs = V._build_model_runs(
        {"id": "job-yunet-provenance", "parameters": {"job_identity": {"config": {}}}},
        {"face": SimpleNamespace(code_identity=identity)}, [],
        {"status": "not_run", "scenes": []}, None, [], {}, "0" * 64, None)
    return next(run for run in runs if run["id"] == V._module_run_id("job-yunet-provenance", "face"))


def test_yunet_model_record_describes_per_frame_sizing_and_separate_licenses():
    record = face_record({"provider": "opencv", "model_name": "YuNet", "detector_kind": "yunet",
                          "weights_path": str(MODEL), "weights_sha256": WEIGHT_SHA,
                          "license": "legacy combined description"})
    assert record["weights_sha256"] == WEIGHT_SHA
    assert record["license"] == "MIT"
    assert record["preprocessing"]["runtime_license"] == "Apache-2.0 (OpenCV)"
    assert record["preprocessing"]["weights_license_source"] == LICENSE_URL
    assert record["preprocessing"]["detector_input"] == "decoded_frame_width_height"
    assert record["preprocessing"]["detector_initial_input"] == [320, 320]
    assert record["status"] == "not_run"  # Metadata alone does not execute inference.


@pytest.mark.parametrize("kind", ["haar", "yunet", None])
def test_custom_and_missing_weights_do_not_inherit_yunet_mit(kind, tmp_path):
    path = tmp_path / "custom.onnx"
    if kind:
        path.write_bytes(b"custom detector weights")
    record = face_record({"provider": "opencv", "model_name": "custom", "detector_kind": kind,
                          "weights_path": str(path), "license": "custom weight terms"})
    assert record["license"] == "custom weight terms"
    assert record["preprocessing"]["weights_license_source"] is None


def test_historical_fixed_size_metadata_remains_readable_without_rewrite():
    historical = face_record({"provider": "opencv", "model_name": "YuNet", "detector_kind": "yunet",
                              "weights_path": str(MODEL), "license": "historical license"})
    historical["preprocessing"] = {"input": "BGR frame as decoded", "detector_kind": "yunet", "detector_input": "320x320"}
    import copy
    before = copy.deepcopy(historical)
    assert SS.validate_model_run(historical) == historical
    assert historical == before


def test_real_yunet_uses_each_non_square_decoded_frame_size():
    pytest.importorskip("cv2")
    extractor = FaceExtractor(model_path=str(MODEL))
    assert extractor.detector_kind == "yunet" and extractor.detector is not None
    for width, height in [(160, 96), (96, 160), (160, 96)]:
        result = extractor.detect_faces(np.zeros((height, width, 3), dtype=np.uint8))
        assert result["status"] == "success"
        assert tuple(extractor.detector.getInputSize()) == (width, height)
        assert extractor._input_size == (width, height)
