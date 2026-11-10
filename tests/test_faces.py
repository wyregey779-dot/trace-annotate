import tempfile
from pathlib import Path

import pytest

from trace_annotate.config import MetricThresholds
from trace_annotate.extractors.faces import FaceExtractor

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trace_annotate" / "models" / "face_detection_yunet_2023mar.onnx"


def test_yunet_loads_when_process_temp_dir_is_non_ascii(monkeypatch, tmp_path):
    """A Chinese TEMP must not silently disable YuNet.

    The OpenCV ONNX importer is byte-oriented; the loader therefore copies the
    weights to an ASCII path. If it trusts the process temp dir, a Chinese
    workspace temp root makes every copy non-ASCII and the detector degrades to
    None, which the run reports as face=not_run.
    """
    pytest.importorskip("cv2")
    if not MODEL.exists():
        pytest.skip("bundled YuNet weights are absent")
    chinese = tmp_path / "视频标注具身标注" / "tmp"
    chinese.mkdir(parents=True)
    monkeypatch.setenv("TEMP", str(chinese))
    monkeypatch.setenv("TMP", str(chinese))
    monkeypatch.setattr(tempfile, "tempdir", str(chinese))

    extractor = FaceExtractor(model_path=str(MODEL), thresholds=MetricThresholds())

    assert extractor.detector is not None
    assert extractor.detector_kind == "yunet"
