import numpy as np

from trace_annotate.extractors.clip_engine import CLIPExtractor


def test_unfinished_clip_backend_never_reports_success(tmp_path):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"not model weights")
    extractor = CLIPExtractor(model_name_or_path=str(weights))
    assert extractor.model is None
    # Even an injected model object cannot make placeholder inference a prediction.
    extractor.model = object()
    result = extractor.classify(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result["status"] == "not_run"
    assert result["screen_type"] is None
    assert result["entity_score"] is None
    assert result["clip_scores"] == {}
