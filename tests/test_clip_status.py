"""CLIP extractor honest-status contract and real zero-shot inference.

Two disciplines are pinned: a missing/broken model reports not_run and never
invents labels, and a real forward pass reports success with a full score map.
The real-inference test is skipped when open_clip/torch or the weight download
is unavailable in the current environment.
"""
import numpy as np
import pytest

from trace_annotate.extractors.clip_engine import CLIPExtractor


def test_clip_missing_weights_reports_not_run(tmp_path):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"not model weights")
    extractor = CLIPExtractor(model_name_or_path=str(weights))
    assert extractor.model is None
    # Even an injected model object cannot turn a broken adapter into a
    # prediction when no text features were computed.
    extractor.model = object()
    result = extractor.classify(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result["status"] == "not_run"
    assert result["screen_type"] is None
    assert result["entity_score"] is None
    assert result["clip_scores"] == {}


def test_clip_real_classify_never_fakes_high_confidence():
    extractor = CLIPExtractor()
    if extractor.model is None:
        pytest.skip("CLIP weights unavailable in this environment")
    result = extractor.classify(np.full((224, 224, 3), (200, 150, 60), dtype=np.uint8))
    assert result["status"] == "success"
    assert set(result["clip_scores"]) == {"io", "tod", "weather", "screen"}
    # Honesty contract: a returned label must be backed by a score at or above
    # the configured cutoff for its category. The adapter may stay silent
    # (None) or commit, but it must never commit below the cutoff.
    binary_cutoff = extractor.thresholds.clip_binary_confidence
    multiclass_cutoff = extractor.thresholds.clip_multiclass_confidence
    for label_key, category in (("indoor_outdoor", "io"), ("time_of_day", "tod")):
        label = result[label_key]
        if label is not None:
            assert result["clip_scores"][category][label] >= binary_cutoff
    for label_key, category in (("weather", "weather"), ("screen_type", "screen")):
        label = result[label_key]
        if label is not None:
            assert result["clip_scores"][category][label] >= multiclass_cutoff
