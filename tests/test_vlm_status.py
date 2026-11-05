"""VLM caption adapter: honest not_run on missing weights, real caption draft
when a local model is available, and the strict rule that a caption is only a
draft (callers decide where it lands, never as an authoritative fact).
"""
import numpy as np
import pytest

from trace_annotate.extractors.vlm import VLMExtractor


def test_vlm_missing_weights_reports_not_run(tmp_path):
    broken = tmp_path / "not-a-model"
    broken.mkdir()
    extractor = VLMExtractor(model_path=str(broken))
    assert extractor.model is None
    result = extractor.caption(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result["status"] == "not_run"
    assert result["caption"] == ""


def test_vlm_real_caption_is_a_draft():
    extractor = VLMExtractor()
    if extractor.model is None:
        pytest.skip("VLM weights unavailable in this environment")
    result = extractor.caption(np.full((224, 224, 3), (120, 140, 160), dtype=np.uint8))
    # A real forward pass either succeeds with text or fails loudly; it must
    # never claim not_run once a model is loaded.
    assert result["status"] in {"success", "failed"}
    if result["status"] == "success":
        assert isinstance(result["caption"], str)
        assert result["caption"].strip()
