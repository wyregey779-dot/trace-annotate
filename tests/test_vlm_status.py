"""VLM caption adapter: honest not_run on missing weights, real caption draft
when a local model is available, and the strict rule that a caption is only a
draft (callers decide where it lands, never as an authoritative fact).

A real forward pass that crashes is a *failed* run, not a pass: the success test
therefore asserts status == "success" with non-empty text, and the failure branch
is covered by its own test so the two statuses can never be conflated.
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
    assert result["status"] == "success", f"real inference did not succeed: {result}"
    assert isinstance(result["caption"], str)
    assert result["caption"].strip()


class _RaisingModel:
    def generate(self, **kwargs):
        raise RuntimeError("synthetic inference crash")


class _EmptyProcessor:
    def __call__(self, **kwargs):
        return {}


def test_vlm_runtime_crash_reports_failed():
    extractor = VLMExtractor.__new__(VLMExtractor)
    extractor.model_path = "stub"
    extractor.task_prompt = "<MORE_DETAILED_CAPTION>"
    extractor.model = _RaisingModel()
    extractor.processor = _EmptyProcessor()
    result = extractor.caption(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result["status"] == "failed"
    assert result["caption"] == ""
    assert result["error"]
