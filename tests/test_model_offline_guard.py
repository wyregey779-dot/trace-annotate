"""A missing weight must fail fast and offline, never start a hub download."""
from __future__ import annotations

import os
import time

import pytest

from trace_annotate.extractors import model_loading
from trace_annotate.extractors.clip_engine import CLIPExtractor
from trace_annotate.extractors.vlm import VLMExtractor


@pytest.fixture(autouse=True)
def _offline_default(monkeypatch):
    monkeypatch.delenv(model_loading.ALLOW_DOWNLOAD_ENV, raising=False)


def test_policy_is_offline_by_default_and_opt_in_is_explicit(monkeypatch):
    assert model_loading.offline_mode() is True
    monkeypatch.setenv(model_loading.ALLOW_DOWNLOAD_ENV, "1")
    assert model_loading.downloads_allowed() is True
    assert model_loading.offline_mode() is False


def test_vlm_missing_weight_is_not_run_without_network_delay():
    started = time.monotonic()
    extractor = VLMExtractor(model_path="traceannotate-definitely-not-a-model")
    elapsed = time.monotonic() - started
    assert extractor.model is None
    assert extractor.load_error
    assert elapsed < 20, f"missing weight took {elapsed:.1f}s, hub retries were not disabled"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert extractor.load_policy["offline"] is True
    assert extractor.code_identity["revision"] == "unpinned"
    assert extractor.caption(None)["status"] == "not_run"


def test_clip_missing_weight_degrades_without_network_delay():
    started = time.monotonic()
    extractor = CLIPExtractor(model_name_or_path="traceannotate-definitely-not-a-model")
    elapsed = time.monotonic() - started
    assert extractor.model is None
    assert extractor.load_error
    assert elapsed < 20, f"missing weight took {elapsed:.1f}s, hub retries were not disabled"
    assert extractor.code_identity["offline"] is True
