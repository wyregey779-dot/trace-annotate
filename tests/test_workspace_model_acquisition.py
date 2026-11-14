"""Frozen workspace inference never downloads or trusts unrelated weights."""
from __future__ import annotations

import copy

import pytest

from trace_annotate.workspace import model_worker as M
from trace_annotate.workspace import resource_budget as R
from trace_annotate.workspace.errors import WorkspaceError


def _frame():
    return {"frame_index": "0", "path": "unused.png", "sha256": "a" * 64}


def _caches(tmp_path, monkeypatch):
    cache = tmp_path / "clip"
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(cache))
    monkeypatch.setenv("TRACE_ANNOTATE_VLM_CACHE", str(tmp_path / "vlm"))
    return cache


def test_workspace_download_opt_in_requires_separate_acquisition(tmp_path, monkeypatch):
    _caches(tmp_path, monkeypatch)
    monkeypatch.setenv("TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD", "1")
    config = M.configuration()
    monkeypatch.setattr(R, "run_owned",
                        lambda *args, **kwargs: pytest.fail("inference must not launch a downloader"))
    result = M.run_phase("clip", config, [_frame()], tmp_path / "work",
                         cancelled=lambda: False)
    assert result["execution"] == {
        "status": "not_run", "reason": "model_acquisition_required"}
    assert result["results"][0]["result"]["reason"] == "model_acquisition_required"


def test_default_clip_inventory_ignores_unrelated_shared_checkpoint(tmp_path, monkeypatch):
    cache = _caches(tmp_path, monkeypatch)
    unrelated = cache / "models--other--model/snapshots/rev/weights.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"unrelated")
    monkeypatch.delenv("TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD", raising=False)
    config = M.configuration()
    assert config["clip"]["selection_resolved"] is True
    assert config["clip"]["inventory"] == []
    assert config["clip"]["inventory_root"] == str(cache.absolute())
    assert M._unavailable("clip", config) in {
        "model_runtime_unavailable", "model_weights_unavailable"}


def test_unrelated_clip_cache_change_does_not_change_frozen_default_identity(
        tmp_path, monkeypatch):
    cache = _caches(tmp_path, monkeypatch)
    selected = cache / (
        "models--timm--vit_base_patch32_clip_224.openai/snapshots/rev/weights.bin")
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"selected")
    monkeypatch.delenv("TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD", raising=False)
    before = M.configuration()
    unrelated = cache / "models--other--model/snapshots/rev/weights.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"unrelated")
    assert M.configuration() == before
    selected.write_bytes(b"changed!")
    assert M.configuration() != before


def test_unknown_named_clip_architecture_is_fail_closed(tmp_path, monkeypatch):
    cache = _caches(tmp_path, monkeypatch)
    unrelated = cache / (
        "models--timm--vit_base_patch32_clip_224.openai/snapshots/rev/weights.bin")
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"default model")
    config = M.configuration(clip_model="not-registered")
    assert config["clip"]["selection_resolved"] is False
    assert config["clip"]["inventory"] == []
    assert M._unavailable("clip", config) in {
        "model_runtime_unavailable", "model_selection_unresolved"}


def test_clip_selection_contract_rejects_malformed_resolution_flag(tmp_path, monkeypatch):
    _caches(tmp_path, monkeypatch)
    config = M.configuration()
    invalid = copy.deepcopy(config)
    invalid["clip"]["selection_resolved"] = "yes"
    with pytest.raises(WorkspaceError) as failure:
        M.validate_configuration(invalid)
    assert getattr(failure.value, "code", None) == "SCHEMA_INVALID"


def test_legacy_clip_selection_is_readable_but_cannot_infer_current_weights(
        tmp_path, monkeypatch):
    _caches(tmp_path, monkeypatch)
    config = M.configuration()
    legacy = copy.deepcopy(config)
    legacy["policy"].pop("host_lock")
    legacy["clip"] = {name: legacy["clip"][name]
                      for name in ("requested", "cache", "inventory")}
    checked = M.validate_configuration(legacy)
    assert checked is not legacy
    assert checked["policy"]["host_lock"] == R.policy()["host_lock"]
    assert checked["clip"]["inventory_root"] == checked["clip"]["cache"]
    assert checked["clip"]["selection_resolved"] is False
    assert M._unavailable("clip", legacy) in {
        "model_runtime_unavailable", "model_selection_unresolved"}
    result = M.run_phase("clip", legacy, [_frame()], tmp_path / "legacy-work",
                         cancelled=lambda: False)
    assert result["execution"]["status"] == "not_run"
    assert result["execution"]["reason"] in {
        "model_runtime_unavailable", "model_selection_unresolved"}
