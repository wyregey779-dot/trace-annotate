"""Selected model identity must describe the file supplied to the loader."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from trace_annotate.extractors.clip_engine import CLIPExtractor
from trace_annotate.extractors.vlm import DEFAULT_MODEL_NAME, VLMExtractor


def _weight(directory: Path, filename: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(content)
    return path


def _clip_stub(monkeypatch, selected: Path | None, *, fail=False):
    calls = []
    package = ModuleType("open_clip")
    package.get_pretrained_cfg = lambda model, tag: {
        "hf_hub": "timm/vit_base_patch32_clip_224.openai/",
        "mean": (0.1, 0.2, 0.3), "std": (0.4, 0.5, 0.6),
    } if model == "ViT-B-32" else {}

    def resolve(config, **kwargs):
        if selected is None:
            raise FileNotFoundError("No cached checkpoint")
        return str(selected)

    def create(model_name, **kwargs):
        calls.append((model_name, kwargs))
        if fail:
            raise ValueError("Model cannot be loaded")
        return SimpleNamespace(eval=lambda: None), None, object()

    package.download_pretrained = resolve
    def load_openai(path, **kwargs):
        calls.append(("openai-local", {"path": path, **kwargs}))
        return SimpleNamespace(eval=lambda: None, visual=SimpleNamespace(
            image_size=224, image_mean=(0.1, 0.2, 0.3), image_std=(0.4, 0.5, 0.6)))
    package.load_openai_model = load_openai
    package.pretrained = SimpleNamespace(_has_safetensors=True,
                                         HF_WEIGHTS_NAME="open_clip_pytorch_model.bin",
                                         HF_SAFE_WEIGHTS_NAME="open_clip_model.safetensors")
    package.create_model_and_transforms = create
    package.get_tokenizer = lambda model_name: object()
    monkeypatch.setitem(sys.modules, "open_clip", package)
    transform_module = ModuleType("open_clip.transform")
    transform_module.image_transform = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "open_clip.transform", transform_module)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    torchvision = ModuleType("torchvision")
    transforms = ModuleType("torchvision.transforms")
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda *args, **kwargs: str(selected) if selected is not None else None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(CLIPExtractor, "_precompute_text_features", lambda self: None)
    return calls


def _vlm_stub(monkeypatch, selected: Path | None, *, fail=False):
    calls = []
    transformers = ModuleType("transformers")
    utils = ModuleType("transformers.utils")
    hub = ModuleType("transformers.utils.hub")

    def cached_file(model_path, filename, **kwargs):
        calls.append(("resolve", model_path, filename, kwargs))
        return str(selected) if selected is not None else None

    def load_processor(path, **kwargs):
        calls.append(("processor", path, kwargs))
        if fail:
            raise ValueError("Model cannot be loaded")
        return object()

    def load_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return SimpleNamespace(eval=lambda: None)

    hub.cached_file = cached_file
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=load_processor)
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub)
    return calls


def test_clip_default_loads_resolved_snapshot_instead_of_sorted_first(tmp_path, monkeypatch):
    cache = tmp_path / "clip"
    repo = cache / "models--timm--vit_base_patch32_clip_224.openai"
    _weight(repo / "snapshots/aaa", "open_clip_model.safetensors", b"old checkpoint")
    selected = _weight(repo / "snapshots/zzz", "open_clip_model.safetensors", b"selected checkpoint")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(cache))
    calls = _clip_stub(monkeypatch, selected)
    extractor = CLIPExtractor()
    assert extractor.model is not None
    identity = extractor.code_identity
    assert identity["weights_path"] == str(selected)
    assert identity["selected_revision"] == "zzz"
    assert identity["weights_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert identity["weights_identity_status"] == "loaded"
    assert calls[0][0] == "ViT-B-32"
    assert calls[0][1].get("pretrained", calls[0][1].get("path")) == str(selected)
    assert calls[0][1]["force_quick_gelu"] is True
    assert calls[0][1]["image_mean"] == (0.1, 0.2, 0.3)
    assert calls[0][1]["image_std"] == (0.4, 0.5, 0.6)
    assert identity["weights_license"] is None


def test_clip_custom_architecture_never_claims_default_cache(tmp_path, monkeypatch):
    cache = tmp_path / "clip"
    selected = _weight(cache / "models--timm--vit_base_patch32_clip_224.openai/snapshots/aaa", "open_clip_model.safetensors", b"unrelated default")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(cache))
    _clip_stub(monkeypatch, selected)
    extractor = CLIPExtractor(model_name_or_path="custom-architecture")
    assert extractor.model is None
    assert extractor.classify(None)["status"] == "not_run"
    assert extractor.code_identity["weights_path"] is None
    assert extractor.code_identity["weights_identity_status"] == "unknown"
    assert extractor.code_identity["weights_license"] is None


def test_clip_missing_selected_weight_reports_not_run_and_unknown_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    _clip_stub(monkeypatch, None, fail=True)
    extractor = CLIPExtractor()
    assert extractor.model is None
    assert extractor.classify(None)["status"] == "not_run"
    assert extractor.code_identity["weights_path"] is None
    assert extractor.code_identity["weights_identity_status"] == "unknown"


def test_vlm_explicit_directory_uses_its_weight_not_another_cached_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "vlm"
    _weight(cache / "models--microsoft--Florence-2-base/snapshots/aaa", "model.safetensors", b"unrelated cache")
    selected = _weight(tmp_path / "explicit-model", "model.safetensors", b"custom local checkpoint")
    monkeypatch.setenv("TRACE_ANNOTATE_VLM_CACHE", str(cache))
    calls = _vlm_stub(monkeypatch, selected)
    extractor = VLMExtractor(model_path=str(selected.parent), revision="requested-revision")
    assert extractor.model is not None
    identity = extractor.code_identity
    assert identity["weights_path"] == str(selected)
    assert identity["weights_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert identity["selected_revision"] is None
    assert identity["revision"] == "requested-revision"
    assert identity["weights_license"] is None
    assert [call[1] for call in calls if call[0] in {"processor", "model"}] == [str(selected.parent)] * 2
    assert next(call[2] for call in calls if call[0] == "model")["use_safetensors"] is True


def test_vlm_pinned_hub_revision_loads_the_resolved_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "vlm"
    repo = cache / "models--microsoft--Florence-2-base"
    _weight(repo / "snapshots/aaa", "model.safetensors", b"old checkpoint")
    selected = _weight(repo / "snapshots/zzz", "model.safetensors", b"pinned checkpoint")
    monkeypatch.setenv("TRACE_ANNOTATE_VLM_CACHE", str(cache))
    calls = _vlm_stub(monkeypatch, selected)
    extractor = VLMExtractor(revision="zzz")
    identity = extractor.code_identity
    assert identity["weights_path"] == str(selected)
    assert identity["selected_revision"] == "zzz"
    resolution = next(call for call in calls if call[0] == "resolve")
    assert resolution[1] == DEFAULT_MODEL_NAME
    assert resolution[3]["revision"] == "zzz"
    assert resolution[3]["local_files_only"] is True
    assert [call[1] for call in calls if call[0] in {"processor", "model"}] == [str(selected.parent)] * 2


def test_vlm_missing_custom_model_does_not_claim_unrelated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "vlm"
    _weight(cache / "models--microsoft--Florence-2-base/snapshots/aaa", "model.safetensors", b"unrelated checkpoint")
    monkeypatch.setenv("TRACE_ANNOTATE_VLM_CACHE", str(cache))
    _vlm_stub(monkeypatch, None, fail=True)
    extractor = VLMExtractor(model_path="not-the-default-model")
    assert extractor.model is None
    assert extractor.caption(None)["status"] == "not_run"
    assert extractor.code_identity["weights_path"] is None
    assert extractor.code_identity["weights_license"] is None


def test_vlm_local_snapshot_reports_local_revision_separately_from_requested(tmp_path, monkeypatch):
    selected = _weight(tmp_path / "cache/snapshots/local-revision", "model.safetensors", b"local")
    _vlm_stub(monkeypatch, selected)
    extractor = VLMExtractor(model_path=str(selected.parent), revision="other-request")
    assert extractor.code_identity["selected_revision"] == "local-revision"
    assert extractor.code_identity["revision"] == "other-request"


def _network_guard(monkeypatch):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("Offline adapter reached a network-capable resolver")

    monkeypatch.setattr(sys.modules["open_clip"], "download_pretrained", blocked)
    sys.modules["huggingface_hub"].hf_hub_download = blocked
    sys.modules["huggingface_hub"].constants = SimpleNamespace(HF_HUB_OFFLINE=False)
    return attempts


def test_offline_hf_cache_does_not_trust_preimported_online_constant(tmp_path, monkeypatch):
    selected = _weight(tmp_path / "snapshots/current", "open_clip_model.safetensors", b"cached")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    calls = _clip_stub(monkeypatch, selected)
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor()
    assert extractor.model is not None
    assert network == []
    assert calls[0][1]["pretrained"] == str(selected)


@pytest.mark.parametrize("cached_content", [None, b"corrupt-cache"])
def test_offline_url_miss_or_bad_sha_never_downloads(tmp_path, monkeypatch, cached_content):
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    selected = tmp_path / "ViT-B-32.pt"
    if cached_content is not None:
        selected.write_bytes(cached_content)
    calls = _clip_stub(monkeypatch, selected)
    expected = hashlib.sha256(b"expected checkpoint").hexdigest()
    sys.modules["open_clip"].get_pretrained_cfg = lambda *args: {
        "url": f"https://openaipublic.azureedge.net/clip/models/{expected}/ViT-B-32.pt"}
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor()
    assert extractor.model is None
    assert extractor.classify(None)["status"] == "not_run"
    assert network == []
    assert calls == []
    assert extractor.code_identity["weights_path"] is None


def test_offline_url_cache_hit_checks_sha_and_passes_exact_local_file(tmp_path, monkeypatch):
    selected = _weight(tmp_path, "ViT-B-32.pt", b"valid checkpoint")
    expected = hashlib.sha256(selected.read_bytes()).hexdigest()
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    calls = _clip_stub(monkeypatch, selected)
    sys.modules["open_clip"].get_pretrained_cfg = lambda *args: {
        "url": f"https://openaipublic.azureedge.net/clip/models/{expected}/ViT-B-32.pt"}
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor()
    assert extractor.model is not None
    assert network == []
    assert calls[0][1].get("pretrained", calls[0][1].get("path")) == str(selected)
    assert extractor.code_identity["weights_sha256"] == expected


def test_offline_hf_cache_uses_public_refs_resolution_not_snapshot_sort(tmp_path, monkeypatch):
    public_cache_lookup = pytest.importorskip("huggingface_hub").try_to_load_from_cache
    cache = tmp_path / "clip"
    repo = cache / "models--timm--vit_base_patch32_clip_224.openai"
    _weight(repo / "snapshots/aaa", "open_clip_model.safetensors", b"old checkpoint")
    selected = _weight(repo / "snapshots/zzz", "open_clip_model.safetensors", b"current checkpoint")
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text("zzz", encoding="utf-8")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(cache))
    _clip_stub(monkeypatch, selected)
    sys.modules["huggingface_hub"].try_to_load_from_cache = public_cache_lookup
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor()
    assert extractor.model is not None
    assert extractor.code_identity["weights_path"] == str(selected)
    assert extractor.code_identity["selected_revision"] == "zzz"
    assert network == []


def test_explicit_download_optin_preserves_official_resolver(tmp_path, monkeypatch):
    selected = _weight(tmp_path, "downloaded.safetensors", b"opt-in checkpoint")
    monkeypatch.setenv("TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD", "1")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    calls = _clip_stub(monkeypatch, selected)
    resolutions = []
    sys.modules["open_clip"].download_pretrained = lambda *args, **kwargs: resolutions.append(kwargs) or str(selected)
    extractor = CLIPExtractor()
    assert extractor.model is not None
    assert len(resolutions) == 1
    assert calls[0][1]["pretrained"] == str(selected)


def test_offline_unknown_architecture_never_enters_factory(tmp_path, monkeypatch):
    selected = _weight(tmp_path, "unrelated.safetensors", b"unrelated")
    calls = _clip_stub(monkeypatch, selected)
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor(model_name_or_path="hf-hub:custom/opaque")
    assert extractor.model is None
    assert calls == [] and network == []
    assert extractor.code_identity["weights_identity_status"] == "unknown"


def test_offline_legacy_openai_jit_uses_public_local_path_loader(tmp_path, monkeypatch):
    selected = _weight(tmp_path, "ViT-B-32.pt", b"valid legacy checkpoint")
    expected = hashlib.sha256(selected.read_bytes()).hexdigest()
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    calls = _clip_stub(monkeypatch, selected)
    sys.modules["open_clip"].get_pretrained_cfg = lambda *args: {
        "url": f"https://openaipublic.azureedge.net/clip/models/{expected}/ViT-B-32.pt"}
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor()
    assert extractor.model is not None
    assert calls[0][0] == "openai-local"
    assert calls[0][1] == {"path": str(selected), "device": "cpu", "precision": "fp32"}
    assert network == []


@pytest.mark.parametrize("architecture,repository", [
    ("ViT-L-14", "timm/vit_large_patch14_clip_224.openai/"),
    ("RN50", "timm/resnet50_clip.openai/"),
])
def test_offline_named_openai_architecture_keeps_its_own_cached_checkpoint(tmp_path, monkeypatch, architecture, repository):
    selected = _weight(tmp_path / "models--selected/snapshots/current", "open_clip_model.safetensors", b"selected named checkpoint")
    _weight(tmp_path / "models--timm--vit_base_patch32_clip_224.openai/snapshots/aaa", "open_clip_model.safetensors", b"unrelated B32")
    monkeypatch.setenv("TRACE_ANNOTATE_CLIP_CACHE", str(tmp_path))
    calls = _clip_stub(monkeypatch, selected)
    lookups = []
    sys.modules["open_clip"].get_pretrained_cfg = lambda model, tag: {
        "hf_hub": repository, "mean": (0.2, 0.3, 0.4), "std": (0.5, 0.6, 0.7)} if model == architecture else {}
    sys.modules["huggingface_hub"].try_to_load_from_cache = lambda repo, filename, **kwargs: lookups.append((repo, filename, kwargs)) or str(selected)
    network = _network_guard(monkeypatch)
    extractor = CLIPExtractor(model_name_or_path=architecture)
    assert extractor.model is not None
    assert calls[0][0] == architecture
    assert calls[0][1]["pretrained"] == str(selected)
    assert calls[0][1]["image_mean"] == (0.2, 0.3, 0.4)
    assert calls[0][1]["image_std"] == (0.5, 0.6, 0.7)
    assert lookups[0][0] == repository.rstrip("/")
    assert extractor.code_identity["weights_path"] == str(selected)
    assert extractor.code_identity["weights_license"] is None
    assert network == []
