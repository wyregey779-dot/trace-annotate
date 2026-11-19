"""YuNet Unicode loading must use the supported buffer API before file fallback."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trace_annotate.extractors import faces as F


def install_create(monkeypatch, create):
    monkeypatch.setattr(F, "HAS_CV2", True)
    monkeypatch.setattr(F, "cv2", SimpleNamespace(FaceDetectorYN=SimpleNamespace(create=create)), raising=False)


def test_unicode_model_uses_complete_buffer_overload_without_writing_files(monkeypatch, tmp_path):
    model = tmp_path / "中文模型.onnx"
    model.write_bytes(b"controlled model bytes")
    calls = []
    detector = object()
    def create(*args, **kwargs):
        assert len(args) == 3 and args[0] == "ONNX"
        assert args[1].dtype == np.uint8 and args[1].tobytes() == model.read_bytes()
        assert args[2].dtype == np.uint8 and args[2].size == 0
        assert kwargs["input_size"] == (320, 320)
        calls.append((args, kwargs))
        return detector
    install_create(monkeypatch, create)
    def forbidden(*args, **kwargs):
        pytest.fail("A supported memory loader must not copy weights or create cache directories")
    monkeypatch.setattr(F.os, "makedirs", forbidden)
    monkeypatch.setattr("shutil.copyfile", forbidden)
    extractor = F.FaceExtractor(model_path=str(model))
    assert extractor.detector is detector and extractor.detector_kind == "yunet"
    assert extractor.code_identity["weights_sha256"] == F.hashlib.sha256(
        model.read_bytes()).hexdigest()
    assert extractor.code_identity["weights_load_source"] == "memory_buffer"
    assert len(calls) == 1


def test_ascii_model_uses_the_same_frozen_bytes_for_hash_and_loading(monkeypatch):
    import builtins
    import io
    model = "C:/models/yunet.onnx"
    detector = object()
    monkeypatch.setattr(F.os.path, "exists", lambda path: path == model)
    original_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda path, *args, **kwargs:
                        io.BytesIO(b"controlled model") if path == model else original_open(path, *args, **kwargs))
    def create(*args, **kwargs):
        assert len(args) == 3 and args[0] == "ONNX"
        assert args[1].tobytes() == b"controlled model"
        return detector
    install_create(monkeypatch, create)
    extractor = F.FaceExtractor(model_path=model)
    assert extractor.detector is detector
    assert extractor.code_identity["weights_sha256"] == F.hashlib.sha256(
        b"controlled model").hexdigest()


def test_source_replacement_after_load_cannot_rewrite_loaded_identity(monkeypatch, tmp_path):
    model = tmp_path / "yunet.onnx"
    original = b"A" * 64
    replacement = b"B" * 64
    model.write_bytes(original)
    original_stat = model.stat()
    loaded = []
    detector = object()

    def create(*args, **kwargs):
        loaded.append(args[1].tobytes())
        return detector

    install_create(monkeypatch, create)
    extractor = F.FaceExtractor(model_path=str(model))
    model.write_bytes(replacement)
    F.os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert loaded == [original]
    assert extractor.code_identity["weights_sha256"] == F.hashlib.sha256(original).hexdigest()
    assert F.hashlib.sha256(model.read_bytes()).hexdigest() != (
        extractor.code_identity["weights_sha256"])


@pytest.mark.parametrize("cache_loads", [True, False])
@pytest.mark.parametrize("explicit_cache", [True, False])
def test_legacy_unicode_fallback_honors_the_explicit_cache(monkeypatch, tmp_path, cache_loads, explicit_cache):
    model = tmp_path / "中文源模型.onnx"
    model.write_bytes(b"legacy controlled model bytes")
    if explicit_cache:
        cache = tmp_path / "owned-cache"
        monkeypatch.setenv("TRACE_ANNOTATE_FACE_CACHE", str(cache))
    else:
        monkeypatch.delenv("TRACE_ANNOTATE_FACE_CACHE", raising=False)
        legacy_root = tmp_path / "controlled-legacy-root"
        cache = legacy_root / "traceannotate" / "yunet"
        monkeypatch.setattr(F, "_ascii_temp_root", lambda: str(legacy_root))
    detector = object()
    calls = []
    def create(*args, **kwargs):
        if args:
            raise TypeError("Controlled old OpenCV: buffer overload unavailable")
        path = Path(kwargs["model"])
        assert path.parent == cache and path.read_bytes() == model.read_bytes()
        calls.append(path)
        if not cache_loads:
            raise RuntimeError("Controlled old OpenCV: explicit cache path unavailable")
        return detector
    install_create(monkeypatch, create)
    def forbidden():
        pytest.fail("An explicit cache must never fall back to a global temporary directory")
    if explicit_cache:
        monkeypatch.setattr(F, "_ascii_temp_root", forbidden)
    extractor = F.FaceExtractor(model_path=str(model))
    assert len(calls) == 1
    assert extractor.detector is (detector if cache_loads else None)


@pytest.mark.parametrize("explicit,weights,expected", [
    (None, None, "environment.onnx"), ("explicit.onnx", None, "explicit.onnx"),
    (None, "weights.onnx", "weights.onnx"), ("haar", None, "haar"),
])
def test_normal_default_reads_face_model_environment_without_overriding_explicit_arguments(
        monkeypatch, explicit, weights, expected):
    monkeypatch.setenv("TRACE_ANNOTATE_FACE_MODEL", "environment.onnx")
    monkeypatch.setattr(F.FaceExtractor, "_init_detector", lambda self: None)
    extractor = F.FaceExtractor(model_path=explicit, weights_path=weights)
    assert extractor.model_path == expected
