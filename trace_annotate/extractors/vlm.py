"""
Local VLM caption adapter with an explicit execution-status contract.

The default backend is Microsoft Florence-2 (MIT license) through the
transformers library, which runs on CPU at a usable speed. The adapter only
produces a caption *draft*: callers must keep it out of the authoritative
annotation set until a human reviews it, because any VLM output can hallucinate.
A missing model reports not_run, a decode or runtime crash reports failed, and
only a real forward pass reports success.
"""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from trace_annotate.extractors.base import BaseExtractor, ExtractorStatus
from trace_annotate.extractors.model_loading import (enforce_offline_environment,
                                                    load_policy)
from trace_annotate.core.hasher import safe_load_image


DEFAULT_MODEL_NAME = "microsoft/Florence-2-base"
# The upstream repository ships processor/model code, so a floating `main`
# revision would let the executed code change without notice. Operators pin a
# concrete commit/tag through TRACE_ANNOTATE_VLM_REVISION; when unset the adapter
# reports "unpinned" in its code identity instead of pretending to be frozen.
REVISION_ENV = "TRACE_ANNOTATE_VLM_REVISION"
# The task prompt requests a dense, literal caption (as opposed to a single
# sentence or a decorative description), which is what an annotation draft needs.
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
DEFAULT_WEIGHTS_SHA256 = "03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060"
DEFAULT_WEIGHTS_LICENSE_URL = (
    "https://huggingface.co/microsoft/Florence-2-base/blob/"
    "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/README.md")
DEFAULT_MODEL_CODE_SHA256 = {
    "configuration_florence2.py": "de2e45a975b3582de05d2f4d963a3e9f9a3d20dccf78d28e0052932a0be93bdf",
    "modeling_florence2.py": "5162bf465e61b6e29cc113a467630ec3cb56ed8e4d46eb6207157f10fb9b8a24",
    "processing_florence2.py": "f146023a507c009f425a49ee39aa037f4f25c64e14336e3e4f3f1d7377a68e98",
}


def _cache_dir() -> str:
    override = os.environ.get("TRACE_ANNOTATE_VLM_CACHE")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".cache", "traceannotate", "vlm")


class VLMExtractor(BaseExtractor):
    """Local VLM captioner; missing weights degrade to a disabled adapter."""

    def __init__(self, model_path: Optional[str] = None,
                 task_prompt: Optional[str] = None,
                 revision: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_NAME
        self.task_prompt = task_prompt or CAPTION_TASK
        self.revision = revision or os.environ.get(REVISION_ENV) or None
        self.model = None
        self.processor = None
        self.load_error: Optional[str] = None
        self.load_policy = load_policy()
        self._weights_path: Optional[str] = None
        self._weights_sha256: Optional[str] = None
        self._selected_revision: Optional[str] = None
        self._model_code_sha256: Dict[str, str] = {}
        self._init_model()

    @property
    def name(self) -> str:
        return "vlm"

    def _init_model(self) -> None:
        """Load the VLM once; every failure degrades to not_run."""
        self.model = None
        self.processor = None
        self._weights_path = None
        self._weights_sha256 = None
        self._selected_revision = None
        self._model_code_sha256 = {}
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:  # noqa: BLE001 - missing dependency disables the adapter
            self.load_error = f"transformers/torch unavailable: {exc}"
            return
        try:
            os.makedirs(_cache_dir(), exist_ok=True)
            enforce_offline_environment()
            selected = None
            if os.path.isdir(self.model_path):
                local_weight = os.path.join(self.model_path, "model.safetensors")
                if os.path.isfile(local_weight):
                    selected = local_weight
            else:
                from transformers.utils.hub import cached_file
                try:
                    selected = cached_file(
                        self.model_path, "model.safetensors", cache_dir=_cache_dir(),
                        local_files_only=self.load_policy["offline"], revision=self.revision,
                        _raise_exceptions_for_missing_entries=False)
                except (OSError, ValueError):
                    # Another supported checkpoint layout may still load, but
                    # its weights must not inherit an unrelated cache identity.
                    selected = None
            load_path = self.model_path
            model_options = {}
            if selected and os.path.isfile(selected):
                path = Path(os.path.abspath(selected))
                self._weights_path = str(path)
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                self._weights_sha256 = hasher.hexdigest()
                if path.parent.parent.name == "snapshots":
                    self._selected_revision = path.parent.name
                load_path = str(path.parent)
                model_options["use_safetensors"] = True
                for filename in DEFAULT_MODEL_CODE_SHA256:
                    code_path = path.parent / filename
                    if code_path.is_file():
                        self._model_code_sha256[filename] = hashlib.sha256(code_path.read_bytes()).hexdigest()
            # Florence-2 ships its processor/model code in the repository, so
            # trust_remote_code=True is the vendor-documented requirement. The
            # local_files_only flag keeps a missing weight from reaching the hub.
            self.processor = AutoProcessor.from_pretrained(
                load_path, cache_dir=_cache_dir(), trust_remote_code=True,
                local_files_only=self.load_policy["offline"],
                revision=self.revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path, cache_dir=_cache_dir(), trust_remote_code=True,
                local_files_only=self.load_policy["offline"],
                revision=self.revision, **model_options)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001 - download/load failure disables inference
            self.model = None
            self.processor = None
            self.load_error = str(exc)

    @property
    def code_identity(self) -> Dict[str, Any]:
        """What code was actually executed, for the model-run record."""
        known_weights = self._weights_sha256 == DEFAULT_WEIGHTS_SHA256
        known_code = self._model_code_sha256 == DEFAULT_MODEL_CODE_SHA256
        return {"model_path": self.model_path,
                "weights_path": self._weights_path,
                "weights_sha256": self._weights_sha256,
                "selected_revision": self._selected_revision,
                "weights_identity_status": (
                    "unknown" if self._weights_path is None else
                    "loaded" if self.model is not None else "selected"),
                "runtime_license": "Apache-2.0 (Transformers)",
                "weights_license": "MIT" if known_weights else None,
                "weights_license_source": DEFAULT_WEIGHTS_LICENSE_URL if known_weights else None,
                "model_code_sha256": dict(self._model_code_sha256),
                "model_code_license": "Apache-2.0" if known_code else None,
                "task_prompt": self.task_prompt,
                "trust_remote_code": True,
                "revision": self.revision or "unpinned",
                "revision_env": REVISION_ENV,
                "offline": self.load_policy["offline"],
                "downloads_allowed": self.load_policy["downloads_allowed"]}

    def caption(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Produce one caption draft for a BGR frame; honest status always."""
        if self.model is None or self.processor is None:
            return {"status": ExtractorStatus.NOT_RUN.value,
                    "reason": "vlm_weights_unavailable", "caption": ""}
        try:
            import torch
            import PIL.Image as Image
            if img_bgr is None:
                raise ValueError("frame is empty")
            image = Image.fromarray(img_bgr[..., ::-1])  # BGR -> RGB
            inputs = self.processor(text=self.task_prompt, images=image,
                                    return_tensors="pt")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=192, num_beams=3, do_sample=False)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False)[0]
            caption = self.processor.post_process_generation(
                generated_text, task=self.task_prompt, image_size=image.size)
            text = caption.get(self.task_prompt, "").strip()
            if not text:
                return {"status": ExtractorStatus.FAILED.value,
                        "error": "vlm returned an empty caption", "caption": ""}
            return {"status": ExtractorStatus.SUCCESS.value,
                    "caption": text, "language": "auto"}
        except Exception as exc:  # noqa: BLE001 - runtime crash reports failed
            return {"status": ExtractorStatus.FAILED.value,
                    "error": str(exc), "caption": ""}

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        img = safe_load_image(staged_path)
        if img is None:
            return {"status": ExtractorStatus.FAILED.value, "caption": ""}
        return self.caption(img)


# Alias for backward compatibility
VLMExtractorEngine = VLMExtractor
