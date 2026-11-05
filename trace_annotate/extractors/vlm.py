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
        self._init_model()

    @property
    def name(self) -> str:
        return "vlm"

    def _init_model(self) -> None:
        """Load the VLM once; every failure degrades to not_run."""
        self.model = None
        self.processor = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:  # noqa: BLE001 - missing dependency disables the adapter
            self.load_error = f"transformers/torch unavailable: {exc}"
            return
        try:
            os.makedirs(_cache_dir(), exist_ok=True)
            enforce_offline_environment()
            # Florence-2 ships its processor/model code in the repository, so
            # trust_remote_code=True is the vendor-documented requirement. The
            # local_files_only flag keeps a missing weight from reaching the hub.
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, cache_dir=_cache_dir(), trust_remote_code=True,
                local_files_only=self.load_policy["offline"],
                revision=self.revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, cache_dir=_cache_dir(), trust_remote_code=True,
                local_files_only=self.load_policy["offline"],
                revision=self.revision)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001 - download/load failure disables inference
            self.model = None
            self.processor = None
            self.load_error = str(exc)

    @property
    def code_identity(self) -> Dict[str, Any]:
        """What code was actually executed, for the model-run record."""
        return {"model_path": self.model_path,
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
