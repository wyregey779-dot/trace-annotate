"""
CLIP zero-shot classifier with an explicit execution-status contract.

Real inference is attempted through open_clip (MIT license). When the model
weights, torch or open_clip cannot be loaded the adapter reports not_run and
never invents labels. The honest boundary is strict: a missing model reports
not_run, a decode or runtime crash reports failed, and only a real forward pass
reports success.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from trace_annotate.extractors.base import BaseExtractor, ExtractorStatus
from trace_annotate.core.hasher import safe_load_image
from trace_annotate.config import MetricThresholds


# Prompt text is English on purpose: CLIP's BPE tokenizer was trained on
# English, so Chinese prompt strings encode as near-noise. The label stays
# Chinese (the user-facing tag); the description drives the actual embedding.
# Each option is [label, text] or [label, [text1, text2, ...]] where a list is
# an ensemble averaged before the softmax, which raises zero-shot confidence.
DEFAULT_PROMPTS = {
    "io": [["室内", ["a living room with furniture inside a house",
                    "an office room with desks and chairs",
                    "a kitchen inside a building"]],
           ["室外", ["an outdoor street with buildings and sky",
                    "a beach with sand and ocean",
                    "a mountain landscape with trees outside"]]],
    "tod": [["白天", ["a scene in bright daylight with sun",
                     "a well lit daytime outdoor scene"]],
            ["夜", ["a scene at night with dark sky and artificial lights",
                    "a night time city street with neon lights"]]],
    "weather": [
        ["雨", ["rainy weather with wet streets and umbrellas",
                "heavy rain falling outside"]],
        ["雪", ["snowy weather with snow covered ground",
                "snow falling in winter"]],
        ["雾", ["foggy weather with low visibility and mist",
                "dense fog outdoors"]],
        [None, ["clear sunny weather, blue sky no rain snow or fog"]]
    ],
    "screen": [
        ["无屏", ["a plain scene with no screen no display no glass",
                  "a scene without any monitor or screen"]],
        ["户外大屏", ["a large outdoor LED billboard screen showing content",
                      "an outdoor advertising display screen"]],
        ["橱窗", ["a shop window display with glass showing products",
                  "a storefront glass showcase"]],
        ["店招", ["a store sign with glowing neon characters",
                  "a shop entrance with a lit signboard"]]
    ]
}

# Default open_clip model and pretrained tag; download happens lazily on first
# construction and is cached under the user cache directory.
DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"


def _cache_dir() -> str:
    override = os.environ.get("TRACE_ANNOTATE_CLIP_CACHE")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".cache", "traceannotate", "clip")


class CLIPExtractor(BaseExtractor):
    """Zero-shot CLIP classifier with lazy weight loading and honest status."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_name: Optional[str] = None,
        prompts: Optional[Dict[str, List[List[Any]]]] = None,
        entity_prompts: Optional[Dict[str, str]] = None,
        thresholds: Optional[MetricThresholds] = None
    ):
        self.model_path = model_name_or_path or model_name
        self.prompts = prompts or DEFAULT_PROMPTS
        self.entity_prompts = entity_prompts or {}
        self.thresholds = thresholds or MetricThresholds()
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._text_features: Optional[np.ndarray] = None
        self._prompt_layout: List[Tuple[str, List[Any]]] = []
        self.load_error: Optional[str] = None
        self._init_model()

    @property
    def name(self) -> str:
        return "clip"

    def _init_model(self) -> None:
        """Load weights lazily; every failure degrades to a disabled adapter."""
        self.model = None
        self._text_features = None
        self._prompt_layout = []
        try:
            import open_clip  # MIT license
            import torch
            import torchvision.transforms as T  # noqa: F401  (ensures torchvision present)
        except Exception as exc:  # noqa: BLE001 - every missing dependency is a disabled adapter
            self.load_error = f"open_clip/torch unavailable: {exc}"
            return
        try:
            model_name = self.model_path or DEFAULT_MODEL_NAME
            os.makedirs(_cache_dir(), exist_ok=True)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=DEFAULT_PRETRAINED, cache_dir=_cache_dir(),
                force_quick_gelu=True)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self._precompute_text_features()
        except Exception as exc:  # noqa: BLE001 - download/load failure disables inference
            self.model = None
            self.load_error = str(exc)

    def _precompute_text_features(self) -> None:
        """Encode all prompt texts once; layout keeps the ordered label list.

        A prompt option may carry one text or an ensemble list. An ensemble is
        averaged (L2-normalized per member, mean, L2-normalized again), which
        is the standard CLIP prompt-ensemble recipe and raises zero-shot
        confidence versus a single abstract phrase.
        """
        import torch
        layout: List[Tuple[str, List[Any]]] = []
        all_texts: List[str] = []
        for category, options in self.prompts.items():
            ordered = []
            for option in options:
                ordered.append(option)
                text = option[1]
                if isinstance(text, str):
                    all_texts.append(text)
                else:
                    all_texts.extend(text)
            layout.append((category, ordered))
        self._prompt_layout = layout
        with torch.no_grad():
            tokens = self.tokenizer(all_texts)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()
        # Collapse each option to one averaged vector in layout order.
        collapsed: List[np.ndarray] = []
        cursor = 0
        for category, ordered in layout:
            for option in ordered:
                text = option[1]
                width = 1 if isinstance(text, str) else len(text)
                block = features[cursor:cursor + width]
                cursor += width
                mean = block.mean(axis=0)
                norm = np.linalg.norm(mean)
                collapsed.append(mean / norm if norm > 0 else mean)
        self._text_features = np.stack(collapsed)

    def classify(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Zero-shot classify one BGR frame; returns honest status always."""
        if self.model is None or self._text_features is None:
            return {
                "status": ExtractorStatus.NOT_RUN.value,
                "clip_scores": {},
                "indoor_outdoor": None,
                "time_of_day": None,
                "weather": None,
                "screen_type": None,
                "top_entity": None,
                "entity_score": None,
                "entity_mismatch": False
            }
        try:
            import torch
            if img_bgr is None:
                raise ValueError("frame is empty")
            rgb = img_bgr[..., ::-1]  # BGR -> RGB
            import PIL.Image as Image
            image = self.preprocess(Image.fromarray(rgb)).unsqueeze(0)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp().cpu().numpy()
                logits = logit_scale * (image_features.cpu().numpy() @ self._text_features.T)
            logits_vec = logits[0]
            scores: Dict[str, Dict[str, float]] = {}
            categories: Dict[str, Any] = {"indoor_outdoor": None, "time_of_day": None,
                                          "weather": None, "screen_type": None}
            offset = 0
            binary_cutoff = self.thresholds.clip_binary_confidence
            multiclass_cutoff = self.thresholds.clip_multiclass_confidence
            binary_categories = {"io": "indoor_outdoor", "tod": "time_of_day"}
            for category, ordered in self._prompt_layout:
                width = len(ordered)
                block = logits_vec[offset:offset + width]
                offset += width
                probs = _softmax(block)
                labels = [option[0] for option in ordered]
                scores[category] = {str(label): float(score) for label, score in zip(labels, probs)}
                best_index = int(np.argmax(probs))
                best_score = float(probs[best_index])
                best_label = labels[best_index]
                if category in binary_categories and best_score >= binary_cutoff:
                    categories[binary_categories[category]] = best_label
                elif category in {"weather", "screen"} and best_score >= multiclass_cutoff:
                    categories[{"weather": "weather", "screen": "screen_type"}[category]] = best_label
            top_entity, entity_score, entity_mismatch = self._classify_entity(image_features.cpu().numpy()[0])
            return {
                "status": ExtractorStatus.SUCCESS.value,
                "clip_scores": scores,
                "indoor_outdoor": categories["indoor_outdoor"],
                "time_of_day": categories["time_of_day"],
                "weather": categories["weather"],
                "screen_type": categories["screen_type"],
                "top_entity": top_entity,
                "entity_score": entity_score,
                "entity_mismatch": entity_mismatch
            }
        except Exception as exc:  # noqa: BLE001 - runtime crash reports failed
            return {
                "status": ExtractorStatus.FAILED.value,
                "error": str(exc),
                "clip_scores": {},
                "indoor_outdoor": None,
                "time_of_day": None,
                "weather": None,
                "screen_type": None,
                "top_entity": None,
                "entity_score": None,
                "entity_mismatch": False
            }

    def _classify_entity(self, image_feature: np.ndarray) -> Tuple[Optional[str], Optional[float], bool]:
        """Entity prompts use cosine similarity, not softmax, against the image feature."""
        import torch
        if not self.entity_prompts:
            return None, None, False
        texts = list(self.entity_prompts.values())
        with torch.no_grad():
            tokens = self.tokenizer(texts)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            similarities = features.cpu().numpy() @ image_feature
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        top_entity = list(self.entity_prompts.keys())[best_index]
        mismatch = best_score < self.thresholds.clip_entity_mismatch_cutoff
        return top_entity, best_score, mismatch

    def classify_scene(self, media_path: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convenience wrapper for zero-shot scene classification."""
        if self.model is None:
            return {
                "status": ExtractorStatus.NOT_RUN.value,
                "top_category": None,
                "scores": {}
            }
        return self.extract({}, media_path)

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        img = safe_load_image(staged_path)
        if img is None:
            return {
                "status": ExtractorStatus.FAILED.value,
                "clip_scores": {},
                "indoor_outdoor": None,
                "time_of_day": None,
                "weather": None,
                "screen_type": None,
                "top_entity": None,
                "entity_score": None,
                "entity_mismatch": False
            }
        return self.classify(img)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


# Alias for backward compatibility
CLIPEngine = CLIPExtractor
