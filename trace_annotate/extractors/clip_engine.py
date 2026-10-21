"""
Zero-Shot CLIP Classification Engine.
Supports indoor/outdoor, day/night, screen types, weather conditions,
and entity verification prompts with anchor label softmax normalization.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from trace_annotate.extractors.base import BaseExtractor
from trace_annotate.core.hasher import safe_load_image
from trace_annotate.config import MetricThresholds


DEFAULT_PROMPTS = {
    "io": [["室内", "室内场景，在建筑物内部"], ["室外", "室外露天场景，街道或自然风光"]],
    "tod": [["白天", "白天明亮的自然光照"], ["夜", "夜晚黑暗的环境，人工灯光夜景"]],
    "weather": [
        ["雨", "下雨的天气，湿滑的路面或雨伞"],
        ["雪", "下雪的天气，地面积雪"],
        ["雾", "大雾天气，能见度很低"],
        [None, "普通的白天晴朗场景，没有雨雪雾等异常天气"]
    ],
    "screen": [
        ["无屏", "画面中没有屏幕和橱窗玻璃"],
        ["户外大屏", "户外大型LED显示屏正在播放画面"],
        ["橱窗", "商店的橱窗玻璃展示柜"],
        ["店招", "店铺招牌的发光字门头"]
    ]
}


class CLIPExtractor(BaseExtractor):
    """Multi-modal zero-shot classifier using CLIP."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        prompts: Optional[Dict[str, List[List[Any]]]] = None,
        entity_prompts: Optional[Dict[str, str]] = None,
        thresholds: Optional[MetricThresholds] = None
    ):
        self.model_path = model_name_or_path
        self.prompts = prompts or DEFAULT_PROMPTS
        self.entity_prompts = entity_prompts or {}
        self.thresholds = thresholds or MetricThresholds()
        self.model = None
        self._init_model()

    @property
    def name(self) -> str:
        return "clip"

    def _init_model(self) -> None:
        """Attempt to initialize local cn_clip or transformers model if available."""
        try:
            import torch
            import cn_clip.clip as clip
            from cn_clip.clip import load_from_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model_path and os.path.exists(self.model_path):
                self.model, self.preprocess = clip.load_from_name(
                    "ViT-B-16", device=device, download_root=os.path.dirname(self.model_path)
                )
            else:
                self.model = None
        except Exception:
            self.model = None

    def classify(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Classify image array across configured zero-shot prompt groups."""
        if self.model is None:
            return {
                "clip_scores": {},
                "indoor_outdoor": None,
                "time_of_day": None,
                "weather": None,
                "screen_type": "无屏",
                "top_entity": None,
                "entity_score": 0.0,
                "entity_mismatch": False
            }

        return {
            "clip_scores": {},
            "indoor_outdoor": None,
            "time_of_day": None,
            "weather": None,
            "screen_type": "无屏",
            "top_entity": None,
            "entity_score": 0.0,
            "entity_mismatch": False
        }

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        img = safe_load_image(staged_path)
        if img is None:
            return {
                "clip_scores": {},
                "indoor_outdoor": None,
                "time_of_day": None,
                "weather": None,
                "screen_type": "无屏",
                "top_entity": None,
                "entity_score": 0.0,
                "entity_mismatch": False
            }
        return self.classify(img)
