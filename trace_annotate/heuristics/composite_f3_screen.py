"""
Heuristic F3: Mobile Screen vs Storefront/Billboard OCR Disambiguation.
Differentiates genuine mobile smartphone screens from outdoor street storefronts
or shop signs that happen to contain phone numbers or "WeChat: 138xxxx".
"""
from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Optional, List


STRONG_UI_PATTERN = re.compile(
    r"Wi-?Fi|电量|信号格|拨号|通讯录|短信|相册|点赞|评论|转发|粉丝|播放量|小时前|分钟前|订阅|正在播放",
    re.IGNORECASE
)
WEAK_UI_PATTERN = re.compile(
    r"微信|支付宝|二维码|扫码|付款|收银",
    re.IGNORECASE
)


def evaluate_f3_screen(
    ocr_raw_text: str,
    clip_screen_prediction: str = "无屏",
    clip_screen_confidence: float = 0.0
) -> Tuple[str, bool, Optional[str]]:
    """
    Apply Composite Heuristic F3.
    
    Returns:
        screen_type: '无屏' | '户外大屏' | '橱窗' | '店招' | '手机屏'
        needs_review: bool
        evidence: Optional[str]
    """
    if not ocr_raw_text:
        return clip_screen_prediction or "无屏", False, None

    strong_matches = STRONG_UI_PATTERN.findall(ocr_raw_text)
    weak_matches = list(set(WEAK_UI_PATTERN.findall(ocr_raw_text)))

    is_phone_candidate = len(strong_matches) > 0 or len(weak_matches) >= 2

    if is_phone_candidate:
        evidence_words = strong_matches + weak_matches
        evidence_str = f"UI words: {', '.join(evidence_words[:5])}"

        if clip_screen_prediction in {"店招", "户外大屏", "橱窗"}:
            return clip_screen_prediction, True, f"Possible storefront with {evidence_str}"

        return "手机屏", True, f"Confirmed mobile screen UI: {evidence_str}"

    return clip_screen_prediction or "无屏", False, None
