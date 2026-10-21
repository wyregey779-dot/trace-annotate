"""
Data Models and Enums for Embodied AI Video Annotation.
Standardized according to Annotation Specification v2.1 (EgoProceAssist/Assembly101/EgoOops aligned).
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict


class ClipStatus(str, Enum):
    """First-step clip-level diagnostic values."""
    NORMAL = "正常"
    WEAR_ISSUE = "佩戴问题"
    DESC_MISMATCH_ALL = "描述不符-整段无关"
    DESC_MISMATCH_PART = "描述不符-仅部分工作"
    EMPTY = "这里没有"


class Verdict(str, Enum):
    """Execution verdict for working segments."""
    CORRECT = "对"
    WRONG = "错"
    UNCERTAIN = "无法判断"


class ErrorType(str, Enum):
    """The 7 standardized error sub-categories."""
    MISSING_STEP = "遗漏步骤"
    OUT_OF_ORDER = "顺序错误"
    DUPLICATE_EXTRA = "重复/多做"
    WRONG_OBJECT = "用错物体"
    WRONG_METHOD = "操作方式错误"
    ACCIDENT = "意外事故"
    OTHER = "其他"


@dataclass
class MidSegment:
    """Sub-segment hole within a working segment (idle or anomaly > 3s)."""
    start: float
    end: float
    reason: str = "idle"  # "idle" or "wrist_anomaly"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class WorkingSegment:
    """A continuous interval of task action execution."""
    start: float
    end: float
    step_number: Optional[int] = None
    verdict: Optional[str] = None  # "对" | "错" | "无法判断" or None if unlabelled
    error_type: Optional[str] = None
    evidence: Optional[str] = None
    correction_events: List[str] = field(default_factory=list)
    mid_segments: List[MidSegment] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class IdleSegment:
    """Timeline complement intervals where no task action is happening."""
    start: float
    end: float

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class EmbodiedAnnotationDoc:
    """Complete annotation document for a single video clip."""
    clip: str
    text_description: str
    clip_status: str = "正常"
    working: List[WorkingSegment] = field(default_factory=list)
    idle: List[IdleSegment] = field(default_factory=list)
    report: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
