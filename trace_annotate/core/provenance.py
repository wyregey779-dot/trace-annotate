"""
Provenance & Audit Traceability System.
Implements the 'fill' dictionary mechanism to track the origin, algorithm,
and confidence tier of every single annotated field in the metadata schema.
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional


class FillSource(str, Enum):
    """Standardized provenance tiers for metadata values."""
    AUTO_METRIC = "auto_metric"           # Deterministic algorithm (laplacian, ffprobe, dhash, yunet)
    AUTO_RULE = "auto_rule"               # Deterministic rule-based deduction
    AUTO_PATH = "auto_path"               # Inferred from file path or folder name
    AUTO_PATH_CONFIRMED = "auto_path_confirmed" # Deep folder hierarchy semantic confirmation
    CANDIDATE_ONLY = "candidate_only"     # Low-confidence machine proposal, tagged as candidate
    PENDING_REVIEW = "pending_review"     # Requires human visual inspection
    PENDING_TOOL = "pending_tool"         # Requires external/specialized tool not currently loaded
    HUMAN_CONFIRMED = "human_confirmed"   # Validated by human auditor (highest authority)


class ProvenanceTracker:
    """Helper to record and validate field-level provenance."""

    def __init__(self, initial_fills: Optional[Dict[str, str]] = None):
        self.fills: Dict[str, str] = dict(initial_fills) if initial_fills else {}

    def set_source(self, field_name: str, source: FillSource | str) -> None:
        """Assign a provenance source to a specific schema field."""
        if isinstance(source, FillSource):
            self.fills[field_name] = source.value
        else:
            self.fills[field_name] = str(source)

    def get_source(self, field_name: str) -> Optional[str]:
        return self.fills.get(field_name)

    def is_human_confirmed(self, field_name: str) -> bool:
        return self.fills.get(field_name) == FillSource.HUMAN_CONFIRMED.value

    def to_dict(self) -> Dict[str, str]:
        return dict(self.fills)

    @classmethod
    def summarize_fills(cls, records: list[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate summary of fill counts across all rows."""
        summary: Dict[str, int] = {}
        for r in records:
            fills = r.get("fill", {})
            for field_name, source in fills.items():
                summary[source] = summary.get(source, 0) + 1
        return summary
