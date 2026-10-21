"""
Provenance & Audit Traceability System.
Implements the 'fill' dictionary mechanism to track the origin, algorithm,
and confidence tier of every single annotated field in the metadata schema,
as well as pipeline stage execution provenance with deterministic hashing.
"""
from __future__ import annotations
import os
import json
import time
import hashlib
from enum import Enum
from typing import Dict, Any, Optional, Union, List


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
    """Helper to record and validate field-level and stage-level provenance."""

    def __init__(
        self,
        storage_or_fills: Optional[Union[str, Dict[str, str]]] = None,
        initial_fills: Optional[Dict[str, str]] = None
    ):
        if isinstance(storage_or_fills, str):
            self.storage_dir: Optional[str] = os.path.abspath(storage_or_fills)
            self.fills: Dict[str, str] = dict(initial_fills) if initial_fills else {}
        elif isinstance(storage_or_fills, dict):
            self.storage_dir = None
            self.fills = dict(storage_or_fills)
        else:
            self.storage_dir = None
            self.fills = dict(initial_fills) if initial_fills else {}

        self.stages: Dict[str, Dict[str, Any]] = {}
        self._load_persisted_stages()

    def _load_persisted_stages(self) -> None:
        if self.storage_dir:
            prov_file = os.path.join(self.storage_dir, "provenance.json")
            if os.path.exists(prov_file):
                try:
                    with open(prov_file, "r", encoding="utf-8") as f:
                        self.stages = json.load(f)
                except Exception:
                    self.stages = {}

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

    def record_stage(
        self,
        stage_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record stage execution with deterministic input hash and idempotent tracking.
        """
        norm_inputs = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
        input_hash = hashlib.md5(norm_inputs.encode("utf-8")).hexdigest()

        record = {
            "stage_name": stage_name,
            "input_hash": input_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs_summary": {k: type(v).__name__ for k, v in inputs.items()},
            "outputs_summary": {k: type(v).__name__ for k, v in outputs.items()},
            "metadata": metadata or {}
        }
        self.stages[stage_name] = record

        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            prov_file = os.path.join(self.storage_dir, "provenance.json")
            try:
                with open(prov_file, "w", encoding="utf-8") as f:
                    json.dump(self.stages, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return record

    def get_stage_record(self, stage_name: str) -> Optional[Dict[str, Any]]:
        return self.stages.get(stage_name)

    @classmethod
    def summarize_fills(cls, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate summary of fill counts across all rows."""
        summary: Dict[str, int] = {}
        for r in records:
            fills = r.get("fill", {})
            for field_name, source in fills.items():
                summary[source] = summary.get(source, 0) + 1
        return summary
