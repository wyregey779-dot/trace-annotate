"""
Append-Only Human Overrides Manager.
Applies manual ground-truth judgments without mutating raw history.
Upgrades field provenance to 'human_confirmed' and entity_status to 'confirmed'.
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, List
from trace_annotate.core.provenance import FillSource


class HumanOverrideManager:
    """Manages persistent human override files."""

    def __init__(self, override_file_path: str):
        self.file_path = os.path.abspath(override_file_path)
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load overrides from disk if present."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.overrides = json.load(f)
        else:
            self.overrides = {}

    def save(self) -> None:
        """Save overrides to disk atomically."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        tmp = self.file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.overrides, f, ensure_ascii=False, indent=2)
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        os.rename(tmp, self.file_path)

    def record_override(
        self,
        rel_path: str,
        primary_entity: Optional[str] = None,
        face_status: Optional[str] = None,
        best_frame_ts: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Add or update an override entry (append-only record)."""
        entry = self.overrides.setdefault(rel_path, {})
        if primary_entity is not None:
            entry["primary_entity"] = primary_entity
        if face_status is not None:
            entry["face_status"] = face_status
        if best_frame_ts is not None:
            entry["best_frame_ts"] = best_frame_ts
        if note:
            entry["note"] = note
        self.save()

    def apply_to_record(self, record: Dict[str, Any]) -> bool:
        """
        Apply overrides to a single record in-place.
        Returns True if an override was applied.
        """
        rel = record.get("rel_path", "")
        if rel not in self.overrides:
            return False

        override = self.overrides[rel]
        fill = record.setdefault("fill", {})

        if "primary_entity" in override:
            record["primary_entity"] = override["primary_entity"]
            record["entity_status"] = "confirmed"
            fill["primary_entity"] = FillSource.HUMAN_CONFIRMED.value

        if "face_status" in override:
            record["face_status"] = override["face_status"]
            fill["face_status"] = FillSource.HUMAN_CONFIRMED.value

        if "best_frame_ts" in override:
            record["best_frame_ts"] = override["best_frame_ts"]
            fill["best_frame_ts"] = FillSource.HUMAN_CONFIRMED.value

        if "note" in override:
            tq = record.setdefault("tech_quality", {})
            tq["human_override_note"] = override["note"]

        return True
