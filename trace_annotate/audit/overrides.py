"""
Append-Only Human Overrides & Revision Audit System.
Maintains complete revision logs (who, when, what changed, basis, reason),
supports revision rollback without history deletion, and enforces atomic disk persistence.
"""
from __future__ import annotations
import os
import json
import time
import uuid
import shutil
from typing import Dict, Any, Optional, List
from trace_annotate.core.provenance import FillSource


class HumanOverrideManager:
    """
    Manages persistent human override files with immutable revision audit trails.
    """

    def __init__(self, override_file_path: str):
        path = os.path.abspath(override_file_path)
        if os.path.isdir(path) or not os.path.splitext(path)[1]:
            self.file_path = os.path.join(path, "human_overrides.json")
        else:
            self.file_path = path

        self.bak_path = self.file_path + ".bak"
        # Format: { rel_path: { "current": {...}, "history": [...] } }
        self.records: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load overrides with automatic backup recovery fallback."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._normalize_loaded_data(data)
                    return
            except Exception:
                # Attempt recovery from backup
                if os.path.exists(self.bak_path):
                    with open(self.bak_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._normalize_loaded_data(data)
                        return
                self.records = {}
        elif os.path.exists(self.bak_path):
            with open(self.bak_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._normalize_loaded_data(data)
        else:
            self.records = {}

    def _normalize_loaded_data(self, data: Dict[str, Any]) -> None:
        """Handle legacy format (flat dict per path) and upgrade to history format."""
        self.records = {}
        for rel_path, entry in data.items():
            if isinstance(entry, dict) and "history" in entry and "current" in entry:
                self.records[rel_path] = entry
            else:
                # Upgrade legacy flat entry
                self.records[rel_path] = {
                    "current": dict(entry),
                    "history": [
                        {
                            "revision_id": 1,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "author": "legacy_import",
                            "reason": "Initial migration from legacy overrides",
                            "changes": {k: {"old": None, "new": v} for k, v in entry.items()},
                            "basis": "legacy_record"
                        }
                    ]
                }

    def save(self) -> str:
        """
        Atomic save with fsync and backup snapshot.
        Never deletes destination before replacing.
        """
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        tmp_path = f"{self.file_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"

        # Write to temporary file with explicit fsync
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Create or update backup if original exists
        if os.path.exists(self.file_path):
            try:
                shutil.copy2(self.file_path, self.bak_path)
            except Exception:
                pass

        # Atomic replace (guaranteed atomic on Windows/POSIX in Python 3.3+)
        os.replace(tmp_path, self.file_path)
        return self.file_path

    def save_overrides(self, atomic: bool = True) -> str:
        """Convenience method returning file_path."""
        return self.save()

    def record_override(
        self,
        rel_path: str,
        primary_entity: Optional[str] = None,
        face_status: Optional[str] = None,
        best_frame_ts: Optional[float] = None,
        note: str = "",
        author: str = "human_auditor",
        basis: str = "visual_inspection",
        field_name: Optional[str] = None,
        old_val: Any = None,
        new_val: Any = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Add a new revision to rel_path's audit log and update current active values.
        Supports both predefined entity fields and generic field_name/new_val pairs.
        Returns the assigned revision_id.
        """
        entry = self.records.setdefault(rel_path, {"current": {}, "history": []})
        current = entry["current"]
        history = entry["history"]

        changes = {}

        # Generic field override
        if field_name is not None and new_val is not None:
            prev = current.get(field_name, old_val)
            changes[field_name] = {"old": prev, "new": new_val}
            current[field_name] = new_val

        if primary_entity is not None and primary_entity != current.get("primary_entity"):
            changes["primary_entity"] = {"old": current.get("primary_entity"), "new": primary_entity}
            current["primary_entity"] = primary_entity

        if face_status is not None and face_status != current.get("face_status"):
            changes["face_status"] = {"old": current.get("face_status"), "new": face_status}
            current["face_status"] = face_status

        if best_frame_ts is not None and best_frame_ts != current.get("best_frame_ts"):
            changes["best_frame_ts"] = {"old": current.get("best_frame_ts"), "new": best_frame_ts}
            current["best_frame_ts"] = best_frame_ts

        if note and note != current.get("note"):
            changes["note"] = {"old": current.get("note"), "new": note}
            current["note"] = note

        for k, v in kwargs.items():
            if v != current.get(k):
                changes[k] = {"old": current.get(k), "new": v}
                current[k] = v

        if not changes:
            return len(history)

        rev_id = len(history) + 1
        history.append({
            "revision_id": rev_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author": author,
            "reason": reason or note or f"Updated {list(changes.keys())}",
            "changes": changes,
            "basis": basis
        })

        self.save()
        return rev_id

    def rollback_revision(
        self,
        rel_path: str,
        target_revision_id: int,
        author: str = "system",
        reason: Optional[str] = None
    ) -> bool:
        """
        Roll back current active values to a previous revision state.
        Appends a rollback event to history rather than deleting future records.
        """
        if rel_path not in self.records:
            return False

        entry = self.records[rel_path]
        history = entry["history"]
        target_rev = next((r for r in history if r["revision_id"] == target_revision_id), None)
        if not target_rev:
            return False

        # Reconstruct state up to target_revision_id
        reconstructed = {}
        for r in history:
            if r["revision_id"] <= target_revision_id:
                for k, change in r.get("changes", {}).items():
                    reconstructed[k] = change["new"]

        # Record rollback as a new revision
        changes = {
            k: {"old": entry["current"].get(k), "new": reconstructed.get(k)}
            for k in set(list(entry["current"].keys()) + list(reconstructed.keys()))
            if entry["current"].get(k) != reconstructed.get(k)
        }

        entry["current"] = reconstructed
        entry["history"].append({
            "revision_id": len(history) + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author": author,
            "reason": reason or f"Rollback to revision #{target_revision_id}",
            "changes": changes,
            "basis": f"rollback_to_rev_{target_revision_id}"
        })
        self.save()
        return True

    def get_revision_history(self, rel_path: str) -> List[Dict[str, Any]]:
        """Return full audit log for a given asset path."""
        return list(self.records.get(rel_path, {}).get("history", []))

    def get_history(self, rel_path: str) -> List[Dict[str, Any]]:
        """Alias for get_revision_history."""
        return self.get_revision_history(rel_path)

    def get_effective_override(self, rel_path: str, field_name: str) -> Any:
        """Get the current active value for a specific field."""
        return self.records.get(rel_path, {}).get("current", {}).get(field_name)

    def apply_to_record(self, record: Dict[str, Any]) -> bool:
        """
        Apply current active overrides to a record in-place.
        """
        rel = record.get("rel_path", "")
        if rel not in self.records:
            return False

        override = self.records[rel]["current"]
        fill = record.setdefault("fill", {})

        if "primary_entity" in override and override["primary_entity"] is not None:
            record["primary_entity"] = override["primary_entity"]
            record["entity_status"] = "confirmed"
            fill["primary_entity"] = FillSource.HUMAN_CONFIRMED.value

        if "face_status" in override and override["face_status"] is not None:
            record["face_status"] = override["face_status"]
            fill["face_status"] = FillSource.HUMAN_CONFIRMED.value

        if "best_frame_ts" in override and override["best_frame_ts"] is not None:
            record["best_frame_ts"] = override["best_frame_ts"]
            fill["best_frame_ts"] = FillSource.HUMAN_CONFIRMED.value

        if "note" in override and override["note"]:
            tq = record.setdefault("tech_quality", {})
            tq["human_override_note"] = override["note"]

        return True


# Alias for convenience and backward compatibility
OverrideManager = HumanOverrideManager
