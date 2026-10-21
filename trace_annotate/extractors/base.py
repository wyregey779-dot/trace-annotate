"""
Base Extractor Interface and Standardized Execution Status Enums.
Guarantees explicit distinction between:
  - success (ran and produced prediction)
  - not_run (model weights absent or disabled)
  - failed (decode error or runtime crash)
  - skipped (condition not met)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any


class ExtractorStatus(str, Enum):
    """Standardized execution status for multi-modal feature extractors."""
    SUCCESS = "success"        # Model ran successfully
    NOT_RUN = "not_run"        # Model weights missing or disabled
    FAILED = "failed"          # Image/Video decode error or runtime exception
    SKIPPED = "skipped"        # Segment filtered out before extraction


class BaseExtractor(ABC):
    """Abstract interface for all multi-modal feature extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the extractor."""
        pass

    @abstractmethod
    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        """
        Process the asset at staged_path and return extracted features.
        
        Args:
            asset_item: dictionary of basic inventory info (rel_path, kind, etc.)
            staged_path: local temporary file path safe for reading
            
        Returns:
            Dictionary containing extracted fields, with 'status' key from ExtractorStatus.
        """
        pass
