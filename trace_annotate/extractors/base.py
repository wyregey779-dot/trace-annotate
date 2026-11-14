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
import re
from typing import Dict, Any


_SAFE_FAILURE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,95}\Z")


def is_safe_failure_reason(value: object) -> bool:
    """Return whether a value is a portable, path-free public error code."""
    return isinstance(value, str) and _SAFE_FAILURE_TOKEN.fullmatch(value) is not None


def safe_failure_reason(exc: BaseException, fallback: str) -> str:
    """Return a bounded public error code without exposing exception text.

    Workspace errors may publish their stable ``code`` and our own validation
    errors may publish a single ASCII token.  Third-party exception messages
    are deliberately discarded because they commonly include local paths,
    model filenames, environment details or user content.
    """
    if not is_safe_failure_reason(fallback):
        raise ValueError("invalid_failure_fallback")
    code = getattr(exc, "code", None)
    if is_safe_failure_reason(code):
        return code
    if isinstance(exc, ValueError):
        message = str(exc)
        if is_safe_failure_reason(message):
            return message
    return fallback


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
