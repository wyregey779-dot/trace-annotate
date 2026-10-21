"""
Base Extractor Interface.
All feature extractors inherit from this class to guarantee modularity and decoupling.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


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
            Dictionary of extracted fields and metrics.
        """
        pass
