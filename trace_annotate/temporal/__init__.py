"""General temporal annotation without task-specific step or verdict rules."""

from trace_annotate.temporal.schema import require_valid_document, validate_document

__all__ = ["validate_document", "require_valid_document"]
