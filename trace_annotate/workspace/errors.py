"""Stable error codes for the workspace API (contract section 8)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class WorkspaceError(Exception):
    """A typed workspace error. The code is stable across locales."""

    http_status = 500

    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def body(self, request_id: str) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "request_id": request_id,
                "details": self.details,
            }
        }


class BadRequest(WorkspaceError):
    http_status = 400


class SchemaInvalid(WorkspaceError):
    http_status = 422


class NotFound(WorkspaceError):
    http_status = 404


class Conflict(WorkspaceError):
    http_status = 409


class LimitExceeded(WorkspaceError):
    http_status = 413


class StorageFull(WorkspaceError):
    http_status = 507


class DependencyUnavailable(WorkspaceError):
    http_status = 503


def revision_conflict(expected: int, current: int) -> Conflict:
    return Conflict(
        "REVISION_CONFLICT",
        "Task has been modified by someone else",
        details={"expected_revision": expected, "current_revision": current},
    )
