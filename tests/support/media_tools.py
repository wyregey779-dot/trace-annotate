"""Cross-platform detection of the media tools the tests really need.

``shutil.which`` is the only accepted detector: a bare ``Path("ffmpeg").exists``
is false for an executable on PATH and would silently skip coverage. On a
developer machine a missing tool skips the affected tests; in CI
(``TRACEANNOTATE_REQUIRE_MEDIA_TOOLS=1``) the same condition fails the run, so a
broken toolchain can never look like a green build.
"""
from __future__ import annotations

import os
import shutil

import pytest

REQUIRED = ("ffmpeg", "ffprobe")
STRICT_ENV = "TRACEANNOTATE_REQUIRE_MEDIA_TOOLS"


def media_tools() -> dict:
    """Return the resolved path of every required tool, without raising."""
    return {name: shutil.which(name) for name in REQUIRED}


def missing_tools() -> list:
    return [name for name, path in media_tools().items() if not path]


def require_media_tools() -> tuple:
    """Return ``(ffmpeg, ffprobe)`` or stop the test with an explicit reason."""
    found = media_tools()
    missing = [name for name in REQUIRED if not found[name]]
    if not missing:
        return found["ffmpeg"], found["ffprobe"]
    message = "required media tool(s) not found on PATH: " + ", ".join(missing)
    if os.environ.get(STRICT_ENV) == "1":
        pytest.fail(message + f" ({STRICT_ENV}=1)")
    pytest.skip(message)
