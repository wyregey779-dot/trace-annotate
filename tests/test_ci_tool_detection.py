"""P12: required media tools are detected with ``shutil.which`` and fail in CI.

A developer machine may skip media coverage when FFmpeg is absent, but a CI run
must not turn a broken toolchain into a green build. These tests pin both
branches of that rule without needing the tools to be missing.
"""
from __future__ import annotations

import shutil

import pytest

from tests.support import media_tools


def test_both_tools_resolve_to_real_executables_on_this_machine():
    found = media_tools.media_tools()
    assert set(found) == {"ffmpeg", "ffprobe"}
    missing = media_tools.missing_tools()
    if missing:
        pytest.skip("media tools not installed here: " + ", ".join(missing))
    for name, path in found.items():
        assert path and shutil.which(name) == path


def test_a_missing_tool_skips_locally(monkeypatch):
    monkeypatch.delenv(media_tools.STRICT_ENV, raising=False)
    real_which = shutil.which
    monkeypatch.setattr(media_tools.shutil, "which",
                        lambda name: None if name == "ffmpeg" else real_which(name))
    with pytest.raises(pytest.skip.Exception) as excinfo:
        media_tools.require_media_tools()
    assert "ffmpeg" in str(excinfo.value)


def test_a_missing_tool_fails_in_ci(monkeypatch):
    monkeypatch.setenv(media_tools.STRICT_ENV, "1")
    real_which = shutil.which
    monkeypatch.setattr(media_tools.shutil, "which",
                        lambda name: None if name == "ffprobe" else real_which(name))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        media_tools.require_media_tools()
    assert "ffprobe" in str(excinfo.value)


def test_both_tools_present_returns_their_paths(monkeypatch):
    monkeypatch.setenv(media_tools.STRICT_ENV, "1")
    monkeypatch.setattr(media_tools.shutil, "which",
                        lambda name: f"/opt/tools/{name}" if name in media_tools.REQUIRED else None)
    assert media_tools.require_media_tools() == ("/opt/tools/ffmpeg", "/opt/tools/ffprobe")
    assert media_tools.missing_tools() == []
