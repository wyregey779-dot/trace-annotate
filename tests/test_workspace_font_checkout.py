"""The bundled license has a byte-level identity, including on Windows."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LICENSE = Path("trace_annotate/workspace/fonts/OFL.txt")


def test_bundled_font_license_survives_autocrlf_checkout(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git is required to exercise checkout conversion")
    expected = (ROOT / LICENSE).read_bytes()
    metadata = json.loads((ROOT / LICENSE.parent / "font.json").read_bytes())
    assert hashlib.sha256(expected).hexdigest() == metadata["license_sha256"]
    assert b"\r\n" not in expected and b"\n" in expected

    repo = tmp_path / "repository"
    repo.mkdir()
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    git("init", "-q")
    (repo / LICENSE).parent.mkdir(parents=True)
    (repo / LICENSE).write_bytes(expected)
    git("-c", "core.autocrlf=false", "add", "--", LICENSE.as_posix())
    before = tmp_path / "before"
    before.mkdir()
    git("-c", "core.autocrlf=true", "checkout-index", "--all", "--prefix=" + str(before) + os.sep)
    assert (before / LICENSE).read_bytes() != expected
    assert b"\r\n" in (before / LICENSE).read_bytes()

    (repo / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    git("-c", "core.autocrlf=false", "add", "--", ".gitattributes")
    after = tmp_path / "after"
    after.mkdir()
    git("-c", "core.autocrlf=true", "checkout-index", "--all", "--prefix=" + str(after) + os.sep)
    actual = (after / LICENSE).read_bytes()
    assert actual == expected
    assert hashlib.sha256(actual).hexdigest() == metadata["license_sha256"]
