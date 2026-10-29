"""Validate a wheel/sdist pair and write their SHA-256 checksums.

Requires tomli on Python < 3.11. Run after `python -m build` and before upload;
this script never publishes files or contacts an external service.
"""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
from pathlib import Path
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


def package_version(source: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__version__"
                   for target in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
    raise ValueError("Package __version__ is missing or not a string literal")


def check_metadata(raw: bytes, version: str) -> None:
    metadata = BytesParser().parsebytes(raw)
    if metadata["Name"] != "trace-annotate" or metadata["Version"] != version:
        raise ValueError(f"Unexpected distribution: {metadata['Name']} {metadata['Version']}")
    if not metadata.get_all("Project-URL"):
        raise ValueError("Distribution is missing project URLs")


def validate_assets(dist: Path, expected_tag: str | None = None) -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if package_version((root / "trace_annotate/__init__.py").read_text(encoding="utf-8")) != version:
        raise ValueError("pyproject.toml and package versions differ")
    if expected_tag is not None and expected_tag != f"v{version}":
        raise ValueError(f"Tag {expected_tag!r} does not match package version v{version}")

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Use a clean output directory containing exactly one wheel and one sdist")
    wheel, sdist = wheels[0], sdists[0]
    if wheel.name != f"trace_annotate-{version}-py3-none-any.whl" or sdist.name != f"trace_annotate-{version}.tar.gz":
        raise ValueError("Distribution filenames do not match the package version")

    with zipfile.ZipFile(wheel) as archive:
        check_metadata(archive.read(f"trace_annotate-{version}.dist-info/METADATA"), version)
        source_files = {
            file.relative_to(root).as_posix(): file.read_bytes()
            for file in (root / "trace_annotate").rglob("*")
            if file.is_file() and file.suffix in {".py", ".json", ".xml", ".html", ".js", ".txt"}
        }
        wheel_files = {name for name in archive.namelist() if name.startswith("trace_annotate/")}
        if wheel_files != source_files.keys():
            raise ValueError("Wheel package files do not match the source checkout")
        for name, content in source_files.items():
            if archive.read(name) != content:
                raise ValueError(f"Wheel contains stale content: {name}")

    with tarfile.open(sdist, "r:gz") as archive:
        prefix = f"trace_annotate-{version}/"
        metadata_file = archive.extractfile(prefix + "PKG-INFO")
        if metadata_file is None:
            raise ValueError("Source distribution metadata is missing")
        check_metadata(metadata_file.read(), version)
        for name, content in source_files.items():
            member = archive.extractfile(prefix + name)
            if member is None or member.read() != content:
                raise ValueError(f"Source distribution contains missing/stale content: {name}")
        for name in ("LICENSE", "README.md", "README_CN.md", "CHANGELOG.md", "CONTRIBUTING.md",
                     "docs/QUICKSTART.md", "docs/RELEASING.md",
                     "docs/GENERAL_VIDEO_ANNOTATION.md",
                     "docs/MEDIA_COMPATIBILITY.md",
                     "docs/QUALITY_PIPELINE.md", "examples/annotations/quality-batch.json",
                     "docs/DEMO_MEDIA.md", "docs/media/embodied-editor.png",
                     "docs/media/embodied-qc.png", "docs/media/embodied-demo.mp4",
                     "docs/media/traceannotate-cover.png",
                     "docs/media/embodied-example.json", "docs/media/provenance.json",
                     "scripts/windows_qc.cmd", "scripts/windows_workbench.cmd",
                     "examples/run_quickstart.py", "examples/annotations/assembly.json",
                     "examples/annotations/general-video.json", "scripts/check_general_editor.py",
                     "scripts/check_editor_integrity.py",
                     "scripts/check_media_identity.py",
                     "scripts/check_media_admission.py", "scripts/check_prepared_editor.py",
                     "scripts/check_embodied_editor.py", "scripts/check_workspace_editor.py",
                     "scripts/check_enhanced_editor.py", "tests/test_enhanced_editor_runner.py",
                     "tests/test_workspace_history_review.py", "tests/test_workspace_roundtrip_regressions.py",
                     "tests/test_workspace_drafts.py", "tests/test_workspace_backup.py",
                     "tests/test_workspace_legacy_endpoints.py",
                     "tests/test_workspace_legacy_adapter.py",
                     "docs/WORKSPACE_DEVELOPMENT.md",
                     "scripts/check_installed_package.py", "scripts/release_assets.py"):
            member = archive.extractfile(prefix + name)
            if member is None or member.read() != (root / name).read_bytes():
                raise ValueError(f"Source distribution contains missing/stale supporting file: {name}")
    return [wheel, sdist]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--expected-tag")
    args = parser.parse_args()
    try:
        assets = validate_assets(args.dist, args.expected_tag)
    except (ValueError, KeyError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Release validation failed: {exc}\n")
    lines = [f"{hashlib.sha256(asset.read_bytes()).hexdigest()}  {asset.name}" for asset in assets]
    checksum_path = args.dist / "SHA256SUMS"
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validated wheel/sdist metadata and package contents; wrote {checksum_path}")


if __name__ == "__main__":
    main()
