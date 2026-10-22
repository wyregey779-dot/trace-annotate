"""Smoke-test the installed distribution without importing the source checkout.

Run after installing a built wheel. The child process uses isolated Python from
an empty temporary directory; it does not open a browser or fetch model weights.
"""
import subprocess
import sys
import tempfile
from pathlib import Path


CHECK = r'''
import json
import os
import subprocess
import sysconfig
from importlib import metadata, resources
from pathlib import Path
import xml.etree.ElementTree as ET
import trace_annotate

source_root = Path(__import__("sys").argv[1]).resolve()
installed_path = Path(trace_annotate.__file__).resolve()
source_package = source_root / "trace_annotate"
assert source_package not in installed_path.parents, f"Imported checkout: {installed_path}"
site_roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
assert any(root in installed_path.parents for root in site_roots), f"Not installed in site-packages: {installed_path}"
assert metadata.version("trace-annotate") == trace_annotate.__version__
package = resources.files("trace_annotate")
for name in ("schema_v2.json", "default_vocab.json"):
    json.loads(package.joinpath("templates", name).read_text(encoding="utf-8"))
assert ET.fromstring(package.joinpath("templates", "label_studio_embodied_v2.1.xml").read_text(encoding="utf-8")).tag == "View"
ui = package.joinpath("embodied", "ui", "annotator.html").read_text(encoding="utf-8")
assert "<video" in ui and "TraceAnnotate" in ui
cli = Path(sysconfig.get_path("scripts")) / ("trace-annotate.exe" if os.name == "nt" else "trace-annotate")
assert cli.is_file(), f"Console entry point is missing from the active Python environment: {cli}"
cli_env = os.environ.copy()
cli_env.pop("PYTHONPATH", None)
cli_env.pop("PYTHONHOME", None)
result = subprocess.run([cli, "--help"], check=True, capture_output=True, text=True, env=cli_env)
assert "embodied-ui" in result.stdout
xml_path = Path("label-studio.xml")
subprocess.run([cli, "embodied-xml", "-o", str(xml_path)], check=True, env=cli_env)
assert ET.parse(xml_path).getroot().tag == "View"
print(f"Installed package smoke check passed: version={trace_annotate.__version__}, package={installed_path}")
print("JSON/XML templates, HTML annotator, CLI help and XML export verified outside checkout.")
'''


def main():
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="traceannotate-installed-") as scratch:
        subprocess.run(
            [sys.executable, "-I", "-c", CHECK, str(source_root)],
            cwd=scratch,
            check=True,
        )


if __name__ == "__main__":
    main()
