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
general_ui = package.joinpath("temporal", "ui", "annotator.html").read_text(encoding="utf-8")
assert "<video" in general_ui and "TraceAnnotate" in general_ui
assert "TraceMediaHash" in package.joinpath("temporal", "ui", "media-hash.js").read_text(encoding="utf-8")
assert "js-sha256" in package.joinpath("temporal", "ui", "MEDIA_HASH_LICENSE.txt").read_text(encoding="utf-8")
cli = Path(sysconfig.get_path("scripts")) / ("trace-annotate.exe" if os.name == "nt" else "trace-annotate")
assert cli.is_file(), f"Console entry point is missing from the active Python environment: {cli}"
cli_env = os.environ.copy()
cli_env.pop("PYTHONPATH", None)
cli_env.pop("PYTHONHOME", None)
result = subprocess.run([cli, "--help"], check=True, capture_output=True, text=True, env=cli_env)
assert "embodied-ui" in result.stdout
assert "embodied-pipeline" in result.stdout
assert "video-ui" in result.stdout and "video-import" in result.stdout
assert "video-hash" in result.stdout and "video-bind" in result.stdout
assert "video-prepare" in result.stdout and "video-open" in result.stdout
from trace_annotate.temporal.preparation import prepare_media
from trace_annotate.temporal.project_server import make_project_server
assert callable(prepare_media) and callable(make_project_server)
from trace_annotate.workspace.ui import EDITOR_HTML
from trace_annotate.workspace.frames import FrameService
from trace_annotate.workspace.server import make_server
assert '<video' in EDITOR_HTML and '__TOKEN__' in EDITOR_HTML and '__NONCE__' in EDITOR_HTML
assert callable(FrameService) and callable(make_server)
assert 'workspace' in result.stdout
editor = subprocess.run([cli, "video-ui", "--print-path"], check=True, capture_output=True,
                        text=True, encoding="utf-8", env=cli_env)
assert Path(editor.stdout.strip()).is_file()
general = Path("general.json")
general.write_text(json.dumps({"kind": "temporal", "schema_version": 1, "clip": "test.mp4",
    "duration": 5, "segments": [{"id": "a", "start": 1, "end": 3, "label": "event",
    "text": "A test cue", "tags": [], "track": "events", "review_status": "unreviewed"}]}), encoding="utf-8")
subprocess.run([cli, "video-validate", str(general)], check=True, env=cli_env)
subprocess.run([cli, "video-export", str(general), "--format", "srt", "-o", "test.srt"], check=True, env=cli_env)
assert "A test cue" in Path("test.srt").read_text(encoding="utf-8")
media_path = Path("test.mp4")
media_path.write_bytes(b"synthetic bytes for installed CLI hashing; not a decodable video")
subprocess.run([cli, "video-bind", str(general), "--media", str(media_path), "--duration", "5", "-o", "bound.json"], check=True, env=cli_env)
bound = json.loads(Path("bound.json").read_text(encoding="utf-8"))
assert bound["schema_version"] == 2
assert bound["media_sha256"] == __import__("hashlib").sha256(media_path.read_bytes()).hexdigest()
assert bound["media_size"] == media_path.stat().st_size
subprocess.run([cli, "video-validate", "bound.json"], check=True, env=cli_env)
xml_path = Path("label-studio.xml")
subprocess.run([cli, "embodied-xml", "-o", str(xml_path)], check=True, env=cli_env)
assert ET.parse(xml_path).getroot().tag == "View"

# Exercise the installed quality implementation, not code imported from checkout.
annotation = {"clip": "synthetic-quality.mp4", "duration": 10,
    "text_description": "Synthetic installed-package format test", "clip_status": "正常",
    "expected_step_count": 1,
    "working": [{"start": 1, "end": 3, "step_number": 1, "verdict": "对"}],
    "idle": [{"start": 0, "end": 1}, {"start": 3, "end": 10}]}
quality_source = Path("quality.json")
quality_source.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
original_quality = quality_source.read_bytes()
subprocess.run([cli, "embodied-pipeline", str(quality_source), "--report", "quality-report.html"], check=True, env=cli_env)
assert "PASS_SCOPED" in Path("quality-report.html").read_text(encoding="utf-8")
assert quality_source.read_bytes() == original_quality
old_report = Path("quality-report.html").read_bytes()
repeated = subprocess.run([cli, "embodied-pipeline", str(quality_source), "--report", "quality-report.html"], env=cli_env)
assert repeated.returncode == 2
assert Path("quality-report.html").read_bytes() == old_report
alias = subprocess.run([cli, "embodied-export", str(quality_source), "--format", "srt", "-o", str(quality_source)], env=cli_env)
assert alias.returncode != 0 and quality_source.read_bytes() == original_quality
annotation["working"][0]["start"] = -1
Path("invalid-quality.json").write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
invalid = subprocess.run([cli, "embodied-pipeline", "invalid-quality.json", "--report", "invalid-quality-report.html"], env=cli_env)
assert invalid.returncode == 1
assert "INVALID_TIME" in Path("invalid-quality-report.html").read_text(encoding="utf-8")
print(f"Installed package smoke check passed: version={trace_annotate.__version__}, package={installed_path}")
print("Templates, editors, workspace modules, general segment export, embodied XML and quality failure gates verified outside checkout.")
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
