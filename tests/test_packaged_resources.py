"""Check runtime resource contents; CI also checks a separately installed wheel."""
import json
from importlib import resources
import xml.etree.ElementTree as ET


def test_runtime_resources_are_readable():
    package = resources.files("trace_annotate")
    schema = json.loads(package.joinpath("templates/schema_v2.json").read_text(encoding="utf-8"))
    vocab = json.loads(package.joinpath("templates/default_vocab.json").read_text(encoding="utf-8"))
    xml = package.joinpath("templates/label_studio_embodied_v2.1.xml").read_text(encoding="utf-8")
    ui = package.joinpath("embodied/ui/annotator.html").read_text(encoding="utf-8")
    assert schema["type"] == "object"
    assert "primary_entities" in vocab
    assert ET.fromstring(xml).tag == "View"
    assert "<video" in ui
    assert "TraceAnnotate" in ui
    general_ui = package.joinpath("temporal/ui/annotator.html").read_text(encoding="utf-8")
    assert "<video" in general_ui and "TraceAnnotate" in general_ui
    hasher = package.joinpath("temporal/ui/media-hash.js").read_text(encoding="utf-8")
    notice = package.joinpath("temporal/ui/MEDIA_HASH_LICENSE.txt").read_text(encoding="utf-8")
    assert "TraceMediaHash" in hasher and "media-hash.js" in general_ui
    assert "js-sha256" in notice and "MIT" in notice
