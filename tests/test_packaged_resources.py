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
