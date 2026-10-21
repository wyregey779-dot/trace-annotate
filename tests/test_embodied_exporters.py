import os
import tempfile
from trace_annotate.embodied.exporters.srt import export_to_srt
from trace_annotate.embodied.exporters.cvat import export_to_cvat_xml
from trace_annotate.embodied.exporters.action_dataset import export_to_action_segmentation


def test_srt_exporter():
    doc = {
        "clip": "sample.mp4",
        "working": [
            {"start": 1.0, "end": 5.0, "step_number": 1, "verdict": "对"}
        ],
        "idle": [
            {"start": 0.0, "end": 1.0},
            {"start": 5.0, "end": 10.0}
        ]
    }
    srt = export_to_srt(doc)
    assert "00:00:01,000 --> 00:00:05,000" in srt
    assert "【步骤 1 · 对】" in srt
    assert "IDLE" in srt


def test_cvat_xml_exporter():
    doc = {
        "clip": "sample.mp4",
        "working": [
            {"start": 1.0, "end": 5.0, "step_number": 1, "verdict": "对"}
        ]
    }
    xml_str = export_to_cvat_xml(doc, fps=30.0)
    assert "<annotations>" in xml_str
    assert 'label="working"' in xml_str
    assert 'name="step_number"' in xml_str


def test_action_segmentation_exporter():
    with tempfile.TemporaryDirectory() as temp_dir:
        docs = [
            {
                "clip": "test_clip.mp4",
                "working": [{"start": 1.0, "end": 3.0, "step_number": 1}],
                "idle": [{"start": 0.0, "end": 1.0}, {"start": 3.0, "end": 4.0}]
            }
        ]
        mapping, gts = export_to_action_segmentation(docs, temp_dir, fps=1.0)
        assert os.path.exists(mapping)
        assert len(gts) == 1
        with open(gts[0]) as f:
            lines = [l.strip() for l in f]
        assert lines == ["idle", "step_1", "step_1", "idle"]
