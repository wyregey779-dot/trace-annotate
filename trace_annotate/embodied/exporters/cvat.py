"""
CVAT Video 1.1 XML Exporter.
Converts embodied video temporal annotations into CVAT Video Track XML schema.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, List


def export_to_cvat_xml(
    doc: Dict[str, Any],
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080
) -> str:
    """
    Export annotation document to CVAT XML track format.
    """
    root = ET.Element("annotations")
    version = ET.SubElement(root, "version")
    version.text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    name = ET.SubElement(task, "name")
    name.text = doc.get("clip", "video.mp4")

    # Working tracks
    track_id = 0
    for w in doc.get("working", []):
        s_frame = int(round(float(w["start"]) * fps))
        e_frame = int(round(float(w["end"]) * fps))

        track = ET.SubElement(root, "track", id=str(track_id), label="working")
        track_id += 1

        box = ET.SubElement(track, "box", frame=str(s_frame), outside="0", occluded="0",
                            keyframe="1", xtl="0", ytl="0", xbr=str(width), ybr=str(height))
        # Attributes
        attr_step = ET.SubElement(box, "attribute", name="step_number")
        attr_step.text = str(w.get("step_number", 1))

        attr_verd = ET.SubElement(box, "attribute", name="verdict")
        attr_verd.text = str(w.get("verdict", "对"))

        if w.get("error_type"):
            attr_err = ET.SubElement(box, "attribute", name="error_type")
            attr_err.text = str(w["error_type"])

        # End frame box (outside=1)
        ET.SubElement(track, "box", frame=str(e_frame), outside="1", occluded="0",
                      keyframe="1", xtl="0", ytl="0", xbr=str(width), ybr=str(height))

    # Pretty print XML
    rough_str = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_str)
    return reparsed.toprettyxml(indent="  ")
