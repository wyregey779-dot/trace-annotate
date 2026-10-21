"""
Label Studio Integration & Bi-directional Bridge.
Exports schema v2.1 configuration XML and ingests Label Studio task export JSON
with order-independent multi-pass resolution and strict attribute preservation.
"""
from __future__ import annotations
import json
from typing import Dict, Any, List, Optional, Union
from trace_annotate.embodied.schema import (
    EmbodiedAnnotationDoc,
    WorkingSegment,
    IdleSegment,
    MidSegment,
    ClipStatus,
    Verdict,
    ErrorType
)


LABEL_STUDIO_XML_TEMPLATE = """
<View>
  <Video name="video" value="$video_url" sync="timeline"/>

  <!-- Clip-Level Diagnostics -->
  <Header value="第一步：整段诊断"/>
  <Choices name="clip_status" toName="video" showInline="true" required="true">
    <Choice value="正常"/>
    <Choice value="佩戴问题"/>
    <Choice value="描述不符-整段无关"/>
    <Choice value="描述不符-仅部分工作"/>
    <Choice value="这里没有"/>
  </Choices>
  <TextArea name="report" toName="video" placeholder="若非正常，填写Quick ID及事实依据" rows="2"/>

  <!-- Timeline Work/Idle Segmentation -->
  <Header value="第二步：工作段圈定与细化"/>
  <TimelineLabels name="timeline" toName="video">
    <Label value="working" background="#22c55e"/>
    <Label value="mid_segment" background="#eab308"/>
    <Label value="idle" background="#94a3b8"/>
  </TimelineLabels>

  <!-- Per-Region Working Properties -->
  <Header value="工作段执行对错判断"/>
  <Number name="step_number" toName="video" perRegion="true" min="1" max="100"/>
  <Choices name="verdict" toName="video" perRegion="true" showInline="true">
    <Choice value="对"/>
    <Choice value="错"/>
    <Choice value="无法判断"/>
  </Choices>

  <Header value="错误类型（仅错时填写）"/>
  <Choices name="error_type" toName="video" perRegion="true" showInline="true">
    <Choice value="遗漏步骤"/>
    <Choice value="顺序错误"/>
    <Choice value="重复/多做"/>
    <Choice value="用错物体"/>
    <Choice value="操作方式错误"/>
    <Choice value="意外事故"/>
    <Choice value="其他"/>
  </Choices>
  <TextArea name="evidence" toName="video" perRegion="true" placeholder="填入事实证据（如：t=14s 用了改锥应为扳手）" rows="2"/>
  <TextArea name="correction_events" toName="video" perRegion="true" placeholder="纠错动作描述（如：放回改锥，重新拿起扳手）" rows="2"/>
</View>
""".strip()


def generate_label_studio_xml() -> str:
    """Return the standardized Label Studio v2.1 configuration XML string."""
    return LABEL_STUDIO_XML_TEMPLATE


def parse_single_label_studio_task(raw_task: Dict[str, Any]) -> EmbodiedAnnotationDoc:
    """
    Parse a single Label Studio task item into a standardized EmbodiedAnnotationDoc.
    Multi-pass parsing ensures order-invariance: properties appearing before regions are fully bound.
    Missing attributes are preserved as None rather than silently defaulting to '对' or step 1.
    """
    data = raw_task.get("data", {})
    clip_name = data.get("video_url", "unknown.mp4").split("/")[-1]
    text_desc = data.get("text_description", "")

    doc = EmbodiedAnnotationDoc(
        clip=clip_name,
        text_description=text_desc,
        clip_status="正常"
    )

    annotations = raw_task.get("annotations", [])
    if not annotations:
        return doc

    latest = annotations[-1]
    result = latest.get("result", [])

    working_regions: Dict[str, Dict[str, Any]] = {}
    mid_segs: List[Dict[str, Any]] = []
    idle_segs: List[IdleSegment] = []
    pending_attributes: Dict[str, Dict[str, Any]] = {}

    # Pass 1: Collect clip diagnostics and categorize regions vs attributes
    for r in result:
        from_name = r.get("from_name", "")
        r_type = r.get("type", "")
        val = r.get("value", {})
        rid = r.get("id")

        if from_name == "clip_status":
            choices = val.get("choices", [])
            if choices:
                doc.clip_status = choices[0]
        elif from_name == "report":
            texts = val.get("text", [])
            if texts:
                doc.report = texts[0]
        elif r_type in ("labels", "timelinelabels", "videorectangle"):
            # Temporal boundaries
            start = val.get("start")
            end = val.get("end")
            seq = val.get("sequence", [])
            if start is None and seq:
                start = seq[0].get("time", 0.0)
                end = seq[-1].get("time", start)

            labels = val.get("labels", []) or val.get("timelinelabels", [])
            label = labels[0] if labels else "working"

            if start is not None and end is not None:
                start = float(start)
                end = float(end)
                if label == "working" or r_type == "videorectangle":
                    working_regions[rid] = {
                        "start": start,
                        "end": end,
                        "step_number": None,
                        "verdict": None,
                        "error_type": None,
                        "evidence": None,
                        "correction_events": []
                    }
                elif label == "mid_segment":
                    mid_segs.append({"start": start, "end": end, "reason": "idle"})
                elif label == "idle":
                    idle_segs.append(IdleSegment(start=start, end=end))

        elif rid is not None:
            # Region-bound attribute node (might appear before or after region definition)
            attr_dict = pending_attributes.setdefault(rid, {})

            if r_type == "number" or from_name == "step_number":
                step_val = val.get("number")
                if step_val is not None:
                    attr_dict["step_number"] = int(step_val)
            elif r_type == "choices":
                choices = val.get("choices", [])
                if choices:
                    c_val = choices[0]
                    if c_val in [v.value for v in Verdict]:
                        attr_dict["verdict"] = c_val
                    elif c_val in [e.value for e in ErrorType]:
                        attr_dict["error_type"] = c_val
            elif r_type == "textarea":
                if from_name == "evidence":
                    texts = val.get("text", [])
                    attr_dict["evidence"] = texts[0] if texts else None
                elif from_name == "correction_events":
                    texts = val.get("text", [])
                    if texts:
                        attr_dict.setdefault("correction_events", []).append(texts[0])

    # Pass 2: Order-independent binding of attributes to working regions
    for rid, wdict in working_regions.items():
        if rid in pending_attributes:
            attrs = pending_attributes[rid]
            if "step_number" in attrs and attrs["step_number"] is not None:
                wdict["step_number"] = attrs["step_number"]
            if "verdict" in attrs and attrs["verdict"] is not None:
                wdict["verdict"] = attrs["verdict"]
            if "error_type" in attrs:
                wdict["error_type"] = attrs["error_type"]
            if "evidence" in attrs:
                wdict["evidence"] = attrs["evidence"]
            if "correction_events" in attrs:
                wdict["correction_events"] = attrs["correction_events"]

    # Assemble sorted working segments
    working_list = []
    for wid, wdict in sorted(working_regions.items(), key=lambda x: x[1]["start"]):
        matched_mids = []
        for m in mid_segs:
            if wdict["start"] <= m["start"] and m["end"] <= wdict["end"]:
                matched_mids.append(MidSegment(start=m["start"], end=m["end"], reason=m["reason"]))

        working_list.append(WorkingSegment(
            start=wdict["start"],
            end=wdict["end"],
            step_number=wdict["step_number"],
            verdict=wdict["verdict"],
            error_type=wdict["error_type"],
            evidence=wdict["evidence"],
            correction_events=wdict["correction_events"],
            mid_segments=matched_mids
        ))

    doc.working = working_list
    doc.idle = idle_segs
    return doc


def parse_label_studio_export(
    raw_task_or_tasks: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[EmbodiedAnnotationDoc, List[EmbodiedAnnotationDoc]]:
    """
    Parse a raw Label Studio task or list of tasks export JSON into standardized EmbodiedAnnotationDoc(s).
    """
    if isinstance(raw_task_or_tasks, list):
        return [parse_single_label_studio_task(t) for t in raw_task_or_tasks]
    return parse_single_label_studio_task(raw_task_or_tasks)
