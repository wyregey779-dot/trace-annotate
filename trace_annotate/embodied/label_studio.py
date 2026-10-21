"""
Label Studio Integration & Configuration Generator.
Generates XML task templates and converts Label Studio annotations to standardized format.
"""
from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from trace_annotate.embodied.schema import EmbodiedAnnotationDoc, WorkingSegment, MidSegment, IdleSegment


LABEL_STUDIO_XML_TEMPLATE = """<View>
  <Header value="第一步：整段诊断（先选这个，命中即停）" />
  <Text name="description" value="$text_description" />
  <Choices name="clip_status" toName="video" choice="single" showInline="true" required="true">
    <Choice value="正常" />
    <Choice value="佩戴问题" />
    <Choice value="描述不符-整段无关" />
    <Choice value="描述不符-仅部分工作" />
    <Choice value="这里没有" />
  </Choices>
  <TextArea name="clip_report" toName="video" editable="true" rows="1"
    placeholder="选了佩戴问题/描述不符时必填：Quick ID（视频左上角）+ 一句话现象；正常则留空" />

  <Header value="第二步：画 working 段（绿）；段内 idle/视角异常画段中段（橙），不拆段" />
  <Video name="video" value="$video_url" framerate="30.0" />
  <VideoTimeline name="timeline" />
  <Labels name="videoLabels" toName="video" allowCreate="false">
    <Label value="working" background="#00C853" />
    <Label value="段中段" background="#FF9800" />
    <Label value="idle" background="#9E9E9E" />
  </Labels>

  <Header value="第三步：点选每个绿色 working 段，先挂步骤编号，再判对错（橙/灰段不判）" />
  <Number name="step_number" toName="video" perRegion="true"
    perRegionControl="videoLabels" precision="0" min="1" max="30"
    value="1"
    hint="该段对应流程描述的第几步；工具据此自动 diff 遗漏/乱序/重复" />
  <Choices name="seg_verdict" toName="video" choice="single" perRegion="true"
    perRegionControl="videoLabels">
    <Choice value="对" />
    <Choice value="错" />
    <Choice value="无法判断" />
  </Choices>
  <Choices name="error_type" toName="video" choice="single" perRegion="true"
    perRegionControl="videoLabels"
    visibleWhen="choice-selected" whenChoiceFromName="seg_verdict" whenChoiceValue="错">
    <Choice value="遗漏步骤" />
    <Choice value="顺序错误" />
    <Choice value="重复/多做" />
    <Choice value="用错物体" />
    <Choice value="操作方式错误" />
    <Choice value="意外事故" />
    <Choice value="其他" />
  </Choices>
  <TextArea name="seg_evidence" toName="video" editable="true" perRegion="true"
    perRegionControl="videoLabels" rows="2"
    placeholder="判错必填，按模板填：t=XX 用了A应为B / 第N步全程未出现 / t=XX 物体掉落"
    visibleWhen="choice-selected" whenChoiceFromName="seg_verdict" whenChoiceValue="错" />
  <TextArea name="correction_events" toName="video" editable="true" perRegion="true"
    perRegionControl="videoLabels" rows="1"
    placeholder="（选填）段内做错后自行纠正/拿错秒放的事件：t=XX 曾做错X已自行纠正；无则留空" />
</View>
"""


def generate_label_studio_xml() -> str:
    """Return the standardized Label Studio v2.1 configuration XML string."""
    return LABEL_STUDIO_XML_TEMPLATE


def parse_label_studio_export(raw_task: Dict[str, Any]) -> EmbodiedAnnotationDoc:
    """
    Parse a raw Label Studio task export JSON item into a standardized EmbodiedAnnotationDoc.
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

    working_map = {}
    mid_segs = []
    idle_segs = []

    for r in result:
        val = r.get("value", {})
        from_name = r.get("from_name", "")

        if from_name == "clip_status":
            choices = val.get("choices", [])
            if choices:
                doc.clip_status = choices[0]
        elif from_name == "clip_report":
            text = val.get("text", [])
            if text:
                doc.report = text[0]

        elif from_name == "videoLabels":
            labels = val.get("labels", [])
            start = float(val.get("start", 0.0))
            end = float(val.get("end", 0.0))
            region_id = r.get("id")

            if "working" in labels:
                working_map[region_id] = {
                    "start": start,
                    "end": end,
                    "step_number": 1,
                    "verdict": "对",
                    "error_type": None,
                    "evidence": None,
                    "correction_events": [],
                    "mid_segments": []
                }
            elif "段中段" in labels:
                mid_segs.append({"start": start, "end": end, "reason": "idle"})
            elif "idle" in labels:
                idle_segs.append(IdleSegment(start=start, end=end))

        elif from_name in {"step_number", "seg_verdict", "error_type", "seg_evidence", "correction_events"}:
            region_id = r.get("id") or r.get("parent_id")
            if region_id and region_id in working_map:
                w = working_map[region_id]
                if from_name == "step_number":
                    w["step_number"] = int(val.get("number", 1))
                elif from_name == "seg_verdict":
                    choices = val.get("choices", [])
                    if choices:
                        w["verdict"] = choices[0]
                elif from_name == "error_type":
                    choices = val.get("choices", [])
                    if choices:
                        w["error_type"] = choices[0]
                elif from_name == "seg_evidence":
                    texts = val.get("text", [])
                    if texts:
                        w["evidence"] = texts[0]
                elif from_name == "correction_events":
                    texts = val.get("text", [])
                    if texts:
                        w["correction_events"].append(texts[0])

    working_list = []
    for wid, wdict in sorted(working_map.items(), key=lambda x: x[1]["start"]):
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
