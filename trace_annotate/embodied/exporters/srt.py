"""
SRT Subtitle Exporter for Visual Video Review.
Generates an SRT subtitle file from working/idle/error annotations.
Allows instant video review in standard desktop media players (VLC, PotPlayer, QuickTime).
"""
from __future__ import annotations
from typing import Dict, Any, List


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_to_srt(doc: Dict[str, Any], include_idle: bool = True) -> str:
    """
    Convert an EmbodiedAnnotationDoc dictionary into an SRT subtitle string.
    """
    entries = []

    # Add working segments
    for w in doc.get("working", []):
        start = float(w["start"])
        end = float(w["end"])
        step = w.get("step_number", 1)
        verdict = w.get("verdict", "对")
        err_type = w.get("error_type")
        evidence = w.get("evidence")

        if verdict == "错":
            text = f"【步骤 {step} · 错 ({err_type})】\n证据: {evidence or '无'}"
        elif verdict == "对":
            text = f"【步骤 {step} · 对】(正在执行任务动作)"
        else:
            text = f"【步骤 {step} · 无法判断】"

        entries.append({
            "start": start,
            "end": end,
            "text": text,
            "priority": 1
        })

        # Add mid-segment holes if any
        for m in w.get("mid_segments", []):
            m_s = float(m["start"])
            m_e = float(m["end"])
            m_r = m.get("reason", "idle")
            entries.append({
                "start": m_s,
                "end": m_e,
                "text": f"⚠ [段中段 · {m_r}] (停顿/异常)",
                "priority": 2
            })

    # Add idle segments
    if include_idle:
        for idl in doc.get("idle", []):
            i_s = float(idl["start"])
            i_e = float(idl["end"])
            entries.append({
                "start": i_s,
                "end": i_e,
                "text": "--- [IDLE 空闲/等待] ---",
                "priority": 0
            })

    # Sort chronologically
    sorted_entries = sorted(entries, key=lambda x: (x["start"], -x["priority"]))

    # Build SRT format
    srt_lines = []
    for idx, e in enumerate(sorted_entries, 1):
        t_start = format_srt_time(e["start"])
        t_end = format_srt_time(e["end"])
        srt_lines.append(f"{idx}\n{t_start} --> {t_end}\n{e['text']}\n")

    return "\n".join(srt_lines)
