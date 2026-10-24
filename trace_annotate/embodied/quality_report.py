"""Presentation of actual QC results; never infer video truth from annotation totals."""
from __future__ import annotations

import math
import json
from html import escape


REPORT_CSS = """
:root{color-scheme:dark;--bg:#0b111b;--surface:#121c2a;--border:#29384a;--text:#edf3fa;--muted:#a9b9cb;--green:#4ade80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,'Segoe UI',sans-serif}
main{max-width:1440px;margin:auto;padding:40px 32px}header{border-bottom:1px solid var(--border);padding-bottom:24px}
.eyebrow{color:var(--green);font-size:13px;font-weight:700;letter-spacing:.12em}.brand{font-size:18px;font-weight:750;color:#fff}
h1{font-size:clamp(24px,2.5vw,36px);line-height:1.35;letter-spacing:-.02em;margin:12px 0}h2{font-size:19px;margin:28px 0 12px}
p{margin:10px 0}.muted{color:var(--muted)}.run{font:12px/1.7 ui-monospace,monospace;overflow-wrap:anywhere;color:var(--muted)}
.status{display:inline-block;padding:4px 12px;border:1px solid var(--border);border-radius:6px;font-size:13px;font-weight:700}
.status-pass{background:#123628;color:#86efac}.status-fail{background:#421e28;color:#fda4af}.status-review{background:#392d16;color:#fde68a}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:24px 0}
.metric,.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px}
.metric dt{font-size:13px;color:var(--muted)}.metric dd{font-size:34px;font-weight:700;margin:8px 0 0;line-height:1.2;font-variant-numeric:tabular-nums}
.metric dd small{font-size:16px;font-weight:500;color:var(--muted)}.metric .working{color:#86efac}.metric .idle{color:#cbd5e1}
.ratio{height:18px;border-radius:5px;overflow:hidden;display:flex;background:#475569;margin:14px 0}
.ratio-working{background:#22c55e}.ratio-idle{background:#64748b}.legend{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.notice{border-left:3px solid #627b98;padding:8px 14px;color:var(--muted);background:#0e1723;font-size:13px}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:8px}table{border-collapse:collapse;width:100%;min-width:900px}
caption{text-align:left;padding:12px 16px;background:var(--surface);color:var(--muted);font-size:13px}
th,td{border-bottom:1px solid var(--border);padding:12px 14px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{font-size:12px;color:var(--muted);background:#172232}td{font-size:13px}td:nth-child(3){max-width:220px}td:nth-child(4),td:nth-child(5),td:nth-child(6){white-space:nowrap}td:last-child{min-width:200px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,monospace;margin:0}
details{margin-top:24px;border:1px solid var(--border);border-radius:8px;background:var(--surface)}summary{cursor:pointer;padding:14px;font-weight:600}
details>pre{padding:16px;border-top:1px solid var(--border)}:focus-visible{outline:3px solid #7dd3fc;outline-offset:3px}
footer{margin-top:24px;color:var(--muted);font-size:12px}@media(max-width:700px){main{padding:24px 16px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.metric{padding:16px}.metric dd{font-size:28px}}
@media print{:root{color-scheme:light;--bg:#fff;--surface:#f5f7fa;--text:#152232;--muted:#425466;--border:#b8c4d2}.brand{color:#152232}main{padding:12px}.notice{background:#f5f7fa}.metric .working{color:#166534}.metric .idle{color:#334155}.table-wrap{overflow:visible}table{min-width:0}th{background:#e5ebf2}}
"""


def quality_statistics(report):
    """Time totals are available only for complete, structurally valid records.

    Comparison files are not counted again as extra videos. Any missing/invalid
    source makes totals unknown, rather than showing a deceptively clean zero.
    """
    records = report.get("results", [])
    complete = bool(records) and not report.get("input_errors")
    working = idle = 0.0
    for record in records:
        complete = complete and record.get("structure", {}).get("passed") is True
        doc = record.get("doc", {})
        for key in ("working", "idle"):
            segments = doc.get(key, [])
            if not isinstance(segments, list):
                complete = False
                continue
            for segment in segments:
                try:
                    start, end = segment["start"], segment["end"]
                    if isinstance(start, bool) or isinstance(end, bool):
                        raise ValueError("boolean interval")
                    start, end = float(start), float(end)
                    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                        raise ValueError("invalid interval")
                    if key == "working":
                        working += end - start
                    else:
                        idle += end - start
                except (ValueError, TypeError, KeyError, OverflowError):
                    complete = False
    # Finite per-video intervals can still overflow when aggregated across a
    # batch. Never render infinite totals or a NaN percentage as a clean result.
    complete = complete and math.isfinite(working) and math.isfinite(idle)
    all_records = records + report.get("comparison_results", [])
    errors = len(report.get("input_errors", [])) + sum(len(record.get("errors", [])) for record in all_records)
    pending = sum(record.get("status") == "REVIEW_REQUIRED" for record in all_records)
    if report.get("agreement") is not None and report["agreement"].get("gate_passed") is not True:
        pending += 1
    return {"videos": len(records), "working_seconds": working if complete else None,
            "idle_seconds": idle if complete else None, "time_totals_complete": complete,
            "structural_errors": errors, "review_items": pending,
            "measurement_scope": "submitted annotation intervals; comparison records excluded from time totals"}


def summary_markup(report):
    stats = quality_statistics(report)
    status = str(report["status"])
    status_class = "pass" if status == "PASS_SCOPED" else "review" if status == "REVIEW_REQUIRED" else "fail"
    duration = lambda value: "—" if value is None else f"{value:.1f} <small>s</small>"
    cards = (f'<dl class="metrics" aria-label="输入标注统计"><div class="metric"><dt>视频记录 / Videos</dt><dd>{stats["videos"]}</dd></div>'
             f'<div class="metric"><dt>Working time</dt><dd class="working">{duration(stats["working_seconds"])}</dd></div>'
             f'<div class="metric"><dt>Idle time</dt><dd class="idle">{duration(stats["idle_seconds"])}</dd></div>'
             f'<div class="metric"><dt>结构错误 / Errors</dt><dd>{stats["structural_errors"]}</dd></div></dl>')
    if stats["time_totals_complete"] and stats["working_seconds"] + stats["idle_seconds"] > 0:
        scale = max(stats["working_seconds"], stats["idle_seconds"])
        scaled_working, scaled_idle = stats["working_seconds"] / scale, stats["idle_seconds"] / scale
        ratio = scaled_working / (scaled_working + scaled_idle) * 100
        bar = (f'<div class="ratio" role="img" aria-label="Working {ratio:.1f}%, Idle {100-ratio:.1f}%">'
               f'<span class="ratio-working" style="width:{ratio:.6f}%"></span><span class="ratio-idle" style="width:{100-ratio:.6f}%"></span></div>'
               f'<div class="legend"><span>● Working {ratio:.1f}%</span><span>● Idle {100-ratio:.1f}%</span></div>')
    else:
        bar = '<p class="muted">时间汇总未生成：输入为空、不完整或未通过结构检查。</p>'
    return (f'<p><span class="status status-{status_class}">{escape(status)}</span> '
            f'<span class="muted">待复核项目 {stats["review_items"]}</span></p>' + cards +
            '<section class="panel"><h2 style="margin:0">Working / Idle 标注时间占比</h2>' + bar +
            '<p class="muted" style="font-size:12px">来自本批输入标注；不重复累计对照文件，不代表已观看视频验证动作。</p></section>')


def agreement_markup(report):
    """Keep comparison failures visible even when each file is well formed."""
    agreement = report.get("agreement")
    if agreement is None:
        return ""
    passed = agreement.get("gate_passed") is True
    minimum = agreement.get("minimum_kappa", .80)
    kappa = agreement.get("kappa")
    findings = []
    if kappa is None:
        findings.append("Kappa 不可计算：" + str(agreement.get("kappa_status", "UNKNOWN")))
    elif isinstance(kappa, (int, float)) and isinstance(minimum, (int, float)) and kappa < minimum:
        findings.append(f"Kappa {kappa} 低于要求 {minimum}")
    if agreement.get("coverage_complete") is False:
        findings.append("比较记录覆盖不完整；核对缺失和重复的视频 ID。")
    if agreement.get("semantic_complete") is False:
        findings.append("比较标签不完整、存在未判定内容或未匹配工作段。")
    if agreement.get("is_degenerate_single_class"):
        findings.append("单一类别或极端类别占比需要复核，不能以高时间一致性自动放行。")
    for key, label in (("missing_clips_a", "主输入缺少视频"), ("missing_clips_b", "对照缺少视频"),
                       ("duplicate_clips_a", "主输入重复视频"), ("duplicate_clips_b", "对照重复视频"),
                       ("validation_errors", "比较输入问题"), ("semantic_missing", "缺失标签"),
                       ("semantic_conflicts", "标签冲突"), ("unmatched_segments", "未匹配工作段"),
                       ("boundary_findings", "时间边界差异")):
        if agreement.get(key):
            findings.append(label + "：" + json.dumps(agreement[key], ensure_ascii=False, allow_nan=False))
    if not passed and not findings:
        findings.append("比较门禁未通过，请复核完整比较结果后仲裁。")
    state = "PASS_SCOPED" if passed else "REVIEW_REQUIRED"
    safe = lambda value: escape(str(value), quote=True)
    return ('<section class="panel" aria-labelledby="agreement-heading"><h2 id="agreement-heading" style="margin-top:0">对照比较门禁</h2>'
            f'<p><span class="status status-{"pass" if passed else "review"}">{state}</span></p>'
            f'<p>Kappa: <strong>{safe(kappa if kappa is not None else "不可计算")}</strong> · 最低要求: {safe(minimum)}'
            f' · 状态: {safe(agreement.get("kappa_status", "UNKNOWN"))}</p>'
            f'<p class="muted">共同视频 {safe(agreement.get("common_clips", "未知"))} · 样本 {safe(agreement.get("total_samples", "未知"))}'
            f' · 采样率 {safe(agreement.get("sample_rate_hz", "未记录"))} Hz</p>'
            + ('<ul>' + ''.join('<li><pre>' + safe(item) + '</pre></li>' for item in findings) + '</ul>' if findings else '<p>本次声明的比较范围通过。</p>')
            + '<p class="muted">时间一致性是诊断量；每份记录的结构检查通过不代表二者比较通过，也不证明视频真值。</p></section>')
