"""
Automated Quality Control (QC) & Interactive HTML Inspection Report Generator.
Aggregates annotation metrics, error frequency, step distribution,
and inter-annotator agreement into a standalone visual report.
"""
from __future__ import annotations
import os
import json
import re
from html import escape
from typing import List, Dict, Any, Optional


QC_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TraceAnnotate 具身智能视频标注质量控制报告 (QC Report)</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { color: #38bdf8; font-size: 26px; border-bottom: 2px solid #334155; padding-bottom: 12px; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; }
    .card-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; }
    .card-value { font-size: 28px; font-weight: bold; color: #f1f5f9; }
    .card-sub { font-size: 12px; color: #64748b; margin-top: 4px; }
    .section-title { font-size: 18px; color: #38bdf8; margin: 24px 0 12px; }
    table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }
    th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }
    th { background: #0f172a; color: #94a3b8; }
    tr:hover { background: #24324a; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }
    .badge-success { background: #14532d; color: #4ade80; }
    .badge-error { background: #7f1d1d; color: #f87171; }
    .badge-idle { background: #334155; color: #cbd5e1; }
    .bar-container { background: #334155; border-radius: 4px; height: 16px; width: 100%; overflow: hidden; display: flex; }
    .bar-working { background: #22c55e; height: 100%; }
    .bar-idle { background: #64748b; height: 100%; }
  </style>
</head>
<body>
  <div class="container">
    <h1>⚡ TraceAnnotate 具身智能视频标注 QC 验收报告</h1>

    <div class="grid">
      <div class="card">
        <div class="card-title">质检视频总数</div>
        <div class="card-value">__TOTAL_CLIPS__</div>
        <div class="card-sub">仅统计输入标注，完整性请另行校验</div>
      </div>
      <div class="card">
        <div class="card-title">Working 工作总时长</div>
        <div class="card-value">__TOTAL_WORKING_S__s</div>
        <div class="card-sub">占比 __WORKING_RATIO__%</div>
      </div>
      <div class="card">
        <div class="card-title">Idle 空闲总时长</div>
        <div class="card-value">__TOTAL_IDLE_S__s</div>
        <div class="card-sub">占比 __IDLE_RATIO__%</div>
      </div>
      <div class="card">
        <div class="card-title">错误发生段数</div>
        <div class="card-value" style="color: __ERROR_COLOR__">__ERROR_COUNT__</div>
        <div class="card-sub">总工作段数 __TOTAL_WORKING_SEGS__</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
      <div class="card-title">全库 Working / Idle 时间轴能量分布</div>
      <div class="bar-container" style="margin-top: 10px;">
        <div class="bar-working" style="width: __WORKING_RATIO__%;"></div>
        <div class="bar-idle" style="width: __IDLE_RATIO__%;"></div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-top: 6px;">
        <span>🟢 Working 执行中 (__WORKING_RATIO__%)</span>
        <span>⚪ Idle 停顿等待 (__IDLE_RATIO__%)</span>
      </div>
    </div>

    <div class="section-title">📊 错误 7 子类统计分布</div>
    <table>
      <thead>
        <tr>
          <th>错误类别</th>
          <th>发生频次</th>
          <th>占比</th>
          <th>典型证据案例</th>
        </tr>
      </thead>
      <tbody>
        __ERROR_ROWS__
      </tbody>
    </table>

    <div class="section-title">📋 标注片段明细清单</div>
    <table>
      <thead>
        <tr>
          <th>视频片段 (Clip)</th>
          <th>任务描述</th>
          <th>片段诊断</th>
          <th>Working 段数</th>
          <th>判定结论</th>
        </tr>
      </thead>
      <tbody>
        __CLIP_ROWS__
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def generate_qc_report(
    docs: List[Dict[str, Any]],
    output_html_path: str
) -> str:
    """
    Generate a visual HTML Quality Control inspection report for a set of annotation docs.
    """
    total_clips = len(docs)
    total_working_s = 0.0
    total_idle_s = 0.0
    total_working_segs = 0
    error_count = 0

    error_stats: Dict[str, int] = {}
    error_samples: Dict[str, str] = {}

    clip_rows = []

    for d in docs:
        clip_name = escape(str(d.get("clip", "video.mp4")))
        desc = escape(str(d.get("text_description", "")))
        status = escape(str(d.get("clip_status", "未标注")))
        working = d.get("working", [])
        idle = d.get("idle", [])

        total_working_segs += len(working)
        w_dur = sum([float(w["end"]) - float(w["start"]) for w in working])
        i_dur = sum([float(i["end"]) - float(i["start"]) for i in idle])
        total_working_s += w_dur
        total_idle_s += i_dur

        has_error = False
        for w in working:
            if w.get("verdict") == "错":
                has_error = True
                error_count += 1
                etype = w.get("error_type", "未分类错误")
                error_stats[etype] = error_stats.get(etype, 0) + 1
                if etype not in error_samples and w.get("evidence"):
                    error_samples[etype] = w["evidence"]

        if has_error:
            verd_badge = '<span class="badge badge-error">含错误段</span>'
        elif working and all(w.get("verdict") == "对" for w in working):
            verd_badge = '<span class="badge badge-success">全部正确</span>'
        else:
            verd_badge = '<span class="badge badge-idle">待判定</span>'
        status_badge = f'<span class="badge badge-success">{status}</span>' if status == "正常" else f'<span class="badge badge-idle">{status}</span>'

        clip_rows.append(f"""
        <tr>
          <td><b>{clip_name}</b></td>
          <td>{desc}</td>
          <td>{status_badge}</td>
          <td>{len(working)} 段 ({w_dur:.1f}s)</td>
          <td>{verd_badge}</td>
        </tr>
        """)

    total_time = max(0.1, total_working_s + total_idle_s)
    w_ratio = round((total_working_s / total_time) * 100, 1)
    i_ratio = round((total_idle_s / total_time) * 100, 1)

    error_rows = []
    if error_stats:
        for etype, cnt in sorted(error_stats.items(), key=lambda x: -x[1]):
            pct = round((cnt / max(1, error_count)) * 100, 1)
            sample = error_samples.get(etype, "无详细证据文本")
            error_rows.append(f"""
            <tr>
              <td><span class="badge badge-error">{escape(str(etype))}</span></td>
              <td><b>{cnt}</b> 次</td>
              <td>{pct}%</td>
              <td style="color: #cbd5e1;">{escape(str(sample))}</td>
            </tr>
            """)
    else:
        error_rows.append("<tr><td colspan='4' style='text-align:center; color:#94a3b8;'>暂无错误记录（不代表已全部判定）</td></tr>")

    html_content = QC_HTML_TEMPLATE
    html_content = html_content.replace("__TOTAL_CLIPS__", str(total_clips))
    html_content = html_content.replace("__TOTAL_WORKING_S__", str(round(total_working_s, 1)))
    html_content = html_content.replace("__TOTAL_IDLE_S__", str(round(total_idle_s, 1)))
    html_content = html_content.replace("__WORKING_RATIO__", str(w_ratio))
    html_content = html_content.replace("__IDLE_RATIO__", str(i_ratio))
    html_content = html_content.replace("__ERROR_COUNT__", str(error_count))
    html_content = html_content.replace("__TOTAL_WORKING_SEGS__", str(total_working_segs))
    html_content = html_content.replace("__ERROR_COLOR__", "#f87171" if error_count > 0 else "#4ade80")
    # Substitute row placeholders once: user text may itself contain placeholder names.
    rows = {"__ERROR_ROWS__": "\n".join(error_rows), "__CLIP_ROWS__": "\n".join(clip_rows)}
    html_content = re.sub(r"__ERROR_ROWS__|__CLIP_ROWS__", lambda match: rows[match.group()], html_content)

    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path
