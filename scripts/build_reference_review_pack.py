"""Extract exact boundary frames for reviewing existing reference candidates.

The pack is read-only evidence. It does not decide transition types, certify
ground truth, or turn an assistant's observations into human review.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import build_reference_truth as R


MAX_EVENTS = 128
MAX_FRAMES = 200_000
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MIN_FREE_RESERVE = 2 * 1024 * 1024 * 1024
MAX_PAGE_BYTES = 8 * 1024 * 1024
SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b")

REVIEW_CSS = """:root{color-scheme:dark;font:16px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif;background:#0b1220;color:#e7edf6}
*{box-sizing:border-box}body{margin:0}a{color:#9bd9ff}a:focus-visible,summary:focus-visible{outline:3px solid #f7c964;outline-offset:3px}
.skip{position:absolute;left:1rem;top:-4rem;background:#fff;color:#111;padding:.5rem;z-index:2}.skip:focus{top:1rem}
header,main,footer{max-width:1400px;margin:auto;padding:.7rem clamp(1rem,3vw,2rem)}header{border-bottom:1px solid #34455e;color:#a9bad2}
h1,h2{line-height:1.25}h1{margin:.2rem 0 .65rem}h2{font-size:1.25rem;margin:0 0 .35rem}
.warning{border:2px solid #e2ad52;background:#392a14;color:#fff2d2;padding:.7rem;border-radius:.6rem;font-weight:700}
.intro{max-width:85ch;margin:.65rem 0}.sources,.event{background:#151f30;border:1px solid #33445d;border-radius:.7rem;padding:.8rem;margin:.7rem 0}
.sources summary,.metadata summary{cursor:pointer;min-height:2rem}.brief{margin:.3rem 0;color:#d2e5f9}
dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.25rem 1rem;margin:.5rem 0}dt{color:#a9bad2}dd{margin:0;min-width:0;overflow-wrap:anywhere}
code{font:0.88em/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
small{font-size:.94em;color:#c8d4e8}
.toc{margin:.6rem 0}.toc summary{cursor:pointer;padding:.35rem;min-height:2rem}.toc ol{columns:3;max-height:12rem;overflow:auto;padding-left:2rem}
.frames{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin-top:1rem}figure{min-width:0;margin:0;background:#0a101c;border:1px solid #34455e;border-radius:.5rem;overflow:hidden}
figure img{display:block;width:100%;height:auto;max-height:60vh;object-fit:contain;background:#05080f}figcaption{padding:.75rem;overflow-wrap:anywhere}
.missing{padding:1rem;border:1px dashed #a9bad2;border-radius:.5rem}footer{color:#a9bad2;border-top:1px solid #34455e}
@media(max-width:720px){.frames{grid-template-columns:1fr}.toc ol{columns:1}dl{grid-template-columns:1fr;gap:0}dt{margin-top:.4rem}figure img{max-height:none}}
@media(forced-colors:active){.warning,.sources,.event,figure{border:1px solid CanvasText}}
"""


def _escaped(value: object) -> str:
    if value is None:
        value = "—"
    elif isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value), quote=True)


def _time_markup(value: dict, unit: str) -> str:
    """Show a readable approximation alongside the exact rational timestamp."""
    fraction = R.rational(value)
    if fraction.denominator <= 0:
        raise ValueError("invalid rational time value in review page")
    exact = (str(fraction.numerator) if fraction.denominator == 1
             else f"{fraction.numerator}/{fraction.denominator}")
    approximate = Decimal(fraction.numerator) / Decimal(fraction.denominator)
    precision = 9 if unit == "秒/刻度" else 6
    return (f'<code>{_escaped(exact)}</code> {unit} '
            f'<small>(约 {approximate:.{precision}f} {unit})</small>')


def _frame_html(frame: dict | None, role: str, dimensions: dict, video_role: str, *, lazy: bool) -> str:
    if frame is None:
        return f'<p class="missing">{role}：没有可核对的精确帧。</p>'
    index = frame["frame_index"]
    path = frame["path"]
    if (type(index) is not int or not 0 <= index < MAX_FRAMES
            or path != f"frame-index-{index:06d}.png"):
        raise ValueError("review page image path is not a generated frame name")
    width, height = dimensions["width"], dimensions["height"]
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 16384 or not 1 <= height <= 16384):
        raise ValueError("review page has invalid image dimensions")
    loading = ' loading="lazy"' if lazy else ''
    return (f'<figure><img src="{path}" alt="{role}，第 {index} 帧；画面内容需人工观察" '
            f'width="{width}" height="{height}"{loading}>'
            f'<figcaption><strong>{role} · #{index}</strong>'
            f'<dl><dt>{video_role} PTS</dt><dd><code>{_escaped(frame["pts"])}</code> 刻度</dd>'
            f'<dt>{video_role} PTS 时间基</dt><dd>{_time_markup(frame["time_base"], "秒/刻度")}</dd>'
            f'<dt>标注时间轴秒数</dt><dd>{_time_markup(frame["seconds"], "秒")}</dd>'
            f'<dt>PNG SHA-256</dt><dd><code>{_escaped(frame["sha256"])}</code></dd>'
            f'</dl></figcaption></figure>')


def _render_review_html(manifest: dict) -> str:
    events = manifest["events"]
    role_names = {"source_video": "源视频", "prepared_playback_copy": "播放副本"}
    video_role = role_names.get(manifest["decoded_video_role"])
    if video_role is None:
        raise ValueError("unknown decoded video role in review page")
    out = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
           '<meta name="referrer" content="no-referrer">',
           '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'self\'; base-uri \'none\'; form-action \'none\'">',
           '<link rel="stylesheet" href="review.css">',
           '<title>边界帧审阅包 · TraceAnnotate</title></head><body>',
           '<a class="skip" href="#main-content">跳到主要内容</a>',
           '<header>TraceAnnotate · Reference review pack</header><main>',
           '<h1 id="main-content" tabindex="-1">边界帧审阅包</h1>',
           '<p class="warning">未人工复核 · 没有签核 · 不是真值。此页只读，不能接受、改标或拒绝候选。</p>',
           f'<p class="intro">下列图片来自实际解码的{video_role}，用于人工核对相邻画面和来源。PTS 是{video_role}帧时间戳；标注时间轴秒数按项目时间原点换算，两者可能不同。静帧不能单独判定渐变转场或完整声音；还需回到原视频核对，并另行记录与签核决定。</p>',
           f'<details class="sources"><summary>来源与范围 · {_escaped(manifest["reference_source_id"])} · 已核对 {_escaped(manifest["actual_decoded_frames_verified"])} 帧 · {len(events)} 个候选</summary><dl>',
           f'<dt>视频 SHA-256</dt><dd><code>{_escaped(manifest["video_sha256"])}</code></dd>',
           f'<dt>标注 SHA-256</dt><dd><code>{_escaped(manifest["annotation_sha256"])}</code></dd>',
           f'<dt>帧图 SHA-256</dt><dd><code>{_escaped(manifest["frame_map_sha256"])}</code></dd>',
           f'<dt>候选 SHA-256</dt><dd><code>{_escaped(manifest["reference_sha256"])}</code></dd>',
           f'<dt>已核对解码帧</dt><dd>{_escaped(manifest["actual_decoded_frames_verified"])}</dd>',
           f'<dt>候选事件</dt><dd>{len(events)}</dd>',
           '</dl></details>',
           f'<nav aria-label="事件目录"><details class="toc"><summary>事件目录（{len(events)}）</summary><ol>']
    for number, event in enumerate(events, 1):
        out.append(f'<li><a href="#event-{number}">事件 {number}：{_escaped(event["annotation_id"])}</a></li>')
    out.append('</ol></details></nav>')
    for number, event in enumerate(events, 1):
        out.extend([f'<article class="event" id="event-{number}">',
                    f'<h2>事件 {number} · {_escaped(event["annotation_id"])}</h2>',
                    f'<p class="brief">声明类型：{_escaped(event["declared_kind"])} · 不可入评理由 {len(event["ineligible_reasons"])} 项 · 未人工复核</p>',
                    '<div class="frames">',
                    _frame_html(event["before"], "前一帧", manifest["actual_dimensions"], video_role, lazy=number > 1),
                    _frame_html(event["current"], "当前帧", manifest["actual_dimensions"], video_role, lazy=number > 1),
                    '</div><details class="metadata"><summary>身份、标注与来源详情</summary><dl>',
                    f'<dt>事件 ID</dt><dd><code>{_escaped(event["event_id"])}</code></dd>',
                    f'<dt>轨道 / 标签</dt><dd>{_escaped(event.get("annotation_track"))} / {_escaped(event.get("annotation_label"))}</dd>',
                    f'<dt>声明类型</dt><dd>{_escaped(event["declared_kind"])}</dd>',
                    f'<dt>不可入评原因</dt><dd>{_escaped(event["ineligible_reasons"])}</dd>',
                    f'<dt>帧身份依据</dt><dd>{_escaped(event["frame_identity_basis"])}</dd>',
                    f'<dt>来源记录</dt><dd><code>{_escaped(event["source_provenance"])}</code></dd>',
                    '<dt>复核状态</dt><dd>未人工复核</dd>',
                    '</dl></details></article>'])
    out.extend(['</main><footer>只读证据包；不能代替原片、人工裁决或人工签核。</footer></body></html>'])
    page = '\n'.join(out) + '\n'
    if len(page.encode('utf-8')) > MAX_PAGE_BYTES:
        raise ValueError("review page exceeds bounded output size")
    return page


def _select_expression(indices: list[int]) -> str:
    ranges = []
    first = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((first, previous))
            first = index
        previous = index
    ranges.append((first, previous))
    terms = [f"eq(n\\,{start})" if start == end else f"between(n\\,{start}\\,{end})"
             for start, end in ranges]
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i+1]})" if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return "select=" + terms[0]


def _run(command: list[str], stdout: Path, stderr: Path, timeout: int,
         max_stdout: int = 32 * 1024 * 1024) -> None:
    started = time.monotonic()
    with stdout.open("xb") as output, stderr.open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors)
        try:
            while process.poll() is None:
                if stdout.stat().st_size > max_stdout:
                    raise ValueError("frame probe output exceeds the bounded evidence size")
                if time.monotonic() - started > timeout:
                    raise ValueError(f"decoder exceeded {timeout}s: {command[0]}")
                time.sleep(0.05)
            if stdout.stat().st_size > max_stdout:
                raise ValueError("frame probe output exceeds the bounded evidence size")
            if process.returncode:
                raise ValueError(f"decoder failed with exit {process.returncode}: {command[0]}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def _run_images(command: list[str], work: Path, timeout: int) -> None:
    """Stop the owned decoder before its PNGs exhaust the output budget."""
    started = time.monotonic()
    with (work / "ffmpeg.stdout").open("xb") as output, (work / "ffmpeg.stderr").open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors)
        try:
            while process.poll() is None:
                image_bytes = sum(path.stat().st_size for path in work.glob("decoded-*.png"))
                if image_bytes > MAX_IMAGE_BYTES or shutil.disk_usage(work).free < MIN_FREE_RESERVE:
                    raise ValueError("review images exceeded the byte budget or 2 GiB disk reserve")
                if time.monotonic() - started > timeout:
                    raise ValueError(f"decoder exceeded {timeout}s: {command[0]}")
                time.sleep(0.05)
            if process.returncode:
                raise ValueError(f"decoder failed with exit {process.returncode}: {command[0]}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def _actual_frame_pts(video: Path, rows: list[dict], work: Path, timeout: int,
                      selected_count: int) -> tuple[list[int], list[list[str]], dict]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValueError("ffprobe is required")
    streams = [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
               "-show_streams", "-of", "json", str(video)]
    _run(streams, work / "ffprobe-streams.json", work / "ffprobe-streams.stderr", timeout)
    document = json.loads((work / "ffprobe-streams.json").read_text(encoding="utf-8"))
    video_streams = [entry for entry in document.get("streams", [])
                     if entry.get("codec_type") == "video"]
    if len(video_streams) != 1 or R.rational(video_streams[0]["time_base"]) != R.rational(rows[0]["time_base"]):
        raise ValueError("actual video stream or time base disagrees with the supplied frame map")
    width = R.integer(video_streams[0]["width"])
    height = R.integer(video_streams[0]["height"])
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise ValueError("invalid actual video dimensions")
    estimated_bytes = selected_count * width * height * 4
    if estimated_bytes > MAX_IMAGE_BYTES:
        raise ValueError("selected frames exceed the uncompressed image budget; review fewer candidates")
    frames = [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
              "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=pts",
              "-of", "json", str(video)]
    output = work / "ffprobe-frame-pts.json"
    _run(frames, output, work / "ffprobe-frames.stderr", timeout)
    if output.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("decoded frame probe exceeds the bounded evidence size")
    frame_entries = json.loads(output.read_text(encoding="utf-8")).get("frames", [])
    if len(frame_entries) > MAX_FRAMES:
        raise ValueError("decoded frame count exceeds review-pack limit")
    pts = []
    for entry in frame_entries:
        value = entry.get("pts")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("actual decoded frame has no integer PTS")
        pts.append(value)
    if len(pts) != len(rows) or any(value != row["pts"] for value, row in zip(pts, rows)):
        raise ValueError("actual decoded frame PTS or count disagrees with supplied frame map")
    return pts, [streams, frames], {"width": width, "height": height,
                                   "estimated_uncompressed_bytes": estimated_bytes}


def build_pack(reference_path: Path, output: Path, *, timeout: int = 300) -> dict:
    """Create a new evidence directory; on error retain it with FAILED.json."""
    reference_path = reference_path.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise ValueError("review-pack output already exists")
    if not output.parent.is_dir():
        raise ValueError("review-pack parent directory does not exist")
    if shutil.disk_usage(output.parent).free < MIN_FREE_RESERVE + MAX_IMAGE_BYTES:
        raise ValueError("insufficient free disk for bounded images and 2 GiB reserve")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_sha256 = R.sha256(reference_path)
    if reference.get("schema") != "trace-annotate.reference-candidates.v1":
        raise ValueError("unsupported reference-candidates schema")
    annotation = Path(reference["annotation"]["path"]).resolve(strict=True)
    video = Path(reference["video"]["path"]).resolve(strict=True)
    frame_map = Path(reference["frame_map"]["path"]).resolve(strict=True)
    if len(reference.get("events", [])) > MAX_EVENTS:
        raise ValueError(f"review pack accepts at most {MAX_EVENTS} candidate events")
    regenerated = R.build_reference(annotation, video, frame_map,
                                    reference["frame_map"]["declared_source_sha256"],
                                    reference["source_id"], reference["split"])
    if regenerated != reference:
        raise ValueError("reference candidates differ from current annotation or frame map")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    _, _, media, _, layout = R.annotation_parts(payload)
    rows = R.load_frame_map(frame_map, media, layout)
    if len(rows) > MAX_FRAMES:
        raise ValueError("decoded frame map exceeds review-pack limit")
    selected = set()
    for event in reference["events"]:
        boundary = event["frame_index"]
        if boundary is not None:
            selected.update(index for index in (boundary - 1, boundary) if index >= 0)
    indices = sorted(selected)
    if not indices:
        raise ValueError("reference has no exact frame boundary to review")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required")
    output.mkdir(exist_ok=False)
    try:
        pts, probe_commands, dimensions = _actual_frame_pts(video, rows, output,
                                                             timeout, len(indices))
        filter_expression = _select_expression(indices) + ",showinfo"
        command = [ffmpeg, "-hide_banner", "-nostdin", "-n", "-xerror",
                   "-err_detect", "explode", "-protocol_whitelist", "file,pipe",
                   "-threads", "1", "-filter_threads", "1", "-copyts", "-i", str(video),
                   "-map", "0:v:0", "-vf", filter_expression,
                   "-fps_mode", "passthrough", "-c:v", "png", "-pix_fmt", "rgb24",
                   "-threads:v", "1", "-frames:v", str(len(indices)),
                   "-f", "image2", str(output / "decoded-%06d.png")]
        _run_images(command, output, timeout)
        emitted = [int(match.group(2)) for line in (output / "ffmpeg.stderr").read_text(
            encoding="utf-8", errors="replace").splitlines()
                   if (match := SHOWINFO.search(line))]
        produced = sorted(output.glob("decoded-*.png"))
        if len(produced) != len(indices) or emitted != [pts[index] for index in indices]:
            raise ValueError("FFmpeg selected frame count or PTS disagrees with actual decoded map")
        images = {}
        total_bytes = 0
        for index, file in zip(indices, produced):
            with file.open("rb") as stream:
                header = stream.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                raise ValueError("decoded output is not PNG")
            size = file.stat().st_size
            total_bytes += size
            if total_bytes > MAX_IMAGE_BYTES:
                raise ValueError("review images exceed the bounded output budget")
            named = output / f"frame-index-{index:06d}.png"
            file.rename(named)
            images[index] = {"path": named.name, "sha256": R.sha256(named),
                             "size_bytes": size, "frame_index": index,
                             "pts": pts[index], "time_base": rows[index]["time_base"],
                             "seconds": rows[index]["seconds"]}
        if (R.sha256(video) != reference["video"]["sha256"]
                or R.sha256(frame_map) != reference["frame_map"]["sha256"]
                or R.sha256(annotation) != reference["annotation"]["sha256"]
                or R.sha256(reference_path) != reference_sha256):
            raise ValueError("source bytes changed while review frames were decoded")
        events = []
        for event in reference["events"]:
            index = event["frame_index"]
            events.append({"event_id": event["event_id"], "annotation_id": event["annotation_id"],
                           "declared_kind": event["kind"], "ineligible_reasons": event["ineligible_reasons"],
                           "annotation_track": event["source_annotation"].get("track"),
                           "annotation_label": event["source_annotation"].get("label"),
                           "source_provenance": event["source_annotation"].get("provenance"),
                           "review_status": "unreviewed", "human_review": False,
                           "before": images.get(index - 1) if index is not None else None,
                           "current": images.get(index) if index is not None else None,
                           "frame_identity_basis": event["frame_identity_basis"]})
        manifest = {"schema": "trace-annotate.reference-review-pack.v1",
                    "reference_source_id": reference["source_id"], "reference_split": reference["split"],
                    "decoded_video_role": "source_video" if layout == "professional" else "prepared_playback_copy",
                    "reference_sha256": reference_sha256,
                    "video_sha256": reference["video"]["sha256"],
                    "frame_map_sha256": reference["frame_map"]["sha256"],
                    "annotation_sha256": reference["annotation"]["sha256"],
                    "actual_decoded_frames_verified": len(pts),
                    "actual_dimensions": dimensions,
                    "scope": "observed boundary frames only; no transition adjudication or human signoff",
                    "human_review": False, "holdout_sealed": False,
                    "commands": probe_commands + [command], "total_image_bytes": total_bytes,
                    "events": events}
        page = _render_review_html(manifest)
        pending_css = output / "review.css.incomplete"
        pending_page = output / "index.html.incomplete"
        pending_manifest = output / "manifest.json.incomplete"
        pending_css.write_text(REVIEW_CSS, encoding="utf-8", newline="\n")
        pending_page.write_text(page, encoding="utf-8", newline="\n")
        manifest["review_page"] = {"path": "index.html", "sha256": R.sha256(pending_page),
                                   "stylesheet": {"path": "review.css", "sha256": R.sha256(pending_css)},
                                   "scope": "read-only; no human adjudication"}
        with pending_manifest.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        pending_css.rename(output / "review.css")
        pending_page.rename(output / "index.html")
        pending_manifest.rename(output / "manifest.json")
        return manifest
    except Exception as exc:
        # A failure after any asset was promoted must not leave an apparently
        # complete review page. Preserve the bytes under explicitly incomplete names.
        cleanup_errors = []
        for name in ("manifest.json", "index.html", "review.css"):
            final = output / name
            if final.exists():
                try:
                    final.rename(output / f"{name}.incomplete")
                except OSError as cleanup_exc:
                    cleanup_errors.append(f"{name}: {cleanup_exc}")
        (output / "FAILED.json").write_text(json.dumps({"error": type(exc).__name__,
            "reason": str(exc), "cleanup_errors": cleanup_errors,
            "scope": "incomplete review pack; not usable as evidence"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.timeout <= 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")
        result = build_pack(args.reference, args.out, timeout=args.timeout)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"review pack failed: {exc}\n")
    print(json.dumps({"out": str(args.out), "events": len(result["events"]),
                      "actual_decoded_frames_verified": result["actual_decoded_frames_verified"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
