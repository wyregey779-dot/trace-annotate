"""Serve the two existing editors through a fixed resource allowlist.

Original package resources stay unchanged. The response adds only navigation
and script nonces; it does not attach a workspace token or migrate documents.
"""
from html import escape
from importlib import resources
import re


CLASSIC_DOCUMENTS = {
    "/tools/general/": ("temporal", "通用视频标注"),
    "/tools/embodied/": ("embodied", "具身视频标注"),
}
HASH_SCRIPT = "/tools/general/media-hash.js"


def classic_resource(path, nonce):
    package = resources.files("trace_annotate")
    if path == HASH_SCRIPT:
        return (package.joinpath("temporal", "ui", "media-hash.js").read_bytes(),
                "application/javascript; charset=utf-8")
    if path not in CLASSIC_DOCUMENTS:
        return None
    folder, title = CLASSIC_DOCUMENTS[path]
    html = package.joinpath(folder, "ui", "annotator.html").read_text(encoding="utf-8")
    recovery = ('继续旧成果请重新选择匹配视频并导入原 JSON；新地址不会自动读取旧文件页草稿。'
                if folder == "temporal" else
                '此具身页面支持新建与导出 JSON，尚无 JSON 导入恢复入口；请保留既有文件。')
    # Only trusted, bundled script tags are augmented. User annotation text is
    # neither interpolated into this document nor rendered as HTML.
    html = re.sub(r"<script\b", '<script nonce="' + escape(nonce, quote=True) + '"', html)
    nav = ('<nav aria-label="标注工具" style="padding:12px 20px;background:#10221e;'
           'color:#e9eff8;font:13px system-ui;line-height:1.8">'
           '<strong>' + title + '</strong> · '
           '<a style="color:#9ae6ca" href="/tools/general/" target="_blank" rel="noopener">通用视频标注（新窗口）</a> · '
           '<a style="color:#9ae6ca" href="/tools/embodied/" target="_blank" rel="noopener">具身视频标注（新窗口）</a> · '
           '<a style="color:#9ae6ca" href="/" target="_blank" rel="noopener">批量工作台（试验功能，新窗口）</a>'
           '<p style="margin:4px 0 0">此页使用原有独立文件流程，与批量工作区分别保存。'
           + recovery +
           '结束前请导出标注主文件。</p></nav>')
    html = html.replace("<body>", "<body>\n" + nav, 1)
    return html.encode("utf-8"), "text/html; charset=utf-8"
