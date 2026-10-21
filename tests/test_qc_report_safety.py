from html.parser import HTMLParser

from trace_annotate.embodied.qc_report import generate_qc_report


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))

    def handle_data(self, data):
        self.text.append(data)


def test_annotation_text_is_not_executable_html(tmp_path):
    payload = '<script>window.reviewProbe=1</script><img src=x onerror="alert(1)">'
    docs = [{
        "clip": payload, "text_description": payload, "clip_status": payload,
        "working": [{"start": 0, "end": 1, "verdict": "错", "error_type": payload, "evidence": payload}],
    }]
    path = tmp_path / "report.html"
    generate_qc_report(docs, str(path))
    parser = ReportParser()
    parser.feed(path.read_text(encoding="utf-8"))
    assert not any(tag in {"script", "img"} for tag, _ in parser.tags)
    assert "".join(parser.text).count(payload) == 5


def test_unjudged_segments_are_not_reported_correct(tmp_path):
    path = tmp_path / "report.html"
    generate_qc_report([{"clip": "empty.mp4"}, {"clip": "unknown.mp4", "working": [{"start": 0, "end": 1, "verdict": "无法判断"}]}], str(path))
    report = path.read_text(encoding="utf-8")
    assert "全部正确" not in report
    assert report.count("待判定") == 2
    assert "100% 正确" not in report
    assert "段落完整性 100%" not in report


def test_user_text_does_not_expand_report_placeholders(tmp_path):
    path = tmp_path / "report.html"
    generate_qc_report([{"clip": "__CLIP_ROWS__", "working": [{"start": 0, "end": 1, "verdict": "错", "error_type": "x", "evidence": "__CLIP_ROWS__"}]}], str(path))
    report = path.read_text(encoding="utf-8")
    assert report.count("__CLIP_ROWS__") == 2
