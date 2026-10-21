import os
import tempfile
from trace_annotate.embodied.qc_report import generate_qc_report


def test_qc_report():
    docs = [
        {
            "clip": "test_clip1.mp4",
            "text_description": "安装面板",
            "clip_status": "正常",
            "working": [
                {"start": 0.0, "end": 10.0, "step_number": 1, "verdict": "对"},
                {"start": 12.0, "end": 20.0, "step_number": 2, "verdict": "错", "error_type": "顺序错误", "evidence": "倒序"}
            ],
            "idle": [
                {"start": 10.0, "end": 12.0}
            ]
        }
    ]
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out_path = f.name

    try:
        res = generate_qc_report(docs, out_path)
        assert os.path.exists(res)
        with open(res, "r", encoding="utf-8") as fh:
            content = fh.read()
        assert "TraceAnnotate" in content
        assert "顺序错误" in content
        assert "test_clip1.mp4" in content
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
