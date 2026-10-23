"""Exercise new commands through the public parser and a real child process."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trace_annotate.cli import main


FIXTURE = Path(__file__).resolve().parents[1] / "examples/annotations/general-video.json"


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["trace-annotate", *map(str, args)])
    main()


def test_general_json_and_subtitle_workflow(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, "-m", "trace_annotate.cli", *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8", check=True)

    assert "4 segments, 2 tracks" in run("video-validate", FIXTURE).stdout
    subtitle = tmp_path / "subtitles.vtt"
    run("video-export", FIXTURE, "--format", "vtt", "--track", "字幕", "-o", subtitle)
    imported = tmp_path / "imported.json"
    run("video-import", subtitle, "--format", "vtt", "--clip", "general-demo.mp4",
        "--duration", "12", "-o", imported)
    result = json.loads(imported.read_text(encoding="utf-8"))
    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [
        (2.0, 4.0, "第一句示例字幕。"), (6.0, 9.0, "第二句示例字幕。\n允许连续多行文字。")]
    assert all(s["review_status"] == "unreviewed" for s in result["segments"])
    run("video-validate", imported)


@pytest.mark.parametrize("format_name", ["json", "csv", "srt", "vtt"])
def test_existing_output_is_preserved(tmp_path, monkeypatch, capsys, format_name):
    output = tmp_path / "existing"
    output.write_bytes(b"keep me")
    with pytest.raises(SystemExit) as result:
        invoke(monkeypatch, "video-export", FIXTURE, "--format", format_name,
               *( ["--track", "字幕"] if format_name in {"srt", "vtt"} else []), "-o", output)
    assert result.value.code == 1
    assert output.read_bytes() == b"keep me"
    assert "Error:" in capsys.readouterr().err


def test_ambiguous_track_and_bad_import_do_not_create_output(tmp_path, monkeypatch):
    output = tmp_path / "no-output.srt"
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "video-export", FIXTURE, "--format", "srt", "-o", output)
    assert not output.exists()
    invalid = tmp_path / "bad.srt"
    invalid.write_text("not a subtitle", encoding="utf-8")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "video-import", invalid, "--format", "srt", "--clip", "video.mp4",
               "--duration", "12", "-o", output)
    assert not output.exists()


def test_json_export_preserves_all_fields(tmp_path, monkeypatch):
    output = tmp_path / "copy.json"
    invoke(monkeypatch, "video-export", FIXTURE, "--format", "json", "-o", output)
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_ui_print_path_does_not_open_browser(monkeypatch, capsys):
    monkeypatch.setattr("trace_annotate.temporal.commands.webbrowser.open", lambda _: pytest.fail("opened browser"))
    invoke(monkeypatch, "video-ui", "--print-path")
    assert Path(capsys.readouterr().out.strip()).is_file()
