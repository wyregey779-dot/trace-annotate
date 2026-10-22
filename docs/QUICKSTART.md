# Quickstart: annotation JSON to a review report

Process a synthetic 12-second task annotation, calculate idle intervals, validate its steps and export a review report. No video file, model weights or API key are required.

For the Chinese overview and runnable commands, see [中文说明](../README_CN.md#从一个完整示例开始).

## 1. Install from a source checkout

Use Python 3.9 or newer and Git:

```bash
git clone --branch v1.0.2 --depth 1 https://github.com/wyregey779-dot/trace-annotate.git
cd trace-annotate
python -m venv .venv
```

Activate the environment:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Linux/macOS shell | `source .venv/bin/activate` |

Then install:

```bash
python -m pip install .
trace-annotate --version
```

The version command should report `trace-annotate 1.0.2`. If PowerShell policy blocks activation, use `.venv\Scripts\python.exe -m pip install .`. Run the example through `.venv\Scripts\python.exe`, or invoke CLI commands through `.venv\Scripts\trace-annotate.exe`; changing the machine's execution policy is not required.

## 2. Run the workflow

From the repository root:

```bash
python examples/run_quickstart.py --output output/quickstart
```

The script copies [assembly.json](../examples/annotations/assembly.json) into a new directory, runs the actual CLI commands and checks the resulting intervals and subtitles. It refuses any existing destination, including an empty directory. Choose another path when rerunning it.

The last output should include:

```text
Verified: 2 working segments, 3 idle intervals, 5 subtitle entries.
```

| Output | Purpose |
| --- | --- |
| `output/quickstart/assembly.json` | Annotation document with calculated idle intervals |
| `output/quickstart/qc-report.html` | Local review report; open in a browser |
| `output/quickstart/assembly.srt` | Timed subtitles for inspection or review with a matching video |

![QC report from the quickstart fixture](images/quickstart-qc.jpg)

## 3. Inspect the example data

The fixture contains two working segments:

| Interval | Step | Verdict |
| --- | --- | --- |
| 2–5 seconds | 1 | `对` (correct) |
| 7–10 seconds | 2 | `对` (correct) |

The document duration is 12 seconds. Its labels are synthetic examples, not observations of a recorded participant or model predictions. The `clip` field identifies an example video; a matching video is not needed for these data-processing commands.

After idle calculation, the document contains:

```json
[
  {"start": 0.0, "end": 2.0},
  {"start": 5.0, "end": 7.0},
  {"start": 10.0, "end": 12.0}
]
```

Validation reports one checked record and `Violations: 0`. Step comparison returns empty `missing` and `out_of_order` lists. The SRT file contains five chronological entries: three idle intervals and two working intervals. Its second entry is:

```text
2
00:00:02,000 --> 00:00:05,000
【步骤 1 · 对】(正在执行任务动作)
```

Validation checks document structure and annotation rules. It cannot determine whether an annotator chose the correct action label from the video.

## 4. Run each command yourself

For a separate manual run, create a new directory and copy the fixture. This command fails if `output/manual-example` already exists:

```bash
python -c "from pathlib import Path; import shutil; Path('output/manual-example').mkdir(parents=True, exist_ok=False); shutil.copyfile('examples/annotations/assembly.json', 'output/manual-example/assembly.json')"
```

Then run:

```bash
trace-annotate embodied-idle output/manual-example/assembly.json --inplace
trace-annotate embodied-validate output/manual-example/assembly.json
trace-annotate embodied-diff output/manual-example/assembly.json --nsteps 2
trace-annotate embodied-qc output/manual-example/assembly.json -o output/manual-example/qc-report.html
trace-annotate embodied-export output/manual-example/assembly.json --format srt -o output/manual-example/assembly.srt
```

`--inplace` updates only the copied annotation file here. Omit it to print generated JSON instead. With your own records, preserve the document's actual duration or pass it using `--duration`; otherwise the final idle interval may be missing.

The exact verdict values are `对`, `错` and `无法判断`. Use `无法判断` for unreviewed work. A segment marked `错` needs a supported `error_type` and non-empty evidence. Label conventions are described in the [annotation SOP](EMBODIED_ANNOTATION_SOP.md).

## 5. Use your own video

```bash
trace-annotate embodied-ui
```

Select a browser-supported local video, mark task intervals, assign steps and review verdicts. Export JSON and run `embodied-validate` on the downloaded file before using it downstream. Export your annotations before changing the video: the editor resets its state when new media is loaded.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `trace-annotate` is not found | Activate the environment where you installed it, or invoke the executable from that environment directly |
| Fixture path is missing | Run from a source checkout's root; the installed Python package does not contain this example folder |
| The example refuses the output path | Choose a new path; the script does not overwrite existing directories |
| Invalid verdict | Use `对`, `错` or `无法判断`, rather than translations such as `correct` or `正确` |
| Trailing idle time is missing | Set the actual clip duration; it must be positive and at least the last working end time |
| The browser cannot play a video | Check container/codec support in that browser; installing OpenCV does not add browser codecs |
| Your record fails validation | Read the reported violations, fix a copy and rerun |

For reproducible bug reports, include the package/Python/OS versions, command, expected output and a small synthetic fixture in [GitHub Issues](https://github.com/wyregey779-dot/trace-annotate/issues).
