# TraceAnnotate

Python tools for local media asset review and temporal annotation of embodied AI video.

[![CI](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml/badge.svg)](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | [中文说明](README_CN.md)

TraceAnnotate combines media staging, annotation provenance, and working/idle timeline utilities. It is an early-stage project: the automated tests and synthetic examples demonstrate specific behaviors, while adoption and performance at scale have not been established. Contributions and reproducible bug reports are welcome.

## What it does

| Area | Available tools |
| --- | --- |
| Media preparation | Batched staging, source/staging boundary checks, size and first-MiB MD5 copy checks, image dHash and duplicate clustering |
| Asset review | Brightness/sharpness metrics, configurable face/entity/screen heuristics, contact sheets and prioritized review queues |
| Provenance | Per-field `fill` origins, human overrides, revision history and rollback |
| Temporal annotation | Working/idle complements, expected-step reconciliation, interval validation and sampled temporal agreement |
| Review and exchange | Local browser timeline editor, HTML QC reports, Label Studio configuration/parsing, SRT, CVAT-style XML and action-label text exports |

The heuristics and quality thresholds are review aids, not ground truth. Label Studio “v2.1” in this repository refers to the project's annotation convention, not an official Label Studio product version. Export adapters are covered by local tests; compatibility with every external application or training pipeline has not been verified.

## Install

Use Python 3.9 or newer. See [CI](https://github.com/wyregey779-dot/trace-annotate/actions/workflows/ci.yml) for the versions currently tested.

```bash
git clone https://github.com/wyregey779-dot/trace-annotate.git
cd trace-annotate
python -m venv .venv
```

Activate the virtual environment (`.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS), then install:

```bash
python -m pip install -e ".[dev]"
trace-annotate --help
python -m pytest tests -q
```

The core depends on NumPy, Pillow and tqdm. Install optional components only for the functions you need:

| Extra or dependency | Purpose and limitations |
| --- | --- |
| `python -m pip install -e ".[cv]"` | OpenCV video/motion processing and optional HEIF image support; YuNet face detection also requires a compatible local ONNX model. |
| `python -m pip install -e ".[ocr]"` | RapidOCR integration; model/runtime availability depends on the installed package. |
| `python -m pip install -e ".[full]"` | Optional vision/OCR dependencies; larger install, and not a promise that model inference is complete. |
| `ffmpeg` executable on `PATH` | Required by the FFmpeg scene-cut helper; this executable is not installed by pip. |

**CLIP is unfinished.** The current adapter does not perform complete image/text inference. The `clip` extra installs PyTorch/torchvision, but does not install `cn_clip` or provide model weights. Do not use its output as a model prediction. See the [roadmap](docs/ROADMAP.md).

The core examples and timeline calculations run locally without an OpenAI API key. Installation and optional model packages may need network access. Prepare and verify all dependencies and weights before using the tools in an offline environment.

## Try the examples

```bash
python examples/demo_asset_pipeline.py
python examples/demo_embodied_pipeline.py
```

The asset example creates synthetic images and uses simulated face/CLIP fields to demonstrate the data flow; it does not measure detection accuracy. It creates and removes `examples/_demo_scratch`, so keep that directory free of your own files. The embodied example constructs synthetic annotations and reports idle intervals, step order and agreement.

For a minimal, model-free step check:

```python
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps

working = [
    {"start": 2.0, "end": 10.0, "step_number": 1},
    {"start": 15.0, "end": 20.0, "step_number": 3},
]
print(compute_idle_complement(working, total_duration=30.0))
# [{'start': 0.0, 'end': 2.0}, {'start': 10.0, 'end': 15.0},
#  {'start': 20.0, 'end': 30.0}]
result = diff_expected_steps(working, expected_step_count=3)
print(result["missing"])       # [2]
print(result["out_of_order"])  # []: no decreasing step transition
```

## Command line

These commands use your own annotation files; see the [annotation SOP](docs/EMBODIED_ANNOTATION_SOP.md) for the document structure.

```bash
# Validate an annotation document
trace-annotate embodied-validate clip.json

# Compute idle intervals, including the tail to the known video duration
trace-annotate embodied-idle clip.json --duration 60

# Compare steps with the expected procedure
trace-annotate embodied-diff clip.json --nsteps 5

# Compare two annotators using timeline samples
trace-annotate embodied-kappa annotator_A.jsonl annotator_B.jsonl --fps 2

# Generate a local review report or open the timeline editor
trace-annotate embodied-qc clip.json -o qc_report.html
trace-annotate embodied-ui

# Export the project's Label Studio configuration or subtitles
trace-annotate embodied-xml -o label_studio_config.xml
trace-annotate embodied-export clip.json --format srt -o clip.srt

# Check an asset catalog or create a contact sheet
trace-annotate asset-verify annotations.jsonl --inventory inventory.jsonl
trace-annotate asset-sheet thumb1.jpg thumb2.jpg -o audit_sheet.jpg --cols 4
```

`embodied-idle --inplace` overwrites its input; omit it to inspect JSON on stdout first. Supply the actual `--duration` to preserve idle time after the last working segment. Staging cleanup removes staged copies, so use a dedicated staging directory and retain source backups. Copy checks use file size and the first MiB, not a full-file cryptographic integrity check or a hardware write lock.

## Documentation and contributions

- [Architecture](docs/ARCHITECTURE.md)
- [Embodied annotation SOP](docs/EMBODIED_ANNOTATION_SOP.md)
- [Asset schema and provenance](docs/PROVENANCE_SCHEMA.md)
- [Visual audit workflow](docs/FAST_AUDIT_WORKFLOW.md)
- [Known limitations and roadmap](docs/ROADMAP.md)
- [Contribution guide](CONTRIBUTING.md)

Report bugs and discuss improvements in [GitHub Issues](https://github.com/wyregey779-dot/trace-annotate/issues). Include the command, package version, operating system and a small synthetic reproducer. Do not upload private footage or credentials.

## License

[MIT](LICENSE). External model weights, optional runtimes and datasets may have their own licenses.
