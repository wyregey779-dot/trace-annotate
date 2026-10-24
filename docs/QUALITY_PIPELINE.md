# Annotation quality pipeline

This development-stage workflow checks submitted annotation files and reports unresolved review gates. General video events continue to use `video-validate`; the procedure and verdict rules here apply to embodied annotations.

## Run a complete example

From the current checkout, install `python -m pip install -e ".[dev]"`, then run:

```bash
trace-annotate embodied-pipeline examples/annotations/quality-batch.json --report output/quality-new/report.html
```

The fixture declares its expected two-step procedure and twelve-second duration. It is a synthetic format test, with no associated video or independent human annotation. A successful report says `PASS_SCOPED`, not that video content is correct. Choose a new report path for every run.

For your own exported file:

```bash
trace-annotate embodied-pipeline batch.json --nsteps 5 --report output/batch-new/report.html
trace-annotate embodied-pipeline batch.json --nsteps 5 --auto-idle --corrected-output output/batch-repaired.json --report output/batch-repair/report.html
trace-annotate embodied-kappa annotator-a.json annotator-b.json --fps 2
```

Use the expected procedure supplied independently of the observed labels. Either declare `expected_step_count` in each record or pass `--nsteps` for a batch with a shared procedure. Inferring the expected count from the largest observed step cannot detect an omitted final step and therefore cannot pass the complete procedure gate.

Provide the actual media duration. Automatic idle calculation does not infer duration from the final working segment, does not clip invalid timestamps, and does not silently merge overlapping working intervals. It prepares a separate document in memory. `--corrected-output` publishes a new copy only when all requested gates pass; the original input remains unchanged.

## Results and exit codes

| Exit | Meaning | Action |
| --- | --- | --- |
| `0` | The declared machine-check scope passed | Retain the report and input hashes; perform required video review separately |
| `1` | A validation, procedure or comparison gate failed or needs review | Read the findings, repair a copy where appropriate, or arrange human adjudication |
| `2` | Input, invocation, encoding, parsing or output publication failed | Correct the reported problem and run again using a new output path |

Malformed JSONL lines, empty batches, duplicate clip identities, non-finite numbers and incomplete comparison coverage cannot be silently discarded. Reports identify source files and SHA-256 hashes. Existing report files and output aliases are rejected, so a failed run cannot present a previous report as a new result.

## What the checks establish

- Structural checks cover finite interval bounds, overlap, working/idle conflicts and coverage, positive integer step IDs, verdict fields, error-category evidence templates, mid-segments and correction events.
- Procedure checks expose missing, unexpected, repeated and temporally out-of-order steps. A legitimate annotation may correctly describe an execution error. A procedure difference requests review; it does not establish that the annotation or video action is wrong.
- Sampled Cohen's Kappa measures binary working/idle agreement. The report separately checks comparison coverage, matched segments and supplied semantic labels. It reports single-class degeneracy explicitly and does not convert an undefined Kappa into a perfect production score.
- A high Kappa does not prove that the files were created by independent annotators, that either annotator is correct, or that the boundaries match the video. The sample rate and boundary policy remain part of the measurement.
- The specification gives conflicting instructions for `描述不符-仅部分工作`. Its data is retained and marked `REVIEW_REQUIRED`; the tool does not silently settle that policy conflict.
- Evidence templates check supplied locators and wording. They cannot verify that the described event occurred. Omitted-step evidence may use its step locator without a made-up timestamp.

Reports label their scope and distinguish machine results from unverified video content. Simulated second-annotator fixtures are software regression data and cannot be presented as a real blind-labeling study or an official gold standard.

The dark report dashboard calculates video-record counts and working/idle
durations from the primary annotation input. Comparison files do not double
the totals. Missing or structurally invalid inputs show unavailable time
totals rather than a clean zero; pending procedure/comparison review remains
visible even when the structure-error count is zero. See the [actual browser
and report demonstration](DEMO_MEDIA.md).

## Windows launchers

From a source checkout:

```bat
scripts\windows_qc.cmd "C:\data\batch.json" 5
scripts\windows_qc.cmd "C:\data\batch.json" 5 --no-open
scripts\windows_workbench.cmd --check
```

The QC launcher creates a fresh report directory, propagates the Python exit status, and opens a report automatically only after a successful run. `TRACE_QC_REPORT_ROOT` selects an isolated report root for tests. The workbench `--check` verifies that the editor file exists; actual browser behavior requires separate browser acceptance.

The launchers contain ASCII text with CRLF line endings, also valid under GBK. They select UTF-8 console output for Python. File encoding alone is not evidence that launch, validation or browser playback succeeded.

## Explicit in-place idle repair

`embodied-idle --inplace` remains available when explicitly requested. After validation it creates an exclusive backup of the original bytes, writes and syncs a temporary file in the same directory, rechecks the original and publishes the replacement atomically. Invalid inputs remain unchanged. Normal pipeline runs and the default idle command do not replace the source file.

## Scope of this stage

This stage improves QC reliability, input preservation, error propagation and reproducible reports. It does not add automatic semantic video understanding, spatial object tracking, human ground-truth certification or universal codec support. See [general video annotation](GENERAL_VIDEO_ANNOTATION.md) and [media compatibility](MEDIA_COMPATIBILITY.md) for the separate segment editor and preparation workflow.
