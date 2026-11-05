# Roadmap and acceptance criteria

The next work focuses on filesystem behavior, annotation interchange and optional model integration. Each item below includes the checks needed before it can be described as supported. Priorities may change after reproducible user reports; no delivery dates are committed.

## Current boundaries

- Core processing runs locally without an API key. Optional runtimes and weights need separate installation and verification.
- The CLIP adapter returns `not_run`; complete image/text inference is not implemented.
- Export adapters have local format tests. Compatibility with specific annotation applications and training pipelines still needs end-to-end checks.
- Copy verification covers file size and the first MiB, not the entire file. Use a dedicated staging directory for disposable copies; concurrent filesystem mutation remains outside the guarantees of the current path checks.
- Timeline agreement samples time intervals. It does not replace semantic label review.
- Synthetic fixtures and regression tests do not establish model accuracy, public adoption or performance at scale. No download or deployment counts are claimed.

## Workstreams

| Workstream | Intended result | Evidence required |
| --- | --- | --- |
| Filesystem lifecycle | Preserve source boundaries and staged files marked in use | Regressions for links, junctions, lock aliases and cleanup failures on supported platforms |
| First-run workflow | A reproducible path from an annotation record to reviewable outputs | Versioned fixture, copyable commands, expected results and installed-package checks |
| External formats | Move annotations between tools without silently changing timing or labels | Named tool versions, round-trip fixtures and comparison of retained/lost fields |
| Optional integrations | Distinguish useful predictions from missing or failed extractors | Explicit statuses, dependency/weight instructions and failure-path tests |
| Performance | Publish measurements that others can reproduce | Defined workload, hardware, repeated measurements and correctness checks |

## General and spatial video annotation

The `1.1.0.dev1` preview adds general temporal segments, custom labels/text/tags, overlapping tracks, review status, native JSON recovery and CSV/SRT/WebVTT projections. [The guide](GENERAL_VIDEO_ANNOTATION.md) defines what is retained or rejected. This is manual annotation, with no automatic recognition or transcription claim.

The media-identity part of gate 1 is implemented in the development line: full streaming SHA-256, v2 media binding, explicit v1 migration, cancellation and verified rename recovery. Presented-frame PTS is visible, but a complete frame index, verified adjacent-frame navigation and VFR precision remain open. Gates 2–6 are not completed by this work.

The next spatial stage needs a separate object/track schema: stable object IDs, frame/PTS time basis, rectangles and polygons, keyframes, outside/occluded state, interpolation provenance, human correction, and per-format exports. Its acceptance must include real videos, coordinate scaling, variable-frame-rate media, occlusion and re-entry, cancellation/recovery and round-trip checks against a named consumer. Temporal intervals and whole-frame rectangles must not be presented as completed object tracking.

### Delivery order

Each row is a separate delivery gate. The status above distinguishes the current implementation from planned work; a design entry alone is not a completed feature.

| Order | Scope and prerequisite | Acceptance gate |
| --- | --- | --- |
| 0. Data integrity | Native JSON is authoritative; browser form values replace only changed fields; review follows current content; CLI output publishes a complete new file | Actual browser downloads retain untouched multiline/CRLF text; edits invalidate old review; undo and dirty-form recovery preserve data; interrupted writes and concurrent destinations cannot replace existing output. Run both browser checks and output fault tests. |
| 1. Media identity and timing | After gate 0, add a versioned media manifest, streaming full-content hash and a PTS-based presented-frame index | Different content with identical filename/size/duration cannot silently resume reviewed annotations. CFR, VFR, rotation and codec fixtures retain boundaries on reopen. Adjacent-frame navigation uses actual indexed presentation timestamps, rather than nominal FPS arithmetic. |
| 2. Projects and recovery | Use the media identity contract; design explicit schema migration before adding project records | At least 20 permitted videos across two interleaved projects survive switching, rename/relink, missing-media recovery and restart without segment/draft leakage. Old native documents remain importable; failed imports leave the current project intact. |
| 3. Quality workflow | Build on project persistence; define labels, event boundaries, overlap rules and review revisions | Reproducible sample queues return to the exact media and interval. Injected omissions, wrong labels, boundary disagreements and stale review records are found and resolved. Report review coverage and agreement separately from measured accuracy. |
| 4. Objects and tracks | Require verified media timing/coordinates and recovery; start with manual rectangle keyframes | Stable object IDs survive motion, occlusion, exit and re-entry. Rotation, display scaling and VFR do not change exported coordinates. Interpolated values retain provenance and can be overridden or removed. Add polygons and assisted tracking only after manual keyframes pass. |
| 5. Interchange and assistance | Freeze one named consumer/version and task schema at a time | Actually import, edit and re-export in Label Studio/CVAT; compare retained/lost times, IDs, coordinates and states. Then evaluate one licensed model per task, recording weights/parameters/failures. Candidates require explicit human acceptance and cannot auto-promote themselves to reviewed. |
| 6. Installation and release | Applies throughout; close before a stable release | Install the built artifact in a clean supported Windows environment and complete real-video create/save/close/reopen/export/readback. Tie version, source commit, package hashes and CI evidence together; publish codec/filesystem limitations, security handling and licenses. |

For timing work, keep requested seek time, actual presented-frame PTS and saved boundary distinct. Proxy media require an explicit original-to-proxy mapping and validation. For spatial work, freeze rotation/crop transforms and coordinate conventions before implementing drawing tools; CSS display dimensions are not annotation coordinates.

Performance gates need repeated short/10-minute/1-hour and 1080p/4K workloads, with 100/1,000/10,000 segments where feasible. Record hardware, codecs, peak memory, UI latency, open/save/export times, dropped frames and failures. A successful short clip does not establish long-video or bulk-task reliability. Public fixtures need redistribution permission; private footage and local paths stay outside published evidence.

## CLIP inference

Status: **implemented in the development line `feat/vision-assist-20261103`**, not yet merged or released. The adapter loads `open_clip` ViT-B-32 with the `openai` weights, encodes English prompts against the Chinese labels through a multi-text ensemble, multiplies by `logit_scale.exp()` and applies per-category softmax; thresholds are 0.85 for binary and 0.70 for multiclass decisions, and anything below stays `None` instead of guessing.

Still open before this can be called supported: a permitted fixture with a reproducible accuracy check, published score semantics per label set, and an evaluation dataset with a stated method. Missing weights still report `not_run` and a crashed forward pass reports `failed`; neither emits a prediction.


## Visual semantics: current status and open work

Each row states what the repository can do today, where that behavior lives, and what is still missing. A capability is not supported because a model file or an adapter name exists; it is supported only when the acceptance check passes with real weights.

| Capability | Current state (with evidence) | Open work | Acceptance check before claiming support |
| --- | --- | --- | --- |
| Emotion (valence, arousal, expression, tone) | **Not implemented.** No extractor, no label vocabulary, no schema field: `extractors/` holds scene, faces, ocr, clip, vlm and metrics only; `workspace/professional_schema.validate_annotation` admits `label`, `text`, `tags`, `track`, `attributes` with no emotion slot | Decide the task first (facial expression, voice prosody, or text tone; do not merge them into one number), then add an adapter that writes `tags` plus numeric `attributes` with its own confidence | Real weights on a permitted clip; missing weights report `not_run`, crashes report `failed`; low-confidence frames emit nothing; every output stays `model_candidate`/`unreviewed` and emotion never enters a `reviewed` record without a human edit |
| Scene change | **Partial.** `extractors/scene.py` cuts on frame differences, so a scene change is only detected when it coincides with a hard cut; CLIP contributes `indoor_outdoor` and `screen_type` tags, which classify a frame rather than measure a change | Scene-level embedding (the existing CLIP encoder can be reused) or a scene classifier; separate hard cuts from gradual changes (fade, dissolve, lighting shift); emit change points with confidence | Fixtures with a dissolve, a lighting ramp and a static shot must be told apart; a static shot must yield no change candidate; confidence and threshold recorded in evidence |
| Shot (镜头) | **Partial.** Each cut interval becomes one `镜头候选 N` annotation with `track=镜头`; no shot attributes exist | Shot size (远景/全景/中景/近景/特写) and camera motion (推/拉/摇/移) classification, or ship shot size first and report motion as `not_run` | Human-labelled comparison on the same clip, per-class confidence, and a documented refusal path when weights are absent |
| Action (动作) | **Not implemented.** No video action adapter, no action label set, no time-window mapping | Fix the scope (single-person vs multi-person), choose a model and record its license, define the frame window and the action interval mapping, then write action tags and intervals | Licensed fixture, real inference, start/end agreement report against human labels, `not_run` without weights, `failed` on error |
| Shot boundary (镜头边界) | **Partial.** Cut points come from the ffmpeg/scenedetect frame-difference pass; a candidate is an interval `(start, end)`, so boundaries are implied rather than emitted as events; no boundary type or confidence is recorded | Emit boundary events with type (cut, gradual, flash), the frame numbers on both sides and their PTS; export them as their own list | CFR and VFR fixtures checked frame-by-frame against human boundaries; VFR boundaries must agree with the indexed PTS, not nominal FPS arithmetic |
| Fact vs interpretation (事实/解读分离) | **Not implemented.** An annotation has one `text` field and free-form `tags`; a VLM caption and a human judgement can currently land in the same string and the validator cannot tell them apart | Add an explicit discriminator, either `attributes.claim_type=observation|inference` or separate `facts`/`interpretation` fields, and enforce it in validation and export | A mixed record is rejected; `model_candidate` may only carry observations; interpretation requires a human author; round-trip export keeps the field and its author |

## External annotation round trips

Start with one Label Studio or CVAT version and one documented task configuration. A completed check should include:

- A permitted synthetic fixture containing working, idle, uncertain and error-labelled segments.
- Steps to import, inspect and re-export the fixture in the selected application.
- A comparison of duration, interval boundaries, step identity, verdict and evidence before and after conversion.
- Explicit handling or rejection of unsupported fields.
- An automated format regression and a record of the application version used for the manual check.

Action-recognition adapters need equivalent checks against a named consumer and its time-unit/frame-sampling convention. Parsing generated XML locally does not establish application compatibility.

## Performance measurements

Before publishing throughput or scale claims:

- Provide the fixture generator or redistribution rights, dataset size, operating system, Python version, hardware and storage type.
- Separate file I/O, hashing, timeline processing and model initialization/inference measurements.
- Record elapsed time and peak memory across repeated runs; document warm/cold-cache conditions where relevant.
- Include correctness checks and failure counts alongside throughput.
- Limit conclusions to the measured workload and environment.

## Feedback and contribution

A useful report describes the task, exact command or workflow, expected output, observed output and environment. Prefer small synthetic reproducers to private footage. Maintainer-authored issues can track genuine work but are not evidence of independent users or adoption.

Discuss a focused proposal in [GitHub Issues](https://github.com/wyregey779-dot/trace-annotate/issues), then submit a pull request with matching tests and documentation. See [CONTRIBUTING](../CONTRIBUTING.md) for the workflow and [SECURITY](../SECURITY.md) for security reports.
