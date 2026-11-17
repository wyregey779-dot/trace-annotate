# Roadmap and acceptance criteria

The next work focuses on filesystem behavior, annotation interchange and optional model integration. Each item below includes the checks needed before it can be described as supported. Priorities may change after reproducible user reports; no delivery dates are committed.

## Current boundaries

- Core processing runs locally without an API key. Optional runtimes and weights need separate installation and verification.
- The development branch contains an offline CLIP adapter and pinned local weights with scoped real forward-pass evidence. Missing weights still return `not_run` and inference failures return `failed`; independent semantic accuracy is not yet signed off.
- The vision-assistance branch contains bounded object, relation and 21-point hand observations plus an optional 16-frame VideoMAE window helper. Its default v1 samples one window per eligible scene; the experimental opt-in v2 samples at most eight local windows from already decoded content frames and reports sparse or omitted coverage. Neither policy recovers events absent from those frames or establishes object-interaction actions. Successful execution proves only that the named provider ran on the bound frames; independent accuracy and human sign-off are still open.
- Professional JSON carries a deterministic, explicitly lossy visual cross-section. The durable vision job and project backup remain necessary for omitted raw observations.
- The optional pinned VideoMAE/SSV2 weights are CC-BY-NC-4.0, are not bundled with the MIT package, and must not be treated as a commercial-use grant. Runtime, code, weight and dataset licenses are evaluated separately.
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

The media-identity part of gate 1 is implemented in the development line: full streaming SHA-256, v2 media binding, explicit v1 migration, cancellation and verified rename recovery. The persistent workspace also records decoded frame order and source/proxy PTS, supports frame-index and adjacent-frame inspection, and has scoped CFR/VFR/audio-offset/rotation regressions. The separate `video-ui` editor still lacks that verified frame navigation, and these scoped checks do not establish every real-media timing case. Gates 2–6 remain open.

The next spatial editing stage still needs a separate user-editable object/track schema: stable object IDs, frame/PTS time basis, rectangles and polygons, keyframes, outside/occluded state, interpolation provenance, human correction, and per-format exports. The current read-only model observations and bounded professional-JSON projection do not implement that UI or establish validated continuous tracks. Acceptance must include real videos, coordinate scaling, variable-frame-rate media, occlusion and re-entry, cancellation/recovery and round-trip checks against a named consumer. Temporal intervals, detector boxes and geometric association candidates must not be presented as completed object tracking.

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
| Observable facial cues (not emotion inference) | **Partial.** `extractors/face_landmarks.py` uses a pinned MediaPipe Face Landmarker and `extractors/expressive_cues.py` derives limited visible geometry. Scoped full/short real forward passes exist at the recorded 45855a8 code identity. | Bind cues to face tracks, add pose/size/quality refusal and temporal evidence. Keep internal emotion, personality, identity and gender outside the claim. | A sealed, human-reviewed permitted corpus; macro F1 >= 0.80 and non-unknown coverage >= 0.60; missing weights are `not_run`, execution errors are `failed`, and every candidate remains unreviewed until a person accepts it. |
| Scene and persistent content change | **Partial.** `extractors/scene_change.py`, `shot_boundaries.py` and the workspace transition-discovery path distinguish cut candidates and non-hard-cut evidence. Code and regression fixtures exist, but current independent gradual/content-change quality is not complete. | Complete bounded full-video scanning, gradual intervals, background-motion compensation, persistent-window evidence and confusion negatives for lighting, subtitles, occlusion and camera motion. | Hard cuts: one-to-one within ±1 actual frame, precision >= 0.95 and recall >= 0.90. Gradual: interval IoU >= 0.5 and macro F1 >= 0.80. Persistent content change: precision >= 0.85 and recall >= 0.70, with the frozen negative false-positive gates. |
| Shot attributes (镜头) | **Partial.** `extractors/shot_attributes.py` emits subject-box-based shot-size candidates and `camera_motion.py` emits geometry-based motion evidence with explicit unknown/not-applicable outcomes. Scoped real execution exists; independent class quality is open. | Improve multi-subject association, foreground masking, background inlier coverage and refusal for screen people, statues, low resolution and ambiguous zoom/dolly evidence. | Human-reviewed independent comparison; macro F1 >= 0.80 and non-unknown coverage >= 0.60 for shot size and camera motion, plus fixed-camera false-motion <= 0.05. |
| Object observations, sparse tracks and relations | **Implemented as bounded review assistance, not supported accuracy.** The exact COCO-80 provider can emit evidence-bound boxes; same-scene geometry builds sparse association summaries and conservative `near`, `contact_candidate`, `co_moving_candidate`, `held_candidate` and `control_transfer_candidate` projections. | Add a user-editable spatial schema/UI, named external round trips, occlusion/re-entry handling and an independently reviewed accuracy corpus. Do not treat association or relation candidates as identity, contact, possession or action facts. | Source and installed-package checks must preserve frame/PTS/hash evidence, budgets and loss manifests. Accuracy requires a sealed permitted corpus with per-class detection/association metrics and human review; UI create/edit/save/reopen/export must pass separately. |
| Hand landmarks | **Implemented as bounded observation assistance, not supported accuracy.** The pinned MediaPipe provider runs in an owned process and can retain complete 21-point normalized/pixel/world observations for a uniformly sampled subset. | Evaluate detection and coordinate quality on a permitted reviewed corpus; cover handedness ambiguity, mirror assumptions, occlusion, small hands and multiple people. No touch, grasp, held-state, identity or action inference. | Real model execution, cancellation/process-tree cleanup, deterministic sampling and professional JSON round trip must pass. Report detection/point metrics separately; structural validation is not accuracy evidence. |
| Action (动作) | **Partial.** The frozen vocabulary is 走/跑/拿起/放下/打开/关闭/递交. `extractors/actions.py` can emit review-only walk/run displacement candidates. An optional, exact local VideoMAE/SSV2 checkpoint can now add path-redacted, unreviewed window-semantic hints for close/open/pick-up/put-down-surface through the durable VisionJobs pipeline. Those hints have no participant or state binding and are not action facts. | Add temporal subject/object/recipient and before/after state evidence for pick-up, put-down, open, close and hand-over; compensate camera motion and independently evaluate the auxiliary window model. Preserve real PTS, model identity, rejection reasons and human review. Do not rename unrelated classes or promote a window hint to `action-candidate-v3`. | Per class on independent reviewed clips: same class and subject, interval IoU >= 0.5, precision >= 0.85 and recall >= 0.65; unknown counts as a miss. Separately report auxiliary-hint window classification quality. UI edit/save/reopen/export must also pass for every class. |
| Portable visual projection | **Implemented with explicit loss.** Apply deterministically projects at most 128 KiB of parameters, 12 evidence frames, 24 object observations, 12 track summaries, 12 relation candidates and 4 complete hand observations. The manifest binds the raw candidate digest and exact total/projected/omitted counts. | Keep the durable raw job and verified backup available; add installed-wheel browser and recovery acceptance. Never relabel the retained subset as the complete result. | Tampered frame/PTS/hash bindings fail closed; repeated apply is deterministic; professional export/import/re-export retains all projected data and loss metadata; backup restores the raw result. |
| Shot boundary (镜头边界) | **Partial.** Boundary events can include cut/gradual/flash/unknown types, frame/PTS evidence and workspace discovery. Existing tests and scoped executions do not yet establish the frozen independent-quality gate. | Finish bounded full-video and VFR coverage, block-boundary context, long-dissolve handling and an independently sealed reference corpus. | CFR and VFR frame-by-frame evaluation against signed reference data using actual indexed PTS; meet the hard-cut and gradual gates above without deleting difficult negatives. |
| Fact vs interpretation (事实/解读分离) | **Partial.** Visual-semantics v1/v2/v3 contracts, model-run provenance, unreviewed candidates and the v3 loss manifest distinguish measurements and model observations from reviewed records. Free-form human text and every external round trip are not yet proven to preserve that distinction. | Enforce claim type, author/reviewer and evidence identity through validation, professional JSON, evidence bundles and named external-tool round trips. | Mixed or unsupported records are rejected; model output cannot auto-promote to reviewed; edits invalidate old review; export/import retains claim type, author, source revision, evidence hashes and projection loss. |

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
