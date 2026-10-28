# Changelog

## Unreleased — 1.1.0.dev0

- Add the persistent multi-video browser segment workbench with ordered file admission, task isolation, exact rational times, revision conflict handling, durable save receipt recovery and development snapshot downloads.
- Decode selected frames by display index, verify mapped source/proxy PTS before saving anchors, and reject invalid or nonexistent frame references. Add actual browser pixel-identity and editing regressions; ordinary player seeks remain second-based.
- Add an experimental persistent workspace with cross-platform writer locking, real SQLite transactions, immutable verified objects, revision-bound receipts and exact decoded media maps. Preserve failed intakes and keep repeated-content tasks independent.
- Add durable candidate jobs with bounded dispatch, leases, cancellation fencing, pause/retry and process-crash recovery. Expose batch integrity verification in the CLI and authenticated controls through the loopback API; semantic auto-annotation remains unfinished.
- Reject legacy experimental workspace storage until explicit migration, unsafe exports, missing object references, malformed times and inconsistent task identities. Run the existing CI matrix on feature branches as well as main/master so development pushes receive actual remote checks.

- Distinguish unavailable final-frame duration metadata from ambiguous PTS mapping. Keep old-probe rejection explicit with a toolchain upgrade path; CI uses a fixed, SHA-256-verified FFmpeg build to execute the full media regressions without estimating timing.
- Verify explicit demux-packet durations when older FFprobe versions omit the decoded final-frame duration. Require a unique packet/frame PTS mapping, record the duration source and retain rejection of ambiguous endpoints.
- Require FFmpeg/ffprobe in every CI test environment so media regressions execute. Synchronize the browser seek-control test with completed segment selection before issuing Home/End inputs.
- Add a dark embodied editor with task descriptions, independently declared step counts, segment labels, explicit pending verdicts, seconds-based seeking and a bottom segment list. Retain incomplete drafts and preserve current annotations when media replacement is cancelled or rejected.
- Render actual QC totals in a dark, responsive report, keeping structural errors and pending review distinct. Add a reproducible synthetic-media browser recording, screenshots and regression workflow.
- Quote CSV fields consistently across supported Python versions, including bare carriage returns and multiline values.
- Reject unpaired Unicode surrogates in annotation fields and keys; encode complete text before publishing outputs so invalid strings cannot leave empty report files. Expose comparison-gate failures and sampling settings directly in QC reports.
- Unify embodied QC, idle repair, procedure comparison and agreement behind the packaged quality pipeline and CLI. Reject invalid or incomplete batches and propagate failed/review gates through exit codes and newly created reports.
- Validate working/idle/mid intervals and category-specific evidence; retain precise interval boundaries instead of rounding gaps into empty segments. Require declared media duration and independent expected-step coverage for complete pipeline checks.
- Report incomplete comparison coverage, semantic conflicts and undefined single-class Kappa explicitly. Separate sampled time agreement from human independence and video-content truth.
- Protect original inputs, escape report content, refuse existing report outputs, and retain a backup for explicitly requested atomic in-place idle repair. Add Windows launchers with fresh report paths and real exit-code propagation.

- Add `video-prepare`, `video-open` and prepared-project `video-ui --project` for verified H.264/AAC compatibility copies. Preserve original content identity, decoded frame order/PTS correspondence, final-frame hold and relative audio-track timing; record conversion commands and limitations.
- Serve prepared projects through a tokenized, read-only loopback server with bounded JSON reads and streamed HTTP Range playback. Automatically load prepared media in the editor, retain provenance through edits/clear/undo, and reject conflicting imports or drafts.
- Add explicit HDR-to-SDR preview conversion, compatibility workflow documentation and actual browser preparation/edit/recovery acceptance in CI. This expands media formats without adding automatic semantic annotation.
- Convert full-range MJPEG pixels into the playback range and retain known WMA decoder tail samples with explicitly recorded inferred timing; reject missing interior timestamps instead of guessing a continuous timeline.
- Require decoded video dimensions and a usable first frame before replacing a general-editor project. Reject audio-only files and videos whose unsupported picture track leaves only playable audio; preserve existing annotations and dirty forms on rejection.
- Reject nonzero browser timeline starts and stop editing if loaded duration changes. This does not add transcoding, timestamp normalization, frame indexing or full-file media validation.
- Fix the optional FFmpeg scene helper's missing final shot and discarded sub-second shots. Validate real video input, report decode/probe/limit failures explicitly, and stop returning invented whole-clip results for missing files. The helper remains separate from the manual editor.
- Add a separate general video segment model with custom labels, descriptions, tags, tracks and explicit review state. Overlapping events are supported; steps, working/idle and embodied verdicts remain in the existing workflow.
- Add a local browser editor for creating, editing, saving and restoring general annotations, with JSON, CSV and subtitle projections.
- Add `video-ui`, `video-validate`, `video-export` and `video-import` commands. Require explicit track selection for multi-track subtitle exports and refuse existing output files.
- Validate JSON structure and finite timestamp bounds; reject duplicate IDs and unsupported fields instead of silently dropping data. Preserve full information in native JSON; document subtitle precision and CSV safety transformations.
- Add general segment fixtures, parser/CLI regressions and installed-package resource checks. Bounding boxes, polygons and object tracking remain planned work.
- Preserve unedited native strings when saving a segment; editing only timestamps no longer normalizes line breaks in labels, descriptions or track names. Support multiline labels/tracks and retain dirty-form recovery across reloads.
- Invalidate previous review state when annotation content changes; require explicit confirmation of the current content. Preview lossy export scope and unreviewed counts before browser downloads.
- Publish complete CLI output atomically without overwriting existing files, with fault-injection coverage for partial writes, sync errors and destination races. Require hard-link support on the output filesystem.
- Add a browser integrity regression suite and run it alongside the general editor workflow in CI.
- Bind new browser documents and drafts to full media SHA-256 using bounded chunk reads in a cancellable background Worker. Reject different content with matching filename/size/duration; support explicit relinking after a verified rename.
- Add temporal schema v2 with required media size/hash. Keep v1 parsing/export support and require explicit browser migration, clearing inherited reviews when previously unbound records are attached to media.
- Add `video-hash` and `video-bind` for full local hashing and explicit media binding; bundle the pinned MIT hashing dependency and its provenance notice. Show presented-frame timestamps separately from seconds-based seek positions, without claiming frame-accurate navigation.
- Retain a frame callback received before `seeked`, so HEVC previews do not remain stuck waiting for another frame. Reject media growth during hashing and report excessively nested binding input as a CLI error.

## 1.0.2

- Unlink in-tree file symlinks without deleting their targets during staging cleanup. Skip directory symlinks and Windows junctions.
- Normalize in-use lock paths and retain protection when a file is locked through an alias. Report deletion failures instead of listing a failed removal as completed.
- Add filesystem regression cases for cleanup, link boundaries, locks and dry runs.
- Add a synthetic annotation fixture and a runnable quickstart that produces a validated annotation, a QC report and SRT subtitles without model dependencies.
- Reorganize the English and Chinese guides around supported workflows, with an actual report preview and explicit integration status.
- Include usage guides, examples and maintenance scripts in source distributions. Validate wheel/source contents and generate SHA-256 checksums for release downloads.
- Add a tag-triggered workflow that builds and checks packages before uploading them to a GitHub release.

## 1.0.1

This patch addresses problems found during a reproducible review of the initial release.

- Include the browser annotator and JSON/XML templates in built packages; check an installed wheel outside the source checkout in CI.
- Read standard JSON annotation documents as well as JSONL during validation.
- Reject missing or malformed annotation structures, boolean step numbers, and invalid or too-short durations before in-place idle updates.
- Use document duration when computing idle intervals and persist validated overrides, preserving the trailing idle span in downstream exports.
- Reject unsafe clip output names, case/Unicode aliases, and Windows reserved device names before writing action datasets.
- Continue reporting other invalid segment fields when timestamps are non-finite.
- Resolve filesystem links when checking staging paths and recheck paths before copying or cleaning up.
- Escape annotation text in QC reports and mark empty or unjudged work as pending rather than correct.
- Keep the unfinished CLIP inference adapter explicitly `not_run`, without automatic model loading or placeholder predictions.
- Start the browser editor with empty annotations, reset state on video changes, and use loaded media duration for idle computation and export.
- Replace unverified capability comparisons with documented behavior and limitations; add contributor reporting templates and a security reporting policy.

## 1.0.0

Initial public source release: asset staging and provenance helpers, temporal annotation validation and agreement, exporters, browser annotation UI, CLI, examples, and tests. Optional model adapters have limitations documented in the README.
