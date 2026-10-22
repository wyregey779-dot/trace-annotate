# Changelog

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
