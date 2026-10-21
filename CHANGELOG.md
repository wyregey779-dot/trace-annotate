# Changelog

## 1.0.1

This patch addresses problems found during a reproducible review of the initial release.

- Include the browser annotator and JSON/XML templates in built packages; check an installed wheel outside the source checkout in CI.
- Read standard JSON annotation documents as well as JSONL during validation.
- Reject missing or malformed annotation structures, boolean step numbers, and invalid or too-short durations before in-place idle updates.
- Use document duration when computing idle intervals, preserving the trailing idle span.
- Reject unsafe or colliding clip output names in action dataset export.
- Resolve filesystem links when checking staging paths and recheck paths before copying or cleaning up.
- Escape annotation text in QC reports and mark empty or unjudged work as pending rather than correct.
- Keep the unfinished CLIP inference adapter explicitly `not_run`, without automatic model loading or placeholder predictions.
- Start the browser editor with empty annotations, reset state on video changes, and use loaded media duration for idle computation and export.
- Replace unverified capability comparisons with documented behavior and limitations; add contributor reporting templates and a security reporting policy.

## 1.0.0

Initial public source release: asset staging and provenance helpers, temporal annotation validation and agreement, exporters, browser annotation UI, CLI, examples, and tests. Optional model adapters have limitations documented in the README.
