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

## CLIP inference

Status: **planned; not implemented**. Installing the `clip` extra does not enable a complete inference path.

Completion requires:

1. A supported backend, package versions, local model format and weight license.
2. Explicit weight configuration. Any download must be a documented user action.
3. Actual image/text encoding and scoring, with documented score meaning and output provenance.
4. Tests for success, missing dependency, missing weights and failed inference. An unrun or failed model must not emit placeholder predictions.
5. A small permitted fixture and a reproducible inference check. Accuracy claims require a separate evaluation with a defined dataset and method.

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
