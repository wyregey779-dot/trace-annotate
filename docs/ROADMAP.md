# Known limitations and roadmap

This is a list of current limitations and proposed work, not a claim that the work is implemented or scheduled. Priorities can change after reproducible user feedback.

## Current limitations

- The CLIP adapter is incomplete. Its optional extra does not install `cn_clip` or weights, and its current classifier is not a complete inference path.
- Optional face/OCR/scene integrations depend on external runtimes, model files or the `ffmpeg` executable. Missing dependencies and third-party model behavior need additional coverage.
- Export adapters are tested locally. Import compatibility with specific Label Studio/CVAT releases and action-recognition training tools still needs end-to-end validation with documented versions.
- Timeline agreement is calculated from samples. A high working/idle agreement score alone does not establish semantic label quality.
- Staging copy verification compares size and a hash of the first MiB. It is not full-file verification, a hardware write lock or a guarantee against storage failure. Use a dedicated staging directory and backups.
- Synthetic examples and unit tests do not establish performance at scale, model accuracy or broad adoption. No public download or deployment metrics are claimed.

## Proposed work

- Implement real CLIP inference with local weight selection, explicit download behavior and tests of successful, missing-model and failed-model states.
- Validate representative round trips in external annotation tools and publish the tested formats and versions.
- Improve CLI error reporting for malformed JSON/JSONL, invalid arguments and missing optional dependencies.
- Extend the installed-wheel/resource and model-free example checks added in 1.0.1 to more optional integrations.
- Collect opt-in, reproducible feedback from users and benchmark synthetic or redistributable datasets before publishing performance claims.

Discuss a proposal in [GitHub Issues](https://github.com/wyregey779-dot/trace-annotate/issues) before starting a broad change. Use synthetic reproducers and avoid private footage or credentials.
