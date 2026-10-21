# Contributing to TraceAnnotate

We welcome reproducible bug reports, documentation corrections, tests, and focused improvements to local media review and temporal annotation.

## Report a problem

Open a [GitHub issue](https://github.com/wyregey779-dot/trace-annotate/issues) with the package version, Python/OS versions, command, expected behavior and actual result. Prefer a small synthetic example over private footage. Do not publish credentials or personal data.

For a potential security vulnerability, use the repository's private vulnerability reporting option when available. Do not post working exploit details or sensitive data in a public issue.

## Development setup

```bash
git clone https://github.com/wyregey779-dot/trace-annotate.git
cd trace-annotate
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, `source .venv/bin/activate` on Linux/macOS), then:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
trace-annotate --help
```

Install optional extras only when your change needs them. Vision and OCR integrations may require additional local model weights; the CLIP adapter is incomplete. See [README](README.md) and the [roadmap](docs/ROADMAP.md).

## Make a change

1. Create a branch from the current `main`.
2. Keep the change focused and explain the user-visible problem.
3. Add a regression test for a functional bug or new behavior. Use temporary directories and synthetic data; tests must not delete user files, require private datasets, or download model weights.
4. Run the relevant tests, then the suite when shared behavior changes. For packaging changes, also build and install a wheel outside the checkout to check bundled resources and CLI behavior.
5. Open a pull request with the change, verification commands and any known limitations. Do not claim tests passed unless you ran them; report skipped dependencies and checks.

## Design principles

- Preserve source media. Use dedicated staging directories, validate path boundaries and document cleanup behavior.
- Keep annotation provenance explicit. Distinguish rules, model outputs, unrun/failed extractors and human confirmation.
- Treat heuristics as review aids. Avoid fabricated ground truth and unsupported accuracy claims.
- Keep core processing local. Make any model download or network requirement clear.
- Maintain deterministic data exchange. Validate intervals and schemas, and document limitations of external adapters.

Be respectful in issues and reviews. Discuss behavior and evidence, and help others reproduce problems.

Contributions are distributed under the project's [MIT License](LICENSE). Do not include datasets, model weights or copied code without checking their licenses and your right to redistribute them.
