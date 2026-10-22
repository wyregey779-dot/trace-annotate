# Releasing TraceAnnotate

Releases contain an installable Python wheel, a source distribution and a
`SHA256SUMS` file. GitHub's automatically generated source archives are separate
from these tested Python distributions. Releases currently use GitHub; this
repository does not claim a PyPI publication.

## Prepare a release

1. Update `project.version` in `pyproject.toml` and `__version__` in
   `trace_annotate/__init__.py` together, then update the changelog.
2. Submit and review the change. Wait for all Linux/Windows CI jobs to pass on
   the commit being released; follow up on any new review findings.
3. Build from a clean checkout in a virtual environment. Use a new output
   directory for each attempt to avoid mixing old and new versions:

   ```bash
   python -m pip install build twine pytest tomli
   python -m build --outdir release-dist
   python -m twine check release-dist/*
   python scripts/release_assets.py --dist release-dist --expected-tag v1.0.2
   python -m pip install release-dist/trace_annotate-1.0.2-py3-none-any.whl
   python scripts/check_installed_package.py
   python -m pytest tests -q
   ```

   The asset check compares the wheel and source distribution with the package
   source files and verifies metadata and tag versions. The installed-package
   check imports from outside the checkout and exercises packaged resources and
   the CLI. Test the documented quickstart and affected user workflows too.

4. Push a `v<version>` tag pointing to that reviewed commit. A release tag must
   match the package version exactly; do not move a published tag to a new commit.

## Automated release assets

Pushing a `v*` tag runs **Release packages**. Its read-only build job creates and
validates the distributions, runs tests, checks the installed wheel and runs the
examples. A separate job with repository-content write permission verifies the
checksums and uploads those files using the repository's built-in GitHub token.
No additional token or PyPI credential is required.

If a release already exists, its title, notes and publication state are preserved.
Otherwise, the workflow creates a **draft** release for the maintainer to finish.
Review the assets and write concrete release notes before publishing that draft.
Only report checks that actually passed, and describe remaining limitations.

Uploads do not overwrite existing assets. If a rerun encounters a same-named
asset, inspect the existing release and workflow artifacts before deciding how
to recover. For changed package contents, publish a new version instead of
silently replacing an existing download.

## Verify a downloaded package

Download the wheel or source distribution and `SHA256SUMS` from the same release.
On Linux, `sha256sum --check SHA256SUMS` verifies both files if both are present.
On Windows, use `Get-FileHash <filename> -Algorithm SHA256` and compare the result
with the matching line in `SHA256SUMS`. Checksums detect a changed download; they
are not an independent signature of the publisher's identity.

After verification, install the wheel with `python -m pip install <wheel-file>`.
If publishing to PyPI later, configure a maintainer-owned PyPI project and a
trusted publishing identity first; it is not part of the current workflow.
