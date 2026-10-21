# Security reports

TraceAnnotate is an early-stage project. Use the latest patch release and test it on copies of data before using it in an important workflow. It is not a sandbox for hostile files or processes; do not run it with elevated privileges.

If GitHub's **Report a vulnerability** option is available in this repository's Security tab, use it for sensitive details. If that option is unavailable, open an issue requesting a private reporting channel without including exploit instructions or sensitive data. Do not post private media, access tokens, or production paths in public reports.

Include the affected version, operating system, a minimal synthetic reproduction, impact, and any proposed repair. The maintainer will assess reports as capacity allows; no response-time guarantee is offered. Security fixes are documented in releases and the changelog. Only the latest release is maintained; older releases do not receive separate backports.

Staging and export paths are checked for directory escapes. These checks do not protect against a separate hostile process changing filesystem links concurrently. Run jobs in directories controlled by the current user, keep source backups, and validate imported annotations before exporting them.
