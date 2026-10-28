# Persistent workspace development slice

This is part of `1.1.0.dev0`, on `feat/general-video-annotation`. It is not a
finished batch annotation product or a release candidate. The existing
`video-ui` and embodied editor remain the available annotation interfaces.

## What this slice implements

- A project directory containing SQLite WAL/FULL storage and immutable,
  SHA-256-addressed objects. An OS lock permits one writer process; a reentrant
  transaction lock serializes HTTP threads. Revision, commit and command receipt
  are committed together. Failed transactions do not advance the task pointer.
- Media admission through the existing full FFmpeg decode/preparation pipeline.
  The workspace frame index records actual decoded frame order, source/proxy PTS
  and rational time. Audio/video offsets, rotation and sample aspect ratio are
  retained. Unverifiable input is rejected rather than assigned estimated frames.
- Independent tasks for repeated imports of the same content. Failed inputs stay
  in the intake ledger. Interrupted imports are marked failed when the project
  reopens; this slice does not silently retry them.
- Queue input snapshots, bounded dispatch (default one worker), leases, pause,
  cancellation, explicit retry of failures, and restart recovery. Jobs commit
  separate candidate artifacts and never overwrite manual annotations.
- A loopback HTTP API with Host/Origin checks, a per-instance capability header,
  bounded JSON, revision conflicts and persistent command receipts. Its browser
  page is currently a JSON snapshot viewer, not the W04 editing workbench.
- `verify-media` is a real queue processor that rehashes the admitted playback
  and frame-map objects and checks their frame/time correspondence. It does not
  infer labels or establish annotation correctness.

## Run a batch

Use a local filesystem with hard-link support, Python 3.9 or newer, and FFmpeg
and ffprobe on PATH. Retain the original media. Install the current checkout:

```powershell
python -m pip install -e ".[dev]"
trace-annotate workspace init --project D:\TraceAnnotateProjects\sample
trace-annotate workspace import --project D:\TraceAnnotateProjects\sample C:\Videos\one.mp4 C:\Videos\two.mp4
trace-annotate workspace list --project D:\TraceAnnotateProjects\sample
trace-annotate workspace intakes --project D:\TraceAnnotateProjects\sample
```

Import reports each valid or rejected input and exits nonzero for a partial
failure. Use task IDs from `list` for a fixed selection. Replace the placeholders:

```powershell
trace-annotate workspace batch create --project D:\TraceAnnotateProjects\sample --task-id TASK_ONE --task-id TASK_TWO --command-id cmd-verify-001
trace-annotate workspace batch run --project D:\TraceAnnotateProjects\sample --batch-id BATCH_ID
trace-annotate workspace batch show --project D:\TraceAnnotateProjects\sample --batch-id BATCH_ID
trace-annotate workspace export --project D:\TraceAnnotateProjects\sample --name snapshot-001
```

`batch run` returns nonzero while any selected job is failed, cancelled, paused
or otherwise unfinished. A succeeded verification job has a durable candidate
commit and an artifact hash; the task's annotation revision remains unchanged.
Repeating the same creation command returns its original receipt. A changed
request with the same command ID is rejected.

```powershell
trace-annotate workspace batch control --project D:\TraceAnnotateProjects\sample --batch-id BATCH_ID --action pause_dispatch --command-id cmd-pause-001
trace-annotate workspace batch control --project D:\TraceAnnotateProjects\sample --batch-id BATCH_ID --action resume_dispatch --command-id cmd-resume-001
```

The other actions are `cancel` and `retry_failed`. Pause stops new dispatch;
it does not suspend arbitrary running code. Running processors must cooperate
at their cancellation checkpoints. An expired lease cannot commit output.
The process-local concurrency slot remains occupied until its processor exits.
Separate CLI invocations cannot control a project held by a running server;
use that server's authenticated control endpoint while it owns the project.

## HTTP surface

`workspace serve --project PATH` prints the local URL. Requests under
`/api/v1/projects/current` support `snapshot`, `tasks/{id}`,
`tasks/{id}/commands`, `commands/{id}`, `intakes`, `batches`, `batches/{id}`,
`batches/{id}/control`, `jobs/{id}` and `jobs/{id}/control` as implemented in
`workspace/server.py`. Reads require `X-TraceAnnotate-Token`; writes also require
the exact local Origin and JSON Content-Type. Do not publish the token.

Only add/update/delete manual annotations are available in this slice. Unknown
fields, out-of-range times, unsupported evidence, frame anchors and custom
attributes fail explicitly. The server does not accept arbitrary local file
paths or run unknown queued processors automatically.

## Data and compatibility boundaries

The SQLite storage version and task digest algorithm are explicit. Workspaces
created by the earlier experimental W02 implementation lack that version and
contain approximate frame maps. They are rejected with
`WORKSPACE_MIGRATION_REQUIRED`, preserving their files. Do not delete them or
edit their metadata to bypass this guard. Create a separate new workspace from
the original source; migration of existing annotations needs a dedicated,
verified workflow. Old temporal v1/v2 files and prepared-media v1 are unaffected.

On open, registered object bytes and admitted-media references are verified.
Missing/corrupt data refuses writes with `RECOVERY_REQUIRED`. A recovery viewer
and repair UI are not yet implemented. Published orphan objects can remain after
a failed transaction; their presence is not a successful job. No automatic
garbage collection removes them.

`workspace export` writes a new immutable development snapshot, including
queue state. It is not yet the frozen workspace-v3 exchange contract or W08
portable project package; it does not copy media, support round-trip import,
or implement the complete review/evidence schemas. Preparation diagnostics are
private local evidence and can contain source paths. Do not publish a whole
project directory without a reviewed export/privacy step.

## Verification and remaining work

Regression coverage includes injected transaction failures, two real subprocess
terminations around commit, competing saves, cancelled/stale workers, interrupted
imports, corrupt/missing objects, same-content task isolation, actual HTTP
requests and real FFmpeg CFR/VFR/audio-offset/rotation fixtures. These establish
the stated engineering behavior; they do not establish arbitrary filesystem
power-loss safety, semantic video accuracy or browser frame-accurate editing.

The next slice must integrate the multi-video browser workbench, mapped-frame
navigation, evidence and review history, candidate adoption, complete-partition
QC, migration and portable exports. Intake scheduling is still synchronous in
the CLI; its cancellation/retry protocol is not yet unified with task jobs.
Real sample reference labels, quality thresholds, performance limits and the
release acceptance matrix still require measured validation before release.
