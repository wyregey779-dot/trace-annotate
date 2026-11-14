# Persistent workspace development slice

The core workspace is part of the published `1.1.0.dev1` preview and
development code on `main`; the vision-assistance additions in this checkout
are an unreleased `1.1.0.dev2` branch. The workspace is not included in stable
`v1.0.2` or a finished batch annotation product. `workspace serve`
opens the persistent multi-video segment editor. The existing `video-ui` and
embodied editor remain separate, compatible annotation interfaces.

The workbench header links to the original general and embodied tools in new
tabs (`/tools/general/` and `/tools/embodied/`). These use the unchanged packaged
editors and their independent-file saving workflow. They do not automatically
import workspace tasks or read drafts from the earlier file-page origin. Select
the matching media and import its original JSON to continue old work. See
[the compatibility inventory](EDITOR_COMPATIBILITY.md) for retained features
and the remaining integration gates.

The main play button resumes continuous playback from the current position;
it exits any previous segment-only range. Play selected segment starts again
from that segment's saved start, with optional looping. Exact-frame mode shows
a static image; its play button returns to video at the displayed frame time.

Task report/export endpoints describe saved structure, not semantic approval.
Derived coverage is exact within its documented budget of 20,000 segments and
4,096-bit intermediate rationals. An exceeded budget is reported as unavailable
statistics, never zero coverage; original JSON/CSV task values remain exportable.
The report gives each track a stable `track_id` (SHA-256 of the exact UTF-8
name). For long or special-character names use
`tasks/TASK/export?format=srt&track_id=ID`; `track` and `track_id` are mutually
exclusive. Short encoded names remain supported through `track`.

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
  workbench lists independent tasks, plays admitted media, edits manual segments,
  handles unsaved changes/conflicts, and downloads a development snapshot.
- Exact frame navigation uses a paginated decoded index and locally decoded PNG
  images. Saved frame anchors must match the admitted map's index, source PTS,
  proxy PTS and rational time. Ordinary playback navigation remains second-based.
- `verify-media` is a real queue processor that rehashes the admitted playback
  and frame-map objects and checks their frame/time correspondence. It does not
  infer labels or establish annotation correctness.

## Run a batch

Use a local filesystem with hard-link support, Python 3.9 or newer, and FFmpeg
and ffprobe on PATH. Retain the original media. Install the current checkout:

```powershell
python -m pip install -e ".[dev]"
trace-annotate workspace init --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample
trace-annotate workspace import --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample D:\视频标注具身标注\02_PROJECT_DATA\02_MEDIA_INPUT\one.mp4 D:\视频标注具身标注\02_PROJECT_DATA\02_MEDIA_INPUT\two.mp4
trace-annotate workspace list --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample
trace-annotate workspace intakes --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample
```

Import reports each valid or rejected input and exits nonzero for a partial
failure. Use task IDs from `list` for a fixed selection. Replace the placeholders:

```powershell
trace-annotate workspace batch create --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample --task-id TASK_ONE --task-id TASK_TWO --command-id cmd-verify-001
trace-annotate workspace batch run --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample --batch-id BATCH_ID
trace-annotate workspace batch show --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample --batch-id BATCH_ID
trace-annotate workspace export --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample --name snapshot-001
```

`batch run` returns nonzero while any selected job is failed, cancelled, paused
or otherwise unfinished. A succeeded verification job has a durable candidate
commit and an artifact hash; the task's annotation revision remains unchanged.
Repeating the same creation command returns its original receipt. A changed
request with the same command ID is rejected.

```powershell
trace-annotate workspace batch control --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample --batch-id BATCH_ID --action pause_dispatch --command-id cmd-pause-001
trace-annotate workspace batch control --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample --batch-id BATCH_ID --action resume_dispatch --command-id cmd-resume-001
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
the exact local Origin and JSON Content-Type. Do not publish the token. The
bootstrap document also creates an HttpOnly SameSite=Strict cookie scoped to
media resources, so native video/image requests do not put a token in a URL.

Manual commands include add/update/delete/replace annotations, conditional
`revert_command` and explicit `review_annotation`. Undo names an original
command and only applies to its protected successor/history chain. A newer
independent edit produces `HISTORY_DIVERGED`; refreshing does not reauthorize
an old snapshot. History advances only after a durable receipt.

`tasks/{id}/import-temporal` requires an explicit command ID, expected revision
and content hash. The UI captures these before file reading/confirmation and
retains the original request when its result is unknown. Temporal v2 must match
the admitted source hash/size. Legacy v1 has no strong media identity: workspace
import refuses it with `MEDIA_BINDING_REQUIRED`. Bind it to the verified source
in the classic tool and export v2 first. The original independent-file v1 tool
remains available; an in-workspace weak-binding migration UI is unfinished.

A v2 document whose verified source matches may represent the final exact media
endpoint with six decimal places. Only an end equal to that declared duration,
rounded above the exact endpoint by at most half a microsecond, is mapped back
to the exact rational endpoint. Original numbers and the versioned mapping rule
remain in the source envelope; export disclosures name this adjustment. Internal
boundaries, unbound v1 documents and larger out-of-range times are not clipped.

Review binds saved content, media and template to a local record. It does not
authenticate an independent reviewer or certify semantic correctness. Ordinary
writes cannot mint a review by carrying `review_status=reviewed`. Historical
imported flags remain source metadata; reports distinguish unbound flags from
content-bound local review actions.

Unknown
fields, out-of-range times, unsupported evidence and custom
attributes fail explicitly. The server does not accept arbitrary local file
paths or run unknown queued processors automatically.

## Browser workflow

### Visual analysis: resources and explicit retry

Visual candidates remain unreviewed and separate from saved manual annotations.
Installing the base package does not install optional model runtimes or weights.
An unavailable runtime or missing checkpoint is reported as `not_run`; querying
capabilities does not load a model. A verified local checkpoint that fails to load
is a failed execution, rather than a successful prediction or a missing-model result.

CLIP and VLM run one model at a time in separate owned child processes. The first
process must exit and be reaped before the next model starts. Default admission
requires 2 GiB for the model plus a 1 GiB available host-memory reserve. Admission
waits at most 60 seconds; running workers also check the reserve. These are
conservative local checks, not an OS memory reservation or a guarantee for every
model. The model slot is shared by workspaces in the same service process.

Optional environment settings are `TRACE_ANNOTATE_MODEL_BUDGET_MIB` (default 2048,
minimum 2048), `TRACE_ANNOTATE_MODEL_RESERVE_MIB` (default 1024, minimum 1024),
`TRACE_ANNOTATE_MODEL_WAIT_SECONDS` (default 60, range 0–600),
`TRACE_ANNOTATE_MODEL_FRAME_SECONDS` (default 60, range 5–600) and
`TRACE_ANNOTATE_MODEL_PHASE_SECONDS` (default 3600, range 60–86400). Invalid values
are rejected. Memory measurement failures, exhausted memory, timeouts and worker
failures are explicit failures; they do not commit a candidate result. Cancellation
and timeout terminate the owned model process tree and wait for its wrapper.
On Windows the service owns the last kill-on-close Job Object handle. This does
not establish equivalent cleanup after a sudden parent crash on every platform.

On Windows, publishing model progress retries sharing or access conflicts at
most 20 times, 50 ms apart, while keeping the previous complete JSON visible.
Persistent contention produces `MODEL_PROGRESS_FAILED` and retains the pending
file and process log. Other filesystem errors propagate immediately. This retry
does not change model-result publication or execution deadlines.

The **Retry task** button is available for failed or cancelled visual jobs whose
saved input still matches the current task. It keeps the original command ID
while a response is unconfirmed, queries its durable receipt, and displays queued
only after confirmation. The request to `POST /jobs/{id}/control` uses
`retry_failed` or `retry_cancelled`, `command_id`, `expected_revision` and
`expected_sha256`. Revision and hash are an optional pair for job retries;
legacy callers without this pair keep their existing frozen-input behavior.
Batch controls and cancellation do not accept these additional fields.

For a new guarded retry, the server checks both the frozen job input and current
task inside the write transaction. A changed task returns `REVISION_CONFLICT` or
`MEDIA_CHANGED` without changing the job or recording a successful control.
An already committed command still returns its original receipt after later
edits; replay does not start another attempt. A changed model configuration needs
**Run visual analysis** to submit a new configuration, rather than repeatedly
retrying a job marked `VISION_RESUBMIT_REQUIRED`. Release resources or correct the
reported environment problem before retrying; retry is not a quality approval.

Keep workspaces, model caches and diagnostics under the chosen project root.
For this local deployment that root is `D:\视频标注具身标注`; use
`02_PROJECT_DATA\03_ANALYSIS_OUTPUT` for workspaces/runs and
`02_PROJECT_DATA\06_LOCAL_TOOLS\vision-assist` for model tools/caches. Explicitly
permitted model downloads must be prepared and verified before submitting a job
with its frozen deployment identity. Preserve successful candidate evidence and
failure diagnostics; do not delete worker directories to manufacture a pass.

The resource and sampling checks have distinct scopes: a three-frame real model
run verifies that deployment and process isolation; a 1369-frame decode comparison
verifies sample identities and bounded decoding. Neither replaces a new full-video
pipeline run or semantic/reference-label evaluation. Historical full-video runs
with failed boundary sampling remain failed for those modules.

Boundary analysis also scans every decoded frame in the selected playback range
at 96 x 64 grayscale resolution. It retains at most 128 MiB of raw measurements,
uses 256-frame windows with 64-frame overlap, and has a 180-second time limit.
Byte-budget truncation is explicit. `coverage.complete` describes the decoded
range; it does not certify transition recall. A supplementary dissolve pass
retains at most 2048 rounded grayscale frames and tolerates at most 16 consecutive
unchanged frames. It fits the complete retained interval and records clipped
ranges and buffer limits. All global dissolve proposals are checked against
positive affine and nondecreasing luminance mappings, including gamma changes.
After a whole-run fit fails, at most four observed static plateaus are tested as
split points. Both sides must independently pass before two proposals are kept.
Longer transitions, long quantization plateaus, moving
mixtures and weak endpoint changes can remain unclassified. Unknown sustained
motion is counted without asserting a dissolve.

Retained fade, dissolve and flash proposals include actual frame-map identities
and exact rational half-open time windows. Their full-size before/anchor/end
images are retained separately; flash evidence includes the recovery frame.
Clipped or merged intervals remain uncertain. Fit metrics identify the component
interval they measure, rather than claiming a fit over a merged interval. At most
64 global proposals and four projections per scene are retained, subject to the
existing 32-claim semantic limit. Omitted projections and their budgets are
recorded; raw proposals remain available. A broader global interval remains
available beside an overlapping local candidate, with an explicit overlap for
review. Supported frozen detector configurations retain their read/apply path;
queued jobs using an older configuration require a new submission. Saving still requires an explicit
action and produces unreviewed model candidates, preserving manual annotations.

After initializing a new workspace, run:

```powershell
trace-annotate workspace serve --project D:\视频标注具身标注\02_PROJECT_DATA\03_ANALYSIS_OUTPUT\workspaces\sample
```

Select multiple files in the workbench. Upload and media preparation are separate:
received files enter a persistent intake queue and are prepared in the background,
with at most one active preparation per workspace. A rejected file does not stop
unrelated files. The development upload limit is 512 MiB per file. The interface
provides stop-upload and individual cancel/retry controls; their outcome depends
on the current intake state. Interrupted uploads are not admitted; failed or
interrupted preparation is recorded for explicit review or retry. If the connection
is lost, refresh the task/intake list and reconcile its state before retrying;
do not assume an unknown request failed or blindly submit a duplicate.

Select a task, enter a label and interval, then save. An unchanged time field
retains its original rational value and frame anchor. Editing a time manually
removes that field's anchor. Another tab's save produces a revision conflict;
the editor keeps the draft for comparison rather than overwriting the newer
revision. Unsaved input is guarded during navigation and kept in independently
identified browser/project drafts. Project drafts have their own CAS revision;
old annotation bases remain available for explicit conflict comparison. The
recovery selector lists dirty drafts. Rescue files include complete form/anchor
state, base identity and any pending request; importing one does not submit
annotations. Unknown commands are queried before explicit original-request
resend. Synchronization can fail or be delayed: only confirmed persisted content
is recoverable after a crash. Committed saves survive browser/service restarts.

Draft routes are `GET /drafts`, `GET /drafts/{id}` and `POST /drafts` under the
same authenticated project API and request-size limit. Drafts are private
recovery data, not portable project backups.

The experimental Python APIs `backup_workspace(store, new_directory)` and
`restore_workspace(backup_directory, new_project_directory)` in
`trace_annotate.workspace.backup` snapshot SQLite and registered immutable
objects. They refuse active intakes/queue jobs, existing or overlapping targets,
corrupt objects and unknown versions. Recovery verifies the snapshot and opens
the new project without replaying commands or starting jobs. Arbitrary exports,
logs, temporary files and external original media are not included. A complete
backup manifest is required; an interrupted directory is not a valid backup.
This does not implement migration of unsupported older storage versions or a
portable interchange format, and there is no backup/restore CLI yet.

Use the exact-frame view for frame boundaries. Frames are zero-based, and a
segment is `[start,end)`: selecting frame N as its end excludes frame N. Use
the actual video endpoint to include the last frame's entire hold; no imaginary
end frame is created. Frame images are the compatibility proxy's decoded RGB
pixels, not original-lossless evidence. The decoder selects by display index
from the beginning, has a 120-second timeout and one concurrent decode slot.
The image cache is limited to 32 images/256 MiB; high-resolution/long-video
latency has not been qualified. Source media is never rewritten.

Normal service shutdown requests cancellation of in-flight media admission and
waits for request workers before closing SQLite. Successful frame cache images
owned by that service are removed on normal close; diagnostic logs and interrupted
session remnants are retained. This is not a disk quota or automatic recovery
cleanup policy for crashes.

The snapshot download contains saved revisions and the intake ledger, not the
unsaved form. It contains no media bytes and cannot yet be reimported as a
portable project. Keep the workspace directory to resume saved work.

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
a failed transaction; their presence is not a successful job. The explicit
`sweep_unbound_artifacts` Python API defaults to a dry run and checks registered
references before deletion. Applied cleanup records a durable deletion journal;
startup recovers pending journal entries. This is not a disk-quota policy.

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
power-loss safety or semantic video accuracy. The workbench browser regression
checks actual displayed pixel markers for encoded CFR/VFR/audio-offset fixtures,
then verifies saved anchors, reload, task isolation and revision conflicts.

Current-task professional JSON, explicit content-bound review, candidate reports
and frozen annotation-video exports are described in [PROFESSIONAL_JSON.md](PROFESSIONAL_JSON.md).
Broader evidence interchange, complete-partition QC, bulk edits, migration and
portable project exports still need further qualification. Intake preparation
runs in a background queue; its cancel/retry controls are separate from task jobs.
A mapped anchor establishes a frame reference, not independent semantic approval.
The complete W04/W09 acceptance matrix
(including real-media scale/performance and release qualification) remains open.
Real sample reference labels, quality thresholds, performance limits and the
release acceptance matrix still require measured validation before a stable release.
