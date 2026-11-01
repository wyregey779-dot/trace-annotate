# Professional task JSON

The workspace can export and import `traceannotate.professional-task` version1.
Use **导出专业JSON** and **导入专业JSON** in the existing workspace editor.
The existing Temporal JSON, CSV and SRT projections retain their formats.

## Open the workspace

Install the current development checkout or its built wheel, with FFmpeg6 or
newer and ffprobe on PATH. Choose a new project directory for the first command;
use that same directory to resume saved work:

```text
trace-annotate workspace init --project NEW_PROJECT_DIRECTORY
trace-annotate workspace serve --project NEW_PROJECT_DIRECTORY
```

Open the printed loopback URL, import videos and select a task. Save annotations
before starting **媒体参数与客观分析** or **导出带标注的视频**. Analysis first creates
a candidate; **确认保存分析结果** admits its reports into the task. Professional
JSON exports those saved values. For an annotated video, choose panel position,
font sizes, fields and tracks, preview a frame, then generate and download the
video plus its paired JSON. Cancelling or failing does not produce a partial
download. An old export stays bound to its input revision after further edits.

`video-ui` and `embodied-ui` still open the independent original editors. They
do not share this workspace's database, analysis jobs or render jobs.

## Document contents and storage

The professional document contains the complete current task, exact rational
times and frame anchors, admitted source/playback identities, template,
saved typed analysis reports, dependency assessments, artifact descriptors,
available local audit index, commits and receipts. The index records each
revision's identity, digest and annotation count, rather than embedding every
historical task body again. Media bytes and archived import-document bytes are
not embedded. Keep the project directory or verified workspace backup for those
complete historical records. This avoids quadratic growth during long annotation
sessions and recursive growth when exchanging a document repeatedly.

Opening a storage-version1 workspace with this implementation upgrades its
storage to version2 transactionally after validation. Schema and task digest
versions remain unchanged. Older binaries refuse version2; they must not write
upgraded projects with their older cleanup logic. Preserve a pre-upgrade backup
if an older binary is needed. Rollback means restoring that original backup to
a separate path, not editing the storage-version number back to1.

An import replaces annotations in an existing task after explicit confirmation.
Admit the matching source video first. Source hash, size and exact timeline must
match. Anchored annotations additionally require the corresponding playback
identity and are checked against the receiving project's frame map. Imported
paths are never used to locate or read files. Import creates a new local
revision and receipt, maps annotation IDs and marks imported annotations
unreviewed. The original document, review flags and history are stored as an
immutable, referenced project artifact; the task contains its descriptor.
Garbage collection and backup preserve that artifact through all historical
owners. Its contents are passive source provenance and are not executed or
treated as current approval. A standalone professional JSON carries the current
task in full and historical artifact descriptors, not those artifact bytes.

Version1 accepts the currently supported annotation fields. Nonempty custom
attributes or evidence-file references are rejected until their own admission
workflow is available. The failure is explicit; fields are not silently removed.
Unknown versions, malformed data, invalid digests, stale revisions and reused
command IDs with different requests are rejected without replacing saved data.

The HTTP command/document limit is1MiB; the exporter reserves4KiB of that limit
for the import command envelope. The current task must remain within
768KiB and individual reports within256KiB. Available history can cause an
export to exceed its limit. Keep the complete project/backup in that case;
the exporter refuses rather than dropping historical records.

## Analysis reports

`POST /api/v1/projects/current/tasks/{task_id}/analysis-reports` accepts
`command_id`, `expected_revision`, `expected_sha256` and `report`.
Reports have an immutable ID, file identity, stream/time domain, optional
interval and annotation subject, method/version/parameters, dependency digest,
typed values/units and declared availability. Existing IDs cannot be changed
to different results; publish a new report ID.

File declarations, measurements, observations and inferences have different
types. Unavailable values are null with a reason; nonfinite JSON is rejected.
An observed audio description includes a declared observer and input identity.
Those fields establish attribution, not authenticated identity or proof that
someone actually listened. This endpoint admits externally produced reports;
it does not itself run a media analyzer or certify semantic correctness.

The **媒体参数与客观分析** panel can run objective analysis on the registered
playback proxy. Its durable job records the frozen task/media identity, leases,
cancellation, retries and candidate artifact. The candidate is saved only after
confirmation, in one revision and receipt. Closing refuses to discard an active
worker's database. Editing a bound segment invalidates that interval result.
Original-source reports remain distinct from proxy reports: transcoding can
change decoded peaks and metadata. The workbench does not silently measure an
external original file via a path imported from JSON.

Current objective methods include decoded frame timing and mono/stereo audio
integrated loudness, sample peak, true-peak estimate and a documented silence
rule. Missing tools, no track, unsupported channel layout, silence and invalid
measurement windows have explicit unavailable reasons. True peak is an estimate;
this implementation does not claim metering certification. Waveform statistics
do not establish speech content, timbre, emotion or semantic annotation accuracy.

Export calculates report freshness from its actual dependencies. Adding a
report does not make it immediately stale. An unrelated label change leaves a
whole-file measurement fresh. Changing a bound segment's boundaries makes that
interval report stale while preserving the original result and source window.

## HTTP and CLI

Export: `GET /api/v1/projects/current/tasks/{task_id}/export?format=professional`.
Import: `POST /api/v1/projects/current/tasks/{task_id}/professional-import` with
`command_id`, `expected_revision`, `expected_sha256` and `document`.
All API calls retain the workspace's token/origin checks and stable receipts.

```text
trace-annotate workspace professional export --project PROJECT --task-id TASK --output NEW_FILE.json
trace-annotate workspace professional import --project PROJECT --task-id TASK --input FILE.json --command-id COMMAND --expected-revision REVISION --expected-sha256 TASK_SHA256
```

The CLI refuses to overwrite an existing output. A professional JSON file is
an exchange document, not an installed application, portable media archive,
automatic annotation verdict or configurable annotated-video export.

## Annotated video

The **导出带标注的视频** panel keeps the complete registered playback image
and adds a black panel with white text at the top, bottom, left or right.
Font sizes, panel thickness, fields and tracks are configurable. Full text is
wrapped; overflow or a missing glyph is rejected rather than silently clipped
or reduced. Short segments may require pausing to read. The bundled unmodified
Noto Sans CJK SC font retains its SIL OFL1.1 license and copyright notice in
`trace_annotate/workspace/fonts/`; it is separate from the project MIT license.

The preview is one exact decoded frame, not a short rendered video. Generation
uses a frozen task/media/font/recipe, preserves decoded frame timestamps and
the final-frame duration, and validates the whole output before five immutable
artifacts and their queue commit become downloadable. Cancelled and failed
attempts expose no partial video. Cancelled work can be explicitly resumed with
the same frozen input after its old processor exits.

The job's video and JSON downloads use the same input revision and task digest.
Later edits are shown as a different current revision; they do not rewrite an
old export. The paired JSON contains the complete frozen task and reports, with
`history.scope=not_included`. Use the ordinary professional export plus the
project backup when the local audit index and historical files are required.
Downloads are bounded to512MiB on the server and may be further limited in the
browser. Current rendering supports the explicitly declared H.264/AAC-or-copy
recipe and bounded panel/frame counts; it does not promise every codec, HDR
conversion, complex writing system or arbitrary document size.
