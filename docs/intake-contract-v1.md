# Async Media Admission Contract (intake-contract-v1)

Status: proposed baseline for T07/T08. The old synchronous `POST /imports` and
`store.import_source` are preserved unchanged. This document specifies the new
async route added in `trace_annotate/workspace/intake.py`.

## State machine (per intake row)

```
queued ──claim──▶ preparing ──(commit)──▶ succeeded
  │                  │
  │                  ├──request_cancel──▶ cancel_requested ──worker fence──▶ cancelled
  │                  │
  │                  └──(error)──▶ failed ──retry_failed──▶ queued
  │
  └──request_cancel──▶ cancelled
```

Terminal states: `succeeded`, `failed`, `cancelled`. A previous process that died
while a row was `preparing` is marked `failed / INTERRUPTED` on open — the staged
input is preserved but never silently re-run.

## Request side (returns immediately, HTTP 202)

1. Receive the full upload bytes into a temp file and `fsync`.
2. Verify whole-file `sha256` and `size`. If the client declared
   `expected_sha256` / `expected_size`, mismatch → `422 MEDIA_IDENTITY_MISMATCH`
   before any intake row is written.
3. Move the bytes into the workspace-owned staging area
   `tmp/intakes/<intake_id>/source` and re-hash; mismatch → rejected.
4. Insert one intake row `queued` with `source_sha256`, `source_size`,
   `template_id`, `template_version`, `staged_path`, and (optionally) a
   `command_id` receipt.
5. Return the intake view immediately. 202 is NOT admission success.

## Worker side (conservative single prepare)

- `claim(worker_id)` picks the oldest `queued` row, bumps `attempt` and
  `generation`, sets `preparing` with a lease.
- Re-hash the staged source; it must match the recorded identity.
- Run the verified preparation pipeline (ffmpeg full decode) OUTSIDE the SQLite
  write transaction; the worker cooperatively checks a `cancelled()` callback.
- On success, in one short transaction: insert media (dedup by source sha256),
  bind immutable playback/frame-map artifacts, create exactly one task, mark the
  intake `succeeded` with `media_id`/`task_id`.
- On error: mark `failed` with a machine-readable `error_code`; the staged
  input is kept for an explicit `retry_failed`.

## Idempotency and recovery

- A repeated enqueue with the same `command_id` and identical request body
  returns the original receipt; same `command_id` with different content →
  `IDEMPOTENCY_CONFLICT`.
- A lost response is reconciled by polling the intake row, never by guessing a
  new idempotency key and re-uploading.
- Restart: `preparing` rows owned by a dead session → `failed / INTERRUPTED`
  with input preserved.
- One intake row creates at most one task; running the queue twice cannot
  duplicate the task.

## Explicit non-goals for this slice

- Real multi-worker parallelism is T18 (later). This slice claims one intake at
  a time and shares the existing prepare budget.
- The browser UI wiring (pause/resume/cancel/retry buttons) is T09.
- Portable project export/import is T14.
