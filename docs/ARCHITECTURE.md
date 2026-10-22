# Architecture

TraceAnnotate provides local Python modules, command-line tools and a browser editor for media review and temporal annotation. The core annotation workflow does not require a hosted service or model API.

## Modules and data flow

| Module | Responsibility |
| --- | --- |
| `core` | Staged copies, size and partial-hash verification, perceptual hashes, duplicate clustering and field provenance |
| `extractors` | Optional visual metrics, face detection, OCR and scene-cut adapters; CLIP inference is not implemented |
| `heuristics` | Rules that turn available measurements and path hints into review candidates |
| `audit` | Contact sheets, a prioritized review queue and recorded human overrides |
| `embodied` | Annotation structure checks, idle intervals, expected-step comparison and agreement calculations |
| `exporters` | SRT, frame labels and a CVAT-style XML export |
| `ui` | A packaged HTML editor for loading local video, marking intervals and downloading annotations |

These modules can be used separately. The example scripts show how to compose selected operations; they are not a complete automatic labeling pipeline. External annotation formats require compatibility checks against the destination tool.

A typical temporal annotation workflow is:

```text
Local video → browser editor → annotation JSON
                                  ↓
                       validate and calculate idle
                                  ↓
                          review report / SRT
```

The [quickstart](QUICKSTART.md) exercises the JSON-to-report path with a synthetic fixture.

## Staging lifecycle

`StagingManager` copies a batch from a source root into a separate staging directory. It checks path boundaries, checks available disk space before copying and can verify the copied size and a hash of the file's first megabyte. This is a partial-content check, not a full-file checksum.

Consumers can call `lock_in_use` before processing a staged file and `unlock_in_use` after processing. Cleanup protects locked entries and their resolved targets. It unlinks eligible file symlinks by their directory entry, skips directory symlinks and Windows junctions, and reports deletion failures.

The caller must control the staging directory while operations run. Path checks do not provide atomic protection against concurrent directory replacement. A free-space check is not a reservation, and the `read_only_source` configuration field does not change filesystem permissions. Source permissions, backup policy and consumer lock lifetimes belong to the application using the stager.

## Annotation checks

The project uses its own clip and interval schema. The [annotation SOP](EMBODIED_ANNOTATION_SOP.md) defines its labels and review conventions.

The validator checks supported statuses, segment fields, timing constraints and evidence requirements. It cannot decide whether a label accurately describes the video. Idle calculation needs a known clip duration to include the interval after the last working segment. Agreement statistics compare annotators on sampled time bins; they do not establish label accuracy.

See the [roadmap](ROADMAP.md) for planned format round-trip checks and measured performance work.

## Provenance and human review

A record's `fill` mapping identifies the origin or review state of individual fields:

```json
{
  "primary_entity": "stone sculpture",
  "face_count": 0,
  "fill": {
    "primary_entity": "human_confirmed",
    "face_count": "auto_metric"
  }
}
```

The available states include `auto_metric`, `auto_rule`, `auto_path`, `auto_path_confirmed`, `candidate_only`, `pending_review`, `pending_tool` and `human_confirmed`. They describe how a field was populated; they are not confidence scores.

Human overrides store revisions and support rollback. The caller explicitly applies them to records and saves the resulting table. The JSON files are ordinary local files, not signed audit records or a concurrent database. See [provenance fields](PROVENANCE_SCHEMA.md) and the [review workflow](FAST_AUDIT_WORKFLOW.md).
