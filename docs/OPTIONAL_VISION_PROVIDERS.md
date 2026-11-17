# Optional local person boxes and face landmarks

The workspace can run two additional local providers. Their outputs are
unreviewed candidates. They do not identify people, certify emotions or establish
annotation accuracy. Missing dependencies or weights are reported as `not_run`;
failed loading, corrupt models and invalid outputs are reported as `failed`.

## Install and configure

Install `.[persons]` into the environment that serves the workspace. This adds
ONNX Runtime and OpenCV. Set `TRACE_ANNOTATE_PERSON_MODEL` to a local ONNX file.
No model download occurs inside a job or when opening the workbench.

For landmarks, use a separate virtual environment with `.[landmarks]`. Set
`TRACE_ANNOTATE_LANDMARK_PYTHON` to that environment's Python executable, and
`TRACE_ANNOTATE_LANDMARK_MODEL` to a local `face_landmarker.task` file. The server
uses that interpreter with `-I -B`, loading the adapter from the server's own
package location. It does not copy or import code from a different source tree.
The worker is limited to 180 seconds, 512 decoded frame filenames, four faces
per frame, 16 MiB of JSON output and 1 MiB of diagnostics. Cancellation and
timeout terminate and reap the owned child process. Image files stay in the
workspace; full decoded images are not held in a second batch in the server.

For a PowerShell session, supply actual absolute paths before starting the CLI:

```powershell
$env:TRACE_ANNOTATE_PERSON_MODEL = 'D:\models\object_detection_nanodet_2022nov.onnx'
$env:TRACE_ANNOTATE_LANDMARK_MODEL = 'D:\models\face_landmarker.task'
$env:TRACE_ANNOTATE_LANDMARK_PYTHON = 'D:\venvs\landmarks\Scripts\python.exe'
trace-annotate workspace serve --project 'D:\annotation-workspace'
```

These example paths are deployment placeholders. Environment variables apply to
this shell and its child server; they do not modify system configuration. To
leave a provider disabled, unset its variables before launching. The Python
`VisionJobs` constructor also accepts `person_model_path`,
`landmark_model_path`, and `landmark_python`. An explicit nonexistent model path
selects an unavailable provider for scoped tests; the result must remain
`not_run`. CLIP and VLM have their separate existing constructor settings. These
new settings do not freeze or replace the CLIP/VLM adapters' identity contracts.

## Model provenance and license

The supported NanoDet layout is OpenCV Zoo's
[`object_detection_nanodet`](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_nanodet).
The tested `object_detection_nanodet_2022nov.onnx` is 3,800,954 bytes, SHA-256
`4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`.
OpenCV Zoo distributes this model directory under Apache-2.0. The adapter
independently decodes the pinned six-output ONNX Runtime graph, filters COCO
person class 0, applies person-only NMS, and restores letterboxed boxes to
decoded-frame **xyxy** coordinates. A different file does not inherit that
model identity or license: it is recorded as unverified local weights.

The tested MediaPipe model is the official
[`face_landmarker.task` float16/1 asset](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task),
SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
MediaPipe's [Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
links the model bundle and component model cards. The local deployment must
retain the associated Apache-2.0 license and model cards; model weights are not
bundled by this Python package. Other local task files retain an unverified
license status. The tested CPU worker runtime is MediaPipe 0.10.31 on Windows.
Other Python/OS combinations require their own installation and execution
checks; the optional extra is not a cross-platform certification.

## Evidence and limitations

Each new job freezes model paths, weight digests, adapter implementation digest,
person runtime version, worker Python digest and MediaPipe distribution metadata
digest. A changed configuration creates a different job identity. An older
queued job is rejected with `VISION_RESUBMIT_REQUIRED` when it cannot execute
under its frozen configuration. Previously committed immutable candidates remain
readable and applicable through the historical contract. A model is hashed
again around loading; a changed model is not silently substituted.

New candidates contain separate `person`, `landmarks` and `expressive` status
records and model runs. A successful landmark forward pass with no usable face
does not become a successful expression analysis. Face geometry filters reject
small, rotated, side-facing and out-of-frame faces. Eye openness is a single
frame geometric candidate; it does not prove blinking, gaze, mood or inner
emotion. Brow tightening requires a neutral personal baseline that this adapter
does not supply.

Person boxes take precedence over face-only geometry for framing and sampled
translation tracks. Association is one-to-one per frame, confined to one scene,
and cannot bridge a gap longer than five seconds. This is geometric association,
not person re-identification. A person detector can mistake statues or screen
people for real people, and can miss small, occluded or distant bodies. Zero
detections do not prove the absence of people. Scores are not calibrated
probabilities. No natural-video precision/recall threshold is claimed here.

Only tentative walk/run translation remains supported. Picking up, putting
down, opening, closing and handing over have no provider and stay `not_run`.
Raw sampled-frame measurements remain in the candidate artifact. Portable
segment claims use its representative keyframe and retain their evidence/model
run links. Applying a candidate keeps `origin=model_candidate` and
`review_status=unreviewed`; human signoff is a separate operation.

## Read-only hand and object geometry

For a completed vision job, the authenticated workspace endpoint
`GET /api/v1/projects/current/jobs/{job_id}/hand-object-geometry` derives
same-frame pixel measurements from the verified candidate. The response binds
each pair to its hand and object observation IDs, scene, frame, rational PTS,
and immutable frame hash. It reports the nearest hand landmark's distance to
an object box, hand-box distance, and box intersection over union. A zero
distance or overlapping boxes do not establish touch, possession, or an action.

The query does not change the task or its review decisions. Its source hash
identifies the complete vision result used for the calculation. Missing hand
or object providers return `not_run`; a valid sampled result with no comparable
pairs returns an empty `success` result. Coverage is limited to sampled frames.
The result retains at most three nearest object boxes per hand and 1,024 pairs
overall, with omitted counts and explicit markers for unavailable or truncated
provider output. This derived view is not stored in the candidate or included
in the portable professional JSON projection; retain the vision job and project
backup when these measurements matter.
