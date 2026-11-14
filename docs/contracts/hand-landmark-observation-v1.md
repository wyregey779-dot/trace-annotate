# hand-landmark-observation-v1

## Scope

This optional provider records MediaPipe Hand Landmarker keypoint observations. It does not assert persistent hand/person identity, touch, grasp, support, intent, or any TraceAnnotate action class.

## Frozen model identity

- Provider: `mediapipe-hand-landmarker-float16-v1`
- Official bundle: `hand_landmarker/float16/1/hand_landmarker.task`
- Size: `7819105` bytes
- SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- Source: <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
- Official documentation: <https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker>
- Model card: <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf>
- Model license: Apache License, Version 2.0, stated by the official model card under `LICENSED UNDER`
- Runtime repository license: Apache-2.0, <https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>

The runtime never downloads this file. A missing bundle or missing MediaPipe runtime returns `not_run`; a size, digest, regular-file identity, load, inference, worker-process, or output-contract failure returns `failed`.

The workspace accepts the bundle only through the explicit `hand_model_path` constructor argument or `TRACE_ANNOTATE_HAND_MODEL`. Submission freezes a path identity plus the measured size and SHA-256 without exporting the absolute path. Every current candidate records a `hands` model-run receipt, including an honest `not_run` receipt when the bundle is absent.

## Observation identity

Each observation binds one scene index, canonical source-frame index, reduced source PTS, caller evidence SHA-256, decoded-pixel SHA-256 and hand model-run id. Its identifier is a deterministic digest of those fields plus the hand index.

Each detected hand contains exactly 21 named image landmarks and exactly 21 world landmarks. Image `x/y` are retained in normalized coordinates and projected to pixels; the third value in `landmarks_pixel` remains MediaPipe's normalized relative-depth value and is not a pixel or metric depth. World coordinates are retained in meters as emitted by MediaPipe. The raw `Left`/`Right` category and score are stored as `mediapipe_category_score_uncalibrated`.

The provider sets `handedness_mirrored_input_assumption_verified=false`: no caller-supplied mirror-state evidence exists. Handedness, landmark geometry and boxes remain unreviewed model outputs.

## Isolation, cancellation and resource boundary

The workspace process never imports or instantiates the MediaPipe native runtime. It verifies the exact local bundle, then starts an owned one-shot subprocess for each frame. The parent sends one canonical JSON identity header and the exact bounded raw BGR byte payload. The worker revalidates the request, decoded-pixel SHA-256, model file identity, size and digest before loading the offline model. It returns one canonical JSON response; absolute model paths and native exception messages are not exported.

The parent bounds the request header to 16 KiB, the decoded frame to 16,777,216 pixels and 64 MiB, the worker response to 16 MiB, and worker stderr diagnostics to 1 MiB. Standard output and standard error are drained through capped pipes rather than unbounded files. Exit status, canonical JSON shape, response identity, frame hash, landmark counts, numeric ranges and the configured hand count are validated before an observation is accepted. Unit tests use an explicitly fake worker runtime. `scripts/check_hand_landmarker_real.py` is the separate opt-in acceptance check for the exact official bundle and caller-supplied licensed image fixtures; its report distinguishes runtime execution from whether a hand was detected.

Cancellation and `close()` are polled while inference is running. On Windows each worker is assigned to a kill-on-close Job Object before frame bytes are sent; a host that prevents safe assignment fails with `hand_landmark_worker_isolation_unavailable` instead of silently weakening process-tree ownership. POSIX workers use a new session and `killpg`. Timeout, cancellation, close, oversized output, malformed output, worker crash or native hang terminate the owned process tree. Cancellation raises without returning partial observations; timeout and worker/protocol failures return `failed`. Concurrent calls are tracked independently so `close()` terminates every owned worker rather than only the most recently started process. The frozen result value is `cancellation_boundary=owned_subprocess_polled_during_inference`.

One current vision job admits at most 16 uniformly distributed decoded keyframes to this native provider and enforces a 120-second whole-phase deadline. Every omitted keyframe receives an explicit `not_run` reason (`hand_landmark_sampling_budget_exhausted` or `hand_landmark_phase_timeout`). The model-run receipt records total, admitted, successful and omitted counts plus `complete`; partial coverage is never presented as full-video hand analysis.

## Explicit non-claims

- A hand box overlapping an object is not touch.
- Similar hand/object motion is not grasp or control.
- Handedness does not identify a persistent person.
- No keypoint result alone establishes `拿起`, `放下`, `打开`, `关闭` or `递交`.
- The official model card calls the model experimental, excludes surveillance/identity recognition, and warns that gloves, occlusion, held objects, blur and extreme capture conditions can reduce validity.
