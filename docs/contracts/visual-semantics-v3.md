# visual-semantics-v3 data contract

## Compatibility

- New semantic documents use `visual-semantics-v3`.
- A professional task remains bounded to at most 32 explicit model-run receipts; configured modules are not represented by silently omitting their `not_run` or `failed` receipt.
- `visual-semantics-v1` and `visual-semantics-v2` remain readable and are validated against their original exact field sets and digest rules.
- A v3 claim adds the always-present, nullable fields `object_ref` and `recipient_ref` to the v2 claim. Their values are ASCII identifiers or `null`.
- These references declare identity links only. Structural validation does not prove that a referenced subject, object or recipient was correctly tracked or identified.

## Digest binding

- A v3 claim digest includes every v2 digest field plus `object_ref` and `recipient_ref`; `id` remains excluded.
- The digest input is `{"digest_version":"visual-semantics-v3","claim":<v3 projection>}`.
- A v3 semantic digest uses digest version `visual-semantics-v3`, the literal semantic version and claim digests sorted by claim id.
- Changing either new reference changes the claim digest and the containing annotation review digest. A stale review binding is not retained after such an edit.

## Action candidate contract

An action claim identifies this contract with `method.parameters.action_contract="action-candidate-v3"`. It must bind a reduced, ordered PTS interval, a provider model-run id, a scored output and one or more registered evidence artifacts. Participant requirements are:

| action | required references |
| --- | --- |
| `走` / `跑` | `subject_ref` |
| `拿起` / `放下` / `打开` / `关闭` | `subject_ref`, `object_ref` |
| `递交` | `subject_ref`, `object_ref`, `recipient_ref`; recipient must differ from subject |

The built-in `subject-translation-v1` provider covers only `走` and `跑`. It measures sampled person-box translation and produces review candidates, not confirmed actions or persistent human identities. The five object-interaction classes have no local provider and therefore return `status=not_run`, `reason=provider_unavailable`, and no candidate. A supported class that ran but did not cross its declared threshold returns `status=unknown`, `reason=no_candidate_met_threshold`.

The optional pinned `videomae-small-ssv2-aux-v1` provider is a different
contract. Under the default v1 sampling policy, each scene with at least
sixteen registered decoded content samples has one canonical uniformly selected
sixteen-frame window classified in an owned, cancellable subprocess. The opt-in
v2 local-window policy below has a separate bounded coverage contract. Under
either policy, only `close`, `open`, `pick_up` and
`put_down_surface` may be projected, and only as an `unspecified`, unreviewed
`videomae-window-hint-v1` claim. The claim keeps `subject_ref`, `object_ref` and
`recipient_ref` null and does not set `action_contract`; therefore it cannot
satisfy or replace an `action-candidate-v3` claim. `walk`, `run` and `hand_over`
remain explicitly unsupported by this checkpoint. Missing local bytes are
`not_run`; an unverified identity, worker/protocol error or inference error is
`failed`; none of those outcomes emits a semantic claim.

Every successful auxiliary hint binds the exact registered playback SHA-256,
scene index, reduced playback PTS window, all sixteen decoded-frame identities,
three durable preview artifacts, the pinned model/config/preprocessor hashes,
implementation fingerprint and a path-redacted owned-process receipt. Legacy
candidates whose frozen job configuration predates this module remain readable,
but they may not carry an undeclared auxiliary sampling object or model run.

The default `videomae-window-sampling-v1` policy above is unchanged. New jobs may
opt into the experimental `videomae-window-sampling-v2` policy by setting
`TRACE_ANNOTATE_ACTION_VIDEO_WINDOW_POLICY=per_scene_local_decoded_content_windows_v2`
before submission. The policy is frozen into the job identity and the model-run
record. It partitions existing decoded content samples into local sixteen-frame
windows, admits only windows whose adjacent sampled PTS are at most 0.5 seconds
apart and whose endpoint span is at most 8 seconds, and executes at most eight
windows across the job within a 300-second phase budget. These are conservative
engineering limits, **not** evidence that the windows match VideoMAE's training
distribution or that action recognition accuracy is known. It does not decode
additional frames; an event absent from the upstream content samples cannot be
recovered. A long single shot with sparse samples can have zero eligible windows.
The v2 staging gate admits at most 128 MiB of decoded input per window and
512 MiB across worker calls, while preserving at least 512 MiB of free disk
before each call; an exhausted gate is reported as `not_run` for that window.
`verified_success_input_bytes` counts only successful worker results. The
validator recalculates every such result from its sixteen verified uint8 BGR
frame shapes and requires equality with the worker's `total_input_bytes` and
the aggregate success count. A failed worker may have staged input without a
successful window receipt; the runtime still charges those bytes to the
512 MiB staging gate, but they are not misreported as verified successes.
The 300-second phase deadline is checked around each frame load and passed into
the owned worker's cancellation callback. A single blocking frame read cannot
be interrupted in the middle; the next frame is not read after the deadline.
The result records exact selected frame/PTS/hash identities, omitted window
envelopes with reasons, and intervals outside successful window envelopes.
An envelope is only the first-to-last sampled PTS, never continuous observation:
the time between decoded frames remains unobserved. A failed, cancelled,
unavailable, over-budget or skipped window yields no successful hint. The v2
sampling contract leaves the `videomae-window-hint-v1` authority and its
`unspecified`/unreviewed export unchanged; it never produces an
`action-candidate-v3` interaction fact. Historic v1 results retain their exact
original validation and projection behavior.

An additional, explicitly opt-in `videomae-window-sampling-v3` policy uses
`TRACE_ANNOTATE_ACTION_VIDEO_WINDOW_POLICY=per_scene_local_redecoded_playback_windows_v3`.
It is a separate job identity and does not reinterpret existing v1/v2 records.
For up to eight candidate local intervals, the planner takes at most four
six-second scene/request intersections. Each selected interval must contain
sixteen distinct admitted frame-map entries with strictly increasing playback
PTS, adjacent gaps no greater than 0.5 seconds and an endpoint span no greater
than eight seconds. At most 64 unique frames are re-decoded from the registered
playback in one cancellable FFmpeg pass. The decoder verifies each output PTS
against `proxy_pts × proxy_time_base`, plus the decoded RGB checksum against
the corresponding PNG pixels. The frozen result records the playback and frame
map SHA-256 values, exact source/proxy PTS, PNG hashes, sample-plan fingerprint,
filter and decoder log hashes, and command digest. Source PTS can have a
different origin and is never substituted for playback PTS.
Intervals skipped before frame inspection are listed with
`global_candidate_budget`; they are not mislabeled as ineligible scenes.

The decode phase has a 180-second limit inside a 300-second action phase,
a 512 MiB decoded-PNG cap, a separate 512 MiB staged worker-input cap and a
512 MiB free-disk reserve. Each worker
window is capped at 128 MiB of uint8 BGR input. Missing weights or an
ineligible interval yields `not_run` without decoding; cancellation or lost
lease cannot publish a partial semantic success. The `coverage` object lists
successful sampled envelopes, omitted candidates and unobserved intervals;
neither an envelope nor a negative hint proves continuous observation or
absence of an action. V3 retains the same unreviewed, `unspecified`
`videomae-window-hint-v1` export with null participant references. The
checkpoint's non-commercial license and unknown independent accuracy remain
unchanged. A fake-model structural test does not count as a real forward pass,
human truth or per-action accuracy evidence.

For a committed successful v3 job, the exact decoder PNGs and PTS/filter/command
receipts are registered as immutable `objects/<sha256>` artifacts. Their pins
commit atomically with queue success; `result()` rehashes these objects and
reparses the decoder PTS receipt, including after a formal workspace backup
and restore. After each video job finishes, the runner checks the objects it
newly registered. It reclaims unowned objects before claiming the next job.
An active publisher or processor defers cleanup until another task finishes or
the runner polls again. Each cleanup request covers at most 1,024 object IDs.
Shutdown retries pending cleanup after the other workers have stopped. If
an active writer still prevents cleanup, closing reports the pending count
and the workspace can be closed again when the writer finishes.
A process crash between registration and cleanup can leave unowned objects.
Use `unbound_artifacts()` to inspect them and `sweep_unbound_artifacts(apply=True)`
after processors exit to reclaim them. The
backup closure preserves the committed v3 result, but this does not establish
that every unrelated temporary keyframe/content artifact needed to **apply an
unapplied candidate after restore** is portable; that older workflow requires
separate evidence-closure work and must not be claimed as verified here.

Every vision action candidate binds its source frames, reduced PTS values, exact decoded-sample SHA-256 values, provider id and action model-run id. On import, each referenced sample is rehashed before a compact immutable preview is projected. The complete semantic document is validated against the projected content-addressed artifact ids before those objects are published.

## Optional model licensing and distribution

TraceAnnotate's repository code is MIT licensed, and the VideoMAE adapter uses
the Apache-2.0 `transformers` runtime. The frozen
`MCG-NJU/videomae-small-finetuned-ssv2` checkpoint is a separate work licensed
CC-BY-NC-4.0 at its pinned upstream revision. Its weights are not bundled in
the wheel, source distribution, repository or release assets, do not inherit
the repository's MIT license, and do not provide a commercial-use grant. A
user must obtain and configure the local model directory separately and is
responsible for determining whether the weight and input-data licenses permit
the intended use.

The execution receipt records the pinned checkpoint/config/preprocessor
digests and license source when the identity is verified. Missing local bytes
are `not_run`; a mismatched identity is `failed`. Neither status permits the
adapter to inherit the pinned model name, labels or license attribution.

## Bounded portable visual projection

The durable vision-job result is the authoritative local record of sampled
object boxes, sparse object tracks, geometric relationship candidates and hand
landmarks. A professional JSON document does not embed that complete result:
the task is limited to 768 KiB and the portable document to 1 MiB. Instead,
apply builds one deterministic, path-redacted cross-section and records its
source candidate SHA-256, exact total/projected/omitted counts, byte and frame
budgets, and whether the projection is complete. The complete job artifact and
project backup are required when the omitted raw observations are needed.

The portable cross-section is limited to 128 KiB of canonical method
parameters and twelve exact decoded evidence frames. It includes at most 24
object observations, 12 object-track summaries, 12 geometric relationship
candidates and four complete hand observations. Sampling is deterministic and
spread across the ordered result rather than taking only its prefix. Each data
claim binds an existing model-run receipt and an immutable preview generated
from a keyframe whose scene, frame index, reduced PTS and evidence SHA-256 all
match. A mismatch fails apply; evidence is never borrowed from another decode
that happens to use the same frame number.

`object-observation-projection-v1` states only that the frozen COCO-80 provider
returned unreviewed box candidates on sampled frames. Its same-scene,
same-class geometric associations do not establish identity, object permanence
or a validated continuous track. `hand-landmark-observation-projection-v1`
retains a bounded set of complete 21-point observations but does not infer
touch, grasp, held state, action or person identity. Raw handedness scores stay
uncalibrated and the mirror assumption remains unverified.

`object-relation-candidate-projection-v1` is an `unspecified` geometric summary,
not an action claim. Its portable vocabulary is deliberately conservative:
`near`, `contact_candidate`, `co_moving_candidate`, `held_candidate` and
`control_transfer_candidate`. In particular, the internal v1 rule labels
`held_by` and `control_transfer` are projected only as the latter two candidate
names. The claim never sets `action_contract`, and it cannot establish pick-up,
put-down, open, close, hand-over, grip, possession or intentional transfer.

`visual-candidate-projection-manifest-v1` is a loss manifest, not a statement
about video content. Missing or unavailable providers emit no successful data
projection. When any portable limit is reached, omission is explicit; the
implementation must not silently relabel the retained subset as the full
result.

## Non-claims

- `not_run` does not mean the action is absent.
- `unknown` does not mean a negative classification.
- The five unsupported object actions are not detected, inferred or renamed from another vocabulary.
- A VideoMAE window hint does not establish an action participant, object state change, exact action boundary or completed action.
- A portable visual projection is not the complete vision-job result; its manifest and the project backup define the loss boundary.
- A hand landmark or geometric relationship candidate does not establish touch, grip, possession, control or transfer.
- A syntactically valid participant reference does not establish real-world identity, object permanence or cross-shot tracking accuracy.
