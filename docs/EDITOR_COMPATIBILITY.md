# Editor compatibility and retained workflows

The general and embodied editors remain the primary independent-file tools.
The persistent batch workbench is experimental and is not their replacement.
Existing `video-ui`, `video-open`, `embodied-ui` and format/QC commands remain
available. Workspace links open the original editors in new tabs so a current
workspace draft is not discarded.

This inventory is a source and test map, not a claim that every product gate or
every real-video semantic review has passed. Record actual run results separately.

| ID | Retained capability | Implementation and regression entry |
| --- | --- | --- |
| K01 | Open, play, pause, seek and mark times | `temporal/ui/annotator.html`, `embodied/ui/annotator.html`; `check_general_editor.py`, `check_embodied_editor.py` |
| K02 | Labels, descriptions, tracks and ordered/duplicate tags | temporal editor and `temporal/schema.py`; `check_editor_integrity.py` |
| K03 | Separate tracks and lanes for overlapping segments | temporal editor timeline; `check_general_editor.py` dense/overlap navigation |
| K04 | Add, edit, delete, undo and redo | temporal editor history; `check_general_editor.py`, `check_editor_integrity.py` |
| K05 | Dirty form prompts and explicit browser draft recovery | temporal editor; `check_editor_integrity.py`, `check_media_identity.py` |
| K06 | Native temporal v1/v2 JSON import/export and continued editing | `temporal/schema.py`, `formats.py`; general and media identity browser checks |
| K07 | CSV, SRT, WebVTT, explicit subtitle track and loss disclosures | `temporal/formats.py`, `commands.py`; parsed browser downloads in `check_general_editor.py` |
| K08 | Manual review and invalidation after content changes | temporal editor; `check_editor_integrity.py` |
| K09 | Full media hash, cancel, verified rename and legacy binding | `temporal/ui/media-hash.js`, `media.py`; `check_media_identity.py` |
| K10 | Prepared compatibility copy, original/proxy timing and identity | `temporal/preparation.py`, `project_server.py`; `check_prepared_editor.py`, `check_media_admission.py` |
| K11 | Embodied task description and independent expected steps | embodied editor; `check_embodied_editor.py` and quality tests |
| K12 | Step number, verdict, error type and evidence text | embodied editor/schema; embodied browser and validator tests |
| K13 | Explicit Working/Idle and complement operation | `embodied/complement.py`; embodied browser and complement tests |
| K14 | QC, procedure comparison and agreement | `embodied/quality_pipeline.py`, `quality_report.py`, `differ.py`, `agreement.py`; quality pipeline tests |
| K15 | Embodied subtitle, CVAT and action projections | `embodied/exporters/`; exporter and CLI tests; external application compatibility remains separately qualified |
| K16 | Existing CLI and packaged resources | `cli.py`, `temporal/commands.py`; `check_installed_package.py`, CLI and Windows launcher tests |

The optional workspace adds independent tasks, revision-bound commands, durable
receipts, exact mapped frame navigation and candidate job storage. It currently
does not provide all original undo/draft/review/format workflows through its own
form. A development snapshot is not a portable project package. Opening an old
editor does not migrate or synchronize workspace annotations.

## Source, browser and package gates

`check_workspace_editor.py` covers actual playback buttons, range completion,
looping, frame-to-play transitions, keyboard guards, task isolation, responsive
reflow and the served classic tool routes. Original editor regressions remain
in CI alongside it; the source files are not replaced by the served navigation.

For an installed-wheel check, install the wheel and browser-test dependencies in
an isolated environment and run that environment's Python with `-I`, supplying
`--installed-wheel PATH` to `check_workspace_editor.py`. The check must reject
checkout imports and compare installed resources with the wheel. The isolated
smoke alone is not a browser acceptance or clean-machine qualification.

User-profile Chrome interaction, actual listening, full real-media workflows,
the full K01–K16 task acceptance matrix and G00 sample/reference freezes must be
reported using their own evidence. Headless synthetic checks do not replace them.

## Remaining integration order

1. Verify original tools and fix playback/accessibility regressions.
2. Add explicit, lossless legacy adapters and protected migration to new projects.
3. Connect persistence, ordinary drafts and conditional undo to the original workflow.
4. Integrate templates, full batch intake/cancellation, candidate adoption and review.
5. Provide media-aware portable project round trips and scope-specific release gates.

Any default-entry change requires feature parity and actual workflow evidence.
Source files, original media and old JSON must remain usable throughout.
