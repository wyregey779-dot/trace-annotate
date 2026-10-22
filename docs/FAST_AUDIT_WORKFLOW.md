# Visual review workflow

Use contact sheets to compare candidate frames, then inspect the original image or video before confirming a label. A thumbnail can help locate a questionable result, but cannot establish every detail in the source media.

## Generate a contact sheet

Pass explicit image paths so the command works consistently across shells:

```bash
trace-annotate asset-sheet thumbs/frame-01.jpg thumbs/frame-02.jpg -o output/contact-sheet.jpg --cols 4 --title "Frame review"
```

Create the output directory first. Missing or unreadable images appear as placeholders. Each cell shows a thumbnail and a shortened filename. Keep the source paths available for follow-up inspection.

The Python review-queue helper prioritizes unresolved entities and conflicts, blacklist or face-review flags, video candidates, and unsuitable frames. These priorities are rules for organizing work; they do not replace a reviewer's decision.

## Record a decision

After checking the original frame, record the decision with its author, basis and reason:

```python
from trace_annotate.audit.overrides import HumanOverrideManager

manager = HumanOverrideManager("output/review/human_overrides.json")
revision = manager.record_override(
    "frames/frame-01.jpg",
    primary_entity="stone sculpture",
    author="reviewer-01",
    basis="original_frame_inspection",
    reason="Confirmed the subject against the original image.",
)

record = {
    "rel_path": "frames/frame-01.jpg",
    "primary_entity": "unresolved",
}
manager.apply_to_record(record)

print(record["primary_entity"])
print(record["fill"])
```

`apply_to_record` updates the supplied record in memory and marks applicable fields as `human_confirmed`. The caller must save the updated records or table. Loading an override file does not automatically merge decisions into other project files.

## Inspect and correct history

`get_revision_history(rel_path)` returns the recorded revisions. `rollback_revision(rel_path, revision_id, ...)` appends a rollback revision and makes its selected values current; the earlier entries remain in the history.

Keep one writer per override file and back it up with the associated annotations. The manager uses temporary-file replacement when saving, but it does not coordinate concurrent writers or provide a tamper-proof audit log.

For field definitions, see [provenance and overrides](PROVENANCE_SCHEMA.md). For timing and label conventions, see the [annotation SOP](EMBODIED_ANNOTATION_SOP.md).
