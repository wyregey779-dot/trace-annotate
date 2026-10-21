# Fast Visual Auditing Workflow (Proof Sheet + Append-Only Overrides)

In large datasets ($> 5,000$ assets), opening video clips or images one by one is prohibitively slow. TraceAnnotate implements an ultra-fast visual audit workflow:

```
[Candidate / Conflict Queue]
          │
          ▼
[4x4 Contact Sheet Generator]  ───▶  Reviewer audits 16 clips in 2 seconds
          │
          ▼
[Append-Only human_overrides.json]
          │
          ▼
[In-Place Table Upgrade (fill = human_confirmed, status = confirmed)]
```

---

## 1. 4x4 Contact Sheet Generation

Batch up to 16 thumbnails into a unified proof grid with overlay identifiers:

```bash
trace-annotate asset-sheet thumbs/*.jpg -o audit_grid_01.jpg --cols 4 --title "Statue Face Audit"
```

A human reviewer glances at the image for 2-3 seconds to spot edge cases (e.g., an ancient statue detected as a human face, or a ticket falsely predicted as a scenic location).

---

## 2. Append-Only Overrides

When an error is spotted, the reviewer records a ground-truth judgment in `human_overrides.json`:

```json
{
  "scenic/statue_01.jpg": {
    "primary_entity": "青年毛主席雕像",
    "face_status": "无人物",
    "note": "Audited: historical stone sculpture, no living person present"
  }
}
```

### Key Principles:
1. **Never mutate raw history**: Original machine metrics in the archive remain intact.
2. **Deterministic Precedence**: The pipeline automatically injects `human_overrides.json` during the merge phase.
3. **Fill Elevation**:
   - `entity_status` is automatically elevated to `confirmed`.
   - `fill["primary_entity"]` is marked `human_confirmed`.
   - `fill["face_status"]` is marked `human_confirmed`.
