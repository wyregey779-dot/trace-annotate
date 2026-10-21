# TraceAnnotate System Architecture

## Overview

TraceAnnotate is designed around two interlocking engines:
1. **Media Asset Data Pipeline Engine**: Ingests, hashes, stages, clusters, and semantically tags multi-terabyte raw video/image archives.
2. **Embodied AI Temporal Annotation Suite**: Fine-grained egocentric action segmentation, working/idle phase extraction, 7-class error diagnosis, and inter-annotator agreement evaluation.

```
                    ┌──────────────────────────────────────────────┐
                    │       Cold Storage (Source Drive / H:\)      │
                    └──────────────────────┬───────────────────────┘
                                           │ Staging (R1-R5 protocol)
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Core Pipeline Engine                                 │
│                                                                                  │
│  [1. Technical Grounding]    [2. Multi-Modal Extractors]  [3. Composite Heuristics]│
│  - size & head1mb_md5        - Visual Metrics (Laplacian) - F1: Statue Face (YN)   │
│  - Perceptual dHash          - YuNet Face Detector        - F2: Entity Tiering     │
│  - DSU Near-Dup Clustering   - RapidOCR Sign Extractor    - F3: Mobile Screen OCR  │
│                              - Zero-Shot CLIP             - Calibrated Scene Cut   │
│                                                                                  │
│  [4. Provenance & Consensus Tracking]                                            │
│  fill: { auto_metric | auto_rule | auto_path | pending_review | human_confirmed }│
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Human-in-the-Loop Fast Audit                             │
│  - 4x4 Visual Contact Sheets   - P1-P4 Prioritized Queue   - Append-Only Overrides│
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Embodied AI Video Annotation                             │
│  - Clip-Level Diagnosis         - Hand-Object Working Segment Boundaries           │
│  - Mid-Segment Holes            - Automatic Idle Timeline Complements              │
│  - Expected Step Diffing S=[1..N]- 7-Class Failure Diagnosis & Template Evidence │
│  - Temporal Cohen's Kappa Eval  - Label Studio v2.1 XML / JSON Roundtrip Export  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Cold Storage Safety Protocol (R1-R5)

Handling large-scale footage on external storage (e.g., HDDs or SD cards) presents real hardware risks:
- Device disconnections (`WinError 21`) during high-concurrency read/write.
- Accidental mutation or corruption of raw archival footage.
- Local SSD space exhaustion.

TraceAnnotate enforces strict cold-storage rules:
* **R1**: Source directory is read-only. No processing outputs, cache files, or temporary frames are ever written back to source.
* **R2**: Staging is performed in small, fixed-size batches (default 4 files).
* **R3**: Every staged file is bidirectional verified (`size` + `head1mb_md5`) before processing.
* **R4**: Free disk space on the host machine is guarded by a hard floor (default 2GB).
* **R5**: Staged temporary copies are purged immediately after batch completion.

---

## 2. Field-Level Provenance Architecture

Standard annotation tools produce flat JSONs where human reviewers cannot tell whether a tag was predicted by an AI model, inferred from a file path, or validated by a domain expert.

TraceAnnotate solves this with the `fill` dictionary:
```json
{
  "primary_entity": "杜甫江阁",
  "face_count": 0,
  "indoor_outdoor": "室外",
  "fill": {
    "primary_entity": "human_confirmed",
    "face_count": "auto_metric",
    "indoor_outdoor": "auto_rule"
  }
}
```
Tier Hierarchy:
1. `human_confirmed`: Human judgment. Overrides all machine predictions.
2. `auto_metric`: Direct algorithmic measurement (Laplacian variance, dimensions, dHash, ffprobe).
3. `auto_rule`: Deterministic business logic rules.
4. `auto_path_confirmed`: Deep directory hierarchy match.
5. `candidate_only`: Machine proposal with uncertainty.
6. `pending_review`: Explicitly flagged for human auditor review.
7. `pending_tool`: Awaiting external specialized sensor/tool.
