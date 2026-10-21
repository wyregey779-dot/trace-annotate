<div align="center">

# TraceAnnotate ⚡

**Production-Grade, Cold-Storage Safe Multi-Modal Data Pipeline & Embodied AI Annotation Suite**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](pyproject.toml)
[![Tests: 13/13 Passing](https://img.shields.io/badge/Tests-13%2F13%20Passing-success.svg)](tests/)
[![Offline: 100%](https://img.shields.io/badge/Offline-Zero%20External%20Traffic-orange.svg)](#)

[English](README.md) | [中文说明](README_CN.md)

</div>

---

## 💡 Why TraceAnnotate?

Existing open-source annotation tools (CVAT, Label Studio, AnyLabeling, FiftyOne) primarily focus on **interactive bounding-box GUIs** or **cloud-first MLOps platforms**. 

However, real-world engineering teams handling **multi-terabyte raw media libraries** and **first-person robotic/embodied video datasets** face critical production blockers:
1. **Fragile External Storage**: Direct concurrent I/O on external hard drives or SD cards causes disconnections (`WinError 21`) or disk corruption.
2. **Black-Box AI Tagging**: Downstream users cannot audit whether a tag was predicted by an AI model, derived from path rules, or confirmed by a human.
3. **Dirty Field False Positives**: Statues detected as human faces, street shop signs falsely flagged as smartphone screens, and tickets confused with scenic landmarks.
4. **Labor-Intensive Video Audit**: Opening thousands of high-resolution video clips one-by-one is painfully slow.

**TraceAnnotate** bridges this gap. It combines **cold-storage hardware protection (R1-R5)**, **field-level provenance tracking (`fill` codes)**, **composite domain heuristics (F1-F3)**, and a complete **Embodied AI video temporal annotation suite** into a unified, offline-first Python framework.

---

## 📊 Feature Comparison

| Capability | CVAT / Label Studio | FiftyOne | **TraceAnnotate** (Ours) |
|---|:---:|:---:|:---:|
| **Cold Storage Staging Protocol (R1-R5)** | ❌ (Direct I/O only) | ❌ (No drive guard) | **✅ Chunked copy, read-only guard, double-checksum** |
| **Field-Level Provenance Tracking** | ❌ (Flat values) | ⚠️ (Field tags only) | **✅ `fill` dictionary (`auto_metric`, `human_confirmed`, etc.)** |
| **Statue / Artifact Face Disambiguation (F1)**| ❌ (Counts as person) | ❌ (No heuristic) | **✅ Decouples sculpture faces from living persons** |
| **Mobile Screen vs Storefront OCR (F3)** | ❌ | ❌ | **✅ OCR UI keywords + morphology disambiguation** |
| **Near-Duplicate Clustering** | ⚠️ (Plugin required) | ✅ (Embedding-based) | **✅ Perceptual dHash + Disjoint-Set Union (并查集)** |
| **Rapid 4x4 Visual Audit Sheets** | ❌ | ⚠️ (Web grid only) | **✅ High-density contact sheets (`sheet_generator`)** |
| **Embodied Task Temporal Segmentation** | ⚠️ (Manual UI) | ❌ | **✅ Working/Idle complement, step diff, 7-class failure diagnosis** |
| **Inter-Annotator Temporal Kappa ($\ge 0.80$)**| ❌ | ❌ | **✅ Second-by-second timeline IoU Cohen's Kappa** |
| **100% Offline / Zero Cloud Traffic** | ⚠️ (Self-hostable) | ⚠️ | **✅ Native local execution, zero external downloads** |

---

## 🏗️ Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │       Cold Storage (Source Drive / H:\)      │
                      └──────────────────────┬───────────────────────┘
                                             │ Staging (R1-R5 protocol)
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                Core Pipeline Engine                                │
│                                                                                    │
│  [1. Technical Grounding]      [2. Multi-Modal Extractors]  [3. Composite Heuristics]│
│  - size & head1mb_md5          - Visual Metrics (Laplacian) - F1: Statue Face (YN)   │
│  - Perceptual dHash            - YuNet Face Detector        - F2: Entity Tiering     │
│  - DSU Near-Dup Clustering     - RapidOCR Sign Extractor    - F3: Mobile Screen OCR  │
│                                - Zero-Shot CLIP             - Calibrated Scene Cut   │
│                                                                                    │
│  [4. Provenance & Consensus Tracking]                                              │
│  fill: { auto_metric | auto_rule | auto_path | pending_review | human_confirmed }  │
└────────────────────────────────────────────┬───────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Human-in-the-Loop Fast Audit                             │
│  - 4x4 Visual Contact Sheets   - P1-P4 Prioritized Queue   - Append-Only Overrides │
└────────────────────────────────────────────┬───────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Embodied AI Video Annotation                             │
│  - Clip-Level Diagnosis         - Hand-Object Working Segment Boundaries           │
│  - Mid-Segment Holes            - Automatic Idle Timeline Complements              │
│  - Expected Step Diffing S=[1..N]- 7-Class Failure Diagnosis & Template Evidence   │
│  - Temporal Cohen's Kappa Eval  - Label Studio v2.1 XML / JSON Roundtrip Export    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/trace-annotate.git
cd trace-annotate

# Install lightweight core
pip install -e .

# Or install full multi-modal suite (OpenCV, RapidOCR, PyTorch)
pip install -e ".[full]"
```

### 2. Run Built-In Demos

```bash
# 1. Media Asset Pipeline Demo (Cold staging, dHash clustering, heuristics, 21-point verification)
python examples/demo_asset_pipeline.py

# 2. Embodied AI Annotation Demo (Working/Idle, Step Diff, 7-class failure, Cohen's Kappa)
python examples/demo_embodied_pipeline.py
```

### 3. CLI Reference

```bash
# Verify 21-point consistency of an annotation catalog
trace-annotate asset-verify my_annotations.jsonl --inventory my_inventory.jsonl

# Generate 4x4 visual audit contact sheet
trace-annotate asset-sheet thumbs/*.jpg -o audit_sheet.jpg --cols 4

# Compute idle intervals automatically for embodied task video
trace-annotate embodied-idle clip_001.json --inplace

# Reconcile expected steps S = [1..N] (detect missing, out-of-order, duplicates)
trace-annotate embodied-diff clip_001.json --nsteps 5

# Compute inter-annotator temporal Cohen's Kappa
trace-annotate embodied-kappa annotator_A.jsonl annotator_B.jsonl --fps 2.0

# Export standardized Label Studio v2.1 XML configuration
trace-annotate embodied-xml -o label_studio_config.xml
```

---

## 🧠 Python API Examples

### Example 1: Cold-Storage Staging & Perceptual Clustering

```python
from trace_annotate.core.staging import StagingManager
from trace_annotate.core.hasher import compute_dhash
from trace_annotate.core.cluster import cluster_by_dhash

# Enforce R1-R5 cold-storage safety: 4 files per batch, verified with head1mb_md5
staging = StagingManager(source_root="E:/RawFootage", staging_root="./staging")

hashes = {}
for batch in staging.stage_batch(["clip1.mp4", "clip2.mp4", "img3.jpg"]):
    for rel_path, staged_path in batch.items():
        hashes[rel_path] = compute_dhash(staged_path)
    # Staged copies are automatically purged after the loop body!

# Group near-duplicates (Hamming distance <= 10)
item_to_cluster, clusters = cluster_by_dhash(hashes, hamming_threshold=10)
print(f"Discovered {len(clusters)} unique visual clusters.")
```

### Example 2: Embodied AI Temporal Segmentation & Step Diffing

```python
from trace_annotate.embodied.schema import EmbodiedAnnotationDoc, WorkingSegment, Verdict
from trace_annotate.embodied.complement import compute_idle_complement
from trace_annotate.embodied.differ import diff_expected_steps

doc = EmbodiedAnnotationDoc(
    clip="wrist_assembly_01.mp4",
    text_description="1. Pick base; 2. Insert shaft; 3. Tighten nut",
    working=[
        WorkingSegment(start=2.0, end=15.0, step_number=1, verdict=Verdict.CORRECT.value),
        WorkingSegment(start=18.0, end=30.0, step_number=3, verdict=Verdict.WRONG.value) # Step 3 before 2!
    ]
)

# 1. Automatic Idle Complement calculation
idle_intervals = compute_idle_complement(
    [{"start": w.start, "end": w.end} for w in doc.working],
    total_duration=45.0
)

# 2. Automated Step Reconciliation
diff = diff_expected_steps([{"step_number": w.step_number} for w in doc.working], expected_step_count=3)
print(diff["missing"])       # [2]
print(diff["out_of_order"])  # [{'previous_step': 1, 'current_step': 3}]
```

---

## 🔬 21-Point Integrity Verification Suite

TraceAnnotate includes an automated verification engine that checks 21 quantitative integrity invariants:

- `row_count_match`: Annotation count matches source inventory.
- `asset_id_missing == 0`: No unassigned asset IDs.
- `duplicate_ids == 0`: Primary keys are strictly unique.
- `key_missing == {}`: All 46 schema keys populated in every record.
- `phash_missing == 0`: Perceptual fingerprinting completeness.
- `video_windows_null == 0`: All video assets contain usable editing windows.
- `face_missing == 0`, `ocr_missing == 0`, `clip_missing == 0`.
- `fill_missing == 0`: 100% field-level provenance audit coverage.

---

## 📖 Documentation

- [System Architecture](docs/ARCHITECTURE.md)
- [Embodied Annotation SOP v2.1](docs/EMBODIED_ANNOTATION_SOP.md)
- [46-Key Schema & Provenance Guide](docs/PROVENANCE_SCHEMA.md)
- [Fast Visual Audit Workflow](docs/FAST_AUDIT_WORKFLOW.md)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
