# Embodied AI Egocentric Video Annotation SOP (v2.1)

TraceAnnotate's project-specific convention for first-person task video annotation. The version number describes this convention, not an external annotation standard. Dataset-specific mappings require separate validation.

The timing rules below are suggested human annotation guidelines. The validator checks document structure, working-segment timestamps, steps and verdict consistency; it does not automatically verify semantic correctness or every guideline in this document.

---

## 1. Input & Output

- **Input**: Video clip + Text description defining expected procedure $S = [s_1, s_2, \dots, s_N]$.
- **Output**: Standardized JSON containing clip diagnosis, working segments, mid-segment holes, idle intervals, step numbers, verdicts, error classifications, and template evidence.

---

## 2. Step 1: Clip-Level Diagnosis (Gatekeeper)

Inspect the overall clip before drawing intervals. If non-normal, stop immediately:

| Diagnosis Value | Standard | Action |
|---|---|---|
| **正常 (Normal)** | Wrist/hand perspective visible, content relevant to task | Proceed to Step 2 |
| **佩戴问题 (Wear Issue)** | Hands not visible: occluded, upside down, black screen, selfie | Do not segment. Fill Quick ID + description in report. |
| **描述不符-整段无关** | Footage completely unrelated to text prompt | Do not segment. Report. |
| **描述不符-仅部分工作** | Only a sub-part of task performed | Record a report and leave working empty. Split out relevant footage and annotate it as a separate normal clip if needed. |
| **这里没有 (Empty)** | No work performed (empty background, insects) | Do not segment. Record a report. |

---

## 3. Step 2: Working Segment Boundaries (Green)

A segment is **Working** if the hand is actively executing or directly preparing (tool grabbing, aligning, parameter adjusting) for a described step:
- **Start Boundary**: First frame where hand starts moving toward target object.
  - *Special Case (Object already in hand)*: First frame where manipulation/motion begins.
- **End Boundary**: Frame where hand leaves contact with target object.
- **Micro-Fragments**: Fragments $< 1\text{s}$ merged into adjacent segments; pauses $< 2\text{s}$ do not break segment.
- **Reset Actions**: Returning objects to pre-collection state (e.g., repeatedly unthreading screws, disassembling shelves) is NOT working. Split working segments around resets.

### Walking Rules:
- **Type A (Carrying)**: Carrying relevant items (serving food, transporting assembly parts) $\to$ Keep working.
- **Type B (Operating)**: Walking with tool without operating (sweeping, chopping) $\to$ Split as idle.
- **Pipeline Waiting**: $< 10\text{s}$ keep working; $\ge 10\text{s}$ split out.

---

## 4. Step 3: Mid-Segments & Idle Complements

- **Mid-Segment Holes (Orange)**: Unproductive behaviors inside a working segment $> 3\text{s}$ (touching face, checking phone, zoning out, wrist occlusion). Tagged as mid-segment without breaking the parent working segment.
- **Idle (Gray)**: Timeline complement of all working segments. Automatically computed via `trace-annotate embodied-idle` or `compute_idle_complement()`.

---

## 5. Step 4: Correctness Verdict & 7 Error Sub-Classes

Every working segment must be assigned a `step_number` corresponding to $S = [1..N]$ and a verdict:

| Error Sub-Class | Standard | Template Evidence Example |
|---|---|---|
| **遗漏步骤 (Missing Step)** | Described step never performed | `"第3步全程未出现"` |
| **顺序错误 (Out of Order)** | Steps performed out of chronological order | `"t=22s 第4步出现在第3步之前"` |
| **重复/多做 (Duplicate/Extra)** | Repeated step or extraneous task | `"第2步固定螺母重复执行了两遍"` |
| **用错物体 (Wrong Object)** | Used incorrect tool or material | `"t=14s 用了十字改锥，规定应为一字改锥"` |
| **操作方式错误 (Wrong Method)** | Correct object, incorrect posture/technique | `"t=18s 旋拧手法方向相反"` |
| **意外事故 (Accident)** | Dropping, slipping, spilling items | `"t=30s 零件从工作台滑落"` |
| **其他 (Other)** | Complex multi-error or edge case | Detailed description |

---

## 6. Working/Idle Agreement

- The agreement command samples working/idle states at $2\text{Hz}$ by default. This is not a measurement of semantic label accuracy or per-step temporal IoU.
- The report includes reference flags for $\kappa \ge 0.80$ and $\kappa \ge 0.70$. These are project defaults, not validated production acceptance criteria.
- Choose acceptance criteria for your own dataset, sampling rate and review process. Inspect disagreements and class balance alongside any aggregate score.
