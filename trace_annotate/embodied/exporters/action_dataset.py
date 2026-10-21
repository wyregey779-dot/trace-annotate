"""
Action Segmentation Dataset Exporter (MS-TCN, ASFormer, ActionFormer Format).
Converts continuous timestamp intervals into discrete frame-level action sequences
for training robotic and egocentric action recognition AI models.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple


def export_to_action_segmentation(
    docs: List[Dict[str, Any]],
    output_dir: str,
    fps: float = 1.0
) -> Tuple[str, List[str]]:
    """
    Export annotation documents to standard frame-level action segmentation format.
    
    Generates:
      - mapping.txt: class_index -> action_label
      - groundTruth/{clip_name}.txt: frame-by-frame action names
      
    Returns:
      mapping_path: path to mapping file
      gt_files: list of generated ground truth files
    """
    os.makedirs(output_dir, exist_ok=True)
    gt_dir = os.path.join(output_dir, "groundTruth")
    os.makedirs(gt_dir, exist_ok=True)

    # Discover unique action step labels
    unique_steps = set()
    for d in docs:
        for w in d.get("working", []):
            step = w.get("step_number")
            if step is not None:
                unique_steps.add(f"step_{step}")

    sorted_steps = ["idle"] + sorted(list(unique_steps))
    mapping_path = os.path.join(output_dir, "mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as f:
        for idx, label in enumerate(sorted_steps):
            f.write(f"{idx} {label}\n")

    gt_files = []
    for d in docs:
        clip_name = os.path.splitext(d.get("clip", "video.mp4"))[0]
        working = d.get("working", [])

        # Infer total duration
        max_dur = max(
            [float(w["end"]) for w in working] +
            [float(i["end"]) for i in d.get("idle", [])] +
            [1.0]
        )
        total_frames = max(1, int(round(max_dur * fps)))
        frame_labels = ["idle"] * total_frames

        # Paint working labels
        for w in working:
            step_label = f"step_{w.get('step_number', 1)}"
            s_idx = max(0, int(round(float(w["start"]) * fps)))
            e_idx = min(total_frames, int(round(float(w["end"]) * fps)))
            for k in range(s_idx, e_idx):
                frame_labels[k] = step_label

        gt_file = os.path.join(gt_dir, f"{clip_name}.txt")
        with open(gt_file, "w", encoding="utf-8") as f:
            for lbl in frame_labels:
                f.write(f"{lbl}\n")
        gt_files.append(gt_file)

    return mapping_path, gt_files
