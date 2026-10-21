"""
Action Segmentation Dataset Exporter (MS-TCN, ASFormer, ActionFormer Format).
Converts continuous timestamp intervals into discrete frame-level action sequences
for training robotic and egocentric action recognition AI models.
Supports persistent global mapping, true video duration padding, and unknown step safety.
"""
from __future__ import annotations
import os
import ntpath
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Union


_WINDOWS_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    prefix + digit for prefix in ("COM", "LPT") for digit in "123456789¹²³"
}


def _portable_name_key(name: str) -> str:
    """Reject canonical Unicode and case aliases independently of host OS."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def load_or_init_mapping(
    mapping_source: Optional[Union[Dict[str, int], List[str], str]],
    output_dir: str
) -> Dict[str, int]:
    """
    Load mapping from dictionary, list, file path, or existing output_dir/mapping.txt.
    Guarantees 'idle' is mapped to index 0 unless explicitly overridden.
    """
    mapping: Dict[str, int] = {}

    if isinstance(mapping_source, dict):
        mapping = dict(mapping_source)
    elif isinstance(mapping_source, list):
        mapping = {lbl: idx for idx, lbl in enumerate(mapping_source)}
    elif isinstance(mapping_source, str) and os.path.exists(mapping_source):
        with open(mapping_source, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        mapping[parts[1]] = int(parts[0])
                    except ValueError:
                        pass
    else:
        existing_file = os.path.join(output_dir, "mapping.txt")
        if os.path.exists(existing_file):
            with open(existing_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            mapping[parts[1]] = int(parts[0])
                        except ValueError:
                            pass

    if "idle" not in mapping:
        mapping["idle"] = 0

    return mapping


def export_to_action_segmentation(
    docs: List[Dict[str, Any]],
    output_dir: str,
    fps: float = 1.0,
    global_mapping: Optional[Union[Dict[str, int], List[str], str]] = None,
    clip_durations: Optional[Dict[str, float]] = None
) -> Tuple[str, List[str]]:
    """
    Export annotation documents to standard frame-level action segmentation format.
    
    Args:
        docs: list of annotation dictionaries
        output_dir: target output directory
        fps: frame sampling rate (default 1.0 Hz)
        global_mapping: optional dictionary, list, or mapping.txt path to prevent label drift
        clip_durations: optional dict of clip filename -> true video duration in seconds
        
    Generates:
        - mapping.txt: class_index -> action_label
        - groundTruth/{clip_name}.txt: frame-by-frame action names
      
    Returns:
        mapping_path: path to mapping file
        gt_files: list of generated ground truth files
    """
    # Reject input-controlled paths before writing mapping.txt or any labels.
    # Use Windows path rules too, so a dataset stays safe across platforms.
    clip_names = []
    for d in docs:
        raw_clip = d.get("clip", "video.mp4")
        if not isinstance(raw_clip, str) or not raw_clip.strip():
            raise ValueError("clip must be a non-empty filename")
        if (os.path.isabs(raw_clip) or ntpath.isabs(raw_clip)
                or ntpath.splitdrive(raw_clip)[0] or "/" in raw_clip
                or "\\" in raw_clip or "\x00" in raw_clip or ":" in raw_clip):
            raise ValueError(f"clip must be a filename without directory components: {raw_clip!r}")
        if any(ord(character) < 32 or character in '<>"|?*' for character in raw_clip):
            raise ValueError(f"clip contains characters reserved by Windows: {raw_clip!r}")
        clip_name = os.path.splitext(raw_clip)[0]
        if clip_name in ("", ".", ".."):
            raise ValueError(f"Invalid clip filename: {raw_clip!r}")
        if raw_clip != raw_clip.rstrip(" .") or clip_name != clip_name.rstrip(" ."):
            raise ValueError(f"clip and its stem must not end in spaces or dots: {raw_clip!r}")
        # Windows device aliases remain reserved even with multiple extensions
        # (NUL.tar.gz) or spaces immediately before the first extension.
        device_basename = clip_name.split(".", 1)[0].rstrip(" ").upper()
        if device_basename in _WINDOWS_DEVICE_NAMES:
            raise ValueError(f"clip uses a reserved Windows device name: {raw_clip!r}")
        clip_names.append(clip_name)
    if len({_portable_name_key(name) for name in clip_names}) != len(clip_names):
        raise ValueError("Clip filenames must produce unique ground-truth filenames")

    os.makedirs(output_dir, exist_ok=True)
    gt_dir = os.path.join(output_dir, "groundTruth")
    os.makedirs(gt_dir, exist_ok=True)

    mapping = load_or_init_mapping(global_mapping, output_dir)

    # Discover any newly seen labels from docs
    discovered_labels = []
    for d in docs:
        for w in d.get("working", []):
            step = w.get("step_number")
            step_label = f"step_{step}" if step is not None else "step_unknown"
            if step_label not in mapping and step_label not in discovered_labels:
                discovered_labels.append(step_label)

    discovered_labels.sort()
    for lbl in discovered_labels:
        next_idx = max(mapping.values()) + 1 if mapping else 0
        mapping[lbl] = next_idx

    # Write mapping file sorted by class index
    mapping_path = os.path.join(output_dir, "mapping.txt")
    sorted_by_id = sorted(mapping.items(), key=lambda item: item[1])
    with open(mapping_path, "w", encoding="utf-8") as f:
        for label, idx in sorted_by_id:
            f.write(f"{idx} {label}\n")

    gt_files = []
    for d, clip_name in zip(docs, clip_names):
        raw_clip = d.get("clip", "video.mp4")
        working = d.get("working", [])

        # Determine true duration: clip_durations > doc['duration'] > segment bounds
        true_duration = None
        if clip_durations:
            true_duration = clip_durations.get(raw_clip) or clip_durations.get(clip_name)

        if true_duration is None:
            true_duration = d.get("duration")

        if true_duration is not None and float(true_duration) > 0:
            max_dur = float(true_duration)
        else:
            max_dur = max(
                [float(w["end"]) for w in working] +
                [float(i["end"]) for i in d.get("idle", [])] +
                [1.0]
            )

        total_frames = max(1, int(round(max_dur * fps)))
        frame_labels = ["idle"] * total_frames

        # Paint working labels
        for w in working:
            step = w.get("step_number")
            step_label = f"step_{step}" if step is not None else "step_unknown"
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
