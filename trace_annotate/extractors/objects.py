"""Offline NanoDet COCO-80 object observations in decoded-frame coordinates.

This is an observation provider, not an action recognizer.  A box does not
establish that an object is held, supported, opened, closed or transferred.
Only the exact OpenCV Zoo NanoDet checkpoint is allowed to use the frozen COCO
label projection below; arbitrary ONNX bytes must not inherit its model name,
labels or Apache-2.0 attribution.

The first integration is intentionally independent from ``PersonExtractor``.
It therefore performs a second forward pass when callers request both APIs.
The workspace integration may later share the raw six tensors, but must keep
the public ``nanodet-person-v1`` result unchanged and record that only one
model forward actually occurred.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseExtractor, safe_failure_reason
from .persons import (INPUT_SIZE, MAX_MODEL_BYTES, NMS_THRESHOLD,
                      OFFICIAL_WEIGHTS_SHA256, STRIDES)

PROVIDER = "nanodet-coco80-v1"
OFFICIAL_MODEL_VERSION = "NanoDet-m-plus-1.5x_416_2022nov"
OFFICIAL_UPSTREAM_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
COCO80_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)
COCO80_LABELS_SHA256 = "a2bb9c8218affdef450cc85951e6a5abc5e4d956cd37b1be3d04840ae0120c8e"
LABEL_SOURCE = (
    "OpenCV Zoo NanoDet demo classes tuple at revision "
    + OFFICIAL_UPSTREAM_REVISION
)
SCORE_THRESHOLD = 0.35
MAX_OBJECTS = 200
MAX_OBJECTS_PER_CLASS = 50


def _reject_reparse_components(path: Path) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or bool(getattr(info, "st_file_attributes", 0) & 0x400)):
            raise ValueError("object_model_must_be_regular_file")


def _read_stable_model(path: Path) -> bytes:
    """Read one bounded regular file without accepting links or replacement."""
    _reject_reparse_components(path)
    before = path.lstat()
    if (stat.S_ISLNK(before.st_mode)
            or bool(getattr(before, "st_file_attributes", 0) & 0x400)
            or not stat.S_ISREG(before.st_mode)):
        raise ValueError("object_model_must_be_regular_file")
    if before.st_size > MAX_MODEL_BYTES:
        raise ValueError("object_model_exceeds_byte_budget")
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
            raise ValueError("object_model_changed_during_read")
        data = source.read(MAX_MODEL_BYTES + 1)
        after_read = os.fstat(source.fileno())
    after_path = path.lstat()
    _reject_reparse_components(path)
    if (len(data) > MAX_MODEL_BYTES
            or (after_read.st_dev, after_read.st_ino, after_read.st_size,
                after_read.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (after_path.st_dev, after_path.st_ino, after_path.st_size,
                after_path.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise ValueError("object_model_changed_during_read")
    return data


def _labels_digest() -> str:
    encoded = json.dumps(COCO80_LABELS, ensure_ascii=True,
                         separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


if len(COCO80_LABELS) != 80 or _labels_digest() != COCO80_LABELS_SHA256:
    raise RuntimeError("frozen_coco80_label_contract_changed")


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def decode_object_outputs(outputs: Sequence[Any], frame_size: Sequence[int],
                          transform: Sequence[float], *,
                          threshold: float = SCORE_THRESHOLD,
                          include_person: bool = False,
                          max_objects: int = MAX_OBJECTS,
                          max_per_class: int = MAX_OBJECTS_PER_CLASS) -> List[Dict[str, Any]]:
    """Decode the pinned six-output graph and apply deterministic class NMS.

    ``score`` is the graph's uncalibrated class score.  NMS is class-aware, so
    spatially overlapping observations of different classes never suppress one
    another.  By default class 0 is omitted because the existing person
    provider remains the authoritative subject-box projection.
    """
    if len(outputs) != 6:
        raise ValueError("unexpected_nanodet_output_count")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(float(threshold)) or not 0 <= threshold <= 1):
        raise ValueError("invalid_object_score_threshold")
    if type(max_objects) is not int or type(max_per_class) is not int:
        raise ValueError("invalid_object_output_budget")
    if not 1 <= max_objects <= MAX_OBJECTS or not 1 <= max_per_class <= MAX_OBJECTS_PER_CLASS:
        raise ValueError("invalid_object_output_budget")
    if len(frame_size) != 2 or len(transform) != 4:
        raise ValueError("invalid_frame_transform")
    width, height = frame_size
    sx, sy, left, top = transform
    values = (width, height, sx, sy, left, top)
    if (any(not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) for value in values)
            or width <= 0 or height <= 0 or sx <= 0 or sy <= 0):
        raise ValueError("invalid_frame_transform")

    candidates: List[tuple[int, List[float], float]] = []
    for level, stride in enumerate(STRIDES):
        side = INPUT_SIZE // stride
        count = side * side
        classes = np.asarray(outputs[level])
        boxes = np.asarray(outputs[level + 3])
        if classes.shape != (1, count, 80) or boxes.shape != (1, count, 32):
            raise ValueError("unexpected_nanodet_output_shape")
        if not np.isfinite(classes).all() or not np.isfinite(boxes).all():
            raise ValueError("nonfinite_nanodet_output")
        if np.any(classes < 0) or np.any(classes > 1):
            raise ValueError("invalid_nanodet_score_range")
        class_ids = classes[0].argmax(axis=1)
        class_scores = classes[0, np.arange(count), class_ids]
        selected = np.flatnonzero(class_scores >= threshold)
        for index in selected:
            class_id = int(class_ids[index])
            if class_id == 0 and not include_person:
                continue
            logits = boxes[0, index].reshape(4, 8).astype(np.float64)
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            totals = probabilities.sum(axis=1, keepdims=True)
            if not np.isfinite(totals).all() or np.any(totals <= 0):
                raise ValueError("invalid_nanodet_distribution")
            probabilities /= totals
            distance = (probabilities @ np.arange(8)) * stride
            cx = (index % side) * stride + (stride - 1) * 0.5
            cy = (index // side) * stride + (stride - 1) * 0.5
            box = [
                max(0.0, min(float(width), (cx - distance[0] - left) / sx)),
                max(0.0, min(float(height), (cy - distance[1] - top) / sy)),
                max(0.0, min(float(width), (cx + distance[2] - left) / sx)),
                max(0.0, min(float(height), (cy + distance[3] - top) / sy)),
            ]
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((class_id, box, float(class_scores[index])))

    candidates.sort(key=lambda item: (-item[2], item[0], tuple(item[1])))
    retained: List[tuple[int, List[float], float]] = []
    per_class: Dict[int, int] = {}
    for class_id, box, score in candidates:
        if len(retained) >= max_objects:
            break
        if per_class.get(class_id, 0) >= max_per_class:
            continue
        if any(old_class == class_id and _iou(box, old_box) > NMS_THRESHOLD
               for old_class, old_box, _ in retained):
            continue
        retained.append((class_id, box, score))
        per_class[class_id] = per_class.get(class_id, 0) + 1
    return [{
        "box": [round(float(value), 4) for value in box],
        "class_id": class_id,
        "class_label": COCO80_LABELS[class_id],
        "score": score,
        "score_kind": "model_score_uncalibrated",
        "coordinate_space": "decoded_frame_xyxy",
    } for class_id, box, score in retained]


class ObjectExtractor(BaseExtractor):
    """Exact-checkpoint COCO observations using one bounded CPU session."""

    def __init__(self, model_path: Optional[str] = None, *,
                 expected_sha256: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._status, self._reason = "not_run", "object_weights_unavailable"
        self.weights_sha256 = None
        self.runtime_version = "unavailable"
        if not model_path:
            return
        try:
            path = Path(model_path)
            data = _read_stable_model(path)
            self.weights_sha256 = hashlib.sha256(data).hexdigest()
            if expected_sha256 and self.weights_sha256 != expected_sha256:
                self._status, self._reason = "failed", "weights_hash_mismatch"
                return
            if self.weights_sha256 != OFFICIAL_WEIGHTS_SHA256:
                self._status, self._reason = "failed", "unverified_object_model_identity"
                return
            try:
                import onnxruntime as ort
            except ImportError:
                self._reason = "onnxruntime_unavailable"
                return
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self.runtime_version = ort.__version__
            self.model = ort.InferenceSession(data, sess_options=options,
                                              providers=["CPUExecutionProvider"])
            self._status, self._reason = "success", None
        except FileNotFoundError:
            return
        except ValueError as exc:
            self._status = "failed"
            self._reason = safe_failure_reason(exc, "object_model_load_failed")
        except Exception as exc:
            self._status = "failed"
            self._reason = safe_failure_reason(exc, "object_model_load_failed")

    @property
    def name(self) -> str:
        return "objects"

    @property
    def code_identity(self) -> Dict[str, Any]:
        official = self.weights_sha256 == OFFICIAL_WEIGHTS_SHA256
        return {
            "provider": "onnxruntime",
            "model_name": PROVIDER if official else "unverified_local_onnx",
            "model_version": OFFICIAL_MODEL_VERSION if official else "unverified",
            "runtime_version": self.runtime_version,
            "weights_path": self.model_path,
            "weights_sha256": self.weights_sha256,
            "license": "Apache-2.0 (OpenCV Zoo)" if official else "unverified",
            "upstream_revision": OFFICIAL_UPSTREAM_REVISION if official else None,
            "label_set": "COCO-80" if official else None,
            "label_set_sha256": COCO80_LABELS_SHA256 if official else None,
            "label_source": LABEL_SOURCE if official else None,
            "input": "BGR letterbox 416x416; mean 103.53/116.28/123.675; std 57.375/57.12/58.395",
            "threshold": SCORE_THRESHOLD,
            "nms_iou": NMS_THRESHOLD,
            "forward_sharing": "independent_session_not_shared_with_person_provider",
        }

    def detect_objects(self, frame: Any) -> Dict[str, Any]:
        empty = {"status": self._status, "reason": self._reason,
                 "object_count": None, "objects_detected": [], "provider": PROVIDER,
                 "excluded_class_ids": [0]}
        if self.model is None:
            return empty
        try:
            import cv2
            if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                    or frame.dtype != np.uint8 or min(frame.shape[:2]) <= 0):
                raise ValueError("object_input_requires_uint8_bgr")
            height, width = frame.shape[:2]
            scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            left = (INPUT_SIZE - new_width) // 2
            top = (INPUT_SIZE - new_height) // 2
            image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            image[top:top + new_height, left:left + new_width] = cv2.resize(
                frame, (new_width, new_height))
            normalized = (image.astype(np.float32) - np.array(
                [103.53, 116.28, 123.675], dtype=np.float32)) / np.array(
                    [57.375, 57.12, 58.395], dtype=np.float32)
            tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
            raw = self.model.run(None, {self.model.get_inputs()[0].name: tensor})
            detections = decode_object_outputs(
                raw, (width, height),
                (new_width / width, new_height / height, left, top))
            return dict(empty, status="success", reason=None,
                        object_count=len(detections), objects_detected=detections,
                        uncertainty=("COCO boxes are unreviewed observations. They do not prove "
                                     "identity continuity, manipulation, support, state change or action."))
        except Exception as exc:
            return dict(empty, status="failed",
                        reason=safe_failure_reason(exc, "object_inference_failed"))

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        from ..core.hasher import safe_load_image
        return self.detect_objects(safe_load_image(staged_path))
