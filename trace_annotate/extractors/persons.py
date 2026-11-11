"""Offline NanoDet person boxes in decoded-frame coordinates.

The output layout and normalization follow OpenCV Zoo NanoDet-m-plus-1.5x_416
(Apache-2.0). This adapter independently filters COCO person class 0, performs
bounded person-only NMS, and reverses letterboxing before returning geometry.
Detections are candidates, including possible screen people or statues; no
identity, absence-of-person, action, or calibrated probability is asserted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseExtractor

PROVIDER = 'nanodet-person-v1'
OFFICIAL_WEIGHTS_SHA256 = '4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186'
INPUT_SIZE = 416
STRIDES = (8, 16, 32)
SCORE_THRESHOLD = .5
NMS_THRESHOLD = .5
MAX_PERSONS = 100
MAX_MODEL_BYTES = 32 * 1024 * 1024


def decode_person_outputs(outputs: Sequence[Any], frame_size: Sequence[int],
                          transform: Sequence[float], *,
                          threshold: float = SCORE_THRESHOLD) -> List[Dict[str, Any]]:
    """Decode the pinned ONNX: three class tensors, then three box tensors.

    OpenCV DNN's unconnected-output ordering differs from ONNX Runtime's graph
    outputs. The pinned 2022nov file has only strides 8/16/32. Never guess that
    the four-level, interleaved layout from another demo is interchangeable.
    """
    if len(outputs) != 6:
        raise ValueError('unexpected_nanodet_output_count')
    width, height = frame_size
    sx, sy, left, top = transform
    if width <= 0 or height <= 0 or sx <= 0 or sy <= 0:
        raise ValueError('invalid_frame_transform')
    candidates = []
    for level, stride in enumerate(STRIDES):
        side = INPUT_SIZE // stride
        count = side * side
        cls = np.asarray(outputs[level])
        raw = np.asarray(outputs[level + 3])
        if cls.shape != (1, count, 80) or raw.shape != (1, count, 32):
            raise ValueError('unexpected_nanodet_output_shape')
        if not np.isfinite(cls).all() or not np.isfinite(raw).all():
            raise ValueError('nonfinite_nanodet_output')
        selected = np.flatnonzero((cls[0, :, 0] >= threshold)
                                  & (cls[0].argmax(axis=1) == 0))
        for index in selected:
            logits = raw[0, index].reshape(4, 8).astype(np.float64)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            distance = (probs @ np.arange(8)) * stride
            cx = (index % side) * stride + (stride - 1) * .5
            cy = (index // side) * stride + (stride - 1) * .5
            x1 = max(0., min(float(width), (cx - distance[0] - left) / sx))
            y1 = max(0., min(float(height), (cy - distance[1] - top) / sy))
            x2 = max(0., min(float(width), (cx + distance[2] - left) / sx))
            y2 = max(0., min(float(height), (cy + distance[3] - top) / sy))
            if x2 > x1 and y2 > y1:
                candidates.append(([x1, y1, x2, y2], float(cls[0, index, 0])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept = []
    for box, score in candidates:
        if len(kept) >= MAX_PERSONS:
            break
        area = (box[2] - box[0]) * (box[3] - box[1])
        duplicate = False
        for old in kept:
            other = old['box']
            intersection = max(0., min(box[2], other[2]) - max(box[0], other[0])) * max(
                0., min(box[3], other[3]) - max(box[1], other[1]))
            union = area + (other[2] - other[0]) * (other[3] - other[1]) - intersection
            if union > 0 and intersection / union > NMS_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append({'box': [round(float(x), 4) for x in box],
                         'confidence': score, 'confidence_kind': 'model_score_uncalibrated',
                         'coordinate_space': 'decoded_frame_xyxy', 'class': 'person'})
    return kept


class PersonExtractor(BaseExtractor):
    """Optional local ONNX provider, one CPU thread and no network operations."""
    def __init__(self, model_path: Optional[str] = None, *,
                 expected_sha256: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._status, self._reason = 'not_run', 'person_weights_unavailable'
        self.weights_sha256 = None
        self.runtime_version = 'unavailable'
        if not model_path or not Path(model_path).is_file():
            return
        try:
            if Path(model_path).stat().st_size > MAX_MODEL_BYTES:
                self._status, self._reason = 'failed', 'person_model_exceeds_byte_budget'
                return
            with Path(model_path).open('rb') as source:
                data = source.read(MAX_MODEL_BYTES + 1)
            if len(data) > MAX_MODEL_BYTES:
                self._status, self._reason = 'failed', 'person_model_exceeds_byte_budget'
                return
            self.weights_sha256 = hashlib.sha256(data).hexdigest()
            if expected_sha256 and self.weights_sha256 != expected_sha256:
                self._status, self._reason = 'failed', 'weights_hash_mismatch'
                return
            try:
                import onnxruntime as ort
            except ImportError:
                self._reason = 'onnxruntime_unavailable'
                return
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self.runtime_version = ort.__version__
            self.model = ort.InferenceSession(data, sess_options=options,
                                              providers=['CPUExecutionProvider'])
            self._status, self._reason = 'success', None
        except Exception as exc:
            self._status, self._reason = 'failed', 'person_model_load_failed: ' + str(exc)[:240]

    @property
    def name(self) -> str:
        return 'persons'

    @property
    def code_identity(self) -> Dict[str, Any]:
        official = self.weights_sha256 == OFFICIAL_WEIGHTS_SHA256
        return {'provider': 'onnxruntime', 'model_name': PROVIDER if official else 'unverified_local_onnx',
                'model_version': 'NanoDet-m-plus-1.5x_416_2022nov' if official else 'unverified',
                'runtime_version': self.runtime_version, 'weights_path': self.model_path,
                'weights_sha256': self.weights_sha256, 'license': 'Apache-2.0 (OpenCV Zoo)' if official else 'unverified',
                'input': 'BGR letterbox 416x416; mean 103.53/116.28/123.675; std 57.375/57.12/58.395',
                'threshold': SCORE_THRESHOLD, 'nms_iou': NMS_THRESHOLD}

    def detect_people(self, frame: Any) -> Dict[str, Any]:
        empty = {'status': self._status, 'reason': self._reason,
                 'person_count': None, 'persons_detected': [], 'provider': PROVIDER}
        if self.model is None:
            return empty
        try:
            import cv2
            if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                    or frame.dtype != np.uint8 or min(frame.shape[:2]) <= 0):
                raise ValueError('person_input_requires_uint8_bgr')
            height, width = frame.shape[:2]
            scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
            new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
            left, top = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
            image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            image[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
            normalized = (image.astype(np.float32) - np.array([103.53, 116.28, 123.675],
                          dtype=np.float32)) / np.array([57.375, 57.12, 58.395], dtype=np.float32)
            tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
            raw = self.model.run(None, {self.model.get_inputs()[0].name: tensor})
            detections = decode_person_outputs(raw, (width, height),
                                               (new_w / width, new_h / height, left, top))
            return dict(empty, status='success', reason=None, person_count=len(detections),
                        persons_detected=detections,
                        uncertainty='Model detections may include screen people or statues; zero detections do not prove absence.')
        except Exception as exc:
            return dict(empty, status='failed', reason='person_inference_failed: ' + str(exc)[:240])

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        from ..core.hasher import safe_load_image
        return self.detect_people(safe_load_image(staged_path))
