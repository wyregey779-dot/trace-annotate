"""Offline VideoMAE/SSV2 window evidence with a deliberately narrow scope.

This optional adapter does *not* recognise a complete TraceAnnotate action.
Something-Something-V2 classifies a sixteen-frame window, but it does not bind
the result to a subject, object track, recipient, or exact action interval.  A
successful result is therefore an unreviewed semantic hint which another
provider must corroborate with track/state evidence before an action candidate
can be created.

Only the exact MCG-NJU ``videomae-small-finetuned-ssv2`` checkpoint frozen
below may inherit its label table or CC-BY-NC-4.0 attribution.  Loading is from
an existing local directory and always uses ``local_files_only=True``; this
module never downloads model resources.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaseExtractor, is_safe_failure_reason, safe_failure_reason
from .model_loading import enforce_offline_environment


PROVIDER = "videomae-small-ssv2-aux-v1"
UPSTREAM_MODEL = "MCG-NJU/videomae-small-finetuned-ssv2"
UPSTREAM_REVISION = "55947f6dd36aaeff50bcc796a2d93d8e3c0246ad"
WEIGHTS_FILE = "pytorch_model.bin"
WEIGHTS_SIZE = 87849013
WEIGHTS_SHA256 = "e3a1aa3caef3828c9c3bc3240f7d2023d037962c0952db6d702779d05bbe5e09"
CONFIG_SHA256 = "b83c728afd6ae7e74e1914087ef9d9818acbec66fba0d312a194780d6c9eda71"
PROCESSOR_SHA256 = "c3aa722f22a7ff0d234d407862025fe47672f74f46f92c91f757f7a7354107c2"
WEIGHTS_LICENSE = "CC-BY-NC-4.0"
WEIGHTS_LICENSE_SOURCE = (
    "https://huggingface.co/MCG-NJU/videomae-small-finetuned-ssv2/"
    "blob/55947f6dd36aaeff50bcc796a2d93d8e3c0246ad/README.md"
)
NUM_FRAMES = 16
NUM_CLASSES = 174
MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_FRAME_PIXELS = 16 * 1024 * 1024
MAX_WINDOW_BYTES = 256 * 1024 * 1024
MAX_TOP_K = 10
ISOLATED_CONTRACT = "isolated-videomae-window-v1"
MAX_ISOLATED_REQUEST_BYTES = 256 * 1024
MAX_ISOLATED_OUTPUT_BYTES = 2 * 1024 * 1024
MODEL_IDENTITY_FIELDS = {
    "provider", "model_name", "upstream_model", "upstream_revision",
    "runtime_version", "torch_version", "weights_sha256", "config_sha256",
    "preprocessor_config_sha256", "weights_license", "weights_license_source",
    "runtime_license", "local_files_only", "num_frames", "authority",
    "supported_hints", "unsupported_actions",
}

# Exact labels from the pinned config.  These are semantic-window hints only.
TARGET_LABELS = {
    "Closing [something]": "close",
    "Opening [something]": "open",
    "Picking [something] up": "pick_up",
    "Putting [number of] [something] onto [something]": "put_down_surface",
    "Putting [something] and [something] on the table": "put_down_surface",
    "Putting [something] on a flat surface without letting it roll": "put_down_surface",
    "Putting [something] on a surface": "put_down_surface",
    "Putting [something] onto a slanted surface but it doesn't glide down": "put_down_surface",
    "Putting [something] onto [something]": "put_down_surface",
    "Putting [something] that can't roll onto a slanted surface, so it stays where it is":
        "put_down_surface",
    "Putting [something] upright on the table": "put_down_surface",
    "Putting [something], [something] and [something] on the table": "put_down_surface",
}
EXPECTED_TARGET_IDS = {
    5: "Closing [something]",
    46: "Opening [something]",
    47: "Picking [something] up",
    102: "Putting [number of] [something] onto [something]",
    103: "Putting [something] and [something] on the table",
    108: "Putting [something] on a flat surface without letting it roll",
    109: "Putting [something] on a surface",
    111: "Putting [something] onto a slanted surface but it doesn't glide down",
    112: "Putting [something] onto [something]",
    116: "Putting [something] that can't roll onto a slanted surface, so it stays where it is",
    119: "Putting [something] upright on the table",
    120: "Putting [something], [something] and [something] on the table",
}
EXPECTED_REJECTION_IDS = {
    13: "Failing to put [something] into [something] because [something] does not fit",
    66: "Pretending to close [something] without actually closing it",
    67: "Pretending to open [something] without actually opening it",
    68: "Pretending to pick [something] up",
    74: "Pretending to put [something] on a surface",
    75: "Pretending to put [something] onto [something]",
}
UNSUPPORTED_ACTIONS = ("walk", "run", "hand_over")


def _sha256(path: Path, *, max_bytes: int) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError("model_file_missing_or_exceeds_budget") from exc
    reparse = bool(getattr(before, "st_file_attributes", 0) & 0x400)
    if (stat.S_ISLNK(before.st_mode) or reparse or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= max_bytes):
        raise ValueError("model_file_missing_or_exceeds_budget")
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
            raise ValueError("model_file_changed_during_hash")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            consumed += len(block)
            if consumed > max_bytes:
                raise ValueError("model_file_exceeds_budget")
            digest.update(block)
        after_open = os.fstat(stream.fileno())
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ValueError("model_file_changed_during_hash") from exc
    if ((after_open.st_dev, after_open.st_ino, after_open.st_size, after_open.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise ValueError("model_file_changed_during_hash")
    return digest.hexdigest()


def _labels(mapping: Any) -> Dict[int, str]:
    if not isinstance(mapping, dict) or len(mapping) != NUM_CLASSES:
        raise ValueError("unexpected_ssv2_label_contract")
    try:
        result = {int(key): str(value) for key, value in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("unexpected_ssv2_label_contract") from exc
    if set(result) != set(range(NUM_CLASSES)):
        raise ValueError("unexpected_ssv2_label_contract")
    for index, label in {**EXPECTED_TARGET_IDS, **EXPECTED_REJECTION_IDS}.items():
        if result.get(index) != label:
            raise ValueError("unexpected_ssv2_label_contract")
    return result


def _validate_ranked_result(result: Dict[str, Any], *, top_k: int) -> None:
    """Validate the bounded semantic projection returned by the child.

    The parent does not trust a syntactically valid result file.  In particular,
    it recomputes every label-to-hint/rejection decision that could otherwise be
    edited into an action-like claim after the worker exits.
    """
    ranked = result.get("ranked")
    if not isinstance(ranked, list) or not 0 < len(ranked) <= top_k:
        raise ValueError("invalid_video_action_ranking")
    seen = set()
    previous = float("inf")
    for item in ranked:
        expected = {"class_id", "label", "score", "score_kind",
                    "semantic_hint", "rejection_signal"}
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("invalid_video_action_ranking")
        class_id, label, score = item["class_id"], item["label"], item["score"]
        if (type(class_id) is not int or not 0 <= class_id < NUM_CLASSES
                or class_id in seen or not isinstance(label, str)
                or not label or len(label) > 240
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(float(score)) or not 0 <= float(score) <= 1
                or float(score) > previous
                or item["score_kind"] != "softmax_score_uncalibrated"
                or item["semantic_hint"] != TARGET_LABELS.get(label)
                or item["rejection_signal"] is not _is_rejection_label(label)):
            raise ValueError("invalid_video_action_ranking")
        pinned = {**EXPECTED_TARGET_IDS, **EXPECTED_REJECTION_IDS}.get(class_id)
        if pinned is not None and label != pinned:
            raise ValueError("invalid_video_action_ranking")
        seen.add(class_id)
        previous = float(score)
    first = ranked[0]
    if first["rejection_signal"]:
        expected_decision, expected_hints = "reject", []
        expected_reason = "pretending_or_failed_action_top_class"
    elif first["semantic_hint"] is not None:
        expected_decision, expected_hints = "auxiliary_candidate", [first["semantic_hint"]]
        expected_reason = None
    else:
        expected_decision, expected_hints = "unknown", []
        expected_reason = "top_class_outside_supported_auxiliary_mapping"
    if (result.get("decision") != expected_decision
            or result.get("semantic_hints") != expected_hints
            or result.get("reason") != expected_reason
            or result.get("score_kind") != "softmax_score_uncalibrated"):
        raise ValueError("inconsistent_video_action_decision")


def _is_rejection_label(label: str) -> bool:
    lowered = label.lower()
    return ("pretending" in lowered or "failing" in lowered
            or ("trying" in lowered and "fail" in lowered)
            or "nothing happens" in lowered)


def interpret_logits(logits: Sequence[float], id2label: Dict[Any, Any], *,
                     top_k: int = 5) -> Dict[str, Any]:
    """Interpret one 174-class output without calling it calibrated probability."""
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError("invalid_action_top_k")
    labels = _labels(id2label)
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (NUM_CLASSES,) or not np.isfinite(values).all():
        raise ValueError("unexpected_or_nonfinite_action_logits")
    shifted = values - float(values.max())
    weights = np.exp(shifted)
    denominator = float(weights.sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid_action_softmax")
    scores = weights / denominator
    order = sorted(range(NUM_CLASSES), key=lambda index: (-float(scores[index]), index))[:top_k]
    ranked = []
    for index in order:
        label = labels[index]
        ranked.append({
            "class_id": index,
            "label": label,
            "score": round(float(scores[index]), 8),
            "score_kind": "softmax_score_uncalibrated",
            "semantic_hint": TARGET_LABELS.get(label),
            "rejection_signal": _is_rejection_label(label),
        })
    first = ranked[0]
    if first["rejection_signal"]:
        decision, hints, reason = "reject", [], "pretending_or_failed_action_top_class"
    elif first["semantic_hint"] is not None:
        decision, hints, reason = "auxiliary_candidate", [first["semantic_hint"]], None
    else:
        decision, hints, reason = "unknown", [], "top_class_outside_supported_auxiliary_mapping"
    return {
        "decision": decision,
        "reason": reason,
        "semantic_hints": hints,
        "ranked": ranked,
        "score_kind": "softmax_score_uncalibrated",
        "review_status": "unreviewed",
        "authority": "auxiliary_window_semantics_only",
        "unsupported_actions": list(UNSUPPORTED_ACTIONS),
    }


def _rational(value: Any) -> Tuple[Dict[str, str], Fraction]:
    if not isinstance(value, dict) or set(value) != {"n", "d"}:
        raise ValueError("frame_pts_must_be_a_reduced_rational")
    try:
        fraction = Fraction(int(value["n"]), int(value["d"]))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("frame_pts_must_be_a_reduced_rational") from exc
    canonical = {"n": str(fraction.numerator), "d": str(fraction.denominator)}
    if value != canonical:
        raise ValueError("frame_pts_must_be_a_reduced_rational")
    return canonical, fraction


def _window(samples: Sequence[Dict[str, Any]], *,
            cancelled: Optional[Callable[[], bool]]) -> Tuple[List[Any], Dict[str, Any]]:
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled_must_be_callable")
    if not isinstance(samples, (list, tuple)) or len(samples) != NUM_FRAMES:
        raise ValueError("videomae_requires_exactly_16_frames")
    images: List[Any] = []
    identities: List[Dict[str, Any]] = []
    previous_frame: Optional[int] = None
    previous_pts: Optional[Fraction] = None
    scene_index: Optional[int] = None
    frame_shape: Optional[Tuple[int, int, int]] = None
    total_bytes = 0
    for sample in samples:
        if cancelled is not None and cancelled():
            raise InterruptedError("video_action_analysis_cancelled")
        if not isinstance(sample, dict):
            raise ValueError("invalid_action_frame_sample")
        scene = sample.get("scene_index")
        if type(scene) is not int or scene < 0:
            raise ValueError("scene_index_must_be_a_nonnegative_integer")
        if scene_index is None:
            scene_index = scene
        elif scene != scene_index:
            raise ValueError("video_action_window_must_not_cross_scene_boundary")
        raw_frame = sample.get("frame_index")
        if isinstance(raw_frame, bool):
            raise ValueError("frame_index_must_be_a_canonical_nonnegative_integer")
        try:
            frame_index = int(raw_frame)
        except (TypeError, ValueError) as exc:
            raise ValueError("frame_index_must_be_a_canonical_nonnegative_integer") from exc
        if frame_index < 0 or (not isinstance(raw_frame, int) and str(frame_index) != raw_frame):
            raise ValueError("frame_index_must_be_a_canonical_nonnegative_integer")
        pts, pts_value = _rational(sample.get("pts"))
        if (previous_frame is not None
                and (frame_index <= previous_frame or pts_value <= previous_pts)):
            raise ValueError("action_frames_and_pts_must_be_strictly_increasing")
        image = sample.get("image")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) <= 0
                or int(image.shape[0]) * int(image.shape[1]) > MAX_FRAME_PIXELS):
            raise ValueError("action_frame_requires_bounded_uint8_bgr")
        if frame_shape is None:
            frame_shape = tuple(int(value) for value in image.shape)
        elif tuple(int(value) for value in image.shape) != frame_shape:
            raise ValueError("video_action_window_requires_one_frame_geometry")
        total_bytes += int(image.nbytes)
        if total_bytes > MAX_WINDOW_BYTES:
            raise ValueError("action_window_exceeds_byte_budget")
        evidence = sample.get("evidence_sha256")
        if (not isinstance(evidence, str) or len(evidence) != 64
                or any(character not in "0123456789abcdef" for character in evidence)):
            raise ValueError("action_frame_requires_sha256_evidence")
        from PIL import Image
        images.append(Image.fromarray(image[..., ::-1], mode="RGB"))
        identities.append({"frame_index": str(frame_index), "pts": pts,
                           "scene_index": scene,
                           "evidence_sha256": evidence,
                           "pixel_sha256": hashlib.sha256(
                               np.ascontiguousarray(image).tobytes()).hexdigest(),
                           "shape": list(frame_shape)})
        previous_frame, previous_pts = frame_index, pts_value
    return images, {
        "start_frame": identities[0]["frame_index"],
        "end_frame": identities[-1]["frame_index"],
        "window_pts": {"start": identities[0]["pts"], "end": identities[-1]["pts"]},
        "scene_index": scene_index,
        "frames": identities,
        "total_input_bytes": total_bytes,
    }


class VideoActionAuxiliary(BaseExtractor):
    """Exact local VideoMAE checkpoint; results are never authoritative actions."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.runtime_version = "unavailable"
        self.torch_version = "unavailable"
        self._verified = False
        self._status = "not_run"
        self._reason = "video_action_weights_unavailable"
        self._hashes: Dict[str, Optional[str]] = {
            "weights": None, "config": None, "processor": None}
        if model_path:
            self._load(Path(model_path))

    @property
    def name(self) -> str:
        return "action_video_auxiliary"

    def _load(self, root: Path) -> None:
        try:
            root_state = root.lstat()
            root_reparse = bool(getattr(root_state, "st_file_attributes", 0) & 0x400)
            if (stat.S_ISLNK(root_state.st_mode) or root_reparse
                    or not stat.S_ISDIR(root_state.st_mode)):
                raise ValueError("video_action_model_directory_missing")
            weights = root / WEIGHTS_FILE
            if not weights.is_file() or weights.stat().st_size != WEIGHTS_SIZE:
                raise ValueError("video_action_weights_size_mismatch")
            self._hashes = {
                "weights": _sha256(weights, max_bytes=MAX_MODEL_BYTES),
                "config": _sha256(root / "config.json", max_bytes=1024 * 1024),
                "processor": _sha256(root / "preprocessor_config.json", max_bytes=1024 * 1024),
            }
            if self._hashes != {"weights": WEIGHTS_SHA256, "config": CONFIG_SHA256,
                                "processor": PROCESSOR_SHA256}:
                raise ValueError("unverified_video_action_model_identity")
            enforce_offline_environment()
            import torch
            import transformers
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
            self.runtime_version = transformers.__version__
            self.torch_version = torch.__version__
            self.processor = VideoMAEImageProcessor.from_pretrained(
                str(root), local_files_only=True)
            self.model = VideoMAEForVideoClassification.from_pretrained(
                str(root), local_files_only=True, weights_only=True)
            # The loader consumes paths, so re-check every frozen file after it
            # returns. A replacement or in-place edit invalidates the result.
            after_load = {
                "weights": _sha256(weights, max_bytes=MAX_MODEL_BYTES),
                "config": _sha256(root / "config.json", max_bytes=1024 * 1024),
                "processor": _sha256(root / "preprocessor_config.json", max_bytes=1024 * 1024),
            }
            if after_load != self._hashes:
                raise ValueError("video_action_model_changed_during_load")
            _labels(self.model.config.id2label)
            if int(getattr(self.model.config, "num_frames", -1)) != NUM_FRAMES:
                raise ValueError("unexpected_videomae_frame_contract")
            self.model.eval()
            self._verified = True
            self._status, self._reason = "success", None
        except ImportError:
            self.processor = None
            self.model = None
            self._verified = False
            self._status = "not_run"
            self._reason = "video_action_runtime_unavailable"
        except Exception as exc:
            self.processor = None
            self.model = None
            self._verified = False
            self._status = "failed"
            self._reason = safe_failure_reason(exc, "video_action_model_load_failed")

    @property
    def code_identity(self) -> Dict[str, Any]:
        return {
            "provider": "transformers",
            "model_name": PROVIDER if self._verified else "unverified_local_model",
            "upstream_model": UPSTREAM_MODEL if self._verified else None,
            "upstream_revision": UPSTREAM_REVISION if self._verified else None,
            "runtime_version": self.runtime_version,
            "torch_version": self.torch_version,
            "weights_sha256": self._hashes["weights"],
            "config_sha256": self._hashes["config"],
            "preprocessor_config_sha256": self._hashes["processor"],
            "weights_license": WEIGHTS_LICENSE if self._verified else None,
            "weights_license_source": WEIGHTS_LICENSE_SOURCE if self._verified else None,
            "runtime_license": "Apache-2.0 (Transformers)",
            "local_files_only": True,
            "num_frames": NUM_FRAMES,
            "authority": "auxiliary_window_semantics_only",
            "supported_hints": ["close", "open", "pick_up", "put_down_surface"],
            "unsupported_actions": list(UNSUPPORTED_ACTIONS),
        }

    def analyze_window(self, samples: Sequence[Dict[str, Any]], *, top_k: int = 5,
                       cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        if self.model is None or self.processor is None:
            return {"status": self._status, "reason": self._reason,
                    "decision": "not_run", "semantic_hints": [],
                    "review_status": "unreviewed",
                    "authority": "auxiliary_window_semantics_only"}
        try:
            if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
                raise ValueError("invalid_action_top_k")
            images, window = _window(samples, cancelled=cancelled)
            if cancelled is not None and cancelled():
                raise InterruptedError("video_action_analysis_cancelled")
            inputs = self.processor(images=images, return_tensors="pt")
            if cancelled is not None and cancelled():
                raise InterruptedError("video_action_analysis_cancelled")
            import torch
            with torch.no_grad():
                output = self.model(**inputs)
            if cancelled is not None and cancelled():
                raise InterruptedError("video_action_analysis_cancelled")
            logits = output.logits
            if getattr(logits, "ndim", None) != 2 or tuple(logits.shape) != (1, NUM_CLASSES):
                raise ValueError("unexpected_video_action_output_shape")
            result = interpret_logits(logits[0].detach().cpu().tolist(),
                                      self.model.config.id2label, top_k=top_k)
            result.update({
                "status": "success", "reason": result["reason"],
                "provider": PROVIDER, "model_identity": self.code_identity,
                "window": window, "subject_ref": None, "object_ref": None,
                "recipient_ref": None,
                "uncertainty": (
                    "Window classification does not identify a subject, object track, recipient, "
                    "exact interval, completed action, or calibrated probability."
                ),
            })
            return result
        except InterruptedError:
            raise
        except Exception as exc:
            return {"status": "failed",
                    "reason": safe_failure_reason(
                        exc, "video_action_inference_failed"),
                    "decision": "not_run", "semantic_hints": [],
                    "review_status": "unreviewed",
                    "authority": "auxiliary_window_semantics_only"}

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        return {"status": "not_run", "reason": "video_window_samples_required",
                "decision": "not_run", "semantic_hints": [],
                "review_status": "unreviewed",
                "authority": "auxiliary_window_semantics_only"}


def _isolated_worker_command(request: Path, output: Path, progress: Path) -> List[str]:
    """Build the child command without importing torch in the parent process."""
    package_root = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import os,sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
        "from trace_annotate.workspace.resource_budget import join_owned_job;"
        "join_owned_job(os.environ.get('TRACE_ANNOTATE_OWNED_MODEL_JOB',''));"
        "runpy.run_module('trace_annotate.extractors.action_video_worker',run_name='__main__')"
    )
    return [sys.executable, "-X", "utf8", "-I", "-B", "-c", bootstrap,
            str(package_root), "--request", str(request), "--output", str(output),
            "--progress", str(progress)]


def _write_isolated_request(model_path: str, samples: Sequence[Dict[str, Any]],
                            work: Path, *, top_k: int,
                            cancelled: Callable[[], bool]) -> Tuple[Path, str]:
    if not callable(cancelled):
        raise ValueError("cancelled_must_be_callable")
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError("invalid_action_top_k")
    model_root = Path(model_path)
    if not model_root.is_absolute():
        raise ValueError("video_action_model_directory_must_be_absolute")
    model_root = model_root.resolve(strict=False)
    # This performs all cheap shape/order/evidence checks before a process is
    # admitted.  It does not load the model or call its forward method.
    _, identity = _window(samples, cancelled=cancelled)
    work.mkdir(parents=True, exist_ok=False)
    frames_root = work / "frames"
    frames_root.mkdir()
    frames: List[Dict[str, Any]] = []
    for index, (sample, frame_identity) in enumerate(zip(samples, identity["frames"])):
        if cancelled():
            raise InterruptedError("video_action_analysis_cancelled")
        frame_path = frames_root / f"{index:02d}.npy"
        with frame_path.open("xb") as stream:
            np.save(stream, np.ascontiguousarray(sample["image"]), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        frames.append({
            **frame_identity,
            "file": f"frames/{frame_path.name}",
            "file_sha256": _sha256(frame_path, max_bytes=MAX_WINDOW_BYTES),
        })
    request_value = {
        "contract": ISOLATED_CONTRACT,
        "provider": PROVIDER,
        "model_path": str(model_root),
        "top_k": top_k,
        "window": {key: identity[key] for key in (
            "start_frame", "end_frame", "window_pts", "scene_index",
            "total_input_bytes")},
        "frames": frames,
        "authority": "auxiliary_window_semantics_only",
        "participants_must_remain_null": True,
    }
    raw = json.dumps(request_value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_ISOLATED_REQUEST_BYTES:
        raise ValueError("isolated_action_request_exceeds_byte_budget")
    request_path = work / "request.json"
    with request_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return request_path, hashlib.sha256(raw).hexdigest()


def analyze_window_isolated(model_path: str, samples: Sequence[Dict[str, Any]],
                            work: Path, *, top_k: int = 5,
                            cancelled: Callable[[], bool],
                            timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Run one VideoMAE window in an owned, cancellable process tree.

    The only successful output is a fully decoded and PID-bound result file.
    Cancellation, timeout, memory failure or a protocol mismatch terminates the
    owned tree and removes any incomplete result.  Inputs and process receipts
    remain in ``work`` as local diagnostic evidence.
    """
    from ..workspace import resource_budget as resources
    from ..workspace.errors import WorkspaceError

    target = Path(work)
    if not target.is_absolute():
        raise ValueError("isolated_action_work_directory_must_be_absolute")
    request_path, request_sha256 = _write_isolated_request(
        model_path, samples, target, top_k=top_k, cancelled=cancelled)
    output_path = target / "result.json"
    partial_path = target / "result.json.partial"
    progress_path = target / "progress.json"
    settings = resources.policy()
    timeout = settings["phase_seconds"] if timeout_seconds is None else timeout_seconds
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout)) or not 0 < float(timeout) <= settings["phase_seconds"]):
        raise ValueError("invalid_action_worker_timeout")
    command = _isolated_worker_command(request_path, output_path, progress_path)
    try:
        execution = resources.run_owned(
            command, target / "execution", settings=settings, cancelled=cancelled,
            timeout=float(timeout), progress_path=progress_path)
        if not output_path.is_file() or not 0 < output_path.stat().st_size <= MAX_ISOLATED_OUTPUT_BYTES:
            raise WorkspaceError("MODEL_WORKER_PROTOCOL",
                                 "Video action worker result is missing or exceeds its byte limit")
        try:
            value = json.loads(output_path.read_text("utf-8"))
            request_value = json.loads(request_path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkspaceError("MODEL_WORKER_PROTOCOL",
                                 "Video action worker result cannot be decoded") from exc
        expected_keys = {"contract", "provider", "request_sha256", "result",
                         "actual_pid", "actual_parent_pid", "peak_working_set_bytes"}
        result = value.get("result") if isinstance(value, dict) else None
        wrapper_pid = execution.get("wrapper_pid")
        parent_pid = execution.get("parent_pid")
        actual_pid = value.get("actual_pid") if isinstance(value, dict) else None
        actual_parent_pid = value.get("actual_parent_pid") if isinstance(value, dict) else None
        direct_worker = actual_pid == wrapper_pid and actual_parent_pid == parent_pid
        launcher_child = (type(actual_pid) is int and actual_pid > 0
                          and actual_pid != wrapper_pid
                          and actual_parent_pid == wrapper_pid)
        if (not isinstance(value, dict) or set(value) != expected_keys
                or value.get("contract") != ISOLATED_CONTRACT
                or value.get("provider") != PROVIDER
                or value.get("request_sha256") != request_sha256
                or not isinstance(result, dict)
                or result.get("status") not in {"success", "failed", "not_run"}
                or result.get("authority") != "auxiliary_window_semantics_only"
                or result.get("review_status") != "unreviewed"
                or any(result.get(name) is not None
                       for name in ("subject_ref", "object_ref", "recipient_ref"))
                or execution.get("status") != "completed"
                or type(wrapper_pid) is not int or wrapper_pid <= 0
                or type(parent_pid) is not int or parent_pid <= 0
                or (value.get("peak_working_set_bytes") is not None
                    and (type(value.get("peak_working_set_bytes")) is not int
                         or value["peak_working_set_bytes"] < 0))
                or not (direct_worker or launcher_child)):
            raise WorkspaceError("MODEL_WORKER_PROTOCOL",
                                 "Video action worker identity or authority is inconsistent")
        if result["status"] == "success":
            identity = result.get("model_identity")
            actual_window = result.get("window")
            ranked = result.get("ranked")
            hints = result.get("semantic_hints")
            expected_frames = [
                {key: frame[key] for key in (
                    "frame_index", "pts", "scene_index", "evidence_sha256",
                    "pixel_sha256", "shape")}
                for frame in request_value["frames"]
            ]
            if (result.get("provider") != PROVIDER or not isinstance(identity, dict)
                    or set(identity) != MODEL_IDENTITY_FIELDS
                    or identity.get("provider") != "transformers"
                    or identity.get("model_name") != PROVIDER
                    or identity.get("upstream_model") != UPSTREAM_MODEL
                    or identity.get("weights_sha256") != WEIGHTS_SHA256
                    or identity.get("config_sha256") != CONFIG_SHA256
                    or identity.get("preprocessor_config_sha256") != PROCESSOR_SHA256
                    or identity.get("upstream_revision") != UPSTREAM_REVISION
                    or identity.get("weights_license") != WEIGHTS_LICENSE
                    or identity.get("weights_license_source") != WEIGHTS_LICENSE_SOURCE
                    or identity.get("local_files_only") is not True
                    or identity.get("authority") != "auxiliary_window_semantics_only"
                    or identity.get("num_frames") != NUM_FRAMES
                    or identity.get("supported_hints")
                    != ["close", "open", "pick_up", "put_down_surface"]
                    or identity.get("unsupported_actions") != list(UNSUPPORTED_ACTIONS)
                    or not isinstance(actual_window, dict)
                    or {key: actual_window.get(key) for key in request_value["window"]}
                    != request_value["window"]
                    or actual_window.get("frames") != expected_frames
                    or result.get("decision") not in {"auxiliary_candidate", "reject", "unknown"}
                    or result.get("unsupported_actions") != list(UNSUPPORTED_ACTIONS)
                    or not isinstance(result.get("uncertainty"), str)
                    or not result["uncertainty"].strip()
                    or result.get("score_kind") != "softmax_score_uncalibrated"
                    or not isinstance(ranked, list) or not 0 < len(ranked) <= request_value["top_k"]
                    or not isinstance(hints, list)
                    or not set(hints) <= {"close", "open", "pick_up", "put_down_surface"}):
                raise WorkspaceError("MODEL_WORKER_PROTOCOL",
                                     "Video action worker used an unverified checkpoint")
            try:
                _validate_ranked_result(result, top_k=request_value["top_k"])
            except ValueError as exc:
                raise WorkspaceError("MODEL_WORKER_PROTOCOL",
                                     "Video action worker returned inconsistent semantics") from exc
        elif (not is_safe_failure_reason(result.get("reason"))
              or result.get("decision") != "not_run" or result.get("semantic_hints") != []
              or "ranked" in result or result.get("score_kind") is not None):
            raise WorkspaceError("MODEL_WORKER_PROTOCOL",
                                 "Unavailable video action worker returned a semantic claim")
        result = dict(result)
        result["execution"] = {
            **execution,
            "contract": ISOLATED_CONTRACT,
            "request_sha256": request_sha256,
            "actual_pid": value["actual_pid"],
            "actual_parent_pid": value["actual_parent_pid"],
            "peak_working_set_bytes": value["peak_working_set_bytes"],
        }
        return result
    except BaseException:
        # A killed worker may have opened its temporary output.  It is never a
        # result and must not be mistaken for a resumable or partial candidate.
        for path in (partial_path, output_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
