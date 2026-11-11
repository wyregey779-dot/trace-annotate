"""Local RapidOCR adapters with explicit execution status and weight identity."""
from __future__ import annotations
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
from typing import Dict, Any, Optional
import numpy as np

from trace_annotate.extractors.base import BaseExtractor
from trace_annotate.core.hasher import safe_load_image
from trace_annotate.config import MetricThresholds

LEGACY_MODELS = {"det": "ch_PP-OCRv4_det_infer.onnx",
                 "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                 "rec": "ch_PP-OCRv4_rec_infer.onnx"}
# Published in rapidocr 3.10.0's MODEL_LICENSES.md. Explicit files and matching
# identities keep the modern backend on its bundled ONNX/embedded-dictionary path.
MODERN_MODELS = {
    "det": ("PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    "rec": ("PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_result(status: str, *, reason: Optional[str] = None,
                  error: Optional[str] = None) -> Dict[str, Any]:
    result = {"status": status, "ocr_text": "", "ocr_readable": "",
              "ocr_clean": "", "ocr_boxes": [], "review_status": "unreviewed"}
    if reason is not None:
        result["reason"] = reason
    if error is not None:
        result["error"] = error
    return result


class OCRExtractor(BaseExtractor):
    """Extracts on-screen and sign text using RapidOCR."""

    def __init__(self, thresholds: Optional[MetricThresholds] = None,
                 model_dir: Optional[str] = None):
        self.thresholds = thresholds or MetricThresholds()
        self.engine = None
        self.load_error = None
        self._load_status = "not_run"
        self.runtime_backend = None
        self.model_dir = model_dir or os.environ.get("TRACE_ANNOTATE_OCR_MODELS")
        self._weights_files = []
        self._init_engine()

    @property
    def name(self) -> str:
        return "ocr"

    @property
    def code_identity(self) -> Dict[str, Any]:
        """Hash all three selected files, keeping code and model licenses separate."""
        try:
            distribution = metadata.distribution(self.runtime_backend)
            runtime_version = distribution.version
        except (metadata.PackageNotFoundError, TypeError, ValueError):
            distribution = None
            runtime_version = "unavailable"
        licenses = []
        if distribution is not None:
            for item in distribution.files or []:
                if Path(str(item)).name in {"LICENSE", "MODEL_LICENSES.md"}:
                    path = Path(distribution.locate_file(item))
                    if path.is_file():
                        licenses.append({"path": str(path), "sha256": _file_sha256(path)})
        weights = [dict(item) for item in self._weights_files]
        hashes = {item["role"]: item["sha256"] for item in weights}
        combined = hashlib.sha256(json.dumps(hashes, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")).hexdigest() if len(hashes) == 3 else None
        model_license = ("Apache-2.0" if self.runtime_backend == "rapidocr" and len(weights) == 3
                         and any(Path(item["path"]).name == "MODEL_LICENSES.md" for item in licenses)
                         else "unverified; inspect the selected model's upstream terms")
        return {"provider": "RapidOCR",
                "model_name": "RapidOCR default det+cls+rec",
                "model_version": runtime_version,
                "runtime_backend": self.runtime_backend,
                "weights_path": None,
                "weights_files": weights, "weights_sha256": combined,
                "weights_hash_policy": "sha256 of canonical sorted role-to-file-sha256 JSON",
                "license": "Apache-2.0" if licenses else "unverified",
                "code_license": "Apache-2.0" if licenses else "unverified",
                "weights_license": model_license, "license_evidence": licenses,
                "offline": True, "load_error": self.load_error}

    def _init_engine(self) -> None:
        try:
            try:
                package = importlib.import_module("rapidocr_onnxruntime")
                self.runtime_backend = "rapidocr_onnxruntime"
            except ImportError:
                package = importlib.import_module("rapidocr")
                self.runtime_backend = "rapidocr"
            directory = Path(self.model_dir) if self.model_dir else Path(package.__file__).parent / "models"
            names = (LEGACY_MODELS if self.runtime_backend == "rapidocr_onnxruntime"
                     else {role: entry[0] for role, entry in MODERN_MODELS.items()})
            paths = {role: directory / name for role, name in names.items()}
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                self.load_error = "Local OCR weights unavailable; downloads are disabled: " + ", ".join(missing)
                return
            for role, path in paths.items():
                digest = _file_sha256(path)
                if self.runtime_backend == "rapidocr" and digest != MODERN_MODELS[role][1]:
                    raise ValueError(f"Local {role} weight differs from the supported bundled OCR model")
                self._weights_files.append({"role": role, "path": str(path.resolve()),
                                            "sha256": digest, "size_bytes": path.stat().st_size})
            if self.runtime_backend == "rapidocr_onnxruntime":
                self.engine = package.RapidOCR(**{role + "_model_path": str(path) for role, path in paths.items()})
            else:
                params = {role.capitalize() + ".model_path": str(path) for role, path in paths.items()}
                params.update({"EngineConfig.onnxruntime.intra_op_num_threads": 1,
                               "EngineConfig.onnxruntime.inter_op_num_threads": 1})
                self.engine = package.RapidOCR(params=params)
        except ImportError as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"[-600:]
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"[-600:]
            self._load_status = "failed"

    def extract_text(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Extract text elements from an image array."""
        if (not isinstance(img_bgr, np.ndarray) or img_bgr.ndim not in (2, 3)
                or img_bgr.size == 0 or img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0
                or (img_bgr.ndim == 3 and img_bgr.shape[2] not in (1, 3, 4))
                or img_bgr.dtype.kind not in "buif" or not np.isfinite(img_bgr).all()):
            return _empty_result("failed", reason="invalid_frame", error="OCR requires a nonempty finite image array")
        h = img_bgr.shape[0]
        if self.engine is None:
            return _empty_result(self._load_status, reason="ocr_runtime_unavailable", error=self.load_error)
        try:
            output = self.engine(img_bgr)
            if self.runtime_backend == "rapidocr":
                values = (output.boxes, output.txts, output.scores)
                if all(value is None for value in values):
                    result = []
                else:
                    if any(value is None for value in values) or len({len(value) for value in values}) != 1:
                        raise ValueError("OCR result fields have inconsistent lengths")
                    result = zip(*values)
            else:
                result, _ = output
                if result is None:
                    result = []
            all_texts, readable_texts, clean_texts, boxes_out = [], [], [], []
            for item in result:
                if item is None or len(item) < 3:
                    raise ValueError("OCR returned a malformed text record")
                box, text, score = item[0], str(item[1]).strip(), float(item[2])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("OCR returned an invalid confidence")
                height_ratio = 0.0
                if box is not None:
                    pts = np.asarray(box)
                    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2 or not np.isfinite(pts).all():
                        raise ValueError("OCR returned an invalid quadrilateral")
                    height_ratio = (float(np.max(pts[:, 1])) - float(np.min(pts[:, 1]))) / h
                    box = pts.tolist()
                is_readable = (score >= self.thresholds.ocr_min_confidence
                               and height_ratio >= self.thresholds.ocr_min_height_ratio)
                all_texts.append(text)
                if is_readable:
                    readable_texts.append(text)
                cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
                if len(cleaned) >= 2:
                    clean_texts.append(cleaned)
                boxes_out.append({"text": text, "score": round(score, 3),
                                  "is_readable": is_readable, "box": box})
            return {"status": "success", "ocr_text": " ".join(all_texts),
                    "ocr_readable": " ".join(readable_texts), "ocr_clean": " ".join(clean_texts),
                    "ocr_boxes": boxes_out, "review_status": "unreviewed"}
        except Exception as exc:
            return _empty_result("failed", reason="ocr_inference_failed", error=f"{type(exc).__name__}: {exc}"[-600:])

    def extract(self, asset_item: Dict[str, Any], staged_path: str) -> Dict[str, Any]:
        try:
            return self.extract_text(safe_load_image(staged_path))
        except Exception as exc:
            return _empty_result("failed", reason="invalid_frame", error=f"{type(exc).__name__}: {exc}"[-600:])
