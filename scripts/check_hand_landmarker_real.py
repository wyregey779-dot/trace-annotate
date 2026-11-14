"""Opt-in acceptance check for the frozen MediaPipe Hand Landmarker bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from trace_annotate.extractors.hands import HandLandmarkExtractor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args(argv)

    extractor = HandLandmarkExtractor(str(args.model))
    rows = []
    try:
        for index, path in enumerate(args.images):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            result = extractor.analyze_frame(
                bgr, scene_index=index, frame_index=str(index),
                pts={"n": str(index), "d": "1"},
                evidence_sha256=_sha256(path), model_run_id="run-real-hand-check",
                cancelled=lambda: False)
            rows.append({
                "input_name": path.name,
                "input_sha256": _sha256(path),
                "status": result["status"],
                "reason": result.get("reason"),
                "hand_count": len(result.get("hands") or []),
                "observations": result.get("hands") or [],
                "frame_identity": result.get("frame_identity"),
                "model_identity": result.get("model_identity"),
                "cancellation_boundary": result.get("cancellation_boundary"),
            })
    finally:
        extractor.close()

    all_executed = bool(rows) and all(row["status"] == "success" for row in rows)
    detected = sum(row["hand_count"] for row in rows)
    report = {
        "kind": "traceannotate.hand-provider-real-acceptance",
        "version": 1,
        "all_inputs_executed": all_executed,
        "at_least_one_hand_detected": detected > 0,
        "input_count": len(rows),
        "detected_hand_count": detected,
        "results": rows,
    }
    _write_json(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "all_inputs_executed", "at_least_one_hand_detected", "input_count",
        "detected_hand_count")}, ensure_ascii=False, sort_keys=True))
    return 0 if all_executed and detected > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
