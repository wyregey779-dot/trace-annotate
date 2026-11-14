"""Opt-in acceptance check for the frozen VideoMAE SSV2 auxiliary model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from trace_annotate.extractors import action_video as A


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


def _public_execution(value: object) -> dict | None:
    """Keep reproducibility metrics without publishing host paths or PIDs."""
    if not isinstance(value, dict):
        return None
    allowed = (
        "contract",
        "status",
        "actual_exit_code",
        "elapsed_seconds",
        "wait_seconds",
        "request_sha256",
        "required_bytes",
        "available_at_admission",
        "minimum_available_bytes",
        "peak_working_set_bytes",
    )
    public = {key: value[key] for key in allowed if key in value}
    host_lock = value.get("host_lock")
    if isinstance(host_lock, dict):
        lock_allowed = (
            "contract",
            "kind",
            "scope_sha256",
            "wait_seconds",
            "recovered_abandoned_owner",
        )
        public["host_lock"] = {
            key: host_lock[key] for key in lock_allowed if key in host_lock
        }
    return public


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("images", nargs=A.NUM_FRAMES, type=Path)
    args = parser.parse_args(argv)

    samples = []
    for index, path in enumerate(args.images):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        samples.append({
            "frame_index": str(index),
            "pts": {"n": str(index), "d": "1"},
            "scene_index": 0,
            "evidence_sha256": _sha256(path),
            "image": np.ascontiguousarray(rgb[:, :, ::-1]),
        })
    result = A.analyze_window_isolated(
        str(args.model), samples, args.work, cancelled=lambda: False,
        timeout_seconds=300.0)
    public = {
        "kind": "traceannotate.action-video-real-acceptance",
        "version": 1,
        "input_names": [path.name for path in args.images],
        "status": result.get("status"),
        "decision": result.get("decision"),
        "semantic_hints": result.get("semantic_hints"),
        "ranked": result.get("ranked"),
        "model_identity": result.get("model_identity"),
        "window": result.get("window"),
        "execution": _public_execution(result.get("execution")),
        "authority": result.get("authority"),
        "review_status": result.get("review_status"),
        "uncertainty": result.get("uncertainty"),
    }
    _write_json(args.output, public)
    summary = {key: public[key] for key in (
        "status", "decision", "semantic_hints", "authority", "review_status")}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if (public["status"] == "success"
                 and (public["execution"] or {}).get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
