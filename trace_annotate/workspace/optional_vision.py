"""Frozen local provider configuration and a bounded optional-runtime worker.

This module never searches caches, downloads weights or imports a model runtime
when the workbench asks for capabilities. Paths are explicit deployment inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .errors import SchemaInvalid, WorkspaceError

CONTRACT = "optional-local-providers-v1"
WORKER_TIMEOUT = 180
MAX_WORKER_OUTPUT = 16 * 1024 * 1024
MAX_WORKER_LOG = 1024 * 1024
MAX_MODEL_BYTES = 64 * 1024 * 1024
PROTOCOL = "traceannotate.face-landmarks.v1"


def _file(path: Optional[str], *, limit: int = MAX_MODEL_BYTES) -> Dict[str, Any]:
    if not path:
        return {"path": None, "sha256": None, "state": "missing"}
    target = Path(path).expanduser().absolute()
    result = {"path": str(target), "sha256": None, "state": "missing"}
    try:
        if target.is_file():
            if target.stat().st_size > limit:
                result["state"] = "oversize"
            else:
                # Stream hashes: configuration cannot allocate an entire model.
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                result.update(sha256=digest.hexdigest(), state="available")
    except OSError:
        result["state"] = "unreadable"
    return result


def _worker_runtime(executable: Optional[str]) -> Dict[str, Any]:
    result = {"mediapipe_version": "unavailable", "metadata_sha256": None}
    if not executable:
        return result
    root = Path(executable).parent.parent
    sites = [root / "Lib" / "site-packages"]
    sites.extend(root.glob("lib/python*/site-packages"))
    metadata = [path for site in sites for path in site.glob("mediapipe-*.dist-info/METADATA")]
    if len(metadata) == 1:
        try:
            data = metadata[0].read_bytes()
            version = re.search(rb"(?m)^Version: ([^\r\n]+)", data)
            result = {"mediapipe_version": version[1].decode("ascii") if version else "unknown",
                      "metadata_sha256": hashlib.sha256(data).hexdigest()}
        except (OSError, UnicodeError):
            pass
    return result


def provider_config(*, person_model: Optional[str] = None,
                    landmark_model: Optional[str] = None,
                    landmark_python: Optional[str] = None) -> Dict[str, Any]:
    person_model = person_model if person_model is not None else os.getenv("TRACE_ANNOTATE_PERSON_MODEL")
    landmark_model = landmark_model if landmark_model is not None else os.getenv("TRACE_ANNOTATE_LANDMARK_MODEL")
    landmark_python = landmark_python if landmark_python is not None else os.getenv("TRACE_ANNOTATE_LANDMARK_PYTHON")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "extractors" / "persons.py",
               root / "extractors" / "face_landmarks.py", root / "extractors" / "expressive_cues.py"]
    implementation = {str(path.relative_to(root)): _file(str(path))["sha256"] for path in sources}
    from .schema import sha256_text
    try:
        from importlib.metadata import version
        onnx = version("onnxruntime")
    except Exception:
        onnx = "unavailable"
    person_file, landmark_file, python_file = _file(person_model), _file(landmark_model), _file(landmark_python)
    return {"contract": CONTRACT, "implementation_sha256": sha256_text(implementation),
            "person": {"model_path": person_file["path"],
                       "weights_sha256": person_file["sha256"],
                       "file_state": person_file["state"], "runtime_version": onnx},
            "landmarks": {"model_path": landmark_file["path"],
                          "weights_sha256": landmark_file["sha256"],
                          "file_state": landmark_file["state"],
                          "worker_python": python_file["path"],
                          "worker_python_sha256": python_file["sha256"],
                          "runtime": _worker_runtime(landmark_python)},
            "worker_timeout_seconds": WORKER_TIMEOUT, "max_output_bytes": MAX_WORKER_OUTPUT}


def validate_config(value: Any) -> Dict[str, Any]:
    """Validate a frozen descriptor without consulting current files or code."""
    def digest(item):
        return item is None or isinstance(item, str) and re.fullmatch("[0-9a-f]{64}", item) is not None
    def path(item):
        return item is None or isinstance(item, str) and 0 < len(item) <= 32768
    expected = {"contract", "implementation_sha256", "person", "landmarks", "worker_timeout_seconds", "max_output_bytes"}
    valid = isinstance(value, dict) and set(value) == expected
    if valid:
        p, l = value["person"], value["landmarks"]
        valid = (value["contract"] == CONTRACT and digest(value["implementation_sha256"])
                 and value["implementation_sha256"] is not None
                 and value["worker_timeout_seconds"] == WORKER_TIMEOUT
                 and value["max_output_bytes"] == MAX_WORKER_OUTPUT
                 and isinstance(p, dict) and set(p) == {"model_path", "weights_sha256", "file_state", "runtime_version"}
                 and isinstance(l, dict) and set(l) == {"model_path", "weights_sha256", "file_state", "worker_python", "worker_python_sha256", "runtime"})
        if valid:
            valid = (all(path(item["model_path"]) and digest(item["weights_sha256"])
                         and item["file_state"] in {"missing", "oversize", "unreadable", "available"}
                         and (item["file_state"] != "available" or item["weights_sha256"] is not None)
                         for item in (p, l))
                     and isinstance(p["runtime_version"], str) and len(p["runtime_version"]) <= 120
                     and path(l["worker_python"]) and digest(l["worker_python_sha256"])
                     and isinstance(l["runtime"], dict) and set(l["runtime"]) == {"mediapipe_version", "metadata_sha256"}
                     and isinstance(l["runtime"]["mediapipe_version"], str)
                     and digest(l["runtime"]["metadata_sha256"]))
    if not valid:
        raise SchemaInvalid("SCHEMA_INVALID", "Invalid frozen optional provider configuration")
    return value


def run_worker(command: List[str], log_path: Path, *, cancelled: Callable[[], bool],
               timeout: float) -> int:
    """Wait with bounded diagnostics; cancellation reaps only our child PID."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if cancelled():
        raise WorkspaceError("CANCELLED", "Landmark worker cancelled before launch")
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancelled():
                    raise WorkspaceError("CANCELLED", "Landmark worker cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("landmark_worker_timeout")
                if log_path.stat().st_size > MAX_WORKER_LOG:
                    raise ValueError("landmark_worker_log_budget_exceeded")
                time.sleep(.05)
            return process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def run_landmarks(config: Dict[str, Any], images: List[Dict[str, Any]], work: Path, *,
                  cancelled: Callable[[], bool]) -> Dict[str, Any]:
    """Use the optional MediaPipe venv without importing it into the server."""
    settings = validate_config(config)["landmarks"]
    empty = {"status": "not_run", "results": []}
    if settings["file_state"] == "missing":
        return dict(empty, reason="landmark_weights_unavailable")
    if settings["file_state"] != "available":
        return dict(empty, status="failed", reason="landmark_weights_" + settings["file_state"])
    if not settings["worker_python_sha256"]:
        return dict(empty, reason="landmark_worker_python_unavailable")
    if not images:
        return dict(empty, reason="no_sampled_frames")
    directory = work / "landmarks"
    directory.mkdir(parents=True, exist_ok=False)
    request_path, output_path = directory / "request.json", directory / "output.json"
    request = {"schema": PROTOCOL, "model_path": settings["model_path"],
               "expected_weights_sha256": settings["weights_sha256"], "max_faces": 4,
               "images": images}
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    bootstrap = ("import sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
                 "runpy.run_module('trace_annotate.extractors.face_landmarks',run_name='__main__')")
    package_root = str(Path(__file__).resolve().parents[2])
    command = [settings["worker_python"], "-I", "-B", "-c", bootstrap, package_root,
               "--request-json", str(request_path), "--output-json", str(output_path)]
    try:
        exit_code = run_worker(command, directory / "worker.log", cancelled=cancelled,
                               timeout=config["worker_timeout_seconds"])
        if not output_path.is_file() or not 0 < output_path.stat().st_size <= config["max_output_bytes"]:
            raise ValueError("landmark_worker_output_missing_or_oversize")
        value = json.loads(output_path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or value.get("schema") != PROTOCOL
                or value.get("status") not in {"success", "not_run", "failed"}
                or not isinstance(value.get("results"), list)
                or len(value["results"]) != len(images)
                or any(result.get("frame_index") != item["frame_index"]
                       or result.get("path") != item["path"]
                       for item, result in zip(images, value["results"]))
                or exit_code != (1 if value["status"] == "failed" else 0)):
            raise ValueError("landmark_worker_protocol_mismatch")
        for result in value["results"]:
            identity = result.get("model_identity") or {}
            if (identity.get("weights_sha256") != settings["weights_sha256"]
                    or (settings["runtime"]["mediapipe_version"] != "unavailable"
                        and identity.get("runtime_version") != settings["runtime"]["mediapipe_version"])):
                raise ValueError("landmark_worker_identity_changed")
        return value
    except WorkspaceError:
        raise
    except Exception as exc:
        return dict(empty, status="failed", reason=str(exc)[:300])
