"""One CPU model per owned process, using bounded JSON and decoded PNG inputs."""
from __future__ import annotations

import copy
import hashlib
import ctypes
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import sys
import time

from . import resource_budget as R
from .errors import WorkspaceError

CONTRACT = "isolated-frame-models-v1"
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 512
PROGRESS_REPLACE_ATTEMPTS = 20
PROGRESS_RETRY_SECONDS = .05

# Formal workspace inference supports only named deployments whose cache
# repository can be resolved without importing torch/open_clip.  This prevents
# an unrelated checkpoint elsewhere in a shared cache from making a missing
# selection look available.
CLIP_REPOSITORIES = {
    "ViT-B-32": "timm/vit_base_patch32_clip_224.openai",
    "ViT-L-14": "timm/vit_large_patch14_clip_224.openai",
    "RN50": "timm/resnet50_clip.openai",
}
CLIP_LEGACY_FILES = {
    "ViT-B-32": "ViT-B-32.pt",
    "ViT-L-14": "ViT-L-14.pt",
    "RN50": "RN50.pt",
}
MODEL_IMPLEMENTATION_SOURCE_FILES = (
    "__init__.py",
    "config.py",
    "core/hasher.py",
    "extractors/__init__.py",
    "extractors/base.py",
    "extractors/clip_engine.py",
    "extractors/faces.py",
    "extractors/metrics.py",
    "extractors/model_loading.py",
    "extractors/ocr.py",
    "extractors/scene.py",
    "extractors/vlm.py",
    "workspace/__init__.py",
    "workspace/errors.py",
    "workspace/model_worker.py",
    "workspace/resource_budget.py",
    "workspace/slots.py",
)


class ModelLoadFailed(RuntimeError):
    """A selected local checkpoint failed to construct a usable adapter."""


class ModelProgressFailed(RuntimeError):
    """Windows progress replacement remained unavailable after bounded retries."""


def _publish_progress(output, phase, frame_index=None):
    progress = output.parent / "progress.json"
    temporary = progress.with_suffix(".pending")
    value = {"phase": phase, "started_at": time.time()}
    if frame_index is not None:
        value["frame_index"] = frame_index
    temporary.write_text(json.dumps(value), encoding="utf-8")
    for attempt in range(PROGRESS_REPLACE_ATTEMPTS):
        try:
            temporary.replace(progress)
            return attempt
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            if attempt + 1 == PROGRESS_REPLACE_ATTEMPTS:
                raise ModelProgressFailed(
                    "Progress file replacement failed after " + str(PROGRESS_REPLACE_ATTEMPTS)
                    + " attempts (Windows error " + str(exc.winerror) + ")") from exc
            time.sleep(PROGRESS_RETRY_SECONDS)


def _reject_memory_failure(value):
    detail = str(value or "").lower()
    if any(token in detail for token in ("out of memory", "not enough memory", "bad allocation",
                                         "bad_alloc", "can't allocate", "cannot allocate", "defaultcpuallocator")):
        raise MemoryError(str(value))


def _check_loaded_adapter(adapter):
    _reject_memory_failure(adapter.load_error)
    identity = adapter.code_identity
    if adapter.model is None and identity.get("weights_path") and identity.get("weights_sha256"):
        raise ModelLoadFailed(adapter.load_error or "Selected checkpoint did not produce a usable model")
    return identity


def _digest(path):
    # Windows creation time and restored mtime cannot identify changed bytes.
    # Deployment boundaries deliberately re-read the complete immutable input.
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_identity(package_root=None,
                             relative_files=MODEL_IMPLEMENTATION_SOURCE_FILES):
    """Hash the local source closure that can change worker execution."""
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    return {relative: _digest(root / Path(relative)) for relative in relative_files}


def _inventory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and ".locks" not in path.parts and "__pycache__" not in path.parts:
            if len(result) >= 2048 or path.stat().st_size > 8 * R.GIB:
                raise WorkspaceError("MODEL_IDENTITY_LIMIT", "Model cache exceeds its identity scan budget")
            result.append({"file": path.relative_to(directory).as_posix(),
                           "bytes": path.stat().st_size, "sha256": _digest(path)})
    return result


def _clip_selection(cache, requested):
    """Return one selection root and only files that can satisfy it."""
    path = Path(requested)
    if path.is_file():
        return path.parent, True, [{"file": path.name, "bytes": path.stat().st_size,
                                    "sha256": _digest(path)}]
    if path.is_dir():
        return path, True, _inventory(path)
    repository = CLIP_REPOSITORIES.get(requested)
    if repository:
        selected = cache / ("models--" + repository.replace("/", "--"))
        inventory = []
        for item in _inventory(selected):
            inventory.append({**item, "file": selected.name + "/" + item["file"]})
        legacy = cache / CLIP_LEGACY_FILES[requested]
        if legacy.is_file():
            inventory.append({"file": legacy.name, "bytes": legacy.stat().st_size,
                              "sha256": _digest(legacy)})
        return cache, True, sorted(inventory, key=lambda item: item["file"])
    token = hashlib.sha256(requested.encode("utf-8")).hexdigest()[:16]
    return cache / ".unresolved" / token, False, []


def configuration(*, clip_model=None, vlm_model=None):
    """Freeze deployment inputs without importing torch or loading any model."""
    home = Path.home() / ".cache" / "traceannotate"
    clip_cache = Path(os.environ.get("TRACE_ANNOTATE_CLIP_CACHE", str(home / "clip"))).absolute()
    vlm_cache = Path(os.environ.get("TRACE_ANNOTATE_VLM_CACHE", str(home / "vlm"))).absolute()
    requested_vlm = vlm_model or "microsoft/Florence-2-base"
    local_vlm = Path(requested_vlm).absolute() if Path(requested_vlm).is_dir() else None
    requested_clip = clip_model or "ViT-B-32"
    clip_inventory_root, clip_selection_resolved, clip_inventory = _clip_selection(
        clip_cache, requested_clip)
    # Only the requested model repository belongs to this VLM deployment.
    vlm_inventory_root = local_vlm or vlm_cache / ("models--" + requested_vlm.replace("/", "--"))
    versions = {}
    for distribution in ("torch", "open_clip_torch", "transformers", "tokenizers"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    package = Path(__file__).resolve().parents[1]
    return {"contract": CONTRACT, "policy": R.policy(), "python": str(Path(sys.executable).absolute()),
            "python_sha256": _digest(sys.executable), "versions": versions,
            "implementation": _implementation_identity(package),
            "allow_download": os.environ.get("TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD", "").strip() == "1",
            "clip": {"requested": requested_clip, "cache": str(clip_cache),
                     "inventory_root": str(clip_inventory_root),
                     "selection_resolved": clip_selection_resolved,
                     "inventory": clip_inventory},
            "vlm": {"requested": str(local_vlm) if local_vlm else vlm_model or "microsoft/Florence-2-base",
                    "revision": os.environ.get("TRACE_ANNOTATE_VLM_REVISION") or None,
                    "cache": str(vlm_cache), "inventory_root": str(vlm_inventory_root),
                    "inventory": _inventory(vlm_inventory_root)}}


def validate_configuration(value):
    expected = {"contract", "policy", "python", "python_sha256", "versions", "implementation",
                "allow_download", "clip", "vlm"}
    if not isinstance(value, dict) or set(value) != expected or value["contract"] != CONTRACT:
        raise WorkspaceError("SCHEMA_INVALID", "Invalid model process configuration")
    value = copy.deepcopy(value)
    bounds = {"model_bytes": (2 * R.GIB, 64 * R.GIB), "reserve_bytes": (R.GIB, 64 * R.GIB),
              "wait_seconds": (0, 600), "frame_seconds": (5, 600), "phase_seconds": (60, 86400)}
    resource = value["policy"]
    legacy_resource_names = {"contract", *bounds}
    if isinstance(resource, dict) and set(resource) == legacy_resource_names:
        resource["host_lock"] = R.policy()["host_lock"]
    lock = resource.get("host_lock") if isinstance(resource, dict) else None
    if (not isinstance(resource, dict) or set(resource) != {"contract", "host_lock", *bounds}
            or resource.get("contract") != R.CONTRACT
            or not isinstance(lock, dict)
            or set(lock) != {"contract", "kind", "identity"}
            or lock.get("contract") != R.HOST_SLOT_CONTRACT
            or lock.get("kind") not in {"file", "windows_named_mutex", "linux_abstract_socket"}
            or not isinstance(lock.get("identity"), str) or not lock["identity"]
            or lock["kind"] == "file" and (
                not Path(lock["identity"]).is_absolute()
                or Path(lock["identity"]).name != R.HOST_SLOT_FILENAME)
            or lock["kind"] == "windows_named_mutex"
            and not lock["identity"].startswith("Local\\TraceAnnotateHostModelSlot-")
            or lock["kind"] == "linux_abstract_socket"
            and not lock["identity"].startswith("traceannotate-host-model-slot-")
            or any(type(resource[name]) not in (int, float) or not math.isfinite(resource[name])
                   or not low <= resource[name] <= high for name, (low, high) in bounds.items())):
        raise WorkspaceError("SCHEMA_INVALID", "Invalid model resource policy")
    def digest(item):
        return isinstance(item, str) and len(item) == 64 and all(c in "0123456789abcdef" for c in item)
    if (type(value["allow_download"]) is not bool or not isinstance(value["python"], str)
            or not Path(value["python"]).is_absolute() or not digest(value["python_sha256"])
            or not isinstance(value["versions"], dict) or not isinstance(value["implementation"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value["versions"].items())
            or any(not isinstance(k, str) or not digest(v) for k, v in value["implementation"].items())):
        raise WorkspaceError("SCHEMA_INVALID", "Invalid model runtime identity")
    for module in ("clip", "vlm"):
        settings = value[module]
        current_names = ({"requested", "cache", "inventory", "inventory_root", "selection_resolved"}
                         if module == "clip" else
                         {"requested", "cache", "inventory", "revision", "inventory_root"})
        legacy_names = ({"requested", "cache", "inventory"} if module == "clip" else set())
        if (not isinstance(settings, dict) or frozenset(settings) not in {frozenset(current_names),
                                                                         frozenset(legacy_names)}
                or not isinstance(settings.get("requested"), str) or not settings["requested"]
                or not isinstance(settings.get("cache"), str) or not isinstance(settings.get("inventory"), list)
                or not Path(settings["cache"]).is_absolute() or len(settings["inventory"]) > 2048):
            raise WorkspaceError("SCHEMA_INVALID", "Invalid frozen model selection")
        if ("inventory_root" in settings and
                (not isinstance(settings["inventory_root"], str)
                 or not Path(settings["inventory_root"]).is_absolute()
                 or module == "clip" and type(settings["selection_resolved"]) is not bool)
                or module == "vlm" and settings["revision"] is not None
                and not isinstance(settings["revision"], str)):
            raise WorkspaceError("SCHEMA_INVALID", "Invalid frozen model revision")
        seen = set()
        for item in settings["inventory"]:
            if (not isinstance(item, dict) or set(item) != {"file", "bytes", "sha256"}
                    or not isinstance(item["file"], str) or not item["file"]
                    or "\\" in item["file"] or Path(item["file"]).is_absolute()
                    or any(part in {"", ".", ".."} for part in item["file"].split("/"))
                    or item["file"] in seen or type(item["bytes"]) is not int
                    or not 0 <= item["bytes"] <= 8 * R.GIB or not digest(item["sha256"])):
                raise WorkspaceError("SCHEMA_INVALID", "Invalid frozen checkpoint inventory")
            seen.add(item["file"])
        if module == "clip" and set(settings) == legacy_names:
            # Legacy selections did not bind one unambiguous deployment root.
            # Keep them readable and diagnostically not_run without inferring
            # a runnable checkpoint from current filesystem contents.
            settings["inventory_root"] = settings["cache"]
            settings["selection_resolved"] = False
    return value


def _unavailable(module, config):
    config = validate_configuration(config)
    settings = config[module]
    dependencies = ("torch", "open_clip") if module == "clip" else ("torch", "transformers")
    if any(util.find_spec(name) is None for name in dependencies):
        return "model_runtime_unavailable"
    # Downloads belong to a separate acquisition/import step.  A queued
    # inference job must never change the frozen cache it is validating.
    if config["allow_download"]:
        return "model_acquisition_required"
    if module == "clip" and not settings["selection_resolved"]:
        return "model_selection_unresolved"
    if not any(
            item["file"].endswith((".safetensors", ".bin", ".pt", ".pth"))
            for item in settings["inventory"]):
        return "model_weights_unavailable"
    return None


def _empty_identity(module, config):
    settings = config[module]
    return {"model_name": settings["requested"], "model_path": settings["requested"],
            "revision": settings.get("revision") or "unpinned", "weights_path": None,
            "weights_sha256": None, "selected_revision": None, "weights_identity_status": "unknown",
            "offline": not config["allow_download"], "downloads_allowed": config["allow_download"],
            "task_prompt": "<MORE_DETAILED_CAPTION>" if module == "vlm" else None}


def requested_identity(module, config):
    """Describe the frozen request without claiming that any weights loaded.

    A parent process uses this only after an owned worker failed before it could
    return its loaded identity.  The resulting record preserves the requested
    deployment while keeping the checkpoint digest unknown.
    """
    config = validate_configuration(config)
    if module not in {"clip", "vlm"}:
        raise WorkspaceError("SCHEMA_INVALID", "Unknown isolated model module")
    return _empty_identity(module, config)


def _validate_identity(module, config, identity, available):
    config = validate_configuration(config)
    if module == "clip" and not config["clip"]["selection_resolved"] and available:
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Unresolved legacy model selection cannot report a loaded model")
    path, digest = identity.get("weights_path"), identity.get("weights_sha256")
    if not path:
        if available or digest is not None:
            raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Loaded model has no verifiable checkpoint identity")
        return
    root = Path(config[module]["inventory_root"])
    try:
        relative = Path(path).absolute().relative_to(root).as_posix()
    except ValueError as exc:
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model checkpoint is outside the frozen deployment") from exc
    expected = next((item for item in config[module]["inventory"] if item["file"] == relative), None)
    if expected is None or expected["sha256"] != digest:
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model checkpoint differs from its frozen deployment")
    selected = identity.get("selected_revision")
    if selected is not None and (Path(path).parent.parent.name != "snapshots" or Path(path).parent.name != selected):
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model selected revision differs from its checkpoint path")
    if module == "vlm":
        for name, actual in identity.get("model_code_sha256", {}).items():
            code_relative = (Path(path).parent / name).relative_to(root).as_posix()
            code = next((item for item in config[module]["inventory"] if item["file"] == code_relative), None)
            if code is None or code["sha256"] != actual:
                raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Executed VLM code differs from its frozen deployment")


def run_phase(module, config, images, work, *, cancelled):
    """Return small results and identity after the owned process has exited."""
    config = validate_configuration(config)
    if module not in {"clip", "vlm"} or not 0 < len(images) <= MAX_FRAMES:
        raise WorkspaceError("SCHEMA_INVALID", "Invalid isolated model phase")
    reason = _unavailable(module, config)
    if reason:
        return {"identity": _empty_identity(module, config), "available": False,
                "results": [{"frame_index": item["frame_index"], "path": item["path"],
                             "result": {"status": "not_run", "reason": reason}}
                            for item in images], "execution": {"status": "not_run", "reason": reason}}
    directory = work / "models" / module
    directory.mkdir(parents=True, exist_ok=False)
    request_path, result_path = directory / "request.json", directory / "result.json"
    request = {"contract": CONTRACT, "module": module, "config": config, "images": images}
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    bootstrap = ("import os,sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
                 "from trace_annotate.workspace.resource_budget import join_owned_job;"
                 "join_owned_job(os.environ.get('TRACE_ANNOTATE_OWNED_MODEL_JOB',''));"
                 "runpy.run_module('trace_annotate.workspace.model_worker',run_name='__main__')")
    command = [config["python"], "-X", "utf8", "-I", "-B", "-c", bootstrap, str(Path(__file__).resolve().parents[2]),
               "--request", str(request_path), "--output", str(result_path)]
    timeout = min(config["policy"]["phase_seconds"], 120 + len(images) * config["policy"]["frame_seconds"])
    execution = R.run_owned(command, directory / "execution", settings=config["policy"],
                            cancelled=cancelled, timeout=timeout, progress_path=directory / "progress.json")
    if not result_path.is_file() or not 0 < result_path.stat().st_size <= MAX_OUTPUT_BYTES:
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model result is missing or exceeds its byte limit")
    try:
        value = json.loads(result_path.read_text("utf-8"))
    except (ValueError, OSError) as exc:
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model result cannot be decoded") from exc
    if (not isinstance(value, dict) or value.get("contract") != CONTRACT or value.get("module") != module
            or type(value.get("available")) is not bool or not isinstance(value.get("identity"), dict)
            or not isinstance(value.get("results"), list) or len(value["results"]) != len(images)):
        raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model result identity is inconsistent")
    _validate_identity(module, config, value["identity"], value.get("available") is True)
    for actual, expected in zip(value["results"], images):
        if (not isinstance(actual, dict) or not isinstance(actual.get("result"), dict)
                or actual.get("frame_index") != expected["frame_index"] or actual.get("path") != expected["path"]
                or actual.get("input_sha256") != expected["sha256"]
                or actual["result"].get("status") not in {"success", "failed", "not_run"}
                or not value["available"] and actual["result"].get("status") == "success"):
            raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model frame result is inconsistent")
    value["execution"] = execution
    return value


def _peak_memory():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged",
                "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        value = Counters()
        value.size = ctypes.sizeof(value)
        kernel = R._windows_api()
        function = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if function(kernel.GetCurrentProcess(), ctypes.byref(value), value.size):
            return int(value.peak_working_set)
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    return None


def execute(request, output):
    """Child-only model construction; no model object crosses this boundary."""
    config, module = validate_configuration(request["config"]), request["module"]
    images = request["images"]
    if request.get("contract") != CONTRACT or module not in {"clip", "vlm"} or not 0 < len(images) <= MAX_FRAMES:
        raise ValueError("Invalid model worker request")
    if config["allow_download"]:
        raise ValueError("Workspace inference cannot acquire or modify model resources")
    os.environ["TRACE_ANNOTATE_CLIP_CACHE"] = config["clip"]["cache"]
    os.environ["TRACE_ANNOTATE_VLM_CACHE"] = config["vlm"]["cache"]
    os.environ["TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if config["vlm"]["revision"]:
        os.environ["TRACE_ANNOTATE_VLM_REVISION"] = config["vlm"]["revision"]
    before = configuration(clip_model=config["clip"]["requested"], vlm_model=config["vlm"]["requested"])
    if before != config:
        raise ValueError("Frozen model deployment changed before loading")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        cv2.setNumThreads(1)
    threads = {"torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads(),
               "opencv": cv2.getNumThreads() if cv2 is not None else "unavailable",
               "other_native_libraries": "not_exposed_by_adapter"}
    if module == "clip":
        from ..extractors.clip_engine import CLIPExtractor
        adapter = CLIPExtractor(model_name_or_path=config["clip"]["requested"])
    else:
        from ..extractors.vlm import VLMExtractor
        adapter = VLMExtractor(model_path=config["vlm"]["requested"], revision=config["vlm"]["revision"])
    identity = _check_loaded_adapter(adapter)
    from ..core.hasher import safe_load_image
    results = []
    progress_retries = 0
    for image in images:
        if _digest(image["path"]) != image["sha256"]:
            raise ValueError("Decoded frame identity changed")
        frame = safe_load_image(image["path"])
        started = time.monotonic()
        progress_retries += _publish_progress(output, "frame", image["frame_index"])
        result = adapter.classify(frame) if module == "clip" else adapter.caption(frame)
        _reject_memory_failure(result.get("error"))
        results.append({"frame_index": image["frame_index"], "path": image["path"],
                        "input_sha256": image["sha256"], "result": result,
                        "elapsed_seconds": time.monotonic() - started})
        if time.monotonic() - started > config["policy"]["frame_seconds"]:
            raise TimeoutError("Model frame exceeded its execution budget")
    progress_retries += _publish_progress(output, "finalizing")
    if configuration(clip_model=config["clip"]["requested"], vlm_model=config["vlm"]["requested"]) != config:
        raise ValueError("Frozen model deployment changed during inference")
    value = {"contract": CONTRACT, "module": module, "identity": identity,
             "available": adapter.model is not None, "results": results,
             "thread_policy": threads,
             "progress_publish_retries": progress_retries,
             "actual_pid": os.getpid(), "actual_parent_pid": os.getppid(), "peak_working_set_bytes": _peak_memory()}
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("Model result exceeds its output budget")
    with output.open("xb") as destination:
        destination.write(raw)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.request.stat().st_size > 2 * 1024 * 1024:
        raise SystemExit("Model request exceeds its byte budget")
    try:
        execute(json.loads(args.request.read_text("utf-8")), args.output)
    except MemoryError:
        import traceback
        traceback.print_exc()
        raise SystemExit(75)
    except TimeoutError:
        import traceback
        traceback.print_exc()
        raise SystemExit(76)
    except ModelLoadFailed:
        import traceback
        traceback.print_exc()
        raise SystemExit(77)
    except ModelProgressFailed:
        import traceback
        traceback.print_exc()
        raise SystemExit(78)
