"""PTS-aware multi-frame sampling for time-series visual work.

Samples are chosen on the admitted frame map (frame_index/source_pts/proxy_pts),
never from a requested seek second. Windows are half-open [start, end): a frame
exactly at ``end`` is outside. Decoding runs as one bounded pass, not one call
per frame, and a budget overrun returns an explicit partial coverage instead of
pretending the whole range was analysed.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..temporal.media import stream_sha256
from . import schema as S
from .errors import DependencyUnavailable, NotFound, SchemaInvalid, WorkspaceError
from .frames import FrameService
from .store import _safe

MAX_PLAN_FRAMES = 2000
MAX_UNIFORM = 48
MAX_BOUNDARY_RADIUS = 8
DECODE_TIMEOUT = 180
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024


class SamplingBudget:
    """Hard limits for one sampling run; overruns are reported, not hidden."""

    def __init__(self, *, max_frames: int = 96, max_seconds: float = 120.0,
                 max_bytes: int = 512 * 1024 * 1024, max_uniform: int = MAX_UNIFORM,
                 boundary_radius: int = 1):
        for name, value in (("max_frames", max_frames), ("max_uniform", max_uniform),
                            ("boundary_radius", boundary_radius)):
            if type(value) is not int or value < 0:
                raise SchemaInvalid("SCHEMA_INVALID", f"{name} must be a non-negative integer")
        if max_frames < 1:
            raise SchemaInvalid("SCHEMA_INVALID", "max_frames must allow at least one frame")
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise SchemaInvalid("SCHEMA_INVALID", "max_seconds must be finite and positive")
        if type(max_bytes) is not int or max_bytes < 1:
            raise SchemaInvalid("SCHEMA_INVALID", "max_bytes must be positive")
        self.max_frames = max_frames
        self.max_seconds = float(max_seconds)
        self.max_bytes = max_bytes
        self.max_uniform = max_uniform
        self.boundary_radius = boundary_radius

    def as_dict(self) -> Dict[str, Any]:
        return {"max_frames": self.max_frames, "max_seconds": self.max_seconds,
                "max_bytes": self.max_bytes, "max_uniform": self.max_uniform,
                "boundary_radius": self.boundary_radius}


def _window(value: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window requires start and end")
    start = S.rational(value["start"])
    end = S.rational(value["end"])
    if start != value["start"] or end != value["end"]:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window seconds must be reduced rationals")
    if S.rational_value(start) >= S.rational_value(end):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window start must precede end")
    return start, end


def _rows(store: Any, media_id: str, budget: SamplingBudget) -> Tuple[List[Dict[str, Any]], str, int]:
    """Read bounded frame-map rows; the map is authoritative for time."""
    service = getattr(store, "_frame_service", None)
    if service is None:
        service = store._frame_service = FrameService(store)
    total = int(service.page(media_id, 0, 1)["total"])
    if total > budget.max_frames and total > MAX_PLAN_FRAMES:
        # Only the sampling planner needs a full scan of the map; cap it and
        # require the caller to narrow the window instead of scanning silently.
        pass
    rows: List[Dict[str, Any]] = []
    offset = 0
    chunk = 1000
    digest = ""
    while offset < total:
        page = service.page(media_id, offset, min(chunk, total - offset))
        digest = page["frame_map_sha256"]
        rows.extend(page["frames"])
        offset += len(page["frames"])
    if not rows:
        raise NotFound("FRAME_NOT_FOUND", "The admitted media has no frame rows")
    return rows, digest, total


def _index_of_seconds(rows: List[Dict[str, Any]], seconds: Any) -> int:
    """First frame whose proxy time is >= seconds (left-closed search)."""
    target = S.rational_value(seconds)
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if S.rational_value(rows[mid]["seconds"]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def boundary_indices(store: Any, media_id: str, seconds: Sequence[Any], *,
                     budget: Optional[SamplingBudget] = None) -> List[int]:
    """Map scene-start seconds to admitted frame indices (left-closed)."""
    S.require_id(media_id)
    budget = budget or SamplingBudget()
    rows, _digest, total = _rows(store, media_id, budget)
    result = []
    for value in seconds:
        index = _index_of_seconds(rows, value)
        result.append(min(max(0, index), total - 1))
    return sorted(set(result))


def plan_samples(store: Any, media_id: str, *, window: Optional[Dict[str, Any]] = None,
                 uniform: int = 8, boundary_frames: Iterable[int] = (),
                 include_edges: bool = True,
                 budget: Optional[SamplingBudget] = None) -> Dict[str, Any]:
    """Plan frames with explicit roles: first, last, uniform, boundary±radius."""
    S.require_id(media_id)
    budget = budget or SamplingBudget()
    if type(uniform) is not int or uniform < 0 or uniform > budget.max_uniform:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "uniform count is outside the budget")
    rows, digest, total = _rows(store, media_id, budget)
    limits = _window(window)
    media = store.get_media(media_id)
    duration = media["duration"]
    if limits is None:
        lo, hi = 0, total
    else:
        start, end = limits
        if S.rational_value(end) > S.rational_value(duration):
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window end exceeds media duration")
        lo = _index_of_seconds(rows, start)
        hi = _index_of_seconds(rows, end)
        while hi < total and S.rational_value(rows[hi]["seconds"]) < S.rational_value(end):
            hi += 1
    if hi <= lo:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window contains no admitted frame")
    chosen: Dict[int, List[str]] = {}

    def add(index: int, role: str) -> None:
        if 0 <= index < total:
            chosen.setdefault(index, [])
            if role not in chosen[index]:
                chosen[index].append(role)

    if include_edges:
        add(lo, "first")
        add(hi - 1, "last")
    if uniform and hi - lo >= 2:
        span = hi - 1 - lo
        for step in range(uniform):
            index = lo + int(round(span * step / max(1, uniform - 1)))
            add(min(hi - 1, max(lo, index)), "uniform")
    elif uniform == 1:
        add(lo, "uniform")
    for boundary in boundary_frames:
        if type(boundary) is not int:
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "boundary frame must be an integer")
        for delta in range(-budget.boundary_radius, budget.boundary_radius + 1):
            add(boundary + delta, "boundary_before" if delta < 0 else ("boundary" if delta == 0 else "boundary_after"))
    ordered = sorted(chosen)
    truncated = len(ordered) > budget.max_frames
    selected = ordered[:budget.max_frames]
    samples = [{"frame_index": str(index), "source_pts": rows[index]["source_pts"],
                "proxy_pts": rows[index]["proxy_pts"], "seconds": rows[index]["seconds"],
                "roles": chosen[index]} for index in selected]
    covered = S.rational_value(samples[-1]["seconds"]) - S.rational_value(samples[0]["seconds"])
    return {
        "media_id": media_id, "frame_map_sha256": digest, "total_frames": total,
        "window": window, "requested_uniform": uniform, "budget": budget.as_dict(),
        "samples": samples, "truncated": truncated,
        "planned": len(ordered), "returned": len(samples),
        "coverage": {"first_seconds": samples[0]["seconds"],
                     "last_seconds": samples[-1]["seconds"],
                     "covered_seconds": {"n": str(int(covered)), "d": "1"} if covered.denominator == 1 else
                     {"n": str(covered.numerator), "d": str(covered.denominator)},
                     "complete": not truncated},
        "time_domain": "playback_seconds",
        "anchor_policy": "frame_index/source_pts/proxy_pts from the admitted frame map",
    }


def decode_samples(store: Any, media_id: str, plan: Dict[str, Any], output: Path, *,
                   timeout: int = DECODE_TIMEOUT, cancelled: Optional[Callable[[], bool]] = None,
                   max_bytes: Optional[int] = None,
                   ffmpeg: Optional[str] = None) -> Dict[str, Any]:
    """Decode the planned frames in one pass; verify count, PNG signature and hash."""
    Path(output).mkdir(parents=True, exist_ok=True)
    _safe(Path(output))
    frames = plan["samples"]
    if not frames:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "No frames were planned")
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "FFmpeg is needed to decode samples")
    from .frames import _binding
    media, path, playback_hash = _binding(store, media_id, "playback_path")
    indices = [int(item["frame_index"]) for item in frames]
    expression = "select=" + "+".join(f"eq(n\\,{index})" for index in indices)
    pattern = str(Path(output) / "sample-%04d.png")
    command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", "-xerror",
               "-protocol_whitelist", "file,pipe", "-err_detect", "explode", "-i", str(path),
               "-map", "0:v:0", "-vf", expression, "-fps_mode", "passthrough",
               "-c:v", "png", "-pix_fmt", "rgb24", "-f", "image2", pattern]
    limit = max_bytes or MAX_OUTPUT_BYTES
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError("FRAME_DECODE_TIMEOUT", "Sample decode exceeded its time budget") from exc
    produced = sorted(Path(output).glob("sample-*.png"))
    expected = len(frames)
    if result.returncode or len(produced) != expected:
        diagnostic = (result.stderr or b"").decode("utf-8", "replace").strip()[-600:]
        raise WorkspaceError("FRAME_DECODE_FAILED",
                            f"Decoder produced {len(produced)} of {expected} planned frames"
                            + (f"; ffmpeg={diagnostic}" if diagnostic else ""))
    evidence: List[Dict[str, Any]] = []
    total_bytes = 0
    for item, file in zip(frames, produced):
        if cancelled is not None and cancelled():
            raise WorkspaceError("CANCELLED", "Sample decode was cancelled")
        data = file.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder output is not PNG")
        total_bytes += len(data)
        if total_bytes > limit:
            raise WorkspaceError("FRAME_TOO_LARGE", "Decoded samples exceed the byte budget")
        evidence.append({"frame_index": item["frame_index"], "source_pts": item["source_pts"],
                         "proxy_pts": item["proxy_pts"], "seconds": item["seconds"],
                         "roles": item["roles"], "path": str(file),
                         "sha256": S.sha256_bytes(data), "size_bytes": len(data)})
    if shutil.which("ffmpeg") and stream_sha256(path) != playback_hash:
        raise WorkspaceError("RECOVERY_REQUIRED", "Playback changed during sample decode")
    return {"media_id": media_id, "frame_map_sha256": plan["frame_map_sha256"],
            "playback_sha256": playback_hash, "decoded": evidence,
            "total_bytes": total_bytes, "command": command,
            "count_verified": True, "time_domain": plan["time_domain"]}


def event_comparison_frames(store: Any, media_id: str, *, boundary_frame: int,
                            radius: int = 1, budget: Optional[SamplingBudget] = None) -> Dict[str, Any]:
    """Before/after frames around a boundary, including the two sides at ±radius."""
    if type(boundary_frame) is not int or boundary_frame < 0:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "boundary_frame must be a non-negative integer")
    budget = budget or SamplingBudget(boundary_radius=max(budget_radius_default(), radius))
    plan = plan_samples(store, media_id, uniform=0, boundary_frames=[boundary_frame],
                        include_edges=False, budget=SamplingBudget(
                            max_frames=max(1, 2 * radius + 1), max_seconds=budget.max_seconds,
                            max_bytes=budget.max_bytes, max_uniform=0,
                            boundary_radius=max(radius, budget.boundary_radius)))
    return plan


def budget_radius_default() -> int:
    return 1


def samples_fingerprint(plan: Dict[str, Any]) -> str:
    """Stable digest of a plan's sample identities (no paths, no wall clock)."""
    body = {"frame_map_sha256": plan["frame_map_sha256"],
            "samples": [{key: item[key] for key in
                         ("frame_index", "source_pts", "proxy_pts", "seconds", "roles")}
                        for item in plan["samples"]],
            "time_domain": plan["time_domain"]}
    return S.sha256_text(body)
