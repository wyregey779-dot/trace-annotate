"""PTS-aware multi-frame sampling for time-series visual work.

Samples are chosen on the admitted frame map (frame_index/source_pts/proxy_pts),
never from a requested seek second. Windows are half-open [start, end): a frame
exactly at ``end`` is outside. Decoding runs as one bounded pass, not one call
per frame, and a budget overrun returns an explicit partial coverage instead of
pretending the whole range was analysed.
"""
from __future__ import annotations

import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..temporal.media import stream_sha256
from ..temporal import preparation as P
from . import schema as S
from .errors import DependencyUnavailable, NotFound, SchemaInvalid, WorkspaceError
from .frames import FrameService
from .store import _safe

MAX_PLAN_FRAMES = 2000
MAX_UNIFORM = 48
MAX_BOUNDARY_RADIUS = 8
DECODE_TIMEOUT = 180
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
MAX_PTS_LOG_BYTES = 2 * 1024 * 1024
MAX_EXACT_FRAMES = 64


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


def _frame_service(store: Any) -> FrameService:
    service = getattr(store, "_frame_service", None)
    if service is None:
        service = store._frame_service = FrameService(store)
    return service


def _frame_map_identity(service: FrameService, media_id: str) -> Tuple[str, int]:
    """Return immutable map identity without materialising every frame row."""
    page = service.page(media_id, 0, 1)
    total = int(page["total"])
    if total < 1 or not page["frames"]:
        raise NotFound("FRAME_NOT_FOUND", "The admitted media has no frame rows")
    return page["frame_map_sha256"], total


def _lookup_index(service: FrameService, media_id: str, seconds: Any, *,
                  digest: str, total: int) -> int:
    """Map a playback time to the left-closed insertion index."""
    result = service.lookup(media_id, seconds, "ceil")
    try:
        lookup_total = int(result["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkspaceError("RECOVERY_REQUIRED", "Frame lookup returned an invalid map identity") from exc
    if result.get("frame_map_sha256") != digest or lookup_total != total:
        raise WorkspaceError("RECOVERY_REQUIRED", "Frame map identity changed during sampling")
    value = result.get("frame_index")
    if value is None:
        return total
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkspaceError("RECOVERY_REQUIRED", "Frame lookup returned an invalid index") from exc
    if index < 0 or index > total:
        raise WorkspaceError("RECOVERY_REQUIRED", "Frame lookup escaped the admitted frame map")
    return index


def _selected_rows(service: FrameService, media_id: str, indices: Sequence[int], *,
                   digest: str, total: int) -> Dict[int, Dict[str, Any]]:
    """Read only selected rows, combining adjacent indices into bounded pages."""
    if not indices:
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    start = previous = indices[0]
    ranges: List[Tuple[int, int]] = []
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    for start, end in ranges:
        offset = start
        while offset < end:
            page = service.page(media_id, offset, min(1000, end - offset))
            if page["frame_map_sha256"] != digest or int(page["total"]) != total:
                raise WorkspaceError("RECOVERY_REQUIRED", "Frame map identity changed during sampling")
            rows = page["frames"]
            if len(rows) != min(1000, end - offset):
                raise WorkspaceError("RECOVERY_REQUIRED", "Frame map page is incomplete")
            for position, row in enumerate(rows, start=offset):
                if row.get("frame_index") != str(position):
                    raise WorkspaceError("RECOVERY_REQUIRED", "Frame map page is not index aligned")
                result[position] = row
            offset += len(rows)
    return result


def boundary_indices(store: Any, media_id: str, seconds: Sequence[Any], *,
                     budget: Optional[SamplingBudget] = None) -> List[int]:
    """Map scene-start seconds to admitted frame indices (left-closed)."""
    S.require_id(media_id)
    budget = budget or SamplingBudget()
    service = _frame_service(store)
    digest, total = _frame_map_identity(service, media_id)
    if len(seconds) > MAX_PLAN_FRAMES:
        raise SchemaInvalid(
            "INVALID_SEMANTIC_CLAIM",
            "Boundary lookup exceeds the planner bound; split the request",
        )
    result = []
    for value in seconds:
        index = _lookup_index(service, media_id, value, digest=digest, total=total)
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
    service = _frame_service(store)
    digest, total = _frame_map_identity(service, media_id)
    limits = _window(window)
    media = store.get_media(media_id)
    duration = media["duration"]
    if limits is None:
        lo, hi = 0, total
    else:
        start, end = limits
        if S.rational_value(end) > S.rational_value(duration):
            raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window end exceeds media duration")
        lo = _lookup_index(service, media_id, start, digest=digest, total=total)
        hi = _lookup_index(service, media_id, end, digest=digest, total=total)
    if hi <= lo:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "window contains no admitted frame")
    chosen: Dict[int, List[str]] = {}

    def add(index: int, role: str) -> None:
        # Boundary hints are absolute frame indices. A caller may provide
        # neighbours outside a requested window; they are context, not events
        # in this plan, and must not escape the half-open window.
        if lo <= index < hi:
            if index not in chosen:
                if len(chosen) >= MAX_PLAN_FRAMES:
                    raise SchemaInvalid(
                        "INVALID_SEMANTIC_CLAIM",
                        "Sampling candidates exceed the planner bound; narrow the window or boundary set",
                    )
                chosen[index] = []
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
    if not ordered:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Sampling request selected no frames in the window")
    truncated = len(ordered) > budget.max_frames
    selected = ordered[:budget.max_frames]
    rows = _selected_rows(service, media_id, selected, digest=digest, total=total)
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


def _select_expression(indices: List[int]) -> str:
    """Keep contiguous ranges compact and sparse expressions shallow."""
    if not indices or indices != sorted(set(indices)) or indices[0] < 0:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Sample indices must be unique ascending frame numbers")
    ranges = []
    first = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((first, previous))
            first = index
        previous = index
    ranges.append((first, previous))
    terms = [f"eq(n\\,{first})" if first == last else f"between(n\\,{first}\\,{last})"
             for first, last in ranges]
    # A long left-associated sum exceeds FFmpeg's expression parser depth.
    # Pairing terms gives logarithmic depth without changing selected frames.
    while len(terms) > 1:
        terms = ["(" + terms[index] + "+" + terms[index + 1] + ")"
                 if index + 1 < len(terms) else terms[index]
                 for index in range(0, len(terms), 2)]
    return "select=" + terms[0]


def plan_exact_indices(store: Any, media_id: str,
                       indices: Sequence[int]) -> Dict[str, Any]:
    """Make an untruncated, map-bound plan for at most 64 local action frames."""
    S.require_id(media_id)
    if (not isinstance(indices, (list, tuple)) or not indices
            or len(indices) > MAX_EXACT_FRAMES
            or any(type(index) is not int for index in indices)
            or list(indices) != sorted(set(indices)) or indices[0] < 0):
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Exact frame indices must be unique, sorted and bounded")
    service = _frame_service(store)
    digest, total = _frame_map_identity(service, media_id)
    if indices[-1] >= total:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "Exact frame index exceeds the admitted map")
    rows = _selected_rows(service, media_id, indices, digest=digest, total=total)
    samples = [{"frame_index": str(index), "source_pts": rows[index]["source_pts"],
                "proxy_pts": rows[index]["proxy_pts"], "seconds": rows[index]["seconds"],
                "roles": ["action_video_local"]} for index in indices]
    return {"media_id": media_id, "frame_map_sha256": digest,
            "total_frames": total, "samples": samples,
            "time_domain": "playback_seconds",
            "anchor_policy": "exact admitted frame indices for local action windows",
            "truncated": False, "planned": len(samples), "returned": len(samples)}


def _filter_script_option(ffmpeg: str, directory: Path,
                          control: Callable[[], bool]) -> str:
    """FFmpeg 9 removed filter_script; keep the documented 6+ contract."""
    receipt = P._run([ffmpeg, "-version"], directory, "decode-tool", control)
    with receipt.open("rb") as stream:
        version = stream.read(4096).decode("ascii", errors="replace")
    match = re.match(r"ffmpeg version (?:n)?(\d+)(?:\.|-|\s)", version)
    if match is None or int(match[1]) < 6:
        raise DependencyUnavailable("UNSUPPORTED_TOOLCHAIN",
                                    "Sample decoding requires an identifiable FFmpeg release 6 or newer")
    return "-/filter:v" if int(match[1]) >= 9 else "-filter_script:v"


def decode_samples(store: Any, media_id: str, plan: Dict[str, Any], output: Path, *,
                   timeout: int = DECODE_TIMEOUT, cancelled: Optional[Callable[[], bool]] = None,
                   max_bytes: Optional[int] = None,
                   ffmpeg: Optional[str] = None,
                   verify_output_pts: bool = False) -> Dict[str, Any]:
    """Decode absolute frame indices once, with owned-process and byte limits."""
    started = time.monotonic()
    limit = MAX_OUTPUT_BYTES if max_bytes is None else max_bytes
    if type(limit) is not int or limit < 1:
        raise SchemaInvalid("SCHEMA_INVALID", "Decoded sample byte budget must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise SchemaInvalid("SCHEMA_INVALID", "Sample decode timeout must be finite and positive")
    directory = None
    completed_bytes = 0
    current_output = 1
    monitoring = True
    pts_log = None

    def control() -> bool:
        nonlocal completed_bytes, current_output
        if cancelled is not None and cancelled():
            raise WorkspaceError("CANCELLED", "Sample decode was cancelled")
        if time.monotonic() - started >= timeout:
            raise WorkspaceError("FRAME_DECODE_TIMEOUT",
                                 "Sample decode exceeded its time budget")
        if monitoring and directory is not None:
            # Image2 writes frames in order. Once the next file exists, the
            # preceding file is complete. Include the active file's size too.
            while (directory / f"frame-{current_output + 1:04d}.png").exists():
                completed_bytes += (directory / f"frame-{current_output:04d}.png").stat().st_size
                current_output += 1
            active = directory / f"frame-{current_output:04d}.png"
            active_bytes = active.stat().st_size if active.exists() else 0
            if completed_bytes + active_bytes > limit:
                raise WorkspaceError("FRAME_TOO_LARGE", "Decoded samples exceed the byte budget")
            if pts_log is not None and pts_log.exists() and pts_log.stat().st_size > MAX_PTS_LOG_BYTES:
                raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder PTS evidence exceeds the byte budget")
        return False

    control()
    Path(output).mkdir(parents=True, exist_ok=True)
    _safe(Path(output))
    frames = plan["samples"]
    if not frames:
        raise SchemaInvalid("INVALID_SEMANTIC_CLAIM", "No frames were planned")
    if verify_output_pts:
        service = _frame_service(store)
        digest, total = _frame_map_identity(service, media_id)
        indices = [int(item["frame_index"]) for item in frames]
        if (plan.get("media_id") != media_id or plan.get("frame_map_sha256") != digest
                or plan.get("total_frames") != total or len(indices) > 64
                or indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= total):
            raise WorkspaceError("RECOVERY_REQUIRED", "Exact decode plan is not bound to the admitted frame map")
        rows = _selected_rows(service, media_id, indices, digest=digest, total=total)
        if (plan.get("time_domain") != "playback_seconds"
                or plan.get("truncated") is not False
                or plan.get("planned") != len(frames)
                or plan.get("returned") != len(frames)):
            raise WorkspaceError("RECOVERY_REQUIRED", "Exact decode plan is incomplete or has another time domain")
        for item, index in zip(frames, indices):
            if (set(item) != {"frame_index", "source_pts", "proxy_pts", "seconds", "roles"}
                    or not isinstance(item["roles"], list) or not item["roles"]
                    or len(item["roles"]) != len(set(item["roles"]))
                    or any(not isinstance(role, str) or not role for role in item["roles"])):
                raise WorkspaceError("RECOVERY_REQUIRED", "Exact decode plan sample is malformed")
            if any(item.get(field) != rows[index][field]
                   for field in ("source_pts", "proxy_pts", "seconds")):
                raise WorkspaceError("RECOVERY_REQUIRED", "Exact decode plan frame identity changed")
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "FFmpeg is needed to decode samples")
    from .frames import _binding
    media, path, playback_hash = _binding(store, media_id, "playback_path")
    indices = [int(item["frame_index"]) for item in frames]
    expression = _select_expression(indices)
    directory = Path(output) / ".decode-0000"
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "select.filter"
    with script.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(expression + (",format=rgb24,showinfo" if verify_output_pts else ""))
    try:
        script_option = _filter_script_option(ffmpeg, directory, control)
    except (P.MediaPreparationError, OSError) as exc:
        raise WorkspaceError("FRAME_DECODE_FAILED",
                             f"Sample decoder version probe failed: {str(exc)[-600:]}") from exc
    command = [ffmpeg, "-hide_banner", "-loglevel",
               "level+info" if verify_output_pts else "error",
               "-nostats", "-nostdin", "-n", "-xerror",
               "-protocol_whitelist", "file,pipe", "-err_detect", "explode",
               "-threads", "1", "-filter_threads", "1",
               "-i", str(path), "-map", "0:v:0", script_option, str(script),
               "-fps_mode", "passthrough", "-c:v", "png", "-pix_fmt", "rgb24",
               "-threads:v", "1", "-frames:v", str(len(indices)),
               "-f", "image2", str(directory / "frame-%04d.png")]
    pts_log = directory / "decode.stderr" if verify_output_pts else None
    try:
        if verify_output_pts:
            P._run(command, directory, "decode", control, strict=False)
        else:
            P._run(command, directory, "decode", control)
    except (P.MediaPreparationError, OSError) as exc:
        diagnostic = str(exc)[-600:]
        raise WorkspaceError("FRAME_DECODE_FAILED",
                             f"Sample decoder failed; ffmpeg={diagnostic}") from exc
    control()
    monitoring = False
    produced = sorted(directory.glob("frame-*.png"))
    expected = len(frames)
    if len(produced) != expected:
        raise WorkspaceError("FRAME_DECODE_FAILED",
                             f"Decoder produced {len(produced)} of {expected} planned frames")
    observed_pts = None
    observed_checksums = None
    if verify_output_pts:
        observed = _observed_output_pts(pts_log, expected)
        observed_pts = [item[0] for item in observed]
        observed_checksums = [item[1] for item in observed]
        proxy_time_base = S.rational_value(media["proxy_time_base"])
        planned_pts = [int(item["proxy_pts"]) * proxy_time_base for item in frames]
        if observed_pts != planned_pts:
            raise WorkspaceError("FRAME_DECODE_FAILED",
                                 "Decoder output PTS differ from the admitted frame map")
    for position, file in enumerate(produced, start=1):
        control()
        file.replace(Path(output) / f"sample-{position:04d}.png")
    # Keep the exact filter script and owned-process diagnostics with evidence.
    evidence: List[Dict[str, Any]] = []
    total_bytes = 0
    for item, position in zip(frames, range(1, expected + 1)):
        control()
        file = Path(output) / f"sample-{position:04d}.png"
        size = file.stat().st_size
        total_bytes += size
        if total_bytes > limit:
            raise WorkspaceError("FRAME_TOO_LARGE", "Decoded samples exceed the byte budget")
        with file.open("rb") as stream:
            header = stream.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder output is not PNG")
        if verify_output_pts:
            import zlib
            from PIL import Image
            try:
                with Image.open(file) as decoded_image:
                    if (decoded_image.mode != "RGB" or decoded_image.width * decoded_image.height
                            > 16_000_000):
                        raise ValueError("Unexpected decoded image geometry")
                    pixel_checksum = f"{zlib.adler32(decoded_image.tobytes(), 0) & 0xffffffff:08X}"
            except (OSError, ValueError) as exc:
                raise WorkspaceError("FRAME_DECODE_FAILED", "Decoded PNG pixels are invalid") from exc
            if pixel_checksum != observed_checksums[position - 1]:
                raise WorkspaceError("FRAME_DECODE_FAILED",
                                     "Decoded PNG pixels differ from the selected PTS frame")
        evidence.append({"frame_index": item["frame_index"], "source_pts": item["source_pts"],
                         "proxy_pts": item["proxy_pts"], "seconds": item["seconds"],
                         "roles": item["roles"], "path": str(file),
                         "sha256": stream_sha256(file, cancelled=control), "size_bytes": size})
    if stream_sha256(path, cancelled=control) != playback_hash:
        raise WorkspaceError("RECOVERY_REQUIRED", "Playback changed during sample decode")
    if verify_output_pts:
        final_digest, final_total = _frame_map_identity(_frame_service(store), media_id)
        if (final_digest, final_total) != (plan["frame_map_sha256"], plan["total_frames"]):
            raise WorkspaceError("RECOVERY_REQUIRED", "Frame map changed during exact sample decode")
    return {"media_id": media_id, "frame_map_sha256": plan["frame_map_sha256"],
            "playback_sha256": playback_hash, "decoded": evidence,
            "total_bytes": total_bytes, "command": command,
            "commands": [command],
            "count_verified": True, "time_domain": plan["time_domain"],
            **({"output_pts_verified": True,
                "output_pts": [S.rational(value) for value in observed_pts],
                "output_pixel_checksums": observed_checksums,
                "pts_log_sha256": stream_sha256(pts_log, cancelled=control)}
               if verify_output_pts else {})}


def _observed_output_pts(path: Path, expected: int) -> List[Tuple[Any, str]]:
    """Read bounded FFmpeg showinfo PTS and RGB checksums, rejecting diagnostics."""
    if path is None or not path.is_file() or path.stat().st_size > MAX_PTS_LOG_BYTES:
        raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder PTS evidence is missing or too large")
    return _parse_observed_output_pts(path.read_bytes(), expected)


def _parse_observed_output_pts(raw: bytes, expected: int) -> List[Tuple[Any, str]]:
    """Parse the exact persisted decoder log after its descriptor is verified."""
    if not isinstance(raw, bytes) or len(raw) > MAX_PTS_LOG_BYTES:
        raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder PTS evidence exceeds its byte budget")
    base = None
    ticks = []
    checksums = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.strip() and ("[info]" not in line or re.search(
                r"\b(?:invalid|error|failed|corrupt|fatal|warning)\b", line,
                re.IGNORECASE)):
            raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder reported an unexpected diagnostic")
        if "showinfo" not in line:
            continue
        match_base = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
        if match_base:
            numerator, denominator = (int(value) for value in match_base.groups())
            if numerator < 1 or denominator < 1 or base is not None:
                raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder PTS time base is ambiguous")
            base = S.rational_value({"n": str(numerator), "d": str(denominator)})
        match_frame = re.search(
            r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b.*\bfmt:rgb24\b.*\bchecksum:([0-9A-F]{8})\b", line)
        if match_frame:
            if int(match_frame[1]) != len(ticks) or len(ticks) >= expected:
                raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder PTS rows are not aligned")
            ticks.append(int(match_frame[2]))
            checksums.append(match_frame[3])
        elif re.search(r"\bn:\s*\d+\s+pts:", line):
            raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder RGB checksum is incomplete")
    if base is None or len(ticks) != expected or len(checksums) != expected:
        raise WorkspaceError("FRAME_DECODE_FAILED", "Decoder PTS evidence is incomplete")
    return [(tick * base, checksum) for tick, checksum in zip(ticks, checksums)]


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
