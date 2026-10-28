"""Admit media only after the existing preparation pipeline verifies every frame.

The legacy preparation manifest is retained as evidence. Workspace timestamps
are reconstructed from integer PTS and explicit stream time bases, never its
display-only floating point seconds or a nominal frame rate.
"""
from __future__ import annotations

import errno
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Optional

from ..temporal.media import stream_sha256
from ..temporal.preparation import MediaPreparationError, prepare_media
from .errors import Conflict, DependencyUnavailable, NotFound, SchemaInvalid, StorageFull, WorkspaceError
from .schema import rational


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        raise ValueError(f"Missing or invalid {name}")
    return value


def _fraction(value: Any, name: str, *, positive: bool = False) -> Fraction:
    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid {name}")
    result = Fraction(value)
    if positive and result <= 0:
        raise ValueError(f"Nonpositive {name}")
    return result


def _rotation(stream: dict) -> int:
    rotations = [item["rotation"] for item in stream.get("side_data_list", [])
                 if "rotation" in item]
    if not rotations and "rotate" in stream.get("tags", {}):
        rotations = [stream["tags"]["rotate"]]
    values = {Fraction(str(value)) % 360 for value in rotations} or {Fraction(0)}
    if len(values) != 1 or not values.issubset({0, 90, 180, 270}):
        raise ValueError("A unique right-angle source rotation is required")
    return int(values.pop())


def _stream_endpoint(stream: dict) -> Fraction:
    """Use explicit muxed stream ticks, including audio that outlasts video."""
    base = _fraction(stream.get("time_base"), "playback stream time base", positive=True)
    start = _integer(stream.get("start_pts"), "playback stream start PTS")
    duration = _integer(stream.get("duration_ts"), "playback stream duration", positive=True)
    return (start + duration) * base


def _workspace_map(source: Path, destination: Path, expected_count: int,
                   source_base: Fraction, proxy_base: Fraction, origin: Fraction,
                   cancelled: Optional[Callable[[], bool]]) -> tuple:
    """Stream a precision-safe index and independently check its full pairing."""
    count = 0
    first_proxy = last_proxy = previous_source = None
    with source.open(encoding="utf-8") as original, destination.open("x", encoding="utf-8", newline="\n") as output:
        for line in original:
            if cancelled is not None and cancelled():
                raise InterruptedError("Media preparation cancelled")
            row = json.loads(line)
            index = _integer(row.get("frame"), "frame index")
            source_pts = _integer(row.get("source_pts"), "source PTS")
            proxy_pts = _integer(row.get("proxy_pts"), "proxy PTS")
            if index != count or (previous_source is not None and source_pts <= previous_source):
                raise ValueError("Frame indices and source PTS must increase without gaps")
            if last_proxy is not None and proxy_pts <= last_proxy:
                raise ValueError("Proxy PTS must increase without gaps or duplicates")
            source_time = source_pts * source_base - origin
            proxy_time = proxy_pts * proxy_base
            if proxy_time < 0 or abs(source_time - proxy_time) > min(source_base, proxy_base):
                raise ValueError("Frame correspondence differs from the shared timeline")
            output.write(json.dumps({
                "frame_index": str(index), "source_pts": str(source_pts),
                "proxy_pts": str(proxy_pts), "seconds": rational(proxy_time),
            }, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            if first_proxy is None:
                first_proxy = proxy_pts
            previous_source, last_proxy = source_pts, proxy_pts
            count += 1
    if count != expected_count or count == 0:
        raise ValueError("Frame map does not contain the verified decoded frame count")
    return count, first_proxy, last_proxy


def prepare_workspace_media(source: Path, output_dir: Path, *,
                            cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Return verified media fields and paths, without assigning workspace IDs.

    ``output_dir`` must not exist and its parent must exist. The caller binds
    ``media_fields`` to its media ID and immutable frame-map artifact, then
    commits admission in its own transaction. No database writes occur here.
    A failed workspace conversion may retain a completed legacy preparation
    directory as diagnostic evidence; its existence is never admission.

    ``frame_map_path`` contains JSONL rows with decimal-string frame_index,
    source_pts, proxy_pts and an exact rational ``seconds`` on the proxy clock.
    ``display_dimensions`` is descriptive metadata outside the media schema.
    """
    source, target = Path(source), Path(output_dir).absolute()
    if cancelled is not None and not callable(cancelled):
        raise SchemaInvalid("SCHEMA_INVALID", "cancelled must be a callable or None")
    try:
        if os.path.lexists(target):
            raise Conflict("OUTPUT_EXISTS", "Media preparation requires a new output directory")
        if not target.parent.is_dir():
            raise NotFound("NOT_FOUND", "Media preparation output parent does not exist")
        if not source.is_file():
            raise NotFound("NOT_FOUND", "Media source is not an existing regular file")
        missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing:
            raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "Media preparation tools are unavailable",
                                        details={"tools": missing})
        manifest = prepare_media(source, target, cancelled=cancelled)
        manifest_path = target / "project.json"
        preparation_log_path = target / "preparation-log.json"
        log = json.loads(preparation_log_path.read_text(encoding="utf-8"))
        timeline = manifest["timeline"]
        source_base = _fraction(timeline["source_time_base"], "source time base", positive=True)
        proxy_base = _fraction(timeline["proxy_time_base"], "proxy time base", positive=True)
        origin = _fraction(timeline["origin_time_fraction"], "common origin")
        expected_count = _integer(timeline["frame_count"], "decoded frame count", positive=True)
        source_stream = manifest["conversion"]["source_video_stream"]
        video = next(item for item in log["source_probe"]["streams"]
                     if item["index"] == source_stream and item["codec_type"] == "video")
        width = _integer(video.get("width"), "source width", positive=True)
        height = _integer(video.get("height"), "source height", positive=True)
        rotation = _rotation(video)
        # An unspecified aspect ratio is not proof of square source pixels.
        aspect = _fraction(video.get("sample_aspect_ratio", "").replace(":", "/"),
                           "source sample aspect ratio", positive=True)
        playback = target / "playback.mp4"
        legacy_mapping = target / "frame-map.jsonl"
        if stream_sha256(legacy_mapping, cancelled=cancelled) != log["frame_mapping_sha256"]:
            raise Conflict("MEDIA_CHANGED", "Prepared frame map changed before workspace admission")
        if (stream_sha256(playback, cancelled=cancelled) != manifest["video"]["sha256"]
                or playback.stat().st_size != manifest["video"]["size"]):
            raise Conflict("MEDIA_CHANGED", "Prepared playback changed before workspace admission")
        mapping = target / "workspace-frame-map.jsonl"
        count, first, last = _workspace_map(legacy_mapping, mapping, expected_count,
                                           source_base, proxy_base, origin, cancelled)
        tail = _integer(timeline["last_frame_duration"]["playback_ticks"],
                        "verified last frame duration", positive=True)
        video_start, video_end = first * proxy_base, (last + tail) * proxy_base
        playback_streams = [item for item in log["playback_probe"]["streams"]
                            if item.get("codec_type") in {"video", "audio"}]
        duration = max([video_end] + [_stream_endpoint(item) for item in playback_streams])
        playback_video = next(item for item in playback_streams if item["codec_type"] == "video")
        playback_width = _integer(playback_video.get("width"), "playback width", positive=True)
        playback_height = _integer(playback_video.get("height"), "playback height", positive=True)
        if not 0 <= video_start < video_end <= duration:
            raise ValueError("Invalid common media timeline")
        identity = manifest["source"]
        media_fields = {
            "source": {"sha256": identity["sha256"], "size_bytes": str(identity["size"]),
                       "display_name": identity["name"]},
            "playback": {"sha256": manifest["video"]["sha256"], "size_bytes": str(manifest["video"]["size"]),
                         "display_name": "playback.mp4"},
            "duration": rational(duration), "video_start": rational(video_start), "video_end": rational(video_end),
            "common_origin": rational(origin), "source_time_base": rational(source_base),
            "proxy_time_base": rational(proxy_base), "frame_count": str(count),
            "source_video_stream": source_stream,
            "coordinate_policy": "source_decoded_pixels_with_explicit_transform",
            "source_width": width, "source_height": height, "rotation_degrees": rotation,
            "pixel_aspect_ratio": rational(aspect),
        }
        display_width, display_height = width * aspect, Fraction(height)
        if rotation in (90, 270):
            display_width, display_height = display_height, display_width
        return {
            "source_sha256": identity["sha256"], "source_size": identity["size"],
            "playback_path": playback, "frame_map_path": mapping,
            "manifest_path": manifest_path, "preparation_log_path": preparation_log_path,
            "media_fields": media_fields,
            "frame_map_sha256": stream_sha256(mapping, cancelled=cancelled),
            "display_dimensions": {"source_width": width, "source_height": height,
                                   "playback_width": playback_width, "playback_height": playback_height,
                                   "display_width": rational(display_width), "display_height": rational(display_height),
                                   "rotation_degrees": rotation, "pixel_aspect_ratio": rational(aspect)},
        }
    except WorkspaceError:
        raise
    except InterruptedError as exc:
        raise WorkspaceError("CANCELLED", "Media preparation was cancelled") from exc
    except FileExistsError as exc:
        raise Conflict("OUTPUT_EXISTS", "Media preparation requires new output files") from exc
    except FileNotFoundError as exc:
        raise NotFound("NOT_FOUND", "A required media preparation file is missing") from exc
    except OSError as exc:
        if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
            raise StorageFull("STORAGE_FULL", "Media preparation has insufficient storage", retryable=True) from exc
        raise WorkspaceError("MEDIA_IO_ERROR", "Media preparation could not access its local files") from exc
    except (MediaPreparationError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError) as exc:
        # Raw tool diagnostics can contain source paths; retain the exception
        # chain for internal diagnosis, without putting it in an HTTP response.
        raise SchemaInvalid("UNSUPPORTED_MEDIA", "Media lacks a verified timeline or supported coordinate metadata",
                            details={"stage": "media_preparation"}) from exc
