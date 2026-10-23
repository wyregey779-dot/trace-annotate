"""CLI entry points for general video segments; embodied commands stay separate."""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import webbrowser

from .formats import export_csv, export_subtitles, import_subtitles, load_document
from .media import bind_document, identify_media, stream_sha256


def _positive_duration(raw: str) -> float:
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("duration must be a finite positive number") from None


def _port(raw: str) -> int:
    try:
        value = int(raw)
        if not 0 <= value <= 65535:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer between 0 and 65535") from None


def _serve_project(directory: str, *, port: int = 0, open_browser: bool = True) -> None:
    from .project_server import make_project_server
    server = make_project_server(directory, port=port)
    try:
        print(server.editor_url, flush=True)
        print("Local project is open. Keep this command running; press Ctrl+C to close. Export JSON in the editor to save edits.", file=sys.stderr, flush=True)
        if open_browser:
            webbrowser.open(server.editor_url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        server.server_close()


def _write_new(path: str, content: str) -> None:
    """Publish a complete UTF-8 file without replacing any existing entry.

    The temporary file is created beside the requested output so a hard link
    can publish it atomically. Filesystems without hard-link support fail
    closed; copying or replacing the destination would break this guarantee.
    File data is synced before publication. This does not promise directory
    entry durability across a power failure on every supported filesystem.
    """
    encoded = content.encode("utf-8")
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "Output already exists", str(target))

    temporary = None
    identity = None
    failed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name[:48]}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            created = os.fstat(stream.fileno())
            identity = (created.st_dev, created.st_ino)
            remaining = memoryview(encoded)
            while remaining:
                written = stream.write(remaining)
                if written is None or written <= 0 or written > len(remaining):
                    raise OSError(errno.EIO, "Output write did not make valid progress", str(target))
                remaining = remaining[written:]
            stream.flush()
            os.fsync(stream.fileno())

        current = os.lstat(temporary)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise OSError(errno.EIO, "Temporary output changed before publication", str(target))
        try:
            os.link(temporary, target)
        except (AttributeError, NotImplementedError) as exc:
            raise OSError(errno.ENOTSUP, "Filesystem cannot safely publish a new output file", str(target)) from exc
    except BaseException:
        failed = True
        raise
    finally:
        if temporary is not None:
            try:
                current = os.lstat(temporary)
                if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == identity:
                    os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as exc:
                if not failed:
                    raise OSError(exc.errno, "Output was written, but temporary cleanup failed", str(temporary)) from exc


def _run(args: argparse.Namespace) -> None:
    try:
        if args.command == "video-ui":
            if args.project:
                if args.print_path:
                    raise ValueError("--print-path is for the standalone editor; use --project with --no-browser to print and serve a project URL")
                _serve_project(args.project, port=args.port, open_browser=not args.no_browser)
                return
            ui = Path(__file__).resolve().parent / "ui" / "annotator.html"
            if not ui.is_file():
                raise ValueError("The general video editor is missing from this installation")
            if args.port:
                raise ValueError("--port requires --project")
            if args.print_path or args.no_browser:
                print(ui)
            elif not webbrowser.open(ui.as_uri()):
                print(f"Open this file in your browser: {ui}")
        elif args.command in {"video-prepare", "video-open"}:
            from .preparation import prepare_media
            print("Inspecting and preparing a compatibility copy; the original file is retained.", file=sys.stderr, flush=True)
            options = {"tone_map_hdr": True} if args.tone_map_hdr else {}
            manifest = prepare_media(args.file, args.output, **options)
            if args.command == "video-open":
                _serve_project(args.output, port=args.port, open_browser=not args.no_browser)
            else:
                print(json.dumps({"project": str(Path(args.output).resolve()), "source": manifest["source"],
                                  "video": manifest["video"], "timeline": manifest["timeline"]}, ensure_ascii=False, indent=2))
        elif args.command == "video-validate":
            doc = load_document(args.file)
            tracks = {segment["track"] for segment in doc["segments"]}
            print(f"Valid: {len(doc['segments'])} segments, {len(tracks)} tracks, {doc['duration']} seconds")
        elif args.command == "video-hash":
            print(stream_sha256(args.file))
        elif args.command == "video-bind":
            doc = load_document(args.file)
            identity = identify_media(args.media, args.duration)
            bound = bind_document(doc, identity, allow_rename=args.allow_rename)
            _write_new(args.output, json.dumps(bound, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            print("Duration was caller-declared; no media decoding or label verification was performed.", file=sys.stderr)
            print(f"Bound: SHA256 {identity.media_sha256}; written {args.output}")
        elif args.command == "video-export":
            doc = load_document(args.file)
            if args.format == "json":
                if args.track is not None:
                    raise ValueError("--track applies only to SRT/WebVTT; JSON always retains the full document")
                content = json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            elif args.format == "csv":
                if args.track is not None:
                    raise ValueError("--track applies only to SRT/WebVTT; CSV contains all segment rows")
                content = "\ufeff" + export_csv(doc)
            else:
                content = export_subtitles(doc, format=args.format, track=args.track)
            _write_new(args.output, content)
            if args.format != "json":
                print("Projection exported: retain JSON for full fidelity. CSV protects formula-like cells; subtitles omit IDs, tags, tracks and review state.", file=sys.stderr)
            print(f"Written: {args.output}")
        elif args.command == "video-import":
            content = Path(args.file).read_text(encoding="utf-8-sig")
            doc = import_subtitles(content, format=args.format, clip=args.clip,
                                   duration=args.duration, track=args.track)
            _write_new(args.output, json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            print(f"Imported: {len(doc['segments'])} subtitle cues as unreviewed segments; written {args.output}")
    except KeyboardInterrupt:
        print("Cancelled: no incomplete project was published.", file=sys.stderr)
        raise SystemExit(130) from None
    except RecursionError:
        print("Error: Temporal JSON nesting is too deep to process safely", file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def register_video_commands(subparsers) -> None:
    ui = subparsers.add_parser("video-ui", help="Open the local general-purpose video segment editor")
    ui.add_argument("--print-path", action="store_true", help="Print the packaged editor path without opening a browser")
    ui.add_argument("--project", help="Open an existing prepared project through a read-only local server")
    ui.add_argument("--no-browser", action="store_true", help="Print the standalone editor path or serve the project URL without opening a browser")
    ui.add_argument("--port", type=_port, default=0, help="Local project port; 0 selects a free port")
    ui.set_defaults(func=_run)

    for name in ("video-prepare", "video-open"):
        prepare = subparsers.add_parser(name, help="Prepare a verified compatibility copy" + (" and open its editor" if name == "video-open" else ""))
        prepare.add_argument("file", help="Original video; never rewritten")
        prepare.add_argument("-o", "--output", required=True, help="New project directory; existing entries are refused")
        prepare.add_argument("--tone-map-hdr", action="store_true", help="Explicitly convert HDR/PQ/HLG to an SDR BT.709 preview; original HDR pixels remain in the source")
        if name == "video-open":
            prepare.add_argument("--no-browser", action="store_true", help="Print the local URL without opening a browser")
            prepare.add_argument("--port", type=_port, default=0, help="Local port; 0 selects a free port")
        prepare.set_defaults(func=_run)

    validate = subparsers.add_parser("video-validate", help="Validate a general temporal annotation JSON document")
    validate.add_argument("file", help="General annotation JSON (not an embodied working/idle record)")
    validate.set_defaults(func=_run)

    hash_media = subparsers.add_parser("video-hash", help="Print a full-file SHA256 using bounded reads")
    hash_media.add_argument("file", help="Regular media file; no decoding is performed")
    hash_media.set_defaults(func=_run)

    bind = subparsers.add_parser("video-bind", help="Explicitly bind annotations to a full-file media hash")
    bind.add_argument("file", help="Temporal annotation JSON; v1 migration resets reviews")
    bind.add_argument("--media", required=True, help="Matching local regular media file")
    bind.add_argument("--duration", type=_positive_duration, required=True,
                      help="Caller-declared actual media duration in seconds; this command does not decode it")
    bind.add_argument("--allow-rename", action="store_true",
                      help="Allow a renamed v2 file only after its full hash matches; never permits a v1 rename")
    bind.add_argument("-o", "--output", required=True, help="New v2 JSON document; no overwrite")
    bind.set_defaults(func=_run)

    export = subparsers.add_parser("video-export", help="Export general segments to JSON, CSV, SRT or WebVTT")
    export.add_argument("file")
    export.add_argument("--format", choices=["json", "csv", "srt", "vtt"], required=True)
    export.add_argument("--track", help="Select one track for subtitles; required for a multi-track document")
    export.add_argument("-o", "--output", required=True, help="New output file; existing files are never overwritten")
    export.set_defaults(func=_run)

    imp = subparsers.add_parser("video-import", help="Import plain-text SRT/WebVTT cues as general segments")
    imp.add_argument("file")
    imp.add_argument("--format", choices=["srt", "vtt"], required=True)
    imp.add_argument("--clip", required=True, help="Matching video filename, without directory components")
    imp.add_argument("--duration", type=_positive_duration, required=True, help="Actual media duration in seconds")
    imp.add_argument("--track", default="subtitles")
    imp.add_argument("-o", "--output", required=True, help="New JSON document; no overwrite")
    imp.set_defaults(func=_run)
