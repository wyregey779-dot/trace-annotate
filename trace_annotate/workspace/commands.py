"""CLI entry for the workspace package: `trace-annotate workspace ...`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .errors import WorkspaceError
from .store import WorkspaceStore


def _open_store(project: Path) -> WorkspaceStore:
    store = WorkspaceStore(project)
    store.open()
    return store


def cmd_init(args: argparse.Namespace) -> int:
    store = WorkspaceStore(Path(args.project))
    store.create()
    print(json.dumps({"created": str(store.root)}, ensure_ascii=False))
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    store = _open_store(Path(args.project))
    try:
        results = []
        for f in args.files:
            r = store.import_source(Path(f))
            results.append(r)
        print(json.dumps({"imported": results}, ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


def cmd_list(args: argparse.Namespace) -> int:
    store = _open_store(Path(args.project))
    try:
        tasks = store.list_tasks()
        for t in tasks:
            media = store.get_media(t["media_id"])
            print(f'{t["task_id"]}  rev={t["revision"]}  {media["source"]["display_name"]}'
                  f'  duration={media["duration"]["n"]}/{media["duration"]["d"]}s')
        return 0
    finally:
        store.close()


def cmd_task(args: argparse.Namespace) -> int:
    store = _open_store(Path(args.project))
    try:
        task = store.get_task(args.task_id)
        print(json.dumps(task, ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


def cmd_export(args: argparse.Namespace) -> int:
    store = _open_store(Path(args.project))
    try:
        out = store.export_snapshot(args.name)
        print(json.dumps({"exported": str(out)}, ensure_ascii=False))
        return 0
    finally:
        store.close()


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve
    store = _open_store(Path(args.project))
    try:
        serve(store, host="127.0.0.1", port=args.port, open_browser=not args.no_browser)
    finally:
        store.close()
    return 0


def register_workspace_commands(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("workspace", help="Persistent multi-video batch annotation workspace")
    sp = p.add_subparsers(dest="ws_command", required=True)

    p_init = sp.add_parser("init", help="Create a new empty workspace project")
    p_init.add_argument("--project", required=True)
    p_init.set_defaults(func=cmd_init)

    p_imp = sp.add_parser("import", help="Import video files (hash + ffprobe + admit)")
    p_imp.add_argument("--project", required=True)
    p_imp.add_argument("files", nargs="+")
    p_imp.set_defaults(func=cmd_import)

    p_ls = sp.add_parser("list", help="List tasks")
    p_ls.add_argument("--project", required=True)
    p_ls.set_defaults(func=cmd_list)

    p_task = sp.add_parser("task", help="Print one task as JSON")
    p_task.add_argument("--project", required=True)
    p_task.add_argument("task_id")
    p_task.set_defaults(func=cmd_task)

    p_exp = sp.add_parser("export", help="Export immutable workspace snapshot")
    p_exp.add_argument("--project", required=True)
    p_exp.add_argument("--name")
    p_exp.set_defaults(func=cmd_export)

    p_srv = sp.add_parser("serve", help="Start the loopback editor server")
    p_srv.add_argument("--project", required=True)
    p_srv.add_argument("--port", type=int, default=0)
    p_srv.add_argument("--no-browser", action="store_true")
    p_srv.set_defaults(func=cmd_serve)
