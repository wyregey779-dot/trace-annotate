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
    try:
        print(json.dumps({"created": str(store.root)}, ensure_ascii=False))
    finally:
        store.close()
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    store = _open_store(Path(args.project))
    try:
        results, failures = [], []
        for f in args.files:
            try:
                results.append(store.import_source(Path(f)))
            except WorkspaceError as error:
                failures.append({"display_name": Path(f).name, "error_code": error.code})
        print(json.dumps({"imported": results, "failed": failures}, ensure_ascii=False, indent=2))
        return 1 if failures else 0
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


def cmd_batch(args: argparse.Namespace) -> int:
    from .processors import OPERATION, VERSION, media_verifier
    from .queue import DurableQueue
    store = _open_store(Path(args.project))
    try:
        queue = DurableQueue(store)
        action = args.batch_command
        if action == "create":
            tasks = [store.get_task(tid) for tid in args.task_id]
            result = queue.create_batch(
                [{"task_id": task["id"], "expected_revision": task["revision"],
                  "expected_sha256": task["content_sha256"]} for task in tasks],
                operation=OPERATION, processor_version=VERSION, command_id=args.command_id)
        elif action == "run":
            results = queue.run_pending(media_verifier(store), batch_id=args.batch_id)
            result = {"executed": results, "batches": queue.list_batches()}
        elif action == "show":
            result = queue.get_batch(args.batch_id) if args.batch_id else {"batches": queue.list_batches()}
        else:
            result = queue.control_batch(args.batch_id, args.action, command_id=args.command_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if action == "run":
            batches = [queue.get_batch(args.batch_id)] if args.batch_id else queue.list_batches()
            return 0 if all(job["state"] == "succeeded" for batch in batches for job in batch["jobs"]) else 1
        return 0
    finally:
        store.close()


def cmd_intakes(args: argparse.Namespace) -> int:
    store = _open_store(Path(args.project))
    try:
        print(json.dumps({"intakes": store.list_intakes()}, ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


def run_workspace_command(args: argparse.Namespace) -> int:
    """Preserve nonzero partial/failure exits without exposing private tracebacks."""
    import sys
    try:
        return args.func(args)
    except WorkspaceError as error:
        print(json.dumps(error.body("req-cli"), ensure_ascii=False), file=sys.stderr)
        return 1


def register_workspace_commands(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("workspace", help="Persistent multi-video batch annotation workspace")
    sp = p.add_subparsers(dest="ws_command", required=True)

    p_init = sp.add_parser("init", help="Create a new empty workspace project")
    p_init.add_argument("--project", required=True)
    p_init.set_defaults(func=cmd_init)

    p_imp = sp.add_parser("import", help="Import videos after full decoding and exact frame mapping")
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

    p_intakes = sp.add_parser("intakes", help="Show all import attempts, including failures")
    p_intakes.add_argument("--project", required=True)
    p_intakes.set_defaults(func=cmd_intakes)

    p_batch = sp.add_parser("batch", help="Durable integrity-verification batches; no automatic semantic labels")
    batch_sub = p_batch.add_subparsers(dest="batch_command", required=True)
    for action in ("create", "run", "show", "control"):
        command = batch_sub.add_parser(action)
        command.add_argument("--project", required=True)
        command.set_defaults(func=cmd_batch)
        if action == "create":
            command.add_argument("--task-id", required=True, action="append")
            command.add_argument("--command-id", required=True)
        else:
            command.add_argument("--batch-id", required=action == "control")
        if action == "control":
            command.add_argument("--action", required=True, choices=("pause_dispatch", "resume_dispatch", "cancel", "retry_failed"))
            command.add_argument("--command-id", required=True)
