"""Exit a real async admission at one of three barriers (P02).

Run as a subprocess so the crash is a real process exit: no exception handler,
no ``finally`` block and no interpreter shutdown hook can clean up after it.
The marker file proves which barrier was reached, after a real preparation.

Usage::

    python crash_admission.py <repo_root> <project_root> <point> <marker>

``point`` is one of ``before_tx``, ``in_tx``, ``after_tx``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXIT_CODES = {"before_tx": 70, "in_tx": 71, "after_tx": 72}


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    repo_root, project_root, point, marker = argv[1:5]
    if point not in EXIT_CODES:
        print(f"unknown crash point {point!r}", file=sys.stderr)
        return 2
    sys.path.insert(0, repo_root)
    from trace_annotate.workspace.intake import AsyncIntakes
    from trace_annotate.workspace.store import WorkspaceStore

    state = {"point": point, "prepared": False, "objects": 0}
    marker_path = Path(marker)

    def mark(**extra) -> None:
        payload = dict(state)
        payload.update(extra)
        marker_path.write_text(json.dumps(payload), encoding="utf-8")

    store = WorkspaceStore(Path(project_root))
    store.open()
    intakes = AsyncIntakes(store)

    if point == "before_tx":
        original_put = WorkspaceStore._put_file
        armed = {"value": False}

        def put(self, path):
            value = original_put(self, path)
            state["prepared"] = True
            state["objects"] += 1
            if state["objects"] >= 4:
                armed["value"] = True
            return value

        original_transaction = WorkspaceStore.transaction

        def transaction(self, *args, **kwargs):
            if armed["value"]:
                mark(crashed="before_tx")
                os._exit(EXIT_CODES["before_tx"])
            return original_transaction(self, *args, **kwargs)

        WorkspaceStore._put_file = put
        WorkspaceStore.transaction = transaction
    elif point == "in_tx":
        def task_digest(self, task):
            mark(crashed="in_tx")
            os._exit(EXIT_CODES["in_tx"])

        WorkspaceStore.task_digest = task_digest
    else:
        def finish(self, intake_id, generation, worker_id, result):
            mark(crashed="after_tx", task_id=result["task_id"], media_id=result["media_id"])
            os._exit(EXIT_CODES["after_tx"])

        AsyncIntakes.finish = finish

    results = intakes.run_pending(worker_id="crash-worker")
    # Reaching this line means the barrier never fired: report it loudly.
    mark(crashed=None, results=len(results))
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
