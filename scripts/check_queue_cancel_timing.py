"""Measure real cancellation visibility and cooperative stop latency.

One controlled job runs in a real DurableQueue on a real workspace. The test
waits until the processor is inside its loop, requests cancellation through the
same persisted API used by HTTP control, and measures:
- the time until the cancel command is durably visible;
- the time until the processor observes cancellation and exits.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from trace_annotate.workspace.queue import DurableQueue
from trace_annotate.workspace.store import WorkspaceStore


def write_json(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    if not args.ffmpeg:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing output directory")
    args.output.mkdir(parents=True)
    clip = args.output / "cancel.mp4"
    subprocess.run([args.ffmpeg, "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=blue:s=48x32:r=10:d=0.6", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", str(clip)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)

    store = WorkspaceStore(args.output / "workspace")
    store.create()
    try:
        store.import_source(clip)
        task = store.get_task(store.list_tasks()[0]["task_id"])
        queue = DurableQueue(store, lease_seconds=30)
        job = queue.enqueue(task["id"], operation="controlled-cancel", processor_version="test-v1",
                            parameters={"name": "cancel-latency"})
        entered = threading.Event()
        stopped = threading.Event()
        result = {}

        def processor(value, context):
            entered.set()
            while not context.cancelled():
                time.sleep(0.02)
            stopped.set()
            return {"cooperative_stop": True}

        worker = threading.Thread(target=lambda: result.setdefault(
            "job", queue.execute_once(processor, worker_id="timing-worker")))
        worker.start()
        if not entered.wait(10):
            raise AssertionError("Processor never entered")
        requested = time.monotonic()
        control = queue.cancel(job["id"])
        visible = time.monotonic()
        if control["state"] != "cancel_requested":
            raise AssertionError("Cancellation was not visible as cancel_requested")
        if not stopped.wait(10):
            raise AssertionError("Processor did not stop within 10 seconds after cancellation")
        stopped_at = time.monotonic()
        worker.join(10)
        if worker.is_alive():
            raise AssertionError("execute_once did not return after cooperative stop")
        final = result.get("job")
        if not final or final["state"] != "cancelled":
            raise AssertionError("Cooperative stop did not settle the job as cancelled")
        summary = {
            "cancel_visible_seconds": round(visible - requested, 3),
            "processor_stop_seconds": round(stopped_at - requested, 3),
            "cancel_state": control["state"],
            "final_state": final["state"],
            "attempts": queue.get_job(job["id"])["attempts"],
            "worker_result_state": final["state"],
            "checks": ["Cancel command is durably visible within 2 seconds",
                       "Processor observes cancellation and exits within 10 seconds",
                       "Cooperative stop settles the job as cancelled"],
            "ok": True,
        }
        if summary["cancel_visible_seconds"] > 2:
            raise AssertionError("Cancellation visibility exceeded 2 seconds")
        if summary["processor_stop_seconds"] > 10:
            raise AssertionError("Processor stop exceeded 10 seconds")
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("CANCEL TIMING ACCEPTANCE PASS")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
