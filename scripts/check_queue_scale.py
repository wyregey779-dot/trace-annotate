"""Scale and isolation check for the local intake queue.

Generates 20 small but genuinely different clips (frame rate, size, duration and
audio shape), admits them through two independent workspaces at the same time,
and adds one duplicate command plus one corrupted file to each queue's edge
cases. It also samples this process tree's resident memory, child count and CPU
while both drains run.

Only local synthetic media is created. An existing output directory is refused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import psutil

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from trace_annotate.workspace.intake import AsyncIntakes
from trace_annotate.workspace.store import WorkspaceStore

SPECS = [
    (5, "32x32", 0.25, False), (8, "48x32", 0.4, True),
    (10, "64x36", 0.6, False), (12, "80x44", 0.8, True),
    (15, "96x54", 1.0, False), (20, "160x90", 1.5, True),
    (24, "64x36", 2.0, False), (25, "80x44", 2.5, True),
    (30, "96x54", 3.0, False), (50, "160x90", 4.0, True),
    (5, "48x32", 0.5, True), (10, "80x44", 1.2, False),
    (12, "96x54", 0.7, True), (15, "160x90", 1.8, False),
    (20, "32x32", 2.2, True), (24, "64x36", 0.9, False),
    (25, "96x54", 1.1, True), (30, "80x44", 2.8, False),
    (50, "160x90", 0.6, True), (8, "32x32", 3.5, False),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_clip(ffmpeg, target, color, width, height, fps, seconds, audio):
    command = [ffmpeg, "-hide_banner", "-v", "error", "-y",
               "-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}:r={fps}:d={seconds}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency={300 + fps * 7}:sample_rate=44100:duration={seconds}",
                    "-c:a", "aac", "-b:a", "48k"]
    command += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", str(target)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90)
    return {"path": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
            "width": width, "height": height, "fps": fps, "seconds": seconds, "audio": audio}


def sample_tree(stop, samples):
    root = psutil.Process(os.getpid())
    for child in root.children(recursive=True):
        try:
            child.cpu_percent(None)
        except psutil.NoSuchProcess:
            pass
    while not stop.wait(0.05):
        try:
            members = [root] + root.children(recursive=True)
            rss = 0
            cpu = 0.0
            for member in members:
                try:
                    rss += member.memory_info().rss
                    cpu += member.cpu_percent(None)
                except psutil.NoSuchProcess:
                    continue
            samples.append({"rss_bytes": rss, "children": len(members) - 1, "cpu_percent": cpu})
        except psutil.NoSuchProcess:
            break


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

    clips = []
    colors = ["blue", "red", "green", "yellow", "magenta", "cyan"]
    for index, (fps, size, seconds, audio) in enumerate(SPECS):
        width, height = (int(value) for value in size.split("x"))
        clips.append(make_clip(args.ffmpeg, args.output / f"clip-{index + 1:02d}.mp4",
                               colors[index % len(colors)], width, height, fps, seconds, audio))
    corrupt = args.output / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a video\n")
    summary = {"clips": clips, "workspaces": {}, "page_errors": [],
               "corrupt": {"sha256": digest(corrupt), "bytes": corrupt.stat().st_size}}

    roots = {"A": args.output / "workspace-a", "B": args.output / "workspace-b"}
    stores = {}
    intakes = {}
    enqueued = {}
    for name, root in roots.items():
        store = WorkspaceStore(root)
        store.create()
        stores[name] = store
        intakes[name] = AsyncIntakes(store)
    duplicate_path = args.output / "clip-01-copy.mp4"
    shutil.copy2(args.output / "clip-01.mp4", duplicate_path)
    for index, clip in enumerate(clips):
        workspace = "A" if index < 10 else "B"
        command_id = "cmd-" + "a" * 32 if workspace == "A" and index == 0 else None
        row = intakes[workspace].enqueue(args.output / clip["path"], display_name=clip["path"],
                                         command_id=command_id)
        enqueued.setdefault(workspace, []).append((clip["path"], row))
    duplicate_original = enqueued["A"][0][1]
    duplicate = intakes["A"].enqueue(duplicate_path, display_name=enqueued["A"][0][0],
                                     command_id="cmd-" + "a" * 32)
    corrupt_row = intakes["A"].enqueue(corrupt, display_name="corrupt.mp4")

    start = time.monotonic()
    samples = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_tree, args=(stop, samples), daemon=True)
    monitor.start()
    errors = []

    def drain(name):
        try:
            intakes[name].run_pending(worker_id="scale-" + name)
        except Exception as exc:  # noqa: BLE001 - report, then fail the check
            errors.append(name + ": " + repr(exc))

    threads = [threading.Thread(target=drain, args=(name,)) for name in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start
    stop.set()
    monitor.join(timeout=5)
    if errors:
        raise AssertionError("; ".join(errors))

    peak = {"rss_bytes": max((item["rss_bytes"] for item in samples), default=0),
            "children": max((item["children"] for item in samples), default=0),
            "cpu_percent": max((item["cpu_percent"] for item in samples), default=0.0)}
    for name, store in stores.items():
        rows = store.list_intakes()
        tasks = store.list_tasks()
        medias = store.get_snapshot()["media"]
        summary["workspaces"][name] = {
            "intakes": len(rows), "tasks": len(tasks), "media": len(medias),
            "failed": [{"display_name": row["display_name"], "state": row["state"],
                        "error_code": row.get("error_code")} for row in rows
                       if row["state"] == "failed"],
            "succeeded": [{"display_name": row["display_name"], "state": row["state"],
                           "task_id": row.get("task_id")} for row in rows
                          if row["state"] == "succeeded"],
            "task_ids": [task["task_id"] for task in tasks],
        }
    summary["duplicate"] = {"requested_id": duplicate_original["id"], "returned_id": duplicate["id"],
                            "same_row": duplicate_original["id"] == duplicate["id"],
                            "state": duplicate.get("state")}
    final_corrupt = next(row for row in stores["A"].list_intakes() if row["id"] == corrupt_row["id"])
    final_duplicate = next(row for row in stores["A"].list_intakes() if row["id"] == duplicate["id"])
    summary["duplicate"]["state"] = final_duplicate["state"]
    summary["corrupt_intake"] = {"id": final_corrupt["id"], "state": final_corrupt["state"],
                                 "error_code": final_corrupt.get("error_code")}
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["peak_process_tree"] = peak
    summary["sample_count"] = len(samples)

    checks = []
    if len(clips) != 20:
        raise AssertionError("Expected 20 clips")
    checks.append("20 distinct synthetic clips were generated")
    for name in ("A", "B"):
        info = summary["workspaces"][name]
        if info["tasks"] != 10 or len(info["succeeded"]) != 10:
            raise AssertionError(name + " did not admit 10 valid clips")
        if name == "B" and info["failed"]:
            raise AssertionError("Workspace B has an unexpected failed intake")
        if name == "A" and not (len(info["failed"]) == 1
                                and info["failed"][0]["display_name"] == "corrupt.mp4"):
            raise AssertionError("Workspace A did not isolate exactly the corrupted input")
    checks.append("Two interleaved workspaces each admitted 10 clips")
    all_ids = summary["workspaces"]["A"]["task_ids"] + summary["workspaces"]["B"]["task_ids"]
    if len(set(all_ids)) != 20:
        raise AssertionError("Task ids were not isolated across workspaces")
    checks.append("Task ids and media did not cross workspaces")
    if not summary["duplicate"]["same_row"]:
        raise AssertionError("Duplicate command created a second row")
    if final_duplicate["state"] != "succeeded":
        raise AssertionError("Duplicate replay did not settle as succeeded")
    checks.append("Duplicate command replayed one intake row")
    if summary["corrupt_intake"]["state"] != "failed":
        raise AssertionError("Corrupted input was not isolated as a failed intake")
    if summary["workspaces"]["A"]["intakes"] != 11:
        raise AssertionError("Corrupt and duplicate did not stay in workspace A")
    checks.append("Corrupted input failed alone without blocking valid clips")
    if peak["rss_bytes"] <= 0:
        raise AssertionError("No process-tree memory samples were recorded")
    checks.append("Memory, child-count and CPU peaks were sampled during both drains")

    summary["checks"] = checks
    summary["ok"] = len(checks) == 6
    write_json(args.output / "summary.json", summary)
    for name, store in stores.items():
        store.close()
    print(json.dumps({"ok": summary["ok"], "checks": checks,
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "peak_process_tree": peak}, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit("Scale check did not complete")
    print("QUEUE SCALE ACCEPTANCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
