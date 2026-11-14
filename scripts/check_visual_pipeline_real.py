"""Run the real visual-assistance pipeline and its professional JSON round trip.

This check is intentionally opt-in.  It uses installed local providers without
monkeypatching, keeps private workspaces outside the source tree, and publishes
only a path/PID-free acceptance summary.  Model outputs remain unreviewed; an
unknown action or zero detections is an honest successful execution outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from trace_annotate.extractors.base import safe_failure_reason
from trace_annotate.workspace import professional_store as PStore
from trace_annotate.workspace import schema as S
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.store import WorkspaceStore


ABSOLUTE_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]"
    r"|(?<!:)(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+"
    r"|/Users/|/home/)"
)
PROJECTION_METHODS = frozenset(V.PORTABLE_PROJECTION_METHODS)
PUBLIC_COMMAND = [
    "python", "-B", "scripts/check_visual_pipeline_real.py",
    "--source", "<fixed-input>",
    "--hand-model", "<fixed-hand-model>",
    "--action-model", "<fixed-action-model-directory>",
    "--person-model", "<fixed-object-model>",
    "--work", "<new-private-work-directory>",
    "--output", "<public-evidence-report>",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if _contains_absolute_path(value):
        raise ValueError("public_report_contains_absolute_path")
    data = S.canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return ABSOLUTE_PATH.search(value) is not None
    if isinstance(value, dict):
        return any(_contains_absolute_path(key) or _contains_absolute_path(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _write_private(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    return {"name": path.name, "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _execution_summary(*, window_seconds: int, exit_code: int,
                       duration_seconds: float) -> dict[str, Any]:
    return {
        "command": PUBLIC_COMMAND + ["--window-seconds", str(window_seconds)],
        "exit_code": exit_code,
        "duration_seconds": round(duration_seconds, 6),
    }


def _model_run_summary(value: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value.get(key) for key in (
        "id", "provider", "model_name", "model_version", "weights_sha256",
        "license", "input_sha256", "output_sha256", "status",
        "failure_reason", "score_kind")}
    # A not-run local adapter can retain its requested model path as its
    # model_name.  That identity remains in the private professional export;
    # the public acceptance summary records only that it was a local path.
    for key, item in summary.items():
        if isinstance(item, str) and ABSOLUTE_PATH.search(item):
            summary[key] = "<redacted-local-path>"
    return summary


def _claims(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = {}
    for annotation in task.get("annotations") or []:
        semantic = (annotation.get("attributes") or {}).get("semantic") or {}
        for claim in semantic.get("claims") or []:
            method = ((claim.get("method") or {}).get("id"))
            if isinstance(method, str):
                values.setdefault(method, []).append(claim)
    return values


def _artifact_summary(document: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(({
        "id": item["id"], "sha256": item["sha256"],
        "size_bytes": item["size_bytes"],
    } for item in document.get("artifacts") or []), key=lambda item: item["id"])


def _projection_summary(task: dict[str, Any]) -> dict[str, Any]:
    by_method = _claims(task)
    projected = {method: len(by_method.get(method) or [])
                 for method in sorted(PROJECTION_METHODS)}
    evidence_ids = sorted({evidence
                           for method in PROJECTION_METHODS
                           for claim in by_method.get(method) or []
                           for evidence in claim.get("evidence_ids") or []})
    manifests = by_method.get("visual-candidate-projection-manifest-v1") or []
    return {
        "claim_counts": projected,
        "manifest_count": len(manifests),
        "projection_evidence_ids": evidence_ids,
        "projection_claims_sha256": S.sha256_text({
            method: by_method.get(method) or [] for method in sorted(PROJECTION_METHODS)}),
    }


def _assert_projection_evidence(task: dict[str, Any], document: dict[str, Any]) -> None:
    artifacts = {item["id"] for item in document.get("artifacts") or []}
    for annotation in task.get("annotations") or []:
        annotation_evidence = set(annotation.get("evidence_ids") or [])
        semantic = (annotation.get("attributes") or {}).get("semantic") or {}
        for claim in semantic.get("claims") or []:
            method = ((claim.get("method") or {}).get("id"))
            if method not in PROJECTION_METHODS:
                continue
            evidence = set(claim.get("evidence_ids") or [])
            if not evidence or not evidence <= annotation_evidence or not evidence <= artifacts:
                raise ValueError("portable_projection_evidence_binding_invalid")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    for path in (args.source, args.hand_model, args.action_model, args.person_model):
        if not path.is_absolute() or not path.exists():
            raise ValueError("required_local_resource_unavailable")
    if not args.work.is_absolute() or args.work.exists():
        raise ValueError("acceptance_work_directory_must_be_new_and_absolute")
    if not args.output.is_absolute():
        raise ValueError("acceptance_output_must_be_absolute")
    args.work.mkdir(parents=True, exist_ok=False)

    source_store = WorkspaceStore(args.work / "source-workspace")
    destination_store: WorkspaceStore | None = None
    jobs: V.VisionJobs | None = None
    source_store.create()
    try:
        admitted = source_store.import_source(args.source)
        task_id = admitted["task_id"]
        before = source_store.get_task(task_id)
        jobs = V.VisionJobs(
            source_store,
            clip_model_path=str(args.work / "unavailable-clip"),
            vlm_model_path=str(args.work / "unavailable-vlm"),
            person_model_path=str(args.person_model),
            hand_model_path=str(args.hand_model),
            action_video_model_path=str(args.action_model),
            slot_wait_seconds=900,
        )
        window = {"start": S.rational(0), "end": S.rational(args.window_seconds)}
        batch = jobs.submit(
            task_id, command_id="real-visual-submit",
            expected_revision=before["revision"],
            expected_sha256=before["content_sha256"], window=window)
        job_id = batch["jobs"][0]["id"]
        completed = jobs.run_pending(batch_id=batch["id"])
        if len(completed) != 1 or completed[0]["state"] != "succeeded":
            raise ValueError("real_visual_job_did_not_succeed")
        candidate = jobs.result(job_id)
        jobs._validate_candidate(candidate, jobs.queue.get_job(job_id))
        if source_store.get_task(task_id) != before:
            raise ValueError("candidate_job_edited_task_before_apply")
        jobs.apply(
            job_id, command_id="real-visual-apply",
            expected_revision=before["revision"],
            expected_sha256=before["content_sha256"], request_body=b"{}")
        applied = source_store.get_task(task_id)
        if not applied["annotations"] or applied["revision"] <= before["revision"]:
            raise ValueError("real_visual_candidate_was_not_applied")

        source_export = PStore.export_professional(source_store, task_id)["data"]
        source_document = json.loads(source_export)
        _assert_projection_evidence(applied, source_document)
        source_file = _write_private(args.work / "source-professional.json", source_export)

        destination_store = WorkspaceStore(args.work / "destination-workspace")
        destination_store.create()
        target_id = destination_store.import_source(args.source)["task_id"]
        target = destination_store.get_task(target_id)
        PStore.apply_import(
            destination_store, target_id, command_id="real-visual-import",
            expected_revision=target["revision"],
            expected_sha256=target["content_sha256"],
            document=source_document, request_body=b"{}")
        restored = destination_store.get_task(target_id)
        restored_export = PStore.export_professional(destination_store, target_id)["data"]
        restored_document = json.loads(restored_export)
        _assert_projection_evidence(restored, restored_document)
        restored_file = _write_private(
            args.work / "restored-professional.json", restored_export)

        source_projection = _projection_summary(applied)
        restored_projection = _projection_summary(restored)
        if source_projection != restored_projection:
            raise ValueError("portable_projection_roundtrip_mismatch")
        source_attributes = [item.get("attributes") for item in applied["annotations"]]
        restored_attributes = [item.get("attributes") for item in restored["annotations"]]
        if source_attributes != restored_attributes:
            raise ValueError("annotation_attributes_roundtrip_mismatch")
        source_runs = ((applied.get("extensions") or {}).get("professional") or {}).get(
            "model_runs") or []
        restored_runs = ((restored.get("extensions") or {}).get("professional") or {}).get(
            "model_runs") or []
        if source_runs != restored_runs:
            raise ValueError("model_runs_roundtrip_mismatch")

        run_by_name = {item["model_name"]: item for item in candidate["model_runs"]}
        expected_ids = {V._module_run_id(job_id, module)
                        for module in ("object", "object_relations", "hands", "action_video")}
        actual_ids = {item["id"] for item in candidate["model_runs"]}
        if not expected_ids <= actual_ids:
            raise ValueError("required_visual_model_run_missing")
        hand_frames = [frame for frame in candidate["keyframes"]
                       if (frame.get("hands") or {}).get("status") == "success"]
        hand_count = sum(len((frame.get("hands") or {}).get("hands") or [])
                         for frame in candidate["keyframes"])
        object_count = sum(len((frame.get("object") or {}).get("objects_detected") or [])
                           for frame in candidate["keyframes"])
        action_values = candidate.get("action_video_sampling") or {}
        action_results = action_values.get("windows") or []
        elapsed = time.monotonic() - started
        public = {
            "kind": "traceannotate.real-visual-pipeline-acceptance",
            "version": 1,
            "status": "accepted",
            "scope": (
                "one fixed real video window; local providers; unreviewed model candidates; "
                "explicit apply; professional JSON export/import/re-export"
            ),
            "non_claims": [
                "model_accuracy_or_f1", "human_review_or_ground_truth",
                "complete_action_recognition", "clean_machine_installation",
                "release_readiness",
            ],
            "input": {"name": args.source.name, "size_bytes": args.source.stat().st_size,
                      "sha256": _sha256(args.source),
                      "window_seconds": args.window_seconds},
            "job": {"id": job_id, "candidate_sha256": S.sha256_text(candidate),
                    "scene_count": len(candidate["scenes"]),
                    "keyframe_count": len(candidate["keyframes"]),
                    "object_observation_count": object_count,
                    "object_track_count": len(((candidate.get("object_sampling") or {})
                                               .get("tracking") or {}).get("tracks") or []),
                    "relation_candidate_count": len((candidate.get("object_relations") or {})
                                                     .get("relations") or []),
                    "successful_hand_frame_count": len(hand_frames),
                    "hand_observation_count": hand_count,
                    "action_window_count": len(action_results),
                    "action_window_statuses": [item.get("status") for item in action_results],
                    "action_window_decisions": [item.get("decision") for item in action_results]},
            "model_runs": [_model_run_summary(item) for item in candidate["model_runs"]],
            "portable_projection": source_projection,
            "roundtrip": {
                "attributes_equal": True, "model_runs_equal": True,
                "projection_equal": True,
                "source_artifacts": _artifact_summary(source_document),
                "restored_artifacts": _artifact_summary(restored_document),
                "source_export": source_file, "restored_export": restored_file,
            },
            "duration_seconds": round(elapsed, 6),
            "execution": _execution_summary(
                window_seconds=args.window_seconds, exit_code=0,
                duration_seconds=elapsed),
        }
        # Keep static analysis honest about a candidate index without exposing
        # provider-specific runtime objects or local paths.
        if not run_by_name:
            raise ValueError("visual_model_runs_are_empty")
        return public
    finally:
        if jobs is not None:
            jobs.close(timeout=60)
        if destination_store is not None:
            destination_store.close()
        source_store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--hand-model", required=True, type=Path)
    parser.add_argument("--action-model", required=True, type=Path)
    parser.add_argument("--person-model", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-seconds", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.window_seconds <= 60:
        parser.error("--window-seconds must be between 1 and 60")
    run_started = time.monotonic()
    try:
        report = _run(args)
    except BaseException as exc:
        report = {
            "kind": "traceannotate.real-visual-pipeline-acceptance",
            "version": 1, "status": "failed",
            "error_code": safe_failure_reason(
                exc, "real_visual_pipeline_acceptance_failed"),
            "execution": _execution_summary(
                window_seconds=args.window_seconds, exit_code=1,
                duration_seconds=time.monotonic() - run_started),
        }
        _write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 1
    _write_json(args.output, report)
    print(json.dumps({
        "status": report["status"], "job": report["job"],
        "portable_projection": report["portable_projection"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
