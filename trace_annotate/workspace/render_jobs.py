"""Snapshot-bound rendered-video candidates on the existing durable queue.

Only registered playback/map bytes and a trusted configured font are used.
Five fixed validated files are published and pinned in the queue finish
transaction. Candidate existence alone never authorizes a download.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import stat
import threading
import time
import uuid

from ..temporal import preparation as P
from ..temporal.media import stream_sha256
from . import render_layout as L, rendering as R, schema as S
from .analysis_jobs import AnalysisJobs
from .errors import Conflict, DependencyUnavailable, SchemaInvalid, WorkspaceError
from .http_media import resolve_media_object
from .professional_export import build_document
from .queue import MAX_RESULT_BYTES
from .slots import SlotClosed, SlotTimeout, get_prepare_slot
from .store import _safe

OPERATION = "render-annotations"
VERSION = "1"
KIND = "traceannotate.render-candidate"
FILES = {"annotation.mp4": "video", "render-plan.json": "layout", "verification.json": "verification",
         "audit.json": "audit", "project.json": "project"}
DEFAULT_FONT = Path(__file__).parent / "fonts/NotoSansCJKsc-Regular.otf"
MAX_JSON_BYTES = 16 * 1024 * 1024
DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024
# These layouts are readable only as already committed immutable candidates.
# They are never selected for a new request or executed by the current planner.
READ_ONLY_LAYOUT_VERSIONS = frozenset({"annotation-panel-v1"})


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SchemaInvalid("SCHEMA_INVALID", name + " must be finite positive seconds")
    return float(value)


def _binding(task):
    return {"task_id": task["id"], "revision": task["revision"], "content_sha256": task["content_sha256"]}


def _json_file(path, limit=MAX_JSON_BYTES):
    _safe(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise WorkspaceError("RECOVERY_REQUIRED", "Render JSON file exceeds its regular-file bound")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size:
        raise WorkspaceError("RECOVERY_REQUIRED", "Render JSON changed while being read")
    try:
        value = json.loads(raw)
        S.canonical_json(value)
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise WorkspaceError("RECOVERY_REQUIRED", "Render JSON is corrupt") from exc


class _Reader:
    def __init__(self, stream, control):
        self.stream, self.control = stream, control

    def read(self, size):
        if self.control():
            raise WorkspaceError("CANCELLED", "Render publication was cancelled")
        return self.stream.read(size)


class RenderJobs(AnalysisJobs):
    """Reuse the tested slot/drainer/close lifecycle, not its analysis processor."""

    def __init__(self, store, *, queue=None, closing=None, font_path=None, font_license=None,
                 deadline_seconds=1800, max_output_bytes=DOWNLOAD_MAX_BYTES,
                 poll_seconds=.1, slot_wait_seconds=900):
        if type(max_output_bytes) is not int or not 1024 * 1024 <= max_output_bytes <= DOWNLOAD_MAX_BYTES:
            raise SchemaInvalid("INVALID_RENDER_BUDGET", "Render output cap must be1MiB through512MiB")
        self.deadline_seconds = _positive(deadline_seconds, "deadline_seconds")
        self.max_output_bytes = max_output_bytes
        self.font_path = Path(font_path).absolute() if font_path is not None else DEFAULT_FONT.absolute()
        self._default_font = font_path is None
        self._font_license = font_license
        self._font_cache = None
        super().__init__(store, queue=queue, closing=closing, poll_seconds=poll_seconds,
                         slot_wait_seconds=slot_wait_seconds)
        self._recover_pins()

    def _font(self):
        _safe(self.font_path)
        if not self.font_path.is_file():
            raise DependencyUnavailable("RENDER_FONT_UNAVAILABLE", "Configured render font is unavailable")
        if self._default_font:
            config = _json_file(self.font_path.parent / "font.json", 16384)
            keys = {"id", "file", "sha256", "license", "license_file", "license_sha256", "source", "unmodified"}
            if (not isinstance(config, dict) or set(config) != keys or config["file"] != self.font_path.name
                    or config["license_file"] != "OFL.txt" or config["unmodified"] is not True):
                raise WorkspaceError("RECOVERY_REQUIRED", "Bundled font metadata is invalid")
            S.require_id(config["id"]); S.require_hash(config["sha256"]); S.require_hash(config["license_sha256"])
            license_path = self.font_path.parent / "OFL.txt"
            _safe(license_path)
            if not license_path.is_file() or license_path.stat().st_size > 65536 or stream_sha256(license_path) != config["license_sha256"]:
                raise WorkspaceError("RENDER_FONT_CHANGED", "Bundled font license identity changed")
            digest, license_text, ident = config["sha256"], config["license"], config["id"]
        else:
            digest = stream_sha256(self.font_path)
            license_text, ident = self._font_license, "configured-local-font"
        if not isinstance(license_text, str) or not license_text.strip() or len(license_text) > 1024:
            raise SchemaInvalid("RENDER_FONT_UNAVAILABLE", "Trusted font license declaration is required")
        # Re-hash on every boundary; cached glyph data may not hide replaced bytes.
        if self.font_path.stat().st_size > L.MAX_FONT_BYTES or stream_sha256(self.font_path) != digest:
            raise WorkspaceError("RENDER_FONT_CHANGED", "Trusted font bytes changed")
        if self._font_cache is None or self._font_cache.sha256 != digest:
            self._font_cache = L.FontFace(self.font_path, digest)
        return self._font_cache, {"id": ident, "sha256": digest, "license": license_text}

    def trusted_font(self):
        return self._font()[0]

    def options(self):
        try:
            _, font = self._font()
        except (WorkspaceError, OSError) as exc:
            return {"available": False, "font": None, "default_recipe": None,
                    "layout_version": L.VERSION, "renderer_version": R.VERSION,
                    "limits": self._limits(), "error_code": getattr(exc, "code", "RENDER_FONT_UNAVAILABLE"),
                    "reason": "Trusted configured font or its license is unavailable or invalid"}
        recipe = {"version": 1, "position": "bottom", "panel_extent_px": 420,
                  "body_font_px": 22, "title_font_px": 26, "padding_px": 16, "line_spacing_px": 6,
                  "fields": ["label", "text", "interval"], "tracks": [], "font_sha256": font["sha256"],
                  "font_license": font["license"],
                  "encoding": {"video_codec": "libx264", "pixel_format": "yuv420p", "crf": 18, "preset": "fast",
                               "audio_codec": "copy", "audio_bitrate_kbps": 192, "color_policy": "preserve_declared_proxy"}}
        missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
        return {"available": not missing, "font": font, "default_recipe": L.normalize_recipe(recipe), "limits": self._limits(),
                "layout_version": L.VERSION, "renderer_version": R.VERSION,
                **({"error_code": "DEPENDENCY_UNAVAILABLE", "reason": "Render tools are unavailable: " + ", ".join(missing)} if missing else {})}

    def _limits(self):
        return {"deadline_seconds": self.deadline_seconds, "max_output_bytes": str(self.max_output_bytes),
                "download_max_bytes": str(DOWNLOAD_MAX_BYTES),
                "maximum_panels": R.MAX_UNIQUE_PANELS, "maximum_spans": R.MAX_SPANS,
                "maximum_frames": L.MAX_FRAME_COUNT, "scope": "registered_playback_only"}

    def _recipe(self, recipe):
        recipe = L.normalize_recipe(recipe)
        face, font = self._font()
        if recipe["font_sha256"] != font["sha256"] or recipe["font_license"] != font["license"]:
            raise SchemaInvalid("RENDER_FONT_CHANGED", "Recipe must use the trusted configured font and license")
        return recipe, face

    def _snapshot_artifacts(self, task, media):
        ids = {media["frame_map_artifact"]}
        archived = task.get("extensions", {}).get("professional", {}).get("import_source", {}).get("document_artifact")
        if archived is not None:
            ids.add(archived["id"])
        result = []
        for ident in sorted(ids):
            row = self.store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE id=?", (ident,)).fetchone()
            if row is None or row[0] != "objects/" + row[1]:
                raise WorkspaceError("RECOVERY_REQUIRED", "Frozen render artifact is missing")
            descriptor = {"id": ident, "relative_path": row[0], "sha256": row[1], "size_bytes": str(row[2])}
            if archived is not None and ident == archived["id"] and archived != descriptor:
                raise WorkspaceError("RECOVERY_REQUIRED", "Frozen source archive binding is inconsistent")
            result.append(descriptor)
        return result

    def _request_parameters(self, task, recipe):
        media = self.store.get_media(task["media_id"])
        artifacts = self._snapshot_artifacts(task, media)
        template = next((t for t in self.store.list_templates() if
                         (t["id"], t["version"]) == (task["template_id"], task["template_version"])), None)
        if template is None:
            raise WorkspaceError("RECOVERY_REQUIRED", "Frozen render template is missing")
        return {"recipe": recipe, "renderer_version": R.VERSION, "layout_version": L.VERSION,
                "font": {"sha256": recipe["font_sha256"], "license": recipe["font_license"]},
                "frame_map": next(a for a in artifacts if a["id"] == media["frame_map_artifact"]),
                "template": template, "artifacts": artifacts, "deadline_seconds": self.deadline_seconds,
                "max_output_bytes": self.max_output_bytes}

    def submit(self, task_id, *, command_id, expected_revision, expected_sha256, recipe):
        S.require_id(command_id); S.require_id(task_id); S.require_hash(expected_sha256)
        if type(expected_revision) is not int or not 0 <= expected_revision <= S.MAX_SAFE_INTEGER:
            raise SchemaInvalid("SCHEMA_INVALID", "Render needs a nonnegative exact revision")
        recipe = L.normalize_recipe(recipe)
        with self.store.transaction():
            self.store.get_task(task_id)
            row = self.store.conn.execute("SELECT body,content_sha256 FROM task_revisions WHERE task_id=? AND revision=?",
                                           (task_id, expected_revision)).fetchone()
            if row is None:
                raise Conflict("REVISION_CONFLICT", "Expected task revision is not available")
            task = json.loads(row[0])
            if (task.get("id") != task_id or task.get("revision") != expected_revision
                    or task.get("content_sha256") != row[1] or self.store.task_digest(task) != row[1]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Historical render snapshot is inconsistent")
            if row[1] != expected_sha256:
                raise Conflict("MEDIA_CHANGED", "Expected render snapshot digest differs")
            # Use the requested historical snapshot to rebuild receipt identity.
            # A later professional import can add a different source archive;
            # it must not change an already accepted enqueue's parameters.
            parameters = self._request_parameters(task, recipe)
            items = [{"task_id": task_id, "expected_revision": expected_revision, "expected_sha256": expected_sha256}]
            request = {"kind": "queue.create_batch", "items": items, "operation": OPERATION,
                       "processor_version": VERSION, "parameters": parameters}
            # A layout upgrade changes new job identity, but must not change
            # the response to an acknowledged old command. Rebuild only the
            # known historical layout field; every other frozen request byte
            # must still match its stored canonical digest. Unknown/changed
            # requests go through the ordinary conflict check, never enqueue
            # with historical parameters.
            receipt = self.store.conn.execute("SELECT request_sha256 FROM command_receipts WHERE command_id=?",
                                               (command_id,)).fetchone()
            replay_request = request
            if receipt is not None and receipt[0] != S.sha256_text(request):
                for layout_version in sorted(READ_ONLY_LAYOUT_VERSIONS):
                    historical = {**request, "parameters": {**parameters, "layout_version": layout_version}}
                    if receipt[0] == S.sha256_text(historical):
                        replay_request = historical
                        break
            replay = self.queue._receipt(command_id, replay_request)
            if replay is not None:
                return replay
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace accepts no new renders")
            self._recipe(recipe)
            return self.queue.create_batch(items, operation=OPERATION, processor_version=VERSION,
                                           parameters=parameters, command_id=command_id)

    def _job(self, job, *, allow_legacy_read=False):
        if job["operation"] != OPERATION or job["processor_version"] != VERSION:
            raise WorkspaceError("UNSUPPORTED_PROCESSOR", "No renderer matches the frozen job")
        p = job["parameters"]
        expected = {"recipe", "renderer_version", "layout_version", "font", "frame_map", "template",
                    "artifacts", "deadline_seconds", "max_output_bytes"}
        if (not isinstance(p, dict) or set(p) != expected or p["renderer_version"] != R.VERSION
                or not isinstance(p["layout_version"], str) or L.normalize_recipe(p["recipe"]) != p["recipe"]
                or p["font"] != {"sha256": p["recipe"]["font_sha256"], "license": p["recipe"]["font_license"]}):
            raise SchemaInvalid("SCHEMA_INVALID", "Frozen render configuration is invalid")
        if p["layout_version"] != L.VERSION and not (
                allow_legacy_read and p["layout_version"] in READ_ONLY_LAYOUT_VERSIONS
                and job["state"] == "succeeded" and job["commit"] is not None):
            raise WorkspaceError("UNSUPPORTED_RENDER_LAYOUT", "Frozen render layout is unavailable for this operation")
        _positive(p["deadline_seconds"], "deadline_seconds")
        if type(p["max_output_bytes"]) is not int or not 1024 * 1024 <= p["max_output_bytes"] <= DOWNLOAD_MAX_BYTES:
            raise SchemaInvalid("INVALID_RENDER_BUDGET", "Frozen output cap is invalid")
        spec = {k: job[k] for k in ("task_id", "operation", "input_revision", "input_sha256", "processor_version", "parameters")}
        if S.sha256_text(spec) != job["idempotency_key"] or job["id"] != "job-" + job["idempotency_key"]:
            raise WorkspaceError("RECOVERY_REQUIRED", "Frozen render job identity is inconsistent")
        self._validate_snapshot(job)
        task, media = job["task_snapshot"], job["media_snapshot"]
        if task["revision"] != job["input_revision"] or task["id"] != job["task_id"] or task["media_id"] != media["id"]:
            raise WorkspaceError("RECOVERY_REQUIRED", "Render task snapshot is inconsistent")
        with self.store.transaction():
            if (self._snapshot_artifacts(task, media) != p["artifacts"] or p["frame_map"] not in p["artifacts"]
                    or p["frame_map"]["id"] != media["frame_map_artifact"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Render frame/source snapshot changed")
            original = next((t for t in self.store.list_templates() if
                             (t["id"], t["version"]) == (task["template_id"], task["template_version"])), None)
            if original != p["template"]:
                raise WorkspaceError("RECOVERY_REQUIRED", "Render template snapshot changed")
        return p

    def _batches(self, batch_id=None):
        batches = [self.queue.get_batch(batch_id)] if batch_id is not None else self.queue.list_batches()
        result = []
        for batch in batches:
            matching = [job["operation"] == OPERATION for job in batch["jobs"]]
            if any(matching) and not all(matching):
                raise WorkspaceError("RECOVERY_REQUIRED", "Mixed render batch is invalid")
            if matching and all(matching):
                if not batch["dispatch_paused"] and any(job["state"] == "queued" for job in batch["jobs"]):
                    result.append(batch["id"])
            elif batch_id is not None:
                raise Conflict("UNSUPPORTED_PROCESSOR", "Batch is not a render batch")
        return result

    def run_pending(self, *, batch_id=None, worker_id="render-local"):
        token = object()
        with self._condition:
            if self.is_closing():
                return []
            # The parent runner ends before our durable pin cleanup. Keep this
            # outer token until cleanup finishes so close cannot release SQLite
            # while the final business operation still uses the connection.
            self._runners.add(token)
        try:
            return super().run_pending(batch_id=batch_id, worker_id=worker_id)
        finally:
            try:
                self._recover_pins()
            finally:
                with self._condition:
                    self._runners.discard(token); self._condition.notify_all()

    def _prepare(self, task, media, recipe, face, work, control, frame_descriptor):
        source, _, _ = resolve_media_object(self.store, media["id"], "playback_path")
        mapping, digest, _ = resolve_media_object(self.store, media["id"], "frame_map_path")
        if digest != frame_descriptor["sha256"]:
            raise Conflict("MEDIA_CHANGED", "Render frame map changed")
        inspect = work / "inspect"; inspect.mkdir(parents=True, exist_ok=False)
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise DependencyUnavailable("DEPENDENCY_UNAVAILABLE", "ffprobe is required for actual render geometry")
        try:
            probe = P._probe(ffprobe, source, inspect, "proxy-probe", control)
        except P.MediaPreparationError as exc:
            raise WorkspaceError("RENDER_INPUT_UNVERIFIED", "Actual proxy geometry could not be verified") from exc
        videos = [s for s in probe["streams"] if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
        if len(videos) != 1:
            raise SchemaInvalid("UNSUPPORTED_RENDER_MEDIA", "Render needs exactly one admitted video stream")
        def rows():
            with mapping.open("rb") as stream:
                count = 0
                while True:
                    line = stream.readline(4097)
                    if not line:
                        break
                    if control():
                        raise WorkspaceError("CANCELLED", "Render planning was cancelled")
                    if len(line) > 4096 or count >= L.MAX_FRAME_COUNT:
                        raise SchemaInvalid("INVALID_FRAME_MAP", "Render frame map exceeds its bound")
                    yield json.loads(line)
                    count += 1
        plan = L.prepare_plan(task, media, rows(), recipe, font=face,
            playback_size=(videos[0].get("width"), videos[0].get("height")), frame_map_sha256=digest, cancelled=control)
        if stream_sha256(mapping, cancelled=control) != digest:
            raise Conflict("MEDIA_CHANGED", "Frame map changed during render planning")
        return source, plan

    def plan(self, task_id, *, expected_revision, expected_sha256, recipe):
        S.require_id(task_id); S.require_hash(expected_sha256)
        if type(expected_revision) is not int or not 0 <= expected_revision <= S.MAX_SAFE_INTEGER:
            raise SchemaInvalid("SCHEMA_INVALID", "Render preview requires a nonnegative exact revision")
        recipe, face = self._recipe(recipe)
        with self.store.transaction():
            task = self.store.get_task(task_id)
            if task["revision"] != expected_revision:
                raise Conflict("REVISION_CONFLICT", "Task changed before render planning")
            if self.store.task_digest(task) != expected_sha256:
                raise Conflict("MEDIA_CHANGED", "Task digest changed before render planning")
            media = self.store.get_media(task["media_id"])
            parameters = self._request_parameters(task, recipe)
        token = object()
        with self._condition:
            if self.is_closing():
                raise Conflict("WORKSPACE_CLOSING", "Workspace is closing")
            self._runners.add(token)
        started = time.monotonic()
        def control():
            if self.is_closing():
                return True
            if time.monotonic() - started >= min(self.deadline_seconds, 120):
                raise WorkspaceError("RENDER_TIMEOUT", "Render preview planning exceeded its deadline")
            return False
        try:
            work = self.store.tmp_dir / "render" / "previews" / ("preview-" + uuid.uuid4().hex)
            _safe(work); work.mkdir(parents=True, exist_ok=False)
            with get_prepare_slot(self.store.root).hold(closing=self.is_closing, timeout=min(self.slot_wait_seconds, self.deadline_seconds, 120)):
                return self._prepare(task, media, recipe, face, work, control, parameters["frame_map"])[1]
        except InterruptedError as exc:
            raise WorkspaceError("CANCELLED", "Render planning cancelled") from exc
        except SlotClosed as exc:
            raise WorkspaceError("CANCELLED", "Render planning stopped during shutdown") from exc
        except SlotTimeout as exc:
            raise WorkspaceError("RENDER_TIMEOUT", "Render planning could not acquire the decode slot in time", retryable=True) from exc
        finally:
            with self._condition:
                self._runners.discard(token); self._condition.notify_all()

    def _processor(self, job, context):
        p = self._job(job)
        recipe, face = self._recipe(p["recipe"])
        started = time.monotonic()
        stop, lost = threading.Event(), threading.Event()
        def renew():
            try:
                while not stop.wait(min(5.0, self.queue.lease_seconds / 3)):
                    try:
                        if context.heartbeat()["cancel_requested"]:
                            return
                    except Exception:
                        lost.set(); return
            finally:
                with self._condition:
                    self._heartbeats.discard(threading.current_thread()); self._condition.notify_all()
        def control():
            if self.is_closing():
                try:
                    self.queue.cancel(job["id"])
                except Conflict as exc:
                    if exc.code != "ALREADY_COMMITTED":
                        raise
                return True
            if time.monotonic() - started >= p["deadline_seconds"]:
                raise WorkspaceError("RENDER_TIMEOUT", "Render exceeded its total attempt deadline", retryable=True)
            if lost.is_set():
                raise Conflict("LEASE_LOST", "Render lease renewal failed")
            return context.cancelled()
        heartbeat = threading.Thread(target=renew, name="render-heartbeat", daemon=True)
        with self._condition:
            self._active_jobs.add(job["id"]); self._heartbeats.add(heartbeat)
        heartbeat.start()
        try:
            if control():
                raise WorkspaceError("CANCELLED", "Render cancelled before planning")
            work = self.store.tmp_dir / "render" / "jobs" / job["id"] / f"{job['attempt']}-{job['lease_generation']}"
            _safe(work); work.mkdir(parents=True, exist_ok=False)
            source, plan = self._prepare(job["task_snapshot"], job["media_snapshot"], recipe, face, work, control, p["frame_map"])
            manifest = R.render_video(source, plan, work / "render", font=face, cancelled=control,
                                      max_output_bytes=p["max_output_bytes"])
            validated = work / "render" / "validated"
            required = sum((validated / name).stat().st_size for name in FILES) + R.MIN_FREE_RESERVE
            if shutil.disk_usage(self.store.root).free < required:
                raise WorkspaceError("STORAGE_FULL", "Insufficient reserve for immutable render publication", retryable=True)
            files = {}
            for filename in FILES:
                if control():
                    raise WorkspaceError("CANCELLED", "Render publication cancelled")
                path = validated / filename
                _safe(path)
                if not path.is_file():
                    raise WorkspaceError("RECOVERY_REQUIRED", "Validated fixed render payload is missing")
                if filename != "annotation.mp4" and path.stat().st_size > MAX_JSON_BYTES:
                    raise WorkspaceError("RENDER_LIMIT_EXCEEDED", "Validated render JSON exceeds its bound")
                if filename == "annotation.mp4" and path.stat().st_size > p["max_output_bytes"]:
                    raise WorkspaceError("RENDER_OUTPUT_LIMIT", "Validated render exceeds its final cap")
                with path.open("rb") as stream:
                    files[filename] = self.store._publish_stream(_Reader(stream, control))
            candidate = {"kind": KIND, "version": 1, "processor_version": VERSION, "job_id": job["id"],
                "input_binding": _binding(job["task_snapshot"]), "render_binding": plan["binding"],
                "recipe": recipe, "renderer": manifest, "files": files,
                "scope": "frozen_registered_playback_render; annotations_not_reclassified; candidate_not_reviewed"}
            self._candidate(candidate, job, verify_files=True, require_pins=False)
            # Pins and normal queue finish are a single durable business tx.
            # execute_once subsequently repeats finish with identical JSON and
            # receives its existing commit, rather than creating a second one.
            with self.store.transaction():
                context.raise_if_cancelled()
                if control():
                    raise WorkspaceError("CANCELLED", "Render cancelled at commit barrier")
                context.heartbeat()
                for filename, role in FILES.items():
                    self.store.bind_artifact_reference(files[filename], owner_kind="render_manifest",
                        owner_id=job["id"], owner_revision=str(job["lease_generation"]), role=role)
                context.raise_if_cancelled()
                self.queue.finish(job["id"], job["lease_generation"], job["worker_id"], candidate)
            return candidate
        except InterruptedError as exc:
            raise WorkspaceError("CANCELLED", "Render was cancelled") from exc
        finally:
            stop.set(); heartbeat.join(timeout=6)
            with self._condition:
                self._active_jobs.discard(job["id"]); self._condition.notify_all()

    def _recover_pins(self):
        """Release only our obsolete uncommitted generations, never files."""
        with self.store.transaction():
            rows = self.store.conn.execute("SELECT artifact_id,owner_id,owner_revision,role FROM artifact_references "
                                           "WHERE owner_kind='render_manifest'").fetchall()
            for artifact_id, owner, revision, role in rows:
                raw = self.store.conn.execute("SELECT input FROM jobs WHERE id=?", (owner,)).fetchone()
                if raw is None or role not in FILES.values():
                    continue
                job = self.queue.get_job(owner)
                if job["operation"] != OPERATION:
                    continue
                generation = int(revision)
                if generation > job["lease_generation"]:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Render pin generation exceeds job generation")
                commit = job["commit"]
                if commit is not None:
                    if job["state"] != "succeeded":
                        raise WorkspaceError("RECOVERY_REQUIRED", "Committed render state is inconsistent")
                    if commit["lease_generation"] == generation:
                        continue
                if job["state"] in ("running", "cancel_requested") or (owner, generation) in self.store._queue_active_processors:
                    continue
                self.store.release_artifact_reference(artifact_id, owner_kind="render_manifest",
                    owner_id=owner, owner_revision=revision, role=role)

    def _artifact(self, descriptor, *, job=None, role=None, control=None, limit=None):
        if not isinstance(descriptor, dict) or set(descriptor) != {"id", "relative_path", "sha256", "size_bytes"}:
            raise WorkspaceError("RECOVERY_REQUIRED", "Render artifact descriptor is invalid")
        S.require_id(descriptor["id"]); S.require_hash(descriptor["sha256"])
        with self.store.transaction():
            row = self.store.conn.execute("SELECT rel_path,sha256,size_bytes FROM artifacts WHERE id=?", (descriptor["id"],)).fetchone()
            if (row is None or descriptor != {"id": descriptor["id"], "relative_path": row[0], "sha256": row[1], "size_bytes": str(row[2])}
                    or row[0] != "objects/" + row[1] or descriptor["id"] != S.make_id("art", row[1])
                    or type(row[2]) is not int or row[2] < 0 or (limit is not None and row[2] > limit)):
                raise WorkspaceError("RECOVERY_REQUIRED", "Render artifact is not registered consistently")
            if job is not None and self.store.conn.execute("SELECT 1 FROM artifact_references WHERE artifact_id=? AND owner_kind='render_manifest' "
                    "AND owner_id=? AND owner_revision=? AND role=?", (descriptor["id"], job["id"], str(job["lease_generation"]), role)).fetchone() is None:
                raise WorkspaceError("RECOVERY_REQUIRED", "Successful render artifact pin is missing")
        path = self.store.root / row[0]
        _safe(path)
        info = path.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size != row[2]
                or stream_sha256(path, cancelled=control) != row[1]):
            raise WorkspaceError("RECOVERY_REQUIRED", "Registered render bytes are missing or corrupt")
        return path

    def _candidate(self, candidate, job, *, verify_files, require_pins, allow_legacy_read=False):
        p = self._job(job, allow_legacy_read=allow_legacy_read)
        expected = {"kind", "version", "processor_version", "job_id", "input_binding", "render_binding",
                    "recipe", "renderer", "files", "scope"}
        if (not isinstance(candidate, dict) or set(candidate) != expected or candidate["kind"] != KIND
                or candidate["version"] != 1 or candidate["processor_version"] != VERSION or candidate["job_id"] != job["id"]
                or candidate["input_binding"] != _binding(job["task_snapshot"]) or candidate["recipe"] != p["recipe"]
                or not isinstance(candidate["files"], dict) or set(candidate["files"]) != set(FILES)):
            raise WorkspaceError("RECOVERY_REQUIRED", "Render candidate snapshot is inconsistent")
        paths = {}
        if verify_files:
            for filename, role in FILES.items():
                cap = p["max_output_bytes"] if filename == "annotation.mp4" else MAX_JSON_BYTES
                paths[filename] = self._artifact(candidate["files"][filename], job=job if require_pins else None, role=role, limit=cap)
                if paths[filename].stat().st_size > cap:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Render artifact exceeds its frozen bound")
            plan = _json_file(paths["render-plan.json"])
            manifest = _json_file(paths["project.json"])
            verification = _json_file(paths["verification.json"])
            L._verify_plan(plan)
            if (manifest != candidate["renderer"] or plan["binding"] != candidate["render_binding"]
                    or plan["recipe"] != p["recipe"] or plan.get("renderer_version") != p["layout_version"]):
                raise WorkspaceError("RECOVERY_REQUIRED", "Render payload differs from candidate")
            binding = plan["binding"]; media = job["media_snapshot"]
            wanted = {"task_id": job["task_id"], "revision": job["input_revision"], "task_content_sha256": job["input_sha256"],
                      "media_id": media["id"], "source_sha256": media["source"]["sha256"], "playback_sha256": media["playback"]["sha256"],
                      "frame_map_artifact_id": media["frame_map_artifact"], "frame_map_sha256": p["frame_map"]["sha256"],
                      "font_sha256": p["font"]["sha256"]}
            if any(binding.get(key) != value for key, value in wanted.items()):
                raise WorkspaceError("RECOVERY_REQUIRED", "Render binding differs from frozen job")
            if (manifest.get("binding") != binding or manifest.get("layout_sha256") != plan["content_sha256"]
                    or manifest.get("renderer_version") != p["renderer_version"]
                    or manifest.get("kind") != "traceannotate.rendered-video" or manifest.get("status") != "verified_not_workspace_published"
                    or S.sha256_text({k:v for k,v in manifest.items() if k != "content_sha256"}) != manifest.get("content_sha256")):
                raise WorkspaceError("RECOVERY_REQUIRED", "Render project manifest is inconsistent")
            if (verification.get("kind") != "traceannotate.render-verification" or verification.get("version") != 1
                    or verification.get("full_decode_exit") != 0 or verification.get("all_frame_pts_exact") is not True
                    or verification.get("final_hold_exact") is not True or verification.get("frame_count") != media["frame_count"]
                    or manifest.get("validation", {}).get("full_decode_exit") != 0
                    or manifest.get("validation", {}).get("all_frame_pts_exact") is not True):
                raise WorkspaceError("RECOVERY_REQUIRED", "Render validation report is incomplete")
            records = manifest.get("files", [])
            if not isinstance(records, list) or len(records) != 4 or {r.get("file") for r in records} != set(FILES) - {"project.json"}:
                raise WorkspaceError("RECOVERY_REQUIRED", "Render project lists unexpected files")
            for record in records:
                descriptor = candidate["files"][record["file"]]
                if record.get("sha256") != descriptor["sha256"] or record.get("size_bytes") != descriptor["size_bytes"]:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Render project file identity differs")
        return paths

    def result(self, job_id):
        S.require_id(job_id)
        try:
            with self.store.transaction():
                job = self.queue.get_job(job_id)
                if job["operation"] != OPERATION:
                    raise Conflict("UNSUPPORTED_PROCESSOR", "Job is not a render job")
                if job["state"] != "succeeded" or job["commit"] is None:
                    raise Conflict("RENDER_NOT_READY", "Render has no successful durable candidate")
                commit = job["commit"]
                durable = self.store.conn.execute("SELECT body FROM durable_commits WHERE id=?", (commit["id"],)).fetchone()
                if (durable is None or json.loads(durable[0]) != commit or commit["disposition"] != "candidate"
                        or commit["job_id"] != job_id or any(commit[k] != job[k] for k in
                            ("task_id", "input_revision", "input_sha256", "processor_version", "idempotency_key", "lease_generation", "worker_id"))):
                    raise WorkspaceError("RECOVERY_REQUIRED", "Render commit is inconsistent")
            path = self._artifact(commit["artifact"], limit=MAX_RESULT_BYTES)
            if path.stat().st_size > MAX_RESULT_BYTES:
                raise WorkspaceError("RECOVERY_REQUIRED", "Render candidate is oversized")
            candidate = _json_file(path, MAX_RESULT_BYTES)
            self._candidate(candidate, job, verify_files=True, require_pins=True, allow_legacy_read=True)
            with self.store.transaction():
                current_task = self.store.get_task(job["task_id"])
                if self.store.task_digest(current_task) != current_task["content_sha256"]:
                    raise WorkspaceError("RECOVERY_REQUIRED", "Current task binding is corrupt")
                current = _binding(current_task)
            return {"kind": "traceannotate.render-result", "version": 1, "job_id": job_id,
                    "task_id": job["task_id"], "input_binding": candidate["input_binding"],
                    "current_binding": current, "manifest": candidate}
        except (OSError, ValueError, TypeError, KeyError, RecursionError, SchemaInvalid) as exc:
            raise WorkspaceError("RECOVERY_REQUIRED", "Render candidate is missing or corrupt") from exc

    def video(self, job_id):
        result = self.result(job_id)
        descriptor = result["manifest"]["files"]["annotation.mp4"]
        return self.store.root / descriptor["relative_path"], descriptor["sha256"], int(descriptor["size_bytes"])

    def frozen_document(self, job_id):
        result = self.result(job_id)
        job = self.queue.get_job(job_id)
        p = self._job(job, allow_legacy_read=True)
        document = build_document(job["task_snapshot"], job["media_snapshot"], p["template"], artifacts=p["artifacts"])
        data = S.canonical_json(document)
        if len(data) > MAX_JSON_BYTES:
            raise WorkspaceError("PROFESSIONAL_LIMIT_EXCEEDED", "Frozen render JSON exceeds its export bound")
        return {"data": data, "content_type": "application/json; charset=utf-8",
                "filename": f"TraceAnnotate-{job['task_id']}-r{job['input_revision']}.render.professional.json",
                "binding": result["input_binding"],
                "losses": "Complete frozen task; history not included; media and archived source bytes not embedded. Pair with this exact render candidate."}
