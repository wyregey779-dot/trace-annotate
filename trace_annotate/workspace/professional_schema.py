"""Bounded, side-effect-free validation for professional task exchange v1.

File declarations, measured values and observations remain distinct. A valid
document is not proof of a semantic review or of possession of external media.
Frame-map row verification belongs to the receiving store's admission path.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import schema as S
from .errors import SchemaInvalid

KIND = "traceannotate.professional-task"
SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_DEPTH = 48
MAX_NODES = 400_000
MAX_ANNOTATIONS = 20_000
MAX_REPORTS = 256
MAX_TEXT_CHARS = 1_000_000


def _invalid(message, code="INVALID_PROFESSIONAL_DOCUMENT"):
    raise SchemaInvalid(code, message)


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        _invalid("Object fields are missing or unknown")


def _text(value, *, nonempty=False):
    if not isinstance(value, str) or len(value) > MAX_TEXT_CHARS or (nonempty and not value):
        _invalid("Text is invalid or exceeds the supported bound")


def _integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= S.MAX_SAFE_INTEGER:
        _invalid("Integer is outside the supported bound")


def _decimal(value, *, signed=False):
    if not isinstance(value, str) or len(value) > 40 or not re.fullmatch(r"-?(0|[1-9][0-9]*)" if signed else r"0|[1-9][0-9]*", value) or value == "-0":
        _invalid("Exact integer must be a canonical decimal string")
    return int(value)


def _rational(value):
    if not isinstance(value, dict) or S.rational(value) != value:
        _invalid("Time must be a reduced rational")
    return S.rational_value(value)


def _list(value, maximum=MAX_ANNOTATIONS):
    if not isinstance(value, list) or len(value) > maximum:
        _invalid("Array is invalid or exceeds the supported bound")
    return value


def bounded_json(value):
    """Reject depth, node, text and non-finite data before canonical encoding."""
    count = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            _invalid("JSON exceeds its node or nesting budget", "PROFESSIONAL_LIMIT_EXCEEDED")
        if isinstance(item, dict):
            for key, child in item.items():
                _text(key)
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            _text(item)
    data = S.canonical_json(value)
    if len(data) > MAX_DOCUMENT_BYTES:
        _invalid("Professional JSON exceeds its byte budget", "PROFESSIONAL_LIMIT_EXCEEDED")
    return data


def document_digest(document):
    return S.sha256_text({key: value for key, value in document.items() if key != "content_sha256"})


def decode_document(data):
    if not isinstance(data, bytes) or len(data) > MAX_DOCUMENT_BYTES:
        _invalid("Professional input must be bounded UTF-8 bytes", "PROFESSIONAL_LIMIT_EXCEEDED")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate JSON key")
            out[key] = value
        return out
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SchemaInvalid("INVALID_PROFESSIONAL_DOCUMENT", "Professional JSON is invalid") from exc
    return validate_document(result)


def _file(value):
    _object(value, {"sha256", "size_bytes", "display_name"})
    S.require_hash(value["sha256"])
    _decimal(value["size_bytes"])
    _text(value["display_name"], nonempty=True)


def validate_media(media):
    _object(media, {"id", "source", "playback", "duration", "frame_count", "frame_map_artifact",
                   "source_time_base", "proxy_time_base", "common_origin", "video_start", "video_end",
                   "source_video_stream", "source_width", "source_height", "pixel_aspect_ratio",
                   "rotation_degrees", "coordinate_policy"})
    S.require_id(media["id"]); S.require_id(media["frame_map_artifact"])
    _file(media["source"]); _file(media["playback"])
    duration = _rational(media["duration"])
    if not 0 <= _rational(media["video_start"]) < _rational(media["video_end"]) <= duration:
        _invalid("Video interval is outside the media duration")
    if _decimal(media["frame_count"]) < 1:
        _invalid("Media must contain decoded video frames")
    for name in ("source_time_base", "proxy_time_base", "pixel_aspect_ratio"):
        if _rational(media[name]) <= 0:
            _invalid("Time bases and pixel aspect ratio must be positive")
    _rational(media["common_origin"])
    _integer(media["source_video_stream"])
    _integer(media["source_width"], 1); _integer(media["source_height"], 1)
    if type(media["rotation_degrees"]) is not int or media["rotation_degrees"] not in (0, 90, 180, 270):
        _invalid("Unsupported rotation declaration")
    if media["coordinate_policy"] != "source_decoded_pixels_with_explicit_transform":
        _invalid("Unsupported coordinate policy")
    return media


def analysis_input_digest(report):
    body = {"dependency_version": "analysis-input-v1", **{key: report[key] for key in
            ("scope", "file_sha256", "stream_index", "time_domain", "method")}}
    if report["scope"] == "interval":
        body["window"] = report["window"]
    body["subject_annotation_id"] = report["subject_annotation_id"]
    return S.sha256_text(body)


def _validate_analysis(report, media):
    bounded_json(report)
    _object(report, {"id", "kind", "scope", "file_sha256", "stream_index", "time_domain", "window",
                     "method", "input_digest", "status", "values", "evidence_ids", "subject_annotation_id"})
    S.require_id(report["id"]); S.require_hash(report["file_sha256"])
    S.require_hash(report["input_digest"])
    if report["kind"] not in ("technical_declared", "loudness", "audio_observation", "color_statistics", "frame_timing"):
        _invalid("Unknown analysis kind")
    if report["scope"] not in ("file", "stream", "interval"):
        _invalid("Unknown analysis scope")
    if report["subject_annotation_id"] is not None:
        S.require_id(report["subject_annotation_id"])
        if report["scope"]!="interval":
            _invalid("Only an interval report can bind an annotation subject")
    if report["stream_index"] is not None:
        _integer(report["stream_index"])
    if report["scope"] == "stream" and report["stream_index"] is None:
        _invalid("Stream analysis requires its actual stream index")
    if report["time_domain"] not in ("original_source_seconds", "admitted_source_seconds", "playback_seconds"):
        _invalid("Unknown analysis time domain")
    expected_file = media["playback"] if report["time_domain"] == "playback_seconds" else media["source"]
    if report["file_sha256"] != expected_file["sha256"]:
        _invalid("Analysis file does not match the declared time domain", "MEDIA_IDENTITY_MISMATCH")
    if report["scope"] == "interval":
        _object(report["window"], {"start", "end"})
        if not 0 <= _rational(report["window"]["start"]) < _rational(report["window"]["end"]) <= _rational(media["duration"]):
            _invalid("Analysis window is outside media")
    elif report["window"] is not None:
        _invalid("File and stream analysis must not claim an interval window")
    _object(report["method"], {"id", "version", "parameters"})
    S.require_id(report["method"]["id"]); _text(report["method"]["version"], nonempty=True)
    if not isinstance(report["method"]["parameters"], dict):
        _invalid("Method parameters must be an object")
    states = {"available", "unavailable", "not_performed", "pending_review"}
    if report["status"] not in states:
        _invalid("Unknown analysis status")
    if report["evidence_ids"] != []:
        _invalid("Evidence files require a separate admission workflow", "UNSUPPORTED_PROFESSIONAL_EVIDENCE")
    names = set()
    for value in _list(report["values"], 1024):
        _object(value, {"name", "value", "unit", "basis", "status", "reason"})
        _text(value["name"], nonempty=True)
        if value["name"] in names:
            _invalid("Analysis value names must be unique")
        names.add(value["name"])
        if value["unit"] not in ("LUFS", "dBFS", "dBTP", "Hz", "fps", "px", "s", "ratio", "none"):
            _invalid("Unknown analysis unit")
        if value["basis"] not in ("declared", "measured", "observed", "inferred") or value["status"] not in states:
            _invalid("Unknown value basis or status")
        permitted={"technical_declared":{"declared"},"loudness":{"measured"},
                   "audio_observation":{"observed","inferred"},
                   "color_statistics":{"measured","observed","inferred"},
                   "frame_timing":{"declared","measured"}}
        if value["basis"] not in permitted[report["kind"]]:
            _invalid("Value basis does not match the declared analysis kind")
        if value["status"] == "available":
            if value["value"] is None or value["reason"] is not None or type(value["value"]) not in (str, int, float, bool):
                _invalid("Available measurement requires a scalar and no unavailability reason")
            if report["status"] != "available":
                _invalid("Unavailable report cannot contain an available observation")
        else:
            if value["value"] is not None:
                _invalid("Unavailable value cannot contain a fabricated measurement")
            _text(value["reason"], nonempty=True)
        if value["basis"] == "inferred" and value["status"] == "available":
            _invalid("Inferences must remain pending review")
        if value["basis"]=="observed" and value["status"]=="available":
            observation=report["method"]["parameters"].get("observation_source")
            _object(observation,{"actor_id","actor_type","input_file_sha256"})
            S.require_id(observation["actor_id"]);S.require_hash(observation["input_file_sha256"])
            if observation["actor_type"] not in ("human","assistant","tool","historical_unverified") or observation["input_file_sha256"]!=report["file_sha256"]:
                _invalid("Observation source must identify its actor role and input file")
        if value["basis"] == "declared" and report["kind"] not in ("technical_declared", "frame_timing"):
            _invalid("Declared file tags cannot stand in for acoustic or visual measurements")
        if value["unit"] in ("LUFS", "dBFS", "dBTP", "Hz", "fps", "px", "s", "ratio") and value["status"] == "available" and type(value["value"]) not in (int, float):
            _invalid("Numeric unit requires a numeric value")
    if report["status"] == "available" and not report["values"]:
        _invalid("Available report requires at least one value")
    if report["input_digest"] != analysis_input_digest(report):
        _invalid("Analysis input digest does not match its dependency fields", "ANALYSIS_DIGEST_MISMATCH")
    return report


def validate_analysis(report, media):
    try:
        return _validate_analysis(report,media)
    except SchemaInvalid:
        raise
    except (KeyError,TypeError,ValueError,OverflowError,RecursionError) as exc:
        raise SchemaInvalid("INVALID_PROFESSIONAL_DOCUMENT","Invalid typed analysis report") from exc


def derived_analysis_status(task, media, report):
    """Assess dependencies without changing stored results or declaring review."""
    if report["status"]!="available":
        return {"id":report["id"],"status":"unavailable","reason":report["status"]}
    file=media["playback"] if report["time_domain"]=="playback_seconds" else media["source"]
    if report["file_sha256"]!=file["sha256"] or report["input_digest"]!=analysis_input_digest(report):
        return {"id":report["id"],"status":"stale","reason":"input_identity_changed"}
    if report["subject_annotation_id"] is not None:
        annotation=next((a for a in task["annotations"] if a["id"]==report["subject_annotation_id"]),None)
        if annotation is None or annotation["type"]!="segment":
            return {"id":report["id"],"status":"stale","reason":"subject_missing_or_not_segment"}
        window={"start":annotation["start"]["seconds"],"end":annotation["end"]["seconds"]}
        if window!=report["window"]:
            return {"id":report["id"],"status":"stale","reason":"subject_interval_changed"}
    return {"id":report["id"],"status":"fresh","reason":None}


def task_content_digest(task, media, template, artifacts):
    media_copy={key:value for key,value in media.items()}
    for key in ("source","playback"):
        media_copy[key]={k:v for k,v in media[key].items() if k!="display_name"}
    frame=next((a for a in artifacts if a["id"]==media["frame_map_artifact"]),None)
    if frame is None:_invalid("Frame-map manifest is required")
    descriptor={key:frame[key] for key in ("id","sha256","size_bytes")}
    return S.sha256_text({"digest_version":"workspace-task-v1","task":S.task_subject(task),
                         "media":media_copy,"template":template,"artifacts":[descriptor]})


def validate_annotation(annotation, media, template):
    _object(annotation, {"id", "type", "label", "text", "tags", "track", "attributes", "evidence_ids", "provenance", "review_status"},
            {"start", "end", "at", "legacy_id", "review_binding"})
    S.require_id(annotation["id"])
    kind = annotation["type"]
    if kind not in ("segment", "event", "whole_clip") or kind not in template["annotation_types"]:
        _invalid("Annotation type is not admitted by its template")
    _text(annotation["label"], nonempty=True); _text(annotation["text"]); _text(annotation["track"], nonempty=True)
    if template["labels"] and annotation["label"] not in template["labels"]:
        _invalid("Annotation label is not admitted by template")
    for tag in _list(annotation["tags"], 4096):
        _text(tag)
    if "legacy_id" in annotation:
        _text(annotation["legacy_id"], nonempty=True)
    if annotation["attributes"] != {} or annotation["evidence_ids"] != []:
        _invalid("Custom annotation attributes and evidence require their own admission", "UNSUPPORTED_PROFESSIONAL_EVIDENCE")
    _object(annotation["provenance"], {"origin", "actor_id", "job_id"})
    if annotation["provenance"]["origin"] not in ("manual", "imported", "model_candidate", "derived"):
        _invalid("Unknown provenance origin")
    S.require_id(annotation["provenance"]["actor_id"])
    if annotation["provenance"]["job_id"] is not None:
        S.require_id(annotation["provenance"]["job_id"])
    if annotation["review_status"] not in ("unreviewed", "reviewed"):
        _invalid("Unknown review status")
    if annotation["provenance"]["origin"] == "model_candidate" and annotation["review_status"] == "reviewed":
        _invalid("Model candidate cannot carry a completed local review")
    if "review_binding" in annotation:
        review=annotation["review_binding"]
        _object(review,{"content_sha256","actor_id","command_id","reviewed_at","task_revision"})
        S.require_hash(review["content_sha256"]); S.require_id(review["actor_id"]); S.require_id(review["command_id"])
        _text(review["reviewed_at"],nonempty=True); _integer(review["task_revision"])
        if annotation["review_status"] != "reviewed":
            _invalid("Unreviewed annotation must not claim a current review binding")
    times={"segment":{"start","end"},"event":{"at"},"whole_clip":set()}[kind]
    if {"start","end","at"}.intersection(annotation) != times:
        _invalid("Annotation has wrong time fields")
    values={}
    duration=_rational(media["duration"])
    for key in times:
        value=annotation[key]
        _object(value,{"seconds"},{"anchor"})
        values[key]=_rational(value["seconds"])
        if not 0<=values[key]<=duration or (key!="end" and values[key]==duration):
            _invalid("Annotation time is outside media duration")
        if "anchor" in value:
            anchor=value["anchor"]
            _object(anchor,{"frame_index","source_pts","proxy_pts"})
            index=_decimal(anchor["frame_index"])
            source_pts=_decimal(anchor["source_pts"],signed=True);proxy_pts=_decimal(anchor["proxy_pts"],signed=True)
            if not 0<=index<_decimal(media["frame_count"]):
                _invalid("Frame anchor is outside media")
            if proxy_pts*_rational(media["proxy_time_base"])!=values[key]:
                _invalid("Frame anchor does not match its exact proxy time")
            if abs(source_pts*_rational(media["source_time_base"])-_rational(media["common_origin"])-values[key])>min(_rational(media["source_time_base"]),_rational(media["proxy_time_base"])):
                _invalid("Frame anchor source/proxy clocks disagree")
    if kind=="segment" and values["start"]>=values["end"]:
        _invalid("Segment start must precede end")
    return annotation


def _validate_document(document):
    bounded_json(document)
    _object(document,{"kind","schema_version","binding","task","media","template","analyses","analysis_assessment","actors","artifacts","history","content_sha256"})
    if document["kind"]!=KIND or type(document["schema_version"]) is not int or document["schema_version"]!=SCHEMA_VERSION:
        _invalid("Unsupported professional document version","UNSUPPORTED_PROFESSIONAL_VERSION")
    S.require_hash(document["content_sha256"])
    if document_digest(document)!=document["content_sha256"]:
        _invalid("Professional document hash mismatch","PROFESSIONAL_DIGEST_MISMATCH")
    media=validate_media(document["media"])
    template=document["template"]
    _object(template,{"id","version","name","annotation_types","labels","attributes","coverage_policy"})
    S.require_id(template["id"]);_integer(template["version"],1);_text(template["name"],nonempty=True)
    types=_list(template["annotation_types"],3)
    if not types or any(t not in ("segment","event","whole_clip") for t in types) or len(set(types))!=len(types):
        _invalid("Unsupported annotation template")
    for label in _list(template["labels"]):_text(label)
    if template["attributes"]!=[] or template["coverage_policy"] not in ("sparse","complete_partition"):
        _invalid("Unsupported template attributes or coverage policy")
    task=document["task"]
    _object(task,{"id","media_id","template_id","template_version","revision","content_sha256","annotations","extensions"},{"legacy_document_metadata","evidence"})
    for name in ("id","media_id","template_id"):S.require_id(task[name])
    _integer(task["revision"]);_integer(task["template_version"],1);S.require_hash(task["content_sha256"])
    if task["media_id"]!=media["id"] or (task["template_id"],task["template_version"])!=(template["id"],template["version"]):
        _invalid("Task media or template binding mismatch")
    if not isinstance(task["extensions"],dict) or not isinstance(task.get("legacy_document_metadata",{}),dict):
        _invalid("Task extension and legacy metadata must be objects")
    if task.get("evidence",[])!=[]:_invalid("Evidence files require admission","UNSUPPORTED_PROFESSIONAL_EVIDENCE")
    seen=set()
    for annotation in _list(task["annotations"]):
        validate_annotation(annotation,media,template)
        if annotation["id"] in seen:_invalid("Duplicate annotation identifier")
        seen.add(annotation["id"])
        if "review_binding" in annotation:
            subject={key:value for key,value in annotation.items() if key not in ("review_status","review_binding")}
            expected_review=S.sha256_text({"annotation":subject,"media_id":task["media_id"],
                 "template_id":task["template_id"],"template_version":task["template_version"]})
            if annotation["review_binding"]["content_sha256"]!=expected_review or annotation["review_binding"]["task_revision"]>task["revision"]:
                _invalid("Review binding does not match current annotation contents","STALE_REVIEW_BINDING")
    _object(document["binding"],{"task_id","media_id","revision","task_content_sha256"})
    binding={"task_id":task["id"],"media_id":task["media_id"],"revision":task["revision"],"task_content_sha256":task["content_sha256"]}
    if document["binding"]!=binding:_invalid("Document binding differs from frozen task")
    report_ids=set()
    for report in _list(document["analyses"],MAX_REPORTS):
        validate_analysis(report,media)
        if report["id"] in report_ids:_invalid("Duplicate analysis identifier")
        report_ids.add(report["id"])
    extension=task["extensions"].get("professional",{})
    if not isinstance(extension,dict):_invalid("Professional extension must be an object")
    if extension.get("analysis_reports",[])!=document["analyses"]:
        _invalid("Analyses differ from the authoritative saved task reports")
    if document["analysis_assessment"]!=[derived_analysis_status(task,media,r) for r in document["analyses"]]:
        _invalid("Analysis assessment does not match current dependencies")
    actors=set()
    for actor in _list(document["actors"],4096):
        _object(actor,{"id","type","display_name","identity_verification"})
        S.require_id(actor["id"]);_text(actor["display_name"],nonempty=True)
        if actor["id"] in actors or actor["type"] not in ("human","assistant","tool","historical_unverified") or actor["identity_verification"] not in ("local_only","unverified","verified_external"):
            _invalid("Invalid or duplicate actor")
        actors.add(actor["id"])
    if any(a["provenance"]["actor_id"] not in actors for a in task["annotations"]):
        _invalid("Annotation actor is missing from actor declarations")
    artifact_ids=set()
    for artifact in _list(document["artifacts"],50_000):
        _object(artifact,{"id","relative_path","sha256","size_bytes"})
        S.require_id(artifact["id"]);S.require_hash(artifact["sha256"]);_decimal(artifact["size_bytes"])
        if artifact["relative_path"]!="objects/"+artifact["sha256"] or artifact["id"] in artifact_ids:
            _invalid("Invalid artifact path or duplicate identifier")
        artifact_ids.add(artifact["id"])
    if media["frame_map_artifact"] not in artifact_ids:
        _invalid("Frame-map manifest is required")
    if task["content_sha256"]!=task_content_digest(task,media,template,document["artifacts"]):
        _invalid("Task content digest does not match exported media/template/annotation state","TASK_DIGEST_MISMATCH")
    history=document["history"]
    _object(history,{"scope","revisions","commits","receipts","source_envelopes"})
    if history["scope"] not in ("available_local_history","available_local_audit_index","not_included"):
        _invalid("Unknown history scope")
    for key in ("revisions","commits","receipts","source_envelopes"):
        for entry in _list(history[key]):
            if not isinstance(entry,dict):_invalid("History entries must be passive JSON objects")
    if history["scope"]=="not_included" and any(history[key] for key in ("revisions","commits","receipts","source_envelopes")):
        _invalid("History scope disagrees with entries")
    return document


def validate_document(document):
    try:
        return _validate_document(document)
    except SchemaInvalid:
        raise
    except (KeyError,TypeError,ValueError,OverflowError,RecursionError) as exc:
        raise SchemaInvalid("INVALID_PROFESSIONAL_DOCUMENT","Invalid professional document fields") from exc
