"""Complete task JSON construction and explicit import identity mapping.

These pure functions never locate files from imported paths or write a store.
The caller must perform transactional persistence and target frame-map checks.
"""
from __future__ import annotations

import copy

from . import schema as S
from . import professional_schema as P
from .errors import SchemaInvalid


def build_document(task, media, template, *, history=None, analyses=None, actors=None, artifacts=None):
    task,media,template=copy.deepcopy(task),copy.deepcopy(media),copy.deepcopy(template)
    if actors is None:
        actors=[{"id":ident,"type":"historical_unverified","display_name":ident,"identity_verification":"unverified"}
                for ident in sorted({ann["provenance"]["actor_id"] for ann in task["annotations"]})]
    if analyses is None:
        professional=task.get("extensions",{}).get("professional",{})
        if not isinstance(professional,dict):
            raise SchemaInvalid("INVALID_PROFESSIONAL_DOCUMENT","Professional extension must be an object")
        analyses=professional.get("analysis_reports",[])
    document={"kind":P.KIND,"schema_version":P.SCHEMA_VERSION,
              "binding":{"task_id":task["id"],"media_id":task["media_id"],"revision":task["revision"],"task_content_sha256":task["content_sha256"]},
              "task":task,"media":media,"template":template,"analyses":copy.deepcopy(analyses),"actors":copy.deepcopy(actors),
              "analysis_assessment":[P.derived_analysis_status(task,media,r) for r in analyses],
              "artifacts":copy.deepcopy(artifacts or []),
              "history":copy.deepcopy(history if history is not None else {"scope":"not_included","revisions":[],"commits":[],"receipts":[],"source_envelopes":[]})}
    document["content_sha256"]=P.document_digest(document)
    return P.validate_document(document)


def import_payload(document, target_media, destination_task_id, *, actor_id="local-user"):
    """Map into an already admitted local task; never execute source history.

    Original review/provenance and all source fields remain in a passive source
    envelope. New local annotations are explicitly imported and unreviewed.
    """
    P.validate_document(document);P.validate_media(target_media)
    S.require_id(destination_task_id);S.require_id(actor_id)
    source_media=document["media"]
    if source_media["source"] != target_media["source"]:
        # Display names are not media identity. Compare only bytes and digest.
        if any(source_media["source"][key]!=target_media["source"][key] for key in ("sha256","size_bytes")):
            raise SchemaInvalid("MEDIA_IDENTITY_MISMATCH","Professional document requires the matching admitted source")
    for key in ("duration","video_start","video_end","frame_count","source_time_base","proxy_time_base","common_origin"):
        if source_media[key]!=target_media[key]:
            raise SchemaInvalid("MEDIA_TIMELINE_MISMATCH","Target media timeline differs from exported media")
    # Exact anchors refer to the corresponding immutable map; require playback
    # identity too rather than guessing a mapping after another transcode.
    if any("anchor" in ann.get(key,{}) for ann in document["task"]["annotations"] for key in ("start","end","at")):
        if source_media["playback"]["sha256"]!=target_media["playback"]["sha256"]:
            raise SchemaInvalid("FRAME_MAP_IDENTITY_MISMATCH","Anchored import needs the matching playback identity")
    mapping={ann["id"]:S.make_id("ann",destination_task_id+":"+ann["id"]) for ann in document["task"]["annotations"]}
    annotations=[]
    for source in document["task"]["annotations"]:
        annotation=copy.deepcopy(source)
        annotation["id"]=mapping[source["id"]]
        annotation["legacy_id"]=source.get("legacy_id") or source["id"]
        annotation["review_status"]="unreviewed"
        annotation.pop("review_binding",None)
        annotation["provenance"]={"origin":"imported","actor_id":actor_id,"job_id":None}
        annotations.append(annotation)
    extensions=copy.deepcopy(document["task"]["extensions"])
    professional=extensions.setdefault("professional",{})
    if not isinstance(professional,dict):
        raise SchemaInvalid("INVALID_PROFESSIONAL_DOCUMENT","Stored professional extension is not an object")
    professional["analysis_reports"]=copy.deepcopy(document["analyses"])
    for report in professional["analysis_reports"]:
        if report["subject_annotation_id"] is not None:
            if report["subject_annotation_id"] not in mapping:
                # A deleted subject is already stale; preserve its absent local
                # identity deterministically rather than assigning a live one.
                report["subject_annotation_id"]=S.make_id("ann",destination_task_id+":"+report["subject_annotation_id"])
            else:
                report["subject_annotation_id"]=mapping[report["subject_annotation_id"]]
            report["input_digest"]=P.analysis_input_digest(report)
        # A report may describe the proxy even when the annotations themselves
        # have only second-based boundaries. Never admit another proxy's
        # measurements merely because its original source is the same.
        P.validate_analysis(report,target_media)
    professional["import_source"]={"source_document_sha256":document["content_sha256"],
        "source_task_id":document["task"]["id"],"source_media_id":source_media["id"],"destination_task_id":destination_task_id,
        "annotation_id_mapping":mapping,"document":copy.deepcopy(document),
        "scope":"historical_source_not_current_review_or_local_receipt"}
    return {"annotations":annotations,"extensions":extensions,
            "legacy_document_metadata":copy.deepcopy(document["task"].get("legacy_document_metadata",{})),
            "source_document_sha256":document["content_sha256"],"annotation_id_mapping":mapping}
