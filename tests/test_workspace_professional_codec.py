"""Pure exchange validation: full state, identity mapping and dependency scope."""
from __future__ import annotations

import copy
import json

import pytest

from trace_annotate.workspace import professional_schema as P
from trace_annotate.workspace import professional_export as E
from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import SchemaInvalid
from trace_annotate.workspace.store import DEFAULT_TEMPLATE


def fixture():
    artifact={"id":S.make_id("art","3"*64),"relative_path":"objects/"+"3"*64,"sha256":"3"*64,"size_bytes":"256"}
    media={"id":"med-source","source":{"sha256":"1"*64,"size_bytes":"1024","display_name":"测试.mp4"},
           "playback":{"sha256":"2"*64,"size_bytes":"2048","display_name":"playback.mp4"},
           "duration":S.rational(2),"video_start":S.rational(0),"video_end":S.rational(2),
           "frame_count":"2","frame_map_artifact":artifact["id"],"source_time_base":S.rational(1),
           "proxy_time_base":S.rational(1),"common_origin":S.rational(0),"source_width":576,"source_height":1024,
           "source_video_stream":1,"rotation_degrees":0,"pixel_aspect_ratio":S.rational(1),
           "coordinate_policy":"source_decoded_pixels_with_explicit_transform"}
    template=copy.deepcopy(DEFAULT_TEMPLATE)
    ann={"id":"ann-original","type":"segment","label":"镜头","text":"全文保留：中文、{大括号}、\\斜线和换行\n不截句。",
         "start":{"seconds":S.rational(0),"anchor":{"frame_index":"0","source_pts":"0","proxy_pts":"0"}},
         "end":{"seconds":S.rational(1),"anchor":{"frame_index":"1","source_pts":"1","proxy_pts":"1"}},
         "track":"镜头内容","tags":["实拍","中文"],"review_status":"unreviewed","attributes":{},"evidence_ids":[],
         "provenance":{"origin":"manual","actor_id":"local-user","job_id":None}}
    task={"id":"tsk-source","media_id":media["id"],"template_id":template["id"],"template_version":1,
          "revision":1,"annotations":[ann],"extensions":{},"legacy_document_metadata":{"original":{"full":"细节"}}}
    task["content_sha256"]=P.task_content_digest(task,media,template,[artifact])
    return task,media,template,[artifact]


def report(media, *, subject=None, scope="stream"):
    r={"id":"analysis-loudness","kind":"loudness","scope":scope,"file_sha256":media["source"]["sha256"],
       "stream_index":0,"time_domain":"admitted_source_seconds","window":None if scope!="interval" else {"start":S.rational(0),"end":S.rational(1)},
       "method":{"id":"ffmpeg-ebur128","version":"test-v1","parameters":{"window_ms":400}},
       "status":"available","values":[{"name":"integrated_loudness","value":-9.0,"unit":"LUFS","basis":"measured","status":"available","reason":None}],
       "evidence_ids":[],"subject_annotation_id":subject}
    r["input_digest"]=P.analysis_input_digest(r)
    return r


def document():
    t,m,tp,arts=fixture()
    return E.build_document(t,m,tp,artifacts=arts)


def resign(doc, *, task=False):
    if task:
        doc["task"]["content_sha256"]=P.task_content_digest(doc["task"],doc["media"],doc["template"],doc["artifacts"])
        doc["binding"]["task_content_sha256"]=doc["task"]["content_sha256"]
    doc["content_sha256"]=P.document_digest(doc)
    return doc


def test_complete_document_is_deterministic_and_does_not_mutate_input():
    t,m,tp,arts=fixture();before=copy.deepcopy((t,m,tp,arts))
    d=E.build_document(t,m,tp,artifacts=arts)
    assert d["task"]==t and (t,m,tp,arts)==before
    assert P.decode_document(S.canonical_json(d))==d
    assert E.build_document(t,m,tp,artifacts=arts)==d
    assert d["task"]["annotations"][0]["text"].endswith("不截句。")


def test_events_whole_clip_and_exact_anchors_preserved():
    t,m,tp,arts=fixture()
    for kind,fields in [("event",{"at":{"seconds":S.rational("3/2")}}),("whole_clip",{})]:
        a=copy.deepcopy(t["annotations"][0]);a.update(id="ann-"+kind,type=kind)
        del a["start"],a["end"];a.update(fields);t["annotations"].append(a)
    t["content_sha256"]=P.task_content_digest(t,m,tp,arts)
    d=E.build_document(t,m,tp,artifacts=arts)
    assert [a["type"] for a in d["task"]["annotations"]]==["segment","event","whole_clip"]
    assert d["task"]["annotations"][0]["end"]["anchor"]["frame_index"]=="1"


def test_import_new_ids_review_downgrade_and_original_full_envelope():
    d=document();a=d["task"]["annotations"][0];a["review_status"]="reviewed"
    subject={key:value for key,value in a.items() if key not in ("review_status","review_binding")}
    review_digest=S.sha256_text({"annotation":subject,"media_id":d["task"]["media_id"],"template_id":d["task"]["template_id"],"template_version":d["task"]["template_version"]})
    a["review_binding"]={"content_sha256":review_digest,"actor_id":"local-user","command_id":"cmd-review","reviewed_at":"2026-11-01T00:00:00Z","task_revision":1}
    resign(d,task=True)
    target=copy.deepcopy(d["media"]);target["id"]="med-local";target["source"]["display_name"]="same-bytes-other-name.mp4"
    payload=E.import_payload(d,target,"tsk-destination")
    new=payload["annotations"][0]
    assert new["id"]!=a["id"] and new["legacy_id"]==a["id"]
    assert new["review_status"]=="unreviewed" and "review_binding" not in new
    assert new["provenance"]=={"origin":"imported","actor_id":"local-user","job_id":None}
    assert new["start"]==a["start"] and new["text"]==a["text"]
    assert payload["extensions"]["professional"]["import_source"]["document"]==d
    assert E.import_payload(d,target,"tsk-destination")==payload


def test_history_is_passive_and_preserved_in_full():
    t,m,tp,arts=fixture()
    history={"scope":"available_local_history","revisions":[copy.deepcopy(t)],"commits":[{"operation":"add_annotation"}],
             "receipts":[{"command_id":"cmd-prior","state":"committed","response":{"revision":1}}],
             "source_envelopes":[{"untrusted_path":"C:/never-read.mp4","instructions":"passive data"}]}
    d=E.build_document(t,m,tp,history=history,artifacts=arts)
    assert d["history"]==history
    assert E.import_payload(d,m,"tsk-next")["extensions"]["professional"]["import_source"]["document"]["history"]==history


def test_report_dependency_and_boundary_freshness_are_independent_of_labels():
    t,m,tp,arts=fixture();r=report(m,subject="ann-original",scope="interval")
    t["extensions"]={"professional":{"analysis_reports":[r]}}
    t["content_sha256"]=P.task_content_digest(t,m,tp,arts)
    d=E.build_document(t,m,tp,artifacts=arts)
    assert d["analysis_assessment"]==[{"id":r["id"],"status":"fresh","reason":None}]
    changed=copy.deepcopy(t);changed["annotations"][0]["label"]="changed-label"
    assert P.derived_analysis_status(changed,m,r)["status"]=="fresh"
    changed["annotations"][0]["end"]={"seconds":S.rational("3/2")}
    assert P.derived_analysis_status(changed,m,r)["reason"]=="subject_interval_changed"
    assert P.analysis_input_digest(r)==r["input_digest"]
    fixed=report(m,scope="interval")
    assert P.derived_analysis_status(changed,m,fixed)["status"]=="fresh"
    payload=E.import_payload(d,m,"tsk-new")
    imported=payload["extensions"]["professional"]["analysis_reports"][0]
    assert imported["subject_annotation_id"]==payload["annotations"][0]["id"]
    assert imported["input_digest"]==P.analysis_input_digest(imported)


@pytest.mark.parametrize("change",[
 lambda d:d.update(unknown=True),lambda d:d.update(schema_version=2),
 lambda d:d["task"]["annotations"][0].update(type="object"),
 lambda d:d["task"]["annotations"][0]["start"].update(seconds={"n":"0","d":"2"}),
 lambda d:d["task"]["annotations"][0]["end"]["anchor"].update(frame_index="2"),
 lambda d:d["task"]["annotations"][0]["end"]["anchor"].update(proxy_pts="0"),
 lambda d:d["task"]["annotations"][0].update(at={"seconds":S.rational(0)}),
 lambda d:d["task"]["annotations"][0].update(attributes={"hidden":1}),
 lambda d:d["task"]["annotations"][0].update(evidence_ids=["ev-missing"]),
 lambda d:d["media"]["source"].update(size_bytes="-1"),
 lambda d:d["artifacts"][0].update(relative_path="C:/outside"),
 lambda d:d["template"].update(annotation_types=[{}]),
 lambda d:d["task"]["annotations"][0].update(label=""),
 lambda d:d["task"].update(content_sha256="0"*64),
 lambda d:d["task"]["annotations"].append(copy.deepcopy(d["task"]["annotations"][0])),
])
def test_invalid_document_fields_are_rejected(change):
    d=document();change(d);resign(d)
    with pytest.raises(SchemaInvalid):P.validate_document(d)


@pytest.mark.parametrize("bad",[b'{"kind":1,"kind":2}',b'{"value":NaN}',b'\xff',b'[]'])
def test_strict_json_rejects_ambiguous_input(bad):
    with pytest.raises(SchemaInvalid):P.decode_document(bad)


@pytest.mark.parametrize("change",[
 lambda r:r.update(file_sha256="9"*64),lambda r:r.update(scope="file",window={"start":S.rational(0),"end":S.rational(1)}),
 lambda r:r.update(subject_annotation_id="ann-original"),lambda r:r["values"][0].update(value=None),
 lambda r:r["values"][0].update(status="unavailable",value=0,reason="no audio"),
 lambda r:r["values"][0].update(unit="volume-percent"),lambda r:r["values"][0].update(basis="inferred"),
 lambda r:r.update(status="not_performed"),lambda r:r.update(input_digest="9"*64),
 lambda r:r.update(status=[]),lambda r:r.update(stream_index=True),
])
def test_invalid_analysis_is_rejected(change):
    _,m,_,_=fixture();r=report(m);change(r)
    with pytest.raises(SchemaInvalid):P.validate_analysis(r,m)


def test_positive_decoded_peaks_are_not_clipped_or_fabricated():
    _,m,_,_=fixture();r=report(m)
    r["values"][0].update(name="sample_peak",value=2.007,unit="dBFS")
    assert P.validate_analysis(r,m)["values"][0]["value"]==2.007
    r["values"][0].update(value=None,status="unavailable",reason="no audio")
    r["status"]="unavailable"
    assert P.validate_analysis(r,m)["values"][0]["value"] is None


def test_export_rejects_divergent_authoritative_reports():
    t,m,tp,arts=fixture();r=report(m)
    with pytest.raises(SchemaInvalid):E.build_document(t,m,tp,artifacts=arts,analyses=[r])


def test_import_rejects_wrong_media_and_timeline():
    d=document();target=copy.deepcopy(d["media"]);target["source"]["sha256"]="9"*64
    with pytest.raises(SchemaInvalid,match="matching admitted source"):E.import_payload(d,target,"tsk-new")
    target=copy.deepcopy(d["media"]);target["duration"]=S.rational(3)
    with pytest.raises(SchemaInvalid,match="timeline"):E.import_payload(d,target,"tsk-new")


def test_hash_and_depth_budgets_reject_before_import():
    d=document();d["task"]["annotations"][0]["text"]="tampered"
    with pytest.raises(SchemaInvalid,match="hash mismatch"):P.validate_document(d)
    nested={};node=nested
    for _ in range(P.MAX_DEPTH+1):node["child"]={};node=node["child"]
    with pytest.raises(SchemaInvalid,match="nesting budget"):P.bounded_json(nested)
    d=document();d["task"]["annotations"][0]["text"]="x"*(P.MAX_TEXT_CHARS+1)
    with pytest.raises(SchemaInvalid,match="bound"):P.validate_document(d)


def test_observed_voice_report_requires_explicit_actor_and_input_declaration():
    _,m,_,_=fixture();r=report(m)
    r["kind"]="audio_observation"
    r["values"][0].update(name="timbre_description",value="Example observation only",unit="none",basis="observed")
    r["input_digest"]=P.analysis_input_digest(r)
    with pytest.raises(SchemaInvalid):P.validate_analysis(r,m)
    r["method"]["parameters"]["observation_source"]={"actor_id":"assistant-reviewer","actor_type":"assistant","input_file_sha256":r["file_sha256"]}
    r["input_digest"]=P.analysis_input_digest(r)
    assert P.validate_analysis(r,m) is r
    r["values"][0]["basis"]="measured"
    with pytest.raises(SchemaInvalid):P.validate_analysis(r,m)


def test_byte_limit_and_frame_map_manifest_are_explicit():
    with pytest.raises(SchemaInvalid):P.decode_document(b" "*(P.MAX_DOCUMENT_BYTES+1))
    d=document();d["artifacts"]=[];resign(d)
    with pytest.raises(SchemaInvalid,match="Frame-map"):P.validate_document(d)


def test_current_review_binding_cannot_be_forged_by_resigning_outer_document():
    d=document();a=d["task"]["annotations"][0]
    a["review_status"]="reviewed"
    a["review_binding"]={"content_sha256":"4"*64,"actor_id":"local-user","command_id":"cmd-review","reviewed_at":"2026-11-01T00:00:00Z","task_revision":1}
    resign(d,task=True)
    with pytest.raises(SchemaInvalid,match="Review binding"):P.validate_document(d)
