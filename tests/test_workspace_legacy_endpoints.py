"""A rounded legacy endpoint is not an arbitrary clipping authorization."""
import copy

import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import SchemaInvalid
from trace_annotate.workspace.legacy_adapter import temporal_import_payload, validate_temporal_import_payload


def fixture():
    media = {"duration": S.rational((2, 3)), "source": {"sha256": "a" * 64, "size_bytes": "12"}}
    doc = {"kind": "temporal", "schema_version": 2, "clip": "real.mp4", "duration": 0.666667,
           "media_sha256": "a" * 64, "media_size": 12,
           "segments": [{"id": "ann-final", "start": 0, "end": 0.666667, "label": "final",
                         "text": "", "track": "main", "tags": [], "review_status": "unreviewed"}]}
    return doc, media


def test_exact_media_endpoint_preserves_original_rounding_provenance():
    doc, media = fixture()
    original = copy.deepcopy(doc)
    payload = temporal_import_payload(doc, media)
    assert payload["annotations"][0]["end"]["seconds"] == S.rational((2, 3))
    record = payload["legacy_document_envelope"]["endpoint_adjustments"][0]
    assert record["original"] == 0.666667
    assert record["mapped_seconds"] == S.rational((2, 3))
    validate_temporal_import_payload(payload, media)
    assert doc == original


@pytest.mark.parametrize("change", ["large", "internal", "hash", "size", "unbound"])
def test_other_outside_endpoints_remain_rejected(change):
    doc, media = fixture()
    if change == "large":
        doc["duration"] = doc["segments"][0]["end"] = .666668
    elif change == "internal":
        doc["duration"] = .7
    elif change == "hash":
        doc["media_sha256"] = "b" * 64
    elif change == "size":
        doc["media_size"] = 13
    else:
        doc["schema_version"] = 1
        del doc["media_sha256"]
    with pytest.raises(SchemaInvalid):
        temporal_import_payload(doc, media)


def test_forged_adjustment_cannot_claim_arbitrary_clipping():
    doc, media = fixture()
    payload = temporal_import_payload(doc, media)
    payload["legacy_document_envelope"]["endpoint_adjustments"][0]["original"] = .9
    with pytest.raises(SchemaInvalid):
        validate_temporal_import_payload(payload, media)


def test_workspace_import_refuses_unbound_v1_even_when_times_fit():
    doc, media = fixture()
    doc["schema_version"] = 1
    del doc["media_sha256"]
    doc["duration"] = .6
    doc["segments"][0]["end"] = .6
    with pytest.raises(SchemaInvalid) as failure:
        temporal_import_payload(doc, media)
    assert failure.value.code == "MEDIA_BINDING_REQUIRED"
