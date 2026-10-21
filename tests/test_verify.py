from trace_annotate.core.verify import verify_annotations, EXPECTED_SCHEMA_KEYS


def test_verification_suite():
    # Valid row
    valid_row = {k: "val" for k in EXPECTED_SCHEMA_KEYS}
    valid_row["asset_id"] = "AST_0001"
    valid_row["kind"] = "image"
    valid_row["dhash"] = "0123456789abcdef"
    valid_row["face_status"] = "无人物"
    valid_row["ocr_readable"] = ""
    valid_row["clip_scores"] = {"a": 1.0}
    valid_row["similar_cluster_id"] = "cluster_0001"
    valid_row["usage_blacklist"] = False
    valid_row["entity_status"] = "confirmed"
    valid_row["scene_sub_segments"] = None
    valid_row["fill"] = {"brightness": "auto_metric"}

    rep = verify_annotations([valid_row])
    assert rep.passed is True
    assert rep.errors == []

    # Invalid row: missing asset_id
    bad_row = dict(valid_row)
    del bad_row["asset_id"]
    rep2 = verify_annotations([bad_row])
    assert rep2.passed is False
    assert any("asset_id" in e for e in rep2.errors)
