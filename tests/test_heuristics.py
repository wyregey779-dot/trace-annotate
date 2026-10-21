from trace_annotate.heuristics.composite_f1_face import evaluate_f1_face
from trace_annotate.heuristics.composite_f2_entity import evaluate_f2_entity_tier
from trace_annotate.heuristics.composite_f3_screen import evaluate_f3_screen


def test_f1_statue_face():
    # Statue path should decouple real face
    kind, status, rev, reason = evaluate_f1_face(
        asset_rel_path="scenic_spot/毛主席雕像/IMG_01.jpg",
        raw_face_count=1,
        raw_face_status="单人特写"
    )
    assert kind == "statue_face"
    assert status == "无人物"
    assert rev is True


def test_f2_entity_tiering():
    # Conflict between CLIP and path
    status, fill, conflict, note = evaluate_f2_entity_tier(
        primary_entity="火宫殿",
        has_human_override=False,
        path_depth_matched=True,
        clip_top_entity="岳麓书院",
        clip_top_score=0.85,
        path_entity_score=0.20
    )
    assert status == "candidate"
    assert conflict is True
    assert "岳麓书院" in note


def test_f3_screen_disambiguation():
    # Mobile UI words
    screen, rev, ev = evaluate_f3_screen(
        ocr_raw_text="Wi-Fi已连接 正在播放 评论(102)",
        clip_screen_prediction="无屏"
    )
    assert screen == "手机屏"
    assert rev is True

    # Storefront with phone/wechat
    screen2, rev2, ev2 = evaluate_f3_screen(
        ocr_raw_text="老字号面馆 微信扫码支付 欢迎光临",
        clip_screen_prediction="店招"
    )
    assert screen2 == "店招"
