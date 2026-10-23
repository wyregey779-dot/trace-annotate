"""General-video structure, strict loading and real format round trips."""
import copy
import csv
import io
import json

import pytest

from trace_annotate.temporal.formats import (
    export_csv, export_subtitles, import_subtitles, load_document,
)
from trace_annotate.temporal.schema import require_valid_document, validate_document


def _segment(segment_id="event-1", **updates):
    segment = {"id": segment_id, "start": 1.25, "end": 3.5, "label": "人物进入",
               "text": "一位行人进入画面。", "tags": ["街景", "人工"],
               "track": "events", "review_status": "unreviewed"}
    segment.update(updates)
    return segment


def _doc(**updates):
    doc = {"kind": "temporal", "schema_version": 1, "clip": "街景.mp4",
           "duration": 65.5, "media_size": 1234, "segments": [_segment()],
           "metadata": {"purpose": "general video", "nested": [None, True, 0.5]}}
    doc.update(updates)
    return doc


def test_general_events_allow_overlaps_without_embodied_fields_or_mutation():
    doc = _doc(segments=[_segment(), _segment("event-2", start=2, end=6),
                         _segment("event-3", start=2.5, end=7, track="sounds")])
    original = copy.deepcopy(doc)
    assert validate_document(doc) == []
    require_valid_document(doc)
    assert doc == original
    assert not any(key in doc for key in ("working", "idle", "clip_status"))


@pytest.mark.parametrize("bad", [None, [], "annotation", 1, True, {},
    {"kind": [], "schema_version": [], "clip": [], "duration": [], "segments": [None]}])
def test_malformed_documents_report_violations_without_crashing(bad):
    assert validate_document(bad)
    with pytest.raises(ValueError, match="Invalid temporal annotation"):
        require_valid_document(bad)


@pytest.mark.parametrize("field,value", [
    ("kind", "embodied"), ("schema_version", True), ("schema_version", 1.0),
    ("schema_version", 2), ("clip", ""), ("clip", "  "), ("clip", "../clip.mp4"),
    ("clip", r"folder\clip.mp4"), ("clip", "C:clip.mp4"), ("clip", "a:b.mp4"),
    ("clip", "."), ("clip", ".."), ("clip", "clip\x00.mp4"),
    ("duration", True), ("duration", "65.5"), ("duration", 0),
    ("duration", float("nan")), ("duration", float("inf")), ("duration", 10 ** 400),
    ("duration", 9007199254740992),
    ("media_size", True), ("media_size", -1), ("media_size", 1.5),
    ("media_size", 9007199254740992), ("segments", {}), ("metadata", []),
])
def test_invalid_document_fields_are_rejected(field, value):
    assert any(field in error for error in validate_document(_doc(**{field: value})))


@pytest.mark.parametrize("field,value", [
    ("id", " "), ("id", []), ("label", ""), ("label", None),
    ("text", None), ("tags", "person"), ("tags", [""]), ("tags", [1]),
    ("track", ""), ("review_status", []), ("review_status", "correct"),
    ("start", True), ("start", "1.25"), ("start", -1), ("start", float("nan")),
    ("end", float("inf")), ("end", 100), ("end", 1.25), ("end", 0),
])
def test_invalid_segment_fields_are_rejected(field, value):
    assert validate_document(_doc(segments=[_segment(**{field: value})]))


def test_unknown_fields_and_duplicate_ids_are_not_silently_discarded():
    doc = _doc(unrecognized="keep this")
    doc["segments"][0]["custom"] = {"source": "manual"}
    doc["segments"].append(_segment())
    errors = "\n".join(validate_document(doc))
    assert "unrecognized" in errors and "custom" in errors and "unique" in errors
    for field in list(_segment()):
        segment = _segment()
        del segment[field]
        assert any(field in error for error in validate_document(_doc(segments=[segment])))


def test_metadata_must_be_finite_json_and_cycles_do_not_crash_validator():
    for metadata in ({"number": float("nan")}, {1: "not a string key"},
                     {"set": {1, 2}}, {"tuple": (1, 2)}):
        assert validate_document(_doc(metadata=metadata))
    recursive = {}
    recursive["self"] = recursive
    assert any("circular" in error for error in validate_document(_doc(metadata=recursive)))


def test_json_load_retains_every_known_field_and_does_not_rewrite_input(tmp_path):
    path = tmp_path / "annotations.json"
    original = json.dumps(_doc(), ensure_ascii=False, indent=2)
    path.write_text("\ufeff" + original, encoding="utf-8")
    before = path.read_bytes()
    assert load_document(path) == _doc()
    assert path.read_bytes() == before


@pytest.mark.parametrize("content, message", [
    ('{"kind":"temporal","kind":"embodied"}', "duplicate JSON key"),
    ('{"metadata":{"note":1,"note":2}}', "duplicate JSON key"),
    ('{"duration":NaN}', "nonfinite JSON"),
    ('{"duration":Infinity}', "nonfinite JSON"),
    ('{"duration":1e999}', "9007199254740991"),
    ('[]', "JSON object"), ('{"kind":', "Expecting value"),
])
def test_json_load_rejects_ambiguous_or_nonfinite_input(tmp_path, content, message):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_document(path)


@pytest.mark.parametrize("lexeme", [
    "9007199254740991.1", "9.0071992547409911e15",
    "-9007199254740991.1", "-9.0071992547409911e15",
    "9007199254740991.00000000000000000000000001", "1e99999999999999999999",
])
def test_json_float_range_is_checked_before_rounding_can_hide_overflow(tmp_path, lexeme):
    path = tmp_path / "outside-range.json"
    content = json.dumps(_doc(metadata={"numeric": "NUMBER_TOKEN"}))
    path.write_text(content.replace('"NUMBER_TOKEN"', lexeme), encoding="utf-8")
    with pytest.raises(ValueError, match="9007199254740991"):
        load_document(path)


def test_json_float_loader_retains_ordinary_ieee754_and_exact_safe_boundary(tmp_path):
    path = tmp_path / "valid-floats.json"
    content = json.dumps(_doc(metadata={"numeric": 0.5005, "boundary": "NUMBER_TOKEN"}))
    path.write_text(content.replace('"NUMBER_TOKEN"', "9007199254740991.0"), encoding="utf-8")
    loaded = load_document(path)
    assert loaded["metadata"]["numeric"] == 0.5005
    assert isinstance(loaded["metadata"]["numeric"], float)
    assert loaded["metadata"]["boundary"] == 9007199254740991


@pytest.mark.parametrize("field", ["schema_version", "media_size"])
def test_json_float_loader_does_not_coerce_integer_only_fields(tmp_path, field):
    path = tmp_path / "float-in-integer-field.json"
    doc = _doc()
    doc[field] = float(doc[field])
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_document(path)


def test_csv_quotes_unicode_multiline_and_json_tags_instead_of_dropping_values():
    doc = _doc(segments=[_segment(text='第一行, "quoted"\n第二行')])
    rows = list(csv.DictReader(io.StringIO(export_csv(doc))))
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == ["id", "start", "end", "label", "text", "tags", "track", "review_status"]
    assert row["text"] == doc["segments"][0]["text"]
    assert json.loads(row["tags"]) == ["街景", "人工"]
    assert float(row["start"]) == 1.25
    assert "metadata" not in row


@pytest.mark.parametrize("payload", ["=1+1", " +SUM(1,2)", "-2+3", "@cmd", "\ttext", "\rtext", "\ntext", " \t=cmd"])
def test_csv_neutralizes_formula_like_text_without_mutating_json(payload):
    doc = _doc(segments=[_segment(segment_id=payload, label=payload, text=payload, track=payload)])
    row = next(csv.DictReader(io.StringIO(export_csv(doc))))
    for field in ("id", "label", "text", "track"):
        assert row[field] == "'" + payload
        assert doc["segments"][0][field] == payload


@pytest.mark.parametrize("fmt,delimiter", [("srt", ","), ("vtt", ".")])
def test_subtitle_rounding_carries_minutes_hours_and_rounds_half_up(fmt, delimiter):
    doc = _doc(duration=4000, segments=[
        _segment("late", start=3599.9999, end=3601, text="跨小时"),
        _segment("early", start=59.9999, end=61.0005, text="跨分钟"),
    ])
    subtitles = export_subtitles(doc, fmt)
    assert f"00:01:00{delimiter}000 --> 00:01:01{delimiter}001" in subtitles
    assert f"01:00:00{delimiter}000 --> 01:00:01{delimiter}000" in subtitles
    assert "1000" not in subtitles
    imported = import_subtitles(subtitles, fmt, doc["clip"], doc["duration"])
    assert [segment["start"] for segment in imported["segments"]] == [60, 3600]
    assert [segment["text"] for segment in imported["segments"]] == ["跨分钟", "跨小时"]
    assert all(segment["review_status"] == "unreviewed" for segment in imported["segments"])


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_plain_multiline_subtitles_round_trip_without_changing_input(fmt):
    doc = _doc(segments=[_segment(text="第一行\n第二行"),
                         _segment("event-2", start=2, end=5, text="同时发生另一事件")])
    original = copy.deepcopy(doc)
    imported = import_subtitles(export_subtitles(doc, fmt), fmt, doc["clip"], doc["duration"])
    assert [(s["start"], s["end"], s["text"]) for s in imported["segments"]] == [
        (s["start"], s["end"], s["text"]) for s in doc["segments"]]
    assert all(s["label"] == "字幕" and s["track"] == "subtitles" for s in imported["segments"])
    assert doc == original


def test_webvtt_escapes_literal_markup_and_ampersands_and_recovers_literal_text():
    text = '<script>alert("x")</script> & <b>非样式</b>'
    doc = _doc(segments=[_segment(text=text)])
    subtitles = export_subtitles(doc, "vtt")
    assert "<script>" not in subtitles and "&lt;script&gt;" in subtitles and "&amp;" in subtitles
    imported = import_subtitles(subtitles, "vtt", doc["clip"], doc["duration"])
    assert imported["segments"][0]["text"] == text
    with pytest.raises(ValueError, match="markup"):
        export_subtitles(doc, "srt")


@pytest.mark.parametrize("text", [r"{\b1}bold{\b0}", r"{\an8}position",
                                   r"first\Nsecond", r"first\nsecond", r"space\hhere"])
def test_srt_rejects_ass_syntax_instead_of_implicitly_styling_or_changing_text(text):
    doc = _doc(segments=[_segment(text=text)])
    with pytest.raises(ValueError, match="ASS overrides/control escapes.*WebVTT or JSON"):
        export_subtitles(doc, "srt")
    content = "1\n00:00:01,250 --> 00:00:03,500\n" + text + "\n"
    with pytest.raises(ValueError, match="ASS overrides/control escapes"):
        import_subtitles(content, "srt", doc["clip"], doc["duration"])
    assert doc["segments"][0]["text"] == text
    webvtt = export_subtitles(doc, "vtt")
    imported = import_subtitles(webvtt, "vtt", doc["clip"], doc["duration"])
    assert imported["segments"][0]["text"] == text


def test_srt_preserves_ordinary_braces_and_noncontrol_backslashes():
    text = r"set {1, 2} and left\right"
    doc = _doc(segments=[_segment(text=text)])
    imported = import_subtitles(export_subtitles(doc), "srt", doc["clip"], doc["duration"])
    assert imported["segments"][0]["text"] == text


def test_subtitles_require_explicit_track_and_do_not_conflate_concurrent_labels():
    doc = _doc(segments=[_segment(), _segment("sound-1", track="sound", text="门铃")])
    with pytest.raises(ValueError, match="explicit track"):
        export_subtitles(doc)
    assert "门铃" not in export_subtitles(doc, track="events")
    assert "门铃" in export_subtitles(doc, track="sound")
    with pytest.raises(ValueError, match="does not exist"):
        export_subtitles(doc, track="unknown")


@pytest.mark.parametrize("text", ["\nleading", "trailing\n", "a\n\nb", "a\n \nb", "a --> b", "tab\ttext", "null\x00text"])
@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_ambiguous_subtitle_text_is_rejected_instead_of_changing_cue_structure(text, fmt):
    with pytest.raises(ValueError):
        export_subtitles(_doc(segments=[_segment(text=text)]), fmt)


def test_label_is_subtitle_fallback_but_review_state_and_tags_are_not_fabricated():
    doc = _doc(segments=[_segment(text="", label="无对白场景")])
    imported = import_subtitles(export_subtitles(doc), "srt", doc["clip"], doc["duration"])
    assert imported["segments"][0]["text"] == "无对白场景"
    assert imported["segments"][0]["tags"] == []
    assert "fps" not in imported and "working" not in imported


def test_submillisecond_interval_is_rejected_instead_of_becoming_zero_length():
    with pytest.raises(ValueError, match="collapses"):
        export_subtitles(_doc(segments=[_segment(start=1.0001, end=1.0002)]))


def test_rounded_duration_overflow_is_not_silently_clipped_on_import():
    doc = _doc(duration=1.2345, segments=[_segment(start=0, end=1.2345)])
    subtitles = export_subtitles(doc)
    assert "00:00:01,235" in subtitles
    with pytest.raises(ValueError, match="duration"):
        import_subtitles(subtitles, "srt", doc["clip"], doc["duration"])


@pytest.mark.parametrize("fmt,content", [
    ("srt", "1\n00:00:02,000 --> 00:00:01,000\nbackwards"),
    ("srt", "1\n00:00:00,000 --> 00:00:01,000\n"),
    ("srt", "1\n00:00:00,000 --> 00:00:01,000 align:start\nsettings"),
    ("srt", "1\n00:00:00,000 --> 00:00:01,000\n<b>markup</b>"),
    ("srt", "1\n00:00:00,000 --> 00:00:01,000\na\n\n1\n00:00:01,000 --> 00:00:02,000\nb"),
    ("vtt", "WEBVTT\n\n00:00.000 --> 00:01.000 position:10%\nsettings"),
    ("vtt", "WEBVTT\n\nidentifier\n00:00.000 --> 00:01.000\ntext"),
    ("vtt", "WEBVTT\n\nNOTE comment"),
    ("vtt", "WEBVTT\n\nSTYLE\n::cue { color: red; }"),
    ("vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\n<b>markup</b>"),
    ("vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\n&#65;"),
    ("vtt", "WEBVTT header metadata\n\n00:00.000 --> 00:01.000\ntext"),
])
def test_unsupported_or_invalid_subtitle_inputs_fail_explicitly(fmt, content):
    with pytest.raises(ValueError):
        import_subtitles(content, fmt, "clip.mp4", 10)


def test_webvtt_import_supports_crlf_bom_short_timestamps_and_named_text_escapes():
    content = "\ufeffWEBVTT\r\n\r\n00:01.250 --> 00:02.500\r\nA &amp; B &lt; C\r\n"
    doc = import_subtitles(content, "webvtt", "clip.mp4", 10, track="captions")
    assert doc["segments"][0]["start"] == 1.25
    assert doc["segments"][0]["text"] == "A & B < C"
    assert doc["segments"][0]["track"] == "captions"


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_empty_document_has_explicit_empty_projection(fmt):
    doc = _doc(segments=[])
    subtitles = export_subtitles(doc, fmt)
    assert subtitles == ("WEBVTT\n\n" if fmt == "vtt" else "")
    assert import_subtitles(subtitles, fmt, doc["clip"], doc["duration"])["segments"] == []


@pytest.mark.parametrize("fmt", [None, "csv", "json", 1])
def test_unknown_subtitle_format_is_rejected(fmt):
    with pytest.raises(ValueError, match="format"):
        export_subtitles(_doc(), fmt)


def test_numbers_outside_javascript_safe_range_are_rejected_before_export():
    doc = _doc(duration=1e100, segments=[_segment(start=1e98, end=1e99)])
    with pytest.raises(ValueError, match="9007199254740991"):
        export_subtitles(doc)
    for number in (9007199254740992, -9007199254740992, 1e100, -1e100):
        assert validate_document(_doc(metadata={"number": number}))
    assert not validate_document(_doc(metadata={"number": 9007199254740991}))
    assert not validate_document(_doc(metadata={"number": -9007199254740991}))


@pytest.mark.parametrize("field", ["clip", "id", "label", "text", "tags", "track"])
@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff", "\ud83d\ude00"])
def test_surrogate_code_units_are_rejected_in_all_canonical_text(field, surrogate):
    doc = _doc()
    if field == "clip":
        doc[field] = surrogate + ".mp4"
    else:
        doc["segments"][0][field] = [surrogate] if field == "tags" else surrogate
    assert any(field in error for error in validate_document(doc))
    with pytest.raises(ValueError, match="Unicode"):
        export_csv(doc)


@pytest.mark.parametrize("metadata", [{"text": "\ud800"}, {"\udfff": "bad key"}, {"nested": ["\ud800"]}])
def test_metadata_unicode_surrogates_are_rejected(metadata):
    assert any("Unicode" in error for error in validate_document(_doc(metadata=metadata)))


def test_json_unicode_emoji_pairs_decode_but_lone_surrogates_fail(tmp_path):
    path = tmp_path / "unicode.json"
    doc = _doc()
    doc["segments"][0]["text"] = "emoji 😀"
    path.write_text(json.dumps(doc, ensure_ascii=True), encoding="utf-8")
    assert load_document(path)["segments"][0]["text"] == "emoji 😀"
    doc["segments"][0]["text"] = "\ud800"
    path.write_text(json.dumps(doc, ensure_ascii=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Unicode"):
        load_document(path)


def test_extreme_subtitle_timestamps_are_rejected_as_value_errors():
    content = "1\n999999999999999999999:00:00,000 --> 999999999999999999999:00:01,000\ntext"
    with pytest.raises(ValueError, match="9007199254740991"):
        import_subtitles(content, "srt", "clip.mp4", 10)
