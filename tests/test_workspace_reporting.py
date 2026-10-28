"""Read-only task report and explicit-loss export routes (G01/G09 surfaces).

These routes must describe structure only, never certify semantic review, and
must carry the binding revision/content digest so a stale export is detectable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import csv
import hashlib
import io
import json
import threading
from urllib.parse import quote

import pytest

from tests.test_workspace_server import api, assert_error, command, http_video, request
from trace_annotate.temporal.formats import import_subtitles
from trace_annotate.workspace import reporting
from trace_annotate.workspace.schema import rational


def _task_path(store, suffix):
    task = store.get_task(store.list_tasks()[0]["task_id"])
    return f"/api/v1/projects/current/tasks/{task['id']}/{suffix}"


def _save(server, store, *, ident="ann-first", start=0, end=1, **changes):
    path, body = command(store, "cmd-" + ident)
    annotation = body["payload"]["annotation"]
    annotation.update(id=ident, start={"seconds": rational(start)}, end={"seconds": rational(end)}, **changes)
    if annotation["type"] == "event":
        del annotation["start"], annotation["end"]
    if annotation["type"] == "whole_clip":
        del annotation["start"], annotation["end"]
    status, _, receipt = request(server, "POST", path, body)
    assert status == 200, receipt
    return receipt


def _fraction(value):
    # Derived report fractions deliberately have a larger budget than stored
    # inputs, so do not parse them through the input's 40-digit validator.
    return Fraction(int(value["n"]), int(value["d"]))


def _csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline="")))


def test_report_on_empty_task_reports_structure_only(api):
    server, store = api
    status, headers, body = request(server, "GET", _task_path(store, "report"))
    assert status == 200, body
    assert body["kind"] == "traceannotate.task-structural-report"
    assert body["scope"] == "structure_and_video_interval_statistics_only"
    assert body["semantic_review"] == "not_performed"
    assert body["checks"]["status"] == "passed"
    assert body["counts"]["annotations"] == 0
    # No track is invented for an empty task, and no Idle labels are inferred.
    assert body["tracks"] == []
    assert any("Gaps mean unannotated" in item for item in body["limitations"])
    assert body["binding"]["task_id"] == store.list_tasks()[0]["task_id"]
    assert headers["Cache-Control"] == "no-store"


def test_report_after_commit_counts_segment_and_coverage(api):
    server, store = api
    path, body = command(store)
    assert request(server, "POST", path, body)[0] == 200
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200
    assert report["counts"]["segments"] == 1
    track = report["tracks"][0]
    assert track["segment_count"] == 1
    # The committed segment spans [0,1) on the active clock.
    assert track["covered_seconds"] == {"n": "1", "d": "1"}
    assert track["uncovered_seconds"] == {"n": "1", "d": "1"}
    assert track["overlap_seconds"] == {"n": "0", "d": "1"}
    assert track["statistics_status"] == "complete"


def test_export_json_carries_binding_and_loss_disclosure(api):
    server, store = api
    path, body = command(store)
    request(server, "POST", path, body)
    status, headers, payload = request(server, "GET", _task_path(store, "export?format=json"))
    assert status == 200
    task = store.get_task(store.list_tasks()[0]["task_id"])
    assert headers["X-TraceAnnotate-Task-Revision"] == str(task["revision"])
    assert headers["X-TraceAnnotate-Content-SHA256"] == task["content_sha256"]
    assert "No media bytes" in headers["X-TraceAnnotate-Export-Losses"]
    assert payload["kind"] == "traceannotate.task-snapshot"
    assert payload["binding"]["task_id"] == task["id"]
    assert payload["task"]["annotations"][0]["label"] == "working"


def test_export_csv_and_srt_roundtrip_one_segment(api):
    server, store = api
    path, body = command(store)
    request(server, "POST", path, body)
    status, headers, csv_body = request(server, "GET", _task_path(store, "export?format=csv"))
    assert status == 200 and "text/csv" in headers["Content-Type"]
    rows = _csv_rows(csv_body)
    task = store.get_task(store.list_tasks()[0]["task_id"])
    assert len(rows) == 1
    assert json.loads(rows[0]["annotation_json"]) == task["annotations"][0]
    assert (rows[0]["start_n"], rows[0]["start_d"], rows[0]["end_n"], rows[0]["end_d"]) == ("0", "1", "1", "1")
    status, headers, srt_body = request(server, "GET", _task_path(store, "export?format=srt"))
    assert status == 200
    srt = srt_body.decode("utf-8")
    restored = import_subtitles(srt, "srt", "playback.mp4", 2)
    assert [(item["start"], item["end"], item["text"]) for item in restored["segments"]] == [(0.0, 1.0, "reviewed")]


def test_export_rejects_missing_or_unknown_format_and_unauthorized(api):
    server, store = api
    task_url = "/api/v1/projects/current/tasks/" + store.list_tasks()[0]["task_id"]
    assert_error(request(server, "GET", task_url + "/export"), 400, "INVALID_QUERY")
    assert_error(request(server, "GET", task_url + "/export?format=parquet"),
                 422, "UNSUPPORTED_EXPORT_FORMAT")
    assert_error(request(server, "GET", task_url + "/export?format=json", authorized=False),
                 403, "INVALID_TOKEN")


def test_large_legal_denominators_keep_exact_report_and_original_exports(api):
    server, store = api
    start = Fraction(1, 100000000000000000003)
    end = Fraction(100000000000000000002, 100000000000000000001)
    _save(server, store, start=start, end=end)
    before = store.get_snapshot()
    task = before["tasks"][0]
    expected = end - start
    assert len(str(expected.denominator)) > 40  # Previously broke all three routes.
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200, report
    assert report["statistics_status"] == "complete"
    assert _fraction(report["tracks"][0]["covered_seconds"]) == expected
    assert _fraction(report["tracks"][0]["uncovered_seconds"]) == 2 - expected
    assert _fraction(report["tracks"][0]["covered_intervals"][0]["duration"]) == expected
    status, _, exported = request(server, "GET", _task_path(store, "export?format=json"))
    assert status == 200 and exported["task"] == task
    status, _, payload = request(server, "GET", _task_path(store, "export?format=csv"))
    assert status == 200
    row = _csv_rows(payload)[0]
    assert json.loads(row["annotation_json"]) == task["annotations"][0]
    assert Fraction(int(row["start_n"]), int(row["start_d"])) == start
    assert Fraction(int(row["end_n"]), int(row["end_d"])) == end
    assert store.get_snapshot() == before


def test_statistics_budget_is_explicit_and_does_not_block_raw_exports(api, monkeypatch):
    server, store = api
    _save(server, store, ident="ann-a", end=Fraction(1, 3))
    _save(server, store, ident="ann-b", start=Fraction(2, 3), end=1)
    before = store.get_snapshot()
    monkeypatch.setattr(reporting, "MAX_STATISTICS_SEGMENTS", 1)
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200 and report["statistics_status"] == "budget_exceeded"
    assert report["statistics_budget"]["max_segments"] == 1
    assert report["checks"]["status"] == "passed"
    assert report["counts"]["segments"] == 2
    assert {"code": "STATISTICS_BUDGET_EXCEEDED", "track": "main"} in report["warnings"]
    for field in ("covered_seconds", "uncovered_seconds", "overlap_seconds", "covered_intervals", "gaps", "overlaps"):
        assert report["tracks"][0][field] is None  # Unavailable is never zero/empty evidence.
    status, _, exported = request(server, "GET", _task_path(store, "export?format=json"))
    assert status == 200 and exported["task"] == before["tasks"][0]
    assert exported["report"]["statistics_status"] == "budget_exceeded"
    status, _, payload = request(server, "GET", _task_path(store, "export?format=csv"))
    assert status == 200
    assert [json.loads(row["annotation_json"]) for row in _csv_rows(payload)] == before["tasks"][0]["annotations"]
    assert store.get_snapshot() == before


def test_big_integer_analysis_budget_bounds_fraction_accumulation():
    # Individually valid short fractions can accumulate a huge common
    # denominator. Test the real default limit, without lowering it.
    intervals = [(Fraction(index, 200), Fraction(index, 200) + Fraction(1, 10**20 + index * 2 + 1))
                 for index in range(200)]
    assert all(len(str(value.denominator)) <= 40 for interval in intervals for value in interval)
    with pytest.raises(reporting._StatisticsBudgetExceeded):
        reporting._coverage(intervals, Fraction(0), Fraction(1))


def test_rational_budget_fallback_keeps_original_json_and_csv(api, monkeypatch):
    server, store = api
    _save(server, store, start=Fraction(1, 100000000000000000003),
          end=Fraction(100000000000000000002, 100000000000000000001))
    original = store.get_task(store.list_tasks()[0]["task_id"])
    # Exercise the same fallback for arithmetic growth, not just item count.
    monkeypatch.setattr(reporting, "MAX_STATISTICS_BITS", 120)
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200 and report["statistics_status"] == "budget_exceeded"
    assert report["statistics_budget"]["max_rational_bits"] == 120
    assert report["tracks"][0]["covered_seconds"] is None
    status, _, exported = request(server, "GET", _task_path(store, "export?format=json"))
    assert status == 200 and exported["task"] == original
    status, _, payload = request(server, "GET", _task_path(store, "export?format=csv"))
    assert status == 200
    assert json.loads(_csv_rows(payload)[0]["annotation_json"]) == original["annotations"][0]


def test_fractional_audio_window_overlap_and_annotation_kinds_are_separate(api):
    server, store = api
    # Real admitted test video is [0,2). Exercise nonzero video_start using the
    # pure exact statistics function: audio-only margins are outside coverage.
    statistics = reporting._coverage([(Fraction(0), Fraction(2)), (Fraction(3, 2), Fraction(5, 2)),
                                     (Fraction(3), Fraction(5))], Fraction(1), Fraction(4))
    assert _fraction(statistics["covered_seconds"]) == Fraction(5, 2)
    assert _fraction(statistics["uncovered_seconds"]) == Fraction(1, 2)
    assert _fraction(statistics["overlap_seconds"]) == Fraction(1, 2)
    assert statistics["max_concurrent_segments"] == 2
    assert [(_fraction(item["start"]), _fraction(item["end"])) for item in statistics["gaps"]] == [(Fraction(5, 2), Fraction(3))]
    _save(server, store, ident="ann-a", start=Fraction(1, 3), end=1, tags=["同一标签", "同一标签"])
    _save(server, store, ident="ann-b", start=Fraction(2, 3), end=Fraction(4, 3))
    _save(server, store, ident="ann-e", type="event", at={"seconds": rational(Fraction(3, 2))})
    _save(server, store, ident="ann-w", type="whole_clip")
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200 and report["checks"]["status"] == "passed"
    assert report["counts"] == {"annotations": 4, "valid_annotations": 4, "segments": 2, "events": 1,
                                "whole_clip": 1, "tag_occurrences": 2, "unique_tags": 1}
    track = report["tracks"][0]
    assert _fraction(track["covered_seconds"]) == 1
    assert _fraction(track["overlap_seconds"]) == Fraction(1, 3)
    assert track["event_count"] == 1 and track["whole_clip_count"] == 1


@pytest.mark.parametrize("track", ["track\tA", "track\nA", "track\r\nA", "中文 + & /=😀", "\x00\x7f"])
def test_multitrack_srt_selects_every_accepted_track_exactly(api, track):
    server, store = api
    _save(server, store, ident="ann-selected", start=Fraction(1, 3), end=Fraction(2, 3), track=track, text="只选此轨")
    _save(server, store, ident="ann-other", track="other", text="不得混入")
    url = _task_path(store, "export?format=srt")
    assert_error(request(server, "GET", url), 422, "TRACK_REQUIRED")
    status, headers, payload = request(server, "GET", url + "&track=" + quote(track, safe=""))
    assert status == 200, payload
    assert "round-half-up milliseconds" in headers["X-TraceAnnotate-Export-Losses"]
    restored = import_subtitles(payload.decode("utf-8"), "srt", "playback.mp4", 2)
    assert [(item["start"], item["end"], item["text"]) for item in restored["segments"]] == [(0.333, 0.667, "只选此轨")]
    assert_error(request(server, "GET", url + "&track=missing"), 422, "TRACK_NOT_FOUND")


def test_long_track_exports_by_short_stable_id_over_real_http(api):
    server, store = api
    # This accepted name would exceed even the HTTP server's request-line
    # limit when percent-encoded. Its ID stays a fixed-size query value.
    name = "长轨道" * 3000
    assert len(quote(name, safe="")) > 65536
    _save(server, store, ident="ann-long", start=Fraction(1, 3), end=Fraction(2, 3), track=name, text="长轨道字幕")
    _save(server, store, ident="ann-other", track="other", text="不能混入")
    before = store.get_snapshot()
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200
    selected = next(item for item in report["tracks"] if item["track"] == name)
    expected_id = hashlib.sha256(name.encode("utf-8")).hexdigest()
    assert selected["track_id"] == expected_id
    url = _task_path(store, "export?format=srt&track_id=" + selected["track_id"])
    status, headers, payload = request(server, "GET", url)
    assert status == 200, payload
    restored = import_subtitles(payload.decode("utf-8"), "srt", "playback.mp4", 2)
    assert [(item["start"], item["end"], item["text"]) for item in restored["segments"]] == [(0.333, 0.667, "长轨道字幕")]
    assert headers["X-TraceAnnotate-Content-SHA256"] == report["binding"]["content_sha256"]
    assert headers["X-TraceAnnotate-Task-Revision"] == str(report["binding"]["revision"])
    # The short selector does not relax query size limits or mutate state.
    assert_error(request(server, "GET", _task_path(store, "export?format=srt&track=" + "x" * 4097)), 404, "NOT_FOUND")
    assert store.get_snapshot() == before


def test_track_ids_keep_exact_case_unicode_and_revision_independent_identity(api):
    server, store = api
    names = ["Track", "track", "é", "e\u0301", "轨道\t\n"]
    for index, name in enumerate(names):
        _save(server, store, ident=f"ann-track-{index}", track=name, text=f"cue-{index}")
    before = store.get_snapshot()
    status, _, report = request(server, "GET", _task_path(store, "report"))
    assert status == 200
    ids = {item["track"]: item["track_id"] for item in report["tracks"]}
    assert len(set(ids.values())) == len(names)
    for index, name in enumerate(names):
        assert ids[name] == hashlib.sha256(name.encode("utf-8")).hexdigest()
        status, _, data = request(server, "GET", _task_path(store, "export?format=srt&track_id=" + ids[name]))
        assert status == 200
        restored = import_subtitles(data.decode("utf-8"), "srt", "playback.mp4", 2)
        assert [item["text"] for item in restored["segments"]] == [f"cue-{index}"]
    assert store.get_snapshot() == before
    # IDs identify exact names, not a report revision or annotation position.
    _save(server, store, ident="ann-later", track="Track", text="later")
    status, _, later = request(server, "GET", _task_path(store, "report"))
    assert status == 200 and later["binding"]["revision"] > report["binding"]["revision"]
    assert {item["track"]: item["track_id"] for item in later["tracks"]} == ids


@pytest.mark.parametrize("query,code", [
    ("format=srt&track_id=" + "0" * 64, "TRACK_NOT_FOUND"),
    ("format=srt&track_id=" + "A" * 64, "INVALID_TRACK_ID"),
    ("format=srt&track_id=short", "INVALID_TRACK_ID"),
    ("format=srt&track_id=", "INVALID_TRACK_ID"),
    ("format=srt&track=main&track_id=" + "0" * 64, "TRACK_SELECTION_CONFLICT"),
    ("format=json&track_id=" + "0" * 64, "UNSUPPORTED_EXPORT_OPTION"),
    ("format=csv&track_id=" + "0" * 64, "UNSUPPORTED_EXPORT_OPTION"),
])
def test_track_id_rejects_unknown_ambiguous_or_unsupported_selection(api, query, code):
    server, store = api
    _save(server, store)
    before = store.get_snapshot()
    assert_error(request(server, "GET", _task_path(store, "export?" + query)), 422, code)
    assert store.get_snapshot() == before


def test_srt_half_up_and_collapsed_interval_are_not_silently_changed(api):
    server, store = api
    _save(server, store, ident="ann-half", start=Fraction(1, 2000), end=Fraction(3, 2000), text="半毫秒")
    status, _, data = request(server, "GET", _task_path(store, "export?format=srt"))
    assert status == 200
    restored = import_subtitles(data.decode("utf-8"), "srt", "playback.mp4", 2)
    assert restored["segments"][0]["start"] == 0.001
    assert restored["segments"][0]["end"] == 0.002
    _save(server, store, ident="ann-small", start=Fraction(1, 10000), end=Fraction(2, 10000))
    assert_error(request(server, "GET", _task_path(store, "export?format=srt")), 422, "SUBTITLE_PRECISION_LOSS")
    assert request(server, "GET", _task_path(store, "export?format=json"))[0] == 200


def test_csv_formula_protection_preserves_original_annotation_json(api):
    server, store = api
    _save(server, store, label='=HYPERLINK("https://invalid", "标签")', text="\t+1\r\n第二行😀",
          track=" @track", tags=["重复", "重复", "=1"], start=Fraction(1, 3), end=Fraction(5, 3))
    before = store.get_snapshot()
    status, _, data = request(server, "GET", _task_path(store, "export?format=csv"))
    assert status == 200
    row = _csv_rows(data)[0]
    assert row["label"].startswith("'=") and row["text"].startswith("'\t+") and row["track"].startswith("' @")
    assert json.loads(row["annotation_json"]) == before["tasks"][0]["annotations"][0]
    assert json.loads(row["tags_json"]) == ["重复", "重复", "=1"]
    assert (row["start_n"], row["start_d"], row["end_n"], row["end_d"]) == ("1", "3", "5", "3")
    assert store.get_snapshot() == before


@pytest.mark.parametrize("query", ["format=srt&track=a&track=b", "format=srt&format=json", "format=srt&track=%FF",
                                   "format=srt&track=%GG", "format=srt&track=%ED%A0%80", "format=srt&unknown=x",
                                   "format=srt%0A", "format=srt&track", "format=srt&track_id=a&track_id=b",
                                   "format=srt&track_id=%FF"])
def test_export_query_keeps_strict_framing_and_utf8(api, query):
    server, store = api
    assert_error(request(server, "GET", _task_path(store, "export?" + query)), 400, "INVALID_QUERY")


@pytest.mark.parametrize("suffix", ["report", "export?format=json", "export?format=csv", "export?format=srt"])
def test_readonly_routes_require_token_and_same_origin_and_preserve_state(api, suffix):
    server, store = api
    _save(server, store)
    before = store.get_snapshot()
    url = _task_path(store, suffix)
    assert_error(request(server, "GET", url, authorized=False), 403, "INVALID_TOKEN")
    for headers in ({"Origin": "https://foreign.invalid"}, {"Host": "foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"}):
        assert_error(request(server, "GET", url, headers=headers), 403, "FORBIDDEN_ORIGIN")
    status, headers, payload = request(server, "GET", url)
    assert status == 200
    task = before["tasks"][0]
    if suffix == "report":
        assert (payload["binding"]["revision"], payload["binding"]["content_sha256"]) == (task["revision"], task["content_sha256"])
    else:
        assert headers["X-TraceAnnotate-Task-Revision"] == str(task["revision"])
        assert headers["X-TraceAnnotate-Content-SHA256"] == task["content_sha256"]
    assert store.get_snapshot() == before


def test_export_binding_and_payload_are_one_revision_during_competing_write(api, monkeypatch):
    server, store = api
    _save(server, store)
    before = store.get_task(store.list_tasks()[0]["task_id"])
    export_started, release_export, write_started, write_finished = [threading.Event() for _ in range(4)]
    original = reporting._report

    def hold_report(*args):
        export_started.set()
        assert release_export.wait(timeout=5)
        return original(*args)

    monkeypatch.setattr(reporting, "_report", hold_report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        export = pool.submit(request, server, "GET", _task_path(store, "export?format=json"))
        assert export_started.wait(timeout=5)

        def save():
            write_started.set()
            result = store.apply_task_command(command_id="cmd-competing", task_id=before["id"],
                expected_revision=before["revision"], expected_sha256=before["content_sha256"],
                operation="update_annotation", payload={"annotation_id": "ann-first", "changes": {"label": "new"}}, request_body=b"competing")
            write_finished.set()
            return result

        write = pool.submit(save)
        assert write_started.wait(timeout=5)
        try:
            assert not write_finished.wait(timeout=0.1)
        finally:
            release_export.set()
        status, headers, data = export.result(timeout=5)
        assert status == 200 and data["task"] == before
        assert data["binding"]["content_sha256"] == before["content_sha256"] == headers["X-TraceAnnotate-Content-SHA256"]
        assert write.result(timeout=5)["applied"] is True
    assert store.get_task(before["id"])["revision"] == before["revision"] + 1
