"""Frame identity, mapping and revision boundaries using actual encoded media."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from PIL import Image, ImageDraw
import pytest

from trace_annotate.workspace import schema as S
from trace_annotate.workspace.errors import WorkspaceError
from trace_annotate.workspace.frames import FrameService
from trace_annotate.workspace.store import WorkspaceStore


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("actual ffmpeg/ffprobe needed for frame identity")
    root = tmp_path_factory.mktemp("workspace-frame-identity")
    for index in range(8):
        picture = Image.new("RGB", (160, 96), "#335577")
        draw = ImageDraw.Draw(picture)
        for bit in range(3):
            draw.rectangle((16 + bit * 40, 16, 47 + bit * 40, 47), fill="white" if index & (1 << bit) else "black")
        picture.save(root / f"input-{index:02}.png")
    return root


def make_source(root, name, rate=10, vfr=False, offset=False):
    source = root / f"{name}.mp4"
    args = [shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-n", "-framerate", str(rate), "-i", str(root / "input-%02d.png")]
    if offset:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3"]
    filters = "setsar=1"
    if vfr:
        filters += r",settb=1/1000,setpts=if(lt(N\,4)\,N*100\,400+(N-4)*200)"
    elif offset:
        filters += ",setpts=PTS+1/TB"
    args += ["-vf", filters, "-fps_mode", "passthrough", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p"]
    if vfr:
        args += ["-enc_time_base", "1:1000"]
    if offset:
        args += ["-c:a", "aac"]
    subprocess.run(args + [str(source)], check=True, capture_output=True, timeout=30)
    return source


def pixel_identity(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return sum((1 << bit) for bit in range(3) if sum(rgb.getpixel((32 + bit * 40, 32))) > 3 * 128)


@pytest.fixture(scope="module")
def admitted(encoded):
    source = make_source(encoded, "anchor-vfr", vfr=True)
    store = WorkspaceStore(encoded / "workspace")
    store.create()
    intake = store.import_source(source)
    yield store, intake, source
    store.close()


@pytest.mark.parametrize("rate,vfr,offset", [(8,False,False), (24,False,False), (25,False,False),
    (30,False,False), (60,False,False), (120,False,False), (10,True,False), (10,False,True)])
def test_decoded_first_last_and_nonadjacent_frames_match_actual_pixels(encoded, rate, vfr, offset):
    name = f"pixels-{rate}-{vfr}-{offset}"
    source = make_source(encoded, name, rate, vfr, offset)
    store = WorkspaceStore(encoded / f"workspace-{name}")
    store.create()
    before = S.sha256_bytes(source.read_bytes())
    try:
        admission = store.import_source(source)
        service = FrameService(store)
        page = service.page(admission["media_id"], 0, 100)
        assert page["total"] == "8"
        assert len(page["frames"]) == 8
        if offset:
            assert S.rational_value(page["frames"][0]["seconds"]) == 1
        if vfr:
            assert S.rational_value(page["frames"][5]["seconds"]) == S.rational_value(S.rational("0.6"))
        for index in (0, 7, 1, 6, 3):
            result = service.image(admission["media_id"], index)
            assert pixel_identity(result["path"]) == index
            assert result["row"] == page["frames"][index]
            assert result["sha256"] == S.sha256_bytes(result["path"].read_bytes())
        assert service.image(admission["media_id"], 3)["path"] == result["path"]
        assert S.sha256_bytes(source.read_bytes()) == before
    finally:
        store.close()


def anchor(row):
    return {"seconds": row["seconds"], "anchor": {key: row[key] for key in ("frame_index", "source_pts", "proxy_pts")}}


def save(store, task_id, annotation, command_id="cmd-anchor"):
    task = store.get_task(task_id)
    return store.apply_task_command(command_id=command_id, task_id=task_id,
        expected_revision=task["revision"], expected_sha256=task["content_sha256"],
        operation="add_annotation", payload={"annotation": annotation}, request_body=b"{}")


def test_frame_anchors_survive_close_reopen_and_snapshot_without_time_rounding(admitted):
    store, intake, _ = admitted
    rows = FrameService(store).page(intake["media_id"])["frames"]
    annotation = {"id": "ann-anchor", "type": "segment", "label": "逐帧边界", "text": "\u0000文字\r\n🧪",
                  "start": anchor(rows[3]), "end": anchor(rows[5])}
    receipt = save(store, intake["task_id"], annotation)
    before = store.get_task(intake["task_id"])
    exported = json.loads(store.export_snapshot().read_bytes())
    assert exported["tasks"][0] == before
    store.close()
    store.open()
    assert store.get_task(intake["task_id"]) == before
    assert store.get_command_receipt(receipt["response"]["command_id"])["state"] == "committed"
    assert before["annotations"][0]["start"] == annotation["start"]


@pytest.mark.parametrize("change", ["index", "source_pts", "proxy_pts", "seconds", "extra", "negative", "bool", "phantom_end"])
def test_forged_or_nonexistent_anchor_is_rejected_without_revision_change(admitted, change):
    store, original, source = admitted
    intake = store.import_source(source)
    rows = FrameService(store).page(original["media_id"])["frames"]
    annotation = {"id": "ann-invalid", "type": "segment", "label": "boundary", "start": anchor(rows[1]), "end": anchor(rows[6])}
    annotation = copy.deepcopy(annotation)
    if change == "index": annotation["start"]["anchor"]["frame_index"] = "2"
    elif change in ("source_pts", "proxy_pts"): annotation["start"]["anchor"][change] = "99999"
    elif change == "seconds": annotation["start"]["seconds"] = S.rational("0.11")
    elif change == "extra": annotation["start"]["anchor"]["nominal_fps"] = 10
    elif change == "negative": annotation["start"]["anchor"]["frame_index"] = "-1"
    elif change == "bool": annotation["start"]["anchor"]["frame_index"] = True
    else:
        annotation["end"] = {"seconds": store.get_media(intake["media_id"])["video_end"],
                             "anchor": {"frame_index": "8", "source_pts": "9999", "proxy_pts": "9999"}}
    before = store.get_task(intake["task_id"])
    with pytest.raises(WorkspaceError):
        save(store, intake["task_id"], annotation, "cmd-bad-" + change)
    assert store.get_task(intake["task_id"]) == before


def test_final_endpoint_accepts_plain_seconds_without_phantom_frame(admitted):
    store, _, source = admitted
    intake = store.import_source(source)
    media = store.get_media(intake["media_id"])
    save(store, intake["task_id"], {"id": "ann-last", "type": "segment", "label": "last",
        "start": anchor(FrameService(store).row(media["id"], 7)), "end": {"seconds": media["video_end"]}}, "cmd-last")
    assert store.get_task(intake["task_id"])["revision"] == 1


@pytest.mark.parametrize("offset,limit", [(-1,1), (True,1), (0,False), (0,1001), (9,1)])
def test_invalid_page_does_not_clamp_or_guess_a_frame(admitted, offset, limit):
    store, intake, _ = admitted
    with pytest.raises(WorkspaceError):
        FrameService(store).page(intake["media_id"], offset, limit)


def test_missing_frame_is_not_replaced_with_last_frame(admitted):
    store, intake, _ = admitted
    service = FrameService(store)
    assert service.page(intake["media_id"], 8, 1)["frames"] == []
    with pytest.raises(WorkspaceError) as error:
        service.image(intake["media_id"], 8)
    assert error.value.code == "FRAME_NOT_FOUND"


def test_cached_map_and_image_changes_are_detected(admitted):
    store, intake, _ = admitted
    service = FrameService(store)
    result = service.image(intake["media_id"], 2)
    original = result["path"].read_bytes()
    result["path"].write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(WorkspaceError) as error:
        service.image(intake["media_id"], 2)
    assert error.value.code == "RECOVERY_REQUIRED"
    media = store.get_media(intake["media_id"])
    record = store.conn.execute("SELECT rel_path FROM artifacts WHERE id=?", (media["frame_map_artifact"],)).fetchone()
    path = store.root / record[0]
    original = path.read_bytes()
    try:
        path.write_bytes(original.replace(b'"frame_index":"0"', b'"frame_index":"9"', 1))
        with pytest.raises(WorkspaceError) as error:
            service.row(intake["media_id"], 0)
        assert error.value.code == "RECOVERY_REQUIRED"
    finally:
        path.write_bytes(original)


def test_http_frame_bytes_remain_valid_after_cache_eviction_and_close(admitted, monkeypatch):
    from trace_annotate.workspace import frames
    store, intake, _ = admitted
    monkeypatch.setattr(frames, "MAX_CACHED_IMAGES", 1)
    service = FrameService(store)
    first = service.image(intake["media_id"], 0)
    second = service.image(intake["media_id"], 1)
    assert not first["path"].exists()
    assert S.sha256_bytes(first["data"]) == first["sha256"]
    assert second["path"].exists()
    service.close()
    assert not second["path"].exists()
    assert S.sha256_bytes(second["data"]) == second["sha256"]
    assert (second["path"].parent / "decode.json").exists()


def test_image_waits_for_a_short_concurrent_decode_instead_of_failing(admitted):
    store, intake, _ = admitted
    service = FrameService(store)
    service._decoder.acquire()
    result = {}

    def decode():
        try:
            result.update(service.image(intake["media_id"], 0))
        except BaseException as error:
            result["error"] = error

    worker = threading.Thread(target=decode, daemon=True)
    worker.start()
    time.sleep(0.25)
    assert worker.is_alive(), result.get("error")
    service._decoder.release()
    worker.join(timeout=60)
    assert not worker.is_alive()
    assert "error" not in result
    assert pixel_identity(result["path"]) == 0


def test_inflight_map_change_is_rejected_before_returning_frame(admitted, monkeypatch):
    from trace_annotate.workspace import frames
    store, intake, _ = admitted
    original_rows = frames._Index.rows
    changed = []
    def tamper(index, offset, limit):
        rows = original_rows(index, offset, limit)
        original = index.path.read_bytes()
        changed.append((index.path, original))
        index.path.write_bytes(original.replace(b'"frame_index":"0"', b'"frame_index":"9"', 1))
        return rows
    monkeypatch.setattr(frames._Index, "rows", tamper)
    try:
        with pytest.raises(WorkspaceError) as error:
            FrameService(store).row(intake["media_id"], 0)
        assert error.value.code == "RECOVERY_REQUIRED"
    finally:
        for path, original in changed:
            path.write_bytes(original)


def test_cancelled_admission_preserves_input_and_records_no_task(encoded):
    source = make_source(encoded, "cancel-before")
    original = source.read_bytes()
    store = WorkspaceStore(encoded / "cancelled-workspace")
    store.create()
    try:
        with pytest.raises(WorkspaceError) as error:
            store.import_source(source, cancelled=lambda: True)
        assert error.value.code == "CANCELLED"
        assert store.list_tasks() == []
        intake = store.list_intakes()[0]
        assert intake["state"] == "cancelled" and intake["error_code"] == "CANCELLED"
        assert intake["task_id"] is None and source.read_bytes() == original
    finally:
        store.close()


def test_cancellation_callback_reaches_actual_preparation_before_admission(encoded):
    source = make_source(encoded, "cancel-during")
    original = source.read_bytes()
    store = WorkspaceStore(encoded / "cancel-during-workspace")
    store.create()
    calls = []
    def cancelled():
        calls.append(True)
        return len(calls) > 8
    try:
        with pytest.raises(WorkspaceError) as error:
            store.import_source(source, cancelled=cancelled)
        assert error.value.code == "CANCELLED" and len(calls) > 8
        assert store.list_tasks() == []
        assert store.list_intakes()[0]["state"] == "cancelled"
        assert source.read_bytes() == original
    finally:
        store.close()



def synthetic_time_service(tmp_path, monkeypatch, times, *, origin=7, tail=1):
    """A valid admitted map without decoding a video or loading model runtimes."""
    from fractions import Fraction
    from trace_annotate.workspace import frames

    times = [Fraction(time) for time in times]
    base = Fraction(1, 10 ** 18)
    media = {"id": "med-" + "a" * 64, "frame_count": str(len(times)),
        "proxy_time_base": S.rational(base), "source_time_base": S.rational(base),
        "common_origin": S.rational(origin), "video_start": S.rational(times[0] if times else 0),
        "video_end": S.rational(times[-1] + tail if times else 1), "duration": S.rational(3000)}
    path = tmp_path / "time-map.jsonl"
    rows = [{"frame_index": str(index), "source_pts": str(int((time + origin) / base)),
             "proxy_pts": str(int(time / base)), "seconds": S.rational(time)} for index, time in enumerate(times)]
    path.write_bytes(b"".join(S.canonical_json(row) + b"\n" for row in rows))
    digest = S.sha256_bytes(path.read_bytes())
    monkeypatch.setattr(frames, "_binding", lambda *args: (media, path, digest))
    return frames.FrameService(None), media, path, rows


def test_time_lookup_uses_global_vfr_pts_and_preserves_nonzero_origin(tmp_path, monkeypatch):
    from fractions import Fraction

    times = [1 + Fraction(index, 1000) for index in range(512)] + [30 + index for index in range(88)]
    service, media, _, rows = synthetic_time_service(tmp_path, monkeypatch, times)
    target = S.rational("1.4505")
    before = service.lookup(media["id"], target, "floor")
    after = service.lookup(media["id"], target, "ceil")
    assert before["frame"] == rows[450] and after["frame"] == rows[451]
    assert before["within_video"] and before["requested_time"] == target
    assert before["frame"]["source_pts"] != before["frame"]["proxy_pts"]
    for index in (0, 255, 256, 511, 512, 599):
        for mode in ("floor", "ceil"):
            assert service.lookup(media["id"], rows[index]["seconds"], mode)["frame"] == rows[index]


def test_time_lookup_leading_audio_tail_and_half_open_endpoint(tmp_path, monkeypatch):
    service, media, _, rows = synthetic_time_service(tmp_path, monkeypatch, [1, 2, 4])
    before = service.lookup(media["id"], S.rational("0.5"), "floor")
    assert before["frame_index"] is None and before["frame"] is None and not before["within_video"]
    assert service.lookup(media["id"], S.rational("0.5"), "ceil")["frame"] == rows[0]
    between = service.lookup(media["id"], S.rational("4.5"), "ceil")
    assert between["frame_index"] == "3" and between["frame"] is None and between["within_video"]
    for time in (5, 20):
        floor = service.lookup(media["id"], S.rational(time), "floor")
        ceil = service.lookup(media["id"], S.rational(time), "ceil")
        assert floor["frame"] == rows[-1] and not floor["within_video"]
        assert ceil["frame_index"] == "3" and ceil["frame"] is None


def test_time_lookup_one_frame_and_large_denominator(tmp_path, monkeypatch):
    from fractions import Fraction

    times = [Fraction(1) + Fraction(index, 10 ** 18) for index in range(3)]
    service, media, _, rows = synthetic_time_service(tmp_path, monkeypatch, times)
    assert service.lookup(media["id"], S.rational(times[1]), "ceil")["frame"] == rows[1]
    assert service.lookup(media["id"], S.rational(times[1]), "floor")["frame"] == rows[1]
    service, media, _, rows = synthetic_time_service(tmp_path, monkeypatch, [2])
    assert service.lookup(media["id"], S.rational(2), "floor")["frame"] == rows[0]
    assert service.lookup(media["id"], S.rational(3), "ceil")["frame_index"] == "1"


@pytest.mark.parametrize("times", [[], [1, 1], [2, 1]])
def test_time_lookup_rejects_empty_or_nonmonotone_admitted_maps(tmp_path, monkeypatch, times):
    service, media, _, _ = synthetic_time_service(tmp_path, monkeypatch, times)
    with pytest.raises(WorkspaceError):
        service.lookup(media["id"], S.rational(1), "floor")


def test_time_lookup_checks_bytes_before_and_after_one_sparse_block(tmp_path, monkeypatch):
    from trace_annotate.workspace import frames

    service, media, path, _ = synthetic_time_service(tmp_path, monkeypatch, range(600))
    service.lookup(media["id"], S.rational(300), "floor")
    original_hash = frames.stream_sha256
    hashes = []
    monkeypatch.setattr(frames, "stream_sha256", lambda path: (hashes.append(path), original_hash(path))[1])
    index = next(iter(service._indices.values()))
    original_rows = index.rows
    reads = []
    monkeypatch.setattr(index, "rows", lambda offset, limit: (reads.append((offset, limit)), original_rows(offset, limit))[1])
    assert service.lookup(media["id"], S.rational("300.5"), "floor")["frame_index"] == "300"
    assert len(hashes) == 2 and reads == [(256, frames.STRIDE + 1)]
    original_lookup = index.at_time
    def changed_during_read(*args):
        result = original_lookup(*args)
        data = path.read_bytes()
        path.write_bytes(data[:-1] + b" ")
        return result
    monkeypatch.setattr(index, "at_time", changed_during_read)
    with pytest.raises(WorkspaceError) as error:
        service.lookup(media["id"], S.rational(300), "floor")
    assert error.value.code == "RECOVERY_REQUIRED"
