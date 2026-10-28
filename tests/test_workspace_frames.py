"""Frame identity, mapping and revision boundaries using actual encoded media."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

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
