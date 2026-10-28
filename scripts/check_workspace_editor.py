"""Exercise the persistent workspace in an isolated, real Chromium browser.

Creates only synthetic, numbered CFR/VFR and audio-offset media. This is a
manual-editing and displayed-frame identity regression, not semantic accuracy
or full W04/release certification. Existing output directories are refused.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import traceback
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--installed-wheel", type=Path)
_bootstrap_args, _ = _bootstrap.parse_known_args()
INSTALLED_WHEEL = _bootstrap_args.installed_wheel
INSTALLED_IDENTITY = None
if INSTALLED_WHEEL is None:
    sys.path.insert(0, str(SOURCE))
else:
    if not sys.flags.isolated:
        raise RuntimeError("Installed acceptance requires the target venv Python with -I")
    import trace_annotate
    package = Path(trace_annotate.__file__).resolve().parent
    roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    if SOURCE in package.parents or not any(root in package.parents for root in roots):
        raise RuntimeError("Installed acceptance must load trace_annotate from target site-packages")
    INSTALLED_WHEEL = INSTALLED_WHEEL.expanduser().resolve(strict=True)
    expected = {}
    with zipfile.ZipFile(INSTALLED_WHEEL) as archive:
        for name in archive.namelist():
            item = Path(name)
            if (name.startswith("trace_annotate/") and item.suffix in {".py", ".html", ".js"}):
                if ".." in item.parts or name in expected:
                    raise RuntimeError("Invalid or duplicated package member in wheel")
                local = package.joinpath(*item.parts[1:])
                contents = archive.read(name)
                if not local.is_file() or local.read_bytes() != contents:
                    raise RuntimeError("Installed resource differs from wheel: " + name)
                expected[name] = hashlib.sha256(contents).hexdigest()
    actual = {"trace_annotate/" + path.relative_to(package).as_posix()
              for path in package.rglob("*") if path.is_file() and path.suffix in {".py", ".html", ".js"}}
    if not expected or set(expected) != actual:
        raise RuntimeError("Installed Python/HTML/JS resource inventory differs from wheel")
    INSTALLED_IDENTITY = {"wheel": str(INSTALLED_WHEEL),
                          "wheel_sha256": hashlib.sha256(INSTALLED_WHEEL.read_bytes()).hexdigest(),
                          "package": str(package), "resources": expected}

from trace_annotate.workspace.server import make_server
from trace_annotate.workspace.store import WorkspaceStore
import trace_annotate
PACKAGE = Path(trace_annotate.__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fraction(value):
    return Fraction(int(value["n"]), int(value["d"]))


def make_sources(output):
    """Encode identifiable pixels; independently verify decoded IDs and PTS."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe are required; media tests cannot be skipped")
    folder = output / "media"
    folder.mkdir()
    frames = folder / "reference-frames"
    frames.mkdir()
    width, height, count = 320, 180, 30
    for index in range(count):
        pixels = bytearray(bytes([40, 60, 80]) * width * height)
        for column in range(5):
            bit = (index >> (4 - column)) & 1
            for y, intensity in ((20, 240 if bit else 16), (135, 16 if bit else 240)):
                line = bytes([intensity] * 3) * 40
                for row in range(y, y + 30):
                    start = (row * width + 20 + column * 56) * 3
                    pixels[start:start + len(line)] = line
        (frames / f"frame-{index:03d}.ppm").write_bytes(
            f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)
    commands = []

    def run(command):
        result = subprocess.run(command, capture_output=True, timeout=90)
        commands.append({"argv": command, "exit_code": result.returncode,
                         "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                         "stdout_bytes": len(result.stdout),
                         "stderr": result.stderr.decode("utf-8", "replace")})
        write_json(folder / "commands.json", commands)
        if result.returncode:
            raise RuntimeError("Fixture command failed; see media/commands.json")
        return result.stdout

    records = {}
    for kind in ("cfr", "vfr", "audio-offset"):
        source = folder / (kind + ".mp4")
        expression = "N*100"
        if kind == "vfr":
            expression = r"if(lt(N\,10)\,N*100\,1000+(N-10)*200)"
        elif kind == "audio-offset":
            expression = "1000+N*100"
        command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n",
                   "-framerate", "10", "-start_number", "0", "-i", str(frames / "frame-%03d.ppm")]
        if kind == "audio-offset":
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=5",
                        "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac"]
        command += ["-vf", "setsar=1,settb=1/1000,setpts=" + expression, "-fps_mode", "passthrough",
                    "-enc_time_base:v", "1:1000", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "10", "-bf", "0", "-pix_fmt", "yuv420p", str(source)]
        run(command)
        probe = json.loads(run([ffprobe, "-v", "error", "-select_streams", "v:0",
                                "-show_streams", "-show_frames", "-show_entries",
                                "stream=time_base,width,height:frame=pts", "-of", "json", str(source)]))
        time_base = Fraction(probe["streams"][0]["time_base"])
        actual_times = [int(row["pts"]) * time_base for row in probe["frames"]]
        expected = [Fraction(i, 10) for i in range(count)]
        if kind == "vfr":
            expected = [Fraction(i, 10) if i < 10 else 1 + Fraction(i - 10, 5) for i in range(count)]
        elif kind == "audio-offset":
            expected = [1 + value for value in expected]
        if actual_times != expected:
            raise AssertionError("Generated source PTS do not match the independent fixture recipe")
        raw = run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-xerror", "-i", str(source),
                   "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        stride = width * height * 3
        if len(raw) != stride * count:
            raise AssertionError("Source full decode did not contain exactly 30 frames")
        for index in range(count):
            upper, lower = [], []
            for y, values in ((35, upper), (150, lower)):
                for column in range(5):
                    offset = index * stride + (y * width + 40 + column * 56) * 3
                    values.append(int(sum(raw[offset:offset + 3]) > 3 * 128))
            observed = sum(bit << (4 - column) for column, bit in enumerate(upper))
            if observed != index or not all(a + b == 1 for a, b in zip(upper, lower)):
                raise AssertionError("Independent source pixel-ID decode failed")
        records[kind] = {"path": str(source), "sha256": digest(source),
                         "frame_times": [str(value) for value in expected], "source_decode_frames": count}
    duplicate = folder / "same-content-renamed.mp4"
    shutil.copyfile(records["cfr"]["path"], duplicate)
    bad = folder / "invalid.mp4"
    bad.write_bytes(b"TraceAnnotate deliberately undecodable acceptance fixture\n")
    records["duplicate"] = {"path": str(duplicate), "sha256": digest(duplicate)}
    records["invalid"] = {"path": str(bad), "sha256": digest(bad)}
    write_json(folder / "manifest.json", records)
    return records


def run_browser(output, sources, channel, summary):
    from playwright.sync_api import sync_playwright, expect
    store = WorkspaceStore(output / "workspace")
    store.create()
    server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    summary["checks"] = []
    summary["page_errors"] = []
    summary["external_requests"] = []
    summary["responses"] = []
    summary["displayed_frame_checks"] = []
    summary["downloads"] = []
    summary["playback_observations"] = []
    summary["layout_observations"] = []
    page = None
    browser = None
    context = None
    held = []

    def check(name):
        summary["checks"].append(name)
        print(name + ": PASS", flush=True)
        write_json(output / "summary.json", summary)

    def observe(target):
        target.on("pageerror", lambda error: summary["page_errors"].append(str(error)))
        target.on("request", lambda request: summary["external_requests"].append(request.url)
                  if request.url.startswith(("http:", "https:")) and not request.url.startswith(server.origin + "/") else None)
        target.on("response", lambda response: summary["responses"].append({
            "path": response.url.split(server.origin, 1)[-1], "status": response.status,
            "content_range": response.headers.get("content-range")}))

    def task_by_name(name):
        intake = next(item for item in store.list_intakes() if item["display_name"] == name)
        return intake["task_id"]

    def select(target, task_id):
        target.locator(f'#task-list button[data-task-id="{task_id}"]').click()
        expect(target.locator(f'#task-list button[data-task-id="{task_id}"]')).to_have_attribute("aria-current", "true")
        expect(target.locator("#new-segment")).to_be_enabled(timeout=30000)

    def fill(target, label, start="0.1", end="1", text=""):
        target.locator("#segment-label").fill(label)
        target.locator("#segment-start").fill(start)
        target.locator("#segment-end").fill(end)
        target.locator("#segment-text").fill(text)

    def save(target, task_id, expected_revision):
        with target.expect_response(lambda response: response.request.method == "POST"
                                    and response.url.endswith(f"/tasks/{task_id}/commands")) as pending:
            target.locator("#save-segment").click()
        if pending.value.status != 200:
            raise AssertionError("Expected a durable command receipt, got " + str(pending.value.status))
        expect(target.locator("#save-segment")).to_be_enabled(timeout=10000)
        if store.get_task(task_id)["revision"] != expected_revision:
            raise AssertionError("Unexpected persisted revision after browser save")

    def snapshot_download(target, name):
        with target.expect_download() as pending:
            target.locator("#export-workspace").click()
        path = output / name
        pending.value.save_as(path)
        body = json.loads(path.read_text(encoding="utf-8"))
        summary["downloads"].append({"path": path.name, "sha256": digest(path)})
        return body

    def wait_value(locator, predicate, argument=None, timeout=30):
        # Playwright's wait_for_function compiles its predicate with eval in
        # the page. Keep the production CSP intact and use CDP evaluation of
        # the selected element, polling its actual state without unsafe-eval.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if locator.evaluate(predicate, argument):
                return
            locator.page.wait_for_timeout(50)
        raise AssertionError("Browser element did not reach the expected state: " + str(locator))

    def show_frame(target, index):
        target.locator("#frame-index").fill(str(index))
        target.locator("#frame-show").click()
        wait_value(target.locator("#frame-image"), """(image, index) => {
            return image && !image.hidden && image.complete && image.naturalWidth > 0 &&
                image.dataset.frameIndex === index;
        }""", str(index))
        pixels = target.locator("#frame-image").evaluate("""image => {
            const canvas = document.createElement('canvas');
            canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
            const context = canvas.getContext('2d', {willReadFrequently:true});
            context.drawImage(image, 0, 0);
            const upper=[], lower=[];
            for (const [y, values] of [[35, upper], [150, lower]]) {
                for (let col=0; col<5; col++) {
                    const rgba=context.getImageData(40+col*56,y,1,1).data;
                    values.push(Number(rgba[0]+rgba[1]+rgba[2] > 3*128));
                }
            }
            return {width:canvas.width,height:canvas.height,
                decoded_id:upper.reduce((sum,bit,col)=>sum+(bit << (4-col)),0),
                complement_valid:upper.every((value,col)=>value+lower[col]===1)};
        }""")
        if pixels != {"width": 320, "height": 180, "decoded_id": index, "complement_valid": True}:
            raise AssertionError(f"Displayed image is not requested frame {index}: {pixels}")
        return pixels

    def video_observation(target, selector="#player"):
        return target.locator(selector).evaluate("""video => {
            const canvas=document.createElement('canvas');canvas.width=320;canvas.height=180;
            const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(video,0,0,320,180);
            const bits=[];for(let col=0;col<5;col++){
                const p=ctx.getImageData(40+col*56,35,1,1).data;
                bits.push(Number(p[0]+p[1]+p[2]>384));
            }
            return {time:video.currentTime,paused:video.paused,ended:video.ended,
                ready:video.readyState,rate:video.playbackRate,src:video.currentSrc,
                pixel_id:bits.reduce((sum,bit,col)=>sum+(bit<<(4-col)),0)};
        }""")

    def exercise_transport(target, task_id, other_task):
        select(target, task_id)
        wait_value(target.locator("#player"), "v=>v.readyState>=2")
        fill(target, "Playback boundary regression", "0.2", "0.8")
        save(target, task_id, 1)
        wait_value(target.locator("#player"), "v=>v.readyState>=2")
        target.locator("#skip-back").click()
        wait_value(target.locator("#player"), "v=>!v.seeking && v.currentTime<0.01")
        initial = video_observation(target)
        target.locator("#play-toggle").click()
        wait_value(target.locator("#player"), "v=>!v.paused && v.currentTime>0.4")
        moving = video_observation(target)
        target.locator("#play-toggle").click()
        wait_value(target.locator("#player"), "v=>v.paused")
        stopped = video_observation(target)
        target.wait_for_timeout(180)
        assert abs(video_observation(target)["time"] - stopped["time"]) < 0.02
        assert moving["pixel_id"] != initial["pixel_id"]
        summary["playback_observations"].append({"case":"click-play-pause", "initial":initial,"moving":moving,"paused":stopped})
        check("Visible play/pause clicks advance decoded pixels and pause the actual video")

        target.locator("#play-selection").click()
        wait_value(target.locator("#player"), "v=>v.paused && Math.abs(v.currentTime-0.8)<0.03")
        target.locator("#play-toggle").click()
        wait_value(target.locator("#player"), "v=>!v.paused && v.currentTime>1.1")
        target.locator("#play-toggle").click()
        check("Main playback continues past a completed 0.2–0.8 second selection")

        target.locator("#loop-selection").check()
        target.locator("#play-selection").click()
        samples, loops, previous = [], 0, None
        deadline=time.monotonic()+8
        while loops < 2 and time.monotonic() < deadline:
            value=target.locator("#player").evaluate("v=>({time:v.currentTime,paused:v.paused})")
            samples.append(value)
            if previous is not None and previous-value["time"]>0.3:
                loops+=1
            previous=value["time"]
            target.wait_for_timeout(40)
        assert loops >= 2 and not samples[-1]["paused"], samples
        target.locator("#play-toggle").click()
        target.locator("#loop-selection").uncheck()
        summary["playback_observations"].append({"case":"selection-loop", "observed_wraps":loops,"samples":samples})
        check("Selected interval loops across its boundary at least twice through actual playback")

        rates=[]
        for rate in ("0.25","1.5"):
            target.locator("#skip-back").click()
            target.locator("#playback-rate").select_option(rate)
            target.locator("#play-toggle").click()
            wait_value(target.locator("#player"), "v=>!v.paused && v.currentTime>0.02")
            begin=time.monotonic(); before=target.locator("#player").evaluate("v=>v.currentTime")
            target.wait_for_timeout(400)
            after=target.locator("#player").evaluate("v=>v.currentTime")
            elapsed=time.monotonic()-begin
            target.locator("#play-toggle").click()
            actual=target.locator("#player").evaluate("v=>v.playbackRate")
            assert actual == float(rate) and after>before
            rates.append({"selected":actual,"media_delta":after-before,"wall_seconds":elapsed})
        assert rates[1]["media_delta"]/rates[1]["wall_seconds"] > 3*rates[0]["media_delta"]/rates[0]["wall_seconds"]
        target.locator("#playback-rate").select_option("1")
        target.locator("#skip-forward").click()
        wait_value(target.locator("#player"), "v=>Math.abs(v.currentTime-v.duration)<0.01")
        target.locator("#skip-back").click()
        wait_value(target.locator("#player"), "v=>v.currentTime<0.01")
        summary["playback_observations"].append({"case":"rates-and-clamps","rates":rates})
        check("Selected speeds change observed progression; five-second jumps clamp at both media ends")

        target.locator("#frame-mode").check()
        wait_value(target.locator("#frame-image"), "image=>!image.hidden && image.complete && image.dataset.frameIndex==='0'")
        show_frame(target, 12)
        expect(target.locator("#player")).to_be_hidden()
        target.locator("#play-toggle").click()
        expect(target.locator("#frame-mode")).not_to_be_checked()
        expect(target.locator("#frame-image")).to_be_hidden()
        wait_value(target.locator("#player"), "v=>!v.paused && v.currentTime>=1.2 && v.currentTime<2.4")
        resumed=video_observation(target)
        target.locator("#play-toggle").click()
        assert resumed["pixel_id"] >= 12
        summary["playback_observations"].append({"case":"frame-to-video","frame":12,"resumed":resumed})
        check("Primary play returns from the displayed deterministic frame to continuous video")

        target.locator("#loop-selection").check()
        target.locator("#play-selection").click()
        wait_value(target.locator("#player"), "v=>!v.paused")
        select(target, other_task)
        wait_value(target.locator("#player"), "v=>v.readyState>=2 && v.paused")
        expect(target.locator("#loop-selection")).not_to_be_checked()
        expect(target.locator("#play-selection")).to_be_disabled()
        target.wait_for_timeout(800)
        assert target.locator("#player").evaluate("v=>v.paused && v.currentTime<0.05")
        select(target, task_id)
        target.locator("#segments tbody button[data-annotation-id]").first.click()
        check("Switching tasks cancels playback and loop state without changing the other task")

        before_fields={name:target.locator("#segment-"+name).input_value() for name in ("label","start","end","text")}
        target.locator("#segment-text").focus()
        for key in ("Space","i","o","ArrowLeft","ArrowRight"):
            target.keyboard.press(key)
        assert target.locator("#player").evaluate("v=>v.paused")
        for name in ("start","end"):
            assert target.locator("#segment-"+name).input_value() == before_fields[name]
        target.locator("#segment-text").fill(before_fields["text"])
        for composing, code in ((True,0),(False,229)):
            for key, key_code, control in ((" ","Space",False),("i","KeyI",False),("o","KeyO",False),("s","KeyS",True)):
                target.locator("body").dispatch_event("keydown", {"key":key,"code":key_code,
                    "bubbles":True,"isComposing":composing,"keyCode":code,"ctrlKey":control})
        assert target.locator("#player").evaluate("v=>v.paused")
        assert store.get_task(task_id)["revision"] == 1
        for name,value in before_fields.items():
            assert target.locator("#segment-"+name).input_value() == value
        check("Text typing and injected IME composition/keyCode-229 events do not trigger playback, marks or save")

        for width,height,scale,label in ((1070,720,1,"1070x720"),(1366,768,1,"1366x768"),(683,384,2,"200-percent-equivalent-reflow")):
            target.set_viewport_size({"width":width,"height":height})
            target.evaluate("window.scrollTo(0,0)")
            if scale==2:
                target.locator("#play-toggle").scroll_into_view_if_needed()
            target.locator("#play-toggle").hover()
            metrics=target.locator("#play-toggle").evaluate("""el=>{
                const r=el.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
                const style=getComputedStyle(el),lum=color=>{
                    const rgb=color.match(/[\\d.]+/g).slice(0,3).map(Number).map(v=>{
                        v/=255;return v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4;});
                    return rgb[0]*0.2126+rgb[1]*0.7152+rgb[2]*0.0722;};
                const a=lum(style.color),b=lum(style.backgroundColor);
                return {x:r.x,y:r.y,width:r.width,height:r.height,viewport_width:innerWidth,
                    viewport_height:innerHeight,unobscured:hit===el||el.contains(hit),
                    hovered_text:style.color,hovered_background:style.backgroundColor,
                    hovered_contrast:(Math.max(a,b)+0.05)/(Math.min(a,b)+0.05)};
            }""")
            assert metrics["x"]>=0 and metrics["y"]>=0 and metrics["x"]+metrics["width"]<=width
            assert metrics["y"]+metrics["height"]<=height and metrics["unobscured"], metrics
            assert metrics["hovered_contrast"] >= 4.5, metrics
            target.locator("#skip-back").click()
            target.locator("#play-toggle").click()
            wait_value(target.locator("#player"), "v=>!v.paused && v.currentTime>0.15")
            target.locator("#play-toggle").click()
            target.screenshot(path=str(output / ("playback-"+label+".png")))
            summary["layout_observations"].append({"case":label,"css_scale_equivalent":scale,**metrics})
        target.set_viewport_size({"width":1500,"height":1060})
        check("Laptop play controls are visible and clickable; 200-percent-equivalent CSS reflow remains reachable")

    def exercise_classic_routes(target):
        for mode in ("general","embodied"):
            links=target.locator('a[href="/tools/'+mode+'/"]')
            expect(links.first).to_be_visible()
        classic=context.new_page();observe(classic)
        classic.on("dialog",lambda dialog: dialog.accept())
        classic.goto(server.origin+"/tools/general/")
        classic.locator("#videoInput").set_input_files(sources["cfr"]["path"])
        expect(classic.locator("#togglePlay")).to_be_enabled(timeout=30000)
        expect(classic.locator("#mediaIdentity")).to_contain_text(sources["cfr"]["sha256"])
        classic.locator("#togglePlay").click()
        wait_value(classic.locator("#videoPlayer"),"v=>!v.paused && v.currentTime>0.3")
        classic.locator("#togglePlay").click()
        for selector,value in (("startInput","0.2"),("endInput","0.8"),("labelInput","Classic label"),("textInput","Classic cue"),("trackInput","subtitles")):
            classic.locator("#"+selector).fill(value)
        classic.locator("#reviewInput").check()
        classic.locator("#saveSegment").click()
        expect(classic.locator("#segmentList button")).to_have_count(1)
        classic.locator("#undo").click();expect(classic.locator("#segmentList button")).to_have_count(0)
        classic.locator("#redo").click();expect(classic.locator("#segmentList button")).to_have_count(1)
        classic.locator("#subtitleTrack").select_option("subtitles")
        files={}
        for fmt in ("JSON","CSV","SRT","VTT"):
            with classic.expect_download() as pending:
                classic.locator("#export"+fmt).click()
            path=output/("classic-general."+fmt.lower());pending.value.save_as(path);files[fmt]=path
            summary["downloads"].append({"path":path.name,"sha256":digest(path)})
        document=json.loads(files["JSON"].read_text(encoding="utf-8"))
        assert document["media_sha256"] == sources["cfr"]["sha256"]
        assert len(document["segments"])==1 and document["segments"][0]["review_status"]=="reviewed"
        assert list(csv.DictReader(io.StringIO(files["CSV"].read_text(encoding="utf-8-sig"))))[0]["text"]=="Classic cue"
        for fmt in ("SRT","VTT"):
            value=files[fmt].read_text(encoding="utf-8")
            assert "Classic cue" in value and ("00:00:00,200 --> 00:00:00,800" if fmt=="SRT" else "00:00:00.200 --> 00:00:00.800") in value
        classic.locator("#newProject").click()
        expect(classic.locator("#segmentList button")).to_have_count(0)
        classic.locator("#jsonInput").set_input_files(str(files["JSON"]))
        expect(classic.locator("#segmentList button")).to_have_count(1)
        check("Served original general editor hashes/plays media, edits/reviews/undoes, downloads formats and reopens JSON")
        classic.goto(server.origin+"/tools/embodied/")
        classic.locator("#fileInput").set_input_files(sources["cfr"]["path"])
        expect(classic.locator("#togglePlay")).to_be_enabled(timeout=30000)
        classic.locator("#togglePlay").click()
        wait_value(classic.locator("#videoPlayer"),"v=>!v.paused && v.currentTime>0.2")
        classic.locator("#togglePlay").click()
        classic.locator("#taskDescription").fill("Synthetic compatibility route check")
        classic.locator("#expectedSteps").fill("1")
        for value,button in (("0.2","markStart"),("0.8","markEnd")):
            classic.locator("#seekSeconds").fill(value);classic.locator("#seekTo").click()
            wait_value(classic.locator("#videoPlayer"),"(v,t)=>!v.seeking && Math.abs(v.currentTime-t)<0.001",float(value))
            classic.locator("#"+button).click()
        classic.locator("#segmentLabel").fill("Synthetic action")
        classic.locator("#selVerdict").select_option("无法判断")
        classic.locator("#computeIdle").click()
        with classic.expect_download() as pending:
            classic.locator("#exportJSON").click()
        path=output/"classic-embodied.json";pending.value.save_as(path)
        data=json.loads(path.read_text(encoding="utf-8"))
        assert len(data["working"])==1 and data["working"][0]["verdict"]=="无法判断"
        assert data["working"][0]["start"]==0.2 and data["working"][0]["end"]==0.8
        assert len(data["idle"])==2 and data["expected_step_count"]==1
        summary["downloads"].append({"path":path.name,"sha256":digest(path)})
        classic.close()
        check("Served original embodied editor plays, marks an unresolved step and exports its explicit Idle complement")

    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch(headless=True, **({"channel": channel} if channel else {}))
        summary["browser"] = browser.version
        context = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 1060})
        page = context.new_page()
        observe(page)
        page.goto(server.editor_url)
        expect(page.locator("#file-input")).to_be_enabled()
        page.locator("#file-input").set_input_files([sources[key]["path"] for key in
                                                    ("cfr", "vfr", "invalid", "audio-offset", "duplicate")])
        expect(page.locator("#import-status")).to_contain_text("本次导入结束", timeout=120000)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(4)
        expect(page.locator("#file-input")).to_be_enabled(timeout=120000)
        intakes = store.list_intakes()
        assert len(intakes) == 5 and sum(item["state"] == "succeeded" for item in intakes) == 4
        assert sum(item["state"] == "failed" for item in intakes) == 1
        assert len(store.get_snapshot()["media"]) == 3
        expect(page.locator("body")).to_contain_text("invalid.mp4")
        check("Multi-file upload preserved a failed intake and admitted later inputs")

        # Admission is already durable if its 200 response was delivered, even
        # when the following list refresh fails. Do not count the file twice or
        # invite a duplicate retry; stop the remaining selected inputs instead.
        refresh_faults = []
        def fail_refresh(route):
            refresh_faults.append(route.request.url)
            route.fulfill(status=503, content_type="application/json", body=json.dumps({
                "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "Injected list refresh fault",
                          "retryable": True, "details": {}}}))
        page.route("**/snapshot", fail_refresh, times=1)
        page.locator("#file-input").set_input_files([sources["cfr"]["path"], sources["vfr"]["path"]])
        expect(page.locator("#import-status")).to_contain_text("本次导入结束", timeout=30000)
        expect(page.locator("#import-status")).to_contain_text("成功 1，失败或未确认 0")
        assert len(refresh_faults) == 1 and len(store.list_intakes()) == 6
        assert len(store.list_tasks()) == 5
        page.locator("#refresh-workspace").click()
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(5)
        check("Admission success with failed list refresh stays successful and stops remaining uploads")

        cfr = task_by_name("cfr.mp4")
        duplicate = task_by_name("same-content-renamed.mp4")
        assert cfr != duplicate and store.get_task(cfr)["media_id"] == store.get_task(duplicate)["media_id"]
        transport_task = next(item["id"] for item in store.get_snapshot()["tasks"]
                              if item["id"] not in (cfr, duplicate)
                              and item["media_id"] == store.get_task(cfr)["media_id"])
        exercise_transport(page, transport_task, cfr)
        exercise_classic_routes(page)
        select(page, cfr)

        page.locator("#new-segment").click()
        label = '<img src=x onerror="window.injectionRan=true">'
        fill(page, "invalid reversed segment", "2", "1")
        page.locator("#save-segment").click()
        expect(page.locator("#status")).to_have_class("notice error")
        expect(page.locator("#status")).not_to_be_empty()
        assert store.get_task(cfr)["revision"] == 0
        fill(page, label, "1/3", "2/3", "中文 😀\nTwo lines & <literal> text")
        page.locator("#segment-track").fill("人工片段")
        page.locator("#segment-tags").fill("重复\n重复\n😀")
        save(page, cfr, 1)
        first = store.get_task(cfr)["annotations"][0]
        assert first["start"]["seconds"] == {"n": "1", "d": "3"}
        assert first["end"]["seconds"] == {"n": "2", "d": "3"}
        assert first["label"] == label and first["tags"] == ["重复", "重复", "😀"]
        assert not page.evaluate("window.injectionRan === true")
        assert page.locator("#segments img").count() == 0
        assert not store.get_task(duplicate)["annotations"]
        check("Exact rational manual save, Unicode/tag preservation, literal HTML and same-media task isolation")

        page.locator("#segment-text").fill("unsaved-draft-marker")
        page.locator(f'#task-list button[data-task-id="{duplicate}"]').click()
        expect(page.locator("#confirm-dialog")).to_be_visible()
        page.locator("#confirm-cancel").click()
        expect(page.locator("#segment-text")).to_have_value("unsaved-draft-marker")
        assert store.get_task(cfr)["revision"] == 1
        page.locator(f'#task-list button[data-task-id="{duplicate}"]').click()
        page.locator("#confirm-accept").click()
        expect(page.locator("#segments tbody button[data-annotation-id]")).to_have_count(0)
        select(page, cfr)
        page.locator(f'#segments tbody button[data-annotation-id="{first["id"]}"]').click()
        expect(page.locator("#segment-text")).to_have_value(first["text"])
        check("Dirty task navigation can cancel or explicitly discard without overwriting persisted data")

        page.locator("#segment-label").fill("renamed-only")
        save(page, cfr, 2)
        renamed = store.get_task(cfr)["annotations"][0]
        assert {key: value for key, value in renamed.items() if key != "label"} == {
            key: value for key, value in first.items() if key != "label"}
        page.reload()
        select(page, cfr)
        page.locator(f'#segments tbody button[data-annotation-id="{first["id"]}"]').click()
        expect(page.locator("#segment-label")).to_have_value("renamed-only")
        check("Field-only update preserves all other content; actual page reload restores committed data")

        page.route("**/tasks/" + duplicate, lambda route: held.append(route), times=1)
        page.locator(f'#task-list button[data-task-id="{duplicate}"]').click()
        expect(page.locator("#save-segment")).to_be_disabled()
        expect(page.locator("#segment-label")).to_be_disabled()
        expect(page.locator("#segment-text")).to_be_disabled()
        page.keyboard.press("Control+s")
        page.wait_for_timeout(50)
        assert len(held) == 1 and store.get_task(cfr)["revision"] == 2
        assert store.get_task(duplicate)["revision"] == 0
        held[0].continue_()
        held.clear()
        expect(page.locator(f'#task-list button[data-task-id="{duplicate}"]')).to_have_attribute("aria-current", "true")
        select(page, cfr)
        page.locator(f'#segments tbody button[data-annotation-id="{first["id"]}"]').click()
        check("Pending task load disables editing and keyboard save until the new task is bound")

        other = context.new_page()
        observe(other)
        other.goto(server.editor_url)
        select(other, cfr)
        other.locator(f'#segments tbody button[data-annotation-id="{first["id"]}"]').click()
        other.locator("#segment-label").fill("stale-other-tab")
        page.locator("#segment-label").fill("winning-tab")
        save(page, cfr, 3)
        with other.expect_response(lambda response: response.request.method == "POST" and
                                   response.url.endswith(f"/tasks/{cfr}/commands")) as pending:
            other.locator("#save-segment").click()
        assert pending.value.status == 409
        expect(other.locator("#conflict-panel")).to_be_visible()
        expect(other.locator("#segment-label")).to_have_value("stale-other-tab")
        assert store.get_task(cfr)["annotations"][0]["label"] == "winning-tab"
        other.locator("#reload-task").click()
        expect(other.locator("#confirm-dialog")).to_be_visible()
        other.locator("#confirm-accept").click()
        other.locator(f'#segments tbody button[data-annotation-id="{first["id"]}"]').click()
        expect(other.locator("#segment-label")).to_have_value("winning-tab")
        other.close()
        check("Two actual tabs reject stale revision and retain the losing draft until explicit reload")

        # Deliver the command to the real server but drop only the response.
        # UI recovery must query its durable receipt instead of duplicating it.
        dropped = []
        def drop_after_commit(route):
            response = route.fetch()
            dropped.append({"status": response.status, "body": response.json()})
            route.abort("failed")
        path_pattern = "**/tasks/" + cfr + "/commands"
        page.route(path_pattern, drop_after_commit, times=1)
        receipt_lookups = []
        def receipt_fault(route):
            receipt_lookups.append(route.request.url)
            if len(receipt_lookups) == 1:
                route.fulfill(status=403, content_type="application/json", body=json.dumps({
                    "error": {"code": "INVALID_TOKEN", "message": "Injected receipt lookup fault",
                              "retryable": False, "details": {"commit_status": "not_committed"}}}))
            else:
                route.continue_()
        page.route("**/commands/*", receipt_fault)
        page.locator("#segment-label").fill("saved-despite-lost-response")
        page.locator("#save-segment").click()
        expect(page.locator("#pending-retry")).to_be_visible(timeout=15000)
        expect(page.locator("#pending-retry")).to_be_enabled(timeout=15000)
        expect(page.locator("#save-segment")).to_be_disabled()
        pending_storage = page.evaluate("""() => Object.entries(sessionStorage)
            .filter(([key])=>key.startsWith('traceannotate-pending:')).map(([,value])=>JSON.parse(value))""")
        assert len(pending_storage) == 1 and pending_storage[0]["command"]["command_id"] == dropped[0]["body"]["command_id"]
        # Synchronous repeated activation probes the busy guard; the second
        # activation must not start another receipt query or command replay.
        page.locator("#pending-retry").evaluate("button => {button.click(); button.click();}")
        expect(page.locator("#pending-retry")).not_to_be_visible(timeout=15000)
        assert len(receipt_lookups) == 2
        page.unroute("**/commands/*", receipt_fault)
        assert len(dropped) == 1 and dropped[0]["status"] == 200
        assert store.get_task(cfr)["revision"] == 4
        assert store.get_task(cfr)["annotations"][0]["label"] == "saved-despite-lost-response"
        check("Lost command response plus forbidden receipt lookup keeps pending state; repeated recovery commits once")

        for case, failing_resource, lose_initial_post in (
                ("committed-snapshot-refresh", "snapshot", False),
                ("committed-task-refresh", "tasks/" + cfr, False),
                ("replayed-command-refresh", "snapshot", True)):
            # A GET error's not_committed metadata describes that read only.
            # It must never erase knowledge that the preceding POST committed,
            # including when the successful POST was an idempotent recovery.
            base_revision = store.get_task(cfr)["revision"]
            writes, refresh_errors = [], []

            def fail_committed_refresh(route):
                refresh_errors.append(route.request.url)
                route.fulfill(status=403, content_type="application/json", body=json.dumps({
                    "error": {"code": "INVALID_TOKEN", "message": "Injected post-commit refresh failure",
                              "retryable": False, "details": {"commit_status": "not_committed"}}}))

            def observe_write(route):
                item = {"request": route.request.post_data_json, "status": None}
                writes.append(item)
                if lose_initial_post and len(writes) == 1:
                    route.abort("failed")
                    return
                response = route.fetch()
                item.update(status=response.status, response=response.json())
                route.fulfill(response=response)

            fault_pattern = "**/" + failing_resource
            page.route(fault_pattern, fail_committed_refresh)
            page.route(path_pattern, observe_write)
            page.locator("#segment-label").fill(case)
            page.locator("#save-segment").click()
            expect(page.locator("#pending-retry")).to_be_visible(timeout=15000)
            expect(page.locator("#pending-retry")).to_be_enabled(timeout=15000)
            if lose_initial_post:
                assert store.get_task(cfr)["revision"] == base_revision
                page.locator("#pending-retry").click()
                deadline = time.monotonic() + 15
                while store.get_task(cfr)["revision"] != base_revision + 1 and time.monotonic() < deadline:
                    page.wait_for_timeout(50)
                expect(page.locator("#pending-retry")).to_be_enabled(timeout=15000)
            assert store.get_task(cfr)["revision"] == base_revision + 1
            assert store.get_task(cfr)["annotations"][0]["label"] == case
            expect(page.locator("#pending-retry")).to_be_visible()
            expect(page.locator("#save-segment")).to_be_disabled()
            expect(page.locator("#segment-label")).to_have_value(case)
            saved_pending = page.evaluate("""() => Object.entries(sessionStorage)
                .filter(([key])=>key.startsWith('traceannotate-pending:')).map(([,value])=>JSON.parse(value))""")
            assert len(saved_pending) == 1
            command_id = saved_pending[0]["command"]["command_id"]
            assert all(write["request"]["command_id"] == command_id for write in writes)
            assert sum(write["status"] == 200 for write in writes) == 1 and refresh_errors
            page.unroute(fault_pattern, fail_committed_refresh)
            page.locator("#pending-retry").click()
            expect(page.locator("#pending-retry")).not_to_be_visible(timeout=15000)
            expect(page.locator("#segment-label")).to_have_value(case)
            assert store.get_task(cfr)["revision"] == base_revision + 1
            assert sum(write["status"] == 200 for write in writes) == 1
            assert store.get_command_receipt(command_id)["state"] == "committed"
            page.unroute(path_pattern, observe_write)
            check(case + " retains the known committed command through read errors and recovers exactly once")

        for kind in ("cfr", "vfr", "audio-offset"):
            task_id = task_by_name(kind + ".mp4")
            select(page, task_id)
            page.locator("#frame-mode").check()
            wait_value(page.locator("#frame-image"), """image => {
                return !image.hidden && image.complete && image.naturalWidth > 0 &&
                    image.dataset.frameIndex === '0';
            }""")
            for index in range(30):
                pixels = show_frame(page, index)
                summary["displayed_frame_checks"].append({"case": kind, "frame_index": index, **pixels})
            page.locator("#frame-image").screenshot(path=str(output / (kind + "-frame-29.png")))
            check(kind + " displayed PNG frame IDs match all 30 actual source frames")
            page.locator("#new-segment").click()
            unchanged_start = page.locator("#segment-start").input_value()
            page.locator("#frame-index").fill("30")
            page.locator("#frame-show").click()
            expect(page.locator("#frame-image")).to_be_hidden()
            page.locator("#mark-start").click()
            expect(page.locator("#segment-start")).to_have_value(unchanged_start)
            show_frame(page, 7)
            page.locator("#mark-start").click()
            show_frame(page, 16)
            page.locator("#mark-end").click()
            page.locator("#segment-label").fill("frame-boundary-" + kind)
            before = store.get_task(task_id)["revision"]
            save(page, task_id, before + 1)
            annotation = store.get_task(task_id)["annotations"][-1]
            expected_times = sources[kind]["frame_times"]
            assert fraction(annotation["start"]["seconds"]) == Fraction(expected_times[7])
            assert fraction(annotation["end"]["seconds"]) == Fraction(expected_times[16])
            assert annotation["start"]["anchor"]["frame_index"] == "7"
            assert annotation["end"]["anchor"]["frame_index"] == "16"
            page.locator("#frame-mode").uncheck()
        check("CFR, VFR and nonzero video-origin frame anchors persist the displayed source identity")

        # Set up valid pre-existing content via the same command store contract.
        # Browser textareas normalize CRLF; a label-only edit must not silently
        # rewrite an untouched field from such a previously saved revision.
        original = {"id": "ann-preservation", "type": "segment", "label": "original",
                    "text": "第一行\r\n第二行 😀", "track": "main",
                    "tags": ["duplicate", "duplicate", "tag\nwith-newline"],
                    "start": {"seconds": {"n": "1", "d": "10"}},
                    "end": {"seconds": {"n": "1", "d": "1"}}}
        current = store.get_task(duplicate)
        setup = {"command_id": "cmd-preservation-fixture", "task_id": duplicate,
                 "expected_revision": current["revision"], "expected_sha256": current["content_sha256"],
                 "operation": "add_annotation", "payload": {"annotation": original}}
        store.apply_task_command(**setup, request_body=json.dumps(setup).encode("utf-8"))
        preserved = store.get_task(duplicate)["annotations"][0]
        page.locator("#refresh-workspace").click()
        select(page, duplicate)
        page.locator('#segments tbody button[data-annotation-id="ann-preservation"]').click()
        page.locator("#segment-label").fill("renamed-with-CRLF-preserved")
        save(page, duplicate, 2)
        changed = store.get_task(duplicate)["annotations"][0]
        assert {key: value for key, value in changed.items() if key != "label"} == {
            key: value for key, value in preserved.items() if key != "label"}
        check("Label-only browser edit preserves untouched CRLF text and newline/duplicate tags")
        with page.expect_response(lambda response: response.request.method == "POST" and
                                  response.url.endswith(f"/tasks/{duplicate}/commands")):
            page.locator("#delete-segment").click()
            expect(page.locator("#confirm-dialog")).to_be_visible()
            page.locator("#confirm-accept").click()
        assert store.get_task(duplicate)["revision"] == 3
        assert not store.get_task(duplicate)["annotations"]
        check("Delete is explicit, durable and isolated to the selected task")
        exported = snapshot_download(page, "browser-workspace.json")
        assert exported["tasks"] == store.get_snapshot()["tasks"]
        assert exported["media"] == store.get_snapshot()["media"]
        assert len(exported["tasks"]) == 5
        assert exported["intakes"] == store.list_intakes()
        page.screenshot(path=str(output / "workspace.png"), full_page=True)
        check("Browser-downloaded snapshot retains all task revisions, exact times and media bindings")
        if summary["page_errors"] or summary["external_requests"]:
            raise AssertionError("Browser had unexpected script errors or external requests")
        if not any(item["path"].endswith("/playback") and item["status"] == 206 for item in summary["responses"]):
            raise AssertionError("Actual browser did not exercise HTTP range playback")
        context.close()
        context = None
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.close()
        store.open()
        server = make_server(store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        context = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 1060})
        page = context.new_page()
        observe(page)
        page.goto(server.editor_url)
        expect(page.locator("#task-list button[data-task-id]")).to_have_count(5)
        reopened = snapshot_download(page, "browser-after-service-restart.json")
        assert reopened["tasks"] == exported["tasks"] and reopened["media"] == exported["media"]
        assert reopened["intakes"] == exported["intakes"]
        check("Service/store restart and fresh browser context preserve committed annotations")
    except BaseException:
        if page is not None:
            try:
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                html = page.content().replace(server.token, "REDACTED_EPHEMERAL_TOKEN")
                (output / "failure-dom.html").write_text(html, encoding="utf-8")
            except Exception:
                pass
        raise
    finally:
        for route in held:
            try:
                route.abort()
            except Exception:
                pass
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.close()
        playwright.stop()
    if any(digest(record["path"]) != record["sha256"] for record in sources.values()):
        raise AssertionError("A source media file changed during acceptance")
    check("Every source file remains byte-for-byte unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channel", help="Optional installed browser channel, e.g. chrome")
    parser.add_argument("--installed-wheel", type=Path,
                        help="Require -I and verify all installed Python/HTML/JS bytes against this wheel")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error("Output must be a new directory; existing evidence preserved: " + str(exc))
    source_files = [Path(__file__).resolve()] + sorted(path for path in PACKAGE.rglob("*")
                    if path.is_file() and path.suffix in {".py", ".html", ".js"})
    def source_key(path):
        return str(path.relative_to(SOURCE)) if INSTALLED_IDENTITY is None else str(path)
    hashes = {source_key(path): digest(path) for path in source_files}
    summary = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "argv": [sys.executable] + sys.argv, "python": sys.version,
               "source_sha256": hashes, "installed_identity": INSTALLED_IDENTITY,
               "headless": True, "native_file_picker_tested": False,
               "scope": "Synthetic click-driven transport, classic editor routes, multi-video manual editing, "
                        "persisted commands and displayed-frame identity. IME is injected-event coverage and "
                        "200-percent layout uses an equivalent CSS viewport, not native browser zoom. "
                        "Not audible-audio quality, automatic annotation accuracy, complete A13 or release readiness."}
    write_json(output / "summary.json", summary)
    try:
        sources = make_sources(output)
        run_browser(output, sources, args.channel, summary)
        if any(digest(path) != hashes[source_key(path)] for path in source_files):
            raise AssertionError("Implementation changed during acceptance; rerun against fixed source")
        if INSTALLED_IDENTITY and digest(INSTALLED_WHEEL) != INSTALLED_IDENTITY["wheel_sha256"]:
            raise AssertionError("Installed acceptance wheel changed during the run")
        summary["status"] = "PASS"
    except Exception as exc:
        summary.update(status="FAIL", error=str(exc))
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("Error: " + str(exc) + "; evidence: " + str(output), file=sys.stderr)
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
