"""Real Node checks of shipped frame navigation and asynchronous ownership."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def run_ui(program):
    node = shutil.which("node")
    if not node:
        pytest.skip("Real Node is required for frame-navigation UI checks")
    source = (Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py").read_text("utf-8")
    parts = [source[source.index("function gcd("):source.index("const state=")],
        source[source.index("function releaseFrame("):source.index("let compareTimer=")],
        source[source.index("async function frameAtTime("):source.index("async function switchTask(")],
        source[source.index("async function showFrame("):source.index("function captureTime(")],
        source[source.index("async function toggleFrameMode("):source.index("async function refreshWorkspace(")]]
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const implementation=JSON.parse(process.argv[1]);
function deferred(){let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve};}
function harness(){
 const fields=new Map();
 const get=id=>{if(!fields.has(id))fields.set(id,{value:'0',checked:false,hidden:false,textContent:'',dataset:{},style:{setProperty(){}},removeAttribute(){}});return fields.get(id)};
 const media={id:'media-A',frame_count:'1000',duration:{n:'40',d:'1'},video_start:{n:'0',d:'1'},video_end:{n:'40',d:'1'},source_width:160,source_height:90,pixel_aspect_ratio:{n:'1',d:'1'},rotation_degrees:0,playback:{sha256:'a'.repeat(64)}};
 const state={task:{id:'task-A'},media,frameSequence:0,frame:null,frameBlob:null,loadingFrame:false,loadingTask:false,compareToken:0,compareBlobs:[]};
 const revoked=[],shown=[],notices=[],calls=[];
 get('frame-mode').checked=true;Object.assign(get('player'),{readyState:2,currentTime:21.14,paused:false,pause(){this.paused=true}});
 const context={state,$:get,console,Uint8Array,Number,BigInt,Blob:class{},URL:{createObjectURL(){return 'blob:frame'},revokeObjectURL(url){revoked.push(url)}},
  Image:class{decode(){return Promise.resolve()}},crypto:{subtle:{digest:async()=>new Uint8Array(32).buffer}},
  json:async()=>{throw new Error('Unexpected JSON request')},request:async()=>{throw new Error('Unexpected image request')},
  renderPlayback(){},updatePlaybackTime(){},pausePlayback(){get('player').pause()},seekPlayback(time){get('player').currentTime=time},
  notice(text){notices.push(text)},describeError:error=>error.message,
  selectedSegment:()=>null,releaseCompare(){state.compareToken++},clearCompareSlots(){},compareSet(){},
  compareSlots:{'start-before':{label:'起点前一帧'},'start-after':{label:'起点帧'},'end-before':{label:'终点前一帧'},'end-after':{label:'终点帧'}}};
 vm.createContext(context);vm.runInContext(implementation,context);
 return {context,state,get,media,revoked,shown,notices,calls};
}
(async()=>{
""" + program + r"""
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-e", harness, json.dumps("\n".join(parts))],
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_enter_frame_mode_uses_current_time_and_reports_endpoint_clamping():
    run_ui(r"""
const h=harness(),c=h.context;
h.get('frame-index').value='0';
c.frameAtTime=async(target,mode,media)=>{
 assert.deepEqual(JSON.parse(JSON.stringify(target)),{n:'1057',d:'50'});
 assert.equal(mode,'floor');assert.equal(media,h.media);
 return {frame_index:'634',within_video:true};
};
c.showFrame=async index=>h.shown.push(index);
await c.toggleFrameMode();assert.deepEqual(h.shown,['634']);assert.equal(h.get('player').paused,true);
for(const [value,expected]of [[1e-7,{n:'1',d:'10000000'}],[1e7,{n:'10000000',d:'1'}]])assert.deepEqual(JSON.parse(JSON.stringify(c.playbackTimeRational(value))),expected);
for(const [index,message]of [[null,'首帧'],['999','末帧']]){
 const edge=harness();edge.context.frameAtTime=async()=>({frame_index:index,within_video:false});
 edge.context.showFrame=async index=>edge.shown.push(index);
 await edge.context.toggleFrameMode();assert.equal(edge.shown[0],index??'0');assert.ok(edge.notices.some(text=>text.includes(message)));
}
for(const [ready,time]of [[0,21.14],[2,NaN],[2,Infinity],[2,-1]]){
 const unready=harness();Object.assign(unready.get('player'),{readyState:ready,currentTime:time});
 unready.context.frameAtTime=async()=>{throw new Error('Must not lookup invalid metadata/time')};
 await unready.context.toggleFrameMode();assert.equal(unready.get('frame-mode').checked,false);assert.equal(unready.state.frame,null);
}
""")


def test_late_cursor_lookup_cannot_override_cancel_manual_frame_or_task_switch():
    run_ui(r"""
for(const change of ['off','task','media','manual','loading']){
 const h=harness(),c=h.context,wait=deferred();c.frameAtTime=()=>wait.promise;c.showFrame=async index=>h.shown.push(index);
 const pending=c.toggleFrameMode();assert.equal(h.state.loadingFrame,true);
 if(change==='off'){h.get('frame-mode').checked=false;c.releaseFrame();}
 if(change==='task'){h.state.task={id:'task-B'};c.releaseFrame();}
 if(change==='media')h.state.media={...h.media,id:'media-B'};
 if(change==='manual')h.state.frameSequence++;
 if(change==='loading')h.state.loadingTask=true;
 wait.resolve({frame_index:'634',within_video:true});await pending;
 assert.deepEqual(h.shown,[],change);
}
""")


def test_late_frame_row_image_and_decode_cannot_publish_stale_identity():
    run_ui(r"""
const row={frame_index:'7',source_pts:'700',proxy_pts:'700',seconds:{n:'7',d:'10'}};
for(const phase of ['row','bytes','decode']){
 const h=harness(),c=h.context,wait=deferred();let imageRequests=0;
 c.json=async()=>phase==='row'?wait.promise:{frames:[row]};
 c.request=async()=>{imageRequests++;return {headers:{get(name){return {'Content-Type':'image/png','X-TraceAnnotate-Frame-Index':'7','X-TraceAnnotate-Source-PTS':'700','X-TraceAnnotate-Proxy-PTS':'700','X-TraceAnnotate-Media-SHA256':'a'.repeat(64),'X-TraceAnnotate-Image-SHA256':'0'.repeat(64)}[name]}},arrayBuffer:async()=>phase==='bytes'?wait.promise:new Uint8Array([1]).buffer}};
 c.Image=class{decode(){return phase==='decode'?wait.promise:Promise.resolve()}};
 const pending=c.showFrame('7');
 // Advance enough microtasks to reach a pending image/decode without timers.
 for(let i=0;i<12;i++)await Promise.resolve();
 h.get('frame-mode').checked=false;c.releaseFrame();
 wait.resolve(phase==='row'?{frames:[row]}:phase==='bytes'?new Uint8Array([1]).buffer:undefined);
 await pending;
 assert.equal(h.state.frame,null,phase);assert.equal(h.get('frame-image').hidden,true,phase);
 if(phase==='row')assert.equal(imageRequests,0);
 if(phase==='decode')assert.ok(h.revoked.includes('blob:frame'));
}
const h=harness();h.state.loadingTask=true;h.context.json=async()=>{throw new Error('No frame request while switching')};await h.context.showFrame('7');assert.equal(h.state.frameSequence,0);
""")


def test_boundary_ceil_uses_endpoint_sentinel_and_counts_available_images():
    run_ui(r"""
const h=harness(),c=h.context;h.media.frame_count='3';h.media.video_end={n:'3',d:'1'};
const annotation={start:{seconds:{n:'0',d:'1'}},end:{seconds:{n:'9',d:'1'}}};
c.selectedSegment=()=>annotation;c.nearestFrameIndex=async(target,media)=>{assert.equal(media,h.media);return target.n==='0'?'0':'3'};
const decoded=[];c.verifiedFramePng=async(index,media)=>{assert.equal(media,h.media);decoded.push(index);return {row:{frame_index:index,source_pts:index,proxy_pts:index,seconds:{n:index,d:'1'}},url:'blob:'+index}};
await c.loadCompare();assert.deepEqual(decoded,['0','2']);
const text=h.get('compare-status').textContent;
assert.ok(text.includes('终点边界 9 s'));assert.ok(text.includes('画面终点 3 s'));assert.ok(text.includes('已核对 2 幅'));
assert.ok(text.includes('无后继视频帧'));assert.ok(!text.includes('首帧 #3'));assert.ok(!text.includes('四格图像均'));
const lookup=harness();lookup.context.frameAtTime=async(target,mode,media)=>{assert.equal(mode,'ceil');assert.equal(media,lookup.media);return {frame_index:'1000'}};
assert.equal(await lookup.context.nearestFrameIndex({n:'40',d:'1'},lookup.media),'1000');
""")


def test_boundary_lookup_and_image_results_are_fenced_after_each_await():
    run_ui(r"""
for(const phase of ['start','end','image']){
 const h=harness(),c=h.context,wait=deferred();let lookups=0,decodes=0,published=0;
 c.selectedSegment=()=>({start:{seconds:{n:'0',d:'1'}},end:{seconds:{n:'1',d:'1'}}});
 c.nearestFrameIndex=async()=>{lookups++;return phase==='start'&&lookups===1||phase==='end'&&lookups===2?wait.promise:lookups===1?'0':'10'};
 c.verifiedFramePng=async(index,media)=>{decodes++;assert.equal(media,h.media);return wait.promise};c.compareSet=()=>published++;
 const pending=c.loadCompare();for(let i=0;i<8;i++)await Promise.resolve();
 h.state.task={id:'task-B'};h.state.media={...h.media,id:'media-B'};h.state.compareToken++;
 wait.resolve(phase==='image'?{row:{frame_index:'0'},url:'blob:stale'}:'0');await pending;
 assert.equal(published,0,phase);
 if(phase==='start')assert.equal(lookups,1);
 if(phase==='end')assert.equal(decodes,0);
 if(phase==='image')assert.ok(h.revoked.includes('blob:stale'));
}
""")
