"""Execute shipped segment playback with controlled native-media event ordering.

The fake media clock deliberately rounds seeks down to microseconds, as observed
in Chrome. Animation callbacks are independent of media events so background-tab
delivery and callbacks left over from a cancelled playback can be reproduced.
This does not certify compositor-frame or sample-accurate playback boundaries.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def run_playback(program):
    node = shutil.which("node")
    if not node:
        pytest.skip("Real Node is required for segment-playback UI checks")
    source = (Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py").read_text("utf-8")
    implementation = source[source.index("function clearPlaybackRAF("):source.index("function keyboardTargetIsInteractive(")]
    bindings = source[source.index(" $('player').addEventListener('timeupdate'"):source.index("$('frame-mode').addEventListener('change'")]
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const [implementation,bindings]=JSON.parse(process.argv[1]);
function deferred(){let resolve;const promise=new Promise(done=>resolve=done);return {promise,resolve};}
function harness({deferredPlay=false}={}){
 const fields=new Map(),callbacks=new Map(),cancelled=new Map(),timers=new Map(),seeks=[];
 let rafId=0,timerId=0;
 const get=id=>{if(!fields.has(id))fields.set(id,{checked:false,value:'1',options:[],hidden:false,events:new Map(),addEventListener(name,callback){this.events.set(name,callback)}});return fields.get(id)};
 const state={task:{id:'task-A'},media:{duration:{n:'200',d:'1'}},playGeneration:0,playIntent:false,playWaiting:false,playError:null,segmentPlayback:null,playRAF:null,loadingTask:false};
 const player=get('player'),playWait=deferred();
 Object.assign(player,{_time:0,duration:200,readyState:4,paused:true,ended:false,playbackRate:1,
  play(){this.paused=false;if(!deferredPlay)this.emit('play');return deferredPlay?playWait.promise:Promise.resolve()},
  pause(){this.paused=true;this.emit('pause')},emit(name){this.events.get(name)?.()}});
 Object.defineProperty(player,'currentTime',{configurable:true,get(){return this._time},set(value){seeks.push(value);this._time=Math.floor(value*1e6)/1e6}});
 const context={state,$:get,console,seconds:time=>Number(time.n)/Number(time.d),renderPlayback(){},updatePlaybackTime(){},releaseFrame(){},playbackAllowed:()=>Boolean(state.task&&!state.loadingTask),selectedSegment:()=>null,notice(){},
  requestAnimationFrame(callback){const id=++rafId;callbacks.set(id,callback);return id},
  cancelAnimationFrame(id){if(callbacks.has(id))cancelled.set(id,callbacks.get(id));callbacks.delete(id)},
  setTimeout(callback){const id=++timerId;timers.set(id,callback);return id},clearTimeout(id){timers.delete(id)}};
 vm.createContext(context);vm.runInContext(implementation+'\n'+bindings,context);
 const segment=(start=983/30,end=527/15)=>({taskId:state.task.id,annotationId:'annotation-A',start,end});
 const advance=time=>{player._time=time;player.emit('timeupdate')};
 const flush=()=>{const id=state.playRAF,callback=callbacks.get(id);assert.ok(callback,'Expected a pending animation callback');callbacks.delete(id);callback()};
 return {state,get,player,context,segment,advance,flush,seeks,callbacks,cancelled,timers,playWait};
}
(async()=>{
""" + program + r"""
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-e", harness, json.dumps([implementation, bindings])],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fractional_start_is_not_cancelled_by_native_seek_rounding():
    run_playback(r"""
for(const start of [1/30,983/30]){
 const h=harness(),segment=h.segment(start,start+2);
 h.get('loop-selection').checked=true;
 await h.context.beginPlayback({segment});
 assert.ok(h.player.currentTime<start,'Reproduce the observed rounded-down seek');
 assert.equal(h.state.segmentPlayback,segment,'An owned seek is not a manual cancellation');
 h.advance(segment.end+.2);
 assert.equal(h.seeks.at(-1),start);assert.equal(h.state.segmentPlayback,segment);
 assert.equal(h.player.paused,false);assert.equal(h.timers.size,0);
}
""")


def test_background_timeupdates_loop_without_any_animation_callback():
    run_playback(r"""
const h=harness(),segment=h.segment(10,12);h.get('loop-selection').checked=true;
await h.context.beginPlayback({segment});
for(let cycle=0;cycle<5;cycle++){
 h.advance(segment.end+.25); // No animation callback is run in this test.
 assert.equal(h.seeks.at(-1),segment.start,'Media events must enforce the bound synchronously');
 assert.equal(h.player.currentTime,segment.start);assert.equal(h.player.paused,false);
}
assert.equal(h.callbacks.size,1,'timeupdate must not multiply animation watchers');
""")


def test_background_nonloop_stops_at_original_end_without_rewriting_segment():
    run_playback(r"""
const h=harness(),segment=h.segment(983/30,527/15),original=JSON.stringify(segment);
await h.context.beginPlayback({segment});h.advance(segment.end+.2);
assert.equal(h.player.paused,true);assert.equal(h.seeks.at(-1),segment.end);
assert.equal(h.state.segmentPlayback,null);assert.equal(h.callbacks.size,0);
assert.equal(JSON.stringify(segment),original);
h.advance(100);assert.equal(h.seeks.at(-1),segment.end,'Stopped events cannot seek again');
""")


def test_native_play_before_or_after_seek_completion_keeps_selected_range():
    run_playback(r"""
for(const ordering of ['before','after']){
 const h=harness({deferredPlay:true}),segment=h.segment();h.get('loop-selection').checked=true;
 const pending=h.context.beginPlayback({segment});
 if(ordering==='before')h.player.emit('play');
 h.playWait.resolve();await pending;
 if(ordering==='after')h.player.emit('play');
 assert.equal(h.state.segmentPlayback,segment);h.advance(segment.end+.4);
 assert.equal(h.seeks.at(-1),segment.start);assert.equal(h.callbacks.size,1);
}
""")


def test_manual_timeline_seek_and_skip_exit_selected_playback():
    run_playback(r"""
for(const action of ['seek','skip','ordinary-play']){
 const h=harness(),segment=h.segment(10,12);h.get('loop-selection').checked=true;
 await h.context.beginPlayback({segment});
 if(action==='seek'){h.get('seek').value='80';h.get('seek').events.get('input')();}
 if(action==='skip')h.context.skipPlayback(5);
 if(action==='ordinary-play')await h.context.beginPlayback();
 assert.equal(h.state.segmentPlayback,null);assert.equal(h.callbacks.size,0);const seeks=h.seeks.length;
 h.advance(90);assert.equal(h.seeks.length,seeks,'An explicit navigation is not pulled back');
 h.context.pausePlayback({clearSegment:true});assert.equal(h.callbacks.size,0);
}
""")


def test_cancelled_animation_cannot_control_replacement_playback():
    run_playback(r"""
const h=harness(),old=h.segment(10,12);h.get('loop-selection').checked=true;
await h.context.beginPlayback({segment:old});const oldHandle=h.state.playRAF;
h.context.pausePlayback({clearSegment:true});const stale=h.cancelled.get(oldHandle);assert.ok(stale);
const next=h.segment(50,52);await h.context.beginPlayback({segment:next});
const liveHandle=h.state.playRAF;h.player._time=70;const count=h.seeks.length;
stale();assert.equal(h.state.playRAF,liveHandle);assert.equal(h.seeks.length,count);
assert.equal(h.callbacks.size,1);h.flush();assert.equal(h.seeks.at(-1),50);
assert.equal(h.callbacks.size,1);
""")


def test_task_switch_and_frame_mode_disable_segment_clock():
    run_playback(r"""
for(const change of ['task','loading','frame','generation']){
 const h=harness(),segment=h.segment(10,12);h.get('loop-selection').checked=true;
 await h.context.beginPlayback({segment});
 if(change==='task')h.state.task={id:'task-B'};
 if(change==='loading')h.state.loadingTask=true;
 if(change==='frame')h.get('frame-mode').checked=true;
 if(change==='generation')h.state.playGeneration++;
 const count=h.seeks.length;h.player._time=20;h.flush();
 assert.equal(h.seeks.length,count,change);assert.equal(h.callbacks.size,0,change);
 if(change!=='generation'){h.player.emit('timeupdate');assert.equal(h.seeks.length,count,change);}
}
""")


def test_foreground_loop_and_pause_resume_keep_single_animation_watcher():
    run_playback(r"""
const h=harness(),segment=h.segment(10,12);h.get('loop-selection').checked=true;
await h.context.beginPlayback({segment});h.player._time=12.1;h.flush();
assert.equal(h.seeks.at(-1),10);assert.equal(h.callbacks.size,1);
h.player.pause();assert.equal(h.callbacks.size,0);assert.equal(h.state.segmentPlayback,segment);
await h.player.play();assert.equal(h.callbacks.size,1);h.advance(12.1);
assert.equal(h.seeks.at(-1),10);
""")


def test_pending_native_seek_uses_completion_before_enforcing_old_clock():
    run_playback(r"""
for(const loop of [true,false]){
 const h=harness(),segment=h.segment();h.get('loop-selection').checked=loop;
 await h.context.beginPlayback({segment});
 h.player.seeking=true;h.player._time=100;const count=h.seeks.length;
 h.player.emit('timeupdate');h.flush();
 assert.equal(h.seeks.length,count,'The old clock during an unfinished seek is not a boundary');
 assert.equal(h.player.paused,false);
 h.player._time=Math.floor(segment.start*1e6)/1e6;h.player.seeking=false;h.player.emit('seeked');
 assert.equal(h.state.segmentPlayback,segment);assert.equal(h.player.paused,false);
 // Native video controls keep selected-loop mode until explicit workspace
 // navigation. An out-of-range native seek is constrained once it completes.
 h.player.seeking=true;h.player._time=100;h.player.emit('timeupdate');
 h.player.seeking=false;h.player.emit('seeked');
 assert.equal(h.seeks.at(-1),loop?segment.start:segment.end);
 assert.equal(h.player.paused,!loop);
}
""")


def test_loop_seek_failure_pauses_once_without_animation_retry_storm():
    run_playback(r"""
const h=harness(),segment=h.segment(10,12);h.get('loop-selection').checked=true;
await h.context.beginPlayback({segment});let attempts=0;
Object.defineProperty(h.player,'currentTime',{configurable:true,get(){return this._time},set(){attempts++;throw new Error('Seek failure')}});
h.advance(12.1);
assert.equal(attempts,1);assert.equal(h.player.paused,true);
assert.equal(h.state.segmentPlayback,null);assert.equal(h.callbacks.size,0);
assert.ok(h.state.playError);h.player.emit('timeupdate');assert.equal(attempts,1);
""")


def test_late_play_event_and_promise_cannot_resume_cancelled_range():
    run_playback(r"""
const h=harness({deferredPlay:true}),segment=h.segment();h.get('loop-selection').checked=true;
const pending=h.context.beginPlayback({segment});
h.context.pausePlayback({clearSegment:true});h.player.emit('play');
h.playWait.resolve();await pending;
assert.equal(h.state.segmentPlayback,null);assert.equal(h.player.paused,true);
assert.equal(h.state.playIntent,false);assert.equal(h.callbacks.size,0);assert.equal(h.timers.size,0);
""")


def test_native_media_end_restarts_only_the_current_loop_or_clears_selection():
    run_playback(r"""
for(const mode of ['loop','once','frame']){
 const h=harness(),segment=h.segment(10,200);h.get('loop-selection').checked=mode!=='once';
 await h.context.beginPlayback({segment});
 h.player._time=200;h.player.paused=true;h.player.ended=true;
 if(mode==='frame')h.get('frame-mode').checked=true;
 h.player.emit('ended');for(let i=0;i<8;i++)await Promise.resolve();
 if(mode==='loop'){
  assert.equal(h.seeks.at(-1),10);assert.equal(h.player.paused,false);
  assert.equal(h.state.segmentPlayback,segment);assert.equal(h.callbacks.size,1);
 }else{
  assert.equal(h.player.paused,true);assert.equal(h.state.segmentPlayback,null);
  assert.equal(h.callbacks.size,0);if(mode==='frame')assert.equal(h.get('frame-mode').checked,true);
 }
}
""")


def test_native_seek_below_owned_clock_exits_without_pulling_back():
    run_playback(r"""
for(const pending of [false,true]){
 const h=harness(),segment=h.segment(10,12);h.get('loop-selection').checked=true;
 await h.context.beginPlayback({segment});
 h.player.seeking=true;h.player._time=1;h.player.emit('timeupdate');
 assert.equal(h.state.segmentPlayback,segment,'Do not inspect an unfinished seek');
 if(pending)h.player.emit('play'); // Native seek can interrupt the owned start seek.
 const count=h.seeks.length;h.player.seeking=false;h.player.emit('seeked');
 assert.equal(h.state.segmentPlayback,null);assert.equal(h.callbacks.size,0);
 assert.equal(h.player.currentTime,1);assert.equal(h.player.paused,false);
 h.advance(40);assert.equal(h.seeks.length,count,'A manual navigation is not pulled back later');
}
""")


def test_lower_boundary_uses_observed_readback_without_a_guessed_epsilon():
    run_playback(r"""
const h=harness(),segment=h.segment();h.get('loop-selection').checked=true;
await h.context.beginPlayback({segment});const observed=h.player.currentTime;
h.player.emit('seeked');assert.equal(h.state.segmentPlayback,segment);
// Just below the actual readback is outside. No arbitrary microsecond/frame
// tolerance is allowed to extend the saved annotation or hold a manual seek.
h.player._time=observed-1e-9;h.player.emit('seeked');
assert.equal(h.state.segmentPlayback,null);assert.equal(h.player.currentTime,observed-1e-9);
assert.equal(h.callbacks.size,0);
""")


def test_failed_new_start_cannot_reuse_previous_clock_or_leave_a_live_range():
    run_playback(r"""
const h=harness(),old=h.segment();h.get('loop-selection').checked=true;
await h.context.beginPlayback({segment:old});const oldClock=h.state.segmentStartClock;
assert.equal(oldClock.active,old);const next=h.segment(50,52);
Object.defineProperty(h.player,'currentTime',{configurable:true,get(){return this._time},set(){throw new Error('Start seek failed')}});
await h.context.beginPlayback({segment:next});
assert.equal(h.player.paused,true);assert.equal(h.state.segmentPlayback,null);
assert.equal(h.state.segmentStartClock,null);assert.equal(h.callbacks.size,0);
assert.ok(h.state.playError);
""")
