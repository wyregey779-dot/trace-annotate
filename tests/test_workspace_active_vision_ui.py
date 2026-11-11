"""Execute shipped JavaScript across save/navigation races, without a fake DOM app."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / 'trace_annotate/workspace/ui.py'


def implementation(begin, end):
    text = UI.read_text(encoding='utf-8')
    start = text.index(begin)
    return text[start:text.index(end, start)]


def node_check(program, snippets):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required to execute shipped browser state logic')
    bootstrap = r'''
const vm=require('node:vm'), assert=require('node:assert/strict');
const snippets=JSON.parse(process.argv[1]);
const fields=new Map();
const element=id=>{if(!fields.has(id))fields.set(id,{hidden:false,disabled:false,checked:false,value:'1',textContent:'',load(){}});return fields.get(id);};
const calls=[];
const context={API:'/api',console,AbortController,setTimeout:()=>77,clearTimeout:()=>{},
  state:{task:{id:'A',revision:0,content_sha256:'old',annotations:[]},media:{id:'MA'},snapshot:{project_id:'P',media:[{id:'MA',duration:{n:'6',d:'1'},video_start:{n:'0',d:'1'},video_end:{n:'6',d:'1'}}]},
    visionGeneration:0,visionJob:{id:'job-A',task_id:'A',input_revision:0,input_sha256:'old',state:'running'},visionResult:null,sequence:0,busy:false,dirty:false,loadingTask:false},
  $:element, clone:x=>JSON.parse(JSON.stringify(x)),
  renderVisionCandidates:()=>{},renderSaveState:()=>{},releaseCompare:()=>{},pausePlayback:()=>{},releaseFrame:()=>{},
  updateUndoRedoButtons:()=>{},fillForm:()=>{},renderTasks:()=>{},updatePlaybackTime:()=>{},
  listRecoverableDrafts:()=>{},loadVisionOptions:()=>{},refreshVisionStatus:()=>{},
  taskName:t=>t.id,seconds:r=>Number(r.n)/Number(r.d),showTime:r=>String(Number(r.n)/Number(r.d)),
  mayLeave:async()=>true,notice:(...x)=>calls.push(['notice',...x]),
  loadSnapshot:async()=>{},recordHistory:()=>{},stashPending:()=>{},queueDraft:()=>{},
  describeError:e=>e.message};
vm.createContext(context);
for(const snippet of snippets)vm.runInContext(snippet,context);
(async()=>{
'''
    result = subprocess.run([node, '-e', bootstrap+program+'\n})().catch(e=>{console.error(e);process.exitCode=1;});', json.dumps(snippets)],
                            capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr


def vision_snippets():
    return [implementation('function resetVision(', 'async function runVision('),
            implementation('async function switchTask(', '// Pixel layout'),
            implementation('async function acceptReceipt(', 'async function recoverPending('),
            implementation('async function applyVision(', 'function fillForm(')]


def test_saved_revision_keeps_running_job_but_old_result_cannot_apply():
    node_check(r'''
context.pollVision=()=>{calls.push(['poll',context.state.visionJob.id]);};
context.state.pending={taskId:'A',command:{command_id:'cmd-save',operation:'add_annotation',payload:{annotation:{id:'ann-1'}}}};
context.json=async path=>{if(path==='/tasks/A')return{id:'A',media_id:'MA',revision:1,content_sha256:'new',annotations:[]};throw Error('unexpected '+path);};
await context.acceptReceipt({applied:true,commit_id:'commit-1',command_id:'cmd-save',task_id:'A',revision:1});
assert.equal(context.state.visionJob?.id,'job-A','save must retain the background job '+JSON.stringify(calls));
assert.equal(context.state.visionJob.input_revision,0,'keep original frozen input identity');
assert.ok(calls.some(c=>c[0]==='poll'));
context.state.visionJob.state='succeeded';context.state.visionResult={scenes:[],keyframes:[],module_status:{}};
context.renderVisionJob();
assert.equal(element('vision-apply').disabled,true,'old revision candidates cannot silently apply');
const before=calls.length;await context.applyVision();
assert.ok(calls.slice(before).some(c=>c[0]==='notice'&&/修订/.test(c[1])),'explain the stale input');
''', vision_snippets())


def test_switching_tasks_restores_only_its_own_job():
    node_check(r'''
context.pollVision=()=>{};
context.json=async path=>({id:path.split('/').at(-1),media_id:'MA',revision:0,content_sha256:'old',annotations:[]});
await context.switchTask('B');
assert.equal(context.state.visionJob,null,'A job must not show on B');
context.state.visionJob={id:'job-B',task_id:'B',input_revision:0,input_sha256:'old',state:'running'};
await context.switchTask('A');
assert.equal(context.state.visionJob?.id,'job-A','returning to A must resume A '+JSON.stringify(calls));
await context.switchTask('B');
assert.equal(context.state.visionJob?.id,'job-B','B keeps its separate background job');
''', vision_snippets())


def test_late_poll_cannot_replace_new_tasks_job():
    node_check(r'''
let resolve;context.json=()=>new Promise(r=>{resolve=r;});
const pending=context.pollVision();
context.resetVision();context.state.task={id:'B',revision:0,content_sha256:'B'};
context.state.visionJob={id:'job-B',state:'running'};
resolve({job_id:'job-A',task_id:'A',state:'succeeded'});
await pending;
assert.equal(context.state.visionJob.id,'job-B');
assert.equal(context.state.visionResult,null);
''', [implementation('function resetVision(', 'async function runVision(')])


@pytest.mark.parametrize('terminal,expected', [('succeeded','admitted'),('failed','failed'),('cancelled','cancelled')])
def test_terminal_import_status_removes_obsolete_queue_note(terminal,expected):
    node_check(r'''
context.state.batch={items:[{intakeId:'intake-1',phase:'accepted',note:'已受理 · queued'}]};
context.state.intakes=[{id:'intake-1',state:TERMINAL,task_id:'task-A',error_code:'DECODE_FAILED'}];
assert.equal(context.syncBatchFromIntakes(),true);
const item=context.state.batch.items[0];assert.equal(item.phase,EXPECTED);
assert.ok(!/queued|已受理/.test(item.note),'terminal state must replace stale queue note');
'''.replace('TERMINAL',json.dumps(terminal)).replace('EXPECTED',json.dumps(expected)),
               [implementation('function syncBatchFromIntakes(', 'async function reconcileBatchItem(')])




def test_changed_hash_at_same_revision_blocks_apply_and_shows_provider_status():
    node_check(r'''
context.state.visionJob={id:'job-A',task_id:'A',input_revision:0,input_sha256:'different',state:'succeeded'};
context.state.visionResult={scenes:[],keyframes:[],module_status:{person:'success',landmarks:'success',expressive:'not_run'}};
context.renderVisionJob();
assert.equal(element('vision-apply').disabled,true);
assert.match(element('vision-job').textContent,/人体\/人脸/);
assert.match(element('vision-job').textContent,/人脸关键点=success/);
assert.match(element('vision-job').textContent,/表情=not_run/);
context.state.visionJob.input_sha256='old';context.renderVisionJob();
assert.equal(element('vision-apply').disabled,false);
''', [implementation('function resetVision(', 'async function runVision(')])


def test_optional_provider_options_are_not_claimed_before_available():
    node_check(r'''
context.json=async()=>({capabilities:{scene:true,person:true,landmarks:false},loaded:false});
await context.loadVisionOptions();
assert.match(element('vision-options').textContent,/人体框 可用/);
assert.match(element('vision-options').textContent,/人脸关键点 不可用/);
''', [implementation('async function loadVisionOptions(', 'function renderVisionCandidates(')])
