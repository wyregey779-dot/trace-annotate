import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const text=fs.readFileSync(new URL('../trace_annotate/workspace/analysis_ui.py',import.meta.url),'utf8');
const script=text.match(/SCRIPT = r'''([\s\S]*?)'''/)[1];
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.style={};this.textContent='';this.disabled=false;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(){}
}
function harness(){
 const elements=new Map(),saved=new Map(),calls=[],notices=[];
 const context={document:{createElement:tag=>new Element(tag)},
  $:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
  state:{snapshot:{project_id:'proj-one'},task:{id:'tsk-one',revision:7,content_sha256:'a'.repeat(64)},
    busy:false,pending:null,loadingTask:false,dirty:false},
  localStorage:{getItem:key=>saved.get(key)||null,setItem:(key,value)=>saved.set(key,value),removeItem:key=>saved.delete(key)},
  clone:structuredClone,commandId:(prefix='cmd')=>prefix+'-stable',
  notice:(...args)=>notices.push(args),describeError:e=>e.code||e.message,
  json:async(...args)=>{calls.push(args);return {};},
  confirmAction:async()=>true,saveCommand:async()=>({revision:8}),
  setTimeout:()=>1,clearTimeout:()=>{},console};
 vm.createContext(context);vm.runInContext(script,context);
 return {context,elements,saved,calls,notices,run:code=>vm.runInContext(code,context)};
}

test('lost submit response keeps the exact original command and base across later edits',async()=>{
 const h=harness();h.run('refreshProfessionalAnalysis=async()=>{}');
 let first=true;h.context.json=async(path,options)=>{
   h.calls.push([path,options]);if(first){first=false;throw Object.assign(new Error('offline'),{code:'CONNECTION_FAILED'});}return {};};
 await h.run('startProfessionalAnalysis()');
 assert.equal(h.saved.size,1);const original=h.calls[0][1].body;
 h.context.state.task.revision=9;h.context.state.task.content_sha256='b'.repeat(64);
 await h.run('startProfessionalAnalysis()');
 assert.equal(h.calls.length,2);assert.equal(h.calls[1][1].body,original);
 assert.equal(JSON.parse(original).expected_revision,7);assert.equal(h.saved.size,0);
 assert.equal(h.context.state.task.revision,9);
});

test('unavailable persistent browser storage prevents an unidentifiable job submission',async()=>{
 const h=harness();h.context.localStorage.setItem=()=>{throw new Error('quota');};
 await h.run('startProfessionalAnalysis()');
 assert.equal(h.calls.length,0);assert.match(h.notices[0][0],/未提交/);
});

test('a pending request for another task is never replayed against the selected task',async()=>{
 const h=harness();h.saved.set('traceannotate-analysis-submit:proj-one',JSON.stringify({taskId:'tsk-other',body:'original'}));
 await h.run('startProfessionalAnalysis()');assert.equal(h.calls.length,0);
 assert.equal(h.saved.size,1);assert.match(h.notices[0][0],/另一任务/);
});

test('explicitly uncommitted rejection clears submit identity but ambiguous failure retains it',async()=>{
 for(const committed of [false,true]){const h=harness();h.context.json=async()=>{throw {status:committed?500:409,code:'FAIL',details:committed?{}:{commit_status:'not_committed'}};};
  await h.run('startProfessionalAnalysis()');assert.equal(h.saved.size,committed?1:0);}
});

test('analysis apply never discards an unsaved annotation draft',async()=>{
 const h=harness();h.context.state.dirty=true;let saves=0;h.context.saveCommand=async()=>{saves++;};
 await h.run("applyAnalysisJob({id:'job-one',task_id:'tsk-one'})");
 assert.equal(h.calls.length,0);assert.equal(saves,0);assert.equal(h.context.state.dirty,true);
});

test('changing a task or draft while its confirmation is open prevents stale application',async()=>{
 for(const change of ['task','draft']){const h=harness();let saves=0;
 h.context.json=async()=>({reports:[{id:'report-one'}]});h.context.saveCommand=async()=>{saves++;};
 h.context.confirmAction=async()=>{if(change==='task')h.context.state.task.id='tsk-other';else h.context.state.dirty=true;return true;};
 await h.run("applyAnalysisJob({id:'job-one',task_id:'tsk-one'})");assert.equal(saves,0);}
});

test('confirmed apply sends only candidate identity through the existing durable save path',async()=>{
 const h=harness();h.run('refreshProfessionalAnalysis=async()=>{}');let saved;
 h.context.json=async()=>({reports:[{id:'analysis-one',values:[{value:2.0}]}]});
 h.context.saveCommand=async(...args)=>{saved=args;return {revision:8};};
 await h.run("applyAnalysisJob({id:'job-one',task_id:'tsk-one'})");
 assert.equal(saved[0],'apply_analysis_job');assert.equal(JSON.stringify(saved[1]),'{}');
 assert.equal(saved[2].endpoint,'/jobs/job-one/apply-analysis');assert.equal(saved[2].base.revision,7);
 assert.equal('document' in saved[2],false);
});

test('late status responses cannot paint another selected task',async()=>{
 const h=harness();let finish;const delayed=new Promise(resolve=>{finish=resolve;});
 h.context.json=path=>path.endsWith('analysis-status')?delayed:Promise.resolve({batches:[]});
 const running=h.run('refreshProfessionalAnalysis()');h.context.state.task={id:'tsk-other',revision:1,content_sha256:'b'.repeat(64)};
 finish({task_id:'tsk-one',revision:7,content_sha256:'a'.repeat(64),reports:[],assessment:[]});await running;
 assert.equal(h.elements.has('analysis-reports'),false);
});

test('reports from a later server revision are not displayed as current editor state',async()=>{
 const h=harness();h.context.json=async path=>path.endsWith('analysis-status')?
  {task_id:'tsk-one',revision:8,content_sha256:'b'.repeat(64),reports:[],assessment:[]}:{batches:[]};
 await h.run('refreshProfessionalAnalysis()');assert.equal(h.elements.has('analysis-reports'),false);
 assert.match(h.elements.get('analysis-status').textContent,/服务器任务已变化/);
});

test('switching tasks clears the previous reports even while their request is in flight',async()=>{
 const h=harness();h.context.json=()=>new Promise(()=>{});
 const reports=h.context.$('analysis-reports'),jobs=h.context.$('analysis-jobs');reports.append(new Element());jobs.append(new Element());
 h.run('analysisView.loading=true; analysisView.key="tsk-old:1"; renderProfessionalAnalysis()');
 assert.equal(reports.children.length,0);assert.equal(jobs.children.length,0);
 assert.match(h.context.$('analysis-status').textContent,/本任务/);
});
