import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const text=fs.readFileSync(new URL('../trace_annotate/workspace/render_ui.py',import.meta.url),'utf8');
const script=text.match(/SCRIPT = r'''([\s\S]*?)'''/)[1];
const panel=text.match(/PANEL = r'''([\s\S]*?)'''/)[1];
const hash='a'.repeat(64),fontHash='f'.repeat(64);
const defaultRecipe={version:1,position:'bottom',panel_extent_px:360,body_font_px:24,title_font_px:28,padding_px:16,
 line_spacing_px:6,fields:['label','text','interval'],tracks:[],font_sha256:fontHash,font_license:'OFL-1.1',
 encoding:{video_codec:'libx264',pixel_format:'yuv420p',crf:18,preset:'medium',audio_codec:'aac',audio_bitrate_kbps:192,color_policy:'preserve_declared_proxy'}};
const layoutVersion='annotation-panel-v2-word-boundaries';
const options={available:true,layout_version:layoutVersion,renderer_version:'annotation-video-v1',font:{id:'Noto CJK',sha256:fontHash,license:'OFL-1.1'},default_recipe:defaultRecipe,limits:{download_max_bytes:512*1048576}};
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.style={};this.value='';this.checked=false;this.hidden=false;this.disabled=false;this.textContent='';this.events=new Map();this.attributes=new Map();this.replacements=0;this.clicked=0;}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;this.replacements++;}
 addEventListener(name,fn){this.events.set(name,fn);}
 removeAttribute(name){this.attributes.delete(name);delete this[name];}
 setAttribute(name,value){this.attributes.set(name,String(value));}
 getAttribute(name){return this.attributes.get(name)??null;}
 contains(element){return this===element||this.children.some(c=>c?.contains?.(element));}
 click(){this.clicked++;}
 remove(){this.removed=true;}
}
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
function plain(value){return JSON.parse(JSON.stringify(value));}
function allText(element){return element.textContent+' '+element.children.map(allText).join(' ');}
function harness(){
 const elements=new Map(),saved=new Map(),calls=[],notices=[],created=[],revoked=[],timers=[],events=new Map(),anchors=[];
 const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 const context={$:get,document:{body:new Element('body'),activeElement:null,createElement:tag=>{const e=new Element(tag);if(tag==='a')anchors.push(e);return e;}},
  window:{addEventListener:(name,fn)=>events.set(name,fn)},
  state:{snapshot:{project_id:'proj-one'},task:{id:'tsk-one',revision:7,content_sha256:hash,annotations:[{id:'ann-one',track:'camera'}]},media:{frame_count:'20'},busy:false,pending:null,loadingTask:false,dirty:false,conflict:false},
  localStorage:{getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,value),removeItem:key=>saved.delete(key)},
  clone:structuredClone,commandId:(prefix='cmd')=>prefix+'-stable',notice:(...args)=>notices.push(args),describeError:e=>e.code||e.message,
  json:async(path,request)=>{calls.push({type:'json',path,request});return path==='/render-options'?structuredClone(options):{batches:[]};},
  request:async(path,request)=>{calls.push({type:'request',path,request});return png();},confirmAction:async()=>true,
  URL:{createObjectURL:blob=>{const url='blob:test-'+(created.length+1);created.push({url,blob});return url;},revokeObjectURL:url=>revoked.push(url)},
  AbortController,Blob,Uint8Array,setTimeout:(fn,ms)=>{timers.push({fn,ms});return timers.length;},clearTimeout:id=>{if(timers[id-1])timers[id-1].cleared=true;},console,initialOptions:structuredClone(options)};
 vm.createContext(context);vm.runInContext(script,context);
 const run=code=>vm.runInContext(code,context);
 run('renderView.options=initialOptions;renderView.optionsProject="proj-one";renderView.recipeReady=true;');
 get('render-position').value='bottom';get('render-panel-extent').value='360';get('render-body-font').value='24';get('render-title-font').value='28';get('render-padding').value='16';get('render-line-spacing').value='6';get('render-tracks').value='';get('render-frame-index').value='0';get('render-download-mib').value='256';
 for(const field of defaultRecipe.fields)get('render-field-'+field).checked=true;
 return {context,elements,saved,calls,notices,created,revoked,timers,events,anchors,get,run};
}
function png(revision=7,identity=hash){return new Response(new Uint8Array([137,80,78,71]),{headers:{'Content-Type':'image/png','Content-Length':'4','X-TraceAnnotate-Task-Revision':String(revision),'X-TraceAnnotate-Content-SHA256':identity}});}
function job(revision=7,identity=hash){return {id:'job-one',task_id:'tsk-one',operation:'render-annotations',parameters:{layout_version:layoutVersion},input_revision:revision,input_sha256:identity,state:'succeeded',commit:{artifact:{sha256:'c'.repeat(64)}}};}
function result(item=job(),current=7,currentHash=hash){return {kind:'traceannotate.render-result',version:1,job_id:item.id,task_id:item.task_id,
 input_binding:{task_id:item.task_id,revision:item.input_revision,content_sha256:item.input_sha256},
 current_binding:{task_id:item.task_id,revision:current,content_sha256:currentHash},manifest:{kind:'verified-fixture-manifest'}};}

test('display controls preserve trusted font/encoding and exact selected track strings',()=>{
 const h=harness();h.get('render-position').value='right';h.get('render-body-font').value='30';h.get('render-tracks').value='camera\n precise name ';
 const recipe=plain(h.run('readRenderRecipe()'));assert.equal(recipe.position,'right');assert.equal(recipe.body_font_px,30);
 assert.equal(recipe.font_sha256,fontHash);assert.equal(recipe.font_license,'OFL-1.1');assert.deepEqual(recipe.encoding,defaultRecipe.encoding);
 assert.deepEqual(recipe.tracks,['camera',' precise name ']);assert.deepEqual(recipe.fields,defaultRecipe.fields);
});

test('invalid geometry/empty fields cannot start a preview or enqueue work',async()=>{
 for(const configure of [h=>{h.get('render-panel-extent').value='361';},h=>{h.get('render-body-font').value='11';},h=>{for(const field of defaultRecipe.fields)h.get('render-field-'+field).checked=false;}]){
  const h=harness();configure(h);await h.run('previewProfessionalRendering()');await h.run('submitProfessionalRendering()');assert.equal(h.calls.length,0);assert.equal(h.saved.size,0);assert.ok(h.notices.length);
 }
 const local=harness();local.get('render-panel-extent').value='361';await local.run('previewProfessionalRendering()');assert.match(local.notices[0][0],/必须为偶数/);
});

test('unsaved annotation drafts block previews and submissions without touching their fields',async()=>{
 const h=harness();h.context.state.dirty=true;h.get('segment-text').value='正在输入中文';h.context.document.activeElement=h.get('segment-text');
 await h.run('previewProfessionalRendering()');await h.run('submitProfessionalRendering()');assert.equal(h.calls.length,0);
 assert.equal(h.context.state.dirty,true);assert.equal(h.get('segment-text').value,'正在输入中文');assert.equal(h.context.document.activeElement,h.get('segment-text'));
});

test('lost render submission preserves identical raw body and command across recipe and revision changes',async()=>{
 const h=harness();h.run('refreshRenderJobs=async()=>{}');let first=true;
 h.context.json=async(path,request)=>{h.calls.push({path,request});if(first){first=false;throw {code:'CONNECTION_FAILED'};}return {};};
 await h.run('submitProfessionalRendering()');const raw=h.calls[0].request.body;assert.equal(h.saved.size,1);
 h.context.state.task.revision=9;h.context.state.task.content_sha256='b'.repeat(64);h.get('render-position').value='left';
 await h.run('submitProfessionalRendering()');assert.equal(h.calls[1].request.body,raw);assert.equal(JSON.parse(raw).expected_revision,7);
 assert.equal(JSON.parse(raw).recipe.position,'bottom');assert.equal(h.saved.size,0);assert.equal(h.context.state.task.revision,9);
});

test('quota failure, corrupt pending records and another task never create unidentifiable work',async()=>{
 const quota=harness();quota.context.localStorage.setItem=()=>{throw new Error('quota');};await quota.run('submitProfessionalRendering()');assert.equal(quota.calls.length,0);
 const corrupt=harness();corrupt.saved.set('traceannotate-render-submit:proj-one','not-json');await corrupt.run('submitProfessionalRendering()');assert.equal(corrupt.calls.length,0);assert.equal(corrupt.saved.get('traceannotate-render-submit:proj-one'),'not-json');
 const other=harness();other.saved.set('traceannotate-render-submit:proj-one',JSON.stringify({version:1,projectId:'proj-one',taskId:'tsk-other',body:JSON.stringify({command_id:'cmd-original',expected_revision:1,expected_sha256:hash,recipe:defaultRecipe})}));await other.run('submitProfessionalRendering()');assert.equal(other.calls.length,0);assert.equal(other.saved.size,1);
});

test('ambiguous and idempotency conflicts preserve recovery while definite rejection clears it',async()=>{
 for(const [status,code,keep] of [[500,'INTERNAL_ERROR',true],[409,'IDEMPOTENCY_CONFLICT',true],[409,'REVISION_CONFLICT',false]]){
  const h=harness();h.context.json=async()=>{throw {status,code,details:{commit_status:'not_committed'}};};await h.run('submitProfessionalRendering()');assert.equal(h.saved.size,keep?1:0);
 }
});

test('repeated activation while submission is pending sends one stable request',async()=>{
 const h=harness(),gate=deferred();h.run('refreshRenderJobs=async()=>{}');h.context.json=async(path,request)=>{h.calls.push({path,request});return gate.promise;};
 const first=h.run('submitProfessionalRendering()');await h.run('submitProfessionalRendering()');assert.equal(h.calls.length,1);gate.resolve({});await first;assert.equal(h.saved.size,0);
});

test('recipe change aborts an in-flight preview and its late image cannot become current',async()=>{
 const h=harness(),gate=deferred();let controller;
 h.context.request=async(path,request)=>{controller=request.signal;return gate.promise;};const pending=h.run('previewProfessionalRendering()');
 h.get('render-position').value='top';h.run('renderRecipeChanged()');assert.equal(controller.aborted,true);gate.resolve(png());await pending;
 assert.equal(h.created.length,0);assert.equal(h.get('render-preview-image').hidden,true);assert.match(h.get('render-preview-help').textContent,/配方已变化/);
});

test('out-of-order previews cannot replace the newer frame or allocate a stale blob URL',async()=>{
 const h=harness(),old=deferred();let call=0;
 h.context.request=async()=>++call===1?old.promise:png();const first=h.run('previewProfessionalRendering()');h.get('render-frame-index').value='3';
 await h.run('previewProfessionalRendering()');assert.match(h.get('render-preview-help').textContent,/帧 #3/);old.resolve(png());await first;
 assert.equal(h.created.length,1);assert.equal(h.get('render-preview-image').src,h.created[0].url);assert.match(h.get('render-preview-help').textContent,/单帧预览/);
});

test('preview sends a canonical decimal frame string in the exact four-field envelope',async()=>{
 const h=harness();h.get('render-frame-index').value='3';await h.run('previewProfessionalRendering()');const body=JSON.parse(h.calls[0].request.body);
 assert.equal(body.frame_index,'3');assert.deepEqual(Object.keys(body).sort(),['expected_revision','expected_sha256','frame_index','recipe']);
});

test('switching tasks clears old previews and aborts pending requests without altering focus',async()=>{
 const h=harness();await h.run('previewProfessionalRendering()');const old=h.created[0].url;h.get('render-tracks').value='my custom input';
 h.context.document.activeElement=h.get('render-tracks');h.run('refreshRenderJobs=async()=>{}');h.context.state.task={id:'tsk-two',revision:0,content_sha256:'b'.repeat(64),annotations:[]};
 h.run('renderProfessionalRendering()');assert.ok(h.revoked.includes(old));assert.equal(h.get('render-preview-image').hidden,true);
 assert.equal(h.get('render-tracks').value,'my custom input');assert.equal(h.context.document.activeElement,h.get('render-tracks'));
});

test('mismatching response binding and explicit overflow fail without showing a false preview',async()=>{
 const h=harness();h.context.request=async()=>png(8);await h.run('previewProfessionalRendering()');assert.equal(h.created.length,0);assert.match(h.get('render-preview-help').textContent,/冻结修订不一致/);
 const overflow=harness();overflow.context.request=async()=>{throw {code:'RENDER_TEXT_OVERFLOW'};};await overflow.run('previewProfessionalRendering()');assert.equal(overflow.created.length,0);assert.match(overflow.get('render-preview-help').textContent,/未截字或自动缩小/);
});

test('a late job-list response never paints another selected task',async()=>{
 const h=harness(),gate=deferred();h.context.json=async()=>gate.promise;const pending=h.run('refreshRenderJobs()');await Promise.resolve();
 h.context.state.task={id:'tsk-two',revision:0,content_sha256:'b'.repeat(64),annotations:[]};const marker=new Element();marker.textContent='new task';h.get('render-jobs').replaceChildren(marker);
 gate.resolve({batches:[{jobs:[job()]}]});await pending;assert.match(allText(h.get('render-jobs')),/new task/);
});

test('server-current binding marks a completed job old even if editor state is stale',async()=>{
 const h=harness();h.context.json=async path=>path==='/batches'?{batches:[{jobs:[job()]}]}:result(job(),8,'b'.repeat(64));
 await h.run('refreshRenderJobs()');assert.match(allText(h.get('render-jobs')),/旧修订工件/);assert.match(allText(h.get('render-jobs')),/输入 r7/);
 const replacements=h.get('render-jobs').replacements;h.context.document.activeElement=h.get('render-tracks');await h.run('refreshRenderJobs()');assert.equal(h.get('render-jobs').replacements,replacements);assert.equal(h.context.document.activeElement,h.get('render-tracks'));
});

test('invalid render-result identities never produce download actions',async()=>{
 const h=harness();h.context.json=async path=>path==='/batches'?{batches:[{jobs:[job()]}]}:{...result(),job_id:'job-other'};
 await h.run('refreshRenderJobs()');assert.match(allText(h.get('render-jobs')),/工件核验失败/);assert.doesNotMatch(allText(h.get('render-jobs')),/下载视频/);
});

test('same-revision historical layouts remain visibly distinct and require confirmation before download',async()=>{
 const h=harness(),historical={...job(),parameters:{layout_version:'annotation-panel-v1'}};h.context.historical=historical;
 h.context.json=async path=>path==='/batches'?{batches:[{jobs:[historical]}]}:result(historical);
 await h.run('refreshRenderJobs()');assert.match(allText(h.get('render-jobs')),/旧版布局 annotation-panel-v1/);assert.match(allText(h.get('render-jobs')),/不包含当前布局/);
 let message;h.context.confirmAction=async text=>{message=text;return false;};
 await h.run("downloadRenderArtifact(historical,'video')");assert.match(message,/旧版布局/);assert.match(message,/输入 r7/);
 assert.equal(h.calls.length,0);assert.equal(h.created.length,0);assert.equal(h.anchors.length,0);
});

test('missing layout metadata is disclosed and a rejected confirmation never requests the file',async()=>{
 const h=harness(),unknown={...job(),parameters:{}};h.context.unknown=unknown;h.context.json=async()=>result(unknown);
 let message;h.context.confirmAction=async text=>{message=text;return false;};await h.run("downloadRenderArtifact(unknown,'json')");
 assert.match(message,/布局版本未核对/);assert.equal(h.calls.length,0);assert.equal(h.created.length,0);
});

test('a cancelled historical layout offers a new generation instead of resuming with a changed algorithm',async()=>{
 const h=harness(),historical={...job(),state:'cancelled',commit:null,parameters:{layout_version:'annotation-panel-v1'}};
 h.context.json=async()=>({batches:[{jobs:[historical]}]});await h.run('refreshRenderJobs()');
 assert.match(allText(h.get('render-jobs')),/以当前布局创建新作业/);assert.doesNotMatch(allText(h.get('render-jobs')),/重新导出/);
});

test('cancelled render exposes resume of its existing job rather than another cached batch',async()=>{
 const h=harness(),cancelled={...job(),state:'cancelled',commit:null};h.context.json=async()=>({batches:[{jobs:[cancelled]}]});
 await h.run('refreshRenderJobs()');assert.match(allText(h.get('render-jobs')),/重新导出/);
 const row=h.get('render-jobs').children[0],button=row.children.find(c=>c.tag==='button'&&c.textContent==='重新导出');assert.ok(button);
 h.run('refreshRenderJobs=async()=>{}');h.context.json=async(path,request)=>{h.calls.push({path,request});return {};};
 await button.events.get('click')();assert.equal(h.calls[0].path,'/jobs/job-one/control');assert.equal(JSON.parse(h.calls[0].request.body).action,'retry_cancelled');
 assert.equal(h.calls.some(call=>call.path.endsWith('/render-jobs')),false);
});

test('lost cancellation/resume control response reuses its original control command',async()=>{
 const h=harness();h.run('refreshRenderJobs=async()=>{}');let first=true;
 h.context.json=async(path,request)=>{h.calls.push({path,request});if(first){first=false;throw {code:'CONNECTION_FAILED'};}return {};};
 await h.run("controlRenderJob('job-one','retry_cancelled')");const raw=h.calls[0].request.body;assert.equal(h.saved.size,1);
 await h.run("controlRenderJob('job-one','retry_cancelled')");assert.equal(h.calls[1].request.body,raw);assert.equal(h.saved.size,0);
});

test('video and JSON downloads use the same frozen job revision, not current task export',async()=>{
 for(const kind of ['video','json']){const h=harness(),frozen=job(5,'d'.repeat(64));h.context.frozen=frozen;let message;
  h.context.confirmAction=async text=>{message=text;return true;};h.context.json=async()=>result(frozen);
  h.context.request=async(path,request)=>{h.calls.push({path,request});return new Response(new Uint8Array([1,2,3]),{headers:{'Content-Type':kind==='video'?'video/mp4':'application/json','Content-Length':'3','X-TraceAnnotate-Task-Revision':'5','X-TraceAnnotate-Content-SHA256':'d'.repeat(64)}});};
  await h.run(`downloadRenderArtifact(frozen,'${kind}')`);assert.equal(h.calls[0].path,'/jobs/job-one/render-'+kind);assert.match(message,/输入 r5/);assert.match(message,/不包含当前修改/);
  assert.equal(h.anchors.length,1);assert.match(h.anchors[0].download,/-r5\./);assert.equal(h.anchors[0].clicked,1);assert.equal(h.created.length,1);
  assert.equal(h.calls.some(call=>call.path.includes('/export')),false);h.timers.find(t=>t.ms===30000).fn();assert.ok(h.revoked.includes(h.created[0].url));
 }
});

test('a task switch during old-revision confirmation prevents all file requests',async()=>{
 const h=harness();h.context.frozen=job(5,'d'.repeat(64));h.context.json=async()=>result(h.context.frozen);
 h.context.confirmAction=async()=>{h.context.state.task.id='tsk-other';return true;};await h.run("downloadRenderArtifact(frozen,'video')");
 assert.equal(h.calls.length,0);assert.equal(h.created.length,0);
});

test('double download activation before manifest resolution does not receive the file twice',async()=>{
 const h=harness(),gate=deferred();h.context.item=job();let manifests=0;
 h.context.json=async()=>{manifests++;return gate.promise;};h.context.request=async(path,request)=>{h.calls.push({path,request});return new Response(new Uint8Array([1]),{headers:{'Content-Type':'video/mp4','Content-Length':'1','X-TraceAnnotate-Task-Revision':'7','X-TraceAnnotate-Content-SHA256':hash}});};
 const first=h.run("downloadRenderArtifact(item,'video')");await h.run("downloadRenderArtifact(item,'video')");assert.equal(manifests,1);gate.resolve(result());await first;
 assert.equal(h.calls.length,1);assert.equal(h.anchors[0].clicked,1);
});

test('download binding headers must match the frozen input rather than the newer current task',async()=>{
 for(const [revision,identity] of [['8','b'.repeat(64)],['7','b'.repeat(64)],['',hash]]){const h=harness();h.context.item=job();h.context.json=async()=>result();
 h.context.request=async()=>new Response(new Uint8Array([1]),{headers:{'Content-Type':'video/mp4','Content-Length':'1','X-TraceAnnotate-Task-Revision':revision,'X-TraceAnnotate-Content-SHA256':identity}});
 await h.run("downloadRenderArtifact(item,'video')");assert.equal(h.created.length,0);assert.equal(h.anchors.length,0);assert.match(h.notices[0][0],/冻结修订不一致/);}
});

test('streaming download caps enforce actual bytes even without Content-Length',async()=>{
 const h=harness();h.context.response=new Response(new Uint8Array([1,2,3,4,5]),{headers:{'Content-Type':'video/mp4'}});h.context.control=new AbortController();
 await assert.rejects(h.run("renderReadBlob(response,4,'video/mp4',control)"),error=>error.code==='DOWNLOAD_LIMIT_EXCEEDED');assert.equal(h.context.control.signal.aborted,true);assert.equal(h.created.length,0);
});

test('declared oversized and truncated files fail before publishing a download',async()=>{
 const h=harness();h.context.control=new AbortController();h.context.response=new Response(new Uint8Array([1]),{headers:{'Content-Type':'video/mp4','Content-Length':'8'}});
 await assert.rejects(h.run("renderReadBlob(response,4,'video/mp4',control)"),error=>error.code==='DOWNLOAD_LIMIT_EXCEEDED');
 h.context.control=new AbortController();h.context.response=new Response(new Uint8Array([1]),{headers:{'Content-Type':'video/mp4','Content-Length':'3'}});
 await assert.rejects(h.run("renderReadBlob(response,4,'video/mp4',control)"),/字节数/);assert.equal(h.created.length,0);
});

test('leaving the page aborts owned work and revokes preview/download object URLs',async()=>{
 const h=harness();await h.run('previewProfessionalRendering()');h.run("renderView.urls.add('blob:download');renderView.downloads.set('x',new AbortController());");
 const controller=h.run("renderView.downloads.get('x')");h.events.get('pagehide')();assert.equal(controller.signal.aborted,true);
 assert.ok(h.revoked.includes('blob:download'));assert.ok(h.revoked.includes(h.created[0].url));assert.equal(h.events.has('keydown'),false);
});

test('panel has unique semantic controls and hidden preview is not overridden by inline display',()=>{
 const ids=Array.from(panel.matchAll(/\bid="([^"]+)"/g),match=>match[1]);assert.equal(ids.length,new Set(ids).size);
 for(const id of ['render-position','render-panel-extent','render-body-font','render-title-font','render-tracks','render-frame-index','render-download-mib'])assert.ok(panel.includes(`for="${id}"`));
 assert.match(panel,/role="status" aria-live="polite"/);assert.doesNotMatch(panel,/<img[^>]*style="[^"]*display:/);assert.match(panel,/单帧图片不能替代完整视频验收/);
});
