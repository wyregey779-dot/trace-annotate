"""Additive, revision-bound annotation-video controls for the workspace."""

PANEL = r'''<details id="professional-render" class="panel" style="margin-bottom:16px;min-width:0">
 <summary style="padding:14px 16px">带标注视频与单帧预览</summary>
 <div class="panel-body" style="min-width:0;overflow-wrap:anywhere">
  <p class="meta-line">在播放代理画面外增加黑底白字标注栏，保留完整画面。先保存片段草稿；预览只验证一个真实解码帧，不代表全片渲染或语义审核通过。正文溢出会明确报错，不截字、不自动缩小字体。</p>
  <p id="render-font-info" class="meta-line" aria-live="polite">正在读取本机渲染配置…</p>
  <fieldset id="render-recipe-fields" style="min-inline-size:0;border:1px solid var(--line);border-radius:8px;padding:12px">
   <legend>标注栏配方</legend>
   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr));gap:12px">
    <label for="render-position">位置<select id="render-position"><option value="bottom">下方</option><option value="top">上方</option><option value="left">左侧</option><option value="right">右侧</option></select></label>
    <label for="render-panel-extent">标注栏厚度（像素，偶数）<input id="render-panel-extent" type="number" min="64" max="4096" step="2"></label>
    <label for="render-body-font">正文字号（像素）<input id="render-body-font" type="number" min="12" max="96" step="1"></label>
    <label for="render-title-font">标签字号（像素）<input id="render-title-font" type="number" min="12" max="128" step="1"></label>
    <label for="render-padding">内边距（像素）<input id="render-padding" type="number" min="8" max="256" step="1"></label>
    <label for="render-line-spacing">行间距（像素）<input id="render-line-spacing" type="number" min="0" max="64" step="1"></label>
   </div>
   <fieldset style="min-inline-size:0;margin:12px 0;border:0;padding:0"><legend>显示字段（至少一项）</legend><div style="display:flex;gap:12px;flex-wrap:wrap">
    <label><input id="render-field-label" type="checkbox"> 标签</label><label><input id="render-field-text" type="checkbox"> 内容</label><label><input id="render-field-interval" type="checkbox"> 时间区间</label><label><input id="render-field-track" type="checkbox"> 轨道</label><label><input id="render-field-tags" type="checkbox"> 标签组</label><label><input id="render-field-review_status" type="checkbox"> 复核状态</label>
   </div></fieldset>
   <label for="render-tracks">选择轨道（每行一个，留空表示全部）<textarea id="render-tracks" rows="2" style="width:100%;max-width:100%;box-sizing:border-box"></textarea></label>
   <p id="render-known-tracks" class="meta-line"></p>
  </fieldset>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,14rem),1fr));gap:12px;margin:12px 0">
   <label for="render-frame-index">预览帧序号（从 0 开始）<input id="render-frame-index" type="text" inputmode="numeric" value="0" aria-describedby="render-preview-help"></label>
   <label for="render-download-mib">浏览器下载上限（MiB）<input id="render-download-mib" type="number" min="1" max="2048" step="1" value="256" aria-describedby="render-download-help"></label>
  </div>
  <p id="render-download-help" class="meta-line">按实际字节限制本次下载；同时受服务端上限约束。大文件会先校验大小，超限停止，不下载残片。</p>
  <div class="form-actions" style="flex-wrap:wrap"><button id="render-preview" type="button">预览这一帧</button><button id="render-submit" type="button">生成带标注视频</button><button id="render-refresh" type="button" class="quiet">刷新渲染状态</button></div>
  <p id="render-status" class="meta-line" role="status" aria-live="polite"></p><p id="render-pending" class="meta-line" aria-live="polite"></p>
  <figure style="margin:12px 0;min-width:0"><img id="render-preview-image" hidden alt="当前已保存修订的单帧标注栏预览" style="max-width:100%;height:auto"><figcaption id="render-preview-help" class="meta-line">尚未生成预览。单帧图片不能替代完整视频验收；预览接收上限为 32 MiB。</figcaption></figure>
  <div id="render-jobs" style="display:grid;gap:10px;min-width:0"></div>
 </div>
</details>'''

SCRIPT = r'''
const renderFields=['label','text','interval','track','tags','review_status'];
const renderView={key:null,options:null,optionsProject:null,optionsPromise:null,recipeReady:false,previewSequence:0,previewController:null,previewURL:null,previewBusy:false,jobsSequence:0,jobsTimer:null,jobsFingerprint:null,submitBusy:false,wasDirty:false,downloads:new Map(),urls:new Set()};
function renderText(tag,text){const el=document.createElement(tag);el.textContent=text;return el;}
function renderBindingKey(){return state.task?state.task.id+':'+state.task.revision+':'+state.task.content_sha256:null;}
function renderPendingKey(){return 'traceannotate-render-submit:'+state.snapshot.project_id;}
function renderError(error){const messages={RENDER_TEXT_OVERFLOW:'完整正文放不进当前标注栏。请增大标注栏、调整字号或选择轨道/字段；未截字或自动缩小。',RENDER_MISSING_GLYPH:'当前字体缺少选中文字，请更换获许可的字体配置或核对文本。',RENDER_TIME_OUTSIDE_VIDEO:'所选标注超出实际画面区间，请核对视频与片段边界。',RENDER_LIMIT_EXCEEDED:'渲染超过服务资源限制，请核对尺寸和素材长度。',RENDER_FONT_UNAVAILABLE:'本机渲染字体尚未就绪。',UNSUPPORTED_RENDER_LAYOUT:'这项作业的布局版本不能由当前程序继续执行。请以新命令重新生成；已完成的受支持旧版工件仍可下载。',RENDER_LINE_BREAK_UNREPRESENTABLE:'当前宽度无法保留完整文字并满足断行规则，请增大标注栏或调整字号。',DOWNLOAD_LIMIT_EXCEEDED:'文件超过本次下载上限，已停止接收，未保存残片。'};return messages[error.code]||(error.code?describeError(error):(error.message||'操作未完成'));}
function renderInvalidatePreview(message='配方或任务已变化，请重新预览。'){
 renderView.previewSequence++;if(renderView.previewController)renderView.previewController.abort();renderView.previewController=null;renderView.previewBusy=false;
 if(renderView.previewURL){URL.revokeObjectURL(renderView.previewURL);renderView.previewURL=null;}
 const image=$('render-preview-image');image.removeAttribute('src');image.hidden=true;$('render-preview-help').textContent=message;
}
function renderAbortDownloads(){for(const controller of renderView.downloads.values())controller.abort();renderView.downloads.clear();}
function renderInteger(id,low,high){const value=$(id).value;if(!/^(0|[1-9][0-9]*)$/.test(value))throw new Error('请输入范围内的整数');const number=Number(value);if(!Number.isSafeInteger(number)||number<low||number>high)throw new Error('数值超出允许范围');return number;}
function readRenderRecipe(){
 if(!renderView.options?.available||!renderView.recipeReady)throw new Error('渲染配置尚未就绪');
 const recipe=clone(renderView.options.default_recipe);recipe.position=$('render-position').value;
 if(!['top','bottom','left','right'].includes(recipe.position))throw new Error('请选择有效位置');
 for(const [field,id,low,high] of [['panel_extent_px','render-panel-extent',64,4096],['body_font_px','render-body-font',12,96],['title_font_px','render-title-font',12,128],['padding_px','render-padding',8,256],['line_spacing_px','render-line-spacing',0,64]])recipe[field]=renderInteger(id,low,high);
 if(recipe.panel_extent_px%2)throw new Error('标注栏厚度必须为偶数像素');
 recipe.fields=renderFields.filter(field=>$('render-field-'+field).checked);if(!recipe.fields.length)throw new Error('请至少选择一个显示字段');
 recipe.tracks=$('render-tracks').value.split('\n').filter(value=>value!=='');if(new Set(recipe.tracks).size!==recipe.tracks.length||recipe.tracks.length>256)throw new Error('轨道不能重复，最多选择 256 条');
 return recipe;
}
function renderLoadPending(){
 const raw=localStorage.getItem(renderPendingKey());if(raw===null)return null;
 try{const pending=JSON.parse(raw),body=JSON.parse(pending.body);if(pending.version!==1||pending.projectId!==state.snapshot.project_id||!/^([A-Za-z][A-Za-z0-9_-]{0,95})$/.test(pending.taskId)||typeof pending.body!=='string'||pending.body.length>65536||Object.keys(body).sort().join(',')!=='command_id,expected_revision,expected_sha256,recipe'||!/^([A-Za-z][A-Za-z0-9_-]{0,95})$/.test(body.command_id)||!Number.isSafeInteger(body.expected_revision)||body.expected_revision<0||!/^[0-9a-f]{64}$/.test(body.expected_sha256)||!body.recipe||typeof body.recipe!=='object'||Array.isArray(body.recipe))throw new Error('invalid');return pending;}
 catch(_){throw new Error('待确认渲染请求损坏；原记录已保留，未自动覆盖或提交。');}
}
function renderButtons(){const blocked=!state.task||state.busy||state.pending||state.loadingTask||state.dirty||state.conflict||!renderView.options?.available||!renderView.recipeReady;
 $('render-preview').disabled=Boolean(blocked||renderView.previewBusy);$('render-submit').disabled=Boolean(blocked||renderView.submitBusy);$('render-recipe-fields').disabled=Boolean(!renderView.options?.available||renderView.submitBusy);
 try{const pending=state.snapshot?renderLoadPending():null;$('render-submit').textContent=pending?'核对/重试原渲染请求':'生成带标注视频';$('render-pending').textContent=pending?'待确认：任务 '+pending.taskId+'，输入 r'+JSON.parse(pending.body).expected_revision+'。重试沿用原始配方与命令；新输入不会覆盖它。':'';}catch(error){$('render-pending').textContent=error.message;$('render-submit').disabled=true;}
}
async function loadRenderOptions(){
 const project=state.snapshot?.project_id;if(!project)return;if(renderView.optionsProject===project&&renderView.options)return renderView.options;
 if(renderView.optionsPromise)return renderView.optionsPromise;
 renderView.optionsPromise=(async()=>{try{const options=await json('/render-options');if(state.snapshot?.project_id!==project)return null;
 renderView.options=options;renderView.optionsProject=project;
 if(!options.available){$('render-font-info').textContent='本机渲染尚不可用，请核对字体与媒体工具配置。';return options;}
 if(!options.default_recipe||!options.font||options.default_recipe.font_sha256!==options.font.sha256)throw new Error('服务端字体与配方身份不一致');
 if(!renderView.recipeReady){const recipe=options.default_recipe;$('render-position').value=recipe.position;for(const [field,id] of [['panel_extent_px','render-panel-extent'],['body_font_px','render-body-font'],['title_font_px','render-title-font'],['padding_px','render-padding'],['line_spacing_px','render-line-spacing']])$(id).value=String(recipe[field]);for(const field of renderFields)$('render-field-'+field).checked=recipe.fields.includes(field);$('render-tracks').value=recipe.tracks.join('\n');renderView.recipeReady=true;}
 $('render-font-info').textContent='字体 '+options.font.id+' · '+options.font.license+' · SHA256 '+options.font.sha256+'。画面外黑底白字；音频按冻结编码配置处理，不承诺原始音质无损。';
 const limit=Number(options.limits?.download_max_bytes);$('render-download-help').textContent='本次上限由下方 MiB 设置和服务端 '+(Number.isSafeInteger(limit)&&limit>0?(limit/1048576)+' MiB':'未声明大小')+' 共同约束。超限停止，不保存残片；下载包含完整接收后的本机 Blob 内存。';return options;
 }catch(error){renderView.options=null;$('render-font-info').textContent='渲染配置读取失败：'+renderError(error);throw error;}finally{renderView.optionsPromise=null;renderButtons();}})();return renderView.optionsPromise;
}
function renderProfessionalRendering(){
 if(!$('professional-render'))return;const key=renderBindingKey();
 if(key!==renderView.key){renderView.key=key;renderView.jobsSequence++;renderView.jobsFingerprint=null;renderInvalidatePreview('任务修订已变化，请对已保存内容重新预览。');renderAbortDownloads();$('render-jobs').replaceChildren();if(renderView.jobsTimer){clearTimeout(renderView.jobsTimer);renderView.jobsTimer=null;}
 if(state.task){$('render-known-tracks').textContent='当前已有轨道：'+Array.from(new Set(state.task.annotations.map(a=>a.track))).join(' / ');$('render-status').textContent='正在读取本任务渲染状态…';void loadRenderOptions().then(()=>refreshRenderJobs()).catch(()=>{});}}
 if(state.dirty&&!renderView.wasDirty){renderInvalidatePreview('片段草稿尚未保存，旧预览已撤下。请先保存或撤回草稿。');$('render-status').textContent='草稿尚未保存；预览和新渲染使用已保存修订。';}renderView.wasDirty=Boolean(state.dirty);renderButtons();
}
function renderCanStart(){if(!state.task||state.busy||state.pending||state.loadingTask||state.conflict)return false;if(state.dirty){notice('请先保存或撤回片段草稿，再预览或生成视频；草稿仍保留。','warning');return false;}return true;}
async function renderReadBlob(response,cap,mime,controller){
 const length=response.headers.get('Content-Length');if(length!==null&&(!/^[0-9]+$/.test(length)||Number(length)>cap)){controller.abort();throw {code:'DOWNLOAD_LIMIT_EXCEEDED'};}
 if(!(response.headers.get('Content-Type')||'').toLowerCase().startsWith(mime)){controller.abort();throw new Error('返回文件类型不符合约定');}
 if(!response.body?.getReader)throw new Error('当前浏览器不支持有界接收，请使用支持 Fetch 流的浏览器');
 const reader=response.body.getReader(),chunks=[];let size=0;try{while(true){const part=await reader.read();if(part.done)break;size+=part.value.byteLength;if(size>cap){controller.abort();await reader.cancel();throw {code:'DOWNLOAD_LIMIT_EXCEEDED'};}chunks.push(part.value);}if(length!==null&&size!==Number(length))throw new Error('接收字节数与文件声明不符');if(!size)throw new Error('返回文件为空');return new Blob(chunks,{type:mime});}finally{reader.releaseLock();}
}
async function previewProfessionalRendering(){
 if(!renderCanStart())return;const task=clone(state.task),key=renderBindingKey();let recipe,index;try{recipe=readRenderRecipe();index=renderInteger('render-frame-index',0,Number(state.media.frame_count)-1);}catch(error){notice(renderError(error),'error');return;}
 renderInvalidatePreview('正在生成真实解码帧的单帧预览…');const sequence=renderView.previewSequence,controller=new AbortController();renderView.previewController=controller;renderView.previewBusy=true;renderButtons();
 try{const response=await request('/tasks/'+encodeURIComponent(task.id)+'/render-preview',{method:'POST',body:JSON.stringify({expected_revision:task.revision,expected_sha256:task.content_sha256,recipe,frame_index:String(index)}),signal:controller.signal});
 if(response.headers.get('X-TraceAnnotate-Task-Revision')!==String(task.revision)||response.headers.get('X-TraceAnnotate-Content-SHA256')!==task.content_sha256)throw new Error('预览响应与冻结修订不一致');
 const blob=await renderReadBlob(response,32*1048576,'image/png',controller);
 if(sequence!==renderView.previewSequence||key!==renderBindingKey()||state.dirty||JSON.stringify(recipe)!==JSON.stringify(readRenderRecipe()))return;
 const url=URL.createObjectURL(blob);renderView.previewURL=url;$('render-preview-image').src=url;$('render-preview-image').hidden=false;$('render-preview-help').textContent='单帧预览 · 输入 r'+task.revision+' · 帧 #'+index+'。浏览器适配显示；尚未验证完整视频、音频或语义。';
 }catch(error){if(sequence===renderView.previewSequence){$('render-preview-help').textContent='预览未生成：'+renderError(error);notice(renderError(error),'error');}}
 finally{if(sequence===renderView.previewSequence){renderView.previewController=null;renderView.previewBusy=false;renderButtons();}}
}
async function submitProfessionalRendering(){
 if(renderView.submitBusy||!renderCanStart())return;let pending;try{pending=renderLoadPending();if(pending&&pending.taskId!==state.task.id){notice('另一任务仍有待确认渲染请求，请先返回任务 '+pending.taskId+' 核对。','warning');return;}
 if(!pending){const task=clone(state.task),recipe=readRenderRecipe();pending={version:1,projectId:state.snapshot.project_id,taskId:task.id,body:JSON.stringify({command_id:commandId('cmd-render'),expected_revision:task.revision,expected_sha256:task.content_sha256,recipe})};localStorage.setItem(renderPendingKey(),JSON.stringify(pending));}}
 catch(error){notice('未提交：'+renderError(error),'error');return;}
 renderView.submitBusy=true;renderButtons();$('render-status').textContent='正在核对冻结修订的渲染请求…';const project=state.snapshot.project_id,storageKey=renderPendingKey();
 try{await json('/tasks/'+encodeURIComponent(pending.taskId)+'/render-jobs',{method:'POST',body:pending.body});try{localStorage.removeItem(storageKey);}catch(_){notice('服务已确认受理，本地恢复记录未能清理；再次核对仍使用原命令。','warning');}if(state.snapshot?.project_id===project&&state.task?.id===pending.taskId)await refreshRenderJobs();}
 catch(error){if(error.status>=400&&error.status<500&&error.details?.commit_status==='not_committed'&&error.code!=='IDEMPOTENCY_CONFLICT'){try{localStorage.removeItem(storageKey);}catch(_){}}notice('渲染请求尚未确认：'+renderError(error)+'。未确认请求保留原命令与原配方。','warning');}
 finally{renderView.submitBusy=false;renderButtons();}
}
async function controlRenderJob(id,action){
 const key='traceannotate-render-control:'+state.snapshot.project_id+':'+id+':'+action;let body;
 try{body=localStorage.getItem(key);if(body===null){body=JSON.stringify({command_id:commandId('cmd-render-control'),action});localStorage.setItem(key,body);}const parsed=JSON.parse(body);if(Object.keys(parsed).sort().join(',')!=='action,command_id'||parsed.action!==action||typeof parsed.command_id!=='string')throw new Error('控制请求恢复记录损坏');await json('/jobs/'+encodeURIComponent(id)+'/control',{method:'POST',body});localStorage.removeItem(key);await refreshRenderJobs();}
 catch(error){notice('渲染控制结果待核对：'+renderError(error)+'；重试沿用原命令。','warning');}
}
function renderLayoutStatus(job){const saved=job.parameters?.layout_version,current=renderView.options?.layout_version;const known=typeof saved==='string'&&saved.length>0&&typeof current==='string'&&current.length>0;return {version:typeof saved==='string'&&saved.length>0?saved:'未声明',requiresConfirmation:!known||saved!==current,message:!known?'布局版本未核对；不能视为当前布局的结果。':saved!==current?'旧版布局 '+saved+'；不包含当前布局 '+current+' 的排版改进。':'当前布局 '+saved};}
function renderValidateResult(result,job){const input=result?.input_binding,current=result?.current_binding;if(result?.kind!=='traceannotate.render-result'||result.version!==1||result.job_id!==job.id||result.task_id!==job.task_id||input?.task_id!==job.task_id||input.revision!==job.input_revision||input.content_sha256!==job.input_sha256||current?.task_id!==job.task_id||!Number.isSafeInteger(current.revision)||!/^[0-9a-f]{64}$/.test(current.content_sha256)||current.revision<0||!result.manifest||typeof result.manifest!=='object'||Array.isArray(result.manifest))throw new Error('渲染工件与作业修订绑定不一致');return result;}
async function refreshRenderJobs(){
 if(!state.task)return;const taskId=state.task.id,key=renderBindingKey(),sequence=++renderView.jobsSequence;
 try{await loadRenderOptions();const queues=await json('/batches');if(sequence!==renderView.jobsSequence||key!==renderBindingKey())return;const unique=new Map();for(const batch of queues.batches)for(const job of batch.jobs)if(job.operation==='render-annotations'&&job.task_id===taskId)unique.set(job.id,job);
 const jobs=Array.from(unique.values()).reverse().slice(0,10),fingerprint=JSON.stringify([key,renderView.options?.layout_version,jobs.map(j=>[j.id,j.state,j.input_revision,j.parameters?.layout_version,j.error_code,j.commit?.artifact?.sha256])]);const active=jobs.some(j=>['queued','running','cancel_requested'].includes(j.state));
 if(fingerprint!==renderView.jobsFingerprint){const rows=[];for(const job of jobs){const row=renderText('div','');row.style.cssText='border:1px solid var(--line);border-radius:8px;padding:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0';
 const layout=renderLayoutStatus(job),states={queued:'排队中',running:'正在渲染',cancel_requested:'正在取消',cancelled:'已取消',failed:'失败',succeeded:'渲染工件已生成'};row.append(renderText('span',(states[job.state]||job.state)+' · 输入 r'+job.input_revision+' · '+layout.message+(job.error_code?' · '+renderError({code:job.error_code}):'')));
 const action=(title,callback)=>{const button=renderText('button',title);button.type='button';button.addEventListener('click',callback);row.append(button);};
 if(['queued','running'].includes(job.state))action('取消渲染',()=>controlRenderJob(job.id,'cancel'));if(job.state==='failed'&&!layout.requiresConfirmation)action('重试渲染',()=>controlRenderJob(job.id,'retry_failed'));if(job.state==='cancelled'&&!layout.requiresConfirmation)action('重新导出',()=>controlRenderJob(job.id,'retry_cancelled'));if(['failed','cancelled'].includes(job.state)&&layout.requiresConfirmation)row.append(renderText('span','请使用上方生成按钮，以当前布局创建新作业。'));
 if(job.state==='succeeded'){try{const result=renderValidateResult(await json('/jobs/'+job.id+'/render-result'),job);if(sequence!==renderView.jobsSequence||key!==renderBindingKey())return;const old=result.input_binding.revision!==state.task.revision||result.input_binding.content_sha256!==state.task.content_sha256||result.input_binding.revision!==result.current_binding.revision||result.input_binding.content_sha256!==result.current_binding.content_sha256;row.append(renderText('span',old?'旧修订工件；不含当前修改':'当前已保存修订的工件'));action('下载视频 · r'+job.input_revision,()=>downloadRenderArtifact(job,'video'));action('下载配对JSON · r'+job.input_revision,()=>downloadRenderArtifact(job,'json'));}catch(error){row.append(renderText('span','工件核验失败：'+renderError(error)));}}
 rows.push(row);}
 if(sequence!==renderView.jobsSequence||key!==renderBindingKey())return;$('render-jobs').replaceChildren(...rows);renderView.jobsFingerprint=fingerprint;}
 $('render-status').textContent=(jobs.length?'本任务显示最近 '+jobs.length+' 个渲染作业。':'本任务尚无渲染作业。')+' 输出始终固定于输入修订；视频与配对JSON来自同一作业，JSON不含历史索引。';
 if(renderView.jobsTimer)clearTimeout(renderView.jobsTimer);renderView.jobsTimer=active?setTimeout(()=>refreshRenderJobs(),1200):null;
 }catch(error){if(sequence===renderView.jobsSequence&&state.task?.id===taskId)$('render-status').textContent='渲染状态读取失败：'+renderError(error);}
}
async function downloadRenderArtifact(job,kind){
 if(!['video','json'].includes(kind)||!state.task||state.task.id!==job.task_id||renderView.downloads.has(job.id+':'+kind))return;const key=renderBindingKey(),taskId=state.task.id,handle=job.id+':'+kind,controller=new AbortController();renderView.downloads.set(handle,controller);
 try{const result=renderValidateResult(await json('/jobs/'+encodeURIComponent(job.id)+'/render-result',{signal:controller.signal}),job);if(key!==renderBindingKey()||controller.signal.aborted)return;const input=result.input_binding;const old=input.revision!==state.task.revision||input.content_sha256!==state.task.content_sha256||input.revision!==result.current_binding.revision||input.content_sha256!==result.current_binding.content_sha256;
 const layout=renderLayoutStatus(job);if((old||state.dirty||layout.requiresConfirmation)&&!await confirmAction('下载固定于输入 r'+input.revision+' 的'+(kind==='video'?'视频':'配对JSON')+'；不包含当前修改或未保存草稿。'+layout.message+' JSON不含历史索引。',{title:'下载冻结修订？',accept:'下载 r'+input.revision,cancel:'返回'}))return;
 if(taskId!==state.task?.id||key!==renderBindingKey())return;const userCap=renderInteger('render-download-mib',1,2048)*1048576,serverCap=Number(renderView.options?.limits?.download_max_bytes);if(!Number.isSafeInteger(serverCap)||serverCap<1)throw new Error('服务端未声明有界下载上限');const cap=Math.min(userCap,serverCap);
 const response=await request('/jobs/'+encodeURIComponent(job.id)+'/render-'+kind,{signal:controller.signal});if(response.headers.get('X-TraceAnnotate-Task-Revision')!==String(input.revision)||response.headers.get('X-TraceAnnotate-Content-SHA256')!==input.content_sha256){controller.abort();throw new Error('下载响应与该作业冻结修订不一致');}const blob=await renderReadBlob(response,cap,kind==='video'?'video/mp4':'application/json',controller);
 if(controller.signal.aborted||taskId!==state.task?.id||key!==renderBindingKey())return;const url=URL.createObjectURL(blob);renderView.urls.add(url);const link=document.createElement('a');link.href=url;link.download='TraceAnnotate-'+job.task_id+'-r'+input.revision+(kind==='video'?'.annotated.mp4':'.professional.json');document.body.append(link);link.click();link.remove();setTimeout(()=>{URL.revokeObjectURL(url);renderView.urls.delete(url);},30000);
 notice('已完整接收输入 r'+input.revision+' 的'+(kind==='video'?'视频':'配对JSON')+'，正在交给浏览器保存。','success');
 }catch(error){if(key===renderBindingKey())notice('下载未完成：'+renderError(error),'error');}
 finally{if(renderView.downloads.get(handle)===controller)renderView.downloads.delete(handle);}
}
function renderRecipeChanged(){renderInvalidatePreview('配方已变化，请重新生成单帧预览；已有作业仍使用各自冻结配方。');renderButtons();}
for(const id of ['render-position','render-panel-extent','render-body-font','render-title-font','render-padding','render-line-spacing','render-tracks','render-frame-index',...renderFields.map(f=>'render-field-'+f)])$(id).addEventListener('input',renderRecipeChanged);
$('render-preview').addEventListener('click',previewProfessionalRendering);$('render-submit').addEventListener('click',submitProfessionalRendering);$('render-refresh').addEventListener('click',()=>{renderView.jobsFingerprint=null;void refreshRenderJobs();});
window.addEventListener('pagehide',()=>{renderInvalidatePreview('页面已离开。');renderAbortDownloads();for(const url of renderView.urls)URL.revokeObjectURL(url);renderView.urls.clear();if(renderView.jobsTimer)clearTimeout(renderView.jobsTimer);});
'''


def enhance(html):
    marker = '  <div class="workspace-grid">'
    if html.count(marker) != 1 or html.count('</script>') != 1:
        raise RuntimeError('Workspace render insertion point is missing or ambiguous')
    return html.replace(marker, PANEL + '\n' + marker).replace('</script>', SCRIPT + '\n</script>')
