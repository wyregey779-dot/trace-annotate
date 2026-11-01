"""Additive objective-analysis panel for the existing workspace editor."""

PANEL = r'''<details id="professional-analysis" class="panel" style="margin-bottom:16px">
 <summary style="padding:14px 16px">媒体参数与客观分析</summary>
 <div class="panel-body">
  <p class="meta-line">测量播放代理的音轨与帧时序，不自动判断旁白内容或音色。原片报告与代理报告分别标明来源；保存结果后可随专业 JSON 导出。</p>
  <div class="form-actions"><button id="analysis-start" type="button">分析播放媒体</button><button id="analysis-refresh" type="button" class="quiet">刷新分析状态</button></div>
  <p id="analysis-status" class="meta-line" role="status" aria-live="polite"></p>
  <div id="analysis-jobs" style="display:grid;gap:8px;margin:12px 0"></div>
  <div id="analysis-reports" style="overflow-wrap:anywhere"></div>
 </div>
</details>'''

SCRIPT = r'''
const analysisView={key:null,loading:false,timer:null,sequence:0,candidates:new Map()};
const analysisNames={integrated_lufs:'综合响度',integrated_loudness:'综合响度',sample_peak_dbfs:'样本峰值',sample_peak:'样本峰值',true_peak_estimate_dbtp:'真峰值估计',true_peak:'真峰值估计',width:'宽度',height:'高度',sample_rate:'采样率',channels:'声道数',codec_name:'编码',frame_count:'帧数',decoded_frame_count:'解码帧数',avg_frame_rate:'声明平均帧率',verified_frame_rate:'验证帧率',color_primaries:'声明色彩原色',color_transfer:'声明传递函数',color_space:'声明矩阵',color_range:'声明色彩范围',timbre_description:'音色观察'};
const analysisKinds={technical_declared:'文件与流声明',loudness:'客观音频测量',frame_timing:'帧时序',audio_observation:'声音观察',color_statistics:'像素统计'};
function analysisPendingKey(){return 'traceannotate-analysis-submit:'+state.snapshot.project_id;}
function analysisText(tag,text){const e=document.createElement(tag);e.textContent=text;return e;}
function showAnalysisReports(data){
 const host=$('analysis-reports');host.replaceChildren();
 if(!data.reports.length){host.append(analysisText('p','尚未保存专业分析报告。'));return;}
 const assessments=new Map(data.assessment.map(a=>[a.id,a]));
 for(const report of data.reports){const d=document.createElement('details');d.style.marginBottom='10px';const a=assessments.get(report.id);const source=report.time_domain==='playback_seconds'?'播放代理':'原片';const freshness=a?.status==='stale'?' · 已失效':a?.status==='unavailable'?' · 有未测项目':'';
 d.append(analysisText('summary',(analysisKinds[report.kind]||report.kind)+' · '+source+(report.stream_index===null?'':' · 流 '+report.stream_index)+freshness));
 const info=analysisText('p','文件 '+report.file_sha256.slice(0,12)+'… · '+report.method.id+' / '+report.method.version+(a?.reason?' · '+a.reason:''));info.className='meta-line';d.append(info);
 const table=document.createElement('table');const body=document.createElement('tbody');
 for(const value of report.values){const row=document.createElement('tr');row.append(analysisText('th',analysisNames[value.name]||value.name));const cell=analysisText('td',value.status==='available'?String(value.value)+(value.unit==='none'?'':' '+value.unit):'未取得：'+(value.reason||'无可验证数据'));cell.style.whiteSpace='pre-wrap';row.append(cell);body.append(row);}table.append(body);d.append(table);host.append(d);}
}
function analysisButtons(){const disabled=!state.task||state.busy||state.pending||state.loadingTask;const start=$('analysis-start');if(start)start.disabled=Boolean(disabled);}
function renderProfessionalAnalysis(){
 if(!$('professional-analysis'))return;analysisButtons();
 if(!state.task)return;const key=state.task.id+':'+state.task.revision;
 if(key!==analysisView.key){analysisView.key=key;$('analysis-reports').replaceChildren();$('analysis-jobs').replaceChildren();$('analysis-status').textContent='正在读取本任务分析状态…';
 if(analysisView.timer){clearTimeout(analysisView.timer);analysisView.timer=null;}void refreshProfessionalAnalysis();}
}
async function refreshProfessionalAnalysis(){
 if(!state.task)return;const taskId=state.task.id,sequence=++analysisView.sequence;analysisView.loading=true;
 try{const [data,queues]=await Promise.all([json('/tasks/'+taskId+'/analysis-status'),json('/batches')]);
 if(state.task?.id!==taskId||sequence!==analysisView.sequence)return;
 if(data.revision!==state.task.revision||data.content_sha256!==state.task.content_sha256){$('analysis-status').textContent='服务器任务已变化，请刷新工作区后核对分析状态。';return;}
 showAnalysisReports(data);const jobs=new Map();for(const batch of queues.batches)for(const job of batch.jobs)if(job.operation==='analyze-media'&&job.task_id===taskId)jobs.set(job.id,job);
 const host=$('analysis-jobs');host.replaceChildren();let active=false;const savedIds=new Set(data.reports.map(r=>r.id));
 const labels={queued:'排队中',running:'正在分析',cancel_requested:'正在取消',cancelled:'已取消',failed:'失败',succeeded:'分析完成，结果为候选'};
 for(const job of Array.from(jobs.values()).reverse().slice(0,10)){const row=document.createElement('div');row.style.cssText='border:1px solid var(--line);border-radius:7px;padding:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap';
 let saved=false,candidateError=null;if(job.state==='succeeded'){try{const key=job.id+':'+job.commit?.artifact?.sha256;let candidate=analysisView.candidates.get(key);if(!candidate){candidate=await json('/jobs/'+job.id+'/analysis-result');analysisView.candidates.set(key,candidate);}saved=candidate.reports.every(r=>savedIds.has(r.id));}catch(error){candidateError=error;}}
 if(state.task?.id!==taskId||sequence!==analysisView.sequence)return;
 row.append(analysisText('span',(saved?'报告已保存':labels[job.state]||job.state)+' · 输入 r'+job.input_revision+(job.error_code?' · '+job.error_code:'')));
 const action=(name,callback)=>{const b=analysisText('button',name);b.type='button';b.addEventListener('click',callback);row.append(b);};
 if(['queued','running','cancel_requested'].includes(job.state)){active=true;if(job.state!=='cancel_requested')action('取消分析',()=>controlAnalysisJob(job.id,'cancel'));}
 if(job.state==='failed')action('重试分析',()=>controlAnalysisJob(job.id,'retry_failed'));
 if(job.state==='cancelled')action('重新分析',()=>controlAnalysisJob(job.id,'retry_cancelled'));
 if(job.state==='succeeded'&&!saved&&!candidateError)action('保存结果到标注',()=>applyAnalysisJob(job));
 if(candidateError)row.append(analysisText('span','候选读取失败：'+describeError(candidateError)));host.append(row);}
 $('analysis-status').textContent='已保存 '+data.reports.length+' 份报告 · '+(active?'后台作业仍在运行，可继续编辑。':'候选结果需明确保存；测量不等于声音听审。');
 if(analysisView.timer)clearTimeout(analysisView.timer);analysisView.timer=active?setTimeout(()=>refreshProfessionalAnalysis(),1200):null;
 }catch(error){if(state.task?.id===taskId)$('analysis-status').textContent='分析状态读取失败：'+describeError(error);}
 finally{analysisView.loading=false;analysisButtons();}
}
async function startProfessionalAnalysis(){
 if(!state.task||state.busy||state.pending||state.loadingTask)return;const task=clone(state.task);
 let pending=null;try{pending=JSON.parse(localStorage.getItem(analysisPendingKey())||'null');}catch(_){}
 if(pending&&pending.taskId!==task.id){notice('另一任务还有待核对的分析请求，请先切回该任务刷新状态。','warning');return;}
 if(!pending){pending={taskId:task.id,body:JSON.stringify({command_id:commandId('cmd-analysis'),expected_revision:task.revision,expected_sha256:task.content_sha256,stream_index:null,window:null,subject_annotation_id:null})};
 try{localStorage.setItem(analysisPendingKey(),JSON.stringify(pending));}catch(_){notice('无法可靠保存分析请求身份，未提交；请恢复浏览器存储后重试。','error');return;}}
 $('analysis-start').disabled=true;$('analysis-status').textContent='正在提交已冻结的播放代理分析…';
 try{await json('/tasks/'+pending.taskId+'/analysis-jobs',{method:'POST',body:pending.body});localStorage.removeItem(analysisPendingKey());await refreshProfessionalAnalysis();}
 catch(error){if(error.status>=400&&error.status<500&&error.details?.commit_status==='not_committed')localStorage.removeItem(analysisPendingKey());notice('分析请求尚未确认：'+describeError(error)+'。重试会复用同一命令身份。','warning');}
 finally{analysisButtons();}
}
async function controlAnalysisJob(id,action){try{await json('/jobs/'+id+'/control',{method:'POST',body:JSON.stringify({command_id:commandId('cmd-analysis-control'),action})});await refreshProfessionalAnalysis();}catch(error){notice(describeError(error),'error');}}
async function applyAnalysisJob(job){
 if(state.task?.id!==job.task_id||state.busy||state.pending||state.loadingTask)return;
 if(state.dirty){notice('请先保存或撤回当前片段草稿，再保存分析结果；草稿仍保留。','warning');return;}
 try{const result=await json('/jobs/'+job.id+'/analysis-result');const task=clone(state.task);
 if(!await confirmAction('将 '+result.reports.length+' 份播放代理客观报告保存为新修订。原片与代理分开记录；这不会给标注勾选人工审核，也不代表听过旁白或确认音色。',{title:'保存客观分析？',accept:'保存报告',cancel:'保留为候选'}))return;
 if(state.task?.id!==task.id||state.task.revision!==task.revision||state.dirty)return;
 const receipt=await saveCommand('apply_analysis_job',{}, {base:task,endpoint:'/jobs/'+job.id+'/apply-analysis'});
 if(receipt){notice('客观报告已保存 · r'+receipt.revision+'；可通过专业JSON导出。','success');await refreshProfessionalAnalysis();}
 }catch(error){notice('分析结果未保存：'+describeError(error),'error');}
}
$('analysis-start').addEventListener('click',startProfessionalAnalysis);
$('analysis-refresh').addEventListener('click',()=>refreshProfessionalAnalysis());
'''

def enhance(html):
    marker='  <div class="workspace-grid">'
    if html.count(marker)!=1:
        raise RuntimeError('Workspace analysis panel insertion point is missing')
    return html.replace(marker,PANEL+'\n'+marker).replace('</script>',SCRIPT+'\n</script>')
