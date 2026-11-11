"""Real Node checks for editor JavaScript helpers that must preserve stored bytes."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _build_changes_implementation() -> str:
    source = (
        Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"
    ).read_text("utf-8")
    begin = source.index("function buildChanges(")
    end = source.index("function restorePendingDraft(", begin)
    return source[begin:end]


def test_label_only_edit_does_not_rewrite_untouched_tag_bytes() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Real Node is required for the editor helper regression")

    program = r"""
const vm = require('node:vm'), assert = require('node:assert/strict');
const implementation = JSON.parse(process.argv[1]);
const times = {
  start: {seconds: {n: '1', d: '10'}},
  end: {seconds: {n: '1', d: '1'}}
};
const fields = {
  'segment-label': {value: 'renamed-with-CRLF-preserved'},
  'segment-text': {value: '第一行\n第二行 😀'},
  'segment-track': {value: 'main'},
  'segment-tags': {value: 'duplicate\nduplicate\ntag\nwith-newline'},
  'segment-transition': {value: ''},
  'gradient-start': {value: ''},
  'gradient-end': {value: ''},
  'segment-reviewed': {checked: false}
};
const state = {
  original: {
    label: 'original',
    text: '第一行\r\n第二行 😀',
    track: 'main',
    tags: ['duplicate', 'duplicate', 'tag\nwith-newline'],
    start: times.start,
    end: times.end,
    review_status: 'unreviewed'
  },
  formBase: {
    label: 'original',
    text: '第一行\n第二行 😀',
    track: 'main',
    tags: 'duplicate\nduplicate\ntag\nwith-newline',
    transition: '',
    gradientStart: '',
    gradientEnd: '',
    reviewed: false
  }
};
const context = {
  state,
  $: id => fields[id],
  validateForm: () => times,
  composedTags: () => ['duplicate', 'duplicate', 'tag', 'with-newline']
};
vm.createContext(context);
vm.runInContext(implementation, context);
const changes = context.buildChanges();
assert.deepEqual(Object.keys(changes).sort(), ['label']);
assert.equal(changes.label, 'renamed-with-CRLF-preserved');
"""

    completed = subprocess.run(
        [node, "-e", program, json.dumps(_build_changes_implementation())],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def _render_vision_job_implementation() -> str:
    source = (
        Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"
    ).read_text("utf-8")
    begin = source.index("function visionInputIsStale(")
    end = source.index("async function pollVision(", begin)
    return source[begin:end]


def test_vision_job_surfaces_insufficient_content_sampling() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Real Node is required for the editor helper regression")

    program = r"""
const vm = require('node:vm'), assert = require('node:assert/strict');
const implementation = JSON.parse(process.argv[1]);
const fields = {
  'vision-job': {hidden: true, textContent: ''},
  'vision-cancel': {hidden: false, disabled: false},
  'vision-apply': {hidden: false, disabled: false}
};
const insufficient = '内容变化检测至少需要 9 个覆盖同一镜头的采样帧，当前只有 8 帧；未判定为无变化。';
const state = {
  task: {id:'task-A', revision:0, content_sha256:'original'},
  visionJob: {id: 'job-12345678', task_id:'task-A', input_revision:0, input_sha256:'original', state: 'succeeded'},
  visionResult: {
    scenes: [{}, {}],
    keyframes: [{}, {}],
    module_status: {
      scene: 'success', transition: 'success', face: 'success', person:'success', landmarks:'success',
      framing: 'success', motion: 'success', actions: 'success',
      expressive: 'not_run', expressive_reason: 'no_landmark_provider_in_this_build',
      acoustics: 'success'
    },
    actions_sampling: {
      status: 'success', supported: ['走', '跑'],
      unsupported: ['拿起', '放下', '打开', '关闭', '递交'], candidates: []
    },
    model_runs: [],
    content_sampling: {
      status: 'insufficient_samples', returned: 8, required_samples: 9,
      insufficient_scenes: 1, changes: 0, filtered: 2,
      scene_results: [{status: 'insufficient_samples', message: insufficient}]
    }
  },
  busy: false, loadingTask: false, dirty: false
};
const context = {state, $: id => fields[id]};
vm.createContext(context);
vm.runInContext(implementation, context);
context.renderVisionJob();
assert.match(fields['vision-job'].textContent,
  /内容采样 insufficient_samples 8 帧 \/ 需 9 帧（1 个镜头不足）/);
assert.match(fields['vision-job'].textContent,
  /六项能力 转场=success 内容=insufficient_samples\(0变化\/2过滤\) 人体\/人脸\/景别\/运镜=success\/success\/success\/success 人脸关键点=success 动作=success\(0候选;支持走\/跑;未支持拿起\/放下\/打开\/关闭\/递交\) 表情=not_run\(no_landmark_provider_in_this_build\) 声学=success/);
assert.equal(fields['vision-apply'].hidden, false);
assert.equal(fields['vision-cancel'].hidden, true);
"""

    completed = subprocess.run(
        [node, "-e", program, json.dumps(_render_vision_job_implementation())],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
