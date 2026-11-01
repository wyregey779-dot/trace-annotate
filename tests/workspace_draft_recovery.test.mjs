// State-machine regression tests only: no browser is launched or controlled.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../trace_annotate/workspace/ui.py', import.meta.url), 'utf8');
function implementation(name, next) {
  const start = source.indexOf(`async function ${name}(`);
  const end = source.indexOf(`async function ${next}(`, start + 1);
  assert.ok(start >= 0 && end > start, `Cannot extract ${name} from the shipped editor`);
  return source.slice(start, end);
}
const restoreSource = implementation('restoreDraftEnvelope', 'listRecoverableDrafts');
const listSource = implementation('listRecoverableDrafts', 'saveSegment');
const copy = value => JSON.parse(JSON.stringify(value));
const hash = 'a'.repeat(64);
function envelope(text = 'recovered words') {
  const form = { label: 'scene', text, track: 'main', tags: '', start: '0', end: '1', reviewed: false };
  return { kind: 'traceannotate-draft-rescue', schema_version: 1, project_id: 'project-a',
    task_id: 'task-a', media_id: 'media-a', source_sha256: hash, playback_sha256: hash,
    base_revision: 0, base_sha256: hash, draft_id: 'draft-a', editor_session_id: 'editor-a',
    draft_revision: 0, local_revision: 1, saved_at: '2026-11-01T00:00:00Z',
    current_form: form, form_base: { ...form, text: '' }, anchors: {}, anchor_base: {},
    original: null, selected_id: null, pending_command: null, dirty: true };
}
function harness({ local = [], server = [], confirm, fetch } = {}) {
  const options = [], notices = [], restored = [], stashed = [], recovered = [];
  const form = envelope('current unsaved form').current_form;
  let confirms = 0;
  const nodes = { 'saved-drafts': { replaceChildren() { options.length = 0; }, append(option) { options.push(option); } },
    'restore-saved-draft': {}, 'draft-status': {} };
  const state = { task: { id: 'task-a', revision: 0, content_sha256: hash, annotations: [] },
    media: { id: 'media-a', source: { sha256: hash }, playback: { sha256: hash } },
    snapshot: { project_id: 'project-a' }, sequence: 1, busy: false, pending: null, loadingTask: false,
    dirty: true, anchors: {}, anchorBase: {}, formBase: { ...form, text: '' }, selected: null, original: null };
  const storage = local.map((value, i) => [`traceannotate-draft:project-a:${i}`, typeof value === 'string' ? value : JSON.stringify(value)]);
  const context = { state, $: id => nodes[id], clone: copy, restoringDraft: false, currentForm: () => copy(form),
    json: fetch || (async () => ({ drafts: server.map(body => ({ task_id: body.task_id, body })) })),
    document: { createElement: () => ({}) },
    localStorage: { get length() { return storage.length; }, key: i => storage[i][0], getItem: key => storage.find(([k]) => k === key)?.[1] },
    confirmAction: async () => { confirms++; return confirm ? confirm(state, form) : true; },
    restorePendingDraft: value => restored.push(copy(value)), stashPending: () => stashed.push(copy(state.pending)),
    recoverPending: async () => recovered.push(copy(state.pending)),
    notice: (...args) => notices.push(args), renderSaveState() {}, queueDraft() {} };
  vm.createContext(context);
  vm.runInContext(restoreSource + '\n' + listSource, context);
  return { context, state, form, options, notices, restored, stashed, recovered, nodes, confirms: () => confirms };
}

test('a newer unsynchronized local copy remains selectable alongside its older server copy', async () => {
  const old = envelope('older synchronized words');
  const newer = { ...envelope('newest unsynchronized words'), local_revision: 2, saved_at: '2026-11-01T00:00:01Z' };
  const h = harness({ local: [newer], server: [old] });
  await h.context.listRecoverableDrafts();
  const texts = h.options.map(option => JSON.parse(option.value).current_form.text);
  assert.ok(texts.includes(newer.current_form.text), 'Server deduplication hid the newer browser copy');
  assert.ok(texts.includes(old.current_form.text), 'Do not discard the other recoverable copy');
  assert.match(h.options[0].textContent, /浏览器/);
});

test('one malformed local record does not hide a separate valid recovery copy', async () => {
  const h = harness({ local: ['{broken-json', envelope()] });
  await h.context.listRecoverableDrafts();
  assert.equal(h.options.length, 1);
  assert.equal(JSON.parse(h.options[0].value).current_form.text, 'recovered words');
});

test('a stale list request cannot replace the recovery list after switching away and back', async () => {
  let release;
  const h = harness({ fetch: () => new Promise(resolve => { release = resolve; }) });
  const pending = h.context.listRecoverableDrafts();
  h.state.sequence += 2;
  const current = { value: 'new-task-session', textContent: 'current session' };
  h.options.push(current);
  release({ drafts: [{ task_id: 'task-a', body: envelope() }] });
  await pending;
  assert.equal(h.options.length, 1);
  assert.equal(h.options[0], current);
});

test('recovery cannot clear an already pending save or replace its form', async () => {
  const h = harness();
  const pending = { command: { command_id: 'unconfirmed-command' } };
  h.state.pending = pending;
  await assert.rejects(h.context.restoreDraftEnvelope(envelope()));
  assert.equal(h.state.pending, pending);
  assert.equal(h.restored.length, 0);
  assert.equal(h.confirms(), 0);
});

for (const field of ['pending', 'busy', 'loadingTask']) {
  test(`recovery rechecks ${field} after asynchronous confirmation`, async () => {
    const next = field === 'pending' ? { command: { command_id: 'save-started-during-confirmation' } } : true;
    const h = harness({ confirm: state => { state[field] = next; return true; } });
    await assert.rejects(h.context.restoreDraftEnvelope(envelope()));
    assert.equal(h.state[field], next);
    assert.equal(h.restored.length, 0);
  });
}

test('recovery cannot apply to a task switched during confirmation', async () => {
  const h = harness({ confirm: state => { state.sequence++; state.task = { id: 'task-b' }; return true; } });
  await assert.rejects(h.context.restoreDraftEnvelope(envelope()));
  assert.equal(h.restored.length, 0);
});

test('an unlocked valid recovery still restores the intended form without submitting a command', async () => {
  const h = harness();
  await h.context.restoreDraftEnvelope(envelope());
  assert.equal(h.restored.length, 1);
  assert.equal(h.restored[0].draft.form.text, 'recovered words');
  assert.equal(h.state.pending, null);
  assert.equal(h.state.conflict, false);
  assert.equal(h.recovered.length, 0);
});

test('a pending command inside a rescue remains recoverable when no other save is active', async () => {
  const h = harness(), draft = envelope();
  const command = { command_id: 'original-command', expected_revision: 0, expected_sha256: hash,
    operation: 'add_annotation', payload: { annotation: { id: 'ann-a' } } };
  draft.pending_command = { projectId: draft.project_id, taskId: draft.task_id, mediaId: draft.media_id,
    endpoint: '/tasks/task-a/commands', command, body: JSON.stringify(command) };
  await h.context.restoreDraftEnvelope(draft);
  assert.equal(h.recovered.length, 1);
  assert.equal(h.recovered[0].command.command_id, command.command_id);
  assert.equal(h.stashed[0].body, JSON.stringify(command));
});

test('a pending save created while a rescue file is being read is retained', async () => {
  const h = harness(), pending = { command: { command_id: 'save-during-file-read' } };
  const file = { size: 500, async text() { h.state.pending = pending; return JSON.stringify(envelope()); } };
  await h.context.restoreDraftFile(file);
  assert.equal(h.state.pending, pending);
  assert.equal(h.restored.length, 0);
  assert.match(h.notices[0][0], /恢复未完成/);
});

test('new words typed while confirmation is pending are not replaced by an older rescue', async () => {
  const h = harness({ confirm: (state, form) => { form.text = 'typed during confirmation'; return true; } });
  await assert.rejects(h.context.restoreDraftEnvelope(envelope()));
  assert.equal(h.form.text, 'typed during confirmation');
  assert.equal(h.restored.length, 0);
});

test('a dirty state change during confirmation invalidates the restoration decision', async () => {
  const h = harness({ confirm: state => { state.dirty = false; return true; } });
  await assert.rejects(h.context.restoreDraftEnvelope(envelope()));
  assert.equal(h.restored.length, 0);
});

test('new words typed while a rescue file is being read remain untouched', async () => {
  const h = harness();
  const file = { size: 500, async text() { h.form.text = 'typed while file was loading'; return JSON.stringify(envelope()); } };
  await h.context.restoreDraftFile(file);
  assert.equal(h.form.text, 'typed while file was loading');
  assert.equal(h.restored.length, 0);
  assert.equal(h.confirms(), 0);
});

test('a task switch while a rescue file is read cannot silently target the newly opened task', async () => {
  const h = harness();
  const file = { size: 500, async text() { h.state.sequence += 2; return JSON.stringify(envelope()); } };
  await h.context.restoreDraftFile(file);
  assert.equal(h.restored.length, 0);
  assert.equal(h.confirms(), 0);
});

test('canceling recovery preserves the existing editor and pending state', async () => {
  const h = harness({ confirm: () => false });
  await h.context.restoreDraftEnvelope(envelope());
  assert.equal(h.restored.length, 0);
  assert.equal(h.state.pending, null);
  assert.equal(h.form.text, 'current unsaved form');
});
