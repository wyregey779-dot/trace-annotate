import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import test from 'node:test';

// Executes the shipped inline JavaScript with a small DOM/storage test double.
// This verifies persistence decisions, not real browser playback, layout or IME.
const editorPath = process.env.TRACE_TEMPORAL_EDITOR_HTML || fileURLToPath(new URL('../trace_annotate/temporal/ui/annotator.html', import.meta.url));
const html = readFileSync(editorPath, 'utf8');
const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1].replace('loadEditor(); render(); loadPreparedProject();', '');
const digest = 'a'.repeat(64);
const primary = 'traceannotate.temporal.v2:sha256:' + digest;
const legacyKey = 'traceannotate.temporal.v1:video.mp4:100';
let writer = 0;

function doc(label = null) {
  return { kind: 'temporal', schema_version: 2, clip: 'video.mp4', duration: 10, media_size: 100, media_sha256: digest,
    segments: label === null ? [] : [{ id: 'seg-1', start: 1, end: 4, label, text: 'Description\n第二行', tags: ['旅行'], track: '主轨', review_status: 'reviewed' }] };
}
function storage() {
  const data = new Map();
  return { data, beforeRead: null, beforeWrite: null, get length() { return data.size; }, key(i) { return [...data.keys()][i] ?? null; },
    getItem(key) { const value = data.get(key) ?? null; this.beforeRead?.(key, value); return value; },
    setItem(key, value) { this.beforeWrite?.(key, value); data.set(key, String(value)); }, removeItem(key) { data.delete(key); } };
}
function element() {
  return { value: '', checked: false, hidden: false, disabled: false, textContent: '', children: [], handlers: {}, style: {}, dataset: {},
    classList: { toggle() {}, contains() { return false; } }, setAttribute() {}, removeAttribute() {}, focus() {}, matches() { return false; },
    append(...nodes) { this.children.push(...nodes); }, replaceChildren(...nodes) { this.children = nodes; }, cloneNode() { return element(); },
    addEventListener(name, handler) { this.handlers[name] = handler; } };
}
function page(store, options = {}) {
  const nodes = new Map(), events = new Map();
  const get = id => { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); };
  const video = get('videoPlayer'); video.currentTime = 0; video.duration = 10;
  const context = vm.createContext({ localStorage: store, crypto: options.crypto || { randomUUID: () => 'writer-' + (++writer) },
    document: { getElementById: get, querySelectorAll: () => [], createElement: element, activeElement: null },
    window: { confirm: () => options.confirm !== false, addEventListener: (name, handler) => events.set(name, handler) },
    location: { protocol: 'file:', hostname: '', pathname: '' }, URL, Blob, TextDecoder, AbortController,
    setTimeout: () => 1, clearTimeout() {}, console });
  vm.runInContext(source + `
    globalThis.api = {
      open(value) { annotation = clone(value); media = { name: value.clip, size: value.media_size, duration: value.duration, sha256: value.media_sha256, url: 'blob:test' }; videoReady = true; pendingDraft = false; pendingDraftKey = null; lastExported = fingerprint(); loadEditor(); findDraft(); },
      replace(value) { annotation = clone(value); selectedId = null; loadEditor(); modified('test edit'); },
      select(id) { selectSegment(id); },
      edit(id, value) { $(id).value = value; contentChanged(); },
      write: writeDraft, restore: restoreDraft,
      state() { return JSON.parse(JSON.stringify({ document: annotation, form: formValues(), dirty: formDirty, pending: pendingDraft, selectedId })); },
      markExported() { lastExported = fingerprint(); },
      refresh() { if (typeof refreshDraftChoices === 'function') return refreshDraftChoices(); return 0; }
    };`, context, { filename: editorPath });
  return { api: context.api, get, events, choose(key) { get('draftChoice').value = key; get('draftChoice').handlers.change?.(); },
    close() { events.get('pagehide')({ persisted: true }); }, ignore() { get('ignoreDraft').handlers.click(); } };
}
function documents(store) { return [...store.data.values()].map(raw => JSON.parse(raw).document); }
function rawEnvelope(document, format = 2) {
  const s = document.segments[0];
  const editor = { startInput: String(s?.start ?? 0), endInput: String(s?.end ?? 1), labelInput: s?.label ?? '', textInput: s?.text ?? '', trackInput: s?.track ?? '主轨', tagsInput: JSON.stringify(s?.tags ?? []), reviewInput: s?.review_status === 'reviewed' };
  return JSON.stringify({ draft_format: format, updated_at: '2026-10-30T12:00:00.000Z', document, selected_id: s?.id ?? null,
    editor, editor_baseline: editor, review_acknowledged_for: null });
}

test('untouched second page cannot erase a saved draft on pagehide', () => {
  const store = storage(), first = page(store), idle = page(store);
  first.api.open(doc()); idle.api.open(doc());
  first.api.replace(doc('A finished annotation')); first.api.write();
  const saved = store.getItem(primary);
  idle.close();
  assert.equal(store.getItem(primary), saved);
  assert.equal(JSON.parse(saved).document.segments[0].label, 'A finished annotation');
});

test('simultaneous editors keep both versions and offer both for recovery', () => {
  const store = storage(), first = page(store), second = page(store);
  first.api.open(doc()); second.api.open(doc());
  first.api.replace(doc('First version')); first.api.write();
  second.api.replace(doc('Second version')); second.api.write();
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'First version'));
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Second version'));
  const recover = page(store); recover.api.open(doc());
  assert.equal(recover.api.refresh(), 2, 'Compatibility mirror is deduplicated; distinct page drafts stay selectable');
  for (const choice of recover.get('draftChoice').children) {
    recover.choose(choice.value); recover.api.restore();
    assert.ok(['First version', 'Second version'].includes(recover.api.state().document.segments[0].label));
  }
});

test('interleaved read/write of the shared compatibility key keeps both page copies', () => {
  const store = storage(), first = page(store), second = page(store);
  first.api.open(doc()); second.api.open(doc());
  first.api.replace(doc('Interleave A')); second.api.replace(doc('Interleave B'));
  let injected = false;
  store.beforeRead = key => { if (key === primary && !injected) { injected = true; second.api.write(); } };
  first.api.write(); store.beforeRead = null;
  assert.ok(injected);
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Interleave A'));
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Interleave B'));
});

test('uncommitted new form survives a conflicting page save and explicit recovery', () => {
  const store = storage(), first = page(store), second = page(store);
  first.api.open(doc()); second.api.open(doc());
  first.api.replace(doc('Saved elsewhere')); first.api.write();
  second.api.edit('labelInput', '未保存片段😀'); second.api.edit('textInput', '还没有点击保存的详细描述'); second.close();
  const key = [...store.data].find(([, raw]) => JSON.parse(raw).editor.labelInput === '未保存片段😀')[0];
  const recover = page(store); recover.api.open(doc()); recover.choose(key); recover.api.restore();
  assert.equal(recover.api.state().form.textInput, '还没有点击保存的详细描述');
  assert.equal(recover.api.state().document.segments.length, 0);
  assert.equal(recover.api.state().dirty, true);
});

test('pending format-2 draft stays untouched until explicit decision', () => {
  const store = storage(), saved = rawEnvelope(doc('Existing')); store.setItem(primary, saved);
  const edit = page(store); edit.api.open(doc()); edit.api.edit('labelInput', 'pending form'); edit.close();
  assert.equal(store.data.size, 1); assert.equal(store.getItem(primary), saved);
  edit.ignore();
  assert.ok([...store.data.values()].includes(saved), 'Continuing must preserve the old raw envelope');
  assert.ok([...store.data.values()].some(raw => JSON.parse(raw).editor.labelInput === 'pending form'));
});

test('restoring and editing an old shared-only draft archives its original bytes', () => {
  const store = storage(), saved = rawEnvelope(doc('Older shared draft')); store.setItem(primary, saved);
  const edit = page(store); edit.api.open(doc()); edit.api.restore(); edit.api.edit('textInput', '新描述'); edit.api.write();
  assert.ok([...store.data.values()].includes(saved));
  assert.equal(JSON.parse(store.getItem(primary)).editor.textInput, '新描述');
  const fresh = page(store); fresh.api.open(doc()); assert.ok(fresh.api.refresh() >= 2);
});

test('ignoring a pending draft without changes and closing remains read only', () => {
  const store = storage(), saved = rawEnvelope(doc('Keep me')); store.setItem(primary, saved);
  const idle = page(store); idle.api.open(doc()); idle.ignore(); idle.close();
  assert.equal(store.data.size, 1); assert.equal(store.getItem(primary), saved);
});

test('legacy v1 recovery preserves source and invalidates old review', () => {
  const store = storage(), value = doc('Legacy'); value.schema_version = 1; delete value.media_sha256;
  const saved = rawEnvelope(value, 1); store.setItem(legacyKey, saved);
  const edit = page(store); edit.api.open(doc()); edit.api.restore(); edit.api.write();
  assert.equal(edit.api.state().document.schema_version, 2);
  assert.equal(edit.api.state().document.segments[0].review_status, 'unreviewed');
  assert.equal(edit.api.state().form.reviewInput, false);
  assert.equal(store.getItem(legacyKey), saved);
});

test('explicit legacy selection preserves coexisting modern page drafts and both remain recoverable', () => {
  const store = storage(), modernKey = primary + ':session:coexisting-modern-page';
  const modern = rawEnvelope(doc('Modern page copy'));
  const legacy = doc('Legacy selected explicitly'); legacy.schema_version = 1; delete legacy.media_sha256;
  const older = rawEnvelope(legacy, 1);
  store.setItem(modernKey, modern); store.setItem(legacyKey, older);
  const edit = page(store); edit.api.open(doc());
  assert.equal(edit.api.refresh(), 2);
  assert.deepEqual(edit.get('draftChoice').children.map(choice => choice.value).sort(), [modernKey, legacyKey].sort());
  edit.choose(legacyKey); edit.api.restore();
  assert.equal(edit.api.state().document.segments[0].label, 'Legacy selected explicitly');
  assert.equal(edit.api.state().document.schema_version, 2);
  assert.equal(edit.api.state().document.segments[0].review_status, 'unreviewed');
  assert.equal(store.getItem(legacyKey), older);
  assert.equal(store.getItem(modernKey), modern);
  const other = page(store); other.api.open(doc()); other.choose(modernKey); other.api.restore();
  assert.equal(other.api.state().document.segments[0].label, 'Modern page copy');
  assert.equal(store.getItem(legacyKey), older);
  assert.equal(store.getItem(modernKey), modern);
});

test('malformed content-bound draft is not silently replaced by a legacy draft', () => {
  const store = storage(), invalid = '{"draft_format":999}'; store.setItem(primary, invalid);
  const value = doc('Legacy fallback'); value.schema_version = 1; delete value.media_sha256;
  store.setItem(legacyKey, rawEnvelope(value, 1));
  const edit = page(store); edit.api.open(doc()); edit.api.restore();
  assert.match(edit.get('status').textContent, /恢复失败/);
  assert.equal(edit.api.state().document.segments.length, 0); assert.equal(store.getItem(primary), invalid);
});

test('quota failure preserves the primary draft and never reports successful saving', () => {
  const store = storage(), first = page(store), second = page(store);
  first.api.open(doc()); second.api.open(doc()); first.api.replace(doc('Protected')); first.api.write();
  const old = store.getItem(primary);
  store.beforeWrite = () => { throw new Error('QuotaExceededError'); };
  second.api.replace(doc('Quota affected')); second.api.write();
  assert.equal(store.getItem(primary), old); assert.match(second.get('draftStatus').textContent, /写入失败/);
  assert.equal(second.api.state().document.segments[0].label, 'Quota affected');
  store.beforeWrite = null; second.api.write();
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Quota affected'));
});

test('writer ID collision does not overwrite the existing page branch', () => {
  const store = storage(), sameId = { randomUUID: () => 'same-id-for-regression' };
  const first = page(store, { crypto: sameId }), second = page(store, { crypto: sameId });
  first.api.open(doc()); second.api.open(doc()); first.api.replace(doc('Collision A')); first.api.write();
  second.api.replace(doc('Collision B')); second.api.write();
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Collision A'));
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Collision B'));
});

test('archive quota failure keeps the old shared draft and retries after capacity returns', () => {
  const store = storage(), saved = rawEnvelope(doc('Archive source')); store.setItem(primary, saved);
  const edit = page(store); edit.api.open(doc()); edit.ignore(); edit.api.replace(doc('Replacement'));
  store.beforeWrite = key => { if (key.includes(':session:preserved-')) throw new Error('QuotaExceededError'); };
  edit.api.write();
  assert.equal(store.getItem(primary), saved); assert.match(edit.get('draftStatus').textContent, /写入失败/);
  assert.ok(documents(store).some(value => value.segments[0]?.label === 'Replacement'), 'Already-written page copy remains available');
  store.beforeWrite = null; edit.api.write();
  assert.ok([...store.data.values()].includes(saved));
  assert.equal(JSON.parse(store.getItem(primary)).document.segments[0].label, 'Replacement');
  assert.match(edit.get('draftStatus').textContent, /已更新/);
});

test('editing after JSON export still updates draft; clean selection does not', () => {
  const store = storage(), edit = page(store); edit.api.open(doc());
  edit.api.replace(doc('Exported')); edit.api.markExported(); edit.api.write();
  const saved = store.getItem(primary);
  edit.api.select('seg-1'); edit.close(); assert.equal(store.getItem(primary), saved);
  edit.api.edit('textInput', 'after export form'); edit.close();
  assert.equal(JSON.parse(store.getItem(primary)).editor.textInput, 'after export form');
});

test('identical dirty forms on different segment IDs retain the new edit target', () => {
  const store = storage(), value = doc('Same content');
  value.segments.push({ ...structuredClone(value.segments[0]), id: 'seg-2' });
  const edit = page(store); edit.api.open(value); edit.api.select('seg-1');
  edit.api.edit('textInput', 'Same unsaved replacement'); edit.api.write();
  assert.equal(JSON.parse(store.getItem(primary)).selected_id, 'seg-1');
  // Confirm discarding A, then edit B before the debounced clean selection is
  // written. The field values match A, but the intended target does not.
  edit.api.select('seg-2'); edit.api.edit('textInput', 'Same unsaved replacement'); edit.close();
  assert.equal(JSON.parse(store.getItem(primary)).selected_id, 'seg-2');
  const recover = page(store); recover.api.open(value); recover.api.restore();
  assert.equal(recover.api.state().selectedId, 'seg-2');
  assert.equal(recover.api.state().form.textInput, 'Same unsaved replacement');
  assert.equal(recover.api.state().dirty, true);
});

test('UUID fallback works without randomUUID and cryptographic globals', () => {
  for (const crypto of [{ getRandomValues(array) { array.fill(123); return array; } }, {}]) {
    const store = storage(), edit = page(store, { crypto }); edit.api.open(doc()); edit.api.replace(doc('Fallback')); edit.api.write();
    assert.equal(JSON.parse(store.getItem(primary)).document.segments[0].label, 'Fallback');
  }
});
