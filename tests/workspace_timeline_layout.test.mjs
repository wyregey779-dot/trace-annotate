// Logic and DOM regressions only; the real Chrome acceptance is separate.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(process.env.TRACEANNOTATE_TIMELINE_SOURCE ||
  new URL('../trace_annotate/workspace/ui.py', import.meta.url), 'utf8');
function implementation(name, next) {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf(next, start + 1);
  assert.ok(start >= 0 && end > start, `Cannot extract ${name}`);
  return source.slice(start, end);
}
const seconds = value => Number(value.n) / Number(value.d);
const time = value => ({ seconds: { n: String(Math.round(value * 1000000)), d: '1000000' } });
const segment = (id, start, end, track = 'shots') => ({ id, type: 'segment', label: id,
  start: time(start), end: time(end), text: 'visible scene', track, review_status: 'unreviewed' });
const duration = 147.666667;
function assignment() {
  return [
    ...Array.from({ length: 78 }, (_, i) => segment(`shot-${i}`, duration * i / 78, duration * (i + 1) / 78)),
    ...Array.from({ length: 11 }, (_, i) => segment(`chapter-${i}`, duration * i / 11, duration * (i + 1) / 11, 'chapters')),
    ...Array.from({ length: 13 }, (_, i) => segment(`material-${i}`, i * 10 + 1, i * 10 + 1.02, 'materials')),
  ];
}
function harness(annotations, width) {
  let document;
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag.toUpperCase(); this.className = ''; this.children = []; this.dataset = {};
      this.attrs = {}; this.handlers = {}; this.clientWidth = width;
      this.style = { setProperty(name, value) { this[name] = value; } };
      this.classList = { contains: name => this.className.split(' ').includes(name) };
    }
    append(...items) {
      for (const item of items) {
        if (item.parentElement) item.parentElement.children = item.parentElement.children.filter(child => child !== item);
        item.parentElement = this; this.children.push(item);
      }
    }
    replaceChildren(...items) { for (const child of this.children) child.parentElement = null; this.children = []; this.append(...items); }
    setAttribute(name, value) { this.attrs[name] = value; }
    addEventListener(name, handler) { this.handlers[name] = handler; }
    focus() { document.activeElement = this; }
  }
  const body = new Element('tbody');
  const nodes = Object.fromEntries(['segments', 'timeline', 'timeline-cursor', 'segment-count', 'revision-label', 'no-segments']
    .map(id => [id, new Element()]));
  nodes.segments.querySelector = name => name === 'tbody' ? body : null;
  document = { activeElement: null, createElement: tag => new Element(tag) };
  const selected = [], state = { task: { annotations, revision: 1 }, media: { duration: time(duration).seconds }, selected: null };
  const context = { document, state, $: id => nodes[id] || null, seconds,
    showTime: seconds, compareTime: (a, b) => seconds(a) - seconds(b),
    selectAnnotation: id => selected.push(id), updateUndoRedoButtons() {} };
  vm.createContext(context);
  if (source.includes('function layoutTimelineTargets(')) {
    vm.runInContext(implementation('layoutTimelineTargets', 'let timelineResizeObserver='), context);
  }
  vm.runInContext(implementation('renderSegments', 'async function selectAnnotation('), context);
  const walk = element => [element, ...element.children.flatMap(walk)];
  function marks() { return walk(nodes.timeline).filter(el => el.classList.contains('timeline-segment')); }
  const px = value => parseFloat(value) * (String(value).endsWith('%') ? nodes.timeline.clientWidth / 100 : 1);
  function boxes() {
    return marks().map(el => ({ element: el, id: el.dataset.annotationId || el.attrs['aria-label'].split('，')[0],
      left: px(el.style['--left']), width: px(el.style['--width']), top: px(el.style['--top'] || '3px'), height: 24 }));
  }
  const render = () => context.renderSegments();
  render();
  return { document, nodes, context, state, selected, marks, boxes, render };
}
function noOcclusion(boxes, width) {
  for (const box of boxes) {
    assert.ok(box.left >= -1e-7 && box.left + box.width <= width + 1e-7, `${box.id} overflows the stage`);
    assert.ok(box.width >= Math.min(24, width) - 1e-7, `${box.id} has an unreachable narrow target`);
    for (const other of boxes) {
      if (box === other || box.top !== other.top) continue;
      assert.ok(box.left + box.width <= other.left + 1e-7 || other.left + other.width <= box.left + 1e-7,
        `${box.id} is covered by ${other.id}`);
    }
  }
}

test('all 102 mixed-track intervals retain separate reachable targets at laptop and narrow widths', () => {
  for (const width of [462, 680, 280, 80]) {
    const input = assignment(), before = JSON.stringify(input), h = harness(input, width);
    assert.equal(h.marks().length, 102);
    assert.equal(new Set(h.boxes().map(box => box.id)).size, 102);
    noOcclusion(h.boxes(), width);
    assert.equal(JSON.stringify(input), before, 'Layout changed annotation data');
  }
});
test('the center of every advertised target activates its own annotation, including covered chapter starts', () => {
  const h = harness(assignment(), 462), boxes = h.boxes();
  for (const wanted of boxes) {
    const x = wanted.left + wanted.width / 2, y = wanted.top + wanted.height / 2;
    const painted = boxes.filter(box => x >= box.left && x < box.left + box.width && y >= box.top && y < box.top + box.height).at(-1);
    assert.ok(painted, `No target at ${wanted.id}`);
    painted.element.handlers.click();
    assert.equal(h.selected.at(-1), wanted.id, `${wanted.id} activated another interval`);
  }
});
test('true interval color geometry remains exact when short hit targets are expanded and end-clamped', () => {
  const input = [segment('first-frame', 0, .000001), segment('last-frame', duration - .000001, duration),
    segment('long', 20, 100)], h = harness(input, 462);
  noOcclusion(h.boxes(), 462);
  for (const box of h.boxes()) {
    const annotation = input.find(item => item.id === box.id);
    const inset = box.element.children.find(el => el.classList.contains('timeline-interval'));
    assert.ok(inset, 'True time interval must stay visible independently of the hit rectangle');
    const start = seconds(annotation.start.seconds) / duration * 462;
    const end = seconds(annotation.end.seconds) / duration * 462;
    assert.ok(Math.abs(box.left + parseFloat(inset.style['--interval-left']) - start) < 1e-6);
    assert.ok(Math.abs(parseFloat(inset.style['--interval-width']) - (end - start)) < 1e-6);
  }
});
test('identical ranges and adjacent subpixel ranges cannot cover one another', () => {
  const input = [segment('cover-a', 0, duration), segment('cover-b', 0, duration),
    ...Array.from({ length: 30 }, (_, i) => segment(`tiny-${i}`, i * .00001, (i + 1) * .00001))];
  const h = harness(input, 250);
  assert.equal(h.marks().length, 32);
  noOcclusion(h.boxes(), 250);
});
test('resize keeps all controls and a focused selected control retains its identity after rebuilding', () => {
  const h = harness(assignment(), 680), focused = h.marks().find(el => el.dataset.annotationId === 'material-12');
  assert.ok(focused, 'Overview needs stable annotation identities');
  focused.focus(); h.state.selected = 'material-12'; h.nodes.timeline.clientWidth = 280; h.render();
  assert.equal(h.marks().length, 102);
  noOcclusion(h.boxes(), 280);
  assert.equal(h.document.activeElement.dataset.annotationId, 'material-12');
  assert.notEqual(h.document.activeElement, focused);
  assert.ok(h.document.activeElement.classList.contains('selected'));
});
test('every rendered interval remains a named native button without a negative tab order', () => {
  const h = harness(assignment(), 462);
  for (const mark of h.marks()) {
    assert.equal(mark.tagName, 'BUTTON'); assert.equal(mark.type, 'button');
    assert.ok(mark.attrs['aria-label']); assert.notEqual(mark.attrs.tabindex, '-1');
    assert.notEqual(mark.disabled, true); assert.equal(typeof mark.handlers.click, 'function');
  }
});
