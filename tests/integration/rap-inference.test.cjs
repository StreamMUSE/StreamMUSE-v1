const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const staticRoot = path.resolve(__dirname, '../../src/streammuse/presentation/rap_demo/static');
const source = path.join(staticRoot, 'js/rap-inference.js');
const sandbox = { window: {} };
vm.createContext(sandbox);
if (fs.existsSync(source)) vm.runInContext(fs.readFileSync(source, 'utf8'), sandbox);
const api = sandbox.window.StreamMuseInference || {};
const copy = (value) => JSON.parse(JSON.stringify(value));

test('unavailable and nonfinite metrics never become zero or successful timings', () => {
  assert.equal(typeof api.formatMetric, 'function');
  for (const value of [null, undefined, '', false, NaN, Infinity, -Infinity, '23']) {
    assert.equal(api.formatMetric(value), '--');
  }
  assert.equal(api.formatMetric(0), '0 ms');
  assert.equal(api.formatMetric(137), '137 ms');
  assert.equal(api.formatMetric(1370), '1.37 s');
  assert.equal(api.formatMetric({ value: null, reason: 'No streaming' }), '--');
});

test('WAIT displays elapsed seconds only, ignoring configured musical durations', () => {
  assert.equal(typeof api.phaseLabel, 'function');
  assert.equal(api.phaseLabel({ phase: 'WAIT', current: { elapsed_ms: 1370, remaining_ticks: 16 }, config: { ticks_per_beat: 4 } }), '1.37 s elapsed');
  assert.equal(api.phaseLabel({ phase: 'WAIT', current: {}, config: {} }), '-- elapsed');
  assert.equal(api.phaseLabel({ phase: 'USER', current: { remaining_ticks: 6 }, config: { ticks_per_beat: 3 } }), '2 beats remaining');
});

test('span layout uses exact arbitrary tick coverage and rejects invalid geometry', () => {
  assert.equal(typeof api.layoutSpans, 'function');
  const spans = [{ span_id: 'a', start_tick: 3, duration_ticks: 2 }, { span_id: 'b', start_tick: 7, duration_ticks: 5 }, { start_tick: 1, duration_ticks: 0 }, { start_tick: NaN, duration_ticks: 2 }];
  const result = api.layoutSpans(spans, 0, 12);
  assert.equal(result.length, 2);
  assert.equal(result[0].left, 25);
  assert.equal(result[0].width, 100 / 6);
  assert.equal(result[1].left, 100 * 7 / 12);
  assert.equal(result[1].width, 100 * 5 / 12);
});

test('config converts bars using reported meter and refuses empty/nonfinite inputs', () => {
  assert.equal(typeof api.readConfig, 'function');
  assert.deepEqual(copy(api.readConfig({ mode: 'cycle', bpm: '60', user: '1', model: '2', unit: 'bars' }, { beats_per_bar: 3 })), { mode: 'cycle', bpm: 60, user_beats: 3, model_beats: 6 });
  for (const value of ['', ' ', 'Infinity', 'NaN', '-1', '0']) {
    assert.throws(() => api.readConfig({ mode: 'cycle', bpm: value, user: '1', model: '2', unit: 'beats' }, {}));
  }
  assert.throws(() => api.readConfig({ mode: 'cycle', bpm: '60', user: '1', model: '2', unit: 'bars' }, {}));
});

class Node {
  constructor(id = '') {
    this.id = id; this.value = ''; this.textContent = ''; this.children = []; this.listeners = {};
    this.dataset = {}; this.style = { setProperty(name, value) { this[name] = value; } };
    this.disabled = false; this.hidden = false; this.tagName = 'DIV';
  }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  removeEventListener(name, callback) { this.listeners[name] = (this.listeners[name] || []).filter((fn) => fn !== callback); }
  async fire(name, event = {}) { for (const callback of this.listeners[name] || []) await callback({ target: this, preventDefault() {}, ...event }); }
  setAttribute(name, value) { this[name] = String(value); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  focus() {}
  closest(selector) { return /input|textarea|select|button/i.test(this.tagName) && selector.includes(this.tagName.toLowerCase()) ? this : null; }
}
function snapshot(extra = {}) {
  return { supported: true, session_id: 'shared', epoch: 1, active_mode: 'cycle', state: 'stopped', phase: 'STOPPED', owner_id: null, cycle_id: 1,
    config: { mode: 'cycle', bpm: 60, beats_per_bar: 4, ticks_per_beat: 4, user_beats: 4, model_beats: 8 },
    capabilities: { renderer: 'fixture', generation_unit: 'bar', configurable: ['mode', 'bpm', 'user_beats', 'model_beats'], startup_buffer_policy: 'minimum-prefix' },
    current: {}, spans: [], requests: [], metrics: {}, error: null, ...extra };
}
function uiFixture(initial = snapshot(), overrides = {}) {
  assert.equal(typeof api.mount, 'function');
  const html = fs.readFileSync(path.join(staticRoot, 'inference.html'), 'utf8');
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map((m) => [m[1], new Node(m[1])]));
  const document = new Node(); document.hidden = false;
  document.getElementById = (id) => { assert.ok(nodes.has(id), `HTML must define ${id}`); return nodes.get(id); };
  document.createElement = () => new Node();
  document.activeElement = null;
  const calls = [], timers = new Map(); let timerId = 0; let time = 1000;
  const storage = new Map(); let uuid = 0;
  const env = new Node();
  Object.assign(env, {
    performance: { now: () => time }, crypto: { randomUUID: () => `uuid-${++uuid}` },
    sessionStorage: { getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    setTimeout: (fn, ms) => { timers.set(++timerId, { fn, ms }); return timerId; }, clearTimeout: (id) => timers.delete(id),
    fetch: async (url, options = {}) => {
      calls.push({ url, ...options, body: options.body ? JSON.parse(options.body) : undefined });
      const body = url === '/api/control/plan' ? { templates: [] } : initial;
      return { ok: true, status: 200, json: async () => body };
    }, ...overrides,
  });
  const controller = api.mount(document, env);
  return { nodes, document, env, controller, calls, timers, advance(ms) { time += ms; }, settle: () => new Promise(setImmediate), close: () => env.fire('pagehide') };
}

test('mount only attaches to the shared session; no Start, Reset, or mode mutation', async () => {
  const ui = uiFixture(); await ui.settle();
  assert.ok(ui.calls.some((call) => call.url === '/api/inference/state'));
  assert.equal(ui.calls.filter((call) => call.method === 'POST').length, 0);
  assert.equal(ui.nodes.get('session-id').textContent, 'shared');
  assert.equal(ui.nodes.get('metric-ttft').textContent, '--');
  await ui.close();
});

test('unsupported runtime disables experimental commands but retains global Stop/Reset', async () => {
  const ui = uiFixture(snapshot({ supported: false, state: 'unavailable', active_mode: 'legacy' })); await ui.settle();
  assert.equal(ui.nodes.get('start').disabled, true);
  assert.equal(ui.nodes.get('save-config').disabled, true);
  assert.equal(ui.nodes.get('stop').disabled, false);
  assert.equal(ui.nodes.get('reset').disabled, false);
  await ui.close();
});

test('taps require USER, owner match, nonrepeated keyboard events and noneditable focus', async () => {
  const state = snapshot({ state: 'running', phase: 'USER', owner_id: 'uuid-1', current: { elapsed_ms: 200 } });
  const ui = uiFixture(state); await ui.settle();
  const capture = ui.nodes.get('capture-surface');
  await capture.fire('keydown', { code: 'Space', key: ' ', repeat: true });
  await ui.document.fire('keydown', { code: 'Space', key: ' ' });
  assert.equal(ui.calls.filter((c) => c.url.endsWith('/tap')).length, 0);
  ui.advance(50);
  await capture.fire('keydown', { code: 'Space', key: ' ', repeat: false }); await ui.settle();
  const taps = ui.calls.filter((c) => c.url.endsWith('/tap'));
  assert.equal(taps.length, 1);
  assert.equal(taps[0].body.owner_id, 'uuid-1');
  assert.equal(taps[0].body.cycle_id, 1);
  assert.equal(taps[0].body.epoch, 1);
  assert.equal(taps[0].body.client_time_ms, 1050);
  assert.ok(taps[0].body.event_id);
  await ui.close();
});

for (const extra of [{ phase: 'WAIT' }, { phase: 'MODEL' }, { owner_id: 'another-tab' }]) {
  test(`capture does not submit taps for ${JSON.stringify(extra)}`, async () => {
    const ui = uiFixture(snapshot({ state: 'running', phase: 'USER', owner_id: 'uuid-1', ...extra })); await ui.settle();
    await ui.nodes.get('capture-surface').fire('keydown', { code: 'Space', key: ' ' });
    assert.equal(ui.calls.filter((c) => c.url.endsWith('/tap')).length, 0);
    await ui.close();
  });
}

test('pending poll cannot restore old lyrics after a reset', async () => {
  let resolvePoll, count = 0;
  const ui = uiFixture(snapshot(), { fetch: async (url) => {
    if (url === '/api/control/plan') return { ok: true, json: async () => ({ templates: [] }) };
    if (url.endsWith('/reset')) return { ok: true, json: async () => snapshot({ epoch: 2 }) };
    count++;
    if (count === 2) return new Promise((resolve) => { resolvePoll = resolve; });
    return { ok: true, json: async () => snapshot({ epoch: count > 2 ? 2 : 1 }) };
  } });
  await ui.settle();
  const pending = ui.controller.refresh(); await ui.settle();
  await ui.nodes.get('reset').fire('click');
  resolvePoll({ ok: true, json: async () => snapshot({ current: { text: 'Expired lyric' } }) });
  await pending; await ui.settle();
  assert.notEqual(ui.nodes.get('current-text').textContent, 'Expired lyric');
  await ui.close();
});

test('unknown TTFT remains unavailable even when total request latency is reported', async () => {
  const ui = uiFixture(snapshot({ requests: [{ request_id: 'r1', state: 'ready', latency_ms: 1000 }], metrics: { response_start_latency_ms: 1370 } })); await ui.settle();
  assert.equal(ui.nodes.get('metric-ttft').textContent, '--');
  assert.equal(ui.nodes.get('metric-response').textContent, '1.37 s');
  await ui.close();
});

test('duplicate tabs inheriting a token cannot both capture; refresh reclaims a released token', () => {
  assert.equal(typeof api.createOwner, 'function');
  const channels = new Set(), tasks = [], storage = new Map(); let uuid = 0;
  class BroadcastChannel {
    constructor() { channels.add(this); }
    postMessage(data) { for (const channel of channels) if (channel !== this) tasks.push(() => channel.onmessage?.({ data })); }
    close() { channels.delete(this); }
  }
  const timers = [];
  const env = { BroadcastChannel, crypto: { randomUUID: () => `token-${++uuid}` },
    sessionStorage: { getItem: (k) => storage.get(k), setItem: (k, v) => storage.set(k, v) },
    setTimeout: (fn) => timers.push(fn), clearTimeout() {} };
  const first = api.createOwner(env, () => {}); timers.shift()();
  const second = api.createOwner(env, () => {});
  while (tasks.length) tasks.shift()();
  timers.shift()();
  assert.notEqual(first.id(), second.id());
  assert.equal(second.ready(), true);
  const original = first.id(); first.close(); second.close(); storage.set('streammuse-inference-owner', original);
  const refreshed = api.createOwner(env, () => {}); timers.shift()();
  assert.equal(refreshed.id(), original); refreshed.close();
});

test('steering displays requested and performed versions without claiming semantic compliance', async () => {
  const state = snapshot({ active_mode: 'steering', state: 'running', phase: 'MODEL', current: { requested_version: 4, playing_version: 2, text: '<img src=x onerror=alert(1)>', status: 'stale context' } });
  state.config.mode = 'steering';
  const ui = uiFixture(state); await ui.settle();
  assert.equal(ui.nodes.get('requested-version').textContent, 'v4');
  assert.equal(ui.nodes.get('playing-version').textContent, 'v2');
  assert.equal(ui.nodes.get('current-text').textContent, '<img src=x onerror=alert(1)>');
  assert.equal(ui.nodes.get('current-text').children.length, 0);
  assert.equal(ui.nodes.get('mode-cycle').disabled, true);
  assert.equal(ui.nodes.get('bpm').disabled, true);
  assert.equal(ui.nodes.get('topic').disabled, false);
  await ui.close();
});

test('audio popover gates shared settings and sends only local gain or session device patches', async () => {
  const calls = [], state = snapshot(); state.capabilities.audio_controls_supported = true;
  const settings = { supported: true, state: 'stopped', capabilities: { drum_gain: true, vocal_gain: true, output_device: true }, values: { drum_gain: 0.7, vocal_gain: 0.8, output_device: 'device-1' }, devices: [{ id: 'device-1', name: 'Speakers', available: true }] };
  const ui = uiFixture(state, { fetch: async (url, options = {}) => {
    calls.push({ url, body: options.body && JSON.parse(options.body) });
    return { ok: true, json: async () => url.includes('/settings') ? settings : url.includes('/plan') ? { templates: [] } : state };
  } });
  await ui.settle(); ui.nodes.get('audio-settings').open = true; await ui.nodes.get('audio-settings').fire('toggle');
  assert.equal(ui.nodes.get('drum-gain').disabled, false);
  assert.equal(ui.nodes.get('output-device').value, 'device-1');
  ui.nodes.get('drum-gain').value = '0.5'; await ui.nodes.get('drum-gain').fire('change');
  assert.deepEqual(calls.find((c) => c.url === '/api/control/settings/local').body, { drum_gain: 0.5 });
  ui.nodes.get('output-device').value = ''; await ui.nodes.get('output-device').fire('change');
  assert.deepEqual(calls.find((c) => c.url === '/api/control/settings/session').body, { output_device: null });
  await ui.close();
});

test('audio controls remain disabled when renderer compatibility is unreported', async () => {
  const ui = uiFixture(); await ui.settle(); ui.nodes.get('audio-settings').open = true; await ui.nodes.get('audio-settings').fire('toggle');
  assert.equal(ui.nodes.get('drum-gain').disabled, true);
  assert.equal(ui.nodes.get('output-device').disabled, true);
  await ui.close();
});

test('reconnect clears a connection failure and permits a fresh explicit Start', async () => {
  let online = false;
  const ui = uiFixture(snapshot(), { fetch: async (url) => {
    if (!online) throw new Error('Offline');
    return { ok: true, json: async () => url.includes('/plan') ? { templates: [] } : snapshot() };
  } });
  await ui.settle(); assert.equal(ui.nodes.get('start').disabled, true);
  online = true; await ui.controller.refresh();
  assert.equal(ui.nodes.get('error').hidden, true);
  assert.equal(ui.nodes.get('start').disabled, false);
  await ui.close();
});

test('steering drafts do not become config writes or block transport after acceptance', async () => {
  const state = snapshot({ active_mode: 'steering' }); state.config.mode = 'steering';
  const ui = uiFixture(state); await ui.settle();
  ui.nodes.get('topic').value = 'Night trains'; await ui.nodes.get('topic').fire('input');
  assert.equal(ui.nodes.get('save-config').disabled, true);
  await ui.nodes.get('steering-form').fire('submit');
  assert.deepEqual(ui.calls.find((call) => call.url.endsWith('/steer')).body, { topic: 'Night trains' });
  assert.equal(ui.nodes.get('start').disabled, false);
  await ui.close();
});

test('page disposal cancels polling but never stops or resets shared audio', async () => {
  const ui = uiFixture(); await ui.settle(); await ui.close();
  assert.equal(ui.timers.size, 0);
  assert.equal(ui.calls.filter((call) => call.method === 'POST').length, 0);
});

test('an old-cycle tap failure cannot contaminate a remotely reset session', async () => {
  let state = snapshot({ state: 'running', phase: 'USER', owner_id: 'uuid-1', current: { elapsed_ms: 200 } }), rejectTap;
  const ui = uiFixture(state, { fetch: async (url) => {
    if (url.endsWith('/tap')) return new Promise((_, reject) => { rejectTap = reject; });
    return { ok: true, json: async () => url.includes('/plan') ? { templates: [] } : state };
  } });
  await ui.settle();
  const pending = ui.nodes.get('capture-surface').fire('keydown', { code: 'Space' });
  await ui.settle(); state = snapshot({ epoch: 2 }); await ui.controller.refresh();
  rejectTap(new Error('Old capture rejected')); await pending;
  assert.equal(ui.nodes.get('error').hidden, true);
  await ui.close();
});

test('audio settings rejection remains visible after request cleanup', async () => {
  const state = snapshot(); state.capabilities.audio_controls_supported = true;
  const settings = { supported: true, capabilities: { drum_gain: true }, values: { drum_gain: 0.7 }, devices: [] };
  const ui = uiFixture(state, { fetch: async (url, options = {}) => {
    if (url.includes('/settings') && options.method === 'POST') return { ok: false, status: 409, json: async () => ({ detail: 'Audio device disconnected' }) };
    return { ok: true, json: async () => url.includes('/settings') ? settings : url.includes('/plan') ? { templates: [] } : state };
  } });
  await ui.settle(); ui.nodes.get('audio-settings').open = true; await ui.nodes.get('audio-settings').fire('toggle');
  ui.nodes.get('drum-gain').value = '0.5'; await ui.nodes.get('drum-gain').fire('change');
  assert.match(ui.nodes.get('audio-status').textContent, /Audio device disconnected/);
  await ui.close();
});

test('successful cycle Start focuses the capture surface without replaying keyboard events', async () => {
  let state = snapshot(); const ui = uiFixture(state, { fetch: async (url) => {
    if (url.endsWith('/start')) state = snapshot({ state: 'running', phase: 'USER', owner_id: 'uuid-1', current: { elapsed_ms: 0 } });
    return { ok: true, json: async () => url.includes('/plan') ? { templates: [] } : state };
  } });
  let focused = false; ui.nodes.get('capture-surface').focus = () => { focused = true; };
  await ui.settle(); await ui.nodes.get('start').fire('click');
  assert.equal(focused, true);
  await ui.close();
});

test('beat and uneven spans retain geometry and show distinct missing/stale states', async () => {
  const ui = uiFixture(snapshot({ spans: [
    { span_id: 'a', start_tick: 0, duration_ticks: 1, state: 'ready' },
    { span_id: 'b', start_tick: 1, duration_ticks: 2, state: 'missing' },
    { span_id: 'c', start_tick: 3, duration_ticks: 7, state: 'stale_context' },
  ] })); await ui.settle();
  const spans = ui.nodes.get('span-timeline').children;
  assert.deepEqual(spans.map((span) => span.style.width), ['10%', '20%', '70%']);
  assert.deepEqual(spans.map((span) => span.dataset.state), ['ready', 'missing', 'stale_context']);
  await ui.close();
});

test('continuous performance follows future musical positions instead of freezing at the first turn', async () => {
  const state = snapshot({ active_mode: 'steering', state: 'running', phase: 'MODEL', current: {
    tick: 65, syllables: [{ text: 'future', start_tick: 64, duration_ticks: 2 }],
  } }); state.config.mode = 'steering';
  const ui = uiFixture(state); await ui.settle();
  const grid = ui.nodes.get('performance-grid');
  assert.equal(grid.children.filter((item) => item.className === 'syllable').length, 1);
  assert.equal(grid.children.find((item) => item.className === 'playhead').style.left, '3.125%');
  assert.doesNotMatch(ui.nodes.get('phase-time').textContent, /remaining/);
  await ui.close();
});

test('flow tick zero is a genuine onset, and a leading rest remains in place', async () => {
  const ui = uiFixture(snapshot({ phase: 'USER', current: { flow: [0, 3], syllables: [{ text: 'late', start_tick: 3, duration_ticks: 2 }] } })); await ui.settle();
  const grid = ui.nodes.get('performance-grid');
  assert.equal(grid.children.filter((item) => item.className === 'flow-mark').length, 2);
  assert.equal(grid.children.find((item) => item.className === 'syllable').style.left, '18.75%');
  await ui.close();
});

test('polls started during Reset cannot restore active performance before the reset acknowledges', async () => {
  let resolveReset;
  const ui = uiFixture(snapshot(), { fetch: async (url) => {
    if (url.endsWith('/reset')) return new Promise((resolve) => { resolveReset = resolve; });
    return { ok: true, json: async () => url.includes('/plan') ? { templates: [] } : snapshot({ current: { text: 'Old performance' } }) };
  } });
  await ui.settle(); const reset = ui.nodes.get('reset').fire('click'); await ui.settle();
  await ui.controller.refresh();
  assert.equal(ui.nodes.get('current-text').textContent, '--');
  resolveReset({ ok: true, json: async () => snapshot({ epoch: 2 }) }); await reset;
  await ui.close();
});

test('browser locks arbitrate inherited capture tokens before either tab becomes ready', async () => {
  const locks = new Set(), storage = new Map(); let uuid = 0;
  const env = { crypto: { randomUUID: () => `lock-${++uuid}` },
    sessionStorage: { getItem: (k) => storage.get(k), setItem: (k, v) => storage.set(k, v) }, clearTimeout() {},
    navigator: { locks: { async request(name, options, callback) {
      await Promise.resolve();
      if (locks.has(name)) return callback(null);
      locks.add(name);
      try { return await callback({ name }); } finally { locks.delete(name); }
    } } },
  };
  const first = api.createOwner(env, () => {}); const original = first.id();
  assert.equal(first.ready(), false); await new Promise(setImmediate); assert.equal(first.ready(), true);
  const second = api.createOwner(env, () => {}); await new Promise(setImmediate);
  assert.notEqual(first.id(), second.id()); assert.equal(second.ready(), true);
  first.close(); await new Promise(setImmediate); storage.set('streammuse-inference-owner', original);
  const refresh = api.createOwner(env, () => {}); await new Promise(setImmediate);
  assert.equal(refresh.id(), original);
  second.close(); refresh.close(); await new Promise(setImmediate);
  assert.equal(locks.size, 0);
});

test('actual service audio capability false overrides provisional flags and reports its reason', async () => {
  const state = snapshot(); Object.assign(state.capabilities, { audio_settings: true, audio_controls_supported: false, audio_controls_reason: 'Live gain control is unavailable' });
  const ui = uiFixture(state); await ui.settle();
  assert.equal(ui.nodes.get('drum-gain').disabled, true);
  assert.equal(ui.nodes.get('audio-status').textContent, 'Live gain control is unavailable');
  await ui.close();
});

test('service duration denominator and response/error counts retain their actual meanings', async () => {
  const ui = uiFixture(snapshot({ metrics: { eligible_duration_ms: 12500, responses: 3, error_count: 2 } })); await ui.settle();
  const values = Object.fromEntries(ui.nodes.get('metric-details').children.map((item) => item.children.map((child) => child.textContent)));
  assert.equal(values['Eligible scheduled duration'], '12.50 s');
  assert.equal(values['Responses started'], '3');
  assert.equal(values.Errors, '2');
  assert.equal(values.Samples, '--');
  await ui.close();
});

test('actual nested service summaries display measured latency and duration-weighted coverage', async () => {
  const ui = uiFixture(snapshot({ metrics: {
    response_latency_summary: { count: 5, p50_ms: 1370, p95_ms: 2200, max_ms: 2500 },
    coverage: { eligible_ms: 10000, missing_ms: 1000, stale_ms: 2000, fallback_ms: 0, wall_ms: 20000, missing_rate: 0.1, stale_rate: 0.2, fallback_rate: 0, on_time_musical_s_per_wall_s: 0.45, on_time_scope: 'Nonmissing audio only' },
  } })); await ui.settle();
  const values = Object.fromEntries(ui.nodes.get('metric-details').children.map((item) => item.children.map((child) => child.textContent)));
  assert.equal(values['Response median'], '1.37 s');
  assert.equal(values['Response p95'], '2.20 s');
  assert.equal(values.Samples, '5');
  assert.equal(values['Missing coverage'], '10%');
  assert.equal(values['Stale coverage'], '20%');
  assert.equal(values['On-time audio seconds / wall second'], '0.45');
  await ui.close();
});

test('candidate yield is based only on explicit evaluated decisions, never audio packets', async () => {
  const ui = uiFixture(snapshot({ requests: [
    { request_id: 'r1', state: 'ready', valid_candidates: null, evaluated_candidates: null, candidates: [{ text: 'a' }, { text: 'b' }] },
    { request_id: 'r2', state: 'ready', valid_candidates: 2, evaluated_candidates: 5 },
  ], spans: [{ start_tick: 0, duration_ticks: 1 }, { start_tick: 1, duration_ticks: 3 }] })); await ui.settle();
  const rows = ui.nodes.get('requests-body').children;
  assert.equal(rows[0].children[5].textContent, '2 valid / 5 evaluated');
  assert.equal(rows[1].children[5].textContent, '--');
  await ui.close();
});

test('legacy audio ownership clears stale inference performance but retains request history', async () => {
  const ui = uiFixture(snapshot({ state: 'running', phase: 'MODEL', active_mode: 'legacy', legacy_active: true, audio_owner: 'legacy',
    current: { text: 'Stale inference lyric', tick: 3, playing_version: 2, requested_version: 3, scheduled_start_ms: 1000, observed_start_ms: 1000 },
    spans: [{ start_tick: 0, duration_ticks: 4, state: 'playing' }],
    requests: [{ request_id: 'past', state: 'ready' }], metrics: { response_start_latency_ms: 1000 },
  })); await ui.settle();
  assert.equal(ui.nodes.get('current-text').textContent, '--');
  assert.equal(ui.nodes.get('playing-version').textContent, '--');
  assert.equal(ui.nodes.get('tick').textContent, '--');
  assert.equal(ui.nodes.get('observed-start').textContent, '--');
  assert.equal(ui.nodes.get('metric-response').textContent, '--');
  assert.equal(ui.nodes.get('span-timeline').children.filter((item) => item.className === 'prepared-span').length, 0);
  assert.equal(ui.nodes.get('requests-body').children.length, 1);
  assert.equal(ui.nodes.get('start').disabled, true);
  assert.equal(ui.nodes.get('stop').disabled, false);
  assert.equal(ui.nodes.get('phase').textContent, 'LEGACY');
  await ui.close();
});

function statusFixture(response) {
  const statusSource = path.join(staticRoot, 'js/rap-inference-status.js');
  const context = { window: {} }; vm.createContext(context);
  if (fs.existsSync(statusSource)) vm.runInContext(fs.readFileSync(statusSource, 'utf8'), context);
  const mount = context.window.StreamMuseInferenceStatus?.mount;
  assert.equal(typeof mount, 'function');
  const status = new Node('inference-mode-status'), env = new Node(), calls = [], timers = new Set();
  const document = { hidden: false, getElementById: (id) => id === status.id ? status : null };
  Object.assign(env, { fetch: async (url, options) => { calls.push({ url, ...options }); if (response instanceof Error) throw response; return { ok: true, json: async () => response }; },
    setTimeout: (fn) => { timers.add(fn); return fn; }, clearTimeout: (timer) => timers.delete(timer) });
  mount(document, env);
  return { status, env, calls, timers, settle: () => new Promise(setImmediate), close: () => env.fire('pagehide') };
}

test('original-page status names active inference mode and phase without writing shared state', async () => {
  const ui = statusFixture(snapshot({ state: 'running', phase: 'MODEL', active_mode: 'cycle', audio_owner: 'inference' })); await ui.settle();
  assert.equal(ui.status.textContent, 'Active: Cyclic exchange / MODEL');
  assert.ok(ui.calls.every((call) => call.url === '/api/inference/state' && (!call.method || call.method === 'GET')));
  await ui.close(); assert.equal(ui.timers.size, 0);
});

test('original-page status does not label stopped or legacy-owned inference history as live', async () => {
  for (const state of [snapshot(), snapshot({ state: 'running', phase: 'MODEL', active_mode: 'legacy', legacy_active: true, audio_owner: 'legacy' })]) {
    const ui = statusFixture(state); await ui.settle(); assert.equal(ui.status.textContent, 'Inference'); await ui.close();
  }
});

test('original-page status reports an unavailable connection instead of asserting inactive playback', async () => {
  const ui = statusFixture(new Error('Offline')); await ui.settle();
  assert.equal(ui.status.textContent, 'Inference status unavailable'); await ui.close();
});

test('inactive error permits configuration and Start without Reset or history loss', async () => {
  let state = snapshot({ state: 'error', phase: 'STOPPED', error: 'Backend timed out',
    metrics: { response_start_latency_ms: 1370 }, requests: [{ request_id: 'retained', state: 'error' }] });
  const calls = [];
  const ui = uiFixture(state, { fetch: async (url, options = {}) => {
    const body = options.body && JSON.parse(options.body); calls.push({ url, body });
    if (url.endsWith('/config')) state = { ...state, config: { ...state.config, ...body } };
    if (url.endsWith('/start')) state = { ...state, state: 'running', phase: 'USER', epoch: 2, owner_id: body.owner_id, error: null };
    return { ok: true, json: async () => url.includes('/plan') ? { templates: [] } : state };
  } });
  await ui.settle();
  assert.equal(ui.nodes.get('start').disabled, false);
  assert.equal(ui.nodes.get('bpm').disabled, false);
  await ui.nodes.get('stop').fire('click');
  assert.equal(ui.nodes.get('start').disabled, false);
  ui.nodes.get('bpm').value = '80'; await ui.nodes.get('bpm').fire('input');
  await ui.nodes.get('save-config').fire('click');
  assert.equal(calls.find((call) => call.url.endsWith('/config')).body.bpm, 80);
  await ui.nodes.get('start').fire('click');
  assert.ok(calls.some((call) => call.url.endsWith('/start')));
  assert.equal(calls.some((call) => call.url.endsWith('/reset')), false);
  assert.equal(ui.nodes.get('metric-response').textContent, '1.37 s');
  assert.match(ui.nodes.get('requests-body').children[0].children[0].children[0].children[0].textContent, /retained/);
  await ui.close();
});

test('error text while running or stopping does not enable inactive-only changes', async () => {
  for (const state of ['running', 'stopping']) {
    const ui = uiFixture(snapshot({ state, phase: 'MODEL', error: 'Reported warning' })); await ui.settle();
    assert.equal(ui.nodes.get('start').disabled, true); assert.equal(ui.nodes.get('bpm').disabled, true); await ui.close();
  }
});

test('candidate counts retain known valid counts and name requested versus evaluated denominators', async () => {
  const ui = uiFixture(snapshot({ requests: [
    { request_id: 'only-valid', valid_candidates: 20, requested_candidates: null, evaluated_candidates: null },
    { request_id: 'requested', valid_candidates: 20, requested_candidates: 64, evaluated_candidates: null, parsed_candidates: null },
    { request_id: 'evaluated', valid_candidates: 20, requested_candidates: 64, evaluated_candidates: 32 },
  ] })); await ui.settle();
  const labels = ui.nodes.get('requests-body').children.map((row) => row.children[5].textContent);
  assert.deepEqual(labels, ['20 valid / 32 evaluated', '20 valid / 64 requested', '20 valid']);
  await ui.close();
});

test('already-open original monitor refreshes real transport controls when inference changes without websocket events', async () => {
  class LegacyNode extends Node {
    constructor(id) { super(id); this.classList = { add() {}, remove() {}, toggle() {} }; this.options = []; this.parentElement = this; this.scrollTop = 0; this.scrollHeight = 0; this.clientWidth = 0; this.offsetLeft = 0; this.offsetWidth = 0; }
    removeAttribute() {}
    remove() {}
  }
  const nodes = new Map(), timers = new Map(), calls = []; let nextTimer = 0;
  const document = new LegacyNode(); document.hidden = false; document.readyState = 'loading';
  document.getElementById = (id) => { if (!nodes.has(id)) nodes.set(id, new LegacyNode(id)); return nodes.get(id); };
  document.createElement = () => new LegacyNode(); document.querySelectorAll = () => [];
  let inference = snapshot({ audio_owner: 'legacy' });
  let legacy = { stopped: true, session_id: 'shared', last_sequence: 1, coordinator_epoch: 1, remote_chunk: null, recent_events: [], frozen_bars: { 0: { text: 'Retained legacy lyric' } }, audio: { state: 'stopped' } };
  const env = new LegacyNode();
  Object.assign(env, { document, Node: LegacyNode, location: { protocol: 'http:', host: 'localhost' },
    setTimeout: (fn, ms) => { timers.set(++nextTimer, { fn, ms }); return nextTimer; }, clearTimeout: (id) => timers.delete(id),
    WebSocket: class { close() {} },
    fetch: async (url, options = {}) => {
      calls.push({ url, ...options });
      if (url === '/api/control/stop') { legacy = { ...legacy, stopped: true, audio: { state: 'stopped' } }; inference = { ...inference, state: 'stopped', phase: 'STOPPED' }; return { ok: true, json: async () => ({ state: 'stopped' }) }; }
      return { ok: true, json: async () => url === '/api/state' ? legacy : url === '/api/inference/state' ? inference : {} };
    },
  });
  env.window = env; const context = vm.createContext(env);
  for (const file of ['rap-demo-state.js', 'rap-demo.js', 'rap-inference-status.js']) vm.runInContext(fs.readFileSync(path.join(staticRoot, 'js', file), 'utf8'), context);
  const settle = () => new Promise(setImmediate);
  const runTimers = async (ms) => { for (const [id, item] of [...timers]) if (item.ms === ms) { timers.delete(id); await item.fn(); } await settle(); };
  await settle(); await runTimers(25);
  assert.equal(nodes.get('stop-runtime').disabled, true);
  inference = snapshot({ state: 'running', phase: 'MODEL', audio_owner: 'inference' });
  legacy = { ...legacy, stopped: false, audio: { state: 'running' } };
  await runTimers(1000); await runTimers(25);
  assert.equal(nodes.get('stop-runtime').disabled, false);
  assert.equal(nodes.get('start-runtime').disabled, true);
  assert.equal(vm.runInContext('monitor.snapshot.frozen_bars[0].text', context), 'Retained legacy lyric');
  assert.equal(calls.some((call) => call.method === 'POST'), false);
  await nodes.get('stop-runtime').fire('click'); await settle(); await runTimers(25);
  assert.ok(calls.some((call) => call.url === '/api/control/stop' && call.method === 'POST'));
  assert.equal(nodes.get('start-runtime').disabled, false);
  assert.equal(nodes.get('stop-runtime').disabled, true);
  await env.fire('pagehide'); await env.fire('beforeunload');
});
