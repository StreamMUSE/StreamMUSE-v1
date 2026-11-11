const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = path.resolve(__dirname, '../../src/streammuse/presentation/rap_demo/static/js/rap-research.js');
const sandbox = { window: {}, Float32Array, ArrayBuffer, DataView, TextEncoder, setTimeout, clearTimeout };
vm.createContext(sandbox);
if (fs.existsSync(source)) vm.runInContext(fs.readFileSync(source, 'utf8'), sandbox);
const api = sandbox.window.StreamMuseResearch || {};

test('WAV serializes little-endian mono PCM16 and clips samples', () => {
  assert.equal(typeof api.encodeWav, 'function');
  const bytes = api.encodeWav(new Float32Array([-2, -1, 0, 0.5, 1, 2, NaN]), 16000);
  const view = new DataView(bytes);
  assert.equal(Buffer.from(bytes).toString('ascii', 0, 4), 'RIFF');
  assert.equal(Buffer.from(bytes).toString('ascii', 8, 12), 'WAVE');
  assert.equal(view.getUint32(4, true), 50);
  assert.equal(view.getUint16(20, true), 1);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint32(28, true), 32000);
  assert.equal(view.getUint16(32, true), 2);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 14);
  assert.deepEqual(Array.from({ length: 7 }, (_, i) => view.getInt16(44 + i * 2, true)), [-32768, -32768, 0, 16384, 32767, 32767, 0]);
});

test('WAV caps duration at 15 seconds and rejects invalid sample rates', () => {
  assert.equal(typeof api.encodeWav, 'function');
  assert.equal(api.encodeWav(new Float32Array(16000 * 16), 16000).byteLength, 44 + 16000 * 15 * 2);
  for (const rate of [0, -1, NaN, 16000.5]) assert.throws(() => api.encodeWav(new Float32Array(1), rate));
});

test('event reduction deduplicates IDs, preserves updates and bounds logs', () => {
  assert.equal(typeof api.reduceEvents, 'function');
  const previous = [{ event_id: 'a', text: 'pending' }];
  const next = api.reduceEvents(previous, [{ event_id: 'a', text: 'done' }, { event_id: 'b', text: 'new' }], 2);
  assert.equal(previous[0].text, 'pending');
  assert.deepEqual(JSON.parse(JSON.stringify(next)), [{ event_id: 'a', text: 'done' }, { event_id: 'b', text: 'new' }]);
  assert.equal(api.reduceEvents(next, [{ event_id: 'c' }], 2)[0].event_id, 'b');
});

test('event reduction keys sequence by run and handles repeated anonymous snapshots', () => {
  assert.equal(typeof api.reduceEvents, 'function');
  const rows = [{ run_id: 'a', sequence: 1 }, { run_id: 'b', sequence: 1 }, { text: 'anonymous' }];
  assert.equal(api.reduceEvents(rows, rows).length, 3);
});

test('microphone reducer ignores stale results and cancels pending permission', () => {
  assert.equal(typeof api.reduceCapture, 'function');
  const waiting = api.reduceCapture({ phase: 'idle', token: 0 }, { type: 'press' });
  const cancelled = api.reduceCapture(waiting, { type: 'cancel' });
  assert.equal(cancelled.phase, 'idle');
  assert.notEqual(cancelled.token, waiting.token);
  assert.equal(api.reduceCapture(cancelled, { type: 'ready', token: waiting.token }).phase, 'idle');
  const active = api.reduceCapture(waiting, { type: 'ready', token: waiting.token });
  assert.equal(active.phase, 'recording');
  assert.equal(api.reduceCapture(active, { type: 'release' }).phase, 'transcribing');
  assert.equal(api.reduceCapture(waiting, { type: 'release' }).phase, 'idle');
});

test('zero log limit clears all retained events', () => {
  assert.equal(api.reduceEvents([{ event_id: 'a' }], [{ event_id: 'b' }], 0).length, 0);
});

test('model label never infers readiness from a model name', () => {
  assert.equal(api.modelLabel({ model: 'test-model' }), 'test-model / Status not reported');
  assert.equal(api.modelLabel({ model: 'test-model', available: false }), 'test-model / Unavailable');
  assert.equal(api.modelLabel({ model: { name: 'test-model', status: 'loading' } }), 'test-model / loading');
  assert.equal(api.modelLabel({ model: 'base.en', state: 'busy', transcriber_state: 'loading', available: true }), 'base.en / loading');
});

function recorderFixture({ permission, resume } = {}) {
  const calls = { permissions: 0, stopped: 0, closed: 0, audio: [], errors: [], states: [] };
  const stream = { getTracks: () => [{ stop() { calls.stopped += 1; } }] };
  const node = () => ({ connect() {}, disconnect() {} });
  let processor;
  class AudioContext {
    sampleRate = 16000;
    state = 'running';
    destination = {};
    resume() { return resume || Promise.resolve(); }
    close() { this.state = 'closed'; calls.closed += 1; return Promise.resolve(); }
    createMediaStreamSource() { return node(); }
    createScriptProcessor() { processor = node(); return processor; }
    createGain() { return { ...node(), gain: { value: 1 } }; }
  }
  const recorder = api.createRecorder({
    env: { AudioContext, navigator: { mediaDevices: { getUserMedia() { calls.permissions += 1; return permission || Promise.resolve(stream); } } } },
    onState: (state) => calls.states.push(state.phase),
    onError: (error) => calls.errors.push(error),
    onAudio: (wav, token, duration) => calls.audio.push({ wav, token, duration }),
  });
  return { recorder, calls, stream, feed(samples) { processor.onaudioprocess({ inputBuffer: { getChannelData: () => samples } }); } };
}

test('recorder has no automatic permission requests and stops tracks before transcription', async () => {
  const fixture = recorderFixture();
  assert.equal(fixture.calls.permissions, 0);
  await fixture.recorder.press();
  fixture.feed(new Float32Array(1600).fill(0.25));
  fixture.recorder.release();
  assert.equal(fixture.calls.permissions, 1);
  assert.equal(fixture.calls.stopped, 1);
  assert.equal(fixture.calls.closed, 1);
  assert.equal(fixture.calls.audio.length, 1);
  assert.equal(fixture.calls.audio[0].duration, 0.1);
  assert.equal(fixture.recorder.state().phase, 'transcribing');
  fixture.recorder.complete(fixture.calls.audio[0].token);
  assert.equal(fixture.recorder.state().phase, 'idle');
});

test('permission granted after cancel stops its tracks without recording', async () => {
  let grant;
  const permission = new Promise((resolve) => { grant = resolve; });
  const fixture = recorderFixture({ permission });
  const waiting = fixture.recorder.press();
  fixture.recorder.cancel();
  grant(fixture.stream);
  await waiting;
  assert.equal(fixture.calls.stopped, 1);
  assert.equal(fixture.calls.closed, 0);
  assert.equal(fixture.calls.audio.length, 0);
  assert.equal(fixture.recorder.state().phase, 'idle');
});

test('cancel while AudioContext resumes cannot reactivate capture', async () => {
  let resume;
  const pending = new Promise((resolve) => { resume = resolve; });
  const fixture = recorderFixture({ resume: pending });
  const starting = fixture.recorder.press();
  await new Promise(setImmediate);
  fixture.recorder.cancel();
  resume();
  await starting;
  assert.equal(fixture.calls.closed, 1);
  assert.ok(fixture.calls.stopped >= 1);
  assert.equal(fixture.calls.audio.length, 0);
  assert.equal(fixture.recorder.state().phase, 'idle');
});

test('recording cancellation discards audio and releases device', async () => {
  const fixture = recorderFixture();
  await fixture.recorder.press();
  fixture.feed(new Float32Array(1600));
  fixture.recorder.cancel();
  fixture.recorder.release();
  assert.equal(fixture.calls.audio.length, 0);
  assert.equal(fixture.calls.stopped, 1);
  assert.equal(fixture.calls.closed, 1);
});

test('sample cap stops capture at exactly 15 seconds without a second release', async () => {
  const fixture = recorderFixture();
  await fixture.recorder.press();
  fixture.feed(new Float32Array(16000 * 16));
  fixture.recorder.release();
  assert.equal(fixture.calls.audio.length, 1);
  assert.equal(fixture.calls.audio[0].duration, 15);
  assert.equal(fixture.calls.audio[0].wav.byteLength, 44 + 16000 * 15 * 2);
  assert.equal(fixture.calls.stopped, 1);
});

test('permission rejection reports an error without producing audio', async () => {
  const denied = Object.assign(new Error('Denied'), { name: 'NotAllowedError' });
  const fixture = recorderFixture({ permission: Promise.reject(denied) });
  await fixture.recorder.press();
  assert.equal(fixture.recorder.state().phase, 'error');
  assert.equal(fixture.calls.errors[0].name, 'NotAllowedError');
  assert.equal(fixture.calls.audio.length, 0);
});

test('release without captured samples reports a clear error', async () => {
  const fixture = recorderFixture();
  await fixture.recorder.press();
  fixture.recorder.release();
  assert.equal(fixture.calls.audio.length, 0);
  assert.match(fixture.calls.errors[0].message, /No audio captured/);
  assert.equal(fixture.calls.stopped, 1);
});

test('story budget counts cumulative UTF-8 bytes instead of characters', () => {
  assert.equal(typeof api.storyBudget, 'function');
  const snapshot = { events: [{ event_id: 'a', text: 'a'.repeat(510) }], limits: { events: 32, text_chars: 512, total_text_utf8_bytes: 512 } };
  assert.equal(api.storyBudget(snapshot, '\u00e9').canSubmit, true);
  const over = api.storyBudget(snapshot, '\u4e2d');
  assert.equal(over.draftBytes, 3);
  assert.equal(over.canSubmit, false);
  assert.match(over.reason, /512/);
});

test('story budget permits same-ID retry at full capacity but rejects additional events', () => {
  assert.equal(typeof api.storyBudget, 'function');
  const events = Array.from({ length: 32 }, (_, i) => ({ event_id: String(i), text: 'a'.repeat(16) }));
  assert.equal(api.storyBudget({ events }, 'b').canSubmit, false);
  assert.equal(api.storyBudget({ events }, events[0].text, '0').canSubmit, true);
  assert.equal(api.storyBudget({ events }, 'different', '0').canSubmit, false);
  assert.equal(api.storyBudget({ events: [] }, '  ').canSubmit, false);
});

test('WAV rejects sample rates above the transcription API limit', () => {
  assert.throws(() => api.encodeWav(new Float32Array(10), 96000));
});

function uiFixture(responses = {}, environment = {}) {
  class Node {
    constructor(id = '') { this.id = id; this.value = ''; this.textContent = ''; this.dataset = {}; this.listeners = {}; this.children = []; this.scrollHeight = 0; this.scrollTop = 0; this.clientHeight = 0; }
    addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
    async fire(name, event = {}) { for (const callback of this.listeners[name] || []) await callback({ preventDefault() {}, ...event }); }
    setAttribute(name, value) { this[name] = value; }
    setCustomValidity(value) { this.validationMessage = value; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    focus() {}
    setPointerCapture() {}
  }
  const html = fs.readFileSync(path.resolve(path.dirname(source), '../research.html'), 'utf8');
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map((match) => [match[1], new Node(match[1])]));
  const tabs = ['story', 'microphone', 'characters', 'coordination'].map((name) => { const tab = nodes.get(`tab-${name}`); tab.dataset.tab = name; return tab; });
  const document = new Node();
  document.hidden = false;
  document.getElementById = (id) => { assert.ok(nodes.has(id), `HTML must define ${id}`); return nodes.get(id); };
  document.createElement = () => new Node();
  document.querySelectorAll = (selector) => selector === '[data-tab]' ? tabs : [];
  const choices = { 'input[name="destination"]:checked': new Node(), 'input[name="character-mode"]:checked': new Node() };
  choices['input[name="destination"]:checked'].value = 'transcript';
  choices['input[name="character-mode"]:checked'].value = 'dialogue';
  document.querySelector = (selector) => choices[selector];
  nodes.get('characters-policy').value = 'fixed_turn';
  nodes.get('coordination-policy').value = 'coordinated';
  const requests = [];
  const window = new Node();
  Object.assign(window, environment);
  let uuid = 0;
  window.crypto = { randomUUID: () => `test-event-uuid${uuid++ ? `-${uuid}` : ''}` };
  window.fetch = async (url, options) => {
    requests.push({ url, options });
    const supplied = responses[url];
    const data = typeof supplied === 'function' ? await supplied(options) : supplied ?? {};
    if (data && typeof data.text === 'function' && typeof data.ok === 'boolean') return data;
    return { ok: true, status: 200, text: async () => JSON.stringify(data) };
  };
  const timers = new Map();
  let timerId = 0;
  const context = { window, document, TextEncoder, AbortController, Float32Array, ArrayBuffer, DataView,
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; }, clearTimeout: (id) => timers.delete(id),
    setInterval: (callback) => { timers.set(++timerId, callback); return timerId; }, clearInterval: (id) => timers.delete(id) };
  vm.runInNewContext(fs.readFileSync(source, 'utf8'), context);
  return { nodes, document, window, requests, choices, settle: () => new Promise(setImmediate) };
}

test('page mounts without microphone access and submits exact story contract', async () => {
  const ui = uiFixture({
    '/api/story': { session_id: 'session-a', enabled: false, version: 0, events: [], continuity: 'directions_only; playback history is not established' },
    '/api/story/events': { session_id: 'session-a', status: 'accepted', event: { event_id: 'test-event-uuid' }, effective_from_bar: 3 },
  });
  await ui.settle();
  assert.ok(ui.requests.some((request) => request.url === '/api/story'));
  assert.ok(ui.requests.every((request) => ['/api/story', '/api/state'].includes(request.url)));
  ui.nodes.get('story-text').value = 'The lights go out.';
  await ui.nodes.get('story-text').fire('input');
  await ui.nodes.get('story-form').fire('submit');
  const write = ui.requests.find((request) => request.options.method === 'POST');
  assert.equal(write.url, '/api/story/events');
  assert.deepEqual(JSON.parse(write.options.body), { text: 'The lights go out.', event_id: 'test-event-uuid', session_id: 'session-a' });
  assert.match(ui.nodes.get('story-feedback').textContent, /effective from bar 4/);
  assert.equal(ui.nodes.get('story-text').value, '');
  assert.equal(ui.nodes.get('story-submit').disabled, true);
  await ui.window.fire('pagehide');
});

test('characters use status route, fixed_turn policy, actor labels, and accepted dialogue', async () => {
  const snapshot = { run_id: 'run-a', state: 'completed', mode: 'dialogue', policy: 'fixed_turn', model: 'qwen-rap',
    actors: [{ actor_id: 'a', persona: 'Mira: inventor' }],
    accepted: [{ event_id: 'run-a:e1', sequence: 1, kind: 'accepted', actor_id: 'a', text: 'Hello.' }], metrics: { model_calls: 1 } };
  const ui = uiFixture({ '/api/characters/status': snapshot, '/api/characters/start': snapshot, '/api/characters/events?after=0&run_id=run-a': { run_id: 'run-a', events: snapshot.accepted, next_cursor: 1 } });
  await ui.settle();
  await ui.nodes.get('tab-characters').fire('click');
  await ui.settle();
  assert.ok(ui.requests.some((request) => request.url === '/api/characters/status'));
  await ui.nodes.get('characters-form').fire('submit');
  const write = ui.requests.find((request) => request.url === '/api/characters/start');
  assert.deepEqual(JSON.parse(write.options.body), { mode: 'dialogue', policy: 'fixed_turn', turns: 8 });
  assert.equal(ui.nodes.get('characters-log').children[0].children[0].children[0].textContent, 'Mira');
  assert.equal(ui.nodes.get('characters-log').children[0].children[1].textContent, 'Hello.');
  await ui.window.fire('pagehide');
});

test('playback projection uses current bar, frozen lyric, and one-based display numbering', () => {
  assert.equal(typeof api.playbackView, 'function');
  const view = api.playbackView({
    audio: { state: 'running' }, current_playback: { bar: 2, tick: 35 },
    current_segment: { bar: 1, topic: 'stale topic', text: 'Stale line.' },
    bars: { 2: { bar: 2, text: 'Draft line.', topic: 'rain', story_version: 9 }, 3: { bar: 3, text: 'Future line.' } },
    frozen_bars: { 2: { bar: 2, text: 'Frozen line.', source: 'generated', story_version: 1 } },
  });
  assert.equal(view.bar, 3);
  assert.equal(view.text, 'Frozen line.');
  assert.equal(view.topic, 'rain');
  assert.equal(view.state, 'running');
  assert.equal(view.storyVersion, 1);
});

test('playback projection does not promote a future lyric into the current bar', () => {
  assert.equal(typeof api.playbackView, 'function');
  const view = api.playbackView({ audio: { state: 'stopped' }, current_playback: { bar: 0 }, current_segment: { bar: 3, text: 'Future line' }, bars: { 3: { bar: 3, text: 'Future line' } } });
  assert.equal(view.text, 'Waiting for a lyric.');
  assert.equal(view.state, 'stopped');
});

test('latency formatter rounds finite milliseconds and handles absent values', () => {
  assert.equal(typeof api.formatMilliseconds, 'function');
  assert.equal(api.formatMilliseconds(1244.9693750022561), '1245 ms');
  assert.equal(api.formatMilliseconds(0), '0 ms');
  assert.equal(api.formatMilliseconds(null), 'Not reported');
  assert.equal(api.formatMilliseconds(NaN), 'Not reported');
});

test('story playback controls use existing start and stop routes without reset', async () => {
  let state = 'stopped';
  const ui = uiFixture({
    '/api/state': () => ({ audio: { state }, current_playback: { bar: 0 }, frozen_bars: { 0: { bar: 0, text: 'A lyric.' } } }),
    '/api/control/start': () => { state = 'running'; return { state: 'priming' }; },
    '/api/control/stop': () => { state = 'stopped'; return { state: 'stop_requested' }; },
  });
  await ui.settle();
  assert.equal(ui.nodes.get('story-playback-text').textContent, 'A lyric.');
  assert.equal(ui.nodes.get('story-start').disabled, false);
  assert.equal(ui.nodes.get('story-stop').disabled, true);
  await ui.nodes.get('story-start').fire('click');
  assert.equal(ui.nodes.get('story-stop').disabled, false);
  await ui.nodes.get('story-stop').fire('click');
  const writes = ui.requests.filter((request) => request.options.method === 'POST');
  assert.deepEqual(writes.map((request) => request.url), ['/api/control/start', '/api/control/stop']);
  assert.equal(ui.nodes.get('story-start').disabled, false);
  await ui.window.fire('pagehide');
});

function microphoneEnvironment() {
  let processor;
  let stopped = 0;
  const node = () => ({ connect() {}, disconnect() {} });
  class AudioContext {
    sampleRate = 16000;
    state = 'running';
    destination = {};
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createMediaStreamSource() { return node(); }
    createScriptProcessor() { processor = node(); return processor; }
    createGain() { return { ...node(), gain: { value: 1 } }; }
  }
  return {
    environment: { AudioContext, navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() { stopped += 1; } }] }) } } },
    feed: () => processor.onaudioprocess({ inputBuffer: { getChannelData: () => new Float32Array(1600).fill(0.25) } }),
    stopped: () => stopped,
  };
}

test('automatic transcript-to-story routing preserves microphone source and event ID', async () => {
  const microphone = microphoneEnvironment();
  const ui = uiFixture({
    '/api/story': { session_id: 'session-a', events: [] },
    '/api/microphone/transcribe': { text: 'The signal returns.', event_id: 'utterance-123', transcription_ms: 1.4 },
    '/api/story/events': { session_id: 'session-a', status: 'accepted', event: { event_id: 'utterance-123', source: 'microphone' } },
  }, microphone.environment);
  await ui.settle();
  ui.choices['input[name="destination"]:checked'].value = 'story';
  await ui.nodes.get('tab-microphone').fire('click');
  await ui.nodes.get('mic-hold').fire('pointerdown', { button: 0, pointerId: 1 });
  await ui.settle();
  microphone.feed();
  await ui.nodes.get('mic-hold').fire('pointerup', { pointerId: 1 });
  await ui.settle();
  const upload = ui.requests.find((request) => request.url === '/api/microphone/transcribe');
  assert.equal(upload.options.headers['Content-Type'], 'audio/wav');
  assert.equal(upload.options.body.byteLength, 44 + 1600 * 2);
  const submission = ui.requests.find((request) => request.url === '/api/story/events');
  assert.ok(submission, 'Story destination must submit the transcript');
  assert.deepEqual(JSON.parse(submission.options.body), { text: 'The signal returns.', event_id: 'utterance-123', source: 'microphone', session_id: 'session-a' });
  assert.equal(microphone.stopped(), 1);
  assert.equal(ui.nodes.get('mic-transcript').textContent, 'The signal returns.');
  assert.equal(ui.nodes.get('mic-feedback').textContent, 'Transcript added to Story.');
  await ui.window.fire('pagehide');
});

const storyWrites = (ui) => ui.requests.filter((request) => request.url === '/api/story/events').map((request) => JSON.parse(request.options.body));
const unavailableResponse = () => new Response(JSON.stringify({ detail: 'temporarily unavailable' }), { status: 503 });

test('typed retry reuses the originally captured session and event ID', async () => {
  let attempts = 0;
  const ui = uiFixture({
    '/api/story': { session_id: 'session-a', events: [] },
    '/api/story/events': () => ++attempts === 1 ? unavailableResponse() : { session_id: 'session-a', status: 'accepted' },
  });
  await ui.settle();
  ui.nodes.get('story-text').value = 'A retained direction.';
  await ui.nodes.get('story-form').fire('submit');
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui).length, 2);
  assert.deepEqual(storyWrites(ui)[0], storyWrites(ui)[1]);
  assert.equal(storyWrites(ui)[1].session_id, 'session-a');
  assert.equal(ui.nodes.get('story-feedback').dataset.state, 'success');
  await ui.window.fire('pagehide');
});

test('Reset clears a typed retry and blocks unchanged draft migration to the new session', async () => {
  let session = 'session-a';
  const ui = uiFixture({
    '/api/story': () => ({ session_id: session, events: [] }),
    '/api/story/events': () => session === 'session-a' ? unavailableResponse() : { session_id: session, status: 'accepted' },
  });
  await ui.settle();
  ui.nodes.get('story-text').value = 'Old session direction.';
  await ui.nodes.get('story-form').fire('submit');
  session = 'session-b';
  await ui.nodes.get('tab-story').fire('click');
  await ui.settle();
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui).length, 1, 'Old retry must not be rebound to the new session');
  assert.equal(ui.nodes.get('story-text').value, 'Old session direction.');
  assert.match(ui.nodes.get('story-feedback').textContent, /session|draft/i);
  ui.nodes.get('story-text').value = 'New session direction.';
  await ui.nodes.get('story-text').fire('input');
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui)[1].session_id, 'session-b');
  assert.notEqual(storyWrites(ui)[1].event_id, storyWrites(ui)[0].event_id);
  await ui.window.fire('pagehide');
});

test('late typed acknowledgement from before Reset cannot clear the draft or report current acceptance', async () => {
  let session = 'session-a';
  let acknowledge;
  const pending = new Promise((resolve) => { acknowledge = resolve; });
  const ui = uiFixture({ '/api/story': () => ({ session_id: session, events: [] }), '/api/story/events': () => pending });
  await ui.settle();
  ui.nodes.get('story-text').value = 'Old direction in flight.';
  const submission = ui.nodes.get('story-form').fire('submit');
  await ui.settle();
  session = 'session-b';
  await ui.nodes.get('tab-story').fire('click');
  await ui.settle();
  acknowledge({ session_id: 'session-a', status: 'accepted' });
  await submission;
  assert.equal(ui.nodes.get('story-text').value, 'Old direction in flight.');
  assert.notEqual(ui.nodes.get('story-feedback').dataset.state, 'success');
  assert.match(ui.nodes.get('story-error').textContent, /session/i);
  assert.equal(storyWrites(ui)[0].session_id, 'session-a');
  await ui.window.fire('pagehide');
});

test('unknown story session never produces an unguarded typed write', async () => {
  const ui = uiFixture({ '/api/story': { events: [] } });
  await ui.settle();
  ui.nodes.get('story-text').value = 'Unknown-session direction.';
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui).length, 0);
  assert.match(ui.nodes.get('story-error').textContent, /session/i);
  assert.equal(ui.nodes.get('story-text').value, 'Unknown-session direction.');
  await ui.window.fire('pagehide');
});

for (const initialSession of ['session-a', null]) {
  test(`microphone Story binding survives Reset during ASR (initial session ${initialSession})`, async () => {
    let session = initialSession;
    let transcribe;
    const pending = new Promise((resolve) => { transcribe = resolve; });
    const microphone = microphoneEnvironment();
    const ui = uiFixture({
      '/api/story': () => ({ session_id: session, events: [] }),
      '/api/microphone/transcribe': () => pending,
      '/api/story/events': () => ({ session_id: session, status: 'accepted' }),
    }, microphone.environment);
    await ui.settle();
    ui.choices['input[name="destination"]:checked'].value = 'story';
    await ui.nodes.get('tab-microphone').fire('click');
    await ui.settle();
    await ui.nodes.get('mic-hold').fire('pointerdown', { button: 0, pointerId: 1 });
    await ui.settle();
    microphone.feed();
    await ui.nodes.get('mic-hold').fire('pointerup', { pointerId: 1 });
    await ui.settle();
    session = 'session-b';
    await ui.nodes.get('tab-microphone').fire('click');
    await ui.settle();
    transcribe({ text: 'Retained transcript.', event_id: 'utterance-reset', transcription_ms: 2 });
    await ui.settle();
    assert.equal(storyWrites(ui).length, 0, 'ASR completion must not capture the new session');
    assert.equal(ui.nodes.get('mic-transcript').textContent, 'Retained transcript.');
    assert.match(ui.nodes.get('microphone-error').textContent, /session/i);
    assert.match(ui.nodes.get('mic-feedback').textContent, /Transcript remains/i);
    await ui.window.fire('pagehide');
  });
}

test('microphone Story session is bound at press, not at release', async () => {
  let session = 'session-a';
  const microphone = microphoneEnvironment();
  const ui = uiFixture({
    '/api/story': () => ({ session_id: session, events: [] }),
    '/api/microphone/transcribe': { text: 'Recorded before Reset.', event_id: 'utterance-held' },
    '/api/story/events': () => ({ session_id: session, status: 'accepted' }),
  }, microphone.environment);
  await ui.settle();
  ui.choices['input[name="destination"]:checked'].value = 'story';
  await ui.nodes.get('tab-microphone').fire('click');
  await ui.settle();
  await ui.nodes.get('mic-hold').fire('pointerdown', { button: 0, pointerId: 1 });
  await ui.settle();
  microphone.feed();
  session = 'session-b';
  await ui.nodes.get('tab-microphone').fire('click');
  await ui.settle();
  await ui.nodes.get('mic-hold').fire('pointerup', { pointerId: 1 });
  await ui.settle();
  assert.equal(storyWrites(ui).length, 0);
  assert.equal(ui.nodes.get('mic-transcript').textContent, 'Recorded before Reset.');
  assert.match(ui.nodes.get('microphone-error').textContent, /session/i);
  await ui.window.fire('pagehide');
});

test('unobserved Reset still sends the press-bound session for atomic backend rejection', async () => {
  let session = 'session-a';
  let transcribe;
  const pending = new Promise((resolve) => { transcribe = resolve; });
  const microphone = microphoneEnvironment();
  const ui = uiFixture({
    '/api/story': () => ({ session_id: session, events: [] }),
    '/api/microphone/transcribe': () => pending,
    '/api/story/events': () => new Response(JSON.stringify({ detail: 'story session changed' }), { status: 409 }),
  }, microphone.environment);
  await ui.settle();
  ui.choices['input[name="destination"]:checked'].value = 'story';
  await ui.nodes.get('tab-microphone').fire('click');
  await ui.settle();
  await ui.nodes.get('mic-hold').fire('pointerdown', { button: 0, pointerId: 1 });
  await ui.settle();
  microphone.feed();
  await ui.nodes.get('mic-hold').fire('pointerup', { pointerId: 1 });
  await ui.settle();
  session = 'session-b';
  transcribe({ text: 'Keep this transcript.', event_id: 'utterance-stale' });
  await ui.settle();
  assert.equal(storyWrites(ui).length, 1);
  assert.equal(storyWrites(ui)[0].session_id, 'session-a');
  assert.equal(ui.nodes.get('mic-transcript').textContent, 'Keep this transcript.');
  assert.notEqual(ui.nodes.get('mic-feedback').dataset.state, 'success');
  assert.match(ui.nodes.get('microphone-error').textContent, /session/i);
  await ui.window.fire('pagehide');
});

test('accepted POST followed by failed confirmation GET retains the exact typed retry identity', async () => {
  let reads = 0;
  let writes = 0;
  const ui = uiFixture({
    '/api/story': () => ++reads === 2 ? unavailableResponse() : { session_id: 'session-a', events: [] },
    '/api/story/events': () => ({ session_id: 'session-a', status: 'accepted', duplicate: ++writes > 1 }),
  });
  await ui.settle();
  ui.nodes.get('story-text').value = 'Possibly already accepted.';
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(ui.nodes.get('story-text').value, 'Possibly already accepted.');
  assert.notEqual(ui.nodes.get('story-feedback').dataset.state, 'success');
  assert.equal(ui.nodes.get('story-submit').disabled, false, 'Unchanged retry must remain available');
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui).length, 2);
  assert.deepEqual(storyWrites(ui)[0], storyWrites(ui)[1]);
  assert.equal(storyWrites(ui)[1].session_id, 'session-a');
  assert.match(ui.nodes.get('story-feedback').textContent, /Already accepted/);
  assert.equal(ui.nodes.get('story-feedback').dataset.state, 'success');
  await ui.window.fire('pagehide');
});

test('confirmation with missing session is unknown, not evidence of Reset', async () => {
  let reads = 0;
  const ui = uiFixture({
    '/api/story': () => ++reads === 2 ? { events: [] } : { session_id: 'session-a', events: [] },
    '/api/story/events': { session_id: 'session-a', status: 'accepted' },
  });
  await ui.settle();
  ui.nodes.get('story-text').value = 'Keep the retry key.';
  await ui.nodes.get('story-form').fire('submit');
  assert.notEqual(ui.nodes.get('story-feedback').dataset.state, 'success');
  await ui.nodes.get('tab-story').fire('click');
  await ui.settle();
  assert.equal(ui.nodes.get('story-submit').disabled, false);
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui).length, 2);
  assert.deepEqual(storyWrites(ui)[0], storyWrites(ui)[1]);
  await ui.window.fire('pagehide');
});

test('superseded confirmation GET preserves typed retry across concurrent microphone submission', async () => {
  let reads = 0;
  let confirm;
  const pending = new Promise((resolve) => { confirm = resolve; });
  const microphone = microphoneEnvironment();
  const ui = uiFixture({
    '/api/story': () => ++reads === 2 ? pending : { session_id: 'session-a', events: [] },
    '/api/story/events': { session_id: 'session-a', status: 'accepted' },
    '/api/microphone/transcribe': { text: 'A concurrent microphone event.', event_id: 'concurrent-mic' },
  }, microphone.environment);
  await ui.settle();
  ui.nodes.get('story-text').value = 'Typed event awaiting confirmation.';
  const typed = ui.nodes.get('story-form').fire('submit');
  await ui.settle();
  assert.equal(reads, 2);
  ui.choices['input[name="destination"]:checked'].value = 'story';
  await ui.nodes.get('tab-microphone').fire('click');
  await ui.settle();
  await ui.nodes.get('mic-hold').fire('pointerdown', { button: 0, pointerId: 1 });
  await ui.settle();
  microphone.feed();
  await ui.nodes.get('mic-hold').fire('pointerup', { pointerId: 1 });
  await ui.settle();
  assert.equal(storyWrites(ui).length, 2);
  assert.equal(ui.nodes.get('mic-feedback').dataset.state, 'success');
  confirm({ session_id: 'session-a', events: [] });
  await typed;
  assert.equal(ui.nodes.get('story-text').value, 'Typed event awaiting confirmation.');
  assert.notEqual(ui.nodes.get('story-feedback').dataset.state, 'success');
  assert.equal(ui.nodes.get('story-submit').disabled, false);
  await ui.nodes.get('story-form').fire('submit');
  assert.equal(storyWrites(ui).length, 3);
  assert.deepEqual(storyWrites(ui)[0], storyWrites(ui)[2]);
  assert.equal(storyWrites(ui)[2].session_id, 'session-a');
  await ui.window.fire('pagehide');
});

test('playback provenance distinguishes frozen context from the latest accepted story version', async () => {
  const ui = uiFixture({
    '/api/state': { session_id: 'session-a', current_playback: { bar: 0 }, frozen_bars: { 0: { bar: 0, text: 'Earlier context lyric.', story_version: 1 } } },
    '/api/story': { session_id: 'session-a', version: 2, events: [] },
  });
  await ui.settle();
  assert.equal(ui.nodes.get('story-playback-version').textContent, 'v1');
  assert.equal(ui.nodes.get('story-latest-version').textContent, 'v2');
  await ui.window.fire('pagehide');
});

test('playback provenance never borrows a draft version or compares versions across sessions', async () => {
  assert.equal(api.playbackView({ current_playback: { bar: 0 }, bars: { 0: { bar: 0, story_version: 4 } } }).storyVersion, null);
  const ui = uiFixture({
    '/api/state': { session_id: 'session-a', current_playback: { bar: 0 }, frozen_bars: { 0: { bar: 0, story_version: 1 } } },
    '/api/story': { session_id: 'session-b', version: 0, events: [] },
  });
  await ui.settle();
  assert.equal(ui.nodes.get('story-playback-version').textContent, '--');
  assert.equal(ui.nodes.get('story-latest-version').textContent, 'v0');
  await ui.window.fire('pagehide');
});

test('live publisher base session plus coordinator epoch matches the story controller session', async () => {
  const ui = uiFixture({
    '/api/state': { session_id: 'rap-live', coordinator_epoch: 7, current_playback: { bar: 0 }, frozen_bars: { 0: { bar: 0, text: 'Actual frozen lyric.', story_version: 2 } } },
    '/api/story': { session_id: 'rap-live-e7', version: 2, events: [] },
  });
  await ui.settle();
  assert.equal(ui.nodes.get('story-playback-version').textContent, 'v2');
  assert.equal(ui.nodes.get('story-latest-version').textContent, 'v2');
  await ui.window.fire('pagehide');
});

for (const epoch of [6, null, undefined]) {
  test(`base session alone cannot authorize playback provenance with epoch ${epoch}`, async () => {
    const ui = uiFixture({
      '/api/state': { session_id: 'rap-live', coordinator_epoch: epoch, current_playback: { bar: 0 }, frozen_bars: { 0: { bar: 0, story_version: 2 } } },
      '/api/story': { session_id: 'rap-live-e7', version: 3, events: [] },
    });
    await ui.settle();
    assert.equal(ui.nodes.get('story-playback-version').textContent, '--');
    assert.equal(ui.nodes.get('story-latest-version').textContent, 'v3');
    await ui.window.fire('pagehide');
  });
}

test('missing story session and missing epoch never establish a provenance match', async () => {
  const ui = uiFixture({
    '/api/state': { session_id: 'rap-live', current_playback: { bar: 0 }, frozen_bars: { 0: { bar: 0, story_version: 2 } } },
    '/api/story': { session_id: null, version: 2, events: [] },
  });
  await ui.settle();
  assert.equal(ui.nodes.get('story-playback-version').textContent, '--');
  await ui.window.fire('pagehide');
});
