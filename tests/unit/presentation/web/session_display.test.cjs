const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const staticDir = path.resolve(__dirname, '../../../../src/streammuse/presentation/web/static');

// Run the actual classic scripts: top-level const bindings are not window properties.
async function viewer() {
    const elements = new Map();
    const listeners = new Map();
    const fills = [];
    let nextFrame;
    let socket;
    let startFails = false;
    const startSnapshots = [];
    const ctx = {
        fillRect(...args) { fills.push({color: this.fillStyle, args}); },
        strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
        fillText() {}, setLineDash() {},
    };
    function element(id) {
        if (!elements.has(id)) {
            elements.set(id, {
                value: id === 'session-bpm' ? '80' : '', textContent: '',
                disabled: false, clientWidth: 1000, clientHeight: 1000,
                handlers: new Map(), getContext: () => ctx,
                addEventListener(name, callback) { this.handlers.set(name, callback); },
                checkValidity: () => true, reportValidity() {}, setCustomValidity() {},
            });
        }
        return elements.get(id);
    }
    function snapshot() {
        fills.length = 0;
        nextFrame();
        return {
            notes: fills.filter(r => ['#2196F3', '#4CAF50', 'rgba(158, 158, 158, 0.4)'].includes(r.color)).length,
            pressed: fills.filter(r => ['#bbdefb', '#1565C0'].includes(r.color)).length,
            tick: element('stat-tick').textContent,
            position: element('stat-position').textContent,
            hitRate: element('stat-hit-rate').textContent,
            latency: element('stat-round-trip').textContent,
        };
    }
    const sandbox = {
        console: {log() {}, error() {}},
        document: {getElementById: element, addEventListener: (name, cb) => listeners.set(name, cb)},
        location: {protocol: 'http:', host: 'localhost'},
        addEventListener() {},
        requestAnimationFrame(callback) { nextFrame = callback; },
        WebSocket: class { constructor() { socket = this; } },
        async fetch(url) {
            if (url === '/api/start') {
                startSnapshots.push(snapshot());
                return {ok: !startFails, json: async () => ({
                    success: !startFails, is_running: !startFails,
                    state: startFails ? 'idle' : 'running',
                })};
            }
            return {ok: true, json: async () => ({success: true, is_running: false, state: 'idle'})};
        },
    };
    sandbox.window = sandbox;
    vm.createContext(sandbox);
    for (const filename of ['piano.js', 'stats.js', 'main.js']) {
        vm.runInContext(readFileSync(path.join(staticDir, 'js', filename), 'utf8'), sandbox, {filename});
    }
    listeners.get('DOMContentLoaded')();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(sandbox.PianoVisualizer, undefined);
    assert.equal(sandbox.Stats, undefined);
    function emit(data) { socket.onmessage({data: JSON.stringify(data)}); }
    function populate() {
        emit({type: 'note', event: 'on', pitch: 60, tick: 28, source: 'user'});
        emit({type: 'note', event: 'on', pitch: 61, tick: 28, source: 'model'});
        emit({type: 'tick', tick: 28, bar: 1, beat: 3});
        emit({type: 'stats', hit_rate: 0.8, round_trip_ms: 150});
        vm.runInContext('PianoVisualizer.replaceNotes(28, [{pitch: 64, tick: 28, source: "model"}])', sandbox);
        assert.ok(snapshot().notes > 0);
        assert.equal(snapshot().pressed, 2);
    }
    return {
        populate, snapshot, startSnapshots, emit, element,
        start: () => element('btn-start').handlers.get('click')(),
        stop: () => element('btn-stop').handlers.get('click')(),
        failStart: () => { startFails = true; },
    };
}

const empty = {notes: 0, pressed: 0, tick: '0', position: 'Bar 0, Beat 0', hitRate: '--%', latency: '-- ms'};

test('Start clears prior notes, held keys, tick and statistics before the API request', async () => {
    const page = await viewer();
    page.populate();
    await page.stop();
    assert.ok(page.snapshot().notes > 0, 'Stop retains the finished session for inspection');
    await page.start();
    assert.deepEqual(page.startSnapshots, [empty]);
    assert.deepEqual(page.snapshot(), empty);
});

test('Stop/Start clears each round without preventing new-session notes from rendering', async () => {
    const page = await viewer();
    for (let round = 0; round < 3; round++) {
        page.populate();
        await page.stop();
        await page.start();
        assert.deepEqual(page.snapshot(), empty);
        page.emit({type: 'note', event: 'off', pitch: 60, tick: 29, source: 'user'});
        assert.equal(page.snapshot().notes, 0, 'Old note-off does not restore a cleared note');
        page.emit({type: 'note', event: 'on', pitch: 67, tick: 0, source: 'user'});
        assert.equal(page.snapshot().notes, 1);
        page.emit({type: 'note', event: 'off', pitch: 67, tick: 1, source: 'user'});
    }
    assert.equal(page.startSnapshots.length, 3);
});

test('Failed Start remains idle with a cleared display and usable Start button', async () => {
    const page = await viewer();
    page.populate();
    page.failStart();
    await page.start();
    assert.deepEqual(page.snapshot(), empty);
    assert.equal(page.element('service-state').textContent, 'Idle');
    assert.equal(page.element('btn-start').disabled, false);
});
