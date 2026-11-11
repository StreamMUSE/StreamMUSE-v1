/* Real Chromium/UI/API acceptance. No mocked responses, media access, or rap controls. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/Users/andrewyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const ROOT = path.resolve(__dirname, '../../..');
const OUT = path.join(ROOT, 'output/research_day_20261105/characters_browser');
const BASE = process.env.DEMO_URL || 'http://127.0.0.1:8019';
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'timed_out']);
const TIMEOUT = 80000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const results = {
    started_at: new Date().toISOString(), base: BASE,
    evidence: 'Real Chromium, mounted character API, and configured model endpoint; no response mocks.',
    scope: 'Character experiments only. No microphone permissions, story writes, or rap runtime controls.',
    runs: [], screenshots: [], page_errors: [], failed_http: [], mutations: [], cleanup: [], passed: false,
  };
  let browser;
  let context;
  let page;
  let ownedRun = null;
  const save = () => fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify(results, null, 2) + '\n');

  async function get(route) {
    const response = await context.request.get(BASE + route, { timeout: 15000 });
    assert.equal(response.ok(), true, `GET ${route}: HTTP ${response.status()} ${await response.text()}`);
    return response.json();
  }

  async function status() { return get('/api/characters/status'); }

  async function events(runId) {
    const data = await get(`/api/characters/events?after=0&run_id=${encodeURIComponent(runId)}`);
    assert.equal(data.run_id, runId, 'Event page must belong to the tested run');
    assert.ok(Array.isArray(data.events));
    assert.equal(new Set(data.events.map((event) => event.event_id)).size, data.events.length, 'Duplicate event IDs');
    for (let i = 0; i < data.events.length; i += 1) {
      assert.equal(data.events[i].sequence, i + 1, 'Event sequence must be contiguous');
      assert.equal(data.events[i].run_id, runId);
    }
    assert.equal(data.next_cursor, data.events.length);
    return data;
  }

  async function waitTerminal(runId) {
    const deadline = Date.now() + TIMEOUT;
    while (Date.now() < deadline) {
      const snapshot = await status();
      assert.equal(snapshot.run_id, runId, 'Another actor replaced this run; refusing to continue');
      if (TERMINAL.has(snapshot.state)) return snapshot;
      await sleep(600);
    }
    throw new Error(`Run ${runId} did not terminate within ${TIMEOUT} ms`);
  }

  async function start(name, mode, policy) {
    const previous = await status();
    assert.ok(previous.state === 'idle' || TERMINAL.has(previous.state), 'Refusing to interrupt an existing character run');
    const tab = mode === 'dialogue' ? 'characters' : 'coordination';
    await page.locator(`#tab-${tab}`).click();
    if (mode === 'dialogue') await page.locator('input[name="character-mode"][value="dialogue"]').check();
    else await page.locator('#coordination-policy').selectOption(policy);
    await page.waitForFunction((id) => !document.getElementById(id).disabled, `${tab}-start`, { timeout: 15000 });
    const pending = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/characters/start' && response.request().method() === 'POST', { timeout: 20000 });
    await page.locator(`#${tab}-start`).click();
    const response = await pending;
    assert.equal(response.status(), 202, `Start failed: ${await response.text()}`);
    assert.deepEqual(response.request().postDataJSON(), { mode, policy, turns: 8 });
    const snapshot = await response.json();
    assert.ok(snapshot.run_id);
    assert.notEqual(snapshot.run_id, previous.run_id, 'Restart must create a new run, not pretend to resume');
    ownedRun = snapshot.run_id;
    const run = { name, mode, policy, run_id: ownedRun, start: snapshot, started_at: new Date().toISOString(), passed: false };
    results.runs.push(run);
    save();
    console.log(`START ${name}: ${ownedRun}`);
    return run;
  }

  function verifyModelEvidence(snapshot) {
    assert.ok(snapshot.model && snapshot.model_url, 'Missing actual model identity/endpoint');
    assert.ok(snapshot.metrics.model_calls > 0, 'No actual model calls');
    const generated = snapshot.proposals.filter((proposal) => proposal.text?.trim());
    assert.ok(generated.length > 0, 'No generated model responses');
    for (const proposal of generated) {
      assert.ok(Number.isFinite(proposal.model_latency_ms) && proposal.model_latency_ms >= 0, 'Missing model response latency');
      assert.ok(proposal.context_hash && proposal.request_id, 'Missing model request provenance');
    }
    assert.equal(snapshot.metrics.errors, 0, 'Transport/model failures cannot count as expected policy behavior');
  }

  function verifyAccepted(snapshot, target) {
    assert.equal(snapshot.state, 'completed', snapshot.error || 'Run failed');
    assert.equal(snapshot.accepted.length, target);
    assert.equal(snapshot.metrics.accepted_count, target);
    assert.equal(snapshot.metrics.success, true);
    assert.equal(snapshot.metrics.correctness_source, 'host_referee');
    assert.equal(snapshot.metrics.gpu_cancellation_confirmed, false);
    assert.equal(new Set(snapshot.accepted.map((event) => event.actor_id)).size, 2);
    for (let i = 0; i < snapshot.accepted.length; i += 1) {
      assert.ok(snapshot.accepted[i].text?.trim(), 'Empty accepted message');
      if (i) assert.notEqual(snapshot.accepted[i].actor_id, snapshot.accepted[i - 1].actor_id, 'Actors must alternate');
    }
    if (snapshot.mode === 'counting') {
      assert.equal(snapshot.target, 10);
      assert.deepEqual(snapshot.accepted.map((event) => event.number), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      assert.equal(snapshot.metrics.valid_prefix_length, 10);
    }
    verifyModelEvidence(snapshot);
  }

  async function verifyRendered(snapshot, tab) {
    await page.locator(`#tab-${tab}`).click();
    await page.waitForFunction(({ state, mode, policy, tab }) =>
      document.getElementById(`${tab}-state`).textContent === state &&
      document.getElementById(`${tab}-active-mode`).textContent === mode &&
      document.getElementById(`${tab}-active-policy`).textContent === policy.replaceAll('_', ' '),
    { state: snapshot.state, mode: snapshot.mode, policy: snapshot.policy, tab }, { timeout: 15000 });
    if (tab === 'characters') {
      const expected = snapshot.accepted.map((event) => event.text);
      await page.waitForFunction((texts) => JSON.stringify(Array.from(document.querySelectorAll('#characters-log .event-content'), (node) => node.textContent)) === JSON.stringify(texts), expected, { timeout: 15000 });
      const names = Object.fromEntries(snapshot.actors.map((actor) => [actor.actor_id, actor.persona.split(':')[0]]));
      assert.deepEqual(await page.locator('#characters-log .event-label strong').allTextContents(), snapshot.accepted.map((event) => names[event.actor_id]));
      const personas = await page.locator('#characters-personas').innerText();
      for (const actor of snapshot.actors) assert.ok(personas.includes(actor.persona), 'Missing persona label');
    } else {
      const expected = Object.fromEntries(Object.entries(snapshot.metrics).map(([key, value]) => [key.replaceAll('_', ' '), typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(2) : String(value ?? '--')]));
      await page.waitForFunction((expectedMetrics) => {
        const actual = Object.fromEntries(Array.from(document.querySelectorAll('#coordination-metrics tr'), (row) => [row.cells[0]?.textContent, row.cells[1]?.textContent]));
        return Object.entries(expectedMetrics).every(([name, value]) => actual[name] === value);
      }, expected, { timeout: 15000 });
      await page.locator('#coordination-log .event-label span').filter({ hasText: /run finished/ }).waitFor({ timeout: 15000 });
      assert.equal(await page.locator('#coordination-start').innerText(), 'Start count');
    }
    const shownError = await page.locator(`#${tab}-error`).isVisible();
    assert.equal(shownError, Boolean(snapshot.error), 'Rendered failure state does not match backend');
    if (snapshot.error) assert.equal(await page.locator(`#${tab}-error`).innerText(), snapshot.error);
  }

  async function screenshot(name, tab, width) {
    await page.setViewportSize({ width, height: width === 360 ? 800 : 1000 });
    await page.locator(`#tab-${tab}`).click();
    const layout = await page.evaluate((tab) => {
      const controls = Array.from(document.querySelectorAll(`#panel-${tab} button, #panel-${tab} select`)).filter((element) => element.getClientRects().length);
      const rectangles = controls.map((element) => ({ id: element.id || element.textContent.trim(), rect: element.getBoundingClientRect().toJSON(), overflow: element.scrollWidth > element.clientWidth + 1 }));
      const overlaps = [];
      for (let i = 0; i < rectangles.length; i += 1) for (let j = i + 1; j < rectangles.length; j += 1) {
        const a = rectangles[i].rect; const b = rectangles[j].rect;
        if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1) overlaps.push([rectangles[i].id, rectangles[j].id]);
      }
      return { viewport: innerWidth, body: document.body.scrollWidth, document: document.documentElement.scrollWidth, rectangles, overlaps };
    }, tab);
    assert.ok(layout.body <= width + 1 && layout.document <= width + 1, `${tab} overflows ${width}px viewport`);
    assert.deepEqual(layout.overlaps, [], 'Controls overlap');
    assert.ok(layout.rectangles.every((item) => !item.overflow), 'Control text overflows');
    const target = page.locator(`#${tab}-start`);
    const before = await target.boundingBox();
    await target.hover();
    const after = await target.boundingBox();
    assert.ok(Math.abs(before.width - after.width) < 0.5 && Math.abs(before.height - after.height) < 0.5, 'Hover changes control dimensions');
    const filename = `${name}-${tab}-${width}.png`;
    await page.screenshot({ path: path.join(OUT, filename), fullPage: true });
    results.screenshots.push({ file: filename, width, tab, layout });
  }

  async function finish(run, expectedKind) {
    const snapshot = await waitTerminal(run.run_id);
    run.final = snapshot;
    run.events = await events(run.run_id);
    save();
    if (expectedKind === 'naive_failure') {
      assert.equal(snapshot.state, 'failed', 'Naive condition unexpectedly succeeded or timed out');
      assert.match(snapshot.error, /^naive publication violated counting rules:/);
      assert.equal(snapshot.metrics.success, false);
      assert.ok(snapshot.metrics.invalid_proposals > 0);
      assert.ok(snapshot.publication_attempts.length > snapshot.accepted.length);
      verifyModelEvidence(snapshot);
    } else verifyAccepted(snapshot, snapshot.mode === 'dialogue' ? 8 : 10);
    for (const tab of ['characters', 'coordination']) {
      await verifyRendered(snapshot, tab);
      await screenshot(run.name, tab, 1440);
      await screenshot(run.name, tab, 360);
    }
    run.passed = true;
    run.finished_at = new Date().toISOString();
    ownedRun = null;
    save();
    console.log(`PASS ${run.name}: ${snapshot.state}, ${snapshot.accepted.length} accepted, ${snapshot.metrics.model_calls} model calls`);
  }

  try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH || '/Users/andrewyang/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing' });
    context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, permissions: [] });
    page = await context.newPage();
    page.setDefaultTimeout(15000);
    page.on('pageerror', (error) => results.page_errors.push(String(error)));
    page.on('request', (request) => {
      if (!['GET', 'HEAD', 'OPTIONS'].includes(request.method())) results.mutations.push({ method: request.method(), path: new URL(request.url()).pathname, body: request.postData() });
    });
    page.on('response', (response) => {
      if (response.status() >= 400 && new URL(response.url()).pathname.startsWith('/api/characters')) results.failed_http.push({ status: response.status(), url: response.url() });
    });
    results.initial_status = await status();
    assert.ok(results.initial_status.state === 'idle' || TERMINAL.has(results.initial_status.state), 'Character engine is already running; no changes made');
    await page.goto(BASE + '/research', { waitUntil: 'domcontentloaded' });
    await page.locator('#tab-characters').click();

    await finish(await start('dialogue', 'dialogue', 'fixed_turn'));
    await finish(await start('coordinated-counting', 'counting', 'coordinated'));
    await finish(await start('naive-counting', 'counting', 'naive'), 'naive_failure');

    const cancelled = await start('cancel-dialogue', 'dialogue', 'fixed_turn');
    const cancelDeadline = Date.now() + 15000;
    let current;
    do {
      current = await status();
      assert.equal(current.run_id, cancelled.run_id);
      assert.equal(current.state, 'running', 'Cancellation run completed before Stop could be tested');
      if (current.metrics.model_calls > 0) break;
      await sleep(50);
    } while (Date.now() < cancelDeadline);
    assert.ok(current.metrics.model_calls > 0, 'Stop must be tested during an actual model request');
    const stopResponse = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/characters/stop' && response.request().method() === 'POST', { timeout: 20000 });
    await page.locator('#characters-stop').click();
    assert.equal((await stopResponse).status(), 200);
    cancelled.final = await waitTerminal(cancelled.run_id);
    cancelled.events = await events(cancelled.run_id);
    assert.equal(cancelled.final.state, 'cancelled');
    assert.ok(cancelled.events.events.some((event) => event.kind === 'cancellation_requested' && event.reason === 'user_stop'));
    assert.ok(cancelled.events.events.some((event) => event.kind === 'run_finished' && event.state === 'cancelled'));
    assert.equal(cancelled.final.metrics.gpu_cancellation_confirmed, false);
    await verifyRendered(cancelled.final, 'characters');
    await screenshot(cancelled.name, 'characters', 1440);
    await screenshot(cancelled.name, 'characters', 360);
    cancelled.passed = true;
    ownedRun = null;
    save();
    console.log(`PASS cancel-dialogue: ${cancelled.final.metrics.model_calls} calls; GPU cancellation remains unconfirmed`);

    await finish(await start('restart-after-cancel', 'dialogue', 'fixed_turn'));
    results.restart_semantics = 'Fresh run after Stop; the API has no same-run resume endpoint.';
    assert.deepEqual(results.page_errors, []);
    assert.deepEqual(results.failed_http, []);
    assert.ok(results.mutations.every((request) => ['/api/characters/start', '/api/characters/stop'].includes(request.path)), 'Unexpected mutation outside the character API');
    results.passed = true;
  } catch (error) {
    results.failure = String(error.stack || error);
    process.exitCode = 1;
    if (page && !page.isClosed()) {
      try { await page.screenshot({ path: path.join(OUT, 'failure.png'), fullPage: true }); } catch (_) { /* Preserve the primary failure. */ }
    }
  } finally {
    // Never stop an unrelated run or touch the rap runtime during cleanup.
    if (ownedRun && context) {
      try {
        const current = await status();
        if (current.run_id === ownedRun && !TERMINAL.has(current.state)) {
          const response = await context.request.post(BASE + '/api/characters/stop', { timeout: 20000 });
          assert.equal(response.ok(), true, 'Owned-run cleanup failed');
          results.cleanup.push({ run_id: ownedRun, result: await response.json() });
        }
      } catch (error) { results.cleanup.push({ error: String(error) }); results.passed = false; process.exitCode = 1; }
    }
    if (browser) await browser.close();
    results.finished_at = new Date().toISOString();
    save();
    console.log(JSON.stringify({ passed: results.passed, runs: results.runs.map((run) => ({ name: run.name, passed: run.passed, state: run.final?.state })), screenshots: results.screenshots.length, summary: path.join(OUT, 'summary.json'), failure: results.failure }, null, 2));
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
