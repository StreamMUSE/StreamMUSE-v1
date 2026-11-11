/* One real keyboard capture and audible response, followed by refresh and Stop. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const base = process.env.DEMO_URL || 'http://127.0.0.1:8018';
  const output = path.resolve(process.env.OUTPUT_DIR || 'output/inference_20261111/browser-cycle');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
  const errors = [], controls = [];
  try {
    const context = await browser.newContext();
    const read = async () => (await context.request.get(`${base}/api/inference/state`)).json();
    const initial = await read();
    assert.equal(initial.state, 'stopped');
    const config = await context.request.post(`${base}/api/inference/config`, {
      data: { mode: 'cycle', bpm: 60, user_beats: 4, model_beats: 8 },
    });
    assert.equal(config.status(), 200);
    const legacy = await context.newPage();
    legacy.on('pageerror', (error) => errors.push(error.message));
    await legacy.goto(`${base}/`);
    const page = await context.newPage();
    await page.setViewportSize({ width: 1440, height: 1000 });
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('response', (response) => {
      if (response.request().method() === 'POST' && response.url().includes('/api/inference/'))
        controls.push({ url: response.url(), status: response.status() });
    });
    await page.goto(`${base}/inference`);
    await page.locator('#start').click();
    await page.waitForFunction(() => document.querySelector('#phase').textContent === 'USER');
    for (let index = 0; index < 7; index++) {
      await page.waitForTimeout(500);
      await page.keyboard.press('Space');
    }
    await page.waitForFunction(() => document.querySelector('#phase').textContent === 'MODEL', null, { timeout: 120000 });
    const performing = await read();
    await legacy.waitForFunction(() => !document.querySelector('#stop-runtime').disabled, null, { timeout: 5000 });
    assert.match(await legacy.locator('#inference-mode-status').textContent(), /Active:/);
    const response = performing.requests.findLast((request) => request.epoch === performing.epoch);
    assert.ok(response.context.flow_ticks.length > 0);
    assert.ok(response.context.flow_ticks[0] > 0, 'A leading rest must not move to tick zero');
    await page.waitForTimeout(500);
    assert.ok(await page.locator('#performance-grid .syllable').count() > 0,
      'Real model syllables must be projected into the musical grid');
    await page.screenshot({ path: path.join(output, 'model-playing.png'), fullPage: true });
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#connection').textContent === 'Connected');
    const refreshed = await read();
    assert.equal(refreshed.session_id, initial.session_id);
    assert.equal(refreshed.epoch, performing.epoch);
    assert.equal(refreshed.owner_id, performing.owner_id);
    assert.equal(refreshed.phase, 'MODEL');
    await page.locator('#stop').click();
    await page.waitForFunction(() => document.querySelector('#runtime-state').textContent === 'stopped', null, { timeout: 12000 });
    const final = await read();
    assert.equal(final.state, 'stopped');
    await legacy.waitForFunction(() => document.querySelector('#stop-runtime').disabled, null, { timeout: 5000 });
    assert.deepEqual(errors, []);
    assert.equal(controls.filter((item) => item.url.endsWith('/start')).length, 1);
    assert.equal(controls.filter((item) => item.url.endsWith('/tap')).length, 7);
    assert.ok(controls.every((item) => item.status < 400), JSON.stringify(controls));
    fs.writeFileSync(path.join(output, 'evidence.json'), JSON.stringify({ performing, refreshed, final, controls, errors }, null, 2) + '\n');
    console.log(JSON.stringify({ keyboard_taps: 7, preserved_leading_rest_tick: response.context.flow_ticks[0],
      refresh_preserved_audio_epoch: true, stop_completed: true, output }));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
