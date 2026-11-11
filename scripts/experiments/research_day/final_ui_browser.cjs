/* Live story provenance acceptance; preserves prior evidence and never resets. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/Users/andrewyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const diagnose = process.argv.includes('--diagnose');
  const phase = diagnose ? 'provenance-before' : 'provenance-after';
  const out = path.resolve(__dirname, '../../../output/research_day_20261105/final_ui');
  fs.mkdirSync(out, { recursive: true });
  const base = process.env.DEMO_URL || 'http://127.0.0.1:8019';
  const result = { phase, base, started_at: new Date().toISOString(), errors: [], samples: [], mutations: [], passed: false };
  const save = () => fs.writeFileSync(path.join(out, `${phase}.json`), JSON.stringify(result, null, 2) + '\n');
  let browser;
  let page;
  let ownsPlayback = false;
  const api = async (route, method = 'GET') => {
    const response = await fetch(base + route, { method, signal: AbortSignal.timeout(20000) });
    assert.ok(response.ok, `${route}: ${response.status} ${await response.clone().text()}`);
    return response.json();
  };
  const stop = async () => {
    const state = await api('/api/state');
    if (['running', 'priming'].includes(state.audio?.state)) await api('/api/control/stop', 'POST');
    const deadline = Date.now() + 20000;
    while (Date.now() < deadline) {
      if ((await api('/api/state')).audio?.state === 'stopped') return;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    throw new Error('Playback did not stop');
  };
  try {
    result.initial_state = await api('/api/state');
    assert.equal(result.initial_state.audio?.state, 'stopped', 'Refusing to interrupt another playback test');
    browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH || '/Users/andrewyang/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing' });
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, permissions: [] });
    page = await context.newPage();
    page.on('pageerror', (error) => result.errors.push(String(error)));
    page.on('request', (request) => { if (request.method() === 'POST') result.mutations.push({ path: new URL(request.url()).pathname, body: request.postData() }); });
    await page.goto(base + '/research', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('story-start').disabled);
    let story = await api('/api/story');
    if (!story.events.length) {
      for (const text of ['A courier finds a brass key hidden inside a city map.', 'The key opens a station locker where a missing notebook waits.']) {
        await page.locator('#story-text').fill(text);
        await page.locator('#story-submit').click();
        await page.waitForFunction(() => document.getElementById('story-feedback').dataset.state === 'success' && document.getElementById('story-text').value === '');
      }
      story = await api('/api/story');
    }
    assert.ok(story.version >= 1 && story.events.length >= 1, 'Need accepted story context for the provenance check');
    result.story = story;
    ownsPlayback = true;
    await page.locator('#story-start').click();
    await page.waitForFunction(() => document.getElementById('story-playback-state').textContent === 'running', null, { timeout: 120000 });
    const deadline = Date.now() + 35000;
    while (Date.now() < deadline) {
      const state = await api('/api/state');
      const current = state.current_playback?.bar ?? state.current_bar;
      const frozen = Object.values(state.frozen_bars || {}).find((bar) => bar.bar === current);
      const ui = await page.evaluate(() => ({
        playing_version: document.getElementById('story-playback-version').textContent,
        accepted_version: document.getElementById('story-latest-version').textContent,
        bar: document.getElementById('story-playback-bar').textContent,
        text: document.getElementById('story-playback-text').textContent,
        state: document.getElementById('story-playback-state').textContent,
      }));
      result.samples.push({ session_id: state.session_id, coordinator_epoch: state.coordinator_epoch, current_bar: current, frozen, ui });
      if (Number.isInteger(frozen?.story_version) && frozen.story_version > 0 &&
          ui.bar === `Bar ${current + 1}` && ui.text === frozen.text &&
          (diagnose || ui.playing_version === `v${frozen.story_version}`)) {
        result.observed_state = state;
        result.observed_ui = ui;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    assert.ok(result.observed_state, 'No matching positive frozen story version and rendered lyric within 35 seconds');
    const state = result.observed_state;
    assert.equal(story.session_id, `${state.session_id}-e${state.coordinator_epoch}`, 'Unexpected live session/epoch contract');
    if (diagnose) {
      assert.equal(result.observed_ui.playing_version, '--', 'Expected to reproduce the pre-fix hidden version');
      result.reproduced = true;
    } else {
      assert.match(result.observed_ui.playing_version, /^v[1-9]\d*$/);
      assert.equal(result.observed_ui.accepted_version, `v${story.version}`);
    }
    for (const width of [1440, 360]) {
      await page.setViewportSize({ width, height: width === 360 ? 800 : 1000 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'Horizontal overflow');
      await page.screenshot({ path: path.join(out, `${phase}-${width}.png`), fullPage: true });
    }
    assert.deepEqual(result.errors, []);
    assert.ok(result.mutations.every((request) => ['/api/story/events', '/api/control/start', '/api/control/stop'].includes(request.path)));
    result.passed = true;
  } catch (error) {
    result.failure = String(error.stack || error);
    process.exitCode = 1;
    if (page) await page.screenshot({ path: path.join(out, `${phase}-failure.png`), fullPage: true }).catch(() => {});
  } finally {
    if (ownsPlayback) {
      try { await stop(); result.final_state = await api('/api/state'); result.final_story = await api('/api/story'); }
      catch (error) { result.cleanup_error = String(error); result.passed = false; process.exitCode = 1; }
    }
    if (browser) await browser.close();
    result.finished_at = new Date().toISOString();
    save();
    console.log(JSON.stringify({ phase, passed: result.passed, reproduced: result.reproduced, observed_ui: result.observed_ui, failure: result.failure, cleanup_error: result.cleanup_error, evidence: path.join(out, `${phase}.json`) }, null, 2));
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
