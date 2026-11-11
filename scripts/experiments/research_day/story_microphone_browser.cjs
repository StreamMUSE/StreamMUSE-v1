/* End-to-end opt-in microphone routing and story playback controls. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('/Users/andrewyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const root = path.resolve(__dirname, '../../..');
  const out = path.join(root, 'output/research_day_20261105/story_microphone_browser');
  fs.mkdirSync(out, {recursive: true});
  const base = 'http://127.0.0.1:8019';
  async function api(route, body) {
    const response = await fetch(base + route, body === undefined ? {} : {
      method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body),
    });
    assert.ok(response.ok, `${route}: ${response.status} ${await response.clone().text()}`);
    return response.json();
  }
  assert.equal((await api('/api/control/settings')).state, 'stopped', 'Do not interrupt another running test');
  await api('/api/control/reset', {});
  const fixture = path.join(root, 'output/rap_alignment_20261028/glass_rooftops/glass_rooftops_bar_00_source.wav');
  const browser = await chromium.launch({headless: true,
    executablePath: '/Users/andrewyang/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${fixture}`],
  });
  const result = {capture_kind: 'prerecorded Chromium fake microphone, real local ASR and H200 rap', errors: []};
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, permissions: ['microphone'],
    recordVideo: {dir: path.join(out, 'video'), size: {width: 1440, height: 1000}}});
  const page = await context.newPage();
  page.on('pageerror', error => result.errors.push(String(error)));
  try {
    await page.goto(base + '/research');
    await page.locator('#story-text').fill('A courier reaches the city at sunrise.');
    await page.locator('#story-submit').click();
    await page.waitForFunction(() => document.querySelector('#story-version').textContent === '1');
    await page.locator('#tab-microphone').click();
    await page.locator('input[name="destination"][value="story"]').check();
    const box = await page.locator('#mic-hold').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.waitForFunction(() => document.querySelector('#mic-state').textContent.toLowerCase() === 'recording');
    await page.waitForTimeout(2900);
    const released = Date.now();
    await page.mouse.up();
    await page.waitForFunction(() => document.querySelector('#mic-feedback').textContent === 'Transcript added to Story.', null, {timeout: 40000});
    result.release_to_story_feedback_ms = Date.now() - released;
    result.transcript = await page.locator('#mic-transcript').innerText();
    result.feedback = await page.locator('#mic-feedback').innerText();
    result.story = await api('/api/story');
    assert.equal(result.story.events.length, 2);
    assert.equal(result.story.events[1].source, 'microphone');
    assert.equal(result.story.events[1].text, result.transcript);
    await page.screenshot({path: path.join(out, 'microphone-to-story.png'), fullPage: true});
    await page.locator('#tab-story').click();
    await page.locator('#story-start').click();
    await page.waitForFunction(() => document.querySelector('#story-playback-state').textContent === 'running', null, {timeout: 120000});
    await page.waitForTimeout(9000);
    result.playing_text = await page.locator('#story-playback-text').innerText();
    result.playing_bar = await page.locator('#story-playback-bar').innerText();
    assert.ok(result.playing_text && !result.playing_text.includes('Waiting'));
    await page.screenshot({path: path.join(out, 'story-playing.png'), fullPage: true});
    await page.locator('#story-stop').click();
    await page.waitForFunction(() => document.querySelector('#story-playback-state').textContent === 'stopped', null, {timeout: 20000});
    assert.equal((await api('/api/story')).events.length, 2, 'Stop preserves story');
    await page.locator('#story-start').click();
    await page.waitForFunction(() => document.querySelector('#story-playback-state').textContent === 'running', null, {timeout: 120000});
    await page.waitForTimeout(1200);
    await page.locator('#story-stop').click();
    await page.waitForFunction(() => document.querySelector('#story-playback-state').textContent === 'stopped', null, {timeout: 20000});
    result.stop_resume_without_reset = true;
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.locator('body').evaluate(el => el.scrollWidth > innerWidth + 1), false);
    await page.screenshot({path: path.join(out, 'story-mobile.png'), fullPage: true});
    assert.deepEqual(result.errors, []);
    result.passed = true;
  } catch (error) {
    result.failure = String(error.stack || error);
    result.passed = false;
    process.exitCode = 1;
  } finally {
    const state = await api('/api/control/settings').catch(() => ({}));
    if (state.state === 'running' || state.state === 'priming') await api('/api/control/stop', {}).catch(() => {});
    await context.close();
    await browser.close();
    fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
