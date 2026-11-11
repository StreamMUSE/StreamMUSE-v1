/* Delay a real transcription response across Reset and verify session fencing. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('/Users/andrewyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const root = path.resolve(__dirname, '../../..');
  const out = path.join(root, 'output/research_day_20261105/microphone_reset_browser');
  fs.mkdirSync(out, {recursive: true});
  const base = 'http://127.0.0.1:8019';
  async function api(route, body) {
    const response = await fetch(base + route, body === undefined ? {} : {
      method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body),
    });
    assert.ok(response.ok, `${route}: ${response.status}`);
    return response.json();
  }
  assert.equal((await api('/api/control/settings')).state, 'stopped');
  await api('/api/control/reset', {});
  const original = await api('/api/story');
  const fixture = path.join(root, 'output/rap_alignment_20261028/glass_rooftops/glass_rooftops_bar_00_source.wav');
  const browser = await chromium.launch({headless: true,
    executablePath: '/Users/andrewyang/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${fixture}`],
  });
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, permissions: ['microphone']});
  const page = await context.newPage();
  const result = {original_session: original.session_id, reset_before_transcript_delivery: false, errors: []};
  page.on('pageerror', error => result.errors.push(String(error)));
  await page.route('**/api/microphone/transcribe', async route => {
    const response = await route.fetch();
    await api('/api/control/reset', {});
    result.reset_before_transcript_delivery = true;
    await route.fulfill({response});
  });
  try {
    await page.goto(base + '/research');
    await page.waitForFunction(() => document.querySelector('#story-version').textContent === '0');
    await page.locator('#tab-microphone').click();
    await page.locator('input[name="destination"][value="story"]').check();
    const box = await page.locator('#mic-hold').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.waitForFunction(() => document.querySelector('#mic-state').textContent.toLowerCase() === 'recording');
    await page.waitForTimeout(2900);
    await page.mouse.up();
    await page.waitForFunction(() => !document.querySelector('#microphone-error').hidden, null, {timeout: 40000});
    result.transcript = await page.locator('#mic-transcript').innerText();
    result.warning = await page.locator('#microphone-error').innerText();
    result.current_story = await api('/api/story');
    assert.ok(result.reset_before_transcript_delivery);
    assert.notEqual(result.current_story.session_id, original.session_id);
    assert.deepEqual(result.current_story.events, []);
    assert.match(result.transcript, /rain|glass|roof|sunrise/i);
    assert.match(result.warning, /session/i);
    assert.deepEqual(result.errors, []);
    await page.screenshot({path: path.join(out, 'reset-warning.png'), fullPage: true});
    result.passed = true;
  } catch (error) {
    result.passed = false;
    result.failure = String(error.stack || error);
    process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
