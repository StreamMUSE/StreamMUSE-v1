/* Real browser/HTTP/ASR checks with a prerecorded microphone fixture. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || '/Users/andrewyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const root = path.resolve(__dirname, '../../..');
  const out = path.join(root, 'output/research_day_20261105/browser');
  fs.mkdirSync(out, {recursive: true});
  const fixture = path.join(root, 'output/rap_alignment_20261028/glass_rooftops/glass_rooftops_bar_00_source.wav');
  const base = process.env.DEMO_URL || 'http://127.0.0.1:8019';
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH || '/Users/andrewyang/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing', args: [
    '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-audio-capture=${fixture}`,
  ]});
  const results = {base, fixture, capture_kind: 'prerecorded WAV through Chromium fake microphone; real browser capture and local ASR', errors: []};
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}, permissions: ['microphone']});
    const page = await context.newPage();
    page.on('pageerror', error => results.errors.push(String(error)));
    await page.goto(base + '/', {waitUntil: 'networkidle'});
    await page.screenshot({path: path.join(out, 'existing-dashboard.png'), fullPage: true});
    assert.equal(await page.locator('body').evaluate(el => el.scrollWidth > innerWidth + 1), false);
    await page.goto(base + '/research', {waitUntil: 'networkidle'});
    await page.locator('#story-version').filter({hasText: /\d/}).waitFor();
    await page.screenshot({path: path.join(out, 'research-desktop.png'), fullPage: true});
    await page.locator('#tab-microphone').click();
    const hold = page.locator('#mic-hold');
    const box = await hold.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.waitForFunction(() => document.querySelector('#mic-state').textContent.toLowerCase().includes('recording'));
    await page.waitForTimeout(3400);
    const released = Date.now();
    await page.mouse.up();
    await page.waitForFunction(() => {
      const text = document.querySelector('#mic-transcript').textContent;
      return text && !text.includes('No transcript yet');
    }, null, {timeout: 40000});
    results.release_to_transcript_ms = Date.now() - released;
    results.transcript = await page.locator('#mic-transcript').innerText();
    results.transcription_display = await page.locator('#mic-latency').innerText();
    results.microphone_feedback = await page.locator('#mic-feedback').innerText();
    assert.match(results.transcript, /rain|glass|roof|sunrise/i);
    await page.screenshot({path: path.join(out, 'microphone-transcript.png'), fullPage: true});
    await page.locator('#tab-characters').click();
    await page.screenshot({path: path.join(out, 'characters-idle.png'), fullPage: true});
    await page.locator('#tab-coordination').click();
    await page.screenshot({path: path.join(out, 'coordination-idle.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    for (const tab of ['story', 'microphone', 'characters', 'coordination']) {
      await page.locator('#tab-' + tab).click();
      assert.equal(await page.locator('body').evaluate(el => el.scrollWidth > innerWidth + 1), false, tab + ' mobile overflow');
      await page.screenshot({path: path.join(out, `${tab}-mobile.png`), fullPage: true});
    }
    assert.deepEqual(results.errors, []);
    results.passed = true;
  } catch (error) {
    results.passed = false;
    results.failure = String(error.stack || error);
    process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(results, null, 2) + '\n');
    console.log(JSON.stringify(results, null, 2));
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
