/* Browser-level layout and shared-session checks; does not start playback. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const base = process.env.DEMO_URL || 'http://127.0.0.1:8018';
  const output = path.resolve(process.env.OUTPUT_DIR || 'output/inference_20261111/browser');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
  const errors = [], evidence = [];
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    page.on('pageerror', (error) => errors.push(error.message));
    const before = await (await context.request.get(`${base}/api/inference/state`)).json();
    assert.equal(before.supported, true, 'A real inference runtime is required');
    for (const [label, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
      await page.setViewportSize({ width, height });
      await page.goto(`${base}/inference`);
      await page.waitForFunction(() => document.querySelector('#connection')?.textContent === 'Connected');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.screenshot({ path: path.join(output, `${label}.png`), fullPage: true });
      evidence.push({ viewport: label, width, height, overflow: false });
    }
    const duplicate = await context.newPage();
    duplicate.on('pageerror', (error) => errors.push(error.message));
    await duplicate.goto(`${base}/inference`);
    await page.reload();
    const after = await (await context.request.get(`${base}/api/inference/state`)).json();
    assert.equal(after.session_id, before.session_id, 'Refresh must retain shared session');
    assert.equal(after.epoch, before.epoch, 'Page attachment must not restart the mode');
    assert.equal(after.state, before.state, 'Page attachment must not start or stop playback');
    assert.equal(after.owner_id, before.owner_id, 'Second tab must not steal capture');
    await page.goto(`${base}/`);
    const legacy = await (await context.request.get(`${base}/api/state`)).json();
    assert.equal(legacy.session_id, before.session_id, 'Both pages must use the same session');
    assert.deepEqual(errors, []);
    const result = { evidence, session_id: before.session_id, epoch: before.epoch,
      state: before.state, page_errors: errors, refresh_and_duplicate_tab_preserved_session: true };
    fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
