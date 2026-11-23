/* Headless browser checks for PIT WALL.
 *
 * Usage:
 *   npm install playwright
 *   node verify_page.js
 *
 * If Playwright's bundled Chromium is not installed, point PITWALL_CHROMIUM at
 * a Chrome/Chromium executable.
 */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const URL = process.env.PITWALL_URL || 'http://localhost:8000/index.html';
const EXE = process.env.PITWALL_CHROMIUM || undefined;
const OUT = __dirname;

function fail(message) {
  console.error('FAIL:', message);
  process.exitCode = 1;
}

(async () => {
  const browser = await chromium.launch({ headless: true, ...(EXE ? { executablePath: EXE } : {}) });
  const page = await browser.newPage({ viewport: { width: 1560, height: 1000 } });
  const errors = [];
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push('console: ' + msg.text()); });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('requestfailed', (r) => errors.push('requestfailed: ' + r.url()));

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.PITWALL && window.PITWALL.debug && window.PITWALL.debug(), null, { timeout: 20000 });
  await page.waitForTimeout(400);

  const initial = await page.evaluate(() => window.PITWALL.state());
  const debug = await page.evaluate(() => window.PITWALL.debug());
  fs.writeFileSync(path.join(OUT, 'page_debug.json'), JSON.stringify(debug, null, 2));

  if (initial.year !== 2011 || initial.raceId !== 843 || initial.lap !== 31) fail('default race state is not 2011/843/lap 31');
  if (initial.focusId !== 20 || initial.aheadId !== 13) fail('default focus/ahead is not VET/MAS');
  if (initial.playing) fail('page opens playing instead of paused');
  if (!debug || Math.abs(Object.values(debug.probs).reduce((a, b) => a + b, 0) - 1) > 1e-9) fail('probabilities do not sum to 1');

  const desktop = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    towerRows: document.querySelectorAll('#towerBody tr').length,
    feedItems: document.querySelectorAll('#feed .feed-item').length,
    probRows: document.querySelectorAll('#probBars .prob-row').length,
    ucVerdict: document.getElementById('ucVerdict').textContent.trim(),
    ucP: document.getElementById('ucP').textContent,
  }));
  if (desktop.scrollWidth > desktop.innerWidth) fail('horizontal overflow at 1560 px');
  if (desktop.towerRows !== 23 || desktop.probRows !== 23) fail('default timing tower or probability bars are incomplete');
  if (!desktop.feedItems) fail('race feed is empty');

  await page.screenshot({ path: path.join(OUT, 'screenshot-dark.png'), fullPage: true });

  await page.click('#themeBtn');
  await page.waitForTimeout(150);
  const light = await page.evaluate(() => ({
    theme: document.documentElement.dataset.theme,
    button: document.getElementById('themeBtn').textContent,
    bg: getComputedStyle(document.body).backgroundColor,
    fg: getComputedStyle(document.body).color,
    panel: getComputedStyle(document.querySelector('.panel')).backgroundColor,
  }));
  if (light.theme !== 'light' || light.bg !== 'rgb(255, 255, 255)' || light.fg !== 'rgb(0, 0, 0)') fail('light theme is not an inversion');
  await page.screenshot({ path: path.join(OUT, 'screenshot-light.png'), fullPage: true });
  await page.click('#themeBtn');

  // Neutral shading: 2011 Canada is heavily neutralised; go to lap 30.
  await page.selectOption('#raceSel', '847');
  await page.waitForFunction(() => window.PITWALL.state().raceId === 847);
  await page.evaluate(() => window.PITWALL.go(30));
  await page.waitForTimeout(300);
  const neutral = await page.evaluate(() => {
    const ids = ['lapChartWrap', 'gapChartWrap', 'paceChartWrap', 'strategyChartWrap', 'probChartWrap'];
    return Object.fromEntries(ids.map((id) => [id, document.querySelectorAll('#' + id + ' .neutral-band').length]));
  });
  for (const [id, count] of Object.entries(neutral)) if (count < 1) fail('no neutral shading in ' + id);

  // Focus interaction and replay transport.
  const nextRow = page.locator('#towerBody tr[data-id]:not(.focus)').first();
  const nextId = Number(await nextRow.getAttribute('data-id'));
  await nextRow.click();
  const switched = await page.evaluate(() => window.PITWALL.state());
  if (switched.focusId !== nextId) fail('clicking a timing-tower row did not change focus');
  await page.click('.speed[data-speed="8"]');
  await page.click('#playBtn');
  await page.waitForTimeout(500);
  await page.click('#playBtn');
  const played = await page.evaluate(() => window.PITWALL.state());
  if (played.lap <= 30 || played.playing) fail('play/pause did not advance and pause');

  // A driver with laps but no results row.
  await page.selectOption('#seasonSel', '2026');
  await page.waitForFunction(() => window.PITWALL.state().year === 2026);
  await page.waitForTimeout(300);
  await page.selectOption('#raceSel', '1188');
  await page.waitForFunction(() => window.PITWALL.state().raceId === 1188);
  await page.waitForTimeout(300);
  const noResult = await page.evaluate(() => ({
    note: document.getElementById('towerNote').textContent,
    fit: window.PITWALL.debug().probs.FIT,
  }));
  if (!noResult.note.includes('FIT') || !noResult.note.includes('no result row in source data')) fail('no-result driver label is missing');
  if (typeof noResult.fit !== 'number') fail('no-result driver has no probability');

  // Mobile overflow after all interactions.
  await page.setViewportSize({ width: 400, height: 900 });
  await page.waitForTimeout(400);
  const mobile = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    bodyWidth: document.body.scrollWidth,
    mainCols: getComputedStyle(document.querySelector('main')).gridTemplateColumns,
  }));
  if (mobile.scrollWidth > mobile.innerWidth || mobile.bodyWidth > mobile.innerWidth) fail('horizontal overflow at 400 px');
  if (!/^[0-9.]+px$/.test(mobile.mainCols)) fail('layout is not a single column at 400 px');
  await page.screenshot({ path: path.join(OUT, 'screenshot-400.png'), fullPage: true });

  if (errors.length) {
    console.error(errors.join('\n'));
    fail(errors.length + ' browser console/page errors');
  }

  console.log('Default:', JSON.stringify(initial));
  console.log('Desktop:', JSON.stringify(desktop));
  console.log('Light:', JSON.stringify(light));
  console.log('Neutral bands:', JSON.stringify(neutral));
  console.log('After focus/play:', JSON.stringify(played));
  console.log('No-result:', JSON.stringify(noResult));
  console.log('Mobile:', JSON.stringify(mobile));
  console.log('Console errors: 0');
  console.log(process.exitCode ? 'FAILED' : 'PASS');
  await browser.close();
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
