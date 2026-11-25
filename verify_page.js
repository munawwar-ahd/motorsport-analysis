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

const PAGE_URL = process.env.PITWALL_URL || 'http://localhost:8000/index.html';
const ORIGIN = new URL(PAGE_URL).origin;
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
  const external = [];
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push('console: ' + msg.text()); });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('requestfailed', (r) => errors.push('requestfailed: ' + r.url()));
  page.on('request', (r) => {
    const u = r.url();
    if (!u.startsWith(ORIGIN + '/') && !u.startsWith('data:')) external.push(u);
  });

  await page.goto(PAGE_URL, { waitUntil: 'networkidle' });
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

  // Circuit SVG and full-colour driver portraits on the replay view.
  await page.waitForFunction(() => {
    const p = document.getElementById('circuitPath');
    if (!p) return !!document.getElementById('circuitFallback');
    const b = p.getBBox();
    return b.width > 40 && b.height > 20;
  }, null, { timeout: 10000 });
  const circuit = await page.evaluate(() => {
    const p = document.getElementById('circuitPath');
    const b = p ? p.getBBox() : null;
    return {
      fallback: !!document.getElementById('circuitFallback'),
      width: b ? b.width : 0,
      height: b ? b.height : 0,
      viewBox: p ? p.ownerSVGElement.getAttribute('viewBox') : '',
      sf: !!document.querySelector('#circuitSvg .circuit-sf'),
      stroke: p ? getComputedStyle(p).strokeWidth : '',
    };
  });
  if (circuit.fallback || circuit.viewBox !== '0 0 220 130' || !circuit.sf) fail('circuit layout SVG is missing or incomplete');
  if (circuit.stroke !== '1.75px') fail('circuit track stroke is not 1.75px');
  await page.waitForFunction(() => {
    const imgs = [...document.querySelectorAll('#towerBody img.driver-img')];
    return imgs.length > 0 && imgs.every((i) => i.complete && i.naturalWidth > 0);
  }, null, { timeout: 10000 });
  const portraits = await page.evaluate(() => {
    const imgs = [...document.querySelectorAll('#towerBody img.driver-img')];
    return {
      count: imgs.length,
      loaded: imgs.filter((i) => i.complete && i.naturalWidth > 0).length,
      filters: [...new Set(imgs.map((i) => getComputedStyle(i).filter))],
    };
  });
  if (portraits.loaded !== portraits.count || portraits.filters.some((f) => f !== 'none')) {
    fail('timing-tower portraits failed to load or are not in full colour');
  }

  // Original constructor colours must reach the tower, probability bars and every chart.
  const team = await page.evaluate(async () => {
    const idx = await fetch('data/index.json').then((r) => r.json());
    const rgb = (hex) => `rgb(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)})`;
    const check = (els, prop) => els.map((el) => {
      const hex = el.dataset.teamColor || '';
      return { hex, actual: getComputedStyle(el)[prop], expected: rgb(hex) };
    });
    const tower = [...document.querySelectorAll('#towerBody tr[data-team-color]')].map((row) => {
      const hex = row.dataset.teamColor;
      return { hex, actual: getComputedStyle(row.querySelector('td')).borderLeftColor, expected: rgb(hex) };
    });
    const paths = check([...document.querySelectorAll(
      '#lapChartWrap path[data-team-color], #gapChartWrap path[data-team-color], #paceChartWrap path[data-team-color], #probChartWrap path[data-team-color]'
    )], 'stroke');
    const stints = check([...document.querySelectorAll('#strategyChartWrap rect[data-team-color]')], 'fill');
    const bars = [...document.querySelectorAll('#probBars .prob-row[data-team-color] .prob-fill')].map((fill) => {
      const hex = fill.parentElement.dataset.teamColor;
      return { hex, actual: getComputedStyle(fill).backgroundColor, expected: rgb(hex) };
    });
    const colors = [...new Set(tower.map((v) => v.hex))];
    const known = ['#FF8000', '#3671C6', '#E8002D', '#27F4D2', '#FF5F0F', '#469BFF'];
    return {
      refs: Object.keys(idx.constructorColors || {}).length,
      towerRows: tower.length,
      colors,
      knownOk: known.every((c) => colors.includes(c)),
      towerOk: tower.every((v) => v.actual === v.expected),
      paths: paths.length,
      pathsOk: paths.length > 20 && paths.every((v) => v.actual === v.expected),
      stints: stints.length,
      stintsOk: stints.length > 20 && stints.every((v) => v.actual === v.expected),
      bars: bars.length,
      barsOk: bars.length === 23 && bars.every((v) => v.actual === v.expected),
    };
  });
  if (team.refs !== 214 || team.towerRows !== 23 || team.colors.length < 8 || !team.knownOk) {
    fail('original constructor colours are missing from the timing tower');
  }
  if (!team.towerOk || !team.pathsOk || !team.stintsOk || !team.barsOk) {
    fail('team colours are not applied in the tower, probability bars and charts: ' + JSON.stringify(team));
  }
  const focusBadge = await page.evaluate(() => ({
    images: document.querySelectorAll('#focusBadge img.driver-img').length,
    text: document.getElementById('focusBadge').textContent.trim(),
  }));
  if (!focusBadge.text || focusBadge.images > 1) fail('focused-driver badge is incomplete');

  // Undercut sliders update the estimate, and Reset restores the live pair.
  const liveUc = await page.evaluate(() => ({
    gap: document.getElementById('ucGapVal').textContent,
    p: document.getElementById('ucP').textContent,
  }));
  await page.evaluate(() => {
    const el = document.getElementById('ucGap');
    el.value = '8';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(100);
  const movedUc = await page.evaluate(() => ({
    gap: document.getElementById('ucGapVal').textContent,
    p: document.getElementById('ucP').textContent,
  }));
  if (!movedUc.gap.startsWith('8.0') || movedUc.p === liveUc.p) fail('undercut slider did not update the estimate');
  await page.click('#ucReset');
  await page.waitForTimeout(100);
  const resetUc = await page.evaluate(() => ({
    gap: document.getElementById('ucGapVal').textContent,
    p: document.getElementById('ucP').textContent,
  }));
  if (resetUc.gap !== liveUc.gap || resetUc.p !== liveUc.p) fail('Reset to live did not restore the live undercut values');

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

  // View switcher and all-time champions table.
  await page.click('#viewChampsBtn');
  await page.waitForFunction(() => window.PITWALL.view() === 'champions');
  const champs = await page.evaluate(async () => {
    const idx = await fetch('data/index.json').then((r) => r.json());
    const imgs = [...document.querySelectorAll('#champRows img.driver-img')];
    const rgb = (hex) => `rgb(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)})`;
    const chips = [...document.querySelectorAll('#champRows .team-chip[data-team-color]')];
    const chipColors = [...new Set(chips.map((c) => c.dataset.teamColor))];
    return {
      hash: location.hash,
      rows: document.querySelectorAll('#champRows .champ-row').length,
      expected: idx.champions.length,
      replayLinks: document.querySelectorAll('#champRows button[data-open-year]').length,
      expectedLinks: idx.champions.filter((c) => c.has_replay).length,
      hiddenMain: document.getElementById('main').hidden,
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      totalImages: imgs.length,
      imageFilters: [...new Set(imgs.map((i) => getComputedStyle(i).filter))],
      chips: chips.length,
      expectedChips: idx.champions.filter((c) => c.constructor && c.constructor.name).length,
      chipColors,
      chipsOk: chips.every((c) => getComputedStyle(c).backgroundColor === rgb(c.dataset.teamColor)),
      chipKnownOk: ['#E8002D', '#3671C6', '#FF8000', '#27F4D2'].every((c) => chipColors.includes(c)),
    };
  });
  if (champs.hash !== '#champions' || !champs.hiddenMain) fail('champions view is not synced with the URL hash');
  if (champs.rows !== champs.expected || champs.rows !== 77) fail('not all champion seasons rendered');
  if (champs.replayLinks !== champs.expectedLinks) fail('replayable champion links are missing');
  if (champs.scrollWidth > champs.innerWidth) fail('horizontal overflow on champions view at 1560 px');
  await page.waitForFunction(() => {
    const imgs = [...document.querySelectorAll('#champRows img.driver-img')];
    return imgs.length > 0 && imgs.every((i) => i.complete && i.naturalWidth > 0);
  }, null, { timeout: 15000 });
  const champImages = await page.evaluate(() => ({
    loaded: [...document.querySelectorAll('#champRows img.driver-img')].filter((i) => i.complete && i.naturalWidth > 0).length,
    total: document.querySelectorAll('#champRows img.driver-img').length,
  }));
  if (champImages.loaded !== champImages.total || champs.imageFilters.some((f) => f !== 'none')) {
    fail('champion portraits failed to load or are not in full colour');
  }
  if (champs.chips !== champs.expectedChips || !champs.chipsOk || !champs.chipKnownOk) {
    fail('champion constructor colour badges are missing or wrong: ' + JSON.stringify(champs));
  }
  await page.fill('#champSearch', '2026');
  await page.waitForTimeout(100);
  const filtered = await page.evaluate(() => document.querySelectorAll('#champRows .champ-row').length);
  if (filtered !== 1) fail('champion search filter did not narrow the table');
  await page.fill('#champSearch', '');
  const firstLink = page.locator('#champRows button[data-open-year]').first();
  const targetYear = Number(await firstLink.getAttribute('data-open-year'));
  const targetRace = Number(await firstLink.getAttribute('data-open-race'));
  await firstLink.click();
  await page.waitForFunction((t) => {
    const s = window.PITWALL.state();
    return window.PITWALL.view() === 'replay' && s.year === t.year && s.raceId === t.race;
  }, { year: targetYear, race: targetRace });
  const jumped = await page.evaluate(() => ({ hash: location.hash, view: window.PITWALL.view(), bbox: document.getElementById('circuitPath').getBBox().width }));
  if (jumped.hash !== '#replay' || jumped.view !== 'replay' || !(jumped.bbox > 40)) fail('OPEN REPLAY did not switch back to the replayed season');
  await page.evaluate(() => window.PITWALL.openRace(2011, 843, 31, 20));
  await page.waitForFunction(() => window.PITWALL.state().raceId === 843 && window.PITWALL.state().lap === 31);

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

  await page.click('#viewChampsBtn');
  await page.waitForFunction(() => window.PITWALL.view() === 'champions');
  await page.waitForTimeout(200);
  const mobileChamps = await page.evaluate(() => ({
    rows: document.querySelectorAll('#champRows .champ-row').length,
    scrollWidth: document.documentElement.scrollWidth,
    bodyWidth: document.body.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  if (mobileChamps.rows !== 77) fail('champion rows missing at 400 px');
  if (mobileChamps.scrollWidth > mobileChamps.innerWidth || mobileChamps.bodyWidth > mobileChamps.innerWidth) fail('horizontal overflow on champions view at 400 px');
  await page.click('#viewReplayBtn');

  if (external.length) fail('external runtime requests: ' + [...new Set(external)].join(', '));
  if (errors.length) {
    console.error(errors.join('\n'));
    fail(errors.length + ' browser console/page errors');
  }

  console.log('Default:', JSON.stringify(initial));
  console.log('Desktop:', JSON.stringify(desktop));
  console.log('Circuit SVG:', JSON.stringify(circuit));
  console.log('Replay portraits:', JSON.stringify(portraits));
  console.log('Team colours:', JSON.stringify(team));
  console.log('Champions:', JSON.stringify({
    rows: champs.rows,
    replayLinks: champs.replayLinks,
    filterRows: filtered,
    images: champImages,
    chips: champs.chips,
    chipColors: champs.chipColors.length,
  }));
  console.log('Mobile champions:', JSON.stringify(mobileChamps));
  console.log('Light:', JSON.stringify(light));
  console.log('Neutral bands:', JSON.stringify(neutral));
  console.log('After focus/play:', JSON.stringify(played));
  console.log('No-result:', JSON.stringify(noResult));
  console.log('Mobile:', JSON.stringify(mobile));
  console.log('Console errors: 0');
  console.log(process.exitCode ? 'FAILED' : 'PASS');
  await browser.close();
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
