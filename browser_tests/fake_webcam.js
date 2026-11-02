// Real Chrome, real getUserMedia, real app loop, with the camera replaced by a file of held-out clips.
// Usage: node fake_webcam.js <fake.mjpeg> <meta.json> [loops] [threshold 40-95] [app_url]
const { chromium } = require('playwright-core'), fs = require('fs');
const [mjpeg, metaPath, loops = '1', thr = '', url = 'http://127.0.0.1:8000/'] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(metaPath));

(async () => {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-webgl', '--ignore-gpu-blocklist', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${mjpeg}`, '--autoplay-policy=no-user-gesture-required'] });
  const ctx = await b.newContext({ permissions: ['camera'] });
  const p = await ctx.newPage();
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForFunction(k => S.slots[k], process.env.SLOT || 'dataset', { timeout: 60000 });
  await p.evaluate(k => useModel(k), process.env.SLOT || 'dataset');
  if (thr) await p.evaluate(v => { const t = $('#thr'); t.value = v; t.dispatchEvent(new Event('input')); }, thr);
  await p.evaluate(() => { window.__t = []; const f = S.model.predict.bind(S.model); S.model.predict = x => { window.__t.push(performance.now()); return f(x); }; });
  await p.click('#camBtn');
  await p.waitForFunction(() => S.camOn && S.buffer.length >= 0, { timeout: 60000 });
  const t0 = Date.now(), log = [];
  const stop = Number(loops) * meta.total_s * 1000 + 15000;
  let shapeSeen = null;
  while (Date.now() - t0 < stop) {
    const s = await p.evaluate(() => {
      const tr = S.buffer.length ? trim(S.buffer) : null;
      return { word: $('#predWord').textContent, conf: parseFloat($('#confFill').style.width) || 0, top3: [...document.querySelectorAll('#top3 > div')].map(d => d.textContent.trim()), sentence: S.sentence.slice(), hands: { L: $('#pL').classList.contains('on'), R: $('#pR').classList.contains('on') }, buffer: S.buffer.length, seqLen: tr ? toSeq(tr).length : null };
    });
    if (s.seqLen) shapeSeen = s.seqLen;
    log.push({ t: (Date.now() - t0) / 1000, ...s });
    await p.waitForTimeout(100);
  }
  const words = log.map(l => l.word);
  const changes = words.filter((w, i) => i && w !== words[i - 1]).length;
  const states = {};
  words.forEach(w => { states[w] = (states[w] || 0) + 1; });
  const fps = await p.evaluate(async () => { const a = S.buffer.length, t = performance.now(); await new Promise(r => setTimeout(r, 2000)); return (S.buffer.length - a) / ((performance.now() - t) / 1000); });
  console.log(JSON.stringify({ threshold: thr || 'default 70', plan: meta.plan.map(x => x.sign), seconds_observed: (Date.now() - t0) / 1000, camera_frames_per_second: +fps.toFixed(1), prediction_gaps_ms: await p.evaluate(() => { const g = window.__t.slice(1).map((v, i) => v - window.__t[i]).filter(v => v < 1000).sort((a, b) => a - b); return { count: g.length, median: Math.round(g[g.length >> 1]), p95: Math.round(g[Math.floor(g.length * .95)]) }; }), sentence_committed: log[log.length - 1].sentence, display_state_counts: states, display_changes: changes, flat_seq_length_seen: shapeSeen, expected_flat_length: 3000, page_errors: errors }, null, 1));
  // Readable timeline: what the screen showed, one line per change
  let last = null;
  for (const l of log) if (l.word !== last) { console.log(`${l.t.toFixed(1).padStart(6)}s  ${l.word.padEnd(22)} conf=${l.conf.toFixed(0).padStart(3)}%  ${l.top3.slice(0, 3).join(' | ')}`); last = l.word; }
  await p.screenshot({ path: process.env.SHOT || '/dev/null', fullPage: true }).catch(() => {});
  await b.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
