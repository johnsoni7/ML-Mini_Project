// Real Chrome: same input, does a TensorFlow.js model give the same answers as Keras? Also times predictions. Usage: node parity_model.js <model.json path> <fixture.json>
const { chromium } = require('playwright-core'), fs = require('fs');
const [path, fixture] = process.argv.slice(2);
const fx = JSON.parse(fs.readFileSync(fixture));
(async () => {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const p = await b.newPage(); const errors = []; p.on('pageerror', e => errors.push(e.message));
  await p.goto('http://127.0.0.1:8000/', { waitUntil: 'load', timeout: 120000 }); await p.waitForFunction(() => S.source, { timeout: 150000 });
  const r = await p.evaluate(async ({ path, clips }) => {
    const m = await tf.loadLayersModel(path); let same1 = 0, same3 = 0, maxDiff = 0; const times = [];
    const rank = a => a.map((v, k) => [v, k]).sort((x, y) => y[0] - x[0]).map(x => x[1]);
    for (const c of clips) {
      const t = performance.now(); const out = tf.tidy(() => m.predict(tf.tensor3d(c.seq.flat(), [1, T, D]))); const o = Array.from(await out.data()); out.dispose(); times.push(performance.now() - t);
      const ra = rank(o), rb = rank(c.keras_probs); same1 += ra[0] === rb[0]; same3 += ra.slice(0, 3).join() === rb.slice(0, 3).join();
      o.forEach((v, k) => { maxDiff = Math.max(maxDiff, Math.abs(v - c.keras_probs[k])); });
    }
    const s = times.slice(5).sort((x, y) => x - y);
    return { clips: clips.length, same_top1: same1, same_top3_order: same3, max_prob_diff: maxDiff, mean_ms: Math.round(s.reduce((x, y) => x + y, 0) / s.length), p95_ms: Math.round(s[Math.floor(s.length * .95)]), input: m.inputs[0].shape, outputs: m.outputs[0].shape[1], backend: tf.getBackend() };
  }, { path, clips: fx.clips });
  console.log(JSON.stringify(r), 'errors', errors);
  await b.close();
})().catch(e => { console.error('FAIL', e.message.slice(0, 400)); process.exit(1); });
