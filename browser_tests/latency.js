// Real Chrome: time predictions for an exported model. Usage: node latency.js <model_url_path> [repeats]
const { chromium } = require('playwright-core');
const [path, repeats = '40'] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const p = await b.newPage(); const errors = []; p.on('pageerror', e => errors.push(e.message));
  await p.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' }); await p.waitForFunction(() => S.source, { timeout: 60000 });
  const r = await p.evaluate(async ({ path, repeats }) => {
    const m = await tf.loadLayersModel(path), times = [];
    const once = async () => { const o = tf.tidy(() => m.predict(tf.randomNormal([1, T, D]))); const a = await o.data(); o.dispose(); return a; };
    for (let i = 0; i < 5; i++) await once();  // the first runs compile shaders
    let finite = true, sum = 0;
    for (let i = 0; i < repeats; i++) { const t = performance.now(); const a = await once(); times.push(performance.now() - t); finite = finite && a.every(Number.isFinite); sum = a.reduce((x, y) => x + y, 0); }
    times.sort((x, y) => x - y);
    return { input: m.inputs[0].shape, outputs: m.outputs[0].shape[1], mean_ms: Math.round(times.reduce((x, y) => x + y, 0) / times.length), p95_ms: Math.round(times[Math.floor(times.length * .95)]), finite, prob_sum: +sum.toFixed(3), backend: tf.getBackend() };
  }, { path, repeats: Number(repeats) });
  console.log(JSON.stringify(r), 'errors', errors);
  await b.close();
})().catch(e => { console.error('FAIL', e.message.slice(0, 300)); process.exit(1); });
