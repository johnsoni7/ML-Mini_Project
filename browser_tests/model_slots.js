// Real Chrome + the real app: which model starts, switching keeps labels and outputs matched, and the fallback works.
// Usage: node model_slots.js [screenshot_dir] [app_url]
const { chromium } = require('playwright-core');
const [shots = '', url = 'http://127.0.0.1:8000/'] = process.argv.slice(2);
const ARGS = ['--headless=new', '--use-angle=metal', '--enable-webgl', '--ignore-gpu-blocklist', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'];

async function open(b, routes = {}) {
  const c = await b.newContext({ viewport: { width: 1512, height: 860 }, permissions: ['camera'] }), p = await c.newPage(), errors = [];
  p.on('pageerror', e => errors.push(e.message));
  for (const [pattern, handler] of Object.entries(routes)) await p.route(pattern, handler);
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForFunction(() => S.source, { timeout: 60000 });
  return { c, p, errors };
}
const state = p => p.evaluate(() => ({ source: S.source, labels: S.labels.length, output: S.model.outputs[0].shape[1], input: S.model.inputs[0].shape.slice(1), trial: tf.tidy(() => S.model.predict(tf.zeros([1, T, D])).dataSync().length), chip: $('#brainChip').textContent, options: [...$('#modelSel').options].map(o => o.textContent), selected: $('#modelSel').selectedOptions[0].textContent, reportRows: document.querySelectorAll('#report tr').length - 1 }));

(async () => {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ARGS });

  let { c, p, errors } = await open(b);
  console.log('start:', await state(p));
  for (const k of ['dataset', 'dataset1000', 'dataset50v2', 'dataset1000v2']) {
    await p.selectOption('#modelSel', k);
    console.log('switched to', k, await state(p));
  }
  await p.click('#camBtn');  // Chrome's test pattern, which has no person in it
  await p.waitForFunction(() => S.camOn, { timeout: 60000 });
  await p.waitForTimeout(4000);
  console.log('no person on camera, word shown:', await p.evaluate(() => $('#predWord').textContent));
  const tick = await p.evaluate(async () => {
    const t = performance.now();
    for (let i = 0; i < 20; i++) { const o = tf.tidy(() => S.model.predict(tf.zeros([1, T, D]))); await o.data(); o.dispose(); }
    return (performance.now() - t) / 20;
  });
  console.log('mean prediction time, ms:', Math.round(tick));
  if (shots) await p.screenshot({ path: shots + '/ui_1000.png' });
  console.log('page errors:', errors);
  await c.close();

  const gone = r => r.fulfill({ status: 404, body: '' }), corrupt = r => r.request().url().endsWith('labels.json') ? r.fallback() : r.fulfill({ status: 200, contentType: 'application/json', body: '{"broken":true}' });
  for (const [name, routes] of [['large model missing', { '**/app_model_1000_v2/**': gone }], ['large model corrupt', { '**/app_model_1000_v2/**': corrupt }], ['only the old small model left', { '**/app_model_1000_v2/**': gone, '**/app_model_v2/**': gone, '**/app_model_1000/**': gone }]]) {
    ({ c, p, errors } = await open(b, routes));
    console.log(name, '->', await state(p), 'page errors:', errors);
    await c.close();
  }
  await b.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
