// Real Chrome: how long the app takes to become ready, and whether a Teach model survives a reload.
const { chromium } = require('playwright-core');
const url = 'http://127.0.0.1:8000/';
(async () => {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-webgl', '--ignore-gpu-blocklist', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
  for (let i = 1; i <= 3; i++) {
    const c = await b.newContext({ permissions: ['camera'] }), p = await c.newPage(), bad = [];
    p.on('response', r => { if (r.status() >= 400) bad.push(r.status() + ' ' + new URL(r.url()).pathname); });
    const t0 = Date.now();
    await p.goto(url, { waitUntil: 'domcontentloaded' });
    await p.waitForFunction(() => typeof S !== 'undefined' && S.source, null, { timeout: 120000 });
    const ready = Date.now() - t0, t1 = Date.now();
    await p.click('#camBtn');
    await p.waitForFunction(() => S.camOn, null, { timeout: 120000 });
    const cam = Date.now() - t1;
    console.log(`load ${i}: model ready ${ready} ms, camera ready ${cam} ms, source ${await p.evaluate(() => S.source)}, failed requests ${JSON.stringify(bad)}`);
    await c.close();
  }
  const c = await b.newContext(), p = await c.newPage();
  await p.goto(url, { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => S.source, null, { timeout: 120000 });
  await p.evaluate(async () => { const m = buildModel(2); await m.save('indexeddb://signo-model'); await DB.setMeta('labels', ['aaa', 'bbb']); await DB.setMeta('activeModel', 'custom'); });
  await p.reload({ waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => S.source, null, { timeout: 120000 });
  console.log('after reload with a saved Teach model:', await p.evaluate(() => ({ source: S.source, labels: S.labels, slots: Object.keys(S.slots).filter(k => S.slots[k]) })));
  await b.close();
})().catch(e => { console.error('FAIL', e.message.slice(0, 300)); process.exit(1); });
