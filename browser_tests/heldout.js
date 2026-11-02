// Real Chrome + the real app: compares the browser's model and MediaPipe with Python on held-out clips.
// Usage: node heldout.js <fixture.json> <videos_dir> [app_url]
const { chromium } = require('playwright-core'), fs = require('fs'), path = require('path');
const [fixturePath, videosDir, url = 'http://127.0.0.1:8000/'] = process.argv.slice(2);
const fx = JSON.parse(fs.readFileSync(fixturePath));

(async () => {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const p = await (await b.newContext()).newPage();
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  await p.route('**/__clip/**', r => r.fulfill({ path: path.join(videosDir, decodeURIComponent(r.request().url().split('/__clip/')[1])), contentType: 'video/mp4' }));
  await p.goto(url, { waitUntil: 'networkidle' });
  await p.waitForFunction(k => S.slots[k], process.env.SLOT || 'dataset', { timeout: 60000 });
  await p.evaluate(k => useModel(k), process.env.SLOT || 'dataset');
  const info = await p.evaluate(() => ({ tfjs: tf.version.tfjs, backend: tf.getBackend(), webgl2: !!document.createElement('canvas').getContext('webgl2'), source: S.source, input: S.model.inputs[0].shape, output: S.model.outputs[0].shape, labels: S.labels }));
  if (JSON.stringify(info.labels) !== JSON.stringify(fx.labels)) throw new Error('label order differs between app_model and the Keras run');
  console.log('model in browser:', { ...info, labels: info.labels.length });

  // 1. Same numbers in: does TF.js give the same answers as Keras?
  const parity = await p.evaluate(async clips => {
    let same1 = 0, same3 = 0, maxDiff = 0;
    for (const c of clips) {
      const o = tf.tidy(() => Array.from(S.model.predict(tf.tensor3d(c.seq.flat(), [1, T, D])).dataSync()));
      const rank = a => a.map((v, k) => [v, k]).sort((x, y) => y[0] - x[0]).map(x => x[1]);
      const ra = rank(o), rb = rank(c.keras_probs);
      same1 += ra[0] === rb[0]; same3 += ra.slice(0, 3).join() === rb.slice(0, 3).join();
      o.forEach((v, k) => { maxDiff = Math.max(maxDiff, Math.abs(v - c.keras_probs[k])); });
    }
    return { clips: clips.length, same_top1: same1, same_top3_order: same3, max_prob_diff: maxDiff };
  }, fx.clips);
  console.log('Keras vs TF.js on identical sequences:', parity);

  // 2. Video in: browser MediaPipe -> browser features -> browser model.
  await p.evaluate(async () => {
    window.H = new Holistic({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}` });
    H.setOptions({ modelComplexity: 1, smoothLandmarks: true, refineFaceLandmarks: false, minDetectionConfidence: .5, minTrackingConfidence: .5 });
    await H.initialize();
  });
  const rows = [];
  for (const c of fx.clips.slice(0, Number(process.env.VIDEO_CLIPS || fx.clips.length))) {
    const r = await p.evaluate(async ({ video, label, seq }) => {
      const v = document.createElement('video'); v.muted = true; v.src = URL.createObjectURL(await (await fetch('/__clip/' + encodeURIComponent(video))).blob());  // a blob is fully seekable
      await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('video failed: ' + video)); });
      H.reset();
      const cv = document.createElement('canvas'); cv.width = Math.min(640, v.videoWidth); cv.height = Math.round(v.videoHeight * cv.width / v.videoWidth); const g = cv.getContext('2d');
      const hashes = new Set(); const frames = []; H.onResults(res => { const ex = extract(res); if (ex) frames.push({ f: ex.f, hands: ex.hands }); });
      const fps = 30, n = Math.floor(v.duration * fps);
      for (let i = 0; i < n; i++) {
        v.currentTime = (i + .5) / fps; await new Promise(res => { v.onseeked = res; });
        g.drawImage(v, 0, 0, cv.width, cv.height); hashes.add(g.getImageData(0, 0, cv.width, cv.height).data.filter((_, q) => q % 5003 === 0).join()); await H.send({ image: cv });
      }
      const tr = trim(frames); if (!tr) return { video, label, pred: null, note: 'no hands', frames: frames.length, handFrames: frames.filter(f => f.hands).length, distinct: hashes.size, n };
      const s = toSeq(tr), py = seq.flat(); let d = 0; for (let k = 0; k < s.length; k++) d += Math.abs(s[k] - py[k]);
      const o = tf.tidy(() => Array.from(S.model.predict(tf.tensor3d(s, [1, T, D])).dataSync())), rank = o.map((x, k) => [S.labels[k], x]).sort((a, b) => b[1] - a[1]);
      return { video, label, frames: frames.length, handFrames: frames.filter(f => f.hands).length, distinct: hashes.size, n, mean_abs_feature_diff: d / s.length, pred: rank[0][0], conf: rank[0][1], top5: rank.slice(0, 5).map(x => x[0]) };
    }, c);
    r.keras_pred = fx.labels[c.keras_probs.indexOf(Math.max(...c.keras_probs))];
    rows.push(r);
    console.log(`${r.label.padEnd(12)} browser=${String(r.pred).padEnd(12)} python=${r.keras_pred.padEnd(12)} conf=${(r.conf || 0).toFixed(2)} featDiff=${(r.mean_abs_feature_diff || 0).toFixed(3)} frames=${r.frames}/${r.n} hands=${r.handFrames} distinct=${r.distinct}`);
  }
  const ok = rows.filter(r => r.pred), mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  console.log(JSON.stringify({
    clips: rows.length, browser_top1: ok.filter(r => r.pred === r.label).length / rows.length, browser_top5: ok.filter(r => r.top5.includes(r.label)).length / rows.length,
    python_top1_same_clips: rows.filter(r => r.keras_pred === r.label).length / rows.length, browser_agrees_with_python: ok.filter(r => r.pred === r.keras_pred).length / rows.length,
    mean_abs_feature_diff: mean(ok.map(r => r.mean_abs_feature_diff)), page_errors: errors,
  }, null, 1));
  await b.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
