# Signo - pose-based word-level sign recognition (prototype)

A cartoony, interactive prototype for the WLASL mini-project: MediaPipe Holistic landmarks -> BiLSTM -> word.
Everything in the web app runs locally in the browser (no video is uploaded).

## Run the app
```
cd signo
python -m http.server 8000
```
Open http://localhost:8000 in Chrome (camera needs localhost or https; an internet connection is needed once to load TF.js / MediaPipe from jsDelivr).

Tabs: **Play** (live recognition + sentence builder + speak), **Teach** (add words, record takes with a countdown),
**Brain** (train in-browser with live loss/accuracy chart, top-1/top-5 + per-class precision/recall, import/export),
**Challenge** (5-round game).

## Training workflow ("train it more, later")
1. **In browser:** Teach -> record 10+ takes per word (+ a `(nothing)` class) -> Brain -> Train. Recordings and the model persist in IndexedDB.
2. **Share data:** Brain -> Save, share & load -> *Export recordings* gives `signo-dataset.json`.
3. **WLASL / bigger models (Python):**
   ```
   cd python && pip install -r requirements.txt
   python extract_wlasl.py --wlasl-json WLASL_v0.3.json --videos ./videos --top-n 100 --out wlasl100.json
   python train.py --data wlasl100.json signo-dataset.json --out model_out --epochs 80
   ```
4. **Back into the app:** Brain -> *Load brain files* -> select `model.json`, the `.bin` file(s) and `labels.json` from `model_out/`.

## Feature format (browser and Python must match - see `python/common.py`)
Per frame 100 floats = 7 pose points + 21 left-hand + 21 right-hand points (x,y, shoulder-centred, divided by shoulder width) + 2 hand-present flags.
Clips are trimmed to the span where a hand is visible and resampled to 30 frames. Dataset JSON: `{version, seqLen:30, featureDim:100, labels, samples:[{label, split?, data:[30][100]}]}`.

## Caveats
* Not yet run end-to-end on real WLASL data or with a live webcam in my environment; expect small fixes.
* Models trained on your own webcam takes recognise *you* well; signer-independent accuracy needs WLASL-scale data.
* `extract_wlasl.py` approximates WLASL-N by instance count; use the official `nslt_N.json` for benchmark-comparable numbers. Many WLASL videos are no longer downloadable.
* Face keypoints are not used yet (nose only); adding face/mouth features is an easy extension in `common.py` + `extract()` in `index.html`.

## Ideas to extend
Transformer / ST-GCN models, continuous (sentence-level) recognition, per-signer normalisation, more augmentation (time warp), ONNX/TFLite export.
