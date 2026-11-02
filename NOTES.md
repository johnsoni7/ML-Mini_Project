# Signo engineering notes (chronological)

All numbers below were read from real command output. Times are IST, 1-2 Nov 2026.

## 23:11 Repository inspection
- Git: one commit (3d69284). Remote `origin` is GitHub `mokshagna1810Y/ML_miniProject`. **Nothing has been pushed.** Work is on local branch `feature/asl-citizen-model`.
- Synopsis found outside the repo: `~/Desktop/John Soni_Mokshagna_WLASL MiniProject Synopsis.pdf`.
- README said itself it had never run end to end, so nothing in it was taken as verified.

### Conflicts between the synopsis and the brief
| Topic | Synopsis says | Brief says | What was done |
|---|---|---|---|
| Dataset | WLASL100/WLASL300 subset | ASL Citizen first, WLASL only as fallback | ASL Citizen, because selective download worked. **Tell the faculty guide about this change.** |
| Features | hand, **face** and body keypoints | frozen spec, no face features | Frozen spec. The nose is the only face point. |
| Model | BiLSTM/GRU or GCN | BiLSTM, optional GRU | BiLSTM. Within synopsis scope. |
| Metrics | top-1/top-5, per-class precision/recall, real-time text demo | same | Followed. |

### Bugs found in the original code
1. `train.py`: val and test merged into one held-out set, used for early stopping and for the final score (test leakage).
2. `extract_wlasl.py`: one MediaPipe tracker shared across all videos, so tracking state crossed between clips.
3. `index.html`: top word displayed even when below the confidence threshold.
4. `index.html`: training or loading a model disposed/overwrote the active model. A dataset model and a Teach model could not coexist.
5. `extract_wlasl.py`: WLASL-N approximated by instance count. Not fixed, since WLASL is not used. Do not cite its numbers as official WLASL100.

## 23:13 Python environment
- Machine: arm64 Mac, 10 CPU cores, 16 GB RAM. Python 3.11 was not installed, so conda env `signo311` supplies 3.11.18 for the venvs.
- `.venv`: tensorflow 2.16.2, tf_keras 2.16.0, mediapipe 0.10.14, numpy 1.26.4, opencv 4.11.0, scikit-learn 1.9.1.
- `.venv-tfjs` (separate, converter only): tensorflow 2.16.2, tf_keras 2.16.0, tensorflowjs 4.22.0.
- TensorFlow sees CPU only. No GPU plugin was installed.

## 23:13 ASL Citizen feasibility (passed in under 2 minutes)
- HEAD: 200, Content-Length 45,924,134,223, Accept-Ranges: bytes, no redirect.
- Range first 1 KB: 206, `Content-Range: bytes 0-1023/45924134223`, 1024 bytes received. Last 1 KB: 206.
- remotezip: opened in 3.2 s, 83,406 entries (83,399 videos). One 460 KB video extracted in 1.02 s.
- Metadata: `splits/{train,val,test}.csv` (columns Participant ID, Video file, Gloss, ASL-LEX Code), `use.txt` (licence).
- Official split: 40,154 / 10,304 / 32,941 clips, 35 / 6 / 11 signers, 2,731 glosses, signer overlap 0 / 0 / 0 (measured).

### Failure and fix
First full run crashed opening the zip: `501 Server Error: Unsupported client range`. Direct curl range requests still returned 206, so the suffix-style request was the suspect (not proven: the error was intermittent). Fix 1 of 2: `RemoteZip(URL, support_suffix_range=False)`, workers lowered from 6 to 4. Worked first time.

## 23:15 Licence (Microsoft Research License Terms, read from `use.txt`)
- Non-commercial, non-revenue research only (s1).
- Data and modifications of it must not be distributed (s1c, s2e). Extracted landmark sequences are treated as modifications, so they stay out of Git.
- Results may be published if no material or substantial portion of the data is included (s5). **No dataset video frames in slides or the report.**
- Personal data must be kept confidential and **destroyed with any backups immediately when the research is complete** (s3).
- The terms do not explicitly mention trained models. `app_model/` is committed locally as the brief requires, but publishing it (e.g. pushing to GitHub) is unresolved: **check with the faculty guide before pushing.** This is a reading of the licence, not legal advice.
- Raw data lives in `~/signo_data/asl_citizen/`, outside the repo and outside the Desktop.
- Cite: Desai et al. (2023), ASL Citizen.

## 23:16 Vocabulary
50 glosses fixed in `python/vocab_asl_citizen.txt` before any training. Not selected using any accuracy.

## 23:20-23:27 Download
`fetch_asl_citizen.py`: 1,535 of 1,535 selected videos, 0 failed, 859.2 MB on disk (about 1.9% of the archive), 431.5 s. Split: 731 train, 188 val, 616 test. About 15 / 4 / 12 per class, below the 15-30 training target, and not padded with duplicates.

## 23:21 Extraction benchmark
| Workers | Clips | Failures | Wall time | Rate |
|---|---|---|---|---|
| 1 | 8 | 0 | 21.3 s | 0.38 clips/s |
| 4 | 20 | 0 | 19.6 s | 1.02 clips/s |

MediaPipe is reliable in spawned worker processes on this Mac. No frame stride was needed.

## 23:23 TF.js conversion test (dummy BiLSTM)
Official converter worked with no fixes. `model.json` + 1 shard. In tfjs 4.22.0 under Node: input `[null,30,100]`, max absolute difference from Keras 3e-8.

## 23:27-23:43 Full extraction, 50 signs
1,529 of 1,535 clips extracted, 6 `no_hands`, 0 corrupt or missing, 6 workers, 997.9 s. Usable: 729 / 186 / 614 clips and 29 / 5 / 11 signers (fewer than the whole dataset because not all signers recorded each word).

## 23:44 BiLSTM, 50 signs, seed 0
Original architecture and settings, no tuning. 64,322 parameters. Stopped at epoch 42, kept epoch 27 (val loss 0.839). 26 s on CPU. Val top-1 82.3%, top-5 96.2%. Declared final, then test evaluated once: **top-1 76.7%, top-5 95.9%**, n=614.

## 23:45-00:19 Browser verification, 50 signs
- First attempt failed with `loadGraph` of undefined: test called `H.reset()` before `H.initialize()`. Test bug.
- Second attempt found no hands in 4 of 5 clips. Cause: Playwright's route does not support Range requests, so seeking stuck on the first frame. Test bug, fixed by loading each clip as a blob.
- Keras vs TF.js on same sequences: 50 of 50 same top-1 and top-3 order, max probability difference 7e-7.
- Full browser pipeline, 50 clips: 72% top-1, 92% top-5. Python on same clips: 76%. Agreement 48 of 50. Mean absolute feature difference 0.048 shoulder widths. No page errors.

## Before 00:03 Real webcam (user report)
User tried the 50-sign model on their own webcam and said some words were recognised correctly. Informal, with no counts. Not a measured accuracy.

## 00:03 User asked for about 1000 words and a bigger video
- Dataset videos sampled (300): about 95% are 640x480, matching the webcam request. Camera request left unchanged.
- 1,991 unnumbered glosses have enough data. 1000 chosen by `make_vocab_1000.py`: keep the 50 demo signs, then most common English words (wordfreq). Fixed before training.
- Full cost would be 30,583 videos, 17 GB, about 5.5 h extraction. **Test split cut to 4 clips per sign**, picked by hash of file name: 22,421 videos, 12.1 GB. All official train and val clips kept. Results must state this.
- Every-second-frame experiment (50 signs, val only): 79.5% vs 82.3% top-1, 96.2% top-5 both, 11 vs 4 no-hands failures on train+val, about 1.7x faster. Gap is about 5 clips on a single run, so inconclusive, but no evidence in favour. Kept every frame.
- UI: video fills the main area, word is a caption on the video, side panel slimmed. Second model slot `app_model_1000/`.
- Measured extraction speed with 8 workers while other jobs ran: 1.43 clips/s, so finish by about 04:15. Deadline about 07:15.

## 03:37-06:31 Three hours lost to a bug in my waiting script
Extraction finished at about 03:37, but the loop waiting for the download used `pgrep -f fetch_asl_citizen`, which matched its own command line, so it never exited. Noticed at 06:31 when the user asked whether it was done. Stopped the loop and built the dataset. The planned longer training was cut back to fit.

## 06:32 Dataset, 1,000 signs
22,372 of 22,421 clips extracted, 49 `no_hands`. 14,626 / 3,763 / 3,983 clips, 35 / 5 / 11 signers, 0 overlap. Each sign has 11-17 train, 2-6 val and 3-4 test clips.

## 06:33-06:49 Round 1, three sizes
Batch 64, patience 8, 15-minute limit, epoch kept by lowest val loss. Validation: 48 units 66.0% / 89.9%, 128 units 56.3% / 83.9%, 256 units 59.6% / 85.2% (stopped by the limit). The logs showed val accuracy still rising after val loss turned upwards, so this rule was keeping weak epochs.

## 06:44 User: 2 hours left, and a report still to write
New deadline about 08:45. Used 25 minutes of it for a second round.

## 06:46-07:12 Round 2
Added `--select val_accuracy`, `--reduce-lr` and `--max-minutes` to `train.py`. Patience 10, 25-minute limit. Dropout 0.5 for the two larger models. Validation: 48 units 69.5% / 90.2% (finished by early stopping), 128 units 68.7% / 89.3% and 256 units 66.9% / 88.1% (both stopped by the limit, and 256 units was still improving).

## 07:21 Selected model: `w1000_u48_acc`
Best val top-1 and top-5, smallest model, smallest train/val gap among round 2. Chosen on validation only. Caveat: the larger models weren't given as long, so this doesn't show they could never do better. Single seed. Validation threshold check: at 70%, a word is shown for 75% of clips and 81% of those are right, so the default stayed at 70%.

## 07:22 Test set, scored once
**Top-1 61.4%, top-5 84.4%**, n=3,983. Mean per-sign precision 65.0%, recall 61.5%. 198 signs at 100% recall and 45 at 0%, but most signs have only 4 clips. No tuning afterwards. Test top-1 is 8.1 points below validation.

## 07:23-07:40 Export and browser checks, 1,000 signs
- Export worked first time: `[null, 30, 100]`, 1,000 labels, 443 KB weights.
- Python: real test clip gave finite output of size 1,000 summing to 1.0, and the correct label.
- TF.js vs Keras on 1,000 test sequences: same top-1 and top-3 order on all, largest difference 1.5e-6.
- Video through browser MediaPipe, features and model, 60 clips: 65% top-1, 85% top-5. Python 65% on the same clips. Agreement 53 of 60. Mean feature difference 0.043. Repeated on 20 clips with the final page: 80% top-1, agreement 18 of 20.
- `model_slots.js`: starts on the 1,000-sign model, labels and outputs stay matched when switching, and a missing or corrupt model falls back to 50 signs. Test pattern shows "No hands in view". About 67 ms per prediction.
- Fake webcam, real app loop, 8 test clips, first pass: 6 right, 1 no word, 1 wrong; then 5 right, 3 no word, 0 wrong; then 5 right, 2 no word, 1 wrong. Coffee was read as "replace" twice. Results vary between runs because timing varies. Six of the 8 signs were picked for good test results, so this isn't an accuracy score.
- Teach, train and reload still work with both dataset models present. The test logged two 404 console errors, probably the missing favicon on its two page loads (not confirmed). A normal page load showed no failed requests.

## 07:30 Bug found by the fake webcam: words vanished too fast
The word was replaced by "No hands in view" as soon as hands left, sometimes after 0.1 s. Now it stays for 2 seconds, with the top 3 kept beside it. After the fix each word stayed on screen for about 2 to 3 seconds.

## Not done
Real webcam test of the 1,000-sign model, more seeds, GRU comparison, `demo_webcam.py`.

## 08:21-08:39 Audit and screening
Baseline on validation: left-dominant signers 46.7%, right-dominant 73.1%. Left-dominant share of clips: 17.0% train, 13.5% validation, 35.3% test. Reweighting validation to the test's mix gives 63.8%. No duplicates. Imbalance about 1.5 to 1. Details in `results/error_analysis.md`. Screening: `plus_plain128` 77.3%, `plus_deep64` 60.1%, `plus_deep128` 47.9%.

## 08:07 TF.js error caught early
A trial model using `Cropping1D` failed to load in TF.js, so the layer was replaced with convolution before any long training.

## 08:39-09:12 Controlled rounds, and a flaw
Mirroring: left-dominant 48.2% to 69.9%, right-dominant 74.3% to 72.5%, overall 70.8% to 72.2%. Wrist-relative hands 70.6%. **Flaw:** a fixed time budget let slow neighbouring jobs cost a run epochs (smoothing got 45 epochs, 256 units got 19, graph got 3). Later comparisons used fixed epochs.

## 09:18-10:00 Fixed-epoch rounds
No warping 70.1%. Half data 65.7%. Weight decay 70.7%. Smoothing 73.3%. `f_long_s0` (180 epochs, 38 minutes) 78.0%.

## 10:00-10:23 Selection and second test look, 1,000 signs
A timing command hung for about 15 minutes, so timeouts were added. `final_v2` = `f_long_s0` + `plus_plain128`: validation 79.4% / 95.0%, chosen before test. An earlier note rejected ensembles, based on a weaker partner and 4 passes, and this was reversed on the new evidence. **Test, logged second look: 73.4% / 92.0%**, n=3,983. Macro precision 75.4%, recall 73.4%, F1 72.5%.

## 10:43 User asked for above 80%, 45 more minutes, up to 10 GB
Decided against downloading. About 45 minutes allows roughly 2,000 to 2,700 new clips, with no time to retrain, for an estimated 1 to 2 points. Trained two more 1,000-sign models and retrained the 50-sign model.

## 11:01 50-sign model
`final50_v2` = `s50_u128_s0` + `s50_u64_s1`: validation 95.2% / 98.4% (186 clips, several combinations within one clip). **Test, logged second look: 91.2% / 97.6%**, n=614. Macro precision 91.7%, recall 91.3%, F1 91.1%. Browser: TF.js matched Keras on 50 of 50, and 50 videos scored 90% top-1 (Python 86%), agreeing on 46.

## 11:20 Three-model 1,000-sign option rejected
Validation 80.3% / 95.5%, but about 291 ms mean, over the 220 ms limit. No third test look. Validation at 80% wouldn't have meant test at 80%.

## 10:37-11:30 Browser checks and fixes
- Fake webcam run 1 failed: camera wasn't ready within 30 seconds. The app waited for all models to load, taking 34 seconds on a busy machine. It now starts on the first model that loads (about 7 seconds idle, plus about 4 for the camera).
- MediaPipe and the model share the main thread: 4 to 11 camera frames per second on 1,000 signs, 12.5 on 50 signs. Prediction gaps had a median of 225 to 266 ms.
- Model timing varied from 107 to 132 ms between sessions on 1,000 signs. One 50-sign reading of 207 ms was taken on a busy machine.
- All 23 tests pass. The 2 console 404s in the Teach check are unexplained, although a plain app load showed no failed requests.

## Storage used
About 1 GB of the 10 GB budget. No new videos.

## Not done
Real webcam measurement of version 2, multiple-seed statistics, separate ablations of geometry, jitter and hand dropout, GRU, Transformer, and fair training of larger or graph models.

## Citation checked
The ASL Citizen citation was checked against arXiv:2304.05934 and the NeurIPS 2023 proceedings. Title, first author and year are correct.

## 08:04 onward: optimisation sprint
User asked for better accuracy (target 85%), then gave a full optimisation brief. Limits: 10 GB extra storage, frozen 30 x 100 input, no test-based choices, stop by about 10:00.

**Storage budget (hard limit 10 GB, 122 GB free at 08:21).** Existing: videos 12 GB, feature cache 217 MB, dataset JSON 468 MB, models 106 MB. Allocation: new videos 5 GB at most (none yet), new features and caches 2 GB (`ds1000.npz` 261 MB so far), checkpoints 1 GB, reserve 2 GB.

**Corrections to the brief.** The 121 ms latency came from a trial of a larger model. The deployed 48-unit model measured about 67 ms. The test set isn't untouched: it was scored once at 07:22, so the final evaluation will be a logged second look.

**Rule for test data.** No new predictions on test clips until selection is final. Data-quality checks used test clips only for their flags and coordinates, with no labels or predictions.
