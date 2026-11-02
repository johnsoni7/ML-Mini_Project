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

## Citation to check
The ASL Citizen citation in the docs was written from memory, not from a downloaded file. Check it against the project page before submission.
