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
