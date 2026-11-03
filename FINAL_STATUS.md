# Signo: final status (version 4)

**The default 1,000-sign model is now version 4, at 83.2% top-1 on test clips never used before. It has not been measured on a real webcam, so try it for 10 minutes before relying on it.**

## Start it

```bash
cd ~/Desktop/signo
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

If it says the address is already in use, the server is already running. Open http://localhost:8000 in Chrome and press **Cmd+Shift+R** once. You need internet, because TensorFlow.js and MediaPipe load from a CDN.

Choose **Large ASL model (1000 signs)** in the dropdown if it isn't already selected. **Previous large ASL model** is version 2, and **Accurate ASL model (50 signs)** is unchanged.

## Results

| Model | Unseen test clips (8,134): top-1 | top-5 | Macro F1 | Earlier test clips (3,983): top-1 | Validation top-1 | Browser time |
|---|---:|---:|---:|---:|---:|---:|
| **Version 4** | **83.2%** (82.4 to 84.0) | **96.4%** | **82.8%** | 83.8% | 85.3%, before adding validation clips to training | 68 ms |
| Version 2 | 74.1% (73.2 to 75.1) | 92.6% | 73.6% | 73.4% | 79.4% | 116 ms |

On the same unseen clips, version 4 is 9.1 points better (95% interval 8.2 to 9.9). Random guessing scores 0.1%. Test signers never appear in training.

## What was checked

| Check | Result |
|---|---|
| Model choice | Validation only, by a rule fixed before the outcome. The final-training plan was written down before any final model existed |
| Test evaluation | Run once by `python/final_eval.py`, which refuses to run again |
| TensorFlow.js export | Input `[null, 30, 100]`, 1,000 outputs, 7.2 MB |
| Chrome vs Python on the same numbers | 300 of 300 same top-1 and top-3 order, largest difference 0.0000007 |
| Test videos through the whole browser pipeline | 40 clips: 82.5% top-1 in the browser, 85% in Python on the same clips, 95% agreement, no page errors |
| Switching and fallback | If the new model is missing or corrupt, the app opens the previous one |
| Existing tests | All 23 pass |
| Storage | See `research/STORAGE_LEDGER.md` |
| Git | Committed to `feature/asl-citizen-model` and pushed to the `johnsoni7` remote. `main` still holds the earlier version |

## Not done

- **Real webcam accuracy is unmeasured.** All scores come from dataset video.
- **`PROJECT_REPORT_v2.pdf` is out of date,** and still says the training gap is about 6 points. The real figure was about 18. Use `PROJECT_REPORT_v4.pdf`.
- Pretraining on the other 1,731 signs was not tried.

## Where things are

| What | Where |
|---|---|
| Project report | `PROJECT_REPORT_v4.pdf` |
| Final numbers and discussion | `results/FINAL_RESULTS.md` |
| Notebook and figures | `SIGNO_SUBMISSION/` |
| What was tried, and why | `research/DECISIONS.md`, `research/RESEARCH_LOG.md`, `research/DIAGNOSIS.md`, `research/LITERATURE_REVIEW.md` |
| Every experiment | `experiments/leaderboard.csv` |
| Browser model | `app_model_1000_v4/` |
| Keras models, not in git | `model_out/final_v4`, `model_out/final_v4_val`, `model_out/fit_exp_022`, `model_out/fit_exp_026` |

## Licence

The licence requires deleting ASL Citizen data when the research is complete. That covers `~/signo_data` and `~/signo_backup_2026-11-02`. Both were kept, because you may still need them.
