# Optimisation report

Every number comes from a saved result file. `results/optimization_comparison.csv` lists all 19 completed configurations. Only the deployed models were timed in the browser, so other runs have no latency value.

## Final results

| Model | Val top-1 | Val top-5 | Test top-1 | Test top-5 | Mean precision | Mean recall | Mean F1 | Weighted F1 | Browser mean / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1,000 signs, version 2** | 79.4% | 95.0% | **73.4%** | **92.0%** | 75.4% | 73.4% | 72.5% | 72.5% | 107-132 / 111-164 ms |
| 1,000 signs, version 1 | 69.5% | 90.2% | 61.4% | 84.4% | 65.0% | 61.5% | 60.5% | Not recorded | 67 / 71 ms |
| **50 signs, version 2** | 95.2% | 98.4% | **91.2%** | **97.6%** | 91.7% | 91.3% | 91.1% | 91.1% | 106-109 / 109-114 ms |
| 50 signs, version 1 | 82.3% | 96.2% | 76.7% | 95.9% | 79.3% | 76.7% | 76.7% | Not recorded | 66 / 70 ms |

Random guessing scores 0.1% top-1 and 0.5% top-5 on 1,000 signs, and 2% and 10% on 50 signs. The 1,000-sign and 50-sign models use different test sets.

**The 85% and 80% targets were not reached on 1,000 signs.** The 50-sign model exceeds both, on a smaller and easier task.

Each test split has been evaluated twice, once per version. Both evaluations are logged in `test_log.json` in `results_1000_v2/` and `results_50_v2/`. Nothing changed after either score was seen. The 1,000-sign model's test split is capped at 4 clips per sign.

## Why the final model is better

1. **It stopped memorising.** New random changes every epoch mean it almost never sees the same input twice. The train-to-validation gap fell from about 23 points to about 6 for the long-trained member.
2. **It handles left-dominant signers.** Validation accuracy for them rose from 46.7% to 76.6%.
3. **It trained for longer, with a falling learning rate,** which let it keep improving instead of overfitting.
4. **Two models average out some of each other's mistakes.**

## Experiments

Validation only. Most used one seed. With 3,763 clips, differences of 1 to 2 points are close to noise.

| Group | Run | What changed | Top-1 | Top-5 | Macro F1 | Left-dominant | Train accuracy | Epochs | Minutes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `w1000_u48_acc` | Version 1, 48 units | 69.5% | 90.2% | 69.1% | 46.7% | 92.3% | 39 | 14.8 |
| Baseline | `plus_plain128` | New augmentation, 128 units | 77.3% | 94.2% | 76.6% | 73.6% | 79.1% | 125 | 30.2 |
| Augmentation | `a_control` | Control | 72.2% | 92.5% | 71.1% | 69.9% | 66.7% | 65 | 15.1 |
| Augmentation | `a_no_mirror` | No mirroring | 70.8% | 91.2% | 70.1% | 48.2% | 73.2% | 65 | 15.0 |
| Augmentation | `c_no_warp` | No time warping | 70.1% | 90.8% | 69.2% | 64.6% | 69.7% | 60 | 16.0 |
| Normalisation | `a_hand_local` | Wrist-relative hands | 70.6% | 92.1% | 69.7% | 68.9% | 64.0% | 64 | 15.0 |
| Training | `c_smoothing60` | Label smoothing 0.1 | 73.3% | 93.3% | 72.6% | 68.9% | 67.8% | 60 | 11.8 |
| Training | `c_weight_decay` | AdamW 1e-4 | 70.7% | 92.0% | 69.6% | 66.9% | 65.6% | 60 | 11.7 |
| Training | `b_smoothing` | Label smoothing, time budget | 70.2% | 92.1% | 69.3% | 68.1% | 61.2% | 45 | 15.2 |
| Training | `f_long_s0` | 180 epochs | 78.0% | 94.5% | 77.5% | 74.6% | 83.9% | 180 | 38.2 |
| Data need | `c_half_data` | Half the training clips | 65.7% | 88.5% | 64.7% | 61.2% | 76.3% | 120 | 17.3 |
| Architecture | `plus_deep64` | CNN + two BiLSTMs, 64 | 60.1% | 86.2% | 58.8% | 58.7% | 52.3% | 72 | 30.3 |
| Architecture | `plus_deep128` | CNN + two BiLSTMs, 128 | 47.9% | 77.8% | 45.8% | 43.1% | 39.1% | 34 | 30.3 |
| Architecture | `b_units256` | 256 units | 56.0% | 84.0% | 54.1% | 54.3% | 44.6% | 19 | 15.3 |
| Architecture | `b_graph` | Lightweight graph model | 2.2% | 8.6% | 0.8% | 1.8% | 1.3% | 3 | 17.0 |
| Ensemble | `g_s1` | New seed | 77.3% | 94.6% | 76.7% | 74.0% | 79.1% | 130 | 30.7 |
| Ensemble | `g_s2_smooth` | New seed, label smoothing | 78.1% | 94.7% | 77.2% | 76.0% | 79.7% | 130 | 29.7 |
| Ensemble | **`final_v2`** | **`f_long_s0` + `plus_plain128`, deployed** | **79.4%** | **95.0%** | **78.9%** | **76.6%** | n/a | n/a | n/a |
| Ensemble | `final_v3` | Three models, rejected at 291 ms | 80.3% | 95.5% | 79.7% | 78.5% | n/a | n/a | n/a |

Best of each measure among deployable models: `final_v2` for top-1, top-5, macro F1 and macro precision (82.3%). Fastest and smallest of the deployed 1,000-sign models: version 1 (67 ms, 110,872 parameters).

### 50-sign runs (186 validation clips)

| Run | Units | Top-1 | Top-5 | Train accuracy | Epochs |
|---|---:|---:|---:|---:|---:|
| `s50_u64_s0` | 64 | 92.5% | 96.8% | 96.9% | 400 |
| `s50_u128_s0` | 128 | 94.6% | 97.8% | 98.0% | 400 |
| `s50_u64_s1` | 64 | 94.1% | 96.8% | 98.9% | 800 |
| `s50_u128_s1` | 128 | 94.6% | 97.8% | 99.9% | 800 |
| **`final50_v2`: 128 s0 + 64 s1** | | **95.2%** | **98.4%** | n/a | n/a |

Several combinations were within one clip of each other, so this choice is not strongly separated from the others.

## Decisions

| Technique | Decision | Why |
|---|---|---|
| Fresh augmentation | Keep | Largest gain |
| Mirroring | Keep | Left-dominant signers 48.2% to 69.9%. Right-dominant 74.3% to 72.5% |
| Time warping | Keep | About 2 points |
| Longer training, cosine learning rate | Keep | 72.2% at about 60 epochs, 78.0% at 180 |
| Two-model average | Keep | 79.4%, 126 ms |
| Three-model average | Reject | 80.3%, but 291 ms is over the 220 ms limit |
| Mirror averaging at prediction time | Reject | No gain on the long model, and it doubles latency |
| Wrist-relative hands | Reject | No gain within the budget |
| Weight decay | Reject | No gain |
| Label smoothing | Not adopted | About 1 point, within noise |
| Class weights, focal loss, balanced sampling | Not tested | Imbalance is only about 1.5 to 1 |
| GRU, Transformer | Not tested | The evidence pointed elsewhere first |

## Not a fair test

- **Larger and convolutional models** were still improving when their time ran out.
- **The graph model** took about 5.5 minutes per epoch on this CPU, so it never really trained.
- **`b_smoothing` and round 1-style runs** used a fixed time budget, so slow neighbouring jobs cost them epochs. Later comparisons used fixed epochs.
- **Individual augmentation ablations** covered mirroring and time warping. Geometry, jitter and hand dropout weren't tested separately.
- **`final_v2` members** differ in training budget and share seed 0.

## Additional data

No videos were downloaded. Half the clips scored about 6.5 points lower, so more data should help. However, processing WLASL clips within the time would have left no time to retrain, and we estimated a gain of only 1 to 2 points. It's the top future step.

## Storage

Budget 10 GB. Used about 1 GB: `model_out` 710 MB, array caches 279 MB, test fixtures 43 MB. No new videos. Trial exports were deleted. 119 GB free.

## Latency

Measured in Chrome with WebGL with nothing else running, except where noted.

| Model | Passes | Mean | p95 |
|---|---:|---:|---:|
| 1,000 signs, version 1 | 1 | 67 ms | 71 ms |
| 1,000 signs, one 128-unit model | 1 | 63 ms | 66 ms |
| **1,000 signs, version 2** | 2 | **107-132 ms** | **111-164 ms** |
| One model + mirror averaging, under training load | 2 | 199 ms | 213 ms |
| Three models (rejected) | 3 | About 291 ms | About 321 ms |
| 50 signs, version 2 | 2 | 106-109 ms | 109-114 ms |

Ranges come from repeated sessions. One 50-sign reading of 207 ms was taken while the machine was busy.

**In the live app loop, prediction doesn't run alone.** MediaPipe and the model share the browser's main thread. With dataset video as a pretend camera:

| Model | Runs | Frames processed per second | Median gap between predictions | p95 gap |
|---|---:|---:|---:|---:|
| 1,000 signs, version 2 | 4 | 4 to 11 | 225 to 266 ms | 252 to 331 ms |
| 50 signs, version 2 | 1 | 12.5 | 236 ms | 328 ms |

So each prediction meets the 220 ms limit, but the full live cycle sometimes takes longer, and the 1,000-sign model can reduce the camera frame rate. A real webcam hasn't been measured.

## Simulated-camera results

Eight test clips played as a pretend camera. Six signs were picked for good version 1 test results, so **this checks the live loop and isn't an accuracy score**.

| Model | Run | Right | No word | Wrong words shown |
|---|---|---:|---:|---:|
| 1,000 signs | 2 | 6 | 2 | 0 |
| 1,000 signs | 3 | 7 | 1 | 0 |
| 1,000 signs | 4 | 5 | 2 | 1 (*boat*) |
| 1,000 signs | 5 | 6 | 2 | 0 |
| 50 signs | 1 | 8 | 0 | 1 extra (*like*) |

Run 1 of the 1,000-sign model failed because the camera wasn't ready within 30 seconds on a busy machine. The app waited for every model to load before it could start. It now starts as soon as the first model is ready, and an idle start measured about 7 seconds for the model and about 4 more for the camera. In the 50-sign run's partial second pass, *book* was once read as *school*.
