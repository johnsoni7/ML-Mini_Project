# Final results

All selection used validation clips. Test clips were scored once, by `python/final_eval.py`, after every choice was fixed.

## Model comparison (validation, same 3,763 clips)

| Model | Validation top-1 | Validation top-5 | Macro F1 | Parameters |
|---|---:|---:|---:|---:|
| Baseline: `final_v2`, deployed before this work | 79.4% | 95.0% | 78.9% | 792,784 |
| Best single run: `exp_022` full_tcn_fair | 84.3% | 96.6% | 83.9% | 1,322,344 |
| Selected ensemble, trained on training clips only | 85.3% | 97.3% | 84.9% | 1,851,136 |

The deployed model is the same ensemble retrained on training plus validation clips, so it has no validation score of its own.

Members: `exp_022` full_tcn_fair (temperature 0.90), `exp_026` always_velocity (temperature 0.84).

## Test results: Unseen test clips (8,134 clips, 11 signers)

These official test clips had never been used. **This is the unbiased score.**

| Model | Top-1 | 95% interval | Top-5 | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Final (training + validation)** | 83.2% | 82.4% to 84.0% | 96.4% | 84.4% | 83.2% | 82.8% | 82.9% |
| Same ensemble, training clips only | 81.1% | 80.2% to 82.0% | 95.6% | 82.2% | 81.0% | 80.6% | 80.8% |
| Baseline `final_v2` | 74.1% | 73.2% to 75.1% | 92.6% | 75.4% | 74.2% | 73.6% | 73.7% |

Final minus baseline on the same clips: +9.1 points (95% interval +8.2 to +9.9).

## Test results: Earlier test clips (3,983 clips, 11 signers)

These clips were scored in versions 1 and 2, so this set is not untouched. It is here to compare with earlier reports.

| Model | Top-1 | 95% interval | Top-5 | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Final (training + validation)** | 83.8% | 82.7% to 84.9% | 96.6% | 85.2% | 83.8% | 83.1% | 83.1% |
| Same ensemble, training clips only | 81.8% | 80.5% to 83.0% | 95.6% | 84.1% | 81.8% | 81.3% | 81.3% |
| Baseline `final_v2` | 73.4% | 72.0% to 74.7% | 92.0% | 75.4% | 73.4% | 72.5% | 72.5% |

Final minus baseline on the same clips: +10.4 points (95% interval +9.2 to +11.6).

## Best and worst signs (unseen test clips)

| Best 10 | F1 | Worst 10 | F1 |
|---|---:|---|---:|
| above | 100.0% | equal | 0.0% |
| important | 100.0% | cost | 13.3% |
| impact | 100.0% | assistant | 16.7% |
| idea | 100.0% | even | 16.7% |
| ice | 100.0% | price | 20.0% |
| hundred | 100.0% | earth | 25.0% |
| high | 100.0% | planet | 25.0% |
| improve | 100.0% | way | 25.0% |
| health | 100.0% | few | 28.6% |
| hate | 100.0% | rescue | 28.6% |

Signs with recall of 50% or less: 88 (baseline 171). The full list is in `results/final_test/final_fresh_per_class.csv`.

## Generalisation gap

For the best single run, clean training accuracy is 100.0% against validation 84.3%, a gap of 15.7 points. The earlier deployed members had a gap of about 18 points. The model still memorises to a degree.

## Browser speed

| Model | Mean ms per prediction |
|---|---:|
| Deployed before this work (final_v2) | 116 |
| Final model (final_v4) | 68 |

## Discussion

**What improved.** On never-used test clips, top-1 rose from 74.1% to 83.2%, top-5 from 92.6% to 96.4% and macro F1 from 73.6% to 82.8%. On the same clips the gain is 9.1 points (95% interval 8.2 to 9.9). Signs with recall of 50% or less fell from 171 to 88. All 11 test signers improved, and the lowest signer rose from 66.7% to 76.6%. The model is also faster in the browser, at 68 ms against 116 ms.

**Which experiment gave the largest gain.** The temporal convolution network, at 84.32% on validation, 6.3 points above the earlier single BiLSTM (95% interval 5.1 to 7.5). Blending clips in every batch (mixup) was next, adding 4.1 points to the same BiLSTM (interval 3.0 to 5.3). Combining two models added 1.0 point on validation.

**Whether another architecture beat the BiLSTM.** Yes, narrowly at equal effort. The TCN beat the best BiLSTM, which ran 240 epochs, by 1.2 points (interval 0.05 to 2.3). The earlier conclusion that convolution does not work here came from a setup fault and runs cut short, not from the architecture. The chosen pair is one TCN and one BiLSTM.

**Whether more data helped.** No new clips exist for these signs. Adding the 3,763 validation clips to the final training, as planned in advance, added 2.1 points on the unseen clips (interval 1.5 to 2.7), which supports data being the main limit. Pretraining on the other 1,731 signs was not tried.

**Whether augmentation helped.** Mixup did. Hand-size and arm-length variation did not: on all clips it scored 79.67% against 80.49% without it. Earlier work had already shown that mirroring and time warping help.

**What did not help (5,000-clip tests against 55.51%, unless stated).**

| Idea | Result | 95% interval |
|---|---:|---:|
| Face points | -6.8 points | -8.2 to -5.3 |
| Face points and bridged hand gaps | -9.4 | -10.9 to -7.9 |
| Frame embedding | -12.1 | -13.7 to -10.6 |
| 45 frames | -2.7 | -4.1 to -1.2 |
| Bridging hand gaps | +0.7 | -0.6 to +2.1 |
| Blended frame sampling | +0.1 | -1.3 to +1.4 |
| Motion input, on all clips and with mixup | -0.9 against the same run without it | -1.9 to +0.2 |
| Label smoothing with mixup, on all clips | 81.19% against 82.12% | |

The face result is the most surprising, since top Kaggle solutions use face points. One possible cause is that face shape lets the model recognise the signer, as the gap widened, but this was not tested. The frame embedding result also disagrees with a published ablation, and it was tested only on 5,000 clips with a BiLSTM, so it is not a general verdict. The BiLSTM with motion input was chosen as the second ensemble member because it added the most when combined, not because motion input helps.

**Whether the model is still overfitting.** Yes. The TCN scores 99.97% on clean training clips against 84.32% on validation, a gap of 15.7 points. Earlier members had a gap of about 18 points, so this has improved only somewhat.

**Whether signer-independent generalisation improved.** Yes, for every test signer, but signers still range from 76.6% to 92.0%. The 16:9 signer's score rose from 78.6% to 87.6%. Correcting for video shape probably contributes, but the model also changed, so the two effects were not separated.

**Validation against test.** The training-only ensemble scored 85.33% on validation and 81.09% on the unseen test clips. Part of that 4.2-point drop is likely from many comparisons on the same validation clips, and part from different signers.

**What remains the main bottleneck.** Too few clips per sign and too few signers, shown by the training gap and by the rise from adding validation clips. The next limit is similar signs: top-5 is 13.2 points above top-1, and the top 100 confused pairs hold 34% of errors, including earth and planet, assist and assistant, experience and experienced, register and rule. Some of these may be very close or the same sign. Real webcam accuracy has still not been measured.

**About the test sets.** The 3,983 earlier clips have now been scored in version 1, version 2 and here, so they are not untouched. The 8,134 clips were first scored in this one run, after all choices were fixed, so that is the score to quote. Both sets come from the same 11 test signers.

