# Diagnosis of the 1,000-sign model

Numbers come from `research/audit.json` and `research/diagnose_raw.json`, made by `python/audit_dataset.py` and `python/diagnose_raw.py`. Model predictions are validation only. Test clips were counted but not predicted.

## Baseline

The deployed model is `final_v2`, two BiLSTMs averaged, in `app_model_1000_v2`.

| Measure | Validation (3,763 clips) |
|---|---:|
| Top-1 | 79.38% |
| Top-5 | 95.00% |
| Macro precision | 82.27% |
| Macro recall | 79.89% |
| Macro F1 | 78.87% |
| Signs with recall of 50% or less | 188 |
| Signs with recall of 0 | 16 |
| Signs with recall of 100% | 512 |

Running the saved model again matches the saved predictions to within 1e-14. The brief's 61.4% figure belongs to version 1, which was replaced before this run.

## What is fine

| Check | Result |
|---|---|
| Signers across splits | No overlap between any two splits (35 train, 5 val, 11 test in this subset) |
| Duplicate or repeated clips | 0 duplicate video IDs, 0 clips with identical features |
| Bad numbers | 0 non-finite values. Coordinates are mostly -1.0 to 1.2, largest magnitude 2.8 |
| Labels | Class order equals sorted training labels, and the saved model's `labels.json` matches. Each experiment repeats this check |
| Feature pipeline | Rebuilding features from raw landmarks matches stored data within rounding (largest difference 0.0007) on 400 of 400 clips |
| Class balance | 11 to 17 training clips per sign, a ratio of 1.5. Recall rises gently with count, from about 74% at 13 clips to 85% at 17 |
| Dropped clips | 49 of 22,421 had too few hand frames: 18 train, 14 val, 17 test |

## Main findings

1. **Lack of data per sign, and memorising signers.** Clean-clip training accuracy on the two deployed members is 97.5% and 95.4%, against 79.4% validation, so the real gap is about 18 points. Earlier reports quote about 6 points, but that compared augmented training clips with clean validation clips, so **that claim is too generous and should be corrected**. Halving the training clips cost about 6.5 points at similar length, which also points to data. The official split has no more clips for these signs.
2. **Training length.** The BiLSTM rose from 72.2% at 65 epochs to 77.3% at 125 and 78.0% at 180, with best epochs near the end. Earlier convolution models stopped at 34 to 72 epochs on a time budget, so they never got a fair test.
3. **Lost hand tracking.** Top-1 falls with the share of the clip without any hand: 84.5% at 5 to 10%, 80.7% at 10 to 20%, 76.0% at 20 to 30%, 62.4% above 30%. This is a correlation, but gaps can be filled before resampling.
4. **Signer differences.** P21, P39 and P12 score 83 to 84%, P26 72% and P5 66%. Tracking explains part of P26: it has the most gaps, the fewest two-hand frames and the most dropped clips of any signer. P5 is not explained by tracking.
5. **Similar signs.** Top-5 is 15.6 points above top-1. The top 100 confused pairs hold 246 of 776 errors, including assist/assistant, star/stars, create/make and holiday/vacation. Some pairs may be very close or the same sign, which sets a ceiling on this vocabulary.
6. **Camera shape.** The features do not correct for video shape, so 16:9 video stretches heights by a third relative to 4:3. All of test signer P42 (390 of 4,000 used test clips) is 16:9, with only 51 of 14,644 such clips in training. Simulating this on validation drops top-1 from 79.38% to 76.53%. The browser asks for 640 x 480 but does not check what it gets, so a camera that returns 16:9 would hit the same problem live.

## Ideas the evidence weakens

- **More than 30 frames.** Median hand span is 38 frames, and 30-frame sampling keeps 95% of fingertip movement at the median (75% at the 10th percentile). There is some effect, 65% top-1 when under 60% of movement is kept against 82% above 90%, but few clips are affected. Lower priority.
- **Class weighting.** Almost no imbalance to correct, so dropped.
- **Face position as the main fix.** 81% of signs bring the fingertips within half a shoulder width of the mouth, and 72% of errors have both signs in that group, close to the 66% expected by chance. Accuracy barely changes with distance to the mouth. Face points are still worth one test because published solutions use them, but this data does not predict a large gain.
- **Low frame rate.** Keeping every 2nd frame costs 0.5 points and every 3rd costs 2.2 points on validation, so slow live frame rates may cost a little. This used the stored 30 frames, so it is only an estimate.

## Calibration

Confidence is reliable on validation: calibration error is 0.019, median confidence 0.96 when right and 0.53 when wrong. Only 13% of wrong answers score above 0.8, so raising the threshold would mostly reject answers rather than fix them.
