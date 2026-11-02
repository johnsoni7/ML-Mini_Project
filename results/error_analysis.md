# Error analysis: why the model makes mistakes

All model errors here come from **validation** clips, except where a line says "test". Test clips were used for model scoring only in the two logged evaluations. Checks of flags and coordinates on test clips used no labels or predictions. No clip was removed from any split.

"Baseline" is the 48-unit model scoring 61.4% on test. "Final" is the two-BiLSTM model scoring 73.4% on test.

## The three highest-priority problems

1. **Left-dominant signers were recognised far less well.** The baseline scored 46.7% on them and 73.1% on right-dominant signers. Fixed by mirroring in training: the final model scores 76.6% and 79.8%.
2. **The baseline memorised its training clips.** It reached 92.3% on training clips and 69.5% on validation. Fixed by making new random changes to clips every epoch: the final runs' gap is about 6 points or less.
3. **There are too few real clips per sign, and some signs look alike.** Training on half the clips scored about 6.5 points lower. This is not fixed. More real data is the top future step.

## Which signer each clip comes from matters

We call a signer "left-dominant" if their left hand is visible more often than their right. This is our inference from the hand flags, because the dataset doesn't label handedness.

| Split | Signers | Left-dominant signers | Share of clips from them |
|---|---:|---:|---:|
| Train | 35 | 7 | 17.0% |
| Validation | 5 | 2 | 13.5% |
| Test | 11 | 4 | 35.3% |

| Validation signer | Clips | Hand | Baseline top-1 | Final top-1 |
|---|---:|---|---:|---:|
| P12 | 296 | Left | 49.3% | 84.1% |
| P5 | 212 | Left | 42.9% | 66.0% |
| P21 | 1,076 | Right | 76.5% | 83.5% |
| P26 | 1,093 | Right | 64.1% | 72.3% |
| P39 | 1,086 | Right | 78.8% | 83.8% |

## The gap between validation and test

| Model | Validation | Test | Gap | Validation reweighted to the test's hand mix |
|---|---:|---:|---:|---:|
| Baseline | 69.5% | 61.4% | 8.1 | 63.8% |
| Final | 79.4% | 73.4% | 6.0 | 78.7% |

**For the baseline,** reweighting to the test's 35% left-dominant share predicted 63.8%, close to the actual 61.4%. So handedness probably explains most of that gap. This is an inference: we didn't score test clips by signer.

**For the final model,** handedness no longer explains the gap, because reweighting predicts 78.7% against an actual 73.4%. The likely remaining causes are:

- **Selection effects:** we chose the best epoch and the best of many runs using validation, which makes that score optimistic.
- **Different people:** the single-signer range on validation is already wide, from 66.0% to 84.1%, and validation has only 5 signers.

We ruled these out as main causes:

| Possible cause | Evidence |
|---|---|
| Different class mix | All 1,000 signs are in every split, with 11-17 train, 2-6 validation and 3-4 test clips per sign |
| Leaked or repeated clips | 0 exact duplicate clips, 0 repeated video IDs, 0 shared signers |
| Broken shoulder normalisation | Shoulder distance is 1.000 in every split, as designed |
| Extreme coordinates | No clip in any split has a coordinate above 5 shoulder widths |
| Python and browser mismatch | Tests pass, and TF.js matches Keras on real sequences |

Label noise can't be ruled out, because we haven't reviewed labels by watching videos.

## Data quality

| Measure | Train | Validation | Test |
|---|---:|---:|---:|
| Average frames with a hand visible | 84.0% | 82.7% | 82.5% |
| Clips with a hand in under half their frames | 1.09% | 3.27% | 1.86% |
| Clips with almost no movement | 0% | 0% | 0% |

Poor hand detection does hurt, but it affects few clips:

| Frames with a hand visible (validation) | Clips | Baseline top-1 | Final top-1 |
|---|---:|---:|---:|
| Under 50% | 123 | 44.7% | 52.8% |
| 50% to 80% | 604 | 61.4% | 72.7% |
| Over 80% | 3,036 | 72.2% | 81.8% |

## Does the number of clips per sign matter?

Class imbalance is mild, at 11 to 17 training clips per sign, or about 1.5 to 1. So we didn't try class weights or focal loss, because there's little imbalance for them to fix.

| Training clips for the sign | Validation clips | Baseline top-1 | Final top-1 |
|---|---:|---:|---:|
| 11 to 13 | 550 | 62.7% | 72.7% |
| 14 | 1,242 | 69.2% | 80.2% |
| 15 | 1,346 | 70.6% | 79.9% |
| 16 to 17 | 625 | 73.9% | 82.4% |

Signs with more clips do a little better, which fits the half-data result.

## Are errors concentrated?

Not strongly. The final model's 776 validation errors fall across 488 signs, and the 100 worst signs account for 35.6% of them. No small set of signs is responsible. When the top guess is wrong, the right sign is in the top 5 for 75.8% of those clips.

**Most confused pairs (final model, validation clips):**

| True sign | Read as | Clips |
|---|---|---:|
| assist | assistant | 5 |
| create | make | 4 |
| safe | rescue | 4 |
| true | really | 3 |
| star | stars | 3 |
| holiday | vacation | 3 |
| obvious | clear | 3 |
| rule | register | 3 |

Many of these pairs are closely related in meaning. Some may be the same or very similar signs under different English words, which would set a limit on any model's score. Dafnis et al. (2022) corrected WLASL's labels for this reason. We didn't merge any labels, because that would change the exam.

**Signs at 0% recall on validation with at least 4 clips:** create, safe, shot, specific, power, script, assist, bother, bright. Each of these rests on only a few clips.

## Test-set results by sign

These come from the final logged evaluation and were not used to make any choice.

- 368 signs scored 100% recall, and 29 scored 0%. Each test clip usually moves a sign's recall by 25%, so these are rough.
- Compared with the baseline, 446 signs improved and 113 got worse.
- Most confused on test: *quiet* as *calm*, *lesson* as *list*, *internet* as *network*, *but* as *different*.
