"""Build SIGNO_SUBMISSION/Signo_Research.ipynb. Every number comes from saved result files when the notebook runs. No dataset clips or landmarks are included."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Signo: recognising 1,000 American Sign Language signs from landmarks

**Students:** John Soni Thomas (CSE-A, Roll No 40, Reg No 240905338), Mokshagna Y (CSE-A, Roll No 42, Reg No 240905354)

This notebook records the research behind Signo's 1,000-sign model. It reads saved results, and does not train anything or read any test clips when it runs. Figures are saved in `figures/`."""))
C.append(code("""import csv, json, os
import pandas as pd
from IPython.display import Image, Markdown, display
R = '../'
load = lambda p: json.load(open(R + p))
audit, raw, base = load('research/audit.json'), load('research/diagnose_raw.json'), load('research/baseline.json')
board = pd.read_csv(R + 'experiments/leaderboard.csv')
choice, rep = load('research/final_choice.json'), load('results/final_test/report.json')
M = rep['models']
show = lambda f: display(Image(filename='figures/' + f))
pc = lambda v: f'{v * 100:.1f}%'
print('Experiments recorded:', len(board), '| Models in the final test:', list(M))"""))
C.append(md("""## 1. Introduction

Sign recognition lets a computer read signing from an ordinary camera. Signo runs in a web browser: MediaPipe finds the body and hands in each frame, and a neural network names the sign. The task here is **isolated** recognition, one sign per clip, across 1,000 signs, on people the model has never seen.

The question for this work: why does the existing model make the mistakes it does, and what real improvement is possible?"""))
C.append(md("""## 2. Research background

| Source | Finding used here |
|---|---|
| Desai et al., ASL Citizen, NeurIPS 2023 | The dataset and its signer-independent protocol. I3D reached 63.10% top-1 and ST-GCN 59.52% on all 2,731 signs. **This is a different vocabulary, so these numbers are not comparable with ours** |
| Vandendriessche et al., 2025, one-shot ISLR | A pose-based PoseFormer reached Rec@1 of about 0.75 on ASL Citizen. Its ablation credits a frame embedding with +0.0278 |
| Hoyeol Sohn, Kaggle ISLR 1st place | 1D convolutions with attention, on lips, hands, nose and eyes, with motion channels |
| Kaggle ISLR 2nd place write-up | Motion, distances, heavy augmentation including mixup, 100 to 210 epochs |
| dos Santos et al., 2025, LIBRAS | A chosen landmark subset beat using all points, and filling gaps helped |
| Bai et al., 2018 | Simple temporal convolutions match or beat recurrent networks on many tasks |

Full notes, including sources that could not be read, are in `research/LITERATURE_REVIEW.md`."""))
C.append(md("## 3. Dataset\n\nASL Citizen is Microsoft's crowdsourced dataset, used under its research licence. No clips or landmarks are included here."))
C.append(code("""display(pd.DataFrame([{'Split': s, 'Clips': audit['clips'][s], 'Signers': len(audit['signers'][s]), 'Clips per sign (min)': audit['per_class'][s]['min'], 'Clips per sign (median)': audit['per_class'][s]['median'], 'Clips per sign (max)': audit['per_class'][s]['max']} for s in ('train', 'val', 'test')]))
print('Signer overlap between splits:', audit['signer_overlap'])
print('Duplicate video IDs:', audit['duplicate_video_ids'], '| Non-finite values:', audit['non_finite_values'], '| Clips dropped for too few hand frames:', audit['dropped_no_hands']['total'])
show('class_distribution.png')"""))
C.append(md("""## 4. Preprocessing

Each clip is trimmed to the span where a hand is visible, with 2 frames either side, then sampled to 30 frames using the nearest frame. Coordinates are centred on the midpoint of the shoulders and divided by shoulder width. Rebuilding features from saved raw landmarks matched the stored values to within 0.0007 on 400 of 400 clips, which checks the pipeline.

Sampling keeps most movement: the median hand span is 38 frames, and 30 frames keep about 95% of fingertip movement at the median."""))
C.append(md("""## 5. Feature extraction

Each frame gives 100 numbers: 7 body points (14), left hand (42), right hand (42) and 2 flags for whether each hand is present. Missing hands are zeros. MediaPipe Holistic uses the same settings in Python and in the browser.

This research tested other features, built separately. None replaced the 30 x 100 input."""))
C.append(md("""## 6. Baseline architecture

The baseline, `final_v2`, averages two bidirectional LSTMs of 128 units, each followed by a dense layer. It takes 114 ms per prediction in the browser."""))
C.append(code("""display(pd.DataFrame([{'Measure': k.replace('_', ' '), 'Baseline, validation': pc(base[k])} for k in ('top1', 'top5', 'macro_precision', 'macro_recall', 'macro_f1')]))
print('Re-running the saved baseline on validation matched its saved predictions to within 1e-14.')"""))
C.append(md("""## 7. Experimental methodology

- **Splits:** the official signer-independent splits. No signer appears in two splits.
- **Selection:** validation only. Test clips are removed before a training process starts.
- **Automation:** `python/auto_optimize.py` trains from a queue, scores each run, compares it, queues follow-ups and keeps `experiments/BEST_MODEL.json`.
- **Comparisons:** a paired bootstrap on shared validation clips gives a 95% interval, so small differences are not called gains.
- **Quick tests:** ideas were first compared on 5,000 training clips. Those scores are only meaningful against each other.
- **Test:** `python/final_eval.py` runs once and refuses to run again. It also uses 8,161 official test clips never used before, because the earlier 3,983 had already been scored twice.

The diagnosis in `research/DIAGNOSIS.md` found the main limit is data: a large gap between training and validation accuracy, and accuracy that rises sharply with the number of clips."""))
C.append(code("show('accuracy_by_tracking_gaps.png')\ndisplay(Markdown('Top-1 falls as more of a clip has no visible hand. This is a correlation, so it was tested directly in section 8.'))"))
C.append(md("## 8. Research-driven improvements\n\nThe quick tests on 5,000 training clips came first, against a control."))
C.append(code("""g = board[board.partial_train.notna()].sort_values('top1', ascending=False)[['name', 'top1', 'top5', 'macro_f1', 'gap']].copy()
display(g.round(4))
show('gate_comparison.png')"""))
C.append(md("Promising ideas then ran on all 14,626 training clips. These are compared with the baseline on the same validation clips."))
C.append(code("""f = board[board.partial_train.isna()].sort_values('top1', ascending=False)[['id', 'name', 'arch', 'top1', 'top5', 'macro_f1', 'gap', 'best_epoch', 'params', 'train_minutes']].copy()
display(f.round(4))
show('training_curves.png')"""))
C.append(md("## 9. Model comparison\n\nThe final model was chosen on validation by a rule fixed in advance: grow an ensemble greedily, stop when a member adds under 0.3 points, at most 4 members."))
C.append(code("""show('model_comparison.png')
display(pd.DataFrame(choice['trace']).round(4))
display(pd.DataFrame([{'Member': m['id'], 'Name': m['name'], 'Own validation top-1': pc(m['solo_top1']), 'Temperature': round(m['temperature'], 2)} for m in choice['members']]))
print('Ensemble on validation:', {k: round(v, 4) for k, v in choice['val'].items()})"""))
C.append(md("Why this model: it scored highest on validation under that rule, it needs no change to the 30 x 100 input, and it is faster in the browser than the baseline. The deployed copy was retrained on training plus validation clips, as planned before any test score existed."))
C.append(md("## 10. Final test results"))
C.append(code("""rows = []
for s, name in (('fresh', 'Unseen test clips'), ('used', 'Earlier test clips (scored before)')):
    for m, label in (('final', 'Final'), ('final_train_only', 'Training clips only'), ('baseline', 'Baseline final_v2')):
        r = M[m][s]
        rows.append({'Test set': name, 'Clips': r['n'], 'Model': label, 'Top-1': pc(r['top1']), '95% interval': f"{pc(r['top1_ci95'][0])} to {pc(r['top1_ci95'][1])}", 'Top-5': pc(r['top5']), 'Macro precision': pc(r['macro_precision']), 'Macro recall': pc(r['macro_recall']), 'Macro F1': pc(r['macro_f1']), 'Weighted F1': pc(r['weighted_f1'])})
display(pd.DataFrame(rows))
print(json.dumps(rep['paired_vs_baseline'], indent=1))"""))
C.append(md("## 11. Confusion matrix\n\nA 1,000 x 1,000 matrix cannot be read on a page, so this shows the 30 lowest-recall signs. The full matrix is in `results/final_test/`."))
C.append(code("show('confusion_matrix.png')"))
C.append(md("## 12. Error analysis"))
C.append(code("""pcs = pd.read_csv(R + 'results/final_test/final_fresh_per_class.csv')
display(pd.concat([pcs.head(10).reset_index(drop=True).add_prefix('best_'), pcs.tail(10).iloc[::-1].reset_index(drop=True).add_prefix('worst_')], axis=1))
show('per_class_performance.png')
sig = M['final']['fresh']['by_signer']
vals = sorted(sig.values())
print(f'Across the {len(vals)} test signers, top-1 ranges from {pc(vals[0])} to {pc(vals[-1])}.')"""))
C.append(md("## 13. Discussion"))
C.append(code("display(Markdown(open(R + 'research/final_discussion.md').read()))"))
C.append(md("""## 14. Limitations

- **No real webcam measurement.** All scores come from dataset video. Real webcam accuracy has not been measured.
- **Same test signers.** Both test sets come from the same 11 official test signers, so the score may not hold for very different signers or cameras.
- **Few clips per sign in testing.** Per-sign scores carry large uncertainty.
- **Validation was used heavily.** Many runs were compared on the same 3,763 clips, so validation scores are somewhat optimistic. The test score is the one to trust.
- **Not all planned work was done.** Pretraining on the other 1,731 signs was not tested, and graph networks had no fair test (one attempt reached 3 epochs on a time budget). Both remain future work.
- **Some signs may be near-duplicates,** such as star and stars, which limits any model on this vocabulary."""))
C.append(md("""## 15. Conclusion

The main limit was data per sign. The largest gains came from training methods and from giving a different architecture a fair test, not from more input features. See section 10 for the final scores."""))
C.append(md("""## 16. References

1. Desai et al. ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition. NeurIPS 2023. https://arxiv.org/abs/2304.05934
2. Microsoft. ASL Citizen code. https://github.com/microsoft/ASL-citizen-code
3. Vandendriessche et al. Representing Signs as Signs: One-Shot ISLR. 2025. https://arxiv.org/abs/2502.20171
4. Hoyeol Sohn. Google Isolated Sign Language Recognition, 1st place solution. https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
5. Kaggle ISLR 2nd place write-up. https://habr.com/en/articles/795109
6. dos Santos et al. Proper Body Landmark Subset for LIBRAS. 2025. https://arxiv.org/html/2510.24887v3
7. Kvanchiani et al. Training Strategies for Isolated Sign Language Recognition. 2024. https://arxiv.org/html/2412.11553v1
8. Bai, Kolter, Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks. 2018. https://arxiv.org/abs/1803.01271
9. Zhang et al. mixup: Beyond Empirical Risk Minimization. ICLR 2018. https://arxiv.org/abs/1710.09412
10. Guo et al. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599"""))
nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "SIGNO_SUBMISSION/Signo_Research.ipynb")
print("wrote", len(C), "cells")
