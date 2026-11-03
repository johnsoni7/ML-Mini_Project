"""Audit the 1,000-sign dataset and the baseline's validation errors, and write research/audit.json. Validation clips only: test clips are counted, never predicted."""
import collections
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/signo_data/asl_citizen")
z = np.load(f"{ROOT}/ds1000.npz")
X, labels, splits, signers, videos = z["X"], np.char.lower(z["label"]), z["split"], z["signer"], z["video"]
A = {}
A["clips"] = dict(collections.Counter(splits.tolist()))
A["classes"] = int(len(set(labels.tolist())))
A["signers"] = {s: sorted(set(signers[splits == s].tolist())) for s in ("train", "val", "test")}
A["signer_overlap"] = {f"{a}-{b}": sorted(set(A["signers"][a]) & set(A["signers"][b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
A["duplicate_video_ids"] = int(len(videos) - len(set(videos.tolist())))
A["video_in_two_splits"] = int(sum(1 for v, c in collections.Counter(videos.tolist()).items() if c > 1))
A["identical_feature_clips"] = int(len(X) - len({hash(x.tobytes()) for x in X}))
per = {s: collections.Counter(labels[splits == s].tolist()) for s in ("train", "val", "test")}
A["per_class"] = {s: {"min": min(c.values()), "median": float(np.median(list(c.values()))), "max": max(c.values()), "classes": len(c)} for s, c in per.items()}
A["imbalance_train_max_over_min"] = max(per["train"].values()) / min(per["train"].values())

man = list(csv.DictReader(open(f"{ROOT}/manifest_1000.csv")))
A["manifest_rows"] = len(man)
missing = set(r["video"] for r in man) - set(videos.tolist())
A["dropped_no_hands"] = {"total": len(missing), "by_split": dict(collections.Counter(r["split"] for r in man if r["video"] in missing)), "by_signer": collections.Counter(r["participant"] for r in man if r["video"] in missing).most_common(8)}
A["glosses_with_variant_number"] = sorted({l for l in set(labels.tolist()) if any(ch.isdigit() for ch in l)})[:40]
base = collections.Counter(l.split(" ")[0] for l in set(labels.tolist()))
A["gloss_groups_with_variants"] = sorted(g for g, c in base.items() if c > 1)

# feature health
A["non_finite_values"] = int((~np.isfinite(X)).sum())
L, R = X[:, :, 98] > 0, X[:, :, 99] > 0
A["hand_presence"] = {s: {"left_frames": float(L[splits == s].mean()), "right_frames": float(R[splits == s].mean()), "both_frames": float((L & R)[splits == s].mean()), "neither_frames": float((~L & ~R)[splits == s].mean()), "clips_never_seen_dominant_hand_flip": float((L[splits == s].mean(1) > R[splits == s].mean(1)).mean())} for s in ("train", "val", "test")}
gaps = []
for hand in (L, R):
    for row in hand:
        on = np.where(row)[0]
        if len(on) > 1:
            gaps.append((np.diff(on) > 1).sum())
A["hand_dropouts_per_hand_track_mean"] = float(np.mean(gaps))
A["coordinate_range"] = {"p1": float(np.percentile(X[:, :, :98], 1)), "p99": float(np.percentile(X[:, :, :98], 99)), "max_abs": float(np.abs(X[:, :, :98]).max())}
step = np.abs(np.diff(X[:, :, 14:98], axis=1))
A["repeated_frames_fraction"] = float((np.abs(np.diff(X[:, :, :98], axis=1)).sum(2) == 0).mean())

# signer table
A["signer_table"] = {}
for s in A["signers"]["val"] + A["signers"]["test"] + A["signers"]["train"]:
    m = signers == s
    sw = np.hypot(X[m, :, 2] - X[m, :, 4], X[m, :, 3] - X[m, :, 5]).mean()
    A["signer_table"][s] = {"split": splits[m][0].item(), "clips": int(m.sum()), "left_more_than_right": float((L[m].mean(1) > R[m].mean(1)).mean()), "shoulder_span_check": float(sw)}

# baseline errors on validation (saved predictions, no new inference)
meta = json.load(open("model_out/final_v2/labels.json"))["labels"]
assert meta == sorted(set(labels.tolist())), "label order differs from sorted training labels"
index = {l: i for i, l in enumerate(meta)}
va = np.where(splits == "val")[0]
y = np.array([index[l] for l in labels[va]])
p = np.load("model_out/final_v2/val_probs.npy")
assert len(p) == len(va)
t = p.argmax(1)
A["baseline_val"] = {"top1": float((t == y).mean()), "top5": float((np.argsort(p, 1)[:, -5:] == y[:, None]).any(1).mean()), "n": int(len(y))}
from sklearn.metrics import precision_recall_fscore_support

P, Rc, F, S = precision_recall_fscore_support(y, t, labels=range(len(meta)), zero_division=0)
A["baseline_val"].update(macro_precision=float(P.mean()), macro_recall=float(Rc.mean()), macro_f1=float(F.mean()), classes_recall_0=int((Rc == 0).sum()), classes_recall_le_50=int((Rc <= 0.5).sum()), classes_recall_100=int((Rc == 1).sum()))
A["baseline_val"]["by_signer"] = {s: {"top1": float((t[signers[va] == s] == y[signers[va] == s]).mean()), "n": int((signers[va] == s).sum())} for s in sorted(set(signers[va].tolist()))}
cor = p.max(1)[t == y]
wrong = p.max(1)[t != y]
A["baseline_val"]["confidence"] = {"correct_median": float(np.median(cor)), "wrong_median": float(np.median(wrong)), "wrong_over_0_8": float((wrong > 0.8).mean())}
bins = np.linspace(0, 1, 11)
A["baseline_val"]["ece"] = float(sum(abs((t[m] == y[m]).mean() - p.max(1)[m].mean()) * m.mean() for m in [(p.max(1) >= a) & (p.max(1) < b) for a, b in zip(bins[:-1], bins[1:])] if m.any()))
pairs = collections.Counter((meta[a], meta[b]) for a, b in zip(y[t != y], t[t != y]))
A["baseline_val"]["top_confusions"] = pairs.most_common(30)
und = collections.Counter()
for (a, b), c in pairs.items():
    und[tuple(sorted((a, b)))] += c
A["baseline_val"]["errors_total"] = int((t != y).sum())
A["baseline_val"]["errors_in_top_100_pairs"] = int(sum(c for _, c in und.most_common(100)))
A["baseline_val"]["errors_within_same_base_gloss"] = int(sum(c for (a, b), c in pairs.items() if a.split(" ")[0] == b.split(" ")[0]))
tr_count = np.array([per["train"][l] for l in meta])
A["baseline_val"]["recall_by_train_count"] = {str(c): {"classes": int((tr_count == c).sum()), "recall": float(Rc[tr_count == c].mean())} for c in sorted(set(tr_count.tolist()))}
frames_hand = (X[va, :, 98:100].sum(2) > 0).mean(1)
both = (X[va, :, 98:100].sum(2) == 2).mean(1)
A["baseline_val"]["top1_by_hand_visible_fraction"] = {f"{a:.1f}-{b:.1f}": {"n": int(((frames_hand >= a) & (frames_hand < b)).sum()), "top1": float((t == y)[(frames_hand >= a) & (frames_hand < b)].mean()) if ((frames_hand >= a) & (frames_hand < b)).any() else None} for a, b in ((0, .6), (.6, .8), (.8, .95), (.95, 1.01))}
A["baseline_val"]["worst_classes"] = sorted(((meta[i], float(Rc[i]), float(P[i])) for i in range(len(meta))), key=lambda r: (r[1], r[2]))[:30]

# aspect ratio, by split and signer
sizes = collections.defaultdict(collections.Counter)
import cv2

sig_ar = collections.defaultdict(collections.Counter)
rng = np.random.RandomState(0)
for r in [man[i] for i in rng.permutation(len(man))]:
    cap = cv2.VideoCapture(f"{ROOT}/videos/{r['video']}")
    ar = round(cap.get(3) / max(cap.get(4), 1), 2)
    cap.release()
    sizes[r["split"]][ar] += 1
    sig_ar[r["participant"]][ar] += 1
A["aspect_by_split"] = {k: dict(v) for k, v in sizes.items()}
A["aspect_by_signer"] = {k: dict(v) for k, v in sig_ar.items() if len(v) > 1 or list(v)[0] != 1.33}
json.dump(A, open("research/audit.json", "w"), indent=1, default=str)
for k in ("clips", "classes", "signer_overlap", "duplicate_video_ids", "identical_feature_clips", "per_class", "imbalance_train_max_over_min", "dropped_no_hands", "gloss_groups_with_variants", "non_finite_values", "hand_presence", "hand_dropouts_per_hand_track_mean", "coordinate_range", "repeated_frames_fraction", "aspect_by_split", "aspect_by_signer"):
    print(k, json.dumps(A[k], default=str)[:900])
print(json.dumps({k: v for k, v in A["baseline_val"].items() if k not in ("worst_classes",)}, default=str)[:3800])
print({s: round(v["left_more_than_right"], 2) for s, v in A["signer_table"].items()})
