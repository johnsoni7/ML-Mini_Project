"""Score ONE chosen run on the held-out TEST set and write everything under results/.

Usage:
  python evaluate.py --run model_out/bilstm_seed0 --data ~/signo_data/asl_citizen/signo_dataset.json --results results

Run this only after the final configuration has been chosen using validation scores. Every evaluation is logged
in results/test_log.json, and a second look at the same test set is refused unless --allow-repeat is passed,
because choosing between models by their test score would make that score optimistic.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from train import assign_splits, fingerprint, load, split_indices, topk, validate_splits

# Chart colours: blue/orange pair checked with the colour-blindness validator; matrix uses a single-hue ramp.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
TRAIN_COL, VAL_COL = "#2a78d6", "#eb6834"
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curve(path, train, val, title, ylabel, best_epoch, percent=False):
    """One measure per chart: train and val share one axis, with the chosen epoch marked."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE, dpi=160)
    style(ax)
    ep = np.arange(1, len(train) + 1)
    scale = 100 if percent else 1
    ax.plot(ep, np.array(train) * scale, color=TRAIN_COL, lw=2, label="Training")
    ax.plot(ep, np.array(val) * scale, color=VAL_COL, lw=2, label="Validation")
    ax.axvline(best_epoch, color=MUTED, lw=1, ls="--")
    ax.annotate(f"model kept: epoch {best_epoch}", (best_epoch, 1), xycoords=("data", "axes fraction"), xytext=(5, -12),
                textcoords="offset points", color=MUTED, fontsize=9)
    if percent:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.legend(frameon=False, loc="center right", labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def matrix_png(path, cm, labels, title):
    """Row-normalised so each row reads as 'of the clips that really were this sign, where did they go?'"""
    from matplotlib.colors import LinearSegmentedColormap

    row = cm.sum(1, keepdims=True).clip(min=1)
    n = len(labels)
    size = max(7, n * 0.24)
    fig, ax = plt.subplots(figsize=(size + 1.2, size), facecolor=SURFACE, dpi=160)
    im = ax.imshow(cm / row, cmap=LinearSegmentedColormap.from_list("blue", RAMP), vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8, color=INK)
    ax.set_yticks(range(n), labels, fontsize=8, color=INK)
    ax.set_xlabel("Predicted sign", color=MUTED, fontsize=11)
    ax.set_ylabel("True sign", color=MUTED, fontsize=11)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=lambda v, _: f"{v * 100:.0f}%")
    bar.set_label("Share of that sign's test clips", color=MUTED, fontsize=9)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--extraction-report", default="", help="defaults to <results>/extraction_report.json")
    ap.add_argument("--allow-repeat", action="store_true", help="only after final selection, e.g. reporting more seeds of a fixed config")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, "config.json")))
    labels = json.load(open(os.path.join(a.run, "labels.json")))["labels"]
    hist = json.load(open(os.path.join(a.run, "history.json")))
    d = load(a.data)
    splits, policy = assign_splits(d["labels"], d["splits"], cfg["seed"])
    summary = validate_splits(d["labels"], splits, d["signers"], d["videos"])
    fp = fingerprint(d["videos"], splits)
    if fp != cfg["split_fingerprint"]:
        raise SystemExit("this dataset's splits differ from the ones the model trained with")
    if sorted(set(d["labels"])) != labels:
        raise SystemExit("label set differs from the trained model")

    os.makedirs(a.results, exist_ok=True)
    log_path = os.path.join(a.results, "test_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    if [e for e in log if e["split_fingerprint"] == fp] and not a.allow_repeat:
        raise SystemExit(f"test set already evaluated: {log}. Pass --allow-repeat only if selection is already final.")

    model = tf.keras.models.load_model(os.path.join(a.run, "model.keras"))
    if tuple(model.input_shape[1:]) != (d["X"].shape[1], d["X"].shape[2]) or model.output_shape[-1] != len(labels):
        raise SystemExit(f"model shape mismatch: {model.input_shape} -> {model.output_shape}")
    y = np.array([labels.index(l) for l in d["labels"]])
    idx = split_indices(splits)
    k5 = min(5, len(labels))

    metrics = {}
    for name in ("val", "test"):
        probs = model.predict(d["X"][idx[name]], verbose=0)
        metrics[name] = {"probs": probs, "y": y[idx[name]], "top1": float((probs.argmax(1) == y[idx[name]]).mean()), "topk": topk(probs, y[idx[name]], k5), "n": int(len(idx[name]))}
    te = metrics["test"]
    pred = te["probs"].argmax(1)
    P, R, F, S = precision_recall_fscore_support(te["y"], pred, labels=range(len(labels)), zero_division=0)
    cm = confusion_matrix(te["y"], pred, labels=range(len(labels)))

    rows = [{"sign": l, "precision": round(float(P[i]), 4), "recall": round(float(R[i]), 4), "f1": round(float(F[i]), 4), "test_clips": int(S[i])} for i, l in enumerate(labels)]
    with open(os.path.join(a.results, "per_class_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.results, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\predicted"] + labels)
        for i, l in enumerate(labels):
            w.writerow([l] + [int(v) for v in cm[i]])

    ranked = sorted(rows, key=lambda r: (r["f1"], r["recall"]), reverse=True)
    off = cm - np.diag(np.diag(cm))
    pairs = sorted(((int(off[i, j]), labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if off[i, j]), reverse=True)[:10]
    ep_best = cfg["best_epoch"]
    tag = cfg["model"].upper()
    curve(os.path.join(a.results, "loss_curve.png"), hist["loss"], hist["val_loss"], f"{tag} loss by epoch", "Loss (lower is better)", ep_best)
    curve(os.path.join(a.results, "accuracy_curve.png"), hist["accuracy"], hist["val_accuracy"], f"{tag} accuracy by epoch", "Accuracy (%)", ep_best, percent=True)
    matrix_png(os.path.join(a.results, "confusion_matrix.png"), cm, labels, f"{tag} test-set confusion matrix ({te['n']} clips, unseen signers)")

    ex_path = a.extraction_report or os.path.join(a.results, "extraction_report.json")
    extraction = json.load(open(ex_path)) if os.path.exists(ex_path) else None
    report = {
        "run": os.path.basename(a.run.rstrip("/")),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_policy": policy,
        "num_classes": len(labels),
        "labels": labels,
        "counts": summary["counts"],
        "signers": summary["signers"],
        "signer_overlap": summary["signer_overlap"],
        "test_top1": te["top1"], f"test_top{k5}": te["topk"], "val_top1": metrics["val"]["top1"], f"val_top{k5}": metrics["val"]["topk"],
        "chance_top1": 1 / len(labels),
        "macro_precision": float(P.mean()), "macro_recall": float(R.mean()),
        "best_classes": ranked[:10], "worst_classes": ranked[-10:][::-1], "top_confusions": [{"count": c, "true": t, "predicted": p} for c, t, p in pairs],
        "per_class": rows,
        "model_config": cfg,
        "extraction": {k: extraction[k] for k in ("selected_videos", "status_counts", "successfully_extracted", "per_split_ok", "wall_seconds", "workers")} if extraction else None,
    }
    json.dump(report, open(os.path.join(a.results, "report.json"), "w"), indent=2)
    log.append({"run": report["run"], "at": report["evaluated_at"], "split_fingerprint": fp, "test_top1": te["top1"], f"test_top{k5}": te["topk"]})
    json.dump(log, open(log_path, "w"), indent=2)

    print(f"VAL   top-1 {metrics['val']['top1']:.4f}  top-{k5} {metrics['val']['topk']:.4f}  n={metrics['val']['n']}")
    print(f"TEST  top-1 {te['top1']:.4f}  top-{k5} {te['topk']:.4f}  n={te['n']}  (chance {1 / len(labels):.4f})")
    print("best :", [(r["sign"], r["f1"]) for r in ranked[:5]])
    print("worst:", [(r["sign"], r["f1"]) for r in ranked[-5:]])
    print("confused:", pairs[:5])


if __name__ == "__main__":
    main()
