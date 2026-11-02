"""Figures for the version 2 report, drawn from saved validation predictions and results files. No dataset video frames."""
import csv
import json
import os
import sys

sys.path.insert(0, "report")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from make_figures import BLUE, GRID, INK, MUTED, ORANGE, bars, save, style


def grouped(ax, names, a, b, la, lb, ylabel, title):
    x = np.arange(len(names))
    for off, vals, col, lab in ((-0.19, a, BLUE, la), (0.19, b, ORANGE, lb)):
        r = ax.bar(x + off, vals, 0.36, color=col, label=lab)
        for q, v in zip(r, vals):
            ax.annotate(f"{v:.1f}", (q.get_x() + q.get_width() / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, names, color=INK, fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK)
    style(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, labelcolor=INK, fontsize=8)


def before_after():
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.85))
    grouped(a[0], ["1,000 signs\ntop-1", "1,000 signs\ntop-5"], [61.4, 84.4], [73.4, 92.0], "Version 1", "Version 2", "Accuracy (%)", "1,000-sign model on its test set")
    grouped(a[1], ["50 signs\ntop-1", "50 signs\ntop-5"], [76.7, 95.9], [91.2, 97.6], "Version 1", "Version 2", "Accuracy (%)", "50-sign model on its test set")
    save(fig, "v2_before_after.png")


def signers():
    z = np.load(os.path.expanduser("~/signo_data/asl_citizen/ds1000.npz"))
    lab, va = np.char.lower(z["label"]), z["split"] == "val"
    labels = json.load(open("model_out/final_v2/labels.json"))["labels"]
    ix = {l: i for i, l in enumerate(labels)}
    y, sg = np.array([ix[l] for l in lab[va]]), z["signer"][va]
    b = np.load("model_out/w1000_u48_acc/val_probs.npy").argmax(1) == y
    n = ((np.load("model_out/f_long_s0/val_probs.npy") + np.load("model_out/plus_plain128/val_probs.npy")) / 2).argmax(1) == y
    order = ["P12", "P5", "P21", "P26", "P39"]
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    grouped(ax, [f"{s}\n{'left' if s in ('P12', 'P5') else 'right'}-dominant" for s in order], [b[sg == s].mean() * 100 for s in order], [n[sg == s].mean() * 100 for s in order], "Version 1", "Version 2", "Top-1 (%)", "Validation accuracy for each signer")
    save(fig, "v2_signers.png")


def ablation():
    r = {}
    for f in ("augmentation/a_control", "augmentation/a_no_mirror", "augmentation/c_no_warp", "data_need/c_half_data", "normalisation/a_hand_local", "training_optimisation/c_smoothing60", "training_optimisation/c_weight_decay"):
        d = json.load(open(f"results/experiments/{f}.json"))
        r[d["experiment"]] = d["val_top1"] * 100
    names = ["Half the\nclips", "No time\nwarping", "Weight\ndecay", "Wrist-relative\nhands", "No\nmirroring", "All kept\n(control)", "Label\nsmoothing"]
    keys = ["c_half_data", "c_no_warp", "c_weight_decay", "a_hand_local", "a_no_mirror", "a_control", "c_smoothing60"]
    fig, ax = plt.subplots(figsize=(7.4, 2.35))
    bars(ax, names, [r[k] for k in keys], "One change at a time (about 60 epochs, validation)", "Top-1 (%)", fmt="{:.1f}")
    ax.set_ylim(50, 80)
    ax.tick_params(axis="x", labelsize=8, colors=INK)
    save(fig, "v2_ablation.png")


def curves():
    h = json.load(open("model_out/f_long_s0/history.json"))
    best = json.load(open("model_out/f_long_s0/config.json"))["best_epoch"]
    ep = range(1, len(h["loss"]) + 1)
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.35))
    for ax, tr, va, title, lab, sc in ((a[0], "loss", "val_loss", "Loss by epoch", "Loss (lower is better)", 1), (a[1], "accuracy", "val_accuracy", "Accuracy by epoch", "Accuracy (%)", 100)):
        ax.plot(ep, [v * sc for v in h[tr]], color=BLUE, lw=2, label="Training")
        ax.plot(ep, [v * sc for v in h[va]], color=ORANGE, lw=2, label="Validation")
        ax.axvline(best, color=MUTED, lw=1, ls="--")
        style(ax)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="center right", labelcolor=INK, fontsize=8)
    a[0].annotate(f"epoch kept: {best}", (best, 1), xycoords=("data", "axes fraction"), xytext=(-4, -12), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    a[1].annotate(f"epoch kept: {best}", (best, 0), xycoords=("data", "axes fraction"), xytext=(-4, 6), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    save(fig, "v2_curves.png")


def recall_threshold():
    r = json.load(open("results_1000_v2/report.json"))
    n = {}
    for x in r["per_class"]:
        k = f"{round(x['recall'] * 100)}%"
        n[k] = n.get(k, 0) + 1
    keys = sorted(n, key=lambda k: int(k[:-1]))
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    bars(ax, keys, [n[k] for k in keys], "Signs at each test recall", "Number of signs")
    ax.set_xlabel("Recall (most signs have 4 test clips)")
    ax.tick_params(axis="x", labelsize=8)
    save(fig, "v2_recall.png")
    t = json.load(open("results_1000_v2/threshold_validation.json"))
    x = [q["threshold"] * 100 for q in t]
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    ax.plot(x, [q["shown"] * 100 for q in t], color=BLUE, lw=2, marker="o", ms=4, label="Clips given a word")
    ax.plot(x, [q["correct_when_shown"] * 100 for q in t], color=ORANGE, lw=2, marker="o", ms=4, label="Words that are correct")
    ax.axvline(70, color=MUTED, lw=1, ls="--")
    ax.annotate("default 70%", (70, 48), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_ylim(35, 104)
    style(ax)
    ax.set_title("Threshold trade-off (validation)", color=INK)
    ax.set_xlabel("Confidence threshold (%)")
    ax.set_ylabel("Share (%)")
    ax.legend(frameon=False, loc="lower left", labelcolor=INK, fontsize=7.8)
    save(fig, "v2_threshold.png")


def matrix():
    rows = list(csv.reader(open("results_1000_v2/confusion_matrix.csv")))
    labels, cm = rows[0][1:], np.array([[int(v) for v in r[1:]] for r in rows[1:]])
    off = cm - np.diag(np.diag(cm))
    keep = []
    for i, j in sorted(zip(*np.nonzero(off)), key=lambda p: (-off[p], labels[p[0]], labels[p[1]])):
        for k in (int(i), int(j)):
            if k not in keep and len(keep) < 30:
                keep.append(k)
    keep.sort(key=lambda k: labels[k])
    sub = cm[np.ix_(keep, keep)] / cm[keep].sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    ramp = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
    im = ax.imshow(sub, cmap=ramp, vmin=0, vmax=1)
    n = len(keep)
    ax.set_xticks(range(n), [labels[k] for k in keep], rotation=90, fontsize=7, color=INK)
    ax.set_yticks(range(n), [labels[k] for k in keep], fontsize=7, color=INK)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=.5)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("True sign")
    for sp in ax.spines.values():
        sp.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.035, pad=.02, format=lambda v, _: f"{v * 100:.0f}%")
    bar.set_label("Share of that sign's test clips", fontsize=8)
    bar.ax.tick_params(labelsize=7.5)
    bar.outline.set_visible(False)
    save(fig, "v2_matrix.png")


for f in (before_after, signers, ablation, curves, recall_threshold, matrix):
    f()
print(sorted(x for x in os.listdir("report/figures") if x.startswith("v2")))
