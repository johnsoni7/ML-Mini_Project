"""Draw the final charts from saved measurements only. A chart whose data does not exist yet is skipped, and the script says so.

Usage: python make_final_figures.py [--out results] [--copy SIGNO_SUBMISSION/figures]
"""
import argparse
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]


def axes(size=(8, 4.5)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE, dpi=160)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, path, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def hbars(path, title, names, values, xlabel, colour=BLUE, marks=None):
    fig, ax = axes((8, max(3, 0.34 * len(names) + 1.2)))
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(names))[::-1]
    cols = marks if marks else [colour] * len(names)
    ax.barh(ys, values, color=cols, height=0.62)
    ax.set_yticks(ys, names, color=INK, fontsize=9)
    for y, v in zip(ys, values):
        ax.text(v + max(values) * 0.008, y, f"{v:.1f}", va="center", color=INK, fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.1)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--copy", default="")
    ap.add_argument("--final", default="model_out/final_v4", help="selected model's folder")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    written = []

    rows = list(csv.DictReader(open("experiments/leaderboard.csv")))
    full = [r for r in rows if not r["partial_train"]]
    gates = [r for r in rows if r["partial_train"]]
    base = json.load(open("research/baseline.json"))

    # model comparison, validation
    names = ["Deployed before this work (final_v2, 2 models)", "Earlier single-model reference (f_long_s0)"] + [r["name"] for r in sorted(full, key=lambda r: -float(r["top1"]))]
    vals = [base["top1"] * 100, 78.02] + [float(r["top1"]) * 100 for r in sorted(full, key=lambda r: -float(r["top1"]))]
    marks = [ORANGE, ORANGE] + [BLUE] * len(full)
    if os.path.exists("research/final_choice.json"):
        ch = json.load(open("research/final_choice.json"))
        names.insert(0, "Selected: " + " + ".join(m["name"] for m in ch["members"]))
        vals.insert(0, ch["val"]["top1"] * 100)
        marks.insert(0, AQUA)
    hbars(f"{a.out}/model_comparison.png", "Validation top-1 on the same 3,763 clips, all training clips used", names, vals, "Top-1 accuracy (%)", marks=marks)
    written.append("model_comparison.png")

    # gates
    s = sorted(gates, key=lambda r: -float(r["top1"]))
    hbars(f"{a.out}/gate_comparison.png", "Quick tests on 5,000 training clips (validation top-1)", [r["name"].replace("gate_", "") + (" (setup fault)" if r["name"] in ("gate_tcn", "gate_cnn_attn") else "") for r in s], [float(r["top1"]) * 100 for r in s], "Top-1 accuracy (%). Compare only with each other", marks=[ORANGE if r["name"] == "gate_base" else BLUE for r in s])
    written.append("gate_comparison.png")

    # training curves of the best run
    b = json.load(open("experiments/BEST_MODEL.json")) if os.path.exists("experiments/BEST_MODEL.json") else None
    if b:
        h = json.load(open(f"{b['dir']}/history.json"))
        ref = json.load(open("model_out/f_long_s0/history.json"))
        fig, ax = axes()
        ax.plot(np.arange(1, len(h["val_accuracy"]) + 1), np.array(h["val_accuracy"]) * 100, color=BLUE, lw=2, label=f"{b['name']}: validation")
        ax.plot(np.arange(1, len(ref["val_accuracy"]) + 1), np.array(ref["val_accuracy"]) * 100, color=ORANGE, lw=2, label="f_long_s0 (earlier reference): validation")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
        finish(fig, ax, f"{a.out}/training_curves.png", "Validation accuracy during training", "Epoch", "Top-1 accuracy (%)")
        written.append("training_curves.png")

    # class distribution, from the audit's per-class counts
    aud = json.load(open("research/audit.json"))
    rc = aud["baseline_val"]["recall_by_train_count"]
    cs = sorted(rc, key=int)
    fig, ax = axes()
    ax.bar([int(c) for c in cs], [rc[c]["classes"] for c in cs], color=BLUE, width=0.7)
    for c in cs:
        ax.text(int(c), rc[c]["classes"] + 6, str(rc[c]["classes"]), ha="center", color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)
    finish(fig, ax, f"{a.out}/class_distribution.png", "Training clips per sign: 1,000 signs are nearly balanced", "Training clips for a sign", "Number of signs")
    written.append("class_distribution.png")

    # hand tracking gaps
    g = json.load(open("research/diagnose_raw.json"))["top1_by_gap_share"]
    ks = [k for k in g if g[k]["n"] >= 50]
    hbars(f"{a.out}/accuracy_by_tracking_gaps.png", "Deployed model, validation: accuracy by share of frames with no hand", [f"{float(k.split('-')[0]) * 100:.0f} to {float(k.split('-')[1]) * 100:.0f}% (n={g[k]['n']})" for k in ks], [g[k]["top1"] * 100 for k in ks], "Top-1 accuracy (%)")
    written.append("accuracy_by_tracking_gaps.png")

    # latency, only if measured
    if os.path.exists("results/final_test/latency.json"):
        lat = json.load(open("results/final_test/latency.json"))
        hbars(f"{a.out}/latency_comparison.png", "Measured prediction time in the browser", list(lat), list(lat.values()), "Milliseconds per prediction (mean)")
        written.append("latency_comparison.png")
    else:
        print("skipped latency chart: not measured yet")

    # final test charts
    r = "results/final_test/report.json"
    if os.path.exists(r):
        rep = json.load(open(r))
        names = json.load(open("model_out/final_v2/labels.json"))["labels"]
        cm = np.loadtxt("results/final_test/final_fresh_confusion.csv", delimiter=",")
        row = cm.sum(1, keepdims=True).clip(min=1)
        worst = np.argsort(np.diag(cm) / row[:, 0])[:30]
        sub = cm[np.ix_(worst, worst)] / row[worst]
        fig, ax = plt.subplots(figsize=(9.5, 8.5), facecolor=SURFACE, dpi=160)
        im = ax.imshow(sub, cmap=LinearSegmentedColormap.from_list("blue", RAMP), vmin=0, vmax=1)
        lab = [names[i] for i in worst]
        ax.set_xticks(range(30), lab, rotation=90, fontsize=7.5, color=INK)
        ax.set_yticks(range(30), lab, fontsize=7.5, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=lambda v, _: f"{v * 100:.0f}%")
        bar.ax.tick_params(labelsize=8, colors=MUTED)
        bar.outline.set_visible(False)
        ax.set_title("Confusion matrix: the 30 lowest-recall signs (unseen test clips)", loc="left", color=INK, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted sign", color=MUTED)
        ax.set_ylabel("True sign", color=MUTED)
        fig.tight_layout()
        fig.savefig(f"{a.out}/confusion_matrix.png", facecolor=SURFACE)
        plt.close(fig)
        written.append("confusion_matrix.png")
        pc = list(csv.DictReader(open("results/final_test/final_fresh_per_class.csv")))
        rec = np.array([float(x["recall"]) for x in pc]) * 100
        fig, ax = axes()
        ax.hist(rec, bins=np.arange(0, 105, 5), color=BLUE, rwidth=0.9)
        ax.grid(axis="x", visible=False)
        ax.axvline(rec.mean(), color=ORANGE, lw=2)
        ax.text(rec.mean() - 1, ax.get_ylim()[1] * 0.92, f"mean {rec.mean():.1f}%", color=INK, ha="right", fontsize=9)
        finish(fig, ax, f"{a.out}/per_class_performance.png", "Recall per sign on unseen test clips", "Recall (%)", "Number of signs")
        written.append("per_class_performance.png")
    else:
        print("skipped final test charts: test not run yet")

    if a.copy:
        os.makedirs(a.copy, exist_ok=True)
        for f in written:
            shutil.copy(f"{a.out}/{f}", f"{a.copy}/{f}")


if __name__ == "__main__":
    main()
