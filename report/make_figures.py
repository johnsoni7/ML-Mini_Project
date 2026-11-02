"""Draw the report's figures from measured results. Uses no dataset video frames."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "report/figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlelocation": "left"})


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=220)
    plt.close(fig)


def bars(ax, names, values, title, ylabel, fmt="{:,}"):
    b = ax.bar(names, values, color=BLUE, width=0.62)
    style(ax)
    ax.set_title(title, color=INK)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.14)
    for r, v in zip(b, values):
        ax.annotate(fmt.format(v), (r.get_x() + r.get_width() / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8.5)


def splits():
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.5))
    bars(a[0], ["Train", "Validation", "Test"], [14626, 3763, 3983], "Usable clips in each split", "Clips")
    bars(a[1], ["Train", "Validation", "Test"], [35, 5, 11], "Signers in each split (none shared)", "Signers")
    save(fig, "splits.png")


def clips_per_sign():
    c = json.load(open("results_1000/clips_per_sign.json"))
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.3))
    for ax, key, title in ((a[0], "train", "Training clips per sign"), (a[1], "val", "Validation clips per sign")):
        bars(ax, list(c[key]), list(c[key].values()), title, "Number of signs")
        ax.set_xlabel("Clips")
    save(fig, "clips_per_sign.png")


def box(ax, x, y, w, h, text, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fill, ec=MUTED, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.8, color=INK)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))


def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.15))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, gap = 0.135, 0.24, 0.0335
    top = ["ASL Citizen\nvideos", "MediaPipe\nHolistic", "Features\n30 x 100", "BiLSTM\ntraining", "Test set\n(scored once)", "TensorFlow.js\nexport"]
    low = ["Webcam", "MediaPipe\nHolistic", "Same features\n30 x 100", "BiLSTM in\nthe browser", "Smoothing and\nthreshold", "Word, top 3,\nconfidence"]
    ax.text(0, 0.95, "Training (Python)", fontsize=9, fontweight="bold", color=INK)
    ax.text(0, 0.42, "Live recognition (browser, on the user's device)", fontsize=9, fontweight="bold", color=INK)
    xs = [i * (w + gap) for i in range(6)]
    for x, t in zip(xs, top):
        box(ax, x + 0.005, 0.60, w, h, t, "#e8f1fc")
    for x, t in zip(xs, low):
        box(ax, x + 0.005, 0.07, w, h, t, "#fdeee7")
    for i in range(5):
        if i != 4:  # the test set only scores the model, it feeds nothing onward
            arrow(ax, (xs[i] + w + 0.008, 0.72), (xs[i + 1] + 0.002, 0.72))
        arrow(ax, (xs[i] + w + 0.008, 0.19), (xs[i + 1] + 0.002, 0.19))
    ax.add_patch(FancyArrowPatch((xs[3] + w * 0.8, 0.6), (xs[5] - 0.005, 0.60), connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    arrow(ax, (xs[5] + w / 2, 0.59), (xs[3] + w + 0.01, 0.30))
    save(fig, "pipeline.png")


def runs():
    rows = json.load(open("results_1000/model_comparison.json"))
    names = {"w1000_u48_acc": "Round 2, 48 units (selected)", "w1000_u128_acc": "Round 2, 128 units", "w1000_u256_acc": "Round 2, 256 units", "w1000_u48": "Round 1, 48 units", "w1000_u256": "Round 1, 256 units", "w1000_u128": "Round 1, 128 units"}
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    y = range(len(rows))[::-1]
    for off, key, col, lab in ((0.19, "train_accuracy", BLUE, "Training accuracy"), (-0.19, "val_top1", ORANGE, "Validation accuracy")):
        b = ax.barh([i + off for i in y], [r[key] * 100 for r in rows], height=0.34, color=col, label=lab)
        for r, s in zip(b, rows):
            ax.annotate(f"{s[key] * 100:.1f}%", (r.get_width(), r.get_y() + r.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(y), [names[r["run"]] for r in rows], color=INK)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Accuracy at the kept epoch (%)")
    style(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, labelcolor=INK, fontsize=8.5)
    save(fig, "runs.png")


def recall():
    r = json.load(open("results_1000/report.json"))
    n = {}
    for x in r["per_class"]:
        k = f"{round(x['recall'] * 100)}%"
        n[k] = n.get(k, 0) + 1
    keys = sorted(n, key=lambda k: int(k[:-1]))
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    bars(ax, keys, [n[k] for k in keys], "Signs at each test recall", "Number of signs")
    ax.set_xlabel("Recall (most signs have 4 test clips)")
    ax.tick_params(axis="x", labelsize=8)
    save(fig, "recall.png")


def threshold():
    t = json.load(open("results_1000/threshold_validation.json"))
    x = [r["threshold"] * 100 for r in t]
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    ax.plot(x, [r["shown"] * 100 for r in t], color=BLUE, lw=2, marker="o", ms=4, label="Clips given a word")
    ax.plot(x, [r["correct_when_shown"] * 100 for r in t], color=ORANGE, lw=2, marker="o", ms=4, label="Words that are correct")
    ax.axvline(70, color=MUTED, lw=1, ls="--")
    ax.annotate("default 70%", (70, 40), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_ylim(35, 104)
    style(ax)
    ax.set_title("Threshold trade-off (validation)", color=INK)
    ax.set_xlabel("Confidence threshold (%)")
    ax.set_ylabel("Share (%)")
    ax.legend(frameon=False, loc="lower left", labelcolor=INK, fontsize=7.8)
    save(fig, "threshold.png")


def curves():
    h = json.load(open("model_out/w1000_u48_acc/history.json"))
    best = json.load(open("model_out/w1000_u48_acc/config.json"))["best_epoch"]
    ep = range(1, len(h["loss"]) + 1)
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.45))
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
    save(fig, "curves.png")


def matrix():
    """Readable at report size: only the signs from the most frequent confusions. Reads the saved table, not the test set."""
    import csv

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    rows = list(csv.reader(open("results_1000/confusion_matrix.csv")))
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
    save(fig, "matrix.png")


if __name__ == "__main__":
    for f in (splits, clips_per_sign, pipeline, runs, recall, threshold, curves, matrix):
        f()
    print(sorted(os.listdir(OUT)))
