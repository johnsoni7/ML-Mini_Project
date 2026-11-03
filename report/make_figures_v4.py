"""Figures for the version 4 report, from saved results only. No dataset video frames, and no new test evaluation."""
import csv
import json
import shutil
import sys

sys.path.insert(0, "report")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from make_figures import BLUE, GRID, INK, MUTED, ORANGE, arrow, box, save, style

AQUA = "#1baf7a"
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
rep = json.load(open("results/final_test/report.json"))
M = rep["models"]
board = list(csv.DictReader(open("experiments/leaderboard.csv")))
choice = json.load(open("research/final_choice.json"))
names = json.load(open("model_out/final_v2/labels.json"))["labels"]


def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.15))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, gap = 0.135, 0.24, 0.0335
    top = ["ASL Citizen\nvideos", "MediaPipe\nHolistic", "Features\n30 x 100", "TCN and BiLSTM\ntraining", "Unseen test clips\n(scored once)", "TensorFlow.js\nexport"]
    low = ["Webcam", "MediaPipe\nHolistic", "Same features\n30 x 100", "TCN + BiLSTM\nin the browser", "Smoothing and\nthreshold", "Word, top 3,\nconfidence"]
    ax.text(0, 0.95, "Training (Python)", fontsize=9, fontweight="bold", color=INK)
    ax.text(0, 0.42, "Live recognition (browser, on the user's device)", fontsize=9, fontweight="bold", color=INK)
    xs = [i * (w + gap) for i in range(6)]
    for x, t in zip(xs, top):
        box(ax, x + 0.005, 0.60, w, h, t, "#e8f1fc")
    for x, t in zip(xs, low):
        box(ax, x + 0.005, 0.07, w, h, t, "#fdeee6")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, 0.72), (b + 0.005, 0.72))
        arrow(ax, (a + w, 0.19), (b + 0.005, 0.19))
    arrow(ax, (xs[5] + w / 2, 0.60), (xs[3] + w, 0.31))
    save(fig, "v4_pipeline.png")


def dataset():
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.4))
    n, v = ["Train", "Val", "Earlier\ntest", "Unseen\ntest"], [14626, 3763, rep["sets"]["used"]["clips"], rep["sets"]["fresh"]["clips"]]
    b = a[0].bar(n, v, color=[BLUE, BLUE, MUTED, ORANGE], width=0.62)
    a[0].set_title("Usable clips in each set", color=INK)
    a[0].set_ylabel("Clips")
    a[0].margins(y=0.14)
    style(a[0])
    for r, x in zip(b, v):
        a[0].annotate(f"{x:,}", (r.get_x() + r.get_width() / 2, x), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8.5)
    rc = json.load(open("research/audit.json"))["baseline_val"]["recall_by_train_count"]
    ks = sorted(rc, key=int)
    b = a[1].bar([int(k) for k in ks], [rc[k]["classes"] for k in ks], color=BLUE, width=0.7)
    for r, k in zip(b, ks):
        a[1].annotate(str(rc[k]["classes"]), (r.get_x() + r.get_width() / 2, rc[k]["classes"]), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8.5)
    a[1].set_title("Training clips per sign", color=INK)
    a[1].set_xlabel("Clips")
    a[1].set_ylabel("Number of signs")
    a[1].margins(y=0.14)
    style(a[1])
    save(fig, "v4_dataset.png")


def hbars(ax, labels, vals, cols, title, xlabel):
    ys = np.arange(len(labels))[::-1]
    ax.barh(ys, vals, color=cols, height=0.62)
    ax.set_yticks(ys, labels, color=INK, fontsize=8)
    for y, v in zip(ys, vals):
        ax.text(v + max(vals) * 0.008, y, f"{v:.1f}", va="center", color=INK, fontsize=7.8)
    ax.set_xlim(0, max(vals) * 1.09)
    ax.set_xlabel(xlabel)
    ax.set_title(title, color=INK)
    style(ax, "x")


LABEL = {"gate_base": "Control", "gate_mixup": "Mixup", "gate_body_mix": "Mixup + body variation", "gate_smooth": "Label smoothing", "gate_velocity": "Motion input", "gate_interp": "Bridged hand gaps", "gate_body_aug": "Body variation", "gate_linear": "Blended frames", "gate_t45": "45 frames", "gate_face": "Face points", "gate_face_interp": "Face + bridged gaps", "gate_embed": "Frame embedding"}


def gates():
    g = [r for r in board if r["partial_train"] and r["name"] in LABEL]
    g.sort(key=lambda r: -float(r["top1"]))
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    hbars(ax, [LABEL[r["name"]] for r in g], [float(r["top1"]) * 100 for r in g], [ORANGE if r["name"] == "gate_base" else BLUE for r in g], "Quick tests on 5,000 training clips (validation top-1)", "Top-1 (%). Compare only with each other")
    save(fig, "v4_gates.png")


FULL = {"full_tcn_fair": "TCN, mixup 50%", "tcn_always_s1": "TCN, mixup 100%, seed 1", "tcn_always_s2": "TCN, mixup 100%, seed 2", "full_mixup_always+moretime": "BiLSTM, mixup 100%, 240 epochs", "full_mixup_always": "BiLSTM, mixup 100%", "full_mixup_always+seed1": "BiLSTM, mixup 100%, seed 1", "full_mixup_u192": "BiLSTM 192 units, mixup 50%", "always_velocity": "BiLSTM, mixup 100%, motion", "full_mixup_always+smooth": "BiLSTM, mixup 100%, smoothing", "full_mixup_smooth": "BiLSTM, mixup 50%, smoothing", "full_mixup": "BiLSTM, mixup 50%", "full_mixup_body": "BiLSTM, mixup 50%, body variation"}


def models():
    f = sorted((r for r in board if r["name"] in FULL), key=lambda r: -float(r["top1"]))
    lab = ["Selected: TCN + BiLSTM"] + [FULL[r["name"]] for r in f] + ["Version 2 (two BiLSTMs)", "Earlier single BiLSTM"]
    val = [choice["val"]["top1"] * 100] + [float(r["top1"]) * 100 for r in f] + [79.38, 78.02]
    col = [AQUA] + [BLUE] * len(f) + [ORANGE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    hbars(ax, lab, val, col, "Validation top-1 on the same 3,763 clips", "Top-1 (%)")
    save(fig, "v4_models.png")


def curves():
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.5))
    runs = [("experiments/exp_022", "TCN", BLUE), ("experiments/exp_026", "BiLSTM with motion", AQUA), ("model_out/f_long_s0", "Earlier BiLSTM", ORANGE)]
    for d, n, c in runs:
        h = json.load(open(d + "/history.json"))
        x = np.arange(1, len(h["val_accuracy"]) + 1)
        a[0].plot(x, np.array(h["val_accuracy"]) * 100, color=c, lw=1.8, label=n)
        a[1].plot(x, h["val_loss"], color=c, lw=1.8, label=n)
    a[0].set_title("Validation accuracy", color=INK)
    a[0].set_ylabel("Top-1 (%)")
    a[0].set_ylim(0, 100)
    a[1].set_title("Validation loss", color=INK)
    a[1].set_ylabel("Loss")
    a[1].set_ylim(bottom=0)
    for ax in a:
        ax.set_xlabel("Epoch")
        style(ax)
    a[0].legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower right")
    save(fig, "v4_curves.png")


def grouped(ax, labels, old, new, title, ylabel, ymax=108):
    x = np.arange(len(labels))
    for off, v, c, l in ((-0.19, old, ORANGE, "Version 2"), (0.19, new, BLUE, "Version 4")):
        r = ax.bar(x + off, v, 0.36, color=c, label=l)
        for q, val in zip(r, v):
            ax.annotate(f"{val:.1f}", (q.get_x() + q.get_width() / 2, val), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, labels, color=INK, fontsize=8.5)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK)
    style(ax)


def before_after():
    fig, a = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    for ax, s, t in ((a[0], "fresh", f"Unseen test clips ({rep['sets']['fresh']['clips']:,})"), (a[1], "used", f"Earlier test clips ({rep['sets']['used']['clips']:,})")):
        k = ["top1", "top5", "macro_f1"]
        grouped(ax, ["Top-1", "Top-5", "Macro F1"], [M["baseline"][s][q] * 100 for q in k], [M["final"][s][q] * 100 for q in k], t, "Score (%)")
    a[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=2, labelcolor=INK, fontsize=8)
    save(fig, "v4_before_after.png")


def signers():
    keys = sorted(M["final"]["fresh"]["by_signer"], key=lambda g: M["final"]["fresh"]["by_signer"][g])
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    grouped(ax, [f"S{i + 1}" for i in range(len(keys))], [M["baseline"]["fresh"]["by_signer"][g] * 100 for g in keys], [M["final"]["fresh"]["by_signer"][g] * 100 for g in keys], "Top-1 for each of the 11 test signers, unseen clips", "Top-1 (%)")
    ax.legend(frameon=False, loc="upper left", ncol=2, labelcolor=INK, fontsize=8)
    save(fig, "v4_signers.png")


def recall_threshold():
    pcs = list(csv.DictReader(open("results/final_test/final_fresh_per_class.csv")))
    rec = np.array([float(x["recall"]) for x in pcs]) * 100
    fig, ax = plt.subplots(figsize=(3.65, 2.55))
    ax.hist(rec, bins=np.arange(0, 105, 10), color=BLUE, rwidth=0.88)
    ax.axvline(rec.mean(), color=ORANGE, lw=1.8)
    ax.text(rec.mean() - 2, ax.get_ylim()[1] * 0.9, f"mean {rec.mean():.1f}%", ha="right", color=INK, fontsize=8)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Number of signs")
    ax.set_title("Recall per sign", color=INK)
    style(ax)
    save(fig, "v4_recall.png")
    y = np.load("results/final_test/fresh_y.npy")
    p = np.load("results/final_test/final_fresh_probs.npy").astype(float)
    c, ok = p.max(1), p.argmax(1) == y
    ts = np.arange(0.3, 0.96, 0.05)
    fig, ax = plt.subplots(figsize=(3.65, 2.55))
    ax.plot(ts * 100, [(c >= t).mean() * 100 for t in ts], color=ORANGE, lw=1.8, label="Clips given a word")
    ax.plot(ts * 100, [ok[c >= t].mean() * 100 for t in ts], color=BLUE, lw=1.8, label="Correct when given")
    ax.axvline(70, color=MUTED, lw=0.9, ls="--")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Confidence threshold (%)")
    ax.set_ylabel("Share (%)")
    ax.set_title("Effect of the threshold", color=INK)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower left")
    style(ax)
    save(fig, "v4_threshold.png")
    t = {f"{x:.1f}": {"shown": float((c >= x).mean()), "correct": float(ok[c >= x].mean())} for x in (0.5, 0.7, 0.8)}
    json.dump(t, open("results/final_test/threshold_fresh.json", "w"), indent=1)


def matrix():
    cm = np.loadtxt("results/final_test/final_fresh_confusion.csv", delimiter=",")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j] + off[j, i], i, j) for i in range(1000) for j in range(i + 1, 1000)), reverse=True)
    pick = []
    for _, i, j in pairs:
        for k in (i, j):
            if k not in pick and len(pick) < 30:
                pick.append(k)
        if len(pick) >= 30:
            break
    row = cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    im = ax.imshow((cm / row)[np.ix_(pick, pick)], cmap=LinearSegmentedColormap.from_list("b", RAMP), vmin=0, vmax=1)
    lab = [names[i] for i in pick]
    ax.set_xticks(range(30), lab, rotation=90, fontsize=7, color=INK)
    ax.set_yticks(range(30), lab, fontsize=7, color=INK)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("True sign")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=lambda v, _: f"{v * 100:.0f}%")
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    save(fig, "v4_matrix.png")


def speed():
    lat = json.load(open("results/final_test/latency.json"))
    fig, ax = plt.subplots(figsize=(7.4, 1.5))
    hbars(ax, ["Version 2 (two BiLSTMs)", "Version 4 (TCN + BiLSTM)"], [lat["Deployed before this work (final_v2)"], lat["Final model (final_v4)"]], [ORANGE, BLUE], "Mean browser time per prediction", "Milliseconds")
    save(fig, "v4_speed.png")


for f in (pipeline, dataset, gates, models, curves, before_after, signers, recall_threshold, matrix, speed):
    f()
shutil.copy("/private/tmp/claude-501/-Users-johnsonithomas-Desktop-signo/75af5c02-c9c5-47dc-a775-b4de4f551f4e/scratchpad/shots/ui_1000.png", "report/figures/app_screenshot_v4.png")
print("figures done")
