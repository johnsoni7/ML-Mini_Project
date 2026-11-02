"""Record one finished run, using validation clips only, and rebuild results/optimization_comparison.csv. The test set is never read.

Usage: python log_experiment.py <group> <run_dir> "<hypothesis>"
Mirror averaging ("tta") predicts each clip and its mirror image and averages the two.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import glob
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from train_plus import mirror

ROOT = "results/experiments"
COLS = ["experiment", "group", "hypothesis", "arch", "hand_local", "units", "dropout", "smoothing", "weight_decay", "lr", "batch", "mirror", "warp", "drop", "geometry", "seed", "train_clips", "val_clips", "epochs_run", "best_epoch", "train_minutes", "train_accuracy_at_kept", "best_val_loss", "val_top1", "val_top5", "val_macro_precision", "val_macro_recall", "val_macro_f1", "val_top1_left_dominant", "val_top1_right_dominant", "val_top1_mirror_avg", "val_top5_mirror_avg", "val_macro_f1_mirror_avg", "val_top1_left_mirror_avg", "params", "model_mb", "browser_ms_mean", "browser_ms_p95"]
LEFT = {"P12", "P5"}  # validation signers whose left hand is visible more often. See results/error_analysis.md


def scores(p, y, K):
    t = p.argmax(1)
    P, R, F, S = precision_recall_fscore_support(y, t, labels=range(K), zero_division=0)
    m = S > 0
    order = np.argsort(-p, 1)[:, :5]
    return {"top1": float((t == y).mean()), "top5": float(np.mean([y[i] in order[i] for i in range(len(y))])), "P": float(P[m].mean()), "R": float(R[m].mean()), "F1": float(F[m].mean())}


def rebuild():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/*/*.json"))]
    rows.sort(key=lambda r: -max(r.get("val_top1") or 0, r.get("val_top1_mirror_avg") or 0))
    with open("results/optimization_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    group, run, hypothesis = sys.argv[1:4]
    z = np.load(os.path.expanduser("~/signo_data/asl_citizen/ds1000.npz"))
    lab, sp, sg = np.char.lower(z["label"]), z["split"], z["signer"]
    va = np.where(sp == "val")[0]
    labels = json.load(open(f"{run}/labels.json"))["labels"]
    index = {l: i for i, l in enumerate(labels)}
    y = np.array([index[l] for l in lab[va]])
    model = tf.keras.models.load_model(f"{run}/model.keras", compile=False)
    p = model.predict(z["X"][va], batch_size=256, verbose=0)
    pm = model.predict(mirror(z["X"][va]), batch_size=256, verbose=0)
    avg = (p + pm) / 2
    np.save(f"{run}/val_probs.npy", p.astype(np.float32))
    np.save(f"{run}/val_probs_mirror.npy", pm.astype(np.float32))
    K, c, h = len(labels), json.load(open(f"{run}/config.json")), json.load(open(f"{run}/history.json"))
    s, sa = scores(p, y, K), scores(avg, y, K)
    left = np.isin(sg[va], list(LEFT))
    e = c["best_epoch"] - 1
    row = {
        "experiment": os.path.basename(run.rstrip("/")), "group": group, "hypothesis": hypothesis, "arch": c["model"], "hand_local": c.get("hand_local", False), "units": c["units"], "dropout": c["dropout"], "smoothing": c.get("smoothing", 0), "weight_decay": c.get("weight_decay", 0), "lr": c["lr"], "batch": c["batch"],
        "mirror": c.get("mirror", False), "warp": c.get("warp", False), "drop": c.get("drop", False), "geometry": c.get("geometry", True), "seed": c["seed"], "train_clips": c.get("train_clips_used", c["split_summary"]["counts"]["train"]), "val_clips": c["split_summary"]["counts"]["val"],
        "epochs_run": c["epochs_run"], "best_epoch": c["best_epoch"], "train_minutes": round(c["train_seconds"] / 60, 1), "train_accuracy_at_kept": round(h["accuracy"][e], 4), "best_val_loss": round(min(h["val_loss"]), 4),
        "val_top1": round(s["top1"], 4), "val_top5": round(s["top5"], 4), "val_macro_precision": round(s["P"], 4), "val_macro_recall": round(s["R"], 4), "val_macro_f1": round(s["F1"], 4),
        "val_top1_left_dominant": round(float((p.argmax(1) == y)[left].mean()), 4), "val_top1_right_dominant": round(float((p.argmax(1) == y)[~left].mean()), 4),
        "val_top1_mirror_avg": round(sa["top1"], 4), "val_top5_mirror_avg": round(sa["top5"], 4), "val_macro_f1_mirror_avg": round(sa["F1"], 4), "val_top1_left_mirror_avg": round(float((avg.argmax(1) == y)[left].mean()), 4),
        "params": c["params"], "model_mb": round(os.path.getsize(f"{run}/model.keras") / 1e6, 2),
    }
    os.makedirs(f"{ROOT}/{group}", exist_ok=True)
    path = f"{ROOT}/{group}/{row['experiment']}.json"
    if os.path.exists(path):
        old = json.load(open(path))
        row["browser_ms_mean"], row["browser_ms_p95"] = old.get("browser_ms_mean"), old.get("browser_ms_p95")
    json.dump(row, open(path, "w"), indent=2)
    rebuild()
    print(json.dumps({k: row[k] for k in ("experiment", "val_top1", "val_top5", "val_macro_f1", "val_top1_left_dominant", "val_top1_right_dominant", "val_top1_mirror_avg", "val_top1_left_mirror_avg", "train_accuracy_at_kept", "epochs_run", "best_epoch", "params")}))


if __name__ == "__main__":
    main()
