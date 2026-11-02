"""Compare finished training runs using validation numbers only. The test set is never read here.

Usage: python compare_runs.py --out results_1000/model_comparison.json model_out/run_a model_out/run_b ...
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        c = json.load(open(os.path.join(run, "config.json")))
        v = json.load(open(os.path.join(run, "val_report.json")))
        h = json.load(open(os.path.join(run, "history.json")))
        e = c["best_epoch"] - 1
        rows.append({
            "run": os.path.basename(run.rstrip("/")), "units": c["units"], "dropout": c["dropout"], "lr": c["lr"],
            "select": c.get("select", "val_loss"), "reduce_lr": c.get("reduce_lr", False),
            "params": c["params"], "model_mb": round(os.path.getsize(os.path.join(run, "model.keras")) / 1e6, 1),
            "best_epoch": c["best_epoch"], "epochs_run": c["epochs_run"], "train_minutes": round(c["train_seconds"] / 60, 1),
            "val_top1": v["top1"], "val_top5": v["top5"], "val_loss": h["val_loss"][e], "train_accuracy": h["accuracy"][e],
            "train_val_gap": h["accuracy"][e] - h["val_accuracy"][e],
        })
    rows.sort(key=lambda r: -r["val_top1"])
    json.dump(rows, open(a.out, "w"), indent=2)
    print("| Run | Units | Dropout | Kept epoch | Val top-1 | Val top-5 | Val loss | Train acc | Gap | Params | MB | Min |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['run']} | {r['units']} | {r['dropout']} | {r['best_epoch']}/{r['epochs_run']} | {r['val_top1'] * 100:.1f}% | {r['val_top5'] * 100:.1f}% | {r['val_loss']:.3f} | {r['train_accuracy'] * 100:.1f}% | {r['train_val_gap'] * 100:.1f} | {r['params']:,} | {r['model_mb']} | {r['train_minutes']} |")


if __name__ == "__main__":
    main()
