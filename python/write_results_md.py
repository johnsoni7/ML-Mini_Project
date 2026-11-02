"""Write results.md from report.json, so the readable table can never disagree with the measured numbers.

Usage: python write_results_md.py --results results --dataset "ASL Citizen, 50 signs"
"""
import argparse
import json
import os


def pct(v):
    return f"{v * 100:.1f}%"


def table(rows):
    out = ["| Sign | Precision | Recall | F1 | Test clips |", "|---|---:|---:|---:|---:|"]
    out += [f"| {r['sign']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | {r['test_clips']} |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-note", default="The official signer-independent split was used exactly as published.")
    a = ap.parse_args()
    r = json.load(open(os.path.join(a.results, "report.json")))
    c, m = r["counts"], r["model_config"]
    ex = r.get("extraction") or {}
    lines = [
        f"# Signo results: {a.dataset}",
        "",
        "Generated from `report.json` by `python/write_results_md.py`. One training run (single seed), so no mean or standard deviation is reported.",
        "",
        "| Measure | Validation | Test |",
        "|---|---:|---:|",
        f"| Top-1 accuracy | {pct(r['val_top1'])} | **{pct(r['test_top1'])}** |",
        f"| Top-5 accuracy | {pct(r['val_top5'])} | **{pct(r['test_top5'])}** |",
        f"| Clips | {c['val']} | {c['test']} |",
        f"| Signers | {r['signers']['val']} | {r['signers']['test']} |",
        "",
        f"Random guessing would score {pct(r['chance_top1'])} top-1 and {pct(5 * r['chance_top1'])} top-5. Mean per-class precision {pct(r['macro_precision'])}, recall {pct(r['macro_recall'])}.",
        "",
        "## Setup",
        "",
        f"- **Dataset:** {a.dataset}. {a.split_note}",
        f"- **Classes:** {r['num_classes']}. **Clips:** {c['train']} train, {c['val']} validation, {c['test']} test.",
        f"- **Signers:** {r['signers']['train']} train, {r['signers']['val']} validation, {r['signers']['test']} test. Overlap between splits: {sum(r['signer_overlap'].values())}.",
        f"- **Model:** {m['model']}, {m['units']} units, dropout {m['dropout']}, Adam learning rate {m['lr']}, batch {m['batch']}, {m['params']:,} parameters.",
        f"- **Training:** seed {m['seed']}, {m['aug']} augmented copies per training clip, kept the epoch with the best {m.get('select', 'val_loss').replace('_', ' ')}, patience {m['patience']}{', learning rate halved when validation loss stalled' if m.get('reduce_lr') else ''}. Stopped at epoch {m['epochs_run']}, kept epoch {m['best_epoch']}, {m['train_seconds']:.0f} s on CPU.",
        f"- **Extraction:** {ex.get('selected_videos')} clips selected, {ex.get('successfully_extracted')} extracted. Status counts: {ex.get('status_counts')}.",
        "",
        "## Figures",
        "",
        "![Loss curve](loss_curve.png)",
        "",
        "![Accuracy curve](accuracy_curve.png)",
        "",
        "![Confusion matrix](confusion_matrix.png)",
        "",
        "## Best signs on the test set",
        "",
        table(r["best_classes"]),
        "",
        "## Worst signs on the test set",
        "",
        table(r["worst_classes"]),
        "",
        "## Most common confusions",
        "",
        "| True sign | Predicted as | Clips |",
        "|---|---|---:|",
    ]
    lines += [f"| {x['true']} | {x['predicted']} | {x['count']} |" for x in r["top_confusions"]]
    lines += ["", "Per-class numbers for every sign are in `per_class_metrics.csv` and the raw matrix is in `confusion_matrix.csv`.", ""]
    cmp_path = os.path.join(a.results, "model_comparison.json")
    if os.path.exists(cmp_path):
        lines += ["## Model comparison (validation only)", "", "The model was chosen from this table before the test set was scored.", "",
                  "| Run | Units | Dropout | Kept epoch | Val top-1 | Val top-5 | Val loss | Train accuracy | Parameters | Minutes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {x['run']} | {x['units']} | {x['dropout']} | {x['best_epoch']}/{x['epochs_run']} | {pct(x['val_top1'])} | {pct(x['val_top5'])} | {x['val_loss']:.3f} | {pct(x['train_accuracy'])} | {x['params']:,} | {x['train_minutes']} |" for x in json.load(open(cmp_path))]
        lines.append("")
    open(os.path.join(a.results, "results.md"), "w").write("\n".join(lines))
    print("wrote", os.path.join(a.results, "results.md"))


if __name__ == "__main__":
    main()
