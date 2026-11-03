"""Write results/FINAL_RESULTS.md and results/experiment_comparison.csv and .json from saved files only."""
import csv
import json
import os


def pc(v):
    return f"{v * 100:.1f}%"


def main():
    rows = list(csv.DictReader(open("experiments/leaderboard.csv")))
    json.dump(rows, open("results/experiment_comparison.json", "w"), indent=1)
    with open("results/experiment_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    base = json.load(open("research/baseline.json"))
    ch = json.load(open("research/final_choice.json"))
    rep = json.load(open("results/final_test/report.json"))
    lat = json.load(open("results/final_test/latency.json")) if os.path.exists("results/final_test/latency.json") else {}
    single = max((r for r in rows if not r["partial_train"]), key=lambda r: float(r["top1"]))
    M = rep["models"]
    o = ["# Final results\n", "All selection used validation clips. Test clips were scored once, by `python/final_eval.py`, after every choice was fixed.\n", "## Model comparison (validation, same 3,763 clips)\n", "| Model | Validation top-1 | Validation top-5 | Macro F1 | Parameters |", "|---|---:|---:|---:|---:|"]
    o.append(f"| Baseline: `final_v2`, deployed before this work | {pc(base['top1'])} | {pc(base['top5'])} | {pc(base['macro_f1'])} | 792,784 |")
    o.append(f"| Best single run: `{single['id']}` {single['name']} | {pc(float(single['top1']))} | {pc(float(single['top5']))} | {pc(float(single['macro_f1']))} | {int(single['params']):,} |")
    o.append(f"| Selected ensemble, trained on training clips only | {pc(ch['val']['top1'])} | {pc(ch['val']['top5'])} | {pc(ch['val']['macro_f1'])} | {M['final_train_only']['params']:,} |")
    o.append("\nThe deployed model is the same ensemble retrained on training plus validation clips, so it has no validation score of its own.\n")
    o.append("Members: " + ", ".join(f"`{m['id']}` {m['name']} (temperature {m['temperature']:.2f})" for m in ch["members"]) + ".\n")
    for key, title, note in (("fresh", "Unseen test clips", "These official test clips had never been used. **This is the unbiased score.**"), ("used", "Earlier test clips", "These clips were scored in versions 1 and 2, so this set is not untouched. It is here to compare with earlier reports.")):
        o += [f"## Test results: {title} ({rep['sets'][key]['clips']:,} clips, {len(rep['sets'][key]['signers'])} signers)\n", note + "\n", "| Model | Top-1 | 95% interval | Top-5 | Macro precision | Macro recall | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, label in (("final", "**Final (training + validation)**"), ("final_train_only", "Same ensemble, training clips only"), ("baseline", "Baseline `final_v2`")):
            r = M[name][key]
            o.append(f"| {label} | {pc(r['top1'])} | {pc(r['top1_ci95'][0])} to {pc(r['top1_ci95'][1])} | {pc(r['top5'])} | {pc(r['macro_precision'])} | {pc(r['macro_recall'])} | {pc(r['macro_f1'])} | {pc(r['weighted_f1'])} |")
        d = rep["paired_vs_baseline"][f"final/{key}"]
        o.append(f"\nFinal minus baseline on the same clips: {d['diff'] * 100:+.1f} points (95% interval {d['ci95'][0] * 100:+.1f} to {d['ci95'][1] * 100:+.1f}).\n")
    pcs = list(csv.DictReader(open("results/final_test/final_fresh_per_class.csv")))
    o += ["## Best and worst signs (unseen test clips)\n", "| Best 10 | F1 | Worst 10 | F1 |", "|---|---:|---|---:|"]
    for a, b in zip(pcs[:10], pcs[-10:][::-1]):
        o.append(f"| {a['sign']} | {pc(float(a['f1']))} | {b['sign']} | {pc(float(b['f1']))} |")
    o.append(f"\nSigns with recall of 50% or less: {M['final']['fresh']['classes_recall_le_50']} (baseline {M['baseline']['fresh']['classes_recall_le_50']}). The full list is in `results/final_test/final_fresh_per_class.csv`.\n")
    o.append("## Generalisation gap\n")
    o.append(f"For the best single run, clean training accuracy is {pc(float(single['train_top1_clean']))} against validation {pc(float(single['top1']))}, a gap of {float(single['gap']) * 100:.1f} points. The earlier deployed members had a gap of about 18 points. The model still memorises to a degree.\n")
    if lat:
        o += ["## Browser speed\n", "| Model | Mean ms per prediction |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in lat.items()] + [""]
    if os.path.exists("research/final_discussion.md"):
        o.append(open("research/final_discussion.md").read())
    open("results/FINAL_RESULTS.md", "w").write("\n".join(o) + "\n")
    print("wrote results/FINAL_RESULTS.md")


if __name__ == "__main__":
    main()
