"""Choose the final model from finished experiments, using validation clips only, and write research/final_choice.json.

Rule, written before seeing the outcome: rank single runs by validation top-1, then grow an ensemble greedily, by averaging calibrated predictions.
Stop when another member adds less than 0.3 points, or at 4 members. Temperature is fitted per member on validation by lowest log loss.
"""
import glob
import json
import os

import numpy as np

MIN_STEP, MAX_MEMBERS = 0.003, 4


def fit_t(p, y):
    lg = np.log(p + 1e-9)
    best = (9e9, 1.0)
    for t in np.arange(0.3, 1.5, 0.01):
        q = np.exp(lg / t)
        q /= q.sum(1, keepdims=True)
        nll = -np.log(q[np.arange(len(y)), y] + 1e-9).mean()
        if nll < best[0]:
            best = (nll, float(t))
    return best[1]


def soften(p, t):
    q = np.exp(np.log(p + 1e-9) / t)
    return q / q.sum(1, keepdims=True)


def main():
    res = [json.load(open(f)) for f in glob.glob("experiments/exp_*/result.json")]
    res = [r for r in res if not r["config"].get("train_limit") and os.path.exists(f"{r['dir']}/model.keras")]
    res.sort(key=lambda r: -r["metrics"]["top1"])
    y = np.load(f"{res[0]['dir']}/val_y.npy")
    vids = np.load(f"{res[0]['dir']}/val_videos.npy").tolist()
    P = {}
    for r in res:
        v = np.load(f"{r['dir']}/val_videos.npy").tolist()
        p = np.load(f"{r['dir']}/val_probs.npy")
        idx = {a: i for i, a in enumerate(v)}
        p = p[[idx[a] for a in vids]]
        r["t"] = fit_t(p, y)
        P[r["id"]] = soften(p, r["t"])
    chosen, trace, cur = [], [], np.zeros_like(next(iter(P.values())))
    while len(chosen) < MAX_MEMBERS:
        scores = {r["id"]: float((((cur * len(chosen) + P[r["id"]]) / (len(chosen) + 1)).argmax(1) == y).mean()) for r in res if r["id"] not in chosen}
        if not scores:
            break
        best = max(scores, key=scores.get)
        prev = trace[-1]["top1"] if trace else 0
        if chosen and scores[best] - prev < MIN_STEP:
            trace.append({"tried": best, "top1": scores[best], "stopped": f"gain {scores[best] - prev:+.4f} is below {MIN_STEP}"})
            break
        cur = (cur * len(chosen) + P[best]) / (len(chosen) + 1)
        chosen.append(best)
        trace.append({"added": best, "top1": scores[best]})
    top5 = float((np.argsort(cur, 1)[:, -5:] == y[:, None]).any(1).mean())
    from sklearn.metrics import precision_recall_fscore_support

    pr, rc, f1, _ = precision_recall_fscore_support(y, cur.argmax(1), labels=range(cur.shape[1]), zero_division=0)
    conf, ok = cur.max(1), cur.argmax(1) == y
    bins = np.linspace(0, 1, 16)
    ece = sum(abs(ok[g].mean() - conf[g].mean()) * g.mean() for g in [(conf >= a) & (conf < b) for a, b in zip(bins[:-1], bins[1:])] if g.any())
    out = {"rule": __doc__, "members": [{"id": r["id"], "name": r["name"], "dir": r["dir"], "temperature": r["t"], "solo_top1": r["metrics"]["top1"], "config": r["config"]} for c in chosen for r in res if r["id"] == c], "trace": trace, "val": {"top1": float(ok.mean()), "top5": top5, "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()), "ece": float(ece), "shown_at_0_7": float((conf >= 0.7).mean()), "correct_when_shown_at_0_7": float(ok[conf >= 0.7].mean())}, "candidates": [{"id": r["id"], "name": r["name"], "top1": r["metrics"]["top1"], "t": r["t"]} for r in res]}
    json.dump(out, open("research/final_choice.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("trace", "val")}, indent=1))
    print([(m["id"], m["name"], round(m["temperature"], 2)) for m in out["members"]])


if __name__ == "__main__":
    main()
