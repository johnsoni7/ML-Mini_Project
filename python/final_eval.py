"""One-time final test evaluation. Refuses to run again once logged, because a second look could steer decisions.

Scores frozen models on two sets of official test clips:
  used:  the 3,983 clips scored twice in earlier versions, so the numbers compare with earlier reports
  fresh: official test clips for the same signs and signers that had never been used, so the score is unbiased
Model choices are final before this runs. Nothing here feeds back into training or selection.

Usage: python final_eval.py --model final=model_out/final_v4 --model baseline=model_out/final_v2 --out results/final_test
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from features_raw import ROOT, build_clip

LOG = "results/final_test_log.json"


def fix_aspect(X, k):
    """Stored features -> features as if measured with true shapes on a 4:3 reference. Each frame holds both shoulders, so this is exact. k is (w/h)/(4/3)."""
    o = X.copy()
    dx, dy = X[..., 2] - X[..., 4], X[..., 3] - X[..., 5]
    r = 1 / np.maximum(np.hypot(k * dx, dy), 1e-6)
    o[..., 0:98:2] = X[..., 0:98:2] * (k * r)[..., None]
    o[..., 1:98:2] = X[..., 1:98:2] * r[..., None]
    return o


def used_test(names):
    z = np.load(f"{ROOT}/ds1000.npz")
    m = z["split"] == "test"
    X, labels, signers, videos = z["X"][m], np.char.lower(z["label"][m]), z["signer"][m], z["video"][m]
    import cv2

    corrected = X.copy()
    for i, v in enumerate(videos):
        cap = cv2.VideoCapture(f"{ROOT}/videos/{v}")
        w, h = cap.get(3), cap.get(4)
        cap.release()
        if abs(w / h - 4 / 3) > 0.01:
            corrected[i] = fix_aspect(X[i], (w / h) / (4 / 3))
    return X, corrected, np.array([names.index(l) for l in labels]), signers, videos


def fresh_test(names):
    rows = [r for r in csv.DictReader(open(f"{ROOT}/manifest_fresh_test.csv")) if r["status"] == "ok"]
    plain, corrected, keep, missing = [], [], [], 0
    for r in rows:
        p = f"{ROOT}/raw_fresh/{os.path.splitext(r['video'])[0]}.npz"
        if not os.path.exists(p):
            missing += 1
            continue
        z = np.load(p)
        a = build_clip(z)
        if a is None:
            continue
        plain.append(a)
        corrected.append(build_clip(z, aspect=True))
        keep.append(r)
    assert missing == 0, f"{missing} fresh clips not extracted"
    print(f"fresh clips: {len(keep)} of {len(rows)} usable, {len(rows) - len(keep)} had too few hand frames")
    return np.array(plain), np.array(corrected), np.array([names.index(r["gloss"].lower()) for r in keep]), np.array([r["participant"] for r in keep]), np.array([r["video"] for r in keep])


def score(p, y, K):
    t = p.argmax(1)
    P, R, F, S = precision_recall_fscore_support(y, t, labels=range(K), zero_division=0)
    m = S > 0
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(y), (2000, len(y)))
    ok = t == y
    return {"n": int(len(y)), "top1": float(ok.mean()), "top1_ci95": [float(v) for v in np.percentile(ok[idx].mean(1), [2.5, 97.5])], "top5": float((np.argsort(p, 1)[:, -5:] == y[:, None]).any(1).mean()), "macro_precision": float(P[m].mean()), "macro_recall": float(R[m].mean()), "macro_f1": float(F[m].mean()), "weighted_f1": float(f1_score(y, t, average="weighted", zero_division=0)), "classes_recall_le_50": int((R[m] <= 0.5).sum()), "classes_recall_0": int((R[m] == 0).sum())}, (P, R, F, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=dir. A name of baseline is scored on features exactly as deployed before this work")
    ap.add_argument("--out", default="results/final_test")
    a = ap.parse_args()
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    if log:
        raise SystemExit(f"final test already run at {log[0]['at']}. It is not repeated.")
    names = json.load(open("model_out/final_v2/labels.json"))["labels"]
    K = len(names)
    sets = {"used": used_test(names), "fresh": fresh_test(names)}
    out = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}, "sets": {k: {"clips": int(len(v[2])), "signers": sorted(set(v[3].tolist()))} for k, v in sets.items()}}
    os.makedirs(a.out, exist_ok=True)
    preds = {}
    for spec in a.model:
        name, d = spec.split("=")
        assert json.load(open(f"{d}/labels.json"))["labels"] == names, "label order differs"
        m = tf.keras.models.load_model(f"{d}/model.keras", compile=False)
        out["models"][name] = {"dir": d, "params": int(m.count_params())}
        for s, (plain, corr, y, sig, vid) in sets.items():
            x = plain if name == "baseline" else corr
            p = m.predict(x, batch_size=256, verbose=0)
            preds[name, s] = p.argmax(1) == y
            r, (P, R, F, S) = score(p, y, K)
            r["by_signer"] = {g: float(preds[name, s][sig == g].mean()) for g in sorted(set(sig.tolist()))}
            out["models"][name][s] = r
            print(name, s, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
            if name != "baseline":
                np.save(f"{a.out}/{name}_{s}_probs.npy", p.astype(np.float16))
                np.savetxt(f"{a.out}/{name}_{s}_confusion.csv", confusion_matrix(y, p.argmax(1), labels=range(K)), fmt="%d", delimiter=",")
                with open(f"{a.out}/{name}_{s}_per_class.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sign", "test_clips", "precision", "recall", "f1"])
                    for i in np.argsort(-F):
                        w.writerow([names[i], int(S[i]), round(float(P[i]), 4), round(float(R[i]), 4), round(float(F[i]), 4)])
                np.save(f"{a.out}/{s}_y.npy", y)
    rng = np.random.RandomState(0)
    out["paired_vs_baseline"] = {}
    for (name, s), ok in preds.items():
        if name == "baseline" or ("baseline", s) not in preds:
            continue
        d = ok.astype(float) - preds["baseline", s].astype(float)
        boots = d[rng.randint(0, len(d), (2000, len(d)))].mean(1)
        out["paired_vs_baseline"][f"{name}/{s}"] = {"diff": float(d.mean()), "ci95": [float(v) for v in np.percentile(boots, [2.5, 97.5])]}
    print(json.dumps(out["paired_vs_baseline"], indent=1))
    json.dump(out, open(f"{a.out}/report.json", "w"), indent=1)
    json.dump([{"at": out["at"], "models": [s.split("=")[0] for s in a.model], "note": "one-time final evaluation"}], open(LOG, "w"), indent=1)


if __name__ == "__main__":
    main()
