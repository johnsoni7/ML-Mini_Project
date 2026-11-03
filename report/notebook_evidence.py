"""Collect everything Signo_Research.ipynb reports into one JSON file. Test results are READ from saved files. No test clip is ever predicted here.

Train and validation clips are predicted, which is allowed. No landmark sequence is written out, because the licence forbids distributing them.
Usage: .venv/bin/python report/notebook_evidence.py <out.json>
"""
import csv
import glob
import json
import os
import re
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

DATA = os.path.expanduser("~/signo_data/asl_citizen")
LEFT = {"P12", "P5"}


def read_json(p):
    return json.load(open(p))


def scores(p, y):
    order = np.argsort(-p, 1)[:, :5]
    eps = 1e-7
    P, R, F, S = precision_recall_fscore_support(y, p.argmax(1), labels=range(p.shape[1]), zero_division=0)
    m = S > 0
    return {"top1": float((p.argmax(1) == y).mean()), "top5": float(np.mean([y[i] in order[i] for i in range(len(y))])), "loss": float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1)).mean()), "macro_f1": float(F[m].mean()), "n": int(len(y))}


def predict(run, X):
    return tf.keras.models.load_model(f"model_out/{run}/model.keras", compile=False).predict(X, batch_size=512, verbose=0)


def run_table():
    rows = []
    for d in sorted(glob.glob("model_out/*")):
        name = os.path.basename(d)
        c, h = read_json(f"{d}/config.json"), read_json(f"{d}/history.json")
        members = c.get("members")
        v = read_json(f"{d}/val_report.json") if os.path.exists(f"{d}/val_report.json") else None
        log = f"logs/train_{name}.log"
        budget = os.path.exists(log) and bool(re.search(r"time (budget|limit)", open(log).read()))
        be = c["best_epoch"]
        rows.append({
            "run": name, "signs": c["split_summary"]["classes"], "arch": c["model"], "units": c["units"], "dropout": c["dropout"], "lr": c["lr"], "batch": c["batch"], "seed": c["seed"],
            "smoothing": c.get("smoothing", 0), "weight_decay": c.get("weight_decay", 0), "hand_local": c.get("hand_local", False), "mirror": c.get("mirror", False), "warp": c.get("warp", False), "drop": c.get("drop", False), "geometry": c.get("geometry", True), "aug": c.get("aug"), "select": c.get("select", "val_loss"),
            "train_clips": c.get("train_clips_used", c["split_summary"]["counts"]["train"]), "params": c["params"], "members": members,
            "epochs_run": None if members else c["epochs_run"], "best_epoch": None if members else be, "minutes": None if members else round(c["train_seconds"] / 60, 1),
            "train_acc": None if members else h["accuracy"][be - 1], "val_loss_kept": None if members else h["val_loss"][be - 1],
            "val_top1": v["top1"] if v else None, "val_top5": v["top5"] if v else None, "stopped_by_budget": budget,
        })
    return rows


def main():
    E = {"runs": run_table()}
    runs = {r["run"]: r for r in E["runs"]}

    # 1,000 signs: train and validation predictions only
    z = np.load(f"{DATA}/ds1000.npz")
    lab, sp, sg, X = np.char.lower(z["label"]), z["split"], z["signer"], z["X"]
    labels = read_json("model_out/final_v2/labels.json")["labels"]
    ix = {l: i for i, l in enumerate(labels)}
    y = np.array([ix[l] for l in lab])
    tr, va = np.where(sp == "train")[0], np.where(sp == "val")[0]
    yv, sv = y[va], sg[va]
    P = {r: np.load(f"model_out/{r}/val_probs.npy") for r in ("w1000_u48_acc", "f_long_s0", "plus_plain128", "g_s2_smooth")}
    V2 = (P["f_long_s0"] + P["plus_plain128"]) / 2
    v3m = read_json("model_out/final_v3/config.json")["members"]
    V3 = np.mean([P[m] for m in v3m], 0)
    E["final_v3_members"] = v3m
    E["val_1000"] = {"v1": scores(P["w1000_u48_acc"], yv), "f_long_s0": scores(P["f_long_s0"], yv), "plus_plain128": scores(P["plus_plain128"], yv), "final_v2": scores(V2, yv), "final_v3": scores(V3, yv)}
    for k, p in (("final_v2", V2), ("final_v3", V3)):
        s = E["val_1000"][k]
        runs[k].update(val_top1=s["top1"], val_top5=s["top5"], val_loss_kept=s["loss"])

    # Every two-model average on validation. The deployed pair was chosen before g_s1 and g_s2_smooth existed
    cand = ("f_long_s0", "plus_plain128", "g_s1", "g_s2_smooth")
    Pc = {r: np.load(f"model_out/{r}/val_probs.npy") for r in cand}
    E["val_pairs"] = sorted(({"pair": [a, b], "top1": float((((Pc[a] + Pc[b]) / 2).argmax(1) == yv).mean()), "top5": float(np.mean([yv[i] in o for i, o in enumerate(np.argsort(-(Pc[a] + Pc[b]), 1)[:, :5])]))} for n, a in enumerate(cand) for b in cand[n + 1:]), key=lambda d: -d["top1"])

    # Clean training accuracy: the same training clips with no random changes, measured the same way as validation
    clean = {}
    for r in ("w1000_u48_acc", "f_long_s0", "plus_plain128"):
        clean[r] = predict(r, X[tr])
    E["clean_train_1000"] = {"v1": float((clean["w1000_u48_acc"].argmax(1) == y[tr]).mean()), "f_long_s0": float((clean["f_long_s0"].argmax(1) == y[tr]).mean()), "plus_plain128": float((clean["plus_plain128"].argmax(1) == y[tr]).mean()), "final_v2": float((((clean["f_long_s0"] + clean["plus_plain128"]) / 2).argmax(1) == y[tr]).mean())}
    del clean

    # Validation error analysis
    v1ok, v2ok = P["w1000_u48_acc"].argmax(1) == yv, V2.argmax(1) == yv
    E["val_by_signer"] = [{"signer": s, "clips": int((sv == s).sum()), "hand": "Left" if s in LEFT else "Right", "v1": float(v1ok[sv == s].mean()), "v2": float(v2ok[sv == s].mean())} for s in ["P12", "P5", "P21", "P26", "P39"]]
    left = np.isin(sv, list(LEFT))
    audit = read_json("results/data_audit.json")
    share_test = 0.353
    E["val_hand"] = {"v1_left": float(v1ok[left].mean()), "v1_right": float(v1ok[~left].mean()), "v2_left": float(v2ok[left].mean()), "v2_right": float(v2ok[~left].mean()), "left_share": {"train": 0.170, "val": float(left.mean()), "test": share_test}}
    E["val_hand"]["v1_reweighted"] = share_test * E["val_hand"]["v1_left"] + (1 - share_test) * E["val_hand"]["v1_right"]
    E["val_hand"]["v2_reweighted"] = share_test * E["val_hand"]["v2_left"] + (1 - share_test) * E["val_hand"]["v2_right"]
    vis = (X[va, :, 98:100].sum(2) > 0).mean(1)
    E["val_by_visibility"] = [{"bin": n, "clips": int(m.sum()), "v1": float(v1ok[m].mean()), "v2": float(v2ok[m].mean())} for n, m in (("Under 50%", vis < .5), ("50% to under 80%", (vis >= .5) & (vis < .8)), ("80% or more", vis >= .8))]
    ntrain = np.bincount(y[tr], minlength=1000)[yv]
    E["val_by_train_count"] = [{"bin": n, "clips": int(m.sum()), "v1": float(v1ok[m].mean()), "v2": float(v2ok[m].mean())} for n, m in (("11 to 13", ntrain <= 13), ("14", ntrain == 14), ("15", ntrain == 15), ("16 to 17", ntrain >= 16))]
    wrong = ~v2ok
    top5 = np.argsort(-V2, 1)[:, :5]
    E["val_errors"] = {"errors": int(wrong.sum()), "signs_with_errors": int(len(set(yv[wrong]))), "worst100_share": float(np.sort(np.bincount(yv[wrong], minlength=1000))[::-1][:100].sum() / wrong.sum()), "wrong_but_in_top5": float(np.mean([yv[i] in top5[i] for i in np.where(wrong)[0]]))}
    pairs = {}
    for t, p_ in zip(yv[wrong], V2.argmax(1)[wrong]):
        pairs[(labels[t], labels[p_])] = pairs.get((labels[t], labels[p_]), 0) + 1
    E["val_confusions"] = [{"true": a, "predicted": b, "clips": n} for (a, b), n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 3]
    conf = V2.max(1)
    E["val_threshold"] = [{"threshold": float(t), "shown": float((conf >= t).mean()), "correct_when_shown": float(v2ok[conf >= t].mean())} for t in (.3, .4, .5, .6, .7, .8, .9, .95)]

    # Dataset and split facts
    def split_facts(split, signer, label):
        out = {}
        for n in ("train", "val", "test"):
            m = split == n
            out[n] = {"clips": int(m.sum()), "signers": int(len(set(signer[m]))), "signs": int(len(set(label[m])))}
        return out
    E["split_1000"] = split_facts(sp, sg, lab)
    E["clips_per_sign_1000"] = {n: {str(k): int(v) for k, v in zip(*np.unique(np.bincount(y[sp == n], minlength=1000), return_counts=True))} for n in ("train", "val", "test")}
    E["signer_clips_1000"] = {n: {s: int((sg[sp == n] == s).sum()) for s in sorted(set(sg[sp == n]))} for n in ("train", "val", "test")}
    E["dominant_hand"] = audit["dominant_hand"]
    E["quality"] = audit["quality_by_split"]
    E["duplicates"] = {"exact_duplicate_groups": audit["exact_duplicate_groups"], "video_id_dups": audit["video_id_dups"]}

    # 50 signs
    z50 = np.load(f"{DATA}/ds50.npz")
    l50, s50, g50, X50 = np.char.lower(z50["label"]), z50["split"], z50["signer"], z50["X"]
    labs50 = read_json("model_out/final50_v2/labels.json")["labels"]
    y50 = np.array([labs50.index(l) for l in l50])
    E["split_50"] = split_facts(s50, g50, l50)
    E["clips_per_sign_50"] = {n: {str(k): int(v) for k, v in zip(*np.unique(np.bincount(y50[s50 == n], minlength=50), return_counts=True))} for n in ("train", "val", "test")}
    E["signer_clips_50"] = {n: {s: int((g50[s50 == n] == s).sum()) for s in sorted(set(g50[s50 == n]))} for n in ("train", "val", "test")}
    va50, tr50 = np.where(s50 == "val")[0], np.where(s50 == "train")[0]
    mem50 = runs["final50_v2"]["members"]
    Pm50 = [np.load(f"model_out/{m}/val_probs.npy") for m in mem50]
    s = scores(np.mean(Pm50, 0), y50[va50])
    runs["final50_v2"].update(val_top1=s["top1"], val_top5=s["top5"], val_loss_kept=s["loss"])
    E["clean_train_50"] = {"v1": float((predict("bilstm_seed0", X50[tr50]).argmax(1) == y50[tr50]).mean()), "final50_v2": float((np.mean([predict(m, X50[tr50]) for m in mem50], 0).argmax(1) == y50[tr50]).mean())}

    # Test results: read from the saved evaluation files, never recomputed from predictions
    E["test"] = {}
    for key, folder in (("1000_v2", "results_1000_v2"), ("50_v2", "results_50_v2"), ("1000_v1", "results_1000"), ("50_v1", "results")):
        r = read_json(f"{folder}/report.json")
        per = list(csv.DictReader(open(f"{folder}/per_class_metrics.csv")))
        cm_rows = list(csv.reader(open(f"{folder}/confusion_matrix.csv")))
        names = cm_rows[0][1:]
        cm = np.array([[int(v) for v in row[1:]] for row in cm_rows[1:]])
        assert abs(np.trace(cm) / cm.sum() - r["test_top1"]) < 1e-9, folder
        P_, R_, F_ = [np.array([float(p[k]) for p in per]) for k in ("precision", "recall", "f1")]
        E["test"][key] = {k: r.get(k) for k in ("run", "evaluated_at", "num_classes", "test_top1", "test_top5", "val_top1", "val_top5", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "counts", "signers", "signer_overlap", "top_confusions", "chance_top1")}
        E["test"][key]["macro_f1"] = r.get("macro_f1") or float(F_.mean())
        E["test"][key]["weighted_f1"] = float(np.average(F_, weights=[int(p["test_clips"]) for p in per]))
        E["test"][key]["correct"] = int(np.trace(cm))
        E["test"][key]["total"] = int(cm.sum())
        E["test"][key]["per_class"] = [{"sign": p["sign"], "precision": float(p["precision"]), "recall": float(p["recall"]), "f1": float(p["f1"]), "clips": int(p["test_clips"])} for p in per]
        i, j = np.nonzero(cm)
        E["test"][key]["labels"] = names
        E["test"][key]["cm_sparse"] = [[int(a), int(b), int(cm[a, b])] for a, b in zip(i, j)]
        E["test"][key]["test_log"] = read_json(f"{folder}/test_log.json")
    E["extraction_1000"] = {k: read_json("results_1000_v2/report.json")["extraction"][k] for k in ("selected_videos", "status_counts", "successfully_extracted", "per_split_ok")}
    E["extraction_50"] = {k: read_json("results_50_v2/report.json")["extraction"][k] for k in ("selected_videos", "status_counts", "successfully_extracted", "per_split_ok")}
    E["fetch_1000"] = read_json(f"{DATA}/manifest_1000.csv.stats.json")
    E["fetch_50"] = read_json(f"{DATA}/manifest.csv.stats.json")
    official = {}
    for n in ("train", "val", "test"):
        rows = list(csv.DictReader(open(f"{DATA}/meta/ASL_Citizen/splits/{n}.csv")))
        official[n] = {"clips": len(rows), "signers": len({r["Participant ID"] for r in rows}), "signs": len({r["Gloss"] for r in rows})}
    E["official"] = official
    E["histories"] = {r: read_json(f"model_out/{r}/history.json") for r in ("f_long_s0", "plus_plain128", "w1000_u48_acc", "s50_u128_s0", "s50_u64_s1", "bilstm_seed0")}
    E["csv_rows"] = list(csv.DictReader(open("results/optimization_comparison.csv")))
    json.dump(E, open(sys.argv[1], "w"))
    print(os.path.getsize(sys.argv[1]) / 1e6, "MB")


if __name__ == "__main__":
    main()
