"""Train a sign classifier from one or more Signo dataset JSON files. The TEST set is never touched here.

Usage:
  python train.py --data ~/signo_data/asl_citizen/signo_dataset.json --out model_out/bilstm_seed0 --seed 0

Three separate jobs for three separate splits:
  train -> the model's weights learn from these clips (plus augmented copies of them)
  val   -> decides when to stop and which checkpoint/configuration is best
  test  -> only loaded here to check for overlap with train/val. Scored once, later, by evaluate.py

Output (--out): model.keras + labels.json + history.json + val_report.json + config.json
Then: evaluate.py scores the chosen run on the test set, export_tfjs.py converts it for the web app.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # tensorflowjs converter needs legacy (tf_keras) models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import collections
import hashlib
import json
import random
import time

import numpy as np
import tensorflow as tf

from common import T, D, augment

SPLITS = ("train", "val", "test")


def load(paths):
    """Dataset JSON files -> dict of arrays. Keeps each clip's split, signer and video id for the leakage checks."""
    xs, ys, splits, signers, vids = [], [], [], [], []
    for p in paths:
        j = json.load(open(p))
        if j["seqLen"] != T or j["featureDim"] != D:
            raise ValueError(f"{p}: feature spec mismatch, expected {T}x{D}")
        for i, s in enumerate(j["samples"]):
            x = np.asarray(s["data"], dtype=np.float32)
            if x.shape != (T, D):
                raise ValueError(f"{p}: sample {i} has shape {x.shape}, expected {(T, D)}")
            if not np.isfinite(x).all():
                raise ValueError(f"{p}: sample {i} contains NaN or infinity")
            xs.append(x)
            ys.append(s["label"].lower())
            splits.append(s.get("split"))
            signers.append(s.get("signer"))
            vids.append(s.get("video_id"))
    return {"X": np.stack(xs), "labels": ys, "splits": splits, "signers": signers, "videos": vids}


def assign_splits(labels, splits, seed=0):
    """Official splits are kept exactly. Only clips with no split at all (browser recordings) get one here,
    70/15/15 per class with a fixed seed. Returns (splits, policy) where policy records which case applied."""
    bad = {s for s in splits if s is not None and s not in SPLITS}
    if bad:
        raise ValueError(f"unknown split names: {bad}")
    if all(s is not None for s in splits):
        return list(splits), "official"
    rng = np.random.RandomState(seed)
    out = list(splits)
    for c in sorted(set(labels)):
        ii = rng.permutation([i for i, (y, s) in enumerate(zip(labels, splits)) if y == c and s is None])
        nv = int(round(len(ii) * 0.15))
        for rank, i in enumerate(ii):
            out[i] = "test" if rank < nv else "val" if rank < 2 * nv else "train"
    return out, "official+stratified_random" if any(s is not None for s in splits) else "stratified_random"


def split_indices(splits):
    s = np.asarray(splits)
    return {name: np.where(s == name)[0] for name in SPLITS}


def validate_splits(labels, splits, signers=None, videos=None, min_train=5):
    """Raises ValueError if the three splits are unusable or leak into one another. Returns a summary dict."""
    idx = split_indices(splits)
    for name in SPLITS:
        if len(idx[name]) == 0:
            raise ValueError(f"the {name} split is empty")
    y = np.asarray(labels)
    classes = sorted(set(labels))
    per_class = {c: {n: int((y[idx[n]] == c).sum()) for n in SPLITS} for c in classes}
    for c, counts in per_class.items():
        if counts["train"] < min_train:
            raise ValueError(f"class {c!r} has only {counts['train']} training clips (need {min_train})")
        if counts["val"] == 0 or counts["test"] == 0:
            raise ValueError(f"class {c!r} is missing from val or test: {counts}")
    if videos is not None and any(v is not None for v in videos):
        seen = collections.Counter(v for v in videos if v is not None)
        dup = [v for v, n in seen.items() if n > 1]
        if dup:
            raise ValueError(f"{len(dup)} video ids appear more than once, e.g. {dup[:3]}")
    signer_overlap = None
    if signers is not None and all(s is not None for s in signers):
        sets = {n: {signers[i] for i in idx[n]} for n in SPLITS}
        signer_overlap = {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }
        if any(signer_overlap.values()):
            raise ValueError(f"a signer appears in more than one split: {signer_overlap}")
    return {
        "classes": len(classes),
        "counts": {n: int(len(idx[n])) for n in SPLITS},
        "min_per_class": {n: min(c[n] for c in per_class.values()) for n in SPLITS},
        "max_per_class": {n: max(c[n] for c in per_class.values()) for n in SPLITS},
        "signers": {n: len({signers[i] for i in idx[n]}) for n in SPLITS} if signer_overlap is not None else None,
        "signer_overlap": signer_overlap,
    }


def build(K, kind="bilstm", units=48, dropout=0.3, lr=3e-3):
    rnn = {
        "bilstm": lambda: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)),
        "gru": lambda: tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units)),
    }[kind]()
    m = tf.keras.Sequential([
        tf.keras.layers.Input((T, D)),
        rnn,
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(K, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def topk(probs, y, k):
    order = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in order[i] for i in range(len(y))]))


def fingerprint(videos, splits):
    """Hash of which clip is in which split, so evaluate.py can prove it scores the same split the model trained with."""
    keys = sorted(f"{v}|{s}" for v, s in zip(videos, splits))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="model_out/bilstm_seed0")
    ap.add_argument("--model", choices=["bilstm", "gru"], default="bilstm")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--aug", type=int, default=4, help="augmented copies per training clip")
    ap.add_argument("--units", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    rng = np.random.RandomState(a.seed)

    d = load(a.data)
    splits, policy = assign_splits(d["labels"], d["splits"], a.seed)
    summary = validate_splits(d["labels"], splits, d["signers"], d["videos"])
    labels = sorted(set(d["labels"]))
    K = len(labels)
    X = d["X"]
    y = np.array([labels.index(l) for l in d["labels"]])
    idx = split_indices(splits)
    tr, va = idx["train"], idx["val"]
    print(f"split policy: {policy}\n{json.dumps(summary)}")

    Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
    for i in tr:
        for _ in range(a.aug):
            Xtr.append(augment(X[i], rng))
            ytr.append(y[i])
    Xtr, ytr = np.stack(Xtr), tf.keras.utils.to_categorical(ytr, K)
    print(f"training on {len(tr)} clips + {a.aug} augmented copies each = {len(Xtr)} sequences")

    os.makedirs(a.out, exist_ok=True)
    model = build(K, a.model, a.units, a.dropout, a.lr)
    best = os.path.join(a.out, "model.keras")
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(best, monitor="val_loss", save_best_only=True),
    ]
    t0 = time.time()
    hist = model.fit(
        Xtr, ytr, validation_data=(X[va], tf.keras.utils.to_categorical(y[va], K)),
        epochs=a.epochs, batch_size=a.batch, shuffle=True, callbacks=cbs, verbose=2,
    )
    seconds = time.time() - t0

    model = tf.keras.models.load_model(best)
    probs = model.predict(X[va], verbose=0)
    k5 = min(5, K)
    val = {"top1": float((probs.argmax(1) == y[va]).mean()), "top5": topk(probs, y[va], k5), "n": int(len(va))}
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    best_epoch = int(np.argmin(h["val_loss"])) + 1
    print(f"\nbest epoch {best_epoch}/{len(h['val_loss'])}  VALIDATION top-1 {val['top1']:.3f}  top-{k5} {val['top5']:.3f}  ({seconds:.0f}s)")

    json.dump({"labels": labels}, open(os.path.join(a.out, "labels.json"), "w"))
    json.dump(h, open(os.path.join(a.out, "history.json"), "w"))
    json.dump(val, open(os.path.join(a.out, "val_report.json"), "w"), indent=2)
    json.dump({
        "model": a.model, "units": a.units, "dropout": a.dropout, "lr": a.lr, "batch": a.batch, "epochs_max": a.epochs,
        "patience": a.patience, "aug": a.aug, "seed": a.seed, "split_policy": policy, "data": [os.path.basename(p) for p in a.data],
        "best_epoch": best_epoch, "epochs_run": len(h["val_loss"]), "train_seconds": round(seconds, 1), "split_summary": summary,
        "split_fingerprint": fingerprint(d["videos"], splits), "params": int(model.count_params()),
    }, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print("saved run to", a.out)


if __name__ == "__main__":
    main()
