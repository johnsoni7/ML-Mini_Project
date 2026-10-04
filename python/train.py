"""Train a BiLSTM sign classifier from one or more Signo dataset JSON files and export it for the web app.

Usage:
  python train.py --data wlasl100.json signo-dataset.json --out model_out --epochs 80

Output (model_out/): model.json + *.bin (TensorFlow.js layers model) + labels.json + model.keras + report.json
Load model.json, the .bin file(s) and labels.json together in the app: Brain > Save, share & load > Load brain files.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # tensorflowjs converter needs legacy (tf_keras) models

import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from common import T, D, augment


def load(paths, min_per_class):
    xs, ys, splits = [], [], []
    for p in paths:
        j = json.load(open(p))
        assert j["seqLen"] == T and j["featureDim"] == D, f"{p}: feature spec mismatch"
        for s in j["samples"]:
            xs.append(np.asarray(s["data"], dtype=np.float32))
            ys.append(s["label"].lower())
            splits.append(s.get("split"))
    cnt = {}
    for y in ys:
        cnt[y] = cnt.get(y, 0) + 1
    keep = sorted(l for l, c in cnt.items() if c >= min_per_class)
    idx = [i for i, y in enumerate(ys) if y in keep]
    return np.stack([xs[i] for i in idx]), [ys[i] for i in idx], [splits[i] for i in idx], keep


def split(y, splits, seed=0):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    if all(s in ("train", "val", "test") for s in splits):
        s = np.asarray(splits)
        return np.where(s == "train")[0], np.where(s != "train")[0]
    tr, va = [], []
    for c in np.unique(y):
        ii = rng.permutation(np.where(y == c)[0])
        nv = max(1, int(round(len(ii) * 0.2))) if len(ii) >= 5 else 0
        va += list(ii[:nv])
        tr += list(ii[nv:])
    return np.array(tr), np.array(va)


def build(K):
    m = tf.keras.Sequential([
        tf.keras.layers.Input((T, D)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(K, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="model_out")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--aug", type=int, default=4, help="augmented copies per training clip")
    ap.add_argument("--min-per-class", type=int, default=3)
    a = ap.parse_args()

    X, labels_per_sample, splits, labels = load(a.data, a.min_per_class)
    y = np.array([labels.index(l) for l in labels_per_sample])
    K = len(labels)
    tr, va = split(labels_per_sample, splits)
    print(f"{len(X)} clips, {K} classes, train {len(tr)}, held-out {len(va)}")

    Xtr = [X[i] for i in tr]
    ytr = [y[i] for i in tr]
    for i in tr:
        for _ in range(a.aug):
            Xtr.append(augment(X[i]))
            ytr.append(y[i])
    Xtr, ytr = np.stack(Xtr), tf.keras.utils.to_categorical(ytr, K)

    model = build(K)
    cbs = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss" if len(va) else "loss")]
    val = (X[va], tf.keras.utils.to_categorical(y[va], K)) if len(va) else None
    model.fit(Xtr, ytr, validation_data=val, epochs=a.epochs, batch_size=32, shuffle=True, callbacks=cbs, verbose=2)

    ev_i = va if len(va) else tr
    probs = model.predict(X[ev_i], verbose=0)
    top1 = float((probs.argmax(1) == y[ev_i]).mean())
    k5 = min(5, K)
    top5 = float(np.mean([y[ev_i][i] in np.argsort(-probs[i])[:k5] for i in range(len(ev_i))]))
    rep = classification_report(y[ev_i], probs.argmax(1), labels=list(range(K)), target_names=labels, output_dict=True, zero_division=0)
    print(f"\n{'held-out' if len(va) else 'TRAIN (optimistic!)'} top-1 {top1:.3f}  top-{k5} {top5:.3f}")
    print(classification_report(y[ev_i], probs.argmax(1), labels=list(range(K)), target_names=labels, zero_division=0))

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, "model.keras"))
    json.dump({"labels": labels}, open(os.path.join(a.out, "labels.json"), "w"))
    json.dump({"top1": top1, "top5": top5, "held_out": bool(len(va)), "per_class": rep}, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, a.out)
    print("exported TF.js model to", a.out)


if __name__ == "__main__":
    main()
