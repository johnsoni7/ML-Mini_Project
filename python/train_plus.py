"""Stronger training for the same 30 x 100 input. The feature spec in common.py and the browser's features are unchanged.

Differences from train.py:
  - training clips get fresh random changes every epoch, including mirroring (left-handed signing) and time warping
  - optional larger model: two convolution layers, two BiLSTMs, pooling. All layers load in TensorFlow.js
  - the learning rate falls smoothly over the time budget
Test clips are never scored here. evaluate.py still does that.

Usage:
  python train_plus.py --out model_out/plus_deep --arch deep --units 128 --max-minutes 25
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import random
import time

import numpy as np
import tensorflow as tf

from common import T, D
from train import fingerprint, split_indices, topk, validate_splits

L = tf.keras.layers


def load_arrays(path, vocab=None):
    z = np.load(path)
    labels = np.char.lower(z["label"])
    keep = np.isin(labels, sorted(vocab)) if vocab else np.ones(len(labels), bool)
    return {"X": z["X"][keep], "labels": labels[keep].tolist(), "splits": z["split"][keep].tolist(), "signers": z["signer"][keep].tolist(), "videos": z["video"][keep].tolist()}


def mirror(x):
    """Flip left and right: negate every x, swap the paired pose points, the two hands and their flags."""
    o = x.copy()
    o[:, :, 0:98:2] = -o[:, :, 0:98:2]
    for a, b in ((1, 2), (3, 4), (5, 6)):
        ia, ib = [a * 2, a * 2 + 1], [b * 2, b * 2 + 1]
        o[:, :, ia], o[:, :, ib] = o[:, :, ib].copy(), o[:, :, ia].copy()
    left, right = o[:, :, 14:56].copy(), o[:, :, 56:98].copy()
    o[:, :, 14:56], o[:, :, 56:98] = right, left
    o[:, :, [98, 99]] = o[:, :, [99, 98]]
    return o


def time_warp(x, rng, min_keep=0.8):
    """Keep a random stretch of the clip and resample it unevenly, so speed and timing vary."""
    n = len(x)
    keep = rng.uniform(min_keep, 1.0, n)
    start = rng.uniform(0, 1, n) * (1 - keep)
    steps = rng.gamma(4.0, 1.0, (n, T))
    pos = np.cumsum(steps, 1)
    pos = (pos - pos[:, :1]) / np.maximum(pos[:, -1:] - pos[:, :1], 1e-6)
    idx = np.rint((start[:, None] + keep[:, None] * pos) * (T - 1)).astype(int).clip(0, T - 1)
    return np.take_along_axis(x, idx[:, :, None], 1)


def drop_hand_frames(x, rng, p=0.3, max_len=4):
    """Imitate MediaPipe briefly losing a hand."""
    o = x.copy()
    for i in np.where(rng.random(len(x)) < p)[0]:
        ln, s = rng.integers(1, max_len + 1), rng.integers(0, T)
        block, flag = ((14, 56), 98) if rng.random() < 0.5 else ((56, 98), 99)
        o[i, s:s + ln, block[0]:block[1]] = 0
        o[i, s:s + ln, flag] = 0
    return o


def augment_batch(x, rng, use_mirror=True, warp=True, drop=True, geometry=True):
    if use_mirror:
        m = rng.random(len(x)) < 0.5
        if m.any():
            x = x.copy()
            x[m] = mirror(x[m])
    if warp:
        x = time_warp(x, rng)
    if drop:
        x = drop_hand_frames(x, rng)
    if not geometry:
        return x.astype(np.float32)
    n = len(x)
    s = rng.uniform(0.85, 1.15, n)[:, None, None]
    r = rng.uniform(-0.14, 0.14, n)
    c, sn = np.cos(r)[:, None, None], np.sin(r)[:, None, None]
    dx, dy = rng.uniform(-0.15, 0.15, (2, n))[:, :, None, None]
    px, py = x[:, :, 0:98:2], x[:, :, 1:98:2]
    present = np.concatenate([np.ones((n, T, 7), bool), np.repeat(x[:, :, 98:99] > 0, 21, 2), np.repeat(x[:, :, 99:100] > 0, 21, 2)], 2)
    jx, jy = rng.normal(0, 0.012, (2, n, T, 49))
    o = x.copy()
    o[:, :, 0:98:2] = np.where(present, (px * c - py * sn) * s + dx + jx, 0)
    o[:, :, 1:98:2] = np.where(present, (px * sn + py * c) * s + dy + jy, 0)
    return o.astype(np.float32)


class Batches(tf.keras.utils.Sequence):
    def __init__(self, X, y, K, batch, seed, **aug):
        self.X, self.y, self.K, self.batch, self.aug = X, y, K, batch, aug
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))

    def __len__(self):
        return len(self.X) // self.batch

    def __getitem__(self, i):
        ii = self.order[i * self.batch:(i + 1) * self.batch]
        return augment_batch(self.X[ii], self.rng, **self.aug), tf.keras.utils.to_categorical(self.y[ii], self.K)

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.X))


def adjacency():
    """Real connections between the 49 landmarks, from MediaPipe's pose and hand layouts. Index 0-6 pose, 7-27 left hand, 28-48 right hand."""
    hand = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (5, 7), (6, 28)]
    edges += [(a + 7, b + 7) for a, b in hand] + [(a + 28, b + 28) for a, b in hand]
    A = np.eye(49, dtype=np.float32)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    return A / A.sum(1, keepdims=True)


def fixed_dense(units, kernel, name=None):
    """A layer with frozen weights. It lets the browser model do fixed maths using only layers TensorFlow.js supports."""
    return L.Dense(units, use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Constant(kernel), name=name)


def hand_local_kernel(scale=3.0):
    """Finger points become offsets from their own wrist, enlarged. The wrist itself stays where it is, so the output is still 100 numbers."""
    k = np.eye(100, dtype=np.float32)
    for block in (14, 56):
        for j in range(1, 21):
            for axis in (0, 1):
                point, wrist = block + 2 * j + axis, block + axis
                k[point, point] = scale
                k[wrist, point] = -scale
    return k


def graph_block(x, channels, dropout, A):
    x = L.Conv2D(channels, 1)(x)
    x = L.Permute((1, 3, 2))(x)
    x = fixed_dense(49, A.T)(x)  # each landmark gathers information from the landmarks it is connected to
    x = L.Permute((1, 3, 2))(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = L.Conv2D(channels, (9, 1), padding="same")(x)  # then each landmark looks along time
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return L.Dropout(dropout)(x)


def build(K, arch, units, dropout, smoothing, hand_local=False):
    i = tf.keras.Input((T, D))
    start = fixed_dense(100, hand_local_kernel())(i) if hand_local else i
    if arch == "graph":
        x = fixed_dense(98, np.eye(100, 98, dtype=np.float32))(start)  # the graph uses coordinates. Missing hands are already zeros
        x = L.Reshape((T, 49, 2))(x)
        A = adjacency()
        for ch in (units // 2, units, units):
            x = graph_block(x, ch, dropout * 0.5, A)
        x = L.GlobalAveragePooling2D()(x)
    elif arch == "plain":
        x = L.Bidirectional(L.LSTM(units))(start)
        x = L.Dropout(dropout)(x)
        x = L.Dense(units, activation="relu")(x)
    else:
        x = L.Conv1D(units * 2, 3, padding="same", activation="relu")(start)  # short filters can learn frame-to-frame movement
        x = L.BatchNormalization()(x)
        x = L.Dropout(dropout * 0.5)(x)
        x = L.Conv1D(units * 2, 5, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.SpatialDropout1D(dropout * 0.5)(x)
        x = L.Bidirectional(L.LSTM(units, return_sequences=True))(x)
        x = L.Bidirectional(L.LSTM(units, return_sequences=True))(x)
        x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
        x = L.Dropout(dropout)(x)
        x = L.Dense(units * 2, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    o = L.Dense(K, activation="softmax")(x)
    m = tf.keras.Model(i, o)
    return m, tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/signo_data/asl_citizen/ds1000.npz"))
    ap.add_argument("--vocab", default="", help="optional file of signs, one per line, to train on only those")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["plain", "deep", "graph"], default="deep")
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="above 0 uses AdamW")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="below 1 uses only part of each sign's training clips, to measure whether more data would help")
    ap.add_argument("--hand-local", action="store_true", help="fingers measured from their own wrist, computed inside the model")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-warp", action="store_true")
    ap.add_argument("--no-drop", action="store_true")
    ap.add_argument("--no-geometry", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=25)
    ap.add_argument("--max-epochs", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=0, help="fixed epoch count. Fair for comparisons, because slow neighbouring jobs can't change it")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.epochs:
        a.max_epochs, a.max_minutes = a.epochs, 100000

    random.seed(a.seed)
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    vocab = {w.strip().lower() for w in open(a.vocab) if w.strip()} if a.vocab else None
    d = load_arrays(a.data, vocab)
    summary = validate_splits(d["labels"], d["splits"], d["signers"], d["videos"], min_train=1)
    labels = sorted(set(d["labels"]))
    K, X = len(labels), d["X"]
    index = {l: i for i, l in enumerate(labels)}
    y = np.array([index[l] for l in d["labels"]])
    idx = split_indices(d["splits"])
    tr, va = idx["train"], idx["val"]
    if a.train_fraction < 1:
        pick = np.random.RandomState(a.seed)
        tr = np.concatenate([(lambda c: pick.permutation(c)[:max(1, round(len(c) * a.train_fraction))])(tr[y[tr] == k]) for k in range(K)])
        print(f"learning-curve run: {len(tr)} training clips")
    print(json.dumps(summary))

    os.makedirs(a.out, exist_ok=True)
    model, loss = build(K, a.arch, a.units, a.dropout, a.smoothing, a.hand_local)
    opt = tf.keras.optimizers.AdamW(a.lr, weight_decay=a.weight_decay, clipnorm=1.0) if a.weight_decay else tf.keras.optimizers.Adam(a.lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    best = os.path.join(a.out, "model.keras")
    t0 = time.time()
    limit = a.max_minutes * 60

    def schedule(epoch, logs=None):
        frac = epoch / a.epochs if a.epochs else min(1.0, (time.time() - t0) / limit)
        opt.learning_rate.assign(max(a.lr * 0.5 * (1 + math.cos(math.pi * frac)), 1e-5))

    def stop_on_time(epoch, logs=None):
        if time.time() - t0 > limit:
            print(f"time budget of {a.max_minutes} minutes used, stopping")
            model.stop_training = True

    hist = model.fit(
        Batches(X[tr], y[tr], K, a.batch, a.seed, use_mirror=not a.no_mirror, warp=not a.no_warp, drop=not a.no_drop, geometry=not a.no_geometry),
        validation_data=(X[va], tf.keras.utils.to_categorical(y[va], K)), validation_batch_size=256,
        epochs=a.max_epochs, verbose=2,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(best, monitor="val_accuracy", mode="max", save_best_only=True),
                   tf.keras.callbacks.LambdaCallback(on_epoch_begin=schedule, on_epoch_end=stop_on_time)],
    )
    seconds = time.time() - t0
    model = tf.keras.models.load_model(best)
    probs = model.predict(X[va], batch_size=256, verbose=0)
    val = {"top1": float((probs.argmax(1) == y[va]).mean()), "top5": topk(probs, y[va], min(5, K)), "n": int(len(va))}
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    best_epoch = int(np.argmax(h["val_accuracy"])) + 1
    print(f"\nbest epoch {best_epoch}/{len(h['val_loss'])}  VALIDATION top-1 {val['top1']:.4f}  top-5 {val['top5']:.4f}  ({seconds:.0f}s)")
    np.save(os.path.join(a.out, "val_probs.npy"), probs.astype(np.float32))
    json.dump({"labels": labels}, open(os.path.join(a.out, "labels.json"), "w"))
    json.dump(h, open(os.path.join(a.out, "history.json"), "w"))
    json.dump(val, open(os.path.join(a.out, "val_report.json"), "w"), indent=2)
    json.dump({
        "model": a.arch, "units": a.units, "dropout": a.dropout, "smoothing": a.smoothing, "lr": a.lr, "batch": a.batch, "weight_decay": a.weight_decay, "hand_local": a.hand_local, "train_fraction": a.train_fraction, "train_clips_used": int(len(tr)), "mirror": not a.no_mirror, "warp": not a.no_warp, "drop": not a.no_drop, "geometry": not a.no_geometry,
        "seed": a.seed, "split_policy": "official", "data": os.path.basename(a.data), "vocab": a.vocab, "select": "val_accuracy", "reduce_lr": "cosine over time budget",
        "patience": 0, "aug": "fresh every epoch", "best_epoch": best_epoch, "epochs_run": len(h["val_loss"]), "train_seconds": round(seconds, 1), "split_summary": summary,
        "split_fingerprint": fingerprint(d["videos"], d["splits"]), "params": int(model.count_params()),
    }, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print("saved run to", a.out)


if __name__ == "__main__":
    main()
