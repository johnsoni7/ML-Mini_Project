"""Train one experiment from a JSON config and record its validation results. Test clips are removed before training and never scored here.

Usage: python train_exp.py experiments/exp_001/config.json
Every architecture uses only layers that TensorFlow.js can load. Fixed maths lives inside the model as frozen layers, so 30 x 100 models need no browser changes.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_fscore_support

from train_plus import time_warp

L = tf.keras.layers
ROOT = os.path.expanduser("~/signo_data/asl_citizen")
HAND_BLOCKS = ((14, 56, 98), (56, 98, 99))  # start, end, flag


def face_pairs(n):
    """Left/right partner of each face point in features_raw.FACE_MODEL."""
    from features_raw import FACE_MODEL

    swap = {61: 291, 0: 0, 17: 17, 40: 270, 91: 321, 1: 1, 4: 4, 168: 168, 33: 263, 133: 362, 70: 300, 105: 334, 107: 336, 10: 10, 152: 152, 234: 454, 58: 288}
    swap.update({v: k for k, v in swap.items()})
    assert len(FACE_MODEL) == n
    return [FACE_MODEL.index(swap[i]) for i in FACE_MODEL]


def mirror_any(x):
    """Mirror a batch of any layout: flip x, swap paired pose points, the two hands, their flags and the paired face points."""
    o = x.copy()
    o[:, :, 0:98:2] = -o[:, :, 0:98:2]
    for a, b in ((1, 2), (3, 4), (5, 6)):
        ia, ib = [a * 2, a * 2 + 1], [b * 2, b * 2 + 1]
        o[:, :, ia], o[:, :, ib] = o[:, :, ib].copy(), o[:, :, ia].copy()
    left = o[:, :, 14:56].copy()
    o[:, :, 14:56] = o[:, :, 56:98]
    o[:, :, 56:98] = left
    o[:, :, [98, 99]] = o[:, :, [99, 98]]
    if x.shape[2] > 100:
        n = (x.shape[2] - 101) // 2
        f = o[:, :, 100:100 + 2 * n].reshape(len(x), x.shape[1], n, 2)
        f[:, :, :, 0] = -f[:, :, :, 0]
        o[:, :, 100:100 + 2 * n] = f[:, :, face_pairs(n)].reshape(len(x), x.shape[1], -1)
    return o


def drop_frames(x, rng, p, max_len):
    o = x.copy()
    T = x.shape[1]
    for i in np.where(rng.random(len(x)) < p)[0]:
        ln, s = rng.integers(1, max_len + 1), rng.integers(0, T)
        a, b, flag = HAND_BLOCKS[rng.integers(0, 2)]
        o[i, s:s + ln, a:b] = 0
        o[i, s:s + ln, flag] = 0
    return o


def vary_body(x, rng, strength):
    """Imitate a different signer: each clip gets its own arm lengths, and each hand its own size and turn around the wrist. Missing hands stay zero."""
    o = x.copy()
    n = len(x)
    for (a, b, flag), shoulder, elbow, wrist in zip(HAND_BLOCKS, (1, 2), (3, 4), (5, 6)):
        S = x[:, :, [shoulder * 2, shoulder * 2 + 1]]
        arm = rng.uniform(1 - 0.15 * strength, 1 + 0.15 * strength, n)[:, None, None]
        for j in (elbow, wrist):
            P = x[:, :, [j * 2, j * 2 + 1]]
            o[:, :, [j * 2, j * 2 + 1]] = S + (P - S) * arm
        H = x[:, :, a:b].reshape(n, x.shape[1], 21, 2)
        W = H[:, :, :1]
        s = rng.uniform(1 - 0.2 * strength, 1 + 0.2 * strength, n)[:, None, None, None]
        r = rng.uniform(-0.3 * strength, 0.3 * strength, n)
        cs, sn = np.cos(r)[:, None, None, None], np.sin(r)[:, None, None, None]
        d = H - W
        rx, ry = (d[..., 0:1] * cs - d[..., 1:2] * sn) * s, (d[..., 0:1] * sn + d[..., 1:2] * cs) * s
        moved = W + (W - S[:, :, None]) * (arm[..., None] - 1)  # the hand travels with the wrist
        out = np.concatenate([rx, ry], 3) + moved
        out = np.where(x[:, :, flag:flag + 1, None] > 0, out, 0)
        o[:, :, a:b] = out.reshape(n, x.shape[1], 42)
    return o


def augment(x, rng, c):
    n, T, D = x.shape
    if c.get("body_aug"):
        x = vary_body(x, rng, c["body_aug"])
    if c.get("mirror", True):
        m = rng.random(n) < 0.5
        if m.any():
            x = x.copy()
            x[m] = mirror_any(x[m])
    if c.get("warp", True):
        x = warp(x, rng)
    if c.get("drop", True):
        x = drop_frames(x, rng, c.get("drop_p", 0.3), c.get("drop_len", 4))
    if not c.get("geometry", True):
        return x.astype(np.float32)
    g = c.get("geo_strength", 1.0)
    s = rng.uniform(1 - 0.15 * g, 1 + 0.15 * g, n)[:, None, None]
    r = rng.uniform(-0.14 * g, 0.14 * g, n)
    cs, sn = np.cos(r)[:, None, None], np.sin(r)[:, None, None]
    dx, dy = rng.uniform(-0.15 * g, 0.15 * g, (2, n))[:, :, None, None]
    ys = rng.uniform(1 - c.get("yscale", 0.0), 1 + c.get("yscale", 0.0), n)[:, None, None]
    cols = [np.arange(0, 98, 2)]
    present = [np.ones((n, T, 7), bool), np.repeat(x[:, :, 98:99] > 0, 21, 2), np.repeat(x[:, :, 99:100] > 0, 21, 2)]
    if D > 100:
        f = (D - 101) // 2
        cols.append(np.arange(100, 100 + 2 * f, 2))
        present.append(np.repeat(x[:, :, -1:] > 0, f, 2))
    ci = np.concatenate(cols)
    pr = np.concatenate(present, 2)
    jx, jy = rng.normal(0, 0.012, (2, n, T, len(ci)))
    px, py = x[:, :, ci], x[:, :, ci + 1] * ys
    o = x.copy()
    o[:, :, ci] = np.where(pr, (px * cs - py * sn) * s + dx + jx, 0)
    o[:, :, ci + 1] = np.where(pr, (px * sn + py * cs) * s + dy + jy, 0)
    return o.astype(np.float32)


def warp(x, rng):
    if x.shape[1] == 30:
        return time_warp(x, rng)
    n, T = len(x), x.shape[1]
    keep = rng.uniform(0.8, 1.0, n)
    start = rng.uniform(0, 1, n) * (1 - keep)
    pos = np.cumsum(rng.gamma(4.0, 1.0, (n, T)), 1)
    pos = (pos - pos[:, :1]) / np.maximum(pos[:, -1:] - pos[:, :1], 1e-6)
    idx = np.rint((start[:, None] + keep[:, None] * pos) * (T - 1)).astype(int).clip(0, T - 1)
    return np.take_along_axis(x, idx[:, :, None], 1)


class Batches(tf.keras.utils.Sequence):
    def __init__(self, X, y, K, c, seed):
        self.X, self.y, self.K, self.c = X, y, K, c
        self.batch = c.get("batch", 64)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))

    def __len__(self):
        return len(self.X) // self.batch

    def __getitem__(self, i):
        ii = self.order[i * self.batch:(i + 1) * self.batch]
        x = augment(self.X[ii], self.rng, self.c)
        t = tf.keras.utils.to_categorical(self.y[ii], self.K)
        mp = self.c.get("mixup", 0.0)
        if mp and self.rng.random() < mp:
            lam = self.rng.beta(0.4, 0.4)
            j = self.rng.permutation(len(x))
            x, t = lam * x + (1 - lam) * x[j], lam * t + (1 - lam) * t[j]
        return x, t

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.X))


def velocity_layer(D):
    """Frozen layer giving each number's change since the previous frame (first frame's change is zero input padding)."""
    k = np.zeros((2, D, D), np.float32)
    k[1] = np.eye(D)
    k[0] = -np.eye(D)
    return L.Conv1D(D, 2, padding="causal", use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Constant(k), name="velocity")


def res_conv(x, ch, k, d, drop):
    y = L.Conv1D(ch, k, padding="same", dilation_rate=d)(x)
    y = L.LayerNormalization()(y)
    y = L.Activation("relu")(y)
    y = L.Dropout(drop)(y)
    y = L.Conv1D(ch, k, padding="same", dilation_rate=d)(y)
    y = L.LayerNormalization()(y)
    y = L.Activation("relu")(y)
    y = L.Dropout(drop)(y)
    return L.Add()([x, y])


def res_conv2(x, ch, k, d, drop):
    """Normalise first, so values stay in range however many blocks are stacked."""
    y = L.LayerNormalization()(x)
    y = L.Conv1D(ch, k, padding="same", dilation_rate=d, activation="relu")(y)
    y = L.Dropout(drop)(y)
    y = L.Conv1D(ch, k, padding="same", dilation_rate=d)(y)
    y = L.Dropout(drop)(y)
    return L.Add()([x, y])


def self_attention(x, dim, heads, drop):
    """Multi-head self-attention from Dense and Dot layers, which TensorFlow.js loads directly."""
    outs = []
    d = dim // heads
    for _ in range(heads):
        q, k, v = L.Dense(d, use_bias=False)(x), L.Dense(d, use_bias=False)(x), L.Dense(d, use_bias=False)(x)
        a = L.Activation("softmax")(L.Dot(axes=(2, 2))([q, k]))
        outs.append(L.Dot(axes=(2, 1))([L.Dropout(drop)(a), v]))
    y = L.Concatenate()(outs) if heads > 1 else outs[0]
    y = L.Dense(dim)(y)
    y = L.Dropout(drop)(y)
    return L.LayerNormalization()(L.Add()([x, y]))


def attention_pool(x):
    a = L.Activation("softmax")(L.Permute((2, 1))(L.Dense(1, use_bias=False)(x)))
    return L.Flatten()(L.Dot(axes=(2, 1))([a, x]))


def part_branches(i, D, units):
    """Separate learned summaries of the body, each hand, and the face, joined each frame."""
    idx = {"pose": list(range(14)), "left": list(range(14, 56)) + [98], "right": list(range(56, 98)) + [99]}
    if D > 100:
        idx["face"] = list(range(100, D))
    out = []
    for name, ii in idx.items():
        k = np.eye(D, len(ii), dtype=np.float32) * 0
        for n, j in enumerate(ii):
            k[j, n] = 1
        s = L.Dense(len(ii), use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Constant(k), name=f"pick_{name}")(i)
        s = L.Dense(units // 2 if name in ("pose", "face") else units, activation="relu")(s)
        out.append(s)
    return L.Concatenate()(out)


def build(c, T, D, K):
    i = tf.keras.Input((T, D))
    arch, u, dr = c["arch"], c.get("units", 128), c.get("dropout", 0.4)
    x = i
    if c.get("velocity"):
        x = L.Concatenate()([i, velocity_layer(D)(i)])
    if c.get("parts"):
        x = part_branches(x if not c.get("velocity") else i, D, u // 2)
    if c.get("input_dropout"):
        x = L.Dropout(c["input_dropout"])(x)
    for n in c.get("embed", []):  # learns relationships between points within each frame before looking along time
        x = L.Dense(n, activation="relu")(x)
    if c.get("embed"):
        x = L.Dropout(dr * 0.25)(x)
    if arch == "bilstm":
        x = L.Bidirectional(L.LSTM(u))(x)
    elif arch == "gru":
        x = L.Bidirectional(L.GRU(u))(x)
    elif arch == "bilstm2":
        x = L.Bidirectional(L.LSTM(u, return_sequences=True))(x)
        x = L.Dropout(dr * 0.5)(x)
        x = L.Bidirectional(L.LSTM(u))(x)
    elif arch == "attn_bilstm":
        x = L.Bidirectional(L.LSTM(u, return_sequences=True))(x)
        x = L.Concatenate()([attention_pool(x), L.GlobalMaxPooling1D()(x)])
    elif arch == "cnn_bilstm":
        x = L.Dense(u * 2)(x)
        x = res_conv(x, u * 2, 5, 1, dr * 0.25)
        x = L.Bidirectional(L.LSTM(u))(x)
    elif arch == "tcn":
        x = L.Dense(u * 2)(x)
        block = res_conv2 if c.get("prenorm") else res_conv
        for d in c.get("dilations", [1, 2, 4, 1, 2, 4]):
            x = block(x, u * 2, c.get("kernel", 5), d, dr * 0.25)
        x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
        if c.get("prenorm"):
            x = L.LayerNormalization()(x)
    elif arch == "cnn_attn":
        x = L.Dense(u * 2)(x)
        block = res_conv2 if c.get("prenorm") else res_conv
        for d in c.get("dilations", [1, 2, 4]):
            x = block(x, u * 2, c.get("kernel", 5), d, dr * 0.25)
        for _ in range(c.get("blocks", 2)):
            x = self_attention(x, u * 2, c.get("heads", 4), dr * 0.25)
        x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
        if c.get("prenorm"):
            x = L.LayerNormalization()(x)
    else:
        raise SystemExit(f"unknown arch {arch}")
    x = L.Dropout(dr)(x)
    x = L.Dense(c.get("head", u), activation="relu")(x)
    x = L.Dropout(dr)(x)
    return tf.keras.Model(i, L.Dense(K, activation="softmax")(x))


def load(c):
    fs = c.get("features", "base")
    path = f"{ROOT}/ds1000.npz" if fs == "base" else f"{ROOT}/fs/{fs}.npz"
    z = np.load(path)
    labels, splits = np.char.lower(z["label"]), z["split"]
    keep = splits != "test"  # test clips never enter a training process
    return z["X"][keep], labels[keep], splits[keep], z["signer"][keep], z["video"][keep]


def scores(p, y, K):
    t = p.argmax(1)
    P, R, F, S = precision_recall_fscore_support(y, t, labels=range(K), zero_division=0)
    m = S > 0
    conf = p.max(1)
    bins = np.linspace(0, 1, 16)
    ece = sum(abs((t[g] == y[g]).mean() - conf[g].mean()) * g.mean() for g in [(conf >= a) & (conf < b) for a, b in zip(bins[:-1], bins[1:])] if g.any())
    return {"top1": float((t == y).mean()), "top5": float((np.argsort(p, 1)[:, -5:] == y[:, None]).any(1).mean()), "macro_precision": float(P[m].mean()), "macro_recall": float(R[m].mean()), "macro_f1": float(F[m].mean()), "weighted_f1": float(f1_score(y, t, average="weighted", zero_division=0)), "ece": float(ece), "classes_recall_le_50": int((R[m] <= 0.5).sum()), "classes_recall_0": int((R[m] == 0).sum())}


def main():
    cfg_path = sys.argv[1]
    c = json.load(open(cfg_path))
    out = os.path.dirname(cfg_path)
    seed = c.get("seed", 0)
    th = c.get("threads", 0)
    if th and c["arch"] in ("tcn", "cnn_attn"):  # convolutions use extra threads well, unlike LSTMs
        th = max(th, 3)
    if th:
        tf.config.threading.set_intra_op_parallelism_threads(th)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    X, labels, splits, signers, videos = load(c)
    names = json.load(open("model_out/final_v2/labels.json"))["labels"]
    assert names == sorted(set(labels.tolist())), "classes differ from the deployed model's classes"
    K, T, D = len(names), X.shape[1], X.shape[2]
    index = {l: i for i, l in enumerate(names)}
    y = np.array([index[l] for l in labels])
    tr, va = np.where(splits == "train")[0], np.where(splits == "val")[0]
    if c.get("train_limit"):  # same clips in every run with this limit, so partial-data comparisons stay fair
        tr = np.array(sorted(tr, key=lambda i: videos[i]))[:c["train_limit"]]
    assert not set(signers[tr]) & set(signers[va]), "a signer is in both training and validation"
    if c.get("final_fit"):  # pre-specified last step: same settings on training plus validation clips, keeping the last epoch
        tr = np.concatenate([tr, va])
        print(f"final fit on {len(tr)} clips from {len(set(signers[tr]))} signers", flush=True)
    model = build(c, T, D, K)
    lr, wd, epochs, warm = c.get("lr", 2e-3), c.get("weight_decay", 0.0), c.get("epochs", 100), c.get("warmup", 0)
    opt = tf.keras.optimizers.AdamW(lr, weight_decay=wd, clipnorm=1.0) if wd else tf.keras.optimizers.legacy.Adam(lr, clipnorm=1.0)  # the legacy Adam is the fast one on Apple chips
    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=c.get("smoothing", 0.0)), metrics=["accuracy"])
    best = os.path.join(out, "model.keras")
    t0 = time.time()
    limit = c.get("max_minutes", 180) * 60

    def schedule(e, logs=None):
        v = lr * (e + 1) / warm if e < warm else lr * 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm)))
        opt.learning_rate.assign(max(v, 1e-5))

    def on_end(e, logs=None):
        print(json.dumps({"epoch": e + 1, "of": epochs, "val_top1": round(logs["val_accuracy"], 4), "s": round(time.time() - t0)}), flush=True)
        if time.time() - t0 > limit:
            model.stop_training = True

    if c.get("final_fit"):
        def on_end_fit(e, logs=None):
            print(json.dumps({"epoch": e + 1, "of": epochs, "loss": round(logs["loss"], 4), "s": round(time.time() - t0)}), flush=True)
            if time.time() - t0 > limit:
                model.stop_training = True

        hist = model.fit(Batches(X[tr], y[tr], K, c, seed), epochs=epochs, verbose=0, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_begin=schedule, on_epoch_end=on_end_fit)])
        model.save(best)
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, open(os.path.join(out, "history.json"), "w"))
        json.dump({"labels": names}, open(os.path.join(out, "labels.json"), "w"))
        json.dump({"final_fit": True, "n_train": int(len(tr)), "epochs_run": len(hist.history["loss"]), "train_seconds": round(time.time() - t0), "params": int(model.count_params())}, open(os.path.join(out, "metrics.json"), "w"), indent=1)
        print("FINISHED final fit", flush=True)
        return
    hist = model.fit(Batches(X[tr], y[tr], K, c, seed), validation_data=(X[va], tf.keras.utils.to_categorical(y[va], K)), validation_batch_size=256, epochs=epochs, verbose=0,
                     callbacks=[tf.keras.callbacks.ModelCheckpoint(best, monitor="val_accuracy", mode="max", save_best_only=True), tf.keras.callbacks.LambdaCallback(on_epoch_begin=schedule, on_epoch_end=on_end)])
    seconds = time.time() - t0
    model = tf.keras.models.load_model(best, compile=False)
    p = model.predict(X[va], batch_size=256, verbose=0)
    np.save(os.path.join(out, "val_probs.npy"), p.astype(np.float32))
    np.save(os.path.join(out, "val_videos.npy"), videos[va])
    np.save(os.path.join(out, "val_y.npy"), y[va])
    sample = np.random.RandomState(0).permutation(tr)[:3000]
    train_clean = float((model.predict(X[sample], batch_size=256, verbose=0).argmax(1) == y[sample]).mean())
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    m = scores(p, y[va], K)
    tp = p.argmax(1) == y[va]
    m.update({"n_val": int(len(va)), "n_train": int(len(tr)), "train_top1_clean": train_clean, "gap": train_clean - m["top1"], "best_epoch": int(np.argmax(h["val_accuracy"])) + 1, "epochs_run": len(h["val_loss"]), "best_val_loss": float(min(h["val_loss"])), "train_seconds": round(seconds), "params": int(model.count_params()), "T": int(T), "D": int(D), "model_mb": round(os.path.getsize(best) / 1e6, 2), "by_signer": {s: float(tp[signers[va] == s].mean()) for s in sorted(set(signers[va].tolist()))}})
    json.dump(h, open(os.path.join(out, "history.json"), "w"))
    json.dump({"labels": names}, open(os.path.join(out, "labels.json"), "w"))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("FINISHED", json.dumps({k: m[k] for k in ("top1", "top5", "macro_f1", "gap", "best_epoch", "params", "train_seconds")}), flush=True)


if __name__ == "__main__":
    main()
