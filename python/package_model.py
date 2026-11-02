"""Combine trained models into one TensorFlow.js-ready model, optionally averaging each clip with its mirror image.

Everything stays inside one model, so index.html and its 30 x 100 features need no changes.
Usage: python package_model.py --out model_out/final --mirror-avg model_out/run_a model_out/run_b
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import shutil

import numpy as np
import tensorflow as tf

from common import T, D
from train_plus import fixed_dense, mirror

L = tf.keras.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mirror-avg", action="store_true")
    a = ap.parse_args()
    labels = [json.load(open(f"{r}/labels.json"))["labels"] for r in a.runs]
    if any(l != labels[0] for l in labels):
        raise SystemExit("the runs use different label orders")
    members = []
    for n, r in enumerate(a.runs):
        m = tf.keras.models.load_model(f"{r}/model.keras", compile=False)
        with tf.name_scope(f"m{n}"):
            c = tf.keras.models.clone_model(m, clone_function=lambda l: type(l).from_config({**l.get_config(), "name": f"m{n}_{l.name}"}))
        c.set_weights(m.get_weights())  # the same trained weights under unique names, which the TensorFlow.js export requires
        c._name = f"member_{n}"
        members.append(c)
    i = tf.keras.Input((T, D))
    inputs = [i]
    if a.mirror_avg:
        inputs.append(fixed_dense(D, mirror(np.eye(D, dtype=np.float32)[None])[0], name="mirror_flip")(i))
    outs = [m(x) for m in members for x in inputs]
    o = L.Average()(outs) if len(outs) > 1 else outs[0]
    model = tf.keras.Model(i, o)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    os.makedirs(a.out, exist_ok=True)
    model.save(f"{a.out}/model.keras")
    json.dump({"labels": labels[0]}, open(f"{a.out}/labels.json", "w"))
    for name in ("history.json", "config.json"):
        shutil.copy(f"{a.runs[0]}/{name}", f"{a.out}/{name}")
    c = json.load(open(f"{a.out}/config.json"))
    c.update({"members": [os.path.basename(r.rstrip("/")) for r in a.runs], "mirror_avg": a.mirror_avg, "params": int(model.count_params())})
    json.dump(c, open(f"{a.out}/config.json", "w"), indent=2)
    print(f"packaged {len(members)} model(s), mirror average {a.mirror_avg}, {len(outs)} passes per prediction, {model.count_params():,} parameters")


if __name__ == "__main__":
    main()
