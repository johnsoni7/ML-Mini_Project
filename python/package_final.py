"""Join chosen runs into one TensorFlow.js-ready model, with each member's calibration built in by rescaling its last layer. Input stays 30 x 100.

Usage: python package_final.py --out model_out/final_v4_val --dirs experiments/exp_017 ... --temps 0.78 ...
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import tensorflow as tf

from common import T, D

L = tf.keras.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--temps", nargs="+", type=float, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    assert len(a.dirs) == len(a.temps)
    labels = [json.load(open(f"{d}/labels.json"))["labels"] for d in a.dirs]
    assert all(l == labels[0] for l in labels), "the runs use different label orders"
    members = []
    for n, (d, t) in enumerate(zip(a.dirs, a.temps)):
        m = tf.keras.models.load_model(f"{d}/model.keras", compile=False)


        def renamed(l):
            cfg = {**l.get_config(), "name": f"m{n}_{l.name}"}
            if not l.trainable and "kernel_initializer" in cfg:  # TensorFlow.js cannot read an array here. The real weights are set below
                cfg["kernel_initializer"] = "zeros"
            if "velocity" in l.name:  # TensorFlow.js gets causal padding's output length wrong. Width 3 with same padding and a zero last step gives identical numbers
                cfg.update(kernel_size=(3,), padding="same")
            return type(l).from_config(cfg)

        c = tf.keras.models.clone_model(m, clone_function=renamed)
        for old, new in zip(m.layers, c.layers):
            w = old.get_weights()
            if "velocity" in old.name:
                w = [np.concatenate([w[0], np.zeros((1,) + w[0].shape[1:], w[0].dtype)], 0)]
            if w:
                new.set_weights(w)
        w = c.layers[-1].get_weights()
        c.layers[-1].set_weights([w[0] / t, w[1] / t])  # dividing the last layer's kernel and bias by t gives softmax(logits / t)
        c._name = f"member_{n}"
        members.append(c)
    i = tf.keras.Input((T, D))
    outs = [m(i) for m in members]
    model = tf.keras.Model(i, L.Average()(outs) if len(outs) > 1 else outs[0])
    os.makedirs(a.out, exist_ok=True)
    model.save(f"{a.out}/model.keras")
    json.dump({"labels": labels[0]}, open(f"{a.out}/labels.json", "w"))
    json.dump({"members": a.dirs, "temperatures": a.temps, "params": int(model.count_params()), "note": a.note, "member_configs": [json.load(open(f"{d}/config.json")) for d in a.dirs]}, open(f"{a.out}/config.json", "w"), indent=1)
    print(f"packaged {len(outs)} model(s), {model.count_params():,} parameters")


if __name__ == "__main__":
    main()
