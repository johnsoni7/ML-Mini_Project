"""Convert a run from train.py into the TensorFlow.js files the browser loads, and check them.

Run with the separate converter environment:
  TF_USE_LEGACY_KERAS=1 .venv-tfjs/bin/python python/export_tfjs.py --run model_out/bilstm_seed0 --out app_model

Writes model.json + *.bin + labels.json. Stale files in --out are removed first so old weights cannot linger.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

from common import T, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="app_model")
    ap.add_argument("--report", default="results/report.json", help="evaluate.py output, shown in the app's Brain tab")
    ap.add_argument("--test-note", default="Official ASL Citizen test split", help="how the test clips were chosen, shown in the app's Brain tab")
    a = ap.parse_args()

    model = tf.keras.models.load_model(os.path.join(a.run, "model.keras"))
    labels = json.load(open(os.path.join(a.run, "labels.json")))["labels"]
    if tuple(model.input_shape[1:]) != (T, D):
        raise SystemExit(f"model input {model.input_shape} is not (None, {T}, {D})")
    if model.output_shape[-1] != len(labels):
        raise SystemExit(f"model has {model.output_shape[-1]} outputs but {len(labels)} labels")
    if not np.isfinite(model.predict(np.zeros((1, T, D), np.float32), verbose=0)).all():
        raise SystemExit("model produces non-finite predictions")

    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "*.bin")) + glob.glob(os.path.join(a.out, "model.json")):
        os.remove(f)
    tfjs.converters.save_keras_model(model, a.out)
    json.dump({"labels": labels}, open(os.path.join(a.out, "labels.json"), "w"))
    if os.path.exists(a.report):
        r = json.load(open(a.report))
        json.dump({
            "usedVal": True, "n": r["counts"]["test"], "top1": r["test_top1"], "top5": r["test_top5"],
            "note": f"{a.test_note}: {r['counts']['test']} clips from {r['signers']['test']} signers this model never saw.",
            "rows": [{"label": x["sign"], "precision": x["precision"], "recall": x["recall"], "support": x["test_clips"]} for x in r["per_class"]],
        }, open(os.path.join(a.out, "report.json"), "w"))
    j = json.load(open(os.path.join(a.out, "model.json")))
    shape = j["modelTopology"]["model_config"]["config"]["layers"][0]["config"]["batch_input_shape"]
    print(f"exported to {a.out}: input {shape}, {len(labels)} labels, files {sorted(os.listdir(a.out))}")


if __name__ == "__main__":
    main()
