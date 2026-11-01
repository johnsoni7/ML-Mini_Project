"""Write a small file of held-out clips + Keras predictions for browser_tests/heldout.js to compare against.

This is an integration check (does the browser agree with Python?). It plays no part in choosing a model.
The fixture holds features derived from the dataset, so it must stay outside the repository.

Usage:
  python make_browser_fixture.py --run model_out/bilstm_seed0 --data ~/signo_data/asl_citizen/signo_dataset.json \
      --out ~/signo_data/asl_citizen/browser_fixture.json --per-class 1
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=1)
    a = ap.parse_args()

    labels = json.load(open(os.path.join(a.run, "labels.json")))["labels"]
    model = tf.keras.models.load_model(os.path.join(a.run, "model.keras"))
    samples = [s for s in json.load(open(a.data))["samples"] if s["split"] == "test"]
    chosen, seen = [], {}
    for s in samples:  # dataset is sorted by video name, which is unrelated to how well a clip is recognised
        if seen.get(s["label"], 0) < a.per_class:
            seen[s["label"]] = seen.get(s["label"], 0) + 1
            chosen.append(s)
    probs = model.predict(np.array([s["data"] for s in chosen], np.float32), verbose=0)
    json.dump({"labels": labels, "clips": [{"video": s["video_id"], "label": s["label"], "seq": s["data"], "keras_probs": p.tolist()} for s, p in zip(chosen, probs)]}, open(a.out, "w"))
    print(f"wrote {len(chosen)} clips to {a.out}")


if __name__ == "__main__":
    main()
