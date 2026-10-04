"""Convert WLASL videos into a Signo dataset JSON (same format the browser exports/imports).

Usage:
  python extract_wlasl.py --wlasl-json WLASL_v0.3.json --videos ./videos --top-n 100 --out wlasl100.json

Notes:
  * Glosses are ranked by number of instances and the top-N are kept. This approximates the
    official WLASL100/300 subsets (which are defined by nslt_N.json); for exact benchmark numbers
    swap in the official split files.
  * Many WLASL source videos are no longer downloadable; missing files are skipped and counted.
  * Each sample keeps the official "split" (train/val/test) so train.py can evaluate properly.
"""
import argparse
import json
import os

import cv2
import mediapipe as mp
from tqdm import tqdm

from common import T, D, extract_features, trim, to_seq


def video_to_frames(path, holistic, start=1, end=-1):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i < start or (end > 0 and i > end):
            continue
        res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ex = extract_features(res)
        if ex is not None:
            frames.append(ex)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl-json", required=True)
    ap.add_argument("--videos", required=True, help="folder containing <video_id>.mp4")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--out", default="wlasl_dataset.json")
    a = ap.parse_args()

    meta = json.load(open(a.wlasl_json))
    meta.sort(key=lambda g: len(g["instances"]), reverse=True)
    meta = meta[: a.top_n]

    samples, missing, empty = [], 0, 0
    hol = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=False)
    for g in tqdm(meta, desc="glosses"):
        for inst in g["instances"]:
            path = os.path.join(a.videos, f"{inst['video_id']}.mp4")
            if not os.path.exists(path):
                missing += 1
                continue
            frames = video_to_frames(path, hol, inst.get("frame_start", 1), inst.get("frame_end", -1))
            tr = trim(frames)
            if tr is None:
                empty += 1
                continue
            seq = to_seq(tr)
            samples.append({
                "label": g["gloss"].lower(),
                "source": "wlasl",
                "split": inst.get("split", "train"),
                "data": [[round(float(v), 4) for v in row] for row in seq],
            })
    hol.close()
    out = {"version": 1, "seqLen": T, "featureDim": D, "labels": [g["gloss"].lower() for g in meta], "samples": samples}
    json.dump(out, open(a.out, "w"))
    print(f"saved {len(samples)} samples to {a.out}  (missing videos: {missing}, no hands found: {empty})")


if __name__ == "__main__":
    main()
