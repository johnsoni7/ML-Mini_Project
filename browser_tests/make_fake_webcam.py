"""Stitch held-out clips, separated by empty frames, into an MJPEG file Chrome can use as a pretend webcam.

Usage: python make_fake_webcam.py <videos_dir> <manifest.csv> <out.mjpeg> <out.json> SIGN [SIGN ...]
Writes dataset frames, so the output must stay outside the repository and be deleted with the videos.
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

W, H, FPS, GAP_SECONDS = 640, 480, 30, 3


def fit(img):
    s = min(W / img.shape[1], H / img.shape[0])
    r = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((H, W, 3), 40, np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def main():
    videos, manifest, out_path, meta_path, *signs = sys.argv[1:]
    rows = [r for r in csv.DictReader(open(manifest)) if r["split"] == "test"]
    blank = cv2.imencode(".jpg", np.full((H, W, 3), 40, np.uint8))[1].tobytes()
    plan, frame = [], 0
    with open(out_path, "wb") as f:
        for _ in range(FPS * GAP_SECONDS):
            f.write(blank)
            frame += 1
        for sign in signs:
            row = next(r for r in rows if r["gloss"] == sign)
            cap = cv2.VideoCapture(os.path.join(videos, row["video"]))
            start = frame
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                f.write(cv2.imencode(".jpg", fit(img), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
                frame += 1
            cap.release()
            plan.append({"sign": sign.lower(), "start_s": start / FPS, "end_s": frame / FPS})
            for _ in range(FPS * GAP_SECONDS):
                f.write(blank)
                frame += 1
    json.dump({"plan": plan, "total_s": frame / FPS}, open(meta_path, "w"), indent=1)
    print(f"{len(plan)} clips, {frame / FPS:.1f}s, {os.path.getsize(out_path) / 1e6:.0f} MB -> {out_path}")


if __name__ == "__main__":
    main()
