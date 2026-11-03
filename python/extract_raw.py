"""Keep every frame's raw MediaPipe landmarks for experiments, so new feature ideas never need the videos again.

The 30 x 100 pipeline in common.py is not touched. Holistic runs with the same settings as the browser and extract_wlasl.py.
Each clip is cached on disk, so an interrupted run resumes where it stopped.

Coordinates are stored as int16 (value x 10000), still in MediaPipe's 0-1 image units. Missing points are 0, with a flag per frame.

Usage:
  python extract_raw.py --manifest ~/signo_data/asl_citizen/manifest_1000.csv --videos ~/signo_data/asl_citizen/videos \
      --cache ~/signo_data/asl_citizen/raw --workers 7
"""
import argparse
import csv
import multiprocessing as mp_proc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np

# Face points that mark where a hand is relative to the face: lips, nose, eyes, eyebrows and the outline of the face.
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
NOSE = [1, 2, 4, 5, 98, 327, 168]
EYES = [33, 133, 159, 145, 263, 362, 386, 374]
BROWS = [70, 105, 107, 300, 334, 336]
OUTLINE = [10, 152, 234, 454, 127, 356, 172, 397, 58, 288]
FACE_IDX = LIPS + NOSE + EYES + BROWS + OUTLINE


def pts(lm, idx=None):
    """Landmark list -> int16 array (n, 3)."""
    use = lm.landmark if idx is None else [lm.landmark[i] for i in idx]
    return np.round(np.array([(p.x, p.y, p.z) for p in use], np.float32).clip(-3, 3) * 10000).astype(np.int16)


def process_one(job):
    row, videos, cache = job
    import cv2

    cv2.setNumThreads(1)
    from extract_wlasl import new_holistic

    stem = os.path.splitext(row["video"])[0]
    out = os.path.join(cache, stem + ".npz")
    cap = cv2.VideoCapture(os.path.join(videos, row["video"]))
    w, h, fps = cap.get(3), cap.get(4), cap.get(5)
    pose, lh, rh, face, flags = [], [], [], [], []
    empty = {"pose": np.zeros((33, 3), np.int16), "hand": np.zeros((21, 3), np.int16), "face": np.zeros((len(FACE_IDX), 3), np.int16)}
    with new_holistic() as hol:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if img.shape[1] > 640:  # same resize as video_to_frames
                img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])), interpolation=cv2.INTER_AREA)
            r = hol.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pose.append(pts(r.pose_landmarks) if r.pose_landmarks else empty["pose"])
            lh.append(pts(r.left_hand_landmarks) if r.left_hand_landmarks else empty["hand"])
            rh.append(pts(r.right_hand_landmarks) if r.right_hand_landmarks else empty["hand"])
            face.append(pts(r.face_landmarks, FACE_IDX) if r.face_landmarks else empty["face"])
            flags.append((r.pose_landmarks is not None, r.left_hand_landmarks is not None, r.right_hand_landmarks is not None, r.face_landmarks is not None))
    cap.release()
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, pose=np.array(pose, np.int16), lh=np.array(lh, np.int16), rh=np.array(rh, np.int16), face=np.array(face, np.int16), flags=np.array(flags, bool), size=np.array([w, h, fps], np.float32))
    os.replace(tmp, out)
    return len(pose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    # Validation first, so early comparisons use every validation clip. Then training clips, shuffled so a partial run is a fair sample. Test clips last.
    np.random.RandomState(0).shuffle(rows)
    rows.sort(key=lambda r: {"val": 0, "train": 1, "test": 2}[r["split"]])
    jobs = [(r, a.videos, a.cache) for r in rows if not os.path.exists(os.path.join(a.cache, os.path.splitext(r["video"])[0] + ".npz"))]
    print(f"{len(rows) - len(jobs)} cached, {len(jobs)} to process", flush=True)
    t0, frames = time.time(), 0
    with mp_proc.get_context("spawn").Pool(a.workers, maxtasksperchild=100) as pool:
        for i, n in enumerate(pool.imap_unordered(process_one, jobs), 1):
            frames += n
            if i % 500 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} clips, {frames} frames, {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
