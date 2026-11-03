"""Build experimental feature sets from the raw landmarks saved by extract_raw.py. common.py and the 30 x 100 pipeline stay as they are.

Layout of every set: pose 7 x 2, left hand 21 x 2, right hand 21 x 2, left and right flags (the same first 100 numbers as common.py),
then, if asked for, face points F x 2 and a face flag. Options are documented on build_clip.

Usage: python features_raw.py --name face_aspect --face --aspect [--interp 4] [--T 30] [--linear] [--splits train,val]
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from extract_raw import FACE_IDX

ROOT = os.path.expanduser("~/signo_data/asl_citizen")
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
# A smaller face set for the model: mouth, nose, eyes, eyebrows, chin, forehead, cheeks. These mark where on the face a sign happens.
FACE_MODEL = [61, 291, 0, 17, 40, 270, 91, 321, 1, 4, 168, 33, 133, 263, 362, 70, 105, 107, 300, 334, 336, 10, 152, 234, 454, 58, 288]
FACE_SEL = [FACE_IDX.index(i) for i in FACE_MODEL]


def fill_gaps(xy, on, max_gap):
    """Join a hand across short tracking losses in a straight line. Before the first and after the last detection stays missing."""
    idx = np.where(on)[0]
    on = on.copy()
    xy = xy.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a)[:, None, None] / (b - a)
            xy[a + 1:b] = xy[a] * (1 - w) + xy[b] * w
            on[a + 1:b] = True
    return xy, on


def build_clip(z, T=30, face=False, aspect=False, interp=0, linear=False):
    """Raw clip -> (T, D) array, or None when it has fewer than 5 hand frames, as in common.trim.
    aspect: use true shapes, with 4:3 as the reference, so 4:3 clips are unchanged and wide clips stop being distorted.
    interp: bridge hand gaps of up to this many frames. linear: blend neighbouring frames when resampling instead of taking the nearest."""
    fl = z["flags"]
    w, h = z["size"][:2]
    pose, lh, rh, fc = [z[k][:, :, :2].astype(np.float32) / 10000 for k in ("pose", "lh", "rh", "face")]
    if aspect and h > 0:
        k = (w / h) / (4 / 3)
        pose, lh, rh, fc = pose * [k, 1], lh * [k, 1], rh * [k, 1], fc * [k, 1]
    a, b = pose[:, 11], pose[:, 12]
    sc = np.hypot(*(a - b).T)
    keep = fl[:, 0] & (sc >= 0.03)
    if not keep.any():
        return None
    c = ((a + b) / 2)[:, None]
    s = sc[:, None, None] + 1e-9
    n = lambda p: ((p - c) / s)[keep]
    P, L, R, F = n(pose[:, POSE_IDX]), n(lh), n(rh), n(fc[:, FACE_SEL])
    fL, fR, fF = fl[keep, 1], fl[keep, 2], fl[keep, 3]
    hands = np.where(fL | fR)[0]
    if len(hands) < 5:
        return None
    lo, hi = max(0, hands[0] - 2), min(len(P), hands[-1] + 3)
    P, L, R, F, fL, fR, fF = P[lo:hi], L[lo:hi], R[lo:hi], F[lo:hi], fL[lo:hi], fR[lo:hi], fF[lo:hi]
    if interp:
        L, fL = fill_gaps(L, fL, interp)
        R, fR = fill_gaps(R, fR, interp)
    L, R, F = L * fL[:, None, None], R * fR[:, None, None], F * fF[:, None, None]  # missing points are exactly zero
    parts = [P.reshape(len(P), -1), L.reshape(len(P), -1), R.reshape(len(P), -1), fL[:, None], fR[:, None]]
    if face:
        parts += [F.reshape(len(P), -1), fF[:, None]]
    f = np.concatenate(parts, 1, dtype=np.float32)
    m = len(f)
    pos = np.arange(T) * (m - 1) / (T - 1)
    if not linear:
        return f[np.round(pos).astype(int)]  # same rounding as common.to_seq on these values
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, m - 1)
    g = (pos - i0)[:, None]
    o = f[i0] * (1 - g) + f[i1] * g
    # a blend with a missing hand would pull the hand to the origin, so frames next to a gap use the nearest frame
    near = f[np.round(pos).astype(int)]
    for flag, blk in ((98, (14, 56)), (99, (56, 98))):
        bad = (f[i0, flag] == 0) | (f[i1, flag] == 0)
        o[bad, blk[0]:blk[1]] = near[bad, blk[0]:blk[1]]
        o[:, flag] = near[:, flag]
    if face:
        o[:, -1] = near[:, -1]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--face", action="store_true")
    ap.add_argument("--aspect", action="store_true")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--interp", type=int, default=0)
    ap.add_argument("--splits", default="train,val")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(f"{ROOT}/manifest_1000.csv")) if r["split"] in a.splits.split(",")]
    X, keep, missing = [], [], 0
    for r in rows:
        p = f"{ROOT}/raw/{os.path.splitext(r['video'])[0]}.npz"
        if not os.path.exists(p):
            missing += 1
            continue
        s = build_clip(np.load(p), a.T, a.face, a.aspect, a.interp, a.linear)
        if s is not None:
            X.append(s)
            keep.append(r)
    out = f"{ROOT}/fs/{a.name}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=np.array(X, np.float32), label=np.array([r["gloss"].lower() for r in keep]), split=np.array([r["split"] for r in keep]), signer=np.array([r["participant"] for r in keep]), video=np.array([r["video"] for r in keep]), T=a.T, face=a.face)
    print(json.dumps({"name": a.name, "clips": len(X), "not_extracted_yet": missing, "dropped": len(rows) - missing - len(X), "shape": list(np.array(X).shape)}))


if __name__ == "__main__":
    main()
