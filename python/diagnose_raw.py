"""More diagnosis from the raw landmarks, validation clips only. Writes research/diagnose_raw.json."""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from features_raw import ROOT, build_clip

names = json.load(open("model_out/final_v2/labels.json"))["labels"]
y = np.load("model_out/final_v2/val_y.npy")
vids = np.load("model_out/final_v2/val_videos.npy").tolist()
p = np.load("model_out/final_v2/val_probs.npy")
ok = p.argmax(1) == y
rows = {r["video"]: r for r in csv.DictReader(open(f"{ROOT}/manifest_1000.csv"))}
rec = []
for i, v in enumerate(vids):
    f = f"{ROOT}/raw/{v[:-4]}.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    fl = z["flags"]
    n = len(fl)
    hands = np.where(fl[:, 1] | fl[:, 2])[0]
    lo, hi = max(0, hands[0] - 2), min(n, hands[-1] + 3)
    lh, rh = z["lh"][:, 0, :2] / 1e4, z["rh"][:, 0, :2] / 1e4  # wrists
    tips = np.concatenate([z["lh"][:, [4, 8, 12, 16, 20], :2], z["rh"][:, [4, 8, 12, 16, 20], :2]], 1) / 1e4
    both = fl[lo:hi, 1:3]
    # movement kept by 30-frame nearest sampling, measured on the more visible hand's fingertips
    use = (z["lh"] if both[:, 0].sum() > both[:, 1].sum() else z["rh"])[lo:hi, :, :2] / 1e4
    on = both[:, 0] if both[:, 0].sum() > both[:, 1].sum() else both[:, 1]
    m = hi - lo
    full = np.linalg.norm(np.diff(use[on], axis=0), axis=2).mean(1).sum() if on.sum() > 1 else 0
    idx = np.round(np.arange(30) * (m - 1) / 29).astype(int)
    idx = idx[on[idx]]
    kept = np.linalg.norm(np.diff(use[idx], axis=0), axis=2).mean(1).sum() if len(idx) > 1 else 0
    # how close the hand gets to the face, in shoulder widths, using lower face points
    sw = np.median(np.hypot(*(z["pose"][:, 11, :2] - z["pose"][:, 12, :2]).T)) / 1e4
    mouth = z["face"][:, 15, :2] / 1e4
    tipsdist = np.linalg.norm(tips - mouth[:, None], axis=2)
    lf = np.where(fl[:, 1])[0]
    rf = np.where(fl[:, 2])[0]
    d = []
    if len(lf):
        d.append(tipsdist[lf, :5].min(1))
    if len(rf):
        d.append(tipsdist[rf, 5:].min(1))
    near = float(np.min([np.percentile(x, 10) for x in d]) / max(sw, 1e-3)) if d else None
    rec.append({"ok": bool(ok[i]), "frames_raw": int(n), "frames_span": int(m), "move_kept": float(kept / full) if full > 0 else None, "near_face": near, "gap_share": float(1 - (both.any(1)).mean()), "signer": rows[v]["participant"], "face_frames": float(fl[lo:hi, 3].mean()), "both_hands": float(both.all(1).mean()), "y": int(y[i])})
R = {"clips": len(rec)}
a = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in rec], float)
o = a("ok").astype(bool)
span, kept, near = a("frames_span"), a("move_kept"), a("near_face")
R["span_frames_percentiles_5_50_95"] = np.percentile(span, [5, 50, 95]).tolist()
R["share_frames_dropped_by_30_sampling_median"] = float(np.median(1 - 30 / np.maximum(span, 30)))
R["movement_kept_median_p10"] = [float(np.nanmedian(kept)), float(np.nanpercentile(kept, 10))]
def by(v, edges):
    return {f"{a_:g}-{b_:g}": {"n": int(((v >= a_) & (v < b_)).sum()), "top1": float(o[(v >= a_) & (v < b_)].mean()) if ((v >= a_) & (v < b_)).any() else None} for a_, b_ in zip(edges[:-1], edges[1:])}
R["top1_by_span_frames"] = by(span, [0, 40, 60, 80, 110, 150, 1000])
R["top1_by_movement_kept"] = by(kept, [0, .6, .7, .8, .9, 2])
R["top1_by_fingertip_distance_to_mouth_in_shoulder_widths"] = by(near, [0, .2, .4, .6, .8, 1.2, 10])
R["top1_by_gap_share"] = by(a("gap_share"), [0, .05, .1, .2, .3, 1.01])
R["signers"] = {s: {"n": sum(r["signer"] == s for r in rec), "top1": float(np.mean([r["ok"] for r in rec if r["signer"] == s])), "gap_share": float(np.mean([r["gap_share"] for r in rec if r["signer"] == s])), "both_hands": float(np.mean([r["both_hands"] for r in rec if r["signer"] == s])), "face_frames": float(np.mean([r["face_frames"] for r in rec if r["signer"] == s]))} for s in sorted({r["signer"] for r in rec})}
# sign-level: signs that happen near the face
per = collections.defaultdict(list)
for r in rec:
    if r["near_face"] is not None:
        per[r["y"]].append(r)
near_sign = {k: np.median([r["near_face"] for r in v]) for k, v in per.items() if len(v) >= 3}
pr = p.argmax(1)
cf = collections.Counter()
for t, q in zip(y[~ok], pr[~ok]):
    if t in near_sign and q in near_sign:
        cf["both_near" if near_sign[t] < .5 and near_sign[q] < .5 else "neither_near" if near_sign[t] >= .5 and near_sign[q] >= .5 else "one_near"] += 1
R["errors_by_whether_true_and_predicted_signs_are_near_face"] = dict(cf)
R["signs_near_face_share"] = float(np.mean([v < .5 for v in near_sign.values()]))
print(json.dumps(R, indent=1))
json.dump(R, open("research/diagnose_raw.json", "w"), indent=1)
