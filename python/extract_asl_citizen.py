"""Turn the downloaded ASL Citizen videos into a Signo dataset JSON (the same format the browser exports).

Each video becomes a (30, 100) array using the frozen spec in common.py. Every clip's result, including
failures, is cached on disk so an interrupted run resumes where it stopped. Nothing is silently discarded:
each failure is counted by reason in the report.

Usage:
  python extract_asl_citizen.py --manifest ~/signo_data/asl_citizen/manifest.csv --videos ~/signo_data/asl_citizen/videos \
      --cache ~/signo_data/asl_citizen/features --out ~/signo_data/asl_citizen/signo_dataset.json \
      --report results/extraction_report.json --workers 6
"""
import argparse
import collections
import csv
import json
import multiprocessing as mp_proc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np

from common import T, D, trim, to_seq


def process_one(job):
    """Runs in a worker process. Returns the row plus status: ok | missing | corrupt | no_pose | no_hands | invalid."""
    row, videos, cache, stride = job
    from extract_wlasl import new_holistic, video_to_frames  # imported here so each worker loads MediaPipe once

    stem = os.path.splitext(row["video"])[0]
    cached = os.path.join(cache, stem + ".npz")
    if os.path.exists(cached):
        z = np.load(cached, allow_pickle=False)
        return {**row, "status": str(z["status"]), "seconds": 0.0, "frames": int(z["frames"]), "from_cache": True}

    path = os.path.join(videos, row["video"])
    t0 = time.time()
    seq, n = None, 0
    if not os.path.exists(path):
        status = "missing"
    else:
        try:
            with new_holistic() as hol:
                frames = video_to_frames(path, hol, stride=stride)
            n = len(frames)
            status = "ok"
            if n == 0:
                import cv2

                cap = cv2.VideoCapture(path)
                readable = cap.isOpened() and cap.read()[0]
                cap.release()
                status = "no_pose" if readable else "corrupt"
            else:
                tr = trim(frames)
                if tr is None:
                    status = "no_hands"
                else:
                    seq = to_seq(tr)
                    if seq.shape != (T, D) or not np.isfinite(seq).all():
                        status, seq = "invalid", None
        except Exception as e:
            print("error on", row["video"], repr(e), flush=True)
            status = "corrupt"
    tmp = cached + ".tmp.npz"
    np.savez_compressed(tmp, seq=seq if seq is not None else np.zeros((0, 0), np.float32), status=status, frames=n)
    os.replace(tmp, cached)
    return {**row, "status": status, "seconds": time.time() - t0, "frames": n, "from_cache": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (only if extraction is too slow)")
    ap.add_argument("--limit", type=int, default=0, help="only process N clips (benchmark)")
    ap.add_argument("--only-existing", action="store_true", help="skip rows whose video is not on disk yet (benchmark)")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.manifest))] if os.path.exists(a.manifest) else None
    if rows is None:  # benchmark before the manifest exists: use whatever videos are already on disk
        rows = [{"video": f, "gloss": "?", "split": "?", "participant": "?"} for f in sorted(os.listdir(a.videos)) if f.endswith(".mp4")]
    if a.only_existing:
        rows = [r for r in rows if os.path.exists(os.path.join(a.videos, r["video"]))]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.cache, exist_ok=True)

    t0 = time.time()
    jobs = [(r, a.videos, a.cache, a.stride) for r in rows]
    results = []
    with mp_proc.get_context("spawn").Pool(a.workers, maxtasksperchild=50) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, jobs), 1):
            results.append(res)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} clips, {time.time() - t0:.0f}s, status so far {dict(collections.Counter(x['status'] for x in results))}", flush=True)
    elapsed = time.time() - t0

    fresh = [r for r in results if not r["from_cache"] and r["seconds"] > 0]
    ok = [r for r in results if r["status"] == "ok"]
    report = {
        "selected_videos": len(rows),
        "status_counts": dict(collections.Counter(r["status"] for r in results)),
        "successfully_extracted": len(ok),
        "per_split_ok": dict(collections.Counter(r["split"] for r in ok)),
        "per_class_ok": dict(sorted(collections.Counter(r["gloss"] for r in ok).items())),
        "per_class_split_ok": {g: dict(collections.Counter(r["split"] for r in ok if r["gloss"] == g)) for g in sorted({r["gloss"] for r in ok})},
        "workers": a.workers,
        "stride": a.stride,
        "wall_seconds": round(elapsed, 1),
        "mean_seconds_per_clip_in_worker": round(float(np.mean([r["seconds"] for r in fresh])), 2) if fresh else None,
        "mean_pose_frames_per_clip": round(float(np.mean([r["frames"] for r in results])), 1) if results else None,
    }
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("per_class")}, indent=2))
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        json.dump(report, open(a.report, "w"), indent=2)

    if a.out:
        samples = []
        for r in sorted(ok, key=lambda x: x["video"]):
            seq = np.load(os.path.join(a.cache, os.path.splitext(r["video"])[0] + ".npz"), allow_pickle=False)["seq"]
            samples.append({
                "label": r["gloss"].lower(),
                "source": "asl_citizen",
                "split": r["split"],
                "signer": r["participant"],
                "video_id": r["video"],
                "data": [[round(float(v), 4) for v in row] for row in seq],
            })
        labels = sorted({s["label"] for s in samples})
        json.dump({"version": 1, "seqLen": T, "featureDim": D, "labels": labels, "samples": samples}, open(a.out, "w"))
        print(f"saved {len(samples)} sequences to {a.out}")


if __name__ == "__main__":
    main()
