"""Download the official test clips for our 1,000 signs that have never been used, so the final score comes from unseen clips.

The earlier evaluations used a sample of about 4 test clips per sign. Signers and split assignments are Microsoft's. Nothing is scored here.
Usage: python fetch_fresh_test.py --out ~/signo_data/asl_citizen/fresh_test_videos --manifest ~/signo_data/asl_citizen/manifest_fresh_test.csv
"""
import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fetch_asl_citizen import ZIP_PREFIX, open_zip

ROOT = os.path.expanduser("~/signo_data/asl_citizen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-mb", type=float, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    used = list(csv.DictReader(open(f"{ROOT}/manifest_1000.csv")))
    vocab, seen = {r["gloss"] for r in used}, {r["video"] for r in used}
    rows = [{"video": r["Video file"], "gloss": r["Gloss"], "split": "test", "participant": r["Participant ID"]} for r in csv.DictReader(open(f"{ROOT}/meta/ASL_Citizen/splits/test.csv")) if r["Gloss"] in vocab and r["Video file"] not in seen]
    os.makedirs(a.out, exist_ok=True)
    with open_zip() as z:
        sizes = {i.filename: i.compress_size for i in z.infolist()}
    mb = sum(sizes.get(ZIP_PREFIX + r["video"], 0) for r in rows) / 1e6
    print(f"{len(rows)} clips, {mb:.0f} MB", flush=True)
    if mb > a.max_mb:
        raise SystemExit(f"refusing: {mb:.0f} MB is above --max-mb {a.max_mb}")
    local, lock, done, t0 = threading.local(), threading.Lock(), [0], time.time()

    def fetch(r):
        dest = os.path.join(a.out, r["video"])
        r["status"] = "ok"
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return r
        for _ in range(3):
            try:
                if not hasattr(local, "z"):
                    local.z = open_zip()
                data = local.z.read(ZIP_PREFIX + r["video"])
                with open(dest + ".part", "wb") as f:
                    f.write(data)
                os.replace(dest + ".part", dest)
                break
            except Exception as e:
                r["status"] = f"error:{type(e).__name__}"
                if hasattr(local, "z"):
                    del local.z
                time.sleep(2)
        else:
            return r
        r["status"] = "ok"
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"{done[0]} downloaded, {time.time() - t0:.0f}s", flush=True)
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(fetch, rows))
    with open(a.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "gloss", "split", "participant", "status"])
        w.writeheader()
        w.writerows(rows)
    print("DONE", sum(r["status"] == "ok" for r in rows), "ok of", len(rows), flush=True)


if __name__ == "__main__":
    main()
