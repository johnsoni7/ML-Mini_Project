"""Download ONLY the selected ASL Citizen videos out of the 45.9 GB archive, using HTTP Range requests.

A zip file keeps an index at its end. remotezip reads just that index, then asks the server for the
exact bytes of each video we want, so the rest of the archive is never transferred.

Usage:
  python fetch_asl_citizen.py --vocab vocab_asl_citizen.txt --meta ~/signo_data/asl_citizen/meta/ASL_Citizen \
      --out ~/signo_data/asl_citizen/videos --manifest ~/signo_data/asl_citizen/manifest.csv
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from remotezip import RemoteZip

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
ZIP_PREFIX = "ASL_Citizen/videos/"


def open_zip():
    # Some download servers reject "last N bytes" requests (HTTP 501), so always ask for explicit byte positions.
    return RemoteZip(URL, support_suffix_range=False)


def read_vocab(path):
    words = []
    for line in open(path):
        if not line.startswith("#"):
            words += line.split()
    assert len(words) == len(set(words)), "duplicate word in vocabulary"
    return words


def read_split_rows(meta_dir, vocab):
    """Official split CSVs -> rows for the chosen glosses. The split is kept exactly as Microsoft defined it."""
    rows = []
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(os.path.join(meta_dir, "splits", split + ".csv"))):
            if r["Gloss"] in vocab:
                rows.append({"video": r["Video file"], "gloss": r["Gloss"], "split": split, "participant": r["Participant ID"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--meta", required=True, help="folder holding splits/train.csv etc.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-mb", type=float, default=2500, help="refuse to download more than this")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-per-class", type=int, default=0, help="keep only N official test clips per sign, picked by a hash of the file name (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N rows (for a quick test)")
    a = ap.parse_args()

    vocab = read_vocab(a.vocab)
    rows = read_split_rows(a.meta, set(vocab))
    missing_glosses = set(vocab) - {r["gloss"] for r in rows}
    assert not missing_glosses, f"not in dataset: {missing_glosses}"
    if a.test_per_class:
        kept, by_gloss = [], collections.defaultdict(list)
        for r in rows:
            if r["split"] == "test":
                by_gloss[r["gloss"]].append(r)
            else:
                kept.append(r)
        for g in sorted(by_gloss):  # the hash is unrelated to clip content, so this is an unbiased pick
            kept += sorted(by_gloss[g], key=lambda r: hashlib.sha256(r["video"].encode()).hexdigest())[: a.test_per_class]
        rows = kept
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    os.makedirs(a.out, exist_ok=True)
    with open(a.manifest + ".planned.csv", "w", newline="") as f:  # lets the extractor start while this still downloads
        w = csv.DictWriter(f, fieldnames=["video", "gloss", "split", "participant"])
        w.writeheader()
        w.writerows(rows)

    t0 = time.time()
    with open_zip() as z:  # raises RangeNotSupported rather than silently fetching the whole file
        sizes = {i.filename: i.compress_size for i in z.infolist()}
    want_mb = sum(sizes.get(ZIP_PREFIX + r["video"], 0) for r in rows) / 1e6
    print(f"{len(vocab)} glosses, {len(rows)} videos, {want_mb:.0f} MB to fetch")
    if want_mb > a.max_mb:
        raise SystemExit(f"refusing: {want_mb:.0f} MB is above --max-mb {a.max_mb}")

    local = threading.local()
    lock = threading.Lock()
    done = {"n": 0}

    def fetch(r):
        name = ZIP_PREFIX + r["video"]
        dest = os.path.join(a.out, os.path.basename(r["video"]))
        r["status"], r["bytes"] = "ok", 0
        if name not in sizes:
            r["status"] = "missing_from_zip"
            return r
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            r["bytes"] = 0  # already here from an earlier run
            r["status"] = "ok"
            return r
        for attempt in range(3):
            try:
                if not hasattr(local, "z"):
                    local.z = open_zip()
                data = local.z.read(name)  # zipfile verifies the CRC checksum while reading
                with open(dest + ".part", "wb") as f:
                    f.write(data)
                os.replace(dest + ".part", dest)
                r["bytes"] = sizes[name]
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
            done["n"] += 1
            if done["n"] % 100 == 0:
                print(f"  {done['n']} downloaded, {time.time() - t0:.0f}s", flush=True)
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(fetch, rows))

    with open(a.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "gloss", "split", "participant", "status", "bytes"])
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    stats = {
        "selected_classes": len(vocab),
        "selected_videos": len(rows),
        "ok_videos": len(ok),
        "failed": collections.Counter(r["status"] for r in rows if r["status"] != "ok"),
        "per_split": collections.Counter(r["split"] for r in ok),
        "mb_downloaded_this_run": round(sum(r["bytes"] for r in rows) / 1e6, 1),
        "mb_on_disk": round(sum(os.path.getsize(os.path.join(a.out, os.path.basename(r["video"]))) for r in ok) / 1e6, 1),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(stats, indent=2))
    json.dump(stats, open(a.manifest + ".stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
