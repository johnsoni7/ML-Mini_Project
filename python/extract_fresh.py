"""Save raw landmarks for the never-used test clips, slowly, alongside training. Nothing is predicted or scored.

Usage: python extract_fresh.py <shard> <shards>
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_raw import process_one

ROOT = os.path.expanduser("~/signo_data/asl_citizen")


def main():
    i, n = int(sys.argv[1]), int(sys.argv[2])
    rows = [r for r in csv.DictReader(open(f"{ROOT}/manifest_fresh_test.csv")) if r["status"] == "ok"]
    cache = f"{ROOT}/raw_fresh"
    os.makedirs(cache, exist_ok=True)
    for k, r in enumerate(rows):
        if k % n == i and not os.path.exists(f"{cache}/{os.path.splitext(r['video'])[0]}.npz"):
            process_one((r, f"{ROOT}/fresh_test_videos", cache))


if __name__ == "__main__":
    main()
