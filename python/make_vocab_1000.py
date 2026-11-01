"""Choose the larger vocabulary from the metadata only, before any model is trained on it.

Rules: no numbered variants (EAT1/EAT2), enough clips in each official split, always keep the 50 demo signs,
then take the most common English words (wordfreq). Accuracy plays no part in the choice.

Usage: python make_vocab_1000.py --meta ~/signo_data/asl_citizen/meta/ASL_Citizen --keep vocab_asl_citizen.txt --n 1000 --out vocab_asl_citizen_1000.txt
"""
import argparse
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wordfreq import zipf_frequency

from fetch_asl_citizen import read_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    counts = {s: collections.Counter(r["Gloss"] for r in csv.DictReader(open(os.path.join(a.meta, "splits", s + ".csv")))) for s in ("train", "val", "test")}
    usable = [g for g in counts["train"] if not re.search(r"\d$", g) and counts["train"][g] >= 10 and counts["val"][g] >= 2 and counts["test"][g] >= 5]
    keep = read_vocab(a.keep)
    rest = sorted((g for g in usable if g not in keep), key=lambda g: (-zipf_frequency(g.lower(), "en"), g))
    chosen = sorted(keep + rest[: a.n - len(keep)])
    with open(a.out, "w") as f:
        f.write(f"# {len(chosen)} ASL Citizen glosses chosen by make_vocab_1000.py before any training. Not selected by accuracy.\n")
        for i in range(0, len(chosen), 10):
            f.write(" ".join(chosen[i:i + 10]) + "\n")
    print(f"{len(usable)} usable glosses, wrote {len(chosen)} to {a.out}")


if __name__ == "__main__":
    main()
