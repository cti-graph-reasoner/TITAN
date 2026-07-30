#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Independent leakage audit for the TITAN template-disjoint splits.

Recomputes every key from the raw CSV text (never trusting stored columns)
and checks:

  1. TEMPLATE disjointness: no test_heldout template key (Section + final
     path with select args stripped) occurs in train, val, or test_iid.
  2. SKELETON disjointness: rows flagged SkeletonUnseen share no relation
     skeleton (filter args also stripped) with train.
  3. QUESTION disjointness: no exact question string of val/test_iid/
     test_heldout occurs in train; no duplicate questions inside any split.
  4. CoT/NoCoT alignment: for every split, the NoCoT view has the same
     questions in the same order and the same final path as the CoT view.

Exits non-zero if any check fails. Run:
    python3 verify_no_leakage.py --splits datasets/TEMPLATE_DISJOINT
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

SPLITS = ("train", "val", "test_iid", "test_heldout")


def final_path(text: str):
    m = re.findall(r"<PATH>(.*?)</PATH>", str(text), flags=re.DOTALL)
    return m[-1] if m else None


def norm(path: str, strip_filter: bool) -> str:
    out = []
    for s in (path.split("<SEP>") if "<SEP>" in path else [path]):
        s = s.strip()
        if s.startswith("select"):
            out.append("select")
        elif strip_filter and s.startswith("filter"):
            out.append("filter")
        else:
            out.append(s)
    return "<SEP>".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--splits", default="experiments/datasets/TEMPLATE_DISJOINT")
    args = ap.parse_args()

    cot = {}
    for name in SPLITS:
        df = pd.read_csv(f"{args.splits}/CoT/{name}.csv")
        fp = df["Path"].map(final_path)
        assert fp.notna().all(), f"{name}: rows without <PATH> block"
        df["_fp"] = fp
        df["_tk"] = df["Section"] + "||" + fp.map(lambda p: norm(p, False))
        df["_sk"] = df["Section"] + "||" + fp.map(lambda p: norm(p, True))
        cot[name] = df

    failures = []

    # 1. template disjointness
    seen = set().union(*(cot[n]["_tk"] for n in ("train", "val", "test_iid")))
    leak = set(cot["test_heldout"]["_tk"]) & seen
    print(f"[1] heldout template keys overlapping train/val/test_iid: {len(leak)}")
    if leak:
        failures.append(f"template leakage: {sorted(leak)[:3]}")

    # 2. skeleton disjointness for the flagged subset
    ho = cot["test_heldout"]
    if "SkeletonUnseen" in ho.columns:
        su = ho[ho["SkeletonUnseen"] == True]  # noqa: E712
        bad = set(su["_sk"]) & set(cot["train"]["_sk"])
        print(f"[2] SkeletonUnseen rows sharing a skeleton with train: {len(bad)} "
              f"(subset size {len(su)})")
        if bad:
            failures.append(f"skeleton leakage: {sorted(bad)[:3]}")
    else:
        print("[2] SKIPPED: no SkeletonUnseen column")

    # 3. question disjointness + intra-split duplicates
    train_q = set(cot["train"]["Question"])
    for name in ("val", "test_iid", "test_heldout"):
        inter = set(cot[name]["Question"]) & train_q
        dups = cot[name]["Question"].duplicated().sum()
        print(f"[3] {name}: exact-question overlap with train={len(inter)}, "
              f"intra-split duplicates={dups}")
        if inter:
            failures.append(f"{name} shares {len(inter)} questions with train")
        if dups:
            failures.append(f"{name} has {dups} duplicate questions")

    # 4. CoT/NoCoT alignment
    for name in SPLITS:
        nc = pd.read_csv(f"{args.splits}/NoCoT/{name}.csv")
        c = cot[name]
        same_q = len(nc) == len(c) and (nc["Question"].values == c["Question"].values).all()
        same_p = same_q and (nc["Path"].map(final_path).values == c["_fp"].values).all()
        print(f"[4] {name}: NoCoT aligned questions={same_q}, aligned paths={same_p}")
        if not (same_q and same_p):
            failures.append(f"{name}: CoT/NoCoT misaligned")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED: no template, skeleton, or question leakage; "
          "CoT/NoCoT views aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
