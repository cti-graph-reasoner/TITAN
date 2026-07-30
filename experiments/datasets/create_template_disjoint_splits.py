#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create TEMPLATE-DISJOINT train/val/test splits for the TITAN dataset.

Motivation
----------
The original splitter (create_dataset_splits.py) splits at the *row* level:
every question template appears in both train and test (verified: 100% of
test rows share their template with train), so path prediction can be solved
by template memorization. This script fixes the protocol by splitting at the
*template* level.

Template identity
-----------------
A row's reasoning path depends only on its template, not on the entities
substituted into it. We therefore identify a template by its
    fine key    = Section || final <PATH> with `select` arguments stripped
                  (select args are entity names; everything else, including
                  `filter` conditions, is fixed by the template)
and additionally track the
    coarse key  = fine key with `filter` arguments also stripped
                  (the pure relation skeleton).

Output splits
-------------
    train.csv          - training rows
    val.csv            - validation rows (templates seen in train,
                         entity combinations unseen)
    test_iid.csv       - i.i.d. test: templates seen in train, rows unseen
                         (comparable to the original protocol)
    test_heldout.csv   - templates NEVER seen in train/val/test_iid.
                         Column `SkeletonUnseen` marks the harder subset
                         whose relation skeleton is also never seen.

Guarantees (asserted before writing):
    * no fine key of test_heldout occurs in train/val/test_iid
    * no exact Question string of any eval split occurs in train
    * within-split exact-duplicate questions are removed

A NoCoT view with identical rows is derived by replacing the CoT response
with its final <PATH>...</PATH> string, so CoT and NoCoT models are trained
and evaluated on exactly the same questions (the original CoT and NoCoT
datasets only shared ~10% of question strings).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Template keys
# ---------------------------------------------------------------------------

def extract_final_path(text: str) -> Optional[str]:
    """Return the content of the last <PATH>...</PATH> block, or None."""
    matches = re.findall(r"<PATH>(.*?)</PATH>", str(text), flags=re.DOTALL)
    return matches[-1] if matches else None


def normalize_path(path_str: str, strip_filter: bool) -> str:
    steps = path_str.split("<SEP>") if "<SEP>" in path_str else [path_str]
    out = []
    for s in steps:
        s = s.strip()
        if s.startswith("select"):
            out.append("select")
        elif strip_filter and s.startswith("filter"):
            out.append("filter")
        else:
            out.append(s)
    return "<SEP>".join(out)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    fp = df["Path"].map(extract_final_path)
    bad = fp.isna().sum()
    if bad:
        print(f"[WARN] Dropping {bad} rows without a parseable <PATH> block.")
    df = df.loc[fp.notna()].copy()
    df["FinalPath"] = fp[fp.notna()]
    df["TemplateKey"] = df["Section"].astype(str) + "||" + df["FinalPath"].map(
        lambda p: normalize_path(p, strip_filter=False))
    df["SkeletonKey"] = df["Section"].astype(str) + "||" + df["FinalPath"].map(
        lambda p: normalize_path(p, strip_filter=True))
    return df


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def _greedy_pick(groups: List[tuple], target: float, cap: float,
                 rng: random.Random) -> List[str]:
    """Pick group keys totalling ~target rows (never exceeding cap),
    leaving at least one group unpicked. groups: [(key, size), ...]."""
    groups = groups[:]
    rng.shuffle(groups)
    picked, mass = [], 0
    for key, size in groups:
        if len(picked) >= len(groups) - 1:
            break  # always leave >=1 group for train
        if mass >= target:
            break
        if mass + size <= cap:
            picked.append(key)
            mass += size
    if not picked and len(groups) > 1:
        # everything was too big for the cap: take the smallest group
        key, _ = min(groups, key=lambda g: g[1])
        picked.append(key)
    return picked


def assign_heldout_templates(
    df: pd.DataFrame,
    heldout_frac: float,
    seed: int,
) -> Dict[str, bool]:
    """
    Per section, choose template (fine) keys for the heldout test set.
    Preference order:
      1. entire coarse (skeleton) groups -> SkeletonUnseen = True
      2. top up with fine groups whose skeleton stays in train
         -> SkeletonUnseen = False
    Returns {fine_key: skeleton_unseen}.
    """
    rng = random.Random(seed)
    heldout: Dict[str, bool] = {}

    for section, sec_df in df.groupby("Section"):
        n = len(sec_df)
        target = heldout_frac * n
        cap = 1.5 * target

        coarse_sizes = sec_df.groupby("SkeletonKey").size()
        fine_sizes = sec_df.groupby("TemplateKey").size()
        fine_to_coarse = sec_df.drop_duplicates("TemplateKey").set_index(
            "TemplateKey")["SkeletonKey"].to_dict()

        picked_coarse: List[str] = []
        if len(coarse_sizes) >= 3:
            # aim roughly half the heldout mass at fully-unseen skeletons
            picked_coarse = _greedy_pick(
                list(coarse_sizes.items()), target / 2, cap / 2, rng)
        for fk, ck in fine_to_coarse.items():
            if ck in picked_coarse:
                heldout[fk] = True

        mass = sum(fine_sizes[fk] for fk in heldout
                   if fine_to_coarse.get(fk, "").startswith(section + "||"))
        remaining = [(fk, sz) for fk, sz in fine_sizes.items()
                     if fk not in heldout]
        if len(remaining) > 1 and mass < target:
            extra = _greedy_pick(remaining, target - mass, cap - mass, rng)
            for fk in extra:
                heldout[fk] = False

    return heldout


# ---------------------------------------------------------------------------
# Main split
# ---------------------------------------------------------------------------

def create_splits(
    input_csvs: List[str],
    out_dir: str,
    heldout_frac: float = 0.15,
    train_frac: float = 0.85,
    val_frac: float = 0.05,
    seed: int = 42,
) -> None:
    df = pd.concat([pd.read_csv(p) for p in input_csvs], ignore_index=True)
    for col in ("Question", "Path", "Section"):
        if col not in df.columns:
            raise ValueError(f"Input is missing required column '{col}'")
    df = add_keys(df)
    df = df.drop_duplicates(subset=["Question", "FinalPath"]).reset_index(drop=True)

    heldout_map = assign_heldout_templates(df, heldout_frac, seed)
    is_heldout = df["TemplateKey"].isin(heldout_map)
    heldout = df[is_heldout].copy()
    heldout["SkeletonUnseen"] = heldout["TemplateKey"].map(heldout_map)
    rest = df[~is_heldout].copy()

    # Row-level split of the remaining (seen-template) rows, per section.
    rng = random.Random(seed + 1)
    parts = {"train": [], "val": [], "test_iid": []}
    for _, sec_df in rest.groupby("Section"):
        idx = list(sec_df.index)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        parts["train"].append(sec_df.loc[idx[:n_train]])
        parts["val"].append(sec_df.loc[idx[n_train:n_train + n_val]])
        parts["test_iid"].append(sec_df.loc[idx[n_train + n_val:]])

    splits = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    splits["test_heldout"] = heldout.reset_index(drop=True)

    # Cross-split exact-question dedup: eval rows must not repeat a training question.
    train_q = set(splits["train"]["Question"])
    for name in ("val", "test_iid", "test_heldout"):
        s = splits[name]
        before = len(s)
        s = s[~s["Question"].isin(train_q)].drop_duplicates(subset=["Question"])
        removed = before - len(s)
        if removed:
            print(f"[INFO] {name}: removed {removed} rows duplicating a train question.")
        splits[name] = s.reset_index(drop=True)

    # ------------------------- hard guarantees -------------------------
    train_val_iid_keys = (set(splits["train"]["TemplateKey"])
                          | set(splits["val"]["TemplateKey"])
                          | set(splits["test_iid"]["TemplateKey"]))
    leak = set(splits["test_heldout"]["TemplateKey"]) & train_val_iid_keys
    assert not leak, f"Template leakage into heldout: {sorted(leak)[:5]}"
    for name in ("val", "test_iid", "test_heldout"):
        dup = set(splits[name]["Question"]) & train_q
        assert not dup, f"{name} shares {len(dup)} exact questions with train"
    skel_unseen = splits["test_heldout"]["SkeletonUnseen"]
    train_skel = set(splits["train"]["SkeletonKey"])
    bad_skel = set(splits["test_heldout"].loc[skel_unseen, "SkeletonKey"]) & train_skel
    assert not bad_skel, f"SkeletonUnseen rows share a skeleton with train: {sorted(bad_skel)[:5]}"

    # ------------------------------ save -------------------------------
    os.makedirs(os.path.join(out_dir, "CoT"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "NoCoT"), exist_ok=True)
    report = {}
    for name, s in splits.items():
        cot = s.copy()
        cot.to_csv(os.path.join(out_dir, "CoT", f"{name}.csv"), index=False)
        nocot = s.copy()
        nocot["Path"] = "<PATH>" + nocot["FinalPath"] + "</PATH>"
        nocot.to_csv(os.path.join(out_dir, "NoCoT", f"{name}.csv"), index=False)

        lengths = s["FinalPath"].map(lambda p: len(p.split("<SEP>")))
        report[name] = {
            "rows": len(s),
            "templates": int(s["TemplateKey"].nunique()),
            "sections": s["Section"].value_counts().to_dict(),
            "path_length": lengths.value_counts().sort_index().to_dict(),
            "operators": {
                op: int(s["FinalPath"].str.contains(pat, regex=True).sum())
                for op, pat in (("filter", r"filter "), ("select", r"select "),
                                ("exec_common", r"exec_common"),
                                ("exec_difference", r"exec_difference"))
            },
        }
        if name == "test_heldout":
            report[name]["skeleton_unseen_rows"] = int(s["SkeletonUnseen"].sum())

    with open(os.path.join(out_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sections"}
                      for k, v in report.items()}, indent=2))
    print(f"\n[OK] All disjointness guarantees hold. Splits written to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--inputs", nargs="+",
                   default=["experiments/datasets/CoT/train_dataset.csv",
                            "experiments/datasets/CoT/val_dataset.csv",
                            "experiments/datasets/CoT/test_dataset.csv"],
                   help="CoT CSVs to pool and re-split (need Question,Path,Section)")
    p.add_argument("--out", default="experiments/datasets/TEMPLATE_DISJOINT",
                   help="Output directory")
    p.add_argument("--heldout", type=float, default=0.15,
                   help="Target fraction of rows whose templates are held out")
    p.add_argument("--train", type=float, default=0.85,
                   help="Train fraction of the remaining (seen-template) rows")
    p.add_argument("--val", type=float, default=0.05,
                   help="Val fraction of the remaining rows")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    create_splits(a.inputs, a.out, a.heldout, a.train, a.val, a.seed)
