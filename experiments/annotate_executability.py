#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate every split row with gold-path executability.

For each row, executes the reference path from the reference starting
entities on the TITAN KG and records:
    GoldExecutable  - True if the final answer set is non-empty
    NumGoldAnswers  - size of the final answer set

Rows where GoldExecutable is False are questions whose gold answer is empty
(usually because the template was instantiated with an entity that lacks the
required edge/attribute). Downstream, train/evaluate either on the full split
or on the validated subset (GoldExecutable == True) — both variants stay
available because this script only adds columns.

Writes <name>.csv -> <name>.annotated.csv next to the originals, plus an
executability_report.json summary.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import pandas as pd

from titan import evaluate as ET
from titan import graph as GA

_G = None
_NAMES = None


def _init(graph_file: str):
    global _G, _NAMES
    _G = GA.load_graph(graph_file)
    _NAMES = ET._node_names(_G)


def _work(arg):
    idx, path_text = arg
    steps = ET.extract_path_steps(path_text)
    if steps is None:
        return idx, False, 0
    entities = ET.extract_entities(path_text)
    try:
        final = ET.execute_path(_G, entities, steps, _NAMES)
    except Exception:
        return idx, False, 0
    return idx, len(final) > 0, len(final)


def annotate(split_dir: str, graph_file: str, workers: int) -> None:
    report = {}
    names = [f for f in ("train", "val", "test_iid", "test_heldout")
             if os.path.exists(os.path.join(split_dir, f + ".csv"))]
    for name in names:
        path = os.path.join(split_dir, name + ".csv")
        df = pd.read_csv(path)
        jobs = list(df["Path"].items())
        with Pool(workers, initializer=_init, initargs=(graph_file,)) as pool:
            results = pool.map(_work, jobs, chunksize=64)
        execu = pd.Series(False, index=df.index)
        nans = pd.Series(0, index=df.index)
        for idx, ok, n in results:
            execu[idx] = ok
            nans[idx] = n
        df["GoldExecutable"] = execu
        df["NumGoldAnswers"] = nans
        out = os.path.join(split_dir, name + ".annotated.csv")
        df.to_csv(out, index=False)

        stats = {
            "rows": len(df),
            "executable": int(df["GoldExecutable"].sum()),
            "executable_rate": float(df["GoldExecutable"].mean()),
            "by_section": df.groupby("Section")["GoldExecutable"].mean().round(3).to_dict(),
        }
        report[name] = stats
        print(f"[{name}] {stats['executable']}/{stats['rows']} executable "
              f"({stats['executable_rate']:.1%}) -> {out}")

    with open(os.path.join(split_dir, "executability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Report: {os.path.join(split_dir, 'executability_report.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--splits", default="experiments/datasets/TEMPLATE_DISJOINT/CoT")
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    annotate(args.splits, args.graph, args.workers)
