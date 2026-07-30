#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer 3, CoT variant: build a "reason step by step, then emit SPARQL"
training set, mirroring the existing DSL CoT fine-tune's teaching format
exactly.

Design: the DSL CoT text (Task Analysis / Objective / entity descriptions /
Step-by-Step Analysis / Path Completion / Final Path) explains WHICH
relations to follow and WHY -- that reasoning is entirely formalism-
independent, identical whether the final answer is expressed as a DSL
<PATH>...</PATH> or a SPARQL query. So rather than writing new reasoning
text, this script takes the existing CoT prose UNCHANGED and replaces both
occurrences of the (byte-identical) <PATH>...</PATH> block with the
compiled SPARQL query -- same content taught, only the terminal answer
formalism differs. No new tag is needed: the SPARQL scorer already
extracts "SELECT DISTINCT ?ans WHERE {...}" by regex regardless of
surrounding prose (see baselines/score_sparql_predictions.py /
sparql_baseline.extract_sparql), so this substitution is directly
scoreable.

Run:
    python3 baselines/build_sparql_cot_training_data.py \
        --in-dir datasets/TEMPLATE_DISJOINT/CoT_TRAIN \
        --out-dir datasets/TEMPLATE_DISJOINT/SPARQL_COT_TRAIN
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan import evaluate as ET  # noqa: E402
from titan import graph as GA  # noqa: E402
from baselines.path_to_sparql import compile_path, load_rel_types  # noqa: E402


def substitute_sparql(cot_text: str, sparql_query: str) -> str:
    """Replace every <PATH>...</PATH> occurrence with the compiled SPARQL
    (both occurrences carry identical DSL content in the source format, so
    both get the same replacement)."""
    return re.sub(r"<PATH>.*?</PATH>", lambda _m: sparql_query, cot_text,
                 flags=re.DOTALL)


def compile_split(df: pd.DataFrame, rel_types: dict, node_type_fn, parse_select_fn) -> pd.DataFrame:
    rows = []
    n_fail = 0
    for r in df.itertuples():
        steps = ET.extract_path_steps(r.Path)
        entities = ET.extract_entities(r.Path)
        if steps is None:
            n_fail += 1
            continue
        try:
            q = compile_path(steps, entities, rel_types, node_type_fn, parse_select_fn)
        except Exception:
            n_fail += 1
            continue
        new_text = substitute_sparql(r.Path, q)
        rows.append({"Question": r.Question, "Path": new_text, "Section": r.Section})
    print(f"[INFO] compiled {len(rows)}/{len(df)} rows ({n_fail} failed)")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in-dir", required=True,
                    help="CoT_TRAIN-style directory (Question, Path=full CoT prose "
                         "ending in <PATH>...</PATH>, Section)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--rel-types", default="rel_target_types.json")
    args = ap.parse_args()

    g = GA.load_graph(args.graph)
    names = ET._node_names(g)
    rel_types = load_rel_types(args.rel_types)
    node_type_fn = lambda n: GA._get_node_type(g, n)  # noqa: E731
    parse_select_fn = lambda s: ET.parse_select_args(s, names)  # noqa: E731

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        path = os.path.join(args.in_dir, f"{split}_dataset.csv")
        df = pd.read_csv(path)
        print(f"[INFO] {split}: {len(df)} input rows from {path}")
        out = compile_split(df, rel_types, node_type_fn, parse_select_fn)
        out_path = os.path.join(args.out_dir, f"{split}_dataset.csv")
        out.to_csv(out_path, index=False)
        print(f"[OK] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
