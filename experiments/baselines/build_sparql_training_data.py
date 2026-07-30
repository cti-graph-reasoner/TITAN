#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer 3: compile the TITAN-DSL training/val/test splits to SPARQL targets,
producing a drop-in replacement dataset directory (Question, Path columns
-- "Path" here holds the compiled SPARQL query text) that train_qwen.py can
consume UNCHANGED, since it just reads whatever is in the Path column
verbatim as the SFT target. This gives a perfectly symmetric comparison to
the existing DSL fine-tune: same base model, same LoRA recipe, same
training questions -- only the target FORMALISM differs (SPARQL vs the
TITAN DSL).

Rows whose gold path fails to compile (rare; see path_to_sparql.py) are
dropped, matching the same executable-only philosophy already used for
DSL fine-tuning few-shot selection elsewhere in the pipeline.

Run:
    python3 baselines/build_sparql_training_data.py \
        --in-dir datasets/TEMPLATE_DISJOINT/NoCoT_TRAIN \
        --out-dir datasets/TEMPLATE_DISJOINT/SPARQL_TRAIN
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan import evaluate as ET  # noqa: E402
from titan import graph as GA  # noqa: E402
from baselines.path_to_sparql import compile_path, load_rel_types  # noqa: E402


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
        rows.append({"Question": r.Question, "Path": q, "Section": r.Section})
    print(f"[INFO] compiled {len(rows)}/{len(df)} rows ({n_fail} failed)")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in-dir", required=True,
                    help="directory with train_dataset.csv/val_dataset.csv/test_dataset.csv "
                         "(DSL Path column)")
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
