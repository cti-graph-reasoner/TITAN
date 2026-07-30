#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Equivalence validation for the path->SPARQL compiler (Layer 1 gate).

For every gold-executable row of a split: execute the reference path with
the native graph executor (titan.evaluate.execute_path — the function that
defines the gold answer sets) AND with the compiled SPARQL over the RDF
conversion, then compare the two answer sets exactly.

High match rate => the RDF world + compiler are faithful, so the SPARQL
baseline (Layer 2) measures query-language difficulty, not substrate
artifacts. Mismatches are dumped with full context for iteration; whatever
remains unmatched gets documented in the paper as constructs our SPARQL
fragment does not capture.

Run:
    python3 baselines/validate_sparql_equivalence.py \
        --data datasets/TEMPLATE_DISJOINT/CoT/test_heldout.annotated.csv \
        --sample 300        # then rerun without --sample for the full pass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
from rdflib import Graph as RDFGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan import evaluate as ET  # noqa: E402
from titan import graph as GA  # noqa: E402
from baselines.path_to_sparql import (  # noqa: E402
    compile_path, load_rel_types, uri_to_name)


KNOWN_BUG1_RELATIONS = {"x_mitre_impact_type", "x_mitre_tactic_type"}


def has_known_executor_bug(steps: list) -> bool:
    """True if this path hits the pre-existing execute_path "_type"-substring
    collision bug (x_mitre_impact_type / x_mitre_tactic_type used as a
    non-seed relation mid-path -- these are MITRE attribute names that
    happen to contain '_type', colliding with the type-seed heuristic).
    Rows affected by this bug have an unreliable GOLD label (not a compiler
    defect), so they are reported separately rather than counted against
    the compiler."""
    return any(s in KNOWN_BUG1_RELATIONS for s in steps[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="experiments/datasets/TEMPLATE_DISJOINT/CoT/test_heldout.annotated.csv")
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--nt", default="titan_graph.nt")
    ap.add_argument("--rel-types", default="rel_target_types.json")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/baselines/sparql_equivalence")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "GoldExecutable" in df.columns:
        df = df[df["GoldExecutable"] == True]  # noqa: E712
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"[INFO] validating {len(df)} gold-executable rows")

    print("[INFO] loading graphs ...")
    g = GA.load_graph(args.graph)
    names = ET._node_names(g)
    rdf = RDFGraph()
    rdf.parse(args.nt, format="nt")
    rel_types = load_rel_types(args.rel_types)
    print(f"[INFO] nx: {g.number_of_nodes()} nodes | rdf: {len(rdf)} triples")

    node_type_fn = lambda n: GA._get_node_type(g, n)  # noqa: E731
    parse_select_fn = lambda s: ET.parse_select_args(s, names)  # noqa: E731

    rows, n_match, n_sparql_err = [], 0, 0
    t0 = time.time()
    for i, r in df.iterrows():
        steps = ET.extract_path_steps(r["Path"])
        entities = ET.extract_entities(r["Path"])
        gold = ET.execute_path(g, entities, steps, names)

        rec = {"Question": r["Question"], "Section": r.get("Section"),
               "path": "<SEP>".join(steps), "n_gold": len(gold),
               "known_gold_bug": has_known_executor_bug(steps)}
        try:
            q = compile_path(steps, entities, rel_types, node_type_fn, parse_select_fn)
            res = {uri_to_name(str(b[0])) for b in rdf.query(q)}
            rec["n_sparql"] = len(res)
            rec["match"] = res == gold
            if not rec["match"]:
                rec["only_gold"] = sorted(gold - res)[:5]
                rec["only_sparql"] = sorted(res - gold)[:5]
        except Exception as e:
            n_sparql_err += 1
            rec["n_sparql"] = -1
            rec["match"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        n_match += bool(rec["match"])
        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(df)}] match={n_match/(i+1):.1%} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(f"{args.out}_per_example.csv", index=False)
    clean = res_df[~res_df["known_gold_bug"]]
    report = {
        "n": len(res_df),
        "match_rate_raw": float(res_df["match"].mean()),
        "n_known_gold_bug_rows": int(res_df["known_gold_bug"].sum()),
        "match_rate_excl_known_gold_bugs": float(clean["match"].mean()) if len(clean) else None,
        "sparql_errors": n_sparql_err,
        "by_section": res_df.groupby("Section")["match"].agg(["mean", "count"])
                            .round(3).to_dict("index"),
    }
    with open(f"{args.out}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"\n=== EQUIVALENCE (raw): {n_match}/{len(res_df)} = {report['match_rate_raw']:.2%} "
          f"(sparql errors: {n_sparql_err}) ===")
    print(f"=== EQUIVALENCE excl. {report['n_known_gold_bug_rows']} known-gold-bug rows: "
          f"{report['match_rate_excl_known_gold_bugs']:.2%} ===")
    for sec, d in sorted(report["by_section"].items()):
        print(f"  {sec:<32} {d['mean']:.2%}  (n={d['count']})")
    print(f"[OK] wrote {args.out}_report.json / _per_example.csv")


if __name__ == "__main__":
    main()
