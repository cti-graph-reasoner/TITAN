#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end evaluation harness for TITAN.

Computes, per example and aggregated (with bootstrap CIs):
  * Path Accuracy      - exact match of normalized step sequences
  * Well-formedness    - generated text contains a parseable <PATH>...</PATH>
  * Executability      - the path executes to a non-empty node set on the KG
  * Entity-level P/R/F1 - final node set of the predicted path vs. the final
                          node set of the reference path (the actual answer),
                          using the reference starting entities for both
                          executions (oracle entity linking), so the metric
                          isolates path quality from entity-resolution noise.

Buckets: path length (L1/L2/L3/L4+), operator (filter/select/exec_common/
exec_difference/plain), Section, and SkeletonUnseen when present.

Modes
-----
1. Gold self-consistency (no --pred): executes the REFERENCE paths and reports
   dataset validity (what fraction of gold paths execute to a non-empty
   answer). Sanity requirement: entity-F1 = 1.0 on every executable row.
2. Prediction scoring (--pred): scores generated outputs against references.
   Accepted prediction formats:
     * JSON list of {"question": ..., "generated_path": ...} (bare steps or
       <PATH>-tagged), joined to --data rows by question string, or
     * CSV with columns Question, Generated.

Execution semantics follow titan/graph.py (same primitives are reused);
unlike follow_graph_n_entities, execute_path() also *returns* the combined
node set produced by exec_common / exec_difference steps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import reduce
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from titan import graph as GA  # noqa: E402


# ---------------------------------------------------------------------------
# Parsing: paths and entities out of model / reference text
# ---------------------------------------------------------------------------

def extract_path_steps(text: str) -> Optional[List[str]]:
    """Steps of the last <PATH>...</PATH> block; bare 'a<SEP>b' also accepted.
    Tolerates malformed openers some LLMs emit, e.g. '(PATH>' or '[PATH]'."""
    s = str(text)
    blocks = re.findall(r"<PATH>(.*?)</PATH>", s, flags=re.DOTALL)
    if blocks:
        content = blocks[-1]
    elif "</PATH>" in s:  # closer present but opener malformed
        content = s.rsplit("</PATH>", 1)[0]
        content = re.sub(r"^.*?[\(\[<]\s*/?PATH\s*[>\)\]]", "", content,
                         count=1, flags=re.DOTALL)
    elif "<SEP>" in s or s.strip():
        content = s
    else:
        return None
    # strip any residual malformed PATH opener on the first step
    content = re.sub(r"^\s*[\(\[<]?\s*PATH\s*[>\)\]]\s*", "", content)
    steps = [p.strip() for p in content.split("<SEP>")]
    return [p for p in steps if p] or None


def normalize_steps(steps: Sequence[str]) -> Tuple[str, ...]:
    return tuple(re.sub(r"\s+", " ", s.strip().lower()) for s in steps)


def extract_entities(reference_text: str) -> List[str]:
    """Starting entities from a CoT response (same cues as test.py)."""
    text = str(reference_text)
    m = re.search(r"starting from the entity '([^']+)'", text)
    if m:
        return [m.group(1)]
    if "We are working with the following entities:" in text:
        seg = text.split("We are working with the following entities:", 1)[1]
        seg = seg.split("Our goal", 1)[0]
        return re.findall(r"- '([^']+)'", seg)
    return []


# ---------------------------------------------------------------------------
# Deterministic path execution (reuses titan.graph primitives)
# ---------------------------------------------------------------------------

def _node_names(graph) -> List[str]:
    """Graph nodes that carry a type edge, longest first (for select parsing)."""
    names = [n for n in graph.nodes if GA._get_node_type(graph, n) is not None]
    return sorted(names, key=len, reverse=True)


_REL_TARGET_TYPES: Optional[Dict[str, str]] = None


def _rel_target_type(label: str) -> Optional[str]:
    """Static per-relation dominant target type (from rel_target_types.json,
    aggregated over ALL edges of that label in the whole graph). Used instead
    of per-node dynamic GA._get_node_type() for exec_common/exec_difference
    accumulation: a handful of MITRE-category names collide across node
    types (e.g. 'Code Signing' exists as both an attack_pattern and a
    course_of_action; see experiments/titan_findings.csv,
    graph_construction_bug), and _get_node_type()'s neighbor-iteration-order
    resolution silently misclassifies such nodes for a *specific* traversal
    context. The relation being followed already tells us the intended
    type; asking the node "what are you, generically" is the wrong
    question here."""
    global _REL_TARGET_TYPES
    if _REL_TARGET_TYPES is None:
        try:
            with open("rel_target_types.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            _REL_TARGET_TYPES = {k: v.get("target_type") for k, v in raw.items()}
        except FileNotFoundError:
            _REL_TARGET_TYPES = {}
    t = _REL_TARGET_TYPES.get(label)
    return None if t in (None, "?") else t


def parse_select_args(arg: str, names_longest_first: Sequence[str]) -> List[str]:
    """Segment 'Cannon LitePower' into known node names (greedy longest match).
    Whatever cannot be matched is kept as whitespace-split tokens."""
    remaining = arg.strip()
    out: List[str] = []
    while remaining:
        hit = next((nm for nm in names_longest_first
                    if remaining == nm or remaining.startswith(nm + " ")), None)
        if hit:
            out.append(hit)
            remaining = remaining[len(hit):].strip()
        else:
            tok, _, rest = remaining.partition(" ")
            out.append(tok)
            remaining = rest.strip()
    return out


def _traverse(graph, current: Set[str], label: str,
              acc_by_type: Dict[Optional[str], Set[str]]) -> Set[str]:
    nxt: Set[str] = set()
    rel_type = _rel_target_type(label)
    for node in current:
        try:
            for nb in graph.neighbors(node):
                if graph[node][nb].get("label") == label:
                    nxt.add(nb)
                    # static per-relation type (see _rel_target_type), not
                    # per-node dynamic GA._get_node_type -- avoids
                    # misclassifying cross-category name-collision nodes
                    acc_by_type.setdefault(rel_type, set()).add(nb)
        except Exception:
            continue
    return nxt


def execute_path(graph, entities: Sequence[str], steps: Sequence[str],
                 names_longest_first: Sequence[str]) -> Set[str]:
    """
    Execute a step sequence and return the FINAL node set (the answer).

    Semantics (mirroring titan/graph.py):
      is_<X>_type       seed all sources of that relation (or restrict, if a
                        current set exists)
      select A B ...    split into one branch per named entity
      filter <cond>     keep nodes whose description matches (and/or/single)
      exec_common T /   set-combine the per-branch accumulated nodes of type T
      exec_difference T (intersection / symmetric difference)
      <relation>        follow edges with that label
    """
    # branches: {branch_id: current node set}; accumulated typed results per branch
    if entities:
        branches: Dict[str, Set[str]] = {e: {e} for e in entities}
    else:
        branches = {"_": set()}
    acc: Dict[str, Dict[Optional[str], Set[str]]] = {b: {} for b in branches}
    final: Set[str] = set()

    for step in steps:
        step = step.strip()

        if step.startswith("select"):
            args = step[len("select"):].strip()
            sel = parse_select_args(args, names_longest_first) if args else []
            if sel:
                branches = {e: {e} for e in sel}
                acc = {e: {e and GA._get_node_type(graph, e): {e}} for e in sel}
            continue

        if step.startswith("exec_"):
            parts = step.split(None, 1)
            op = parts[0][len("exec_"):]
            op_type = parts[1].strip() if len(parts) > 1 else None
            if op_type:
                sets = [acc[b].get(op_type, set()) for b in branches]
            else:
                sets = list(branches.values())
            if not sets:
                final = set()
            elif op == "common":
                final = reduce(lambda a, b: a & b, sets)
            elif op == "difference":
                final = reduce(lambda a, b: a ^ b, sets)
            else:
                final = set()
            branches = {"_": set(final)}
            acc = {"_": {}}
            continue

        if step.startswith("is_") and step.endswith("_type"):
            # exact type-seed convention (is_<category>_type). NOT a plain
            # "_type" substring check: MITRE attribute relations like
            # x_mitre_impact_type / x_mitre_tactic_type also contain that
            # substring and are NOT type-seeding steps -- see
            # experiments/titan_findings.csv, executor_bug. Under the old
            # substring check, reaching one of those after an earlier step
            # had already emptied a branch would incorrectly RESET it to
            # the full source set instead of correctly staying empty.
            seed = set(GA.find_type_sources(graph, step))
            for b in branches:
                branches[b] = (branches[b] & seed) if branches[b] else set(seed)
            continue

        if step.startswith("filter "):
            kw = step[len("filter "):].strip()
            for b in branches:
                branches[b] = GA._get_filtered_nodes(graph, branches[b], kw)
            continue

        for b in branches:
            branches[b] = _traverse(graph, branches[b], step, acc[b])

    if not final:
        final = set().union(*branches.values()) if branches else set()
    return final


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def set_prf(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not gold:
        return (float("nan"),) * 3
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> Tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)


def length_bucket(steps: Sequence[str]) -> str:
    n = len(steps)
    return f"L{n}" if n <= 3 else "L4+"


def operator_bucket(steps: Sequence[str]) -> str:
    joined = " ".join(steps)
    if "exec_common" in joined:
        return "exec_common"
    if "exec_difference" in joined:
        return "exec_difference"
    if any(s.startswith("select") for s in steps):
        return "select"
    if any(s.startswith("filter") for s in steps):
        return "filter"
    return "plain"


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def load_predictions(pred_path: str) -> Dict[str, str]:
    """question -> generated text."""
    if pred_path.endswith(".json"):
        with open(pred_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return {str(it["question"]).strip().strip('"'): it["generated_path"]
                for it in items}
    df = pd.read_csv(pred_path)
    if not {"Question", "Generated"} <= set(df.columns):
        raise ValueError("Prediction CSV needs columns: Question, Generated")
    return dict(zip(df["Question"].astype(str).str.strip(), df["Generated"]))


def evaluate(data_csv: str, graph_file: str, pred_path: Optional[str],
             sample: Optional[int], seed: int, out_prefix: str) -> None:
    df = pd.read_csv(data_csv)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    preds = load_predictions(pred_path) if pred_path else None
    print(f"[INFO] Loading graph {graph_file} ...")
    graph = GA.load_graph(graph_file)
    names = _node_names(graph)
    print(f"[INFO] {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges; "
          f"{len(names)} typed entity names.")

    rows = []
    n_missing_pred = 0
    for _, r in df.iterrows():
        ref_steps = extract_path_steps(r["Path"])
        if ref_steps is None:
            continue
        entities = extract_entities(r["Path"])

        if preds is not None:
            gen_text = preds.get(str(r["Question"]).strip())
            if gen_text is None:
                n_missing_pred += 1
                continue
            gen_steps = extract_path_steps(gen_text)
        else:
            gen_steps = ref_steps  # gold self-consistency mode

        rec = {
            "Question": r["Question"],
            "Section": r.get("Section"),
            "SkeletonUnseen": r.get("SkeletonUnseen"),
            "len_bucket": length_bucket(ref_steps),
            "op_bucket": operator_bucket(ref_steps),
            "well_formed": gen_steps is not None,
            "ref_path": "<SEP>".join(ref_steps),
            "gen_path": "<SEP>".join(gen_steps) if gen_steps else None,
        }
        rec["path_em"] = bool(gen_steps) and (
            normalize_steps(gen_steps) == normalize_steps(ref_steps))

        gold_set = execute_path(graph, entities, ref_steps, names)
        rec["gold_executable"] = len(gold_set) > 0
        if gen_steps is None:
            pred_set: Set[str] = set()
        elif rec["path_em"]:
            pred_set = gold_set  # identical program, identical result
        else:
            pred_set = execute_path(graph, entities, gen_steps, names)
        rec["pred_nonempty"] = len(pred_set) > 0
        p, rr, f = set_prf(pred_set, gold_set)
        rec.update(precision=p, recall=rr, f1=f,
                   n_gold=len(gold_set), n_pred=len(pred_set))
        rows.append(rec)

    res = pd.DataFrame(rows)
    if n_missing_pred:
        print(f"[WARN] {n_missing_pred} data rows had no matching prediction "
              f"(joined on exact question string).")

    # ----------------------------- report ------------------------------
    def agg(sub: pd.DataFrame, label: str) -> dict:
        em_m, em_lo, em_hi = bootstrap_ci(sub["path_em"].astype(float).values, seed=seed)
        f1_m, f1_lo, f1_hi = bootstrap_ci(sub["f1"].values.astype(float), seed=seed)
        return {
            "bucket": label, "n": len(sub),
            "path_em": em_m, "path_em_ci": [em_lo, em_hi],
            "entity_f1": f1_m, "entity_f1_ci": [f1_lo, f1_hi],
            "well_formed": float(sub["well_formed"].mean()),
            "gold_executable": float(sub["gold_executable"].mean()),
            "pred_nonempty": float(sub["pred_nonempty"].mean()),
        }

    report = {"overall": agg(res, "overall"), "by_length": [], "by_operator": [],
              "by_section": []}
    for col, key in (("len_bucket", "by_length"), ("op_bucket", "by_operator"),
                     ("Section", "by_section")):
        for val, sub in res.groupby(col):
            report[key].append(agg(sub, str(val)))
    if res["SkeletonUnseen"].notna().any():
        report["by_skeleton"] = [agg(sub, f"skeleton_unseen={v}")
                                 for v, sub in res.groupby("SkeletonUnseen")]

    res.to_csv(f"{out_prefix}_per_example.csv", index=False)
    with open(f"{out_prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    o = report["overall"]
    print(f"\n=== {out_prefix} (n={o['n']}) ===")
    print(f"Path EM          : {o['path_em']:.3f}  CI95 [{o['path_em_ci'][0]:.3f}, {o['path_em_ci'][1]:.3f}]")
    print(f"Entity F1        : {o['entity_f1']:.3f}  CI95 [{o['entity_f1_ci'][0]:.3f}, {o['entity_f1_ci'][1]:.3f}]")
    print(f"Well-formed      : {o['well_formed']:.3f}")
    print(f"Gold executable  : {o['gold_executable']:.3f}")
    print(f"Pred non-empty   : {o['pred_nonempty']:.3f}")
    for key in ("by_length", "by_operator"):
        print(f"\n-- {key} --")
        for b in sorted(report[key], key=lambda x: x["bucket"]):
            print(f"  {b['bucket']:<16} n={b['n']:<6} EM={b['path_em']:.3f} "
                  f"F1={b['entity_f1']:.3f} gold_exec={b['gold_executable']:.3f}")
    print(f"\n[OK] Wrote {out_prefix}_report.json and {out_prefix}_per_example.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TITAN end-to-end evaluation harness.")
    p.add_argument("--data", required=True,
                   help="Reference CSV (Question, Path[CoT reference], Section)")
    p.add_argument("--graph", default="stix_graph_correct.graphml")
    p.add_argument("--pred", default=None,
                   help="Predictions (json list or CSV); omit for gold self-consistency mode")
    p.add_argument("--sample", type=int, default=None,
                   help="Evaluate a random subsample of N rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval", help="Output file prefix")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(a.data, a.graph, a.pred, a.sample, a.seed, a.out)
