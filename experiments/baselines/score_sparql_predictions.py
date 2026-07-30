#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone, parallelized scorer for already-generated SPARQL predictions
(the *_generated_text* JSON produced by sparql_baseline.py's generation
step). Split out from sparql_baseline.py because rdflib's pure-Python
SPARQL engine measured ~23s/query on these multi-hop BGPs (projected
~100h for the full 15,627-row test_heldout) -- far too slow to be
practical. Uses pyoxigraph (compiled Rust SPARQL engine) instead, which
benchmarked at ~0.31s/query single-threaded (~81 min for the full set);
this script additionally parallelizes across CPU cores via
multiprocessing, each worker loading its own Store (load itself takes
~0.3s, negligible next to query time).

Run:
    python3 baselines/score_sparql_predictions.py \
        --pred baselines/qwen25_72b_sparql_heldout_full.json \
        --data datasets/TEMPLATE_DISJOINT/CoT/test_heldout.csv \
        --workers 16 --out baselines/qwen25_72b_sparql_heldout_full_eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import multiprocessing
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan import evaluate as ET  # noqa: E402
from titan import graph as GA  # noqa: E402
from baselines.path_to_sparql import compile_path, load_rel_types  # noqa: E402

NODE_NS = "http://titan.local/n/"
REL_NS = "http://titan.local/r/"

_STORE = None
_GRAPH = None
_NAMES = None
_REL_TYPES = None


def _init(nt_path: str, graph_path: str, rel_types_path: str):
    global _STORE, _GRAPH, _NAMES, _REL_TYPES
    import pyoxigraph as ox
    _STORE = ox.Store()
    _STORE.bulk_load(path=nt_path, format=ox.RdfFormat.N_TRIPLES)
    _GRAPH = GA.load_graph(graph_path)
    _NAMES = ET._node_names(_GRAPH)
    _REL_TYPES = load_rel_types(rel_types_path)


def normalize_query(q: str) -> str:
    """Collapse all whitespace so cosmetic formatting differences (the only
    thing that should vary between a model's output and the compiler's
    canonical form, since both follow the identical variable-naming and
    structural convention) don't count against exact match."""
    return re.sub(r"\s+", " ", q).strip()


def extract_sparql(text: str) -> Optional[str]:
    text = re.sub(r"```(?:sparql)?", "", text).strip()
    m = re.search(r"(SELECT\s+DISTINCT\s+\?ans\s+WHERE\s*\{.*)", text,
                 flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    q = m.group(1)
    depth, end, started = 0, None, False
    for i, ch in enumerate(q):
        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                end = i + 1
                break
    return q[:end] if end else q


def uri_to_name(uri: str) -> str:
    for ns in (NODE_NS, REL_NS):
        if uri.startswith(ns):
            from urllib.parse import unquote
            return unquote(uri[len(ns):])
    return uri


def set_prf(pred: set, gold: set):
    if not gold:
        return float("nan"), float("nan"), float("nan")
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


class _TimeoutError(Exception):
    pass


class _time_limit:
    """Bounds a single query's execution time so one pathological BGP (e.g. an
    unbounded/malformed pattern from a truncated or malformed model generation)
    can't hang an entire worker for the rest of the run -- observed happening
    on the Phi CoT-SPARQL predictions, where one row's generated query ran for
    18+ minutes while every other worker sat idle waiting for it.

    Uses ITIMER_VIRTUAL (process CPU time), not wall clock: this scorer runs
    many worker processes in parallel, often alongside unrelated GPU/CPU jobs,
    so wall-clock time for a legitimately-slow-but-finite query is not stable
    run-to-run -- it depends on incidental scheduling contention, which would
    make query_em/F1 vary with server load instead of being a fixed property
    of the prediction being scored. CPU time isn't affected by that."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise _TimeoutError(f"query exceeded {self.seconds}s CPU time")

    def __enter__(self):
        signal.signal(signal.SIGVTALRM, self._handler)
        signal.setitimer(signal.ITIMER_VIRTUAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)


def _node_type(n):
    return GA._get_node_type(_GRAPH, n)


def _parse_select(s):
    return ET.parse_select_args(s, _NAMES)


def _score_one(arg):
    question, ref_path, gen_text = arg
    steps = ET.extract_path_steps(ref_path)
    entities = ET.extract_entities(ref_path)
    gold = ET.execute_path(_GRAPH, entities, steps, _NAMES)

    try:
        gold_query = compile_path(steps, entities, _REL_TYPES, _node_type, _parse_select)
    except Exception:
        gold_query = None

    q = extract_sparql(gen_text)
    rec = {
        "Question": question, "n_gold": len(gold), "well_formed": q is not None,
        "len_bucket": ET.length_bucket(steps), "op_bucket": ET.operator_bucket(steps),
    }
    query_em = (q is not None and gold_query is not None
               and normalize_query(q) == normalize_query(gold_query))
    rec["query_em"] = query_em
    if q is None:
        rec.update(precision=0.0, recall=0.0, f1=0.0 if gold else float("nan"), n_pred=0,
                   exec_error=None)
        return rec
    try:
        with _time_limit(10):
            results = list(_STORE.query(q))
        pred = {uri_to_name(row[0].value) for row in results}
        p, rr, f1 = set_prf(pred, gold)
        rec.update(precision=p, recall=rr, f1=f1, n_pred=len(pred), exec_error=None)
    except Exception as e:
        rec.update(precision=0.0, recall=0.0, f1=0.0 if gold else float("nan"), n_pred=0,
                   exec_error=f"{type(e).__name__}: {e}"[:200])
    return rec


def bootstrap_ci(values, n_boot=1000, seed=0, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pred", required=True, help="JSON from sparql_baseline.py generation")
    ap.add_argument("--data", required=True, help="reference CSV with Question, Path, Section")
    ap.add_argument("--nt", default="titan_graph.nt")
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--rel-types", default="rel_target_types.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        preds = {d["question"]: d["generated_text"] for d in json.load(f)}
    ref = pd.read_csv(args.data)
    print(f"[INFO] {len(ref)} reference rows, {len(preds)} predictions")

    has_skeleton = "SkeletonUnseen" in ref.columns
    jobs = []
    sections = []
    skeleton_unseen = []
    for r in ref.itertuples():
        gen = preds.get(r.Question)
        if gen is None:
            continue
        jobs.append((r.Question, r.Path, gen))
        sections.append(r.Section)
        skeleton_unseen.append(getattr(r, "SkeletonUnseen", None) if has_skeleton else None)

    print(f"[INFO] scoring {len(jobs)} rows across {args.workers} workers ...")
    t0 = time.time()
    # Dispatched one-by-one via apply_async (not imap) so each row gets its
    # own wall-clock .get(timeout=...): pyoxigraph's query engine is a Rust
    # extension that doesn't yield back to the Python interpreter during a
    # long call, so the in-worker SIGVTALRM guard in _score_one/_time_limit
    # can sit pending forever and never actually fire for a pathological
    # query stuck inside native code (observed directly: a worker pegged at
    # 100% CPU for 19+ minutes despite the 10s CPU-time guard). This outer
    # timeout doesn't need the stuck worker's cooperation -- it just gives up
    # waiting on that one AsyncResult and scores the row as a failure; the
    # Pool keeps routing new work to whichever workers are still responsive,
    # at the cost of permanently losing that one worker's slot for the rest
    # of the run (acceptable: this class of hang is rare, ~1 in several
    # thousand rows).
    TIMEOUT_S = 15
    n_timeout = 0
    with Pool(args.workers, initializer=_init,
             initargs=(args.nt, args.graph, args.rel_types)) as pool:
        async_results = [pool.apply_async(_score_one, (job,)) for job in jobs]
        rows = []
        for i, ar in enumerate(async_results):
            try:
                rec = ar.get(timeout=TIMEOUT_S)
            except multiprocessing.TimeoutError:
                n_timeout += 1
                question, ref_path, _gen = jobs[i]
                steps = ET.extract_path_steps(ref_path)
                rec = {
                    "Question": question, "n_gold": None, "well_formed": False,
                    "len_bucket": ET.length_bucket(steps), "op_bucket": ET.operator_bucket(steps),
                    "query_em": False, "precision": 0.0, "recall": 0.0, "f1": float("nan"),
                    "n_pred": 0, "exec_error": f"TimeoutError: exceeded {TIMEOUT_S}s wall-clock",
                }
            rows.append(rec)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(jobs) - i - 1) / rate
                print(f"  [{i+1}/{len(jobs)}] {elapsed:.0f}s elapsed, "
                      f"ETA {eta/60:.1f} min", flush=True)
    if n_timeout:
        print(f"[WARN] {n_timeout}/{len(jobs)} rows exceeded {TIMEOUT_S}s and were scored as "
              f"failures (pathological/malformed generated query, not counted against the "
              f"model beyond that -- see exec_error column in the per-example CSV)")

    res = pd.DataFrame(rows)
    res["Section"] = sections
    res["SkeletonUnseen"] = skeleton_unseen
    res.to_csv(f"{args.out}_per_example.csv", index=False)

    def agg(sub: pd.DataFrame, label: str) -> dict:
        em_m, em_lo, em_hi = bootstrap_ci(sub["query_em"].astype(float).values)
        f1_m, f1_lo, f1_hi = bootstrap_ci(sub["f1"].values)
        return {
            "bucket": label, "n": len(sub),
            "query_em": em_m, "query_em_ci": [em_lo, em_hi],
            "entity_f1": f1_m, "entity_f1_ci": [f1_lo, f1_hi],
            "well_formed": float(sub["well_formed"].mean()),
            "exec_error_rate": float(sub["exec_error"].notna().mean()),
            "pred_nonempty": float((sub["n_pred"] > 0).mean()),
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
    with open(f"{args.out}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    o = report["overall"]
    print(f"\n=== SPARQL scoring: {args.pred} (n={o['n']}) ===")
    print(f"Query EM         : {o['query_em']:.3f}  CI95 [{o['query_em_ci'][0]:.3f}, "
          f"{o['query_em_ci'][1]:.3f}]")
    print(f"Entity F1        : {o['entity_f1']:.3f}  CI95 [{o['entity_f1_ci'][0]:.3f}, "
          f"{o['entity_f1_ci'][1]:.3f}]")
    print(f"Well-formed      : {o['well_formed']:.3f}")
    print(f"Execution errors : {o['exec_error_rate']:.3f}")
    print(f"Pred non-empty   : {o['pred_nonempty']:.3f}")
    for key in ("by_length", "by_operator", "by_section"):
        print(f"\n-- {key} --")
        for b in sorted(report[key], key=lambda x: x["bucket"]):
            print(f"  {b['bucket']:<32} n={b['n']:<6} EM={b['query_em']:.3f} "
                  f"F1={b['entity_f1']:.3f} well_formed={b['well_formed']:.3f}")
    if "by_skeleton" in report:
        print("\n-- by_skeleton --")
        for b in report["by_skeleton"]:
            print(f"  {b['bucket']:<32} n={b['n']:<6} EM={b['query_em']:.3f} "
                  f"F1={b['entity_f1']:.3f}")
    print(f"\n[OK] wrote {args.out}_report.json / _per_example.csv")


if __name__ == "__main__":
    main()
