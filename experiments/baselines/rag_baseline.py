#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG baseline for TITAN: retrieve relevant graph passages by embedding
similarity, then have an LLM answer the question in free text from that
context (no multi-hop path, no DSL, no SPARQL -- the "traditional retrieval
system" TITAN's introduction contrasts itself against, actually run instead
of only asserted).

Corpus: one passage per named entity in the graph, "<Name> (<type>):
<description-or-empty>" -- built once from the same GraphML every other
baseline/executor uses, so it's the same universe of entities.

Retrieval: sentence-transformers all-MiniLM-L6-v2 (small, fast, CPU/GPU
either way), cosine similarity, top-k passages concatenated into context.

Generation: any vLLM-servable model answers in free text given the
retrieved context (no DSL/SPARQL formatting requested -- this baseline
represents the "just retrieve and answer" alternative).

Scoring: entity-level P/R/F1, but since the answer is free text (not a
formalism with defined variables), predicted entities are recovered by
greedy longest-match string search over the known node-name vocabulary
(same technique used for `select` parsing elsewhere: parse_select_args),
scanning the ENTIRE generated answer text rather than one placeholder
argument. Compared against the same gold answer sets from
titan.evaluate.execute_path() as every other baseline.

Run:
    source .venv-vllm/bin/activate
    CUDA_VISIBLE_DEVICES=4 python baselines/rag_baseline.py \
        --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
        --test datasets/TEMPLATE_DISJOINT/CoT/test_heldout.csv \
        --out baselines/qwen25_7b_rag_heldout_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan import evaluate as ET  # noqa: E402
from titan import graph as GA  # noqa: E402

SYSTEM_PROMPT = (
    "You are a Cyber Threat Intelligence assistant. Answer the user's question using ONLY "
    "the information in the provided context passages. List the specific named entities, "
    "techniques, tools, mitigations, or values that answer the question. If the context "
    "does not contain the answer, say so briefly."
)


# ---------------------------------------------------------------------------
# Corpus + index
# ---------------------------------------------------------------------------

def build_corpus(graph_file: str) -> tuple[list[str], list[str]]:
    """One passage per named entity: 'Name (type): description'."""
    g = GA.load_graph(graph_file)
    names, passages = [], []
    for n in g.nodes:
        t = GA._get_node_type(g, n)
        if t is None:
            continue
        desc = None
        try:
            for nb in g.neighbors(n):
                if g[n][nb].get("label") == "description":
                    desc = str(nb)
                    break
        except Exception:
            pass
        passage = f"{n} ({t})" + (f": {desc}" if desc else "")
        names.append(str(n))
        passages.append(passage)
    return names, passages


def build_index(passages: List[str], model_name: str, cache_path: Optional[str]):
    from sentence_transformers import SentenceTransformer

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["emb"]
    model = SentenceTransformer(model_name)
    emb = model.encode(passages, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True)
    if cache_path:
        np.savez(cache_path, emb=emb)
    return emb


def retrieve(question_emb, corpus_emb, k: int) -> List[int]:
    sims = corpus_emb @ question_emb
    return np.argsort(-sims)[:k].tolist()


# ---------------------------------------------------------------------------
# Entity extraction from free-text answers
# ---------------------------------------------------------------------------

def extract_entities_from_text(text: str, names_longest_first: List[str],
                               max_names: int = 4000) -> set:
    """Greedy longest-match scan for known entity names anywhere in free
    text (not just a single placeholder arg, unlike parse_select_args)."""
    lower_text = text.lower()
    found = set()
    for name in names_longest_first[:max_names]:
        if len(name) < 3:
            continue
        if name.lower() in lower_text:
            found.add(name)
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RAG baseline for TITAN")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--corpus-cache", default="rag_corpus_embeddings.npz")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[INFO] building retrieval corpus ...")
    names, passages = build_corpus(args.graph)
    print(f"[INFO] {len(passages)} passages")
    corpus_emb = build_index(passages, args.embed_model, args.corpus_cache)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model)

    test = pd.read_csv(args.test)
    if args.sample and args.sample < len(test):
        total = len(test)
        test = (test.groupby("Section", group_keys=False)
                    .apply(lambda g: g.sample(max(1, round(args.sample * len(g) / total)),
                                              random_state=args.seed))
                    .reset_index(drop=True))
    questions = test["Question"].astype(str).tolist()
    print(f"[INFO] {len(questions)} questions; retrieving top-{args.top_k} passages each ...")

    q_emb = embed_model.encode(questions, batch_size=256, show_progress_bar=True,
                               normalize_embeddings=True)
    contexts = []
    for i in range(len(questions)):
        idx = retrieve(q_emb[i], corpus_emb, args.top_k)
        contexts.append("\n".join(f"- {passages[j]}" for j in idx))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, quantization=args.quantization,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
              trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    conversations = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}]
        for q, ctx in zip(questions, contexts)
    ]
    outputs = llm.chat(conversations, sampling, use_tqdm=True)

    results = [{"question": q, "context": ctx, "generated_text": o.outputs[0].text}
              for q, ctx, o in zip(questions, contexts, outputs)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[OK] {len(results)} generations -> {args.out}")

    # ------------------------------- scoring ----------------------------
    g = GA.load_graph(args.graph)
    exec_names = ET._node_names(g)
    names_longest_first = sorted(set(names), key=len, reverse=True)

    rows = []
    t0 = time.time()
    for i, (r, gen) in enumerate(zip(test.itertuples(), results)):
        steps = ET.extract_path_steps(r.Path)
        entities = ET.extract_entities(r.Path)
        gold = ET.execute_path(g, entities, steps, exec_names)

        pred = extract_entities_from_text(gen["generated_text"], names_longest_first)
        p, rr, f1 = ET.set_prf(pred, gold)
        rows.append({"Question": r.Question, "Section": r.Section, "n_gold": len(gold),
                    "n_pred": len(pred), "precision": p, "recall": rr, "f1": f1})
        if (i + 1) % 1000 == 0:
            print(f"  scored [{i+1}/{len(results)}] ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    out_prefix = args.out.rsplit(".", 1)[0] + "_eval"
    res.to_csv(f"{out_prefix}_per_example.csv", index=False)

    def bootstrap_ci(values, n_boot=1000, seed=0, level=0.95):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
        lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(values.mean()), float(lo), float(hi)

    f1_m, f1_lo, f1_hi = bootstrap_ci(res["f1"].values)
    report = {"overall": {"n": len(res), "entity_f1": f1_m, "entity_f1_ci": [f1_lo, f1_hi],
                          "pred_nonempty": float((res["n_pred"] > 0).mean())},
             "by_section": []}
    for sec, sub in res.groupby("Section"):
        m, lo, hi = bootstrap_ci(sub["f1"].values)
        report["by_section"].append({"section": sec, "n": len(sub), "entity_f1": m,
                                     "entity_f1_ci": [lo, hi]})
    with open(f"{out_prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    o = report["overall"]
    print(f"\n=== RAG baseline: {args.model} (n={o['n']}) ===")
    print(f"Entity F1        : {o['entity_f1']:.3f}  CI95 [{o['entity_f1_ci'][0]:.3f}, "
          f"{o['entity_f1_ci'][1]:.3f}]")
    print(f"Pred non-empty   : {o['pred_nonempty']:.3f}")
    print(f"[OK] wrote {out_prefix}_report.json / _per_example.csv")


if __name__ == "__main__":
    main()
