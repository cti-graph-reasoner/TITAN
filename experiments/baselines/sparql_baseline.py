#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer 2: prompted SPARQL generation baseline for TITAN.

Answers the metareview's / Reviewer 4's core objection ("this could be done
by existing SPARQL generators") with data instead of assertion. Structured
to be exactly as fair as possible to the TITAN-DSL baselines
(fewshot_path_baseline.py / vllm_baseline.py):
  - same graph, same test questions, same model, same decoding
  - schema description carries the SAME information (typed relations,
    node types) as the DSL prompt, just rendered as RDF predicates instead
    of DSL relation names
  - the 12 few-shot examples are the SAME questions as the DSL baseline's
    shots (same seed, same executable-verified selection), with SPARQL
    targets COMPILED from their gold paths via baselines/path_to_sparql.py
    (compiler validated at 100.00% equivalence with the native executor on
    5500 sampled rows across two seeds -- see
    baselines/validate_sparql_equivalence.py and titan_findings.csv)
  - the 3 worked examples are translated from the DSL baseline's 3 worked
    examples via the same compiler, so the reasoning content is identical,
    only the target language differs

Scoring is entity-level P/R/F1 only (SPARQL has no DSL-path-EM analogue):
predicted SPARQL is executed against titan_graph.nt (rdflib) and compared
to the gold answer set from titan.evaluate.execute_path() -- the SAME
ground truth every other model is scored against.

Run (vLLM, e.g. Qwen2.5-72B across 2 GPUs):
    source .venv-vllm/bin/activate
    CUDA_VISIBLE_DEVICES=1,2 python baselines/sparql_baseline.py \
        --model unsloth/Qwen2.5-72B-Instruct-bnb-4bit --tp 1 --pp 2 \
        --quantization bitsandbytes \
        --train datasets/TEMPLATE_DISJOINT/NoCoT/train.csv \
        --test datasets/TEMPLATE_DISJOINT/CoT/test_heldout.csv \
        --exec-csv datasets/TEMPLATE_DISJOINT/CoT/train.annotated.csv \
        --out baselines/qwen25_72b_sparql_heldout_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan import evaluate as ET  # noqa: E402
from titan import graph as GA  # noqa: E402
from baselines.fewshot_path_baseline import (  # noqa: E402
    collect_vocabulary, pick_fewshot)
from baselines.path_to_sparql import (  # noqa: E402
    compile_path, load_rel_types, uri_to_name)

NODE_NS = "http://titan.local/n/"
REL_NS = "http://titan.local/r/"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def render_schema(relations: List[str], rel_types: dict) -> Tuple[List[str], List[str]]:
    types_used: set = set()
    lines = []
    for rel in sorted(relations):
        info = rel_types.get(rel)
        if not info:
            continue
        src, tgt = info["source_type"], info["target_type"]
        src = src if src != "?" else "value"
        tgt = tgt if tgt != "?" else "value"
        lines.append(f"  {src} --{rel}--> {tgt}")
        types_used.update({src, tgt})
    return sorted(types_used - {"value"}), lines


def compile_shot(question: str, dsl_target: str, rel_types: dict,
                 node_type_fn, parse_select_fn) -> Optional[str]:
    steps = ET.extract_path_steps(dsl_target)
    entities = ET.extract_entities(dsl_target)
    if steps is None:
        return None
    try:
        return compile_path(steps, entities, rel_types, node_type_fn, parse_select_fn)
    except Exception:
        return None


SYSTEM_TEMPLATE = """You are a Cyber Threat Intelligence assistant that answers questions by \
emitting a SPARQL query over an RDF graph derived from MITRE ATT&CK.

GRAPH COMPOSITION
Every entity (including attribute values like platform names or tactic types) is a URI of
the form <http://titan.local/n/NAME> (spaces and special characters percent-encoded), each
carrying an <http://www.w3.org/2000/01/rdf-schema#label> triple with its plain-text name.
Entities have these types:
  {node_types}
Every edge is a directed RDF triple using a predicate of the form
<http://titan.local/r/RELATION>. These are the relations you can traverse, with the entity
types they connect (source --relation--> target; "value" = an attribute-value entity, still
a URI, not a literal):
{schema}
All entity relations also exist in the reverse direction under their inverse name (e.g.
uses_malware / used_by_malware), so reasoning can move in both directions.

SPARQL FRAGMENT
Write exactly ONE query of the form:
  SELECT DISTINCT ?ans WHERE {{ ... }}
Available constructs:
- entity seed:      VALUES ?x0 {{ <http://titan.local/n/EntityName> }}
- relation step:     ?xi <http://titan.local/r/relation_name> ?xj .
- keyword filter on an entity's description (case-insensitive substring):
                     ?xi <http://titan.local/r/description> ?d . ?d <http://www.w3.org/2000/01/rdf-schema#label> ?dl .
                     FILTER(CONTAINS(LCASE(STR(?dl)), "keyword"))
- type seed (no named entity in the question):
                     ?x0 <http://titan.local/r/is_<type>_type> ?t .
- comparing TWO named entities: two independent sub-patterns (separate variable prefixes),
  each ending by binding its own reached node to ?ans, combined as needed (see examples)
- set intersection ("common"): put both branches' patterns in the SAME WHERE block, sharing
  the variable ?ans (an implicit join = intersection)
- set difference ("difference", SYMMETRIC — items in exactly one side):
  {{ {{ <branch A pattern, ?ans> }} MINUS {{ <branch B pattern, ?ans> }} }}
  UNION
  {{ {{ <branch B pattern, ?ans> }} MINUS {{ <branch A pattern, ?ans> }} }}
Always BIND the final answer node to variable ?ans.

IMPORTANT convention: when the question names a specific entity (a malware, group,
campaign, tool...), the traversal starts implicitly FROM that entity's URI as ?x0 — do NOT
add an is_<type>_type step or VALUES-seed a second time for it. Only seed via is_<type>_type
when the question does not name a starting entity, or when it compares several named
entities.

WORKED EXAMPLES

{worked_examples}

Respond with ONLY the SPARQL query, no explanation, no markdown code fences.

More examples:
{examples}"""


SYSTEM_TEMPLATE_COT = SYSTEM_TEMPLATE.replace(
    "Respond with ONLY the SPARQL query, no explanation, no markdown code fences.",
    "First reason step by step about which relations to follow and why (as in the "
    "worked examples' \"Reasoning:\" lines above), THEN give the final SPARQL query "
    "at the end of your response."
)


def build_worked_examples(rel_types: dict, node_type_fn, parse_select_fn) -> str:
    cases = [
        ("What are the best practices for protecting against .NET malware used by APT32?",
         "Reasoning: APT32 is an intrusion_set, so we start from its URI implicitly as ?x0. "
         "We need its malware (uses_malware), keep only .NET ones (filter .NET via "
         "description), get the techniques those malware use (uses_attack_pattern), and "
         "finally the mitigations (mitigated_by).",
         "<PATH>uses_malware<SEP>filter .NET<SEP>uses_attack_pattern<SEP>mitigated_by</PATH>",
         ["APT32"], "intrusion_set"),
        ("How do security experts typically prevent and respond to FALLCHILL and BlackEnergy 3?",
         "Reasoning: two named malware are compared, so we seed all malware nodes "
         "(is_malware_type), give each entity its own branch (VALUES), follow each branch to "
         "its techniques and mitigations, and INTERSECT the branches by sharing ?ans.",
         "<PATH>is_malware_type<SEP>select FALLCHILL BlackEnergy 3<SEP>uses_attack_pattern"
         "<SEP>mitigated_by<SEP>exec_common course_of_action</PATH>",
         [], None),
        ("What similarities exist in the target assets of spyware and GoldenSpy?",
         "Reasoning: one side is a NAMED entity (GoldenSpy) and the other a FILTERED CLASS "
         "(spyware) — branch A starts implicitly from GoldenSpy's URI; branch B reaches the "
         "same node type by going the OTHER direction (used_by_malware) with a description "
         "filter, since there is no second named entity to seed from. Both branches bind "
         "x_mitre_asset nodes to ?ans, joined (intersection).",
         "<PATH>uses_attack_pattern<SEP>targets<SEP>used_by_attack_pattern<SEP>"
         "used_by_malware<SEP>filter spyware<SEP>exec_common x_mitre_asset</PATH>",
         ["GoldenSpy"], "malware"),
    ]
    out = []
    for q, reasoning, dsl, entities, _ in cases:
        q_compiled = compile_shot(q, dsl, rel_types, node_type_fn, parse_select_fn)
        out.append(f"Q: {q}\n{reasoning}\nA: {q_compiled}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def extract_sparql(text: str) -> Optional[str]:
    text = re.sub(r"```(?:sparql)?", "", text).strip()
    m = re.search(r"(SELECT\s+DISTINCT\s+\?ans\s+WHERE\s*\{.*)", text,
                 flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    q = m.group(1)
    # trim to the balanced closing brace of the outermost WHERE block
    depth, end = 0, None
    started = False
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


def run_query(rdf, query: str) -> set:
    results = rdf.query(query)
    out = set()
    for row in results:
        val = row[0]
        out.add(uri_to_name(str(val)) if str(val).startswith((NODE_NS, REL_NS)) else str(val))
    return out


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


def bootstrap_ci(values, n_boot=1000, seed=0, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SPARQL prompted baseline (Layer 2)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--nt", default="titan_graph.nt")
    ap.add_argument("--rel-types", default="rel_target_types.json")
    ap.add_argument("--exec-csv", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-prompt", default=None)
    ap.add_argument("--cot", action="store_true",
                    help="use the CoT-instruction system prompt (reason step by step, "
                         "then emit the SPARQL query) instead of the direct-answer prompt")
    ap.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"],
                    help="for Harmony-format reasoning models (e.g. gpt-oss): controls "
                         "chain-of-thought verbosity via chat_template_kwargs. Without this, "
                         "gpt-oss's analysis-channel reasoning can consume the entire --max-new "
                         "budget before it ever reaches the SPARQL answer (same false-negative "
                         "failure mode already hit and fixed for the DSL baseline).")
    ap.add_argument("--disable-async-scheduling", action="store_true",
                    help="pass async_scheduling=False to vLLM. Works around a vLLM v1 internal "
                         "bug (AssertionError: request.num_output_placeholders >= 0 in "
                         "async_scheduler.py) observed on a pipeline-parallel + very long "
                         "(--max-new 3500) generation run; leave unset for configs that already "
                         "work fine with the default.")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap concurrent in-flight sequences (vLLM's max_num_seqs). Without a "
                         "cap, vLLM can batch enough concurrent very-long generations (e.g. "
                         "DeepSeek-R1's verbose <think> traces at --max-new 3500) to fill the "
                         "entire KV cache and OOM partway through a run, losing all progress "
                         "since this script only writes output at the very end. Lower this if "
                         "that happens; leave unset for configs that already work fine.")
    args = ap.parse_args()
    if not args.dump_prompt and not args.out:
        ap.error("--out is required unless --dump-prompt is used")

    g = GA.load_graph(args.graph)
    names = ET._node_names(g)
    rel_types = load_rel_types(args.rel_types)
    node_type_fn = lambda n: GA._get_node_type(g, n)  # noqa: E731
    parse_select_fn = lambda s: ET.parse_select_args(s, names)  # noqa: E731

    rels = collect_vocabulary(args.train)
    node_types, schema_lines = render_schema(rels, rel_types)
    worked = build_worked_examples(rel_types, node_type_fn, parse_select_fn)

    shots = pick_fewshot(args.train, args.shots, args.seed, executable_csv=args.exec_csv)
    compiled_shots = []
    for q, dsl in shots:
        c = compile_shot(q, dsl, rel_types, node_type_fn, parse_select_fn)
        if c:
            compiled_shots.append((q, c))
    print(f"[INFO] {len(rels)} relations, {len(schema_lines)} schema lines, "
          f"{len(compiled_shots)}/{len(shots)} shots compiled successfully")

    template = SYSTEM_TEMPLATE_COT if args.cot else SYSTEM_TEMPLATE
    system_prompt = template.format(
        node_types=", ".join(node_types),
        schema="\n".join(schema_lines),
        worked_examples=worked,
        examples="\n".join(f"Q: {q}\nA: {p}" for q, p in compiled_shots),
    )
    print(f"[INFO] system prompt ~{len(system_prompt)} chars")

    if args.dump_prompt:
        with open(args.dump_prompt, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        print(f"[OK] prompt written to {args.dump_prompt} (no model loaded, exiting)")
        return

    test = pd.read_csv(args.test)
    if args.sample and args.sample < len(test):
        total = len(test)
        test = (test.groupby("Section", group_keys=False)
                    .apply(lambda gr: gr.sample(max(1, round(args.sample * len(gr) / total)),
                                                random_state=args.seed))
                    .reset_index(drop=True))
    print(f"[INFO] generating for {len(test)} questions")

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model, tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
        quantization=args.quantization, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
    )
    if args.disable_async_scheduling:
        llm_kwargs["async_scheduling"] = False
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    questions = test["Question"].astype(str).tolist()
    conversations = [[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": q}] for q in questions]
    chat_kwargs = ({"reasoning_effort": args.reasoning_effort}
                   if args.reasoning_effort else None)
    outputs = llm.chat(conversations, sampling, use_tqdm=True,
                       chat_template_kwargs=chat_kwargs)
    raw = [{"question": q, "generated_text": o.outputs[0].text}
          for q, o in zip(questions, outputs)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=1)
    print(f"[OK] {len(raw)} generations -> {args.out}")

    # ------------------------------- scoring ----------------------------
    print("[INFO] loading RDF graph for scoring ...")
    from rdflib import Graph as RDFGraph
    rdf = RDFGraph()
    rdf.parse(args.nt, format="nt")

    rows = []
    t0 = time.time()
    for i, (r, gen) in enumerate(zip(test.itertuples(), raw)):
        steps = ET.extract_path_steps(r.Path)
        entities = ET.extract_entities(r.Path)
        gold = ET.execute_path(g, entities, steps, names)

        q = extract_sparql(gen["generated_text"])
        rec = {"Question": r.Question, "Section": r.Section, "n_gold": len(gold),
               "well_formed": q is not None}
        if q is None:
            rec.update(precision=0.0, recall=0.0, f1=0.0 if gold else float("nan"),
                       exec_error=None, n_pred=0)
        else:
            try:
                pred = run_query(rdf, q)
                p, rr, f1 = set_prf(pred, gold)
                rec.update(precision=p, recall=rr, f1=f1, n_pred=len(pred), exec_error=None)
            except Exception as e:
                rec.update(precision=0.0, recall=0.0, f1=0.0 if gold else float("nan"),
                           n_pred=0, exec_error=f"{type(e).__name__}: {e}"[:200])
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  scored [{i+1}/{len(raw)}] ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    out_prefix = args.out.rsplit(".", 1)[0] + "_eval"
    res.to_csv(f"{out_prefix}_per_example.csv", index=False)

    f1_m, f1_lo, f1_hi = bootstrap_ci(res["f1"].values)
    report = {
        "overall": {
            "n": len(res),
            "entity_f1": f1_m, "entity_f1_ci": [f1_lo, f1_hi],
            "well_formed": float(res["well_formed"].mean()),
            "exec_error_rate": float(res["exec_error"].notna().mean()),
            "pred_nonempty": float((res["n_pred"] > 0).mean()),
        },
        "by_section": [],
    }
    for sec, sub in res.groupby("Section"):
        m, lo, hi = bootstrap_ci(sub["f1"].values)
        report["by_section"].append({"section": sec, "n": len(sub), "entity_f1": m,
                                     "entity_f1_ci": [lo, hi],
                                     "well_formed": float(sub["well_formed"].mean())})
    with open(f"{out_prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    o = report["overall"]
    print(f"\n=== SPARQL baseline: {args.model} (n={o['n']}) ===")
    print(f"Entity F1        : {o['entity_f1']:.3f}  CI95 [{o['entity_f1_ci'][0]:.3f}, "
          f"{o['entity_f1_ci'][1]:.3f}]")
    print(f"Well-formed      : {o['well_formed']:.3f}")
    print(f"Execution errors : {o['exec_error_rate']:.3f}")
    print(f"Pred non-empty   : {o['pred_nonempty']:.3f}")
    print(f"[OK] wrote {out_prefix}_report.json / _per_example.csv")


if __name__ == "__main__":
    main()
