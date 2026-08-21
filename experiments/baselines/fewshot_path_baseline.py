#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Few-shot (no fine-tuning) path-generation baseline for TITAN.

Prompts an instruction-tuned LLM with the TITAN DSL specification (relation
vocabulary + operators) and K few-shot examples drawn from the TRAINING
split, then asks it to emit the <PATH>...</PATH> plan for each test question.
Output is a JSON list [{"question", "generated_path"}, ...] scorable by
titan/evaluate.py.

Run (Llama-3.3-70B 4-bit across two GPUs):
    CUDA_VISIBLE_DEVICES=1,2 python baselines/fewshot_path_baseline.py \
        --model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
        --train datasets/TEMPLATE_DISJOINT/NoCoT/train.csv \
        --test datasets/TEMPLATE_DISJOINT/NoCoT/test_heldout.csv \
        --sample 1000 --out baselines/llama33_70b_heldout.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def collect_vocabulary(train_csv: str) -> list[str]:
    """Distinct plain relation steps observed in training paths."""
    df = pd.read_csv(train_csv)
    rels = set()
    for p in df["Path"].astype(str):
        m = re.search(r"<PATH>(.*?)</PATH>", p, flags=re.DOTALL)
        if not m:
            continue
        for step in m.group(1).split("<SEP>"):
            step = step.strip()
            if step and not step.startswith(("filter ", "select ", "exec_")):
                rels.add(step)
    return sorted(rels)


def collect_typed_schema(graph_file: str, relations: list[str]) -> tuple[list[str], list[str]]:
    """Derive the graph composition from the KG itself: the node-type inventory
    and, per relation, the dominant (source type -> target type) signature."""
    from collections import Counter

    from titan import graph as GA

    g = GA.load_graph(graph_file)
    node_type: dict[str, str | None] = {}

    def ntype(n):
        if n not in node_type:
            node_type[n] = GA._get_node_type(g, n)
        return node_type[n]

    wanted = set(relations)
    sig: dict[str, Counter] = {r: Counter() for r in wanted}
    for u, v, data in g.edges(data=True):
        lab = data.get("label")
        if lab in wanted:
            sig[lab][(ntype(u) or "?", ntype(v) or "value")] += 1

    lines = []
    used_types: set[str] = set()
    for rel in sorted(wanted):
        if not sig[rel]:
            continue
        total = sum(sig[rel].values())
        # every signature carrying >=15% of this relation's edges
        for (src, tgt), n in sig[rel].most_common():
            if n / total < 0.15:
                break
            lines.append(f"  {src} --{rel}--> {tgt}")
            used_types.update({src, tgt})
    types = sorted(used_types - {"?", "value"})
    return types, lines


def pick_fewshot(train_csv: str, k: int, seed: int,
                 executable_csv: str | None = None) -> list[tuple[str, str]]:
    """K examples stratified across sections (question, <PATH> target).

    If executable_csv is given (a *.annotated.csv with a GoldExecutable
    column, row-aligned with train_csv by Question), only rows whose gold
    path actually executes to a non-empty answer are eligible — otherwise a
    few-shot example can teach the model to imitate an unanswerable query
    (~48% of the raw dataset is like this; see executability_report.json)."""
    df = pd.read_csv(train_csv)
    df["_p"] = df["Path"].astype(str).str.extract(r"(<PATH>.*?</PATH>)", flags=re.DOTALL)
    df = df.dropna(subset=["_p"])
    if executable_csv:
        exec_map = pd.read_csv(executable_csv, usecols=["Question", "GoldExecutable"])
        exec_map = exec_map.drop_duplicates(subset=["Question"])
        before = len(df)
        df = df.merge(exec_map, on="Question", how="inner")
        df = df[df["GoldExecutable"] == True]  # noqa: E712
        print(f"[INFO] few-shot pool filtered to executable rows: {len(df)}/{before}")
    per_sec = max(1, k // df["Section"].nunique())
    shots = (df.groupby("Section", group_keys=False)
               .apply(lambda g: g.sample(min(per_sec, len(g)), random_state=seed),
                      include_groups=False))
    if len(shots) > k:
        shots = shots.sample(k, random_state=seed)
    # shots lost the Question index via include_groups=False; recover via df
    shots = df.loc[shots.index]
    return list(zip(shots["Question"], shots["_p"]))


SYSTEM_TEMPLATE = """You are a Cyber Threat Intelligence assistant that answers questions by \
emitting an executable traversal path over a typed MITRE ATT&CK knowledge graph.

GRAPH COMPOSITION
The graph contains nodes of these types:
  {node_types}
Every edge is typed and directed. These are the relations you can traverse, with the
node types they connect (source --relation--> target):
{schema}
Relations whose target is "value" lead to attribute values (platform names, tactic
types, detection text...) rather than entity nodes. All entity relations also exist
in the reverse direction under their inverse name (e.g. uses_malware / used_by_...),
so reasoning can move in both directions.

PATH LANGUAGE
A path is a sequence of steps wrapped in <PATH>...</PATH>, separated by <SEP>. Steps can be:
- a relation from the schema above (follow edges with that label)
- "filter <keywords>"  keep nodes whose description contains the keywords ("and"/"or" supported)
- "is_<type>_type"     start from all nodes of that type (e.g. is_malware_type)
- "select <Name> <Name>" restrict to the named entities, one branch each
- "exec_common <type>" / "exec_difference <type>"  set-intersect / set-difference the branches

IMPORTANT convention: when the question names a specific entity (a malware, group,
campaign, tool...), the traversal starts implicitly FROM that entity's node. Do NOT
put the entity's name or an is_<type>_type/select step in the path — begin directly
with the first relation to follow from it. Only use is_<type>_type (optionally with
select/filter) when the question does not name a starting entity, or when it compares
several named entities (then: is_<type>_type, select <names>, ..., exec_*).

WORKED EXAMPLES
Q: What are the best practices for protecting against .NET malware used by APT32?
Reasoning: APT32 is an intrusion_set, so we start from its node implicitly. We need its
malware (intrusion_set --uses_malware--> malware), keep only .NET ones (filter .NET),
get the techniques those malware use (malware --uses_attack_pattern--> attack_pattern),
and finally the mitigations (attack_pattern --mitigated_by--> course_of_action).
A: <PATH>uses_malware<SEP>filter .NET<SEP>uses_attack_pattern<SEP>mitigated_by</PATH>

Q: How do security experts typically prevent and respond to FALLCHILL and BlackEnergy 3?
Reasoning: two named malware are compared, so we seed all malware nodes
(is_malware_type), select the two entities as separate branches, follow each branch to
its techniques and their mitigations, and intersect the branches' course_of_action sets.
A: <PATH>is_malware_type<SEP>select FALLCHILL BlackEnergy 3<SEP>uses_attack_pattern<SEP>mitigated_by<SEP>exec_common course_of_action</PATH>

Q: What similarities exist in the target assets of spyware and GoldenSpy?
Reasoning: one side is a NAMED entity (GoldenSpy, a malware) and the other is a FILTERED
CLASS (spyware), not a second named entity — so this is not a select/select comparison.
Branch A starts implicitly from GoldenSpy: malware --uses_attack_pattern--> attack_pattern
--targets--> x_mitre_asset. Branch B reaches the same node type by going the OTHER
direction: attack_pattern --used_by_malware--> malware, then filter to keep only spyware,
then --targets--> x_mitre_asset again. Both branches now hold x_mitre_asset sets, so we
exec_common them. (This pattern reaches the filtered class via a reverse traversal
instead of a second select branch — used whenever only one side is a named entity.)
A: <PATH>uses_attack_pattern<SEP>targets<SEP>used_by_attack_pattern<SEP>used_by_malware<SEP>filter spyware<SEP>exec_common x_mitre_asset</PATH>

Respond with ONLY the path, wrapped in <PATH>...</PATH>, no explanation.

More examples:
{examples}"""


SYSTEM_TEMPLATE_COT = SYSTEM_TEMPLATE.replace(
    "Respond with ONLY the path, wrapped in <PATH>...</PATH>, no explanation.",
    "First reason step by step about which relations to follow and why (as in the "
    "worked examples' \"Reasoning:\" lines above), THEN give the final answer wrapped "
    "in <PATH>...</PATH> at the end of your response. Always end with the "
    "<PATH>...</PATH> tag."
)


def build_messages(question: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Few-shot path baseline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True, help="NoCoT train CSV (for vocab + shots)")
    ap.add_argument("--test", required=True, help="Test CSV (Question, Section)")
    ap.add_argument("--graph", default="stix_graph_correct.graphml",
                    help="KG used to derive the typed relation schema")
    ap.add_argument("--exec-csv", default=None,
                    help="*.annotated.csv (Question, GoldExecutable) to restrict "
                         "few-shot examples to answerable gold paths")
    ap.add_argument("--dump-prompt", default=None,
                    help="write the assembled system prompt to this path and exit "
                         "(no model load, no GPU)")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rels = collect_vocabulary(args.train)
    node_types, schema_lines = collect_typed_schema(args.graph, rels)
    shots = pick_fewshot(args.train, args.shots, args.seed, executable_csv=args.exec_csv)
    system_prompt = SYSTEM_TEMPLATE.format(
        node_types=", ".join(node_types),
        schema="\n".join(schema_lines),
        examples="\n".join(f"Q: {q}\nA: {p}" for q, p in shots),
    )
    print(f"[INFO] {len(rels)} relations, {len(schema_lines)} schema lines, "
          f"{len(shots)} shots, system prompt ~{len(system_prompt)} chars")

    if args.dump_prompt:
        with open(args.dump_prompt, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        print(f"[OK] prompt written to {args.dump_prompt} (no model loaded, exiting)")
        return

    test = pd.read_csv(args.test)
    if args.sample and args.sample < len(test):
        # stratified by Section to preserve the operator/length mix
        total = len(test)
        test = (test.groupby("Section", group_keys=False)
                    .apply(lambda g: g.sample(
                        max(1, round(args.sample * len(g) / total)),
                        random_state=args.seed))
                    .reset_index(drop=True))
    print(f"[INFO] evaluating {len(test)} test questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    print(f"[INFO] model loaded across: {set(model.hf_device_map.values())}")

    results = []
    questions = test["Question"].astype(str).tolist()
    for i in range(0, len(questions), args.batch):
        chunk = questions[i:i + args.batch]
        prompts = [tokenizer.apply_chat_template(
            build_messages(q, system_prompt), tokenize=False,
            add_generation_prompt=True) for q in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        for q, seq in zip(chunk, out):
            text = tokenizer.decode(seq[enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            m = re.search(r"<PATH>.*?</PATH>", text, flags=re.DOTALL)
            results.append({"question": q,
                            "generated_path": m.group(0) if m else text.strip()})
        done = min(i + args.batch, len(questions))
        print(f"[{done}/{len(questions)}]", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

    print(f"[OK] {len(results)} generations -> {args.out}")


if __name__ == "__main__":
    main()
