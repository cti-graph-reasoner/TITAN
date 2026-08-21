#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-throughput baseline generation via vLLM, for the FULL test set.

Reuses prompt construction (typed schema extraction, executable-filtered
few-shot sampling, the v3 system prompt) from fewshot_path_baseline.py
unchanged, so every baseline model — Llama-3.3-70B, Qwen2.5-72B,
gpt-oss-120b, ... — is compared under byte-identical prompt content. Only
the inference backend differs (vLLM continuous batching instead of a
transformers generate() loop), because the earlier per-question loop does
not scale to the full ~15.6k-row test_heldout split.

Run in the SEPARATE .venv-vllm environment (kept isolated from the
unsloth/trl training venv to avoid dependency conflicts):
    source .venv-vllm/bin/activate
    CUDA_VISIBLE_DEVICES=2,3 python baselines/vllm_baseline.py \
        --model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
        --tp 2 --quantization bitsandbytes \
        --train datasets/TEMPLATE_DISJOINT/NoCoT/train.csv \
        --test datasets/TEMPLATE_DISJOINT/NoCoT/test_heldout.csv \
        --exec-csv datasets/TEMPLATE_DISJOINT/CoT/train.annotated.csv \
        --out baselines/llama33_70b_heldout_full_v3.json

Omit --sample for the full test set (default). All other CLI flags mirror
fewshot_path_baseline.py so results stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fewshot_path_baseline import (  # noqa: E402
    SYSTEM_TEMPLATE,
    SYSTEM_TEMPLATE_COT,
    collect_typed_schema,
    collect_vocabulary,
    pick_fewshot,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM full-test-set baseline generation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size")
    ap.add_argument("--pp", type=int, default=1,
                    help="pipeline_parallel_size (bnb-prequant models reject TP>1; "
                         "use --tp 1 --pp 2 for those)")
    ap.add_argument("--quantization", default=None,
                    help="e.g. bitsandbytes; omit for an unquantized/native checkpoint")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--exec-csv", default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="omit for the FULL test set")
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-prompt", default=None)
    ap.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"],
                    help="for Harmony-format reasoning models (e.g. gpt-oss): controls "
                         "chain-of-thought verbosity via chat_template_kwargs. Leave unset "
                         "for non-reasoning models.")
    ap.add_argument("--cot", action="store_true",
                    help="use the CoT-instruction system prompt (reason step by step, "
                         "then answer) instead of the direct-answer v3 prompt -- same "
                         "schema/shots/decoding, only the final instruction differs. For "
                         "the {prompted x CoT/NoCoT} x {fine-tuned x CoT/NoCoT} ablation.")
    args = ap.parse_args()

    rels = collect_vocabulary(args.train)
    node_types, schema_lines = collect_typed_schema(args.graph, rels)
    shots = pick_fewshot(args.train, args.shots, args.seed, executable_csv=args.exec_csv)
    template = SYSTEM_TEMPLATE_COT if args.cot else SYSTEM_TEMPLATE
    system_prompt = template.format(
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
        total = len(test)
        test = (test.groupby("Section", group_keys=False)
                    .apply(lambda g: g.sample(max(1, round(args.sample * len(g) / total)),
                                              random_state=args.seed),
                          include_groups=False))
        test = test.reset_index(drop=True)
    questions = test["Question"].astype(str).tolist()
    print(f"[INFO] generating for {len(questions)} questions "
          f"({'full test set' if not args.sample else 'sampled'})")

    from vllm import LLM, SamplingParams  # deferred: only needed in .venv-vllm

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    conversations = [
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": q}]
        for q in questions
    ]

    chat_kwargs = ({"reasoning_effort": args.reasoning_effort}
                   if args.reasoning_effort else None)
    outputs = llm.chat(conversations, sampling, use_tqdm=True,
                       chat_template_kwargs=chat_kwargs)

    results = [{"question": q, "generated_path": o.outputs[0].text.strip()}
              for q, o in zip(questions, outputs)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[OK] {len(results)} generations -> {args.out}")


if __name__ == "__main__":
    main()
