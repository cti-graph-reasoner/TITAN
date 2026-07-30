#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-as-judge sanity check for the paraphrase-robustness experiment
(Figure~\\ref{fig:paraphrase-robustness} in the paper).

Motivation: every reported paraphrase-robustness number implicitly assumes
the paraphrased question still asks for the same answer as the original
(utils/paraphrase.py enforces this at generation time via bracket/malware-term
preservation, but never re-verifies it downstream). If some paraphrases had
silently drifted in meaning, "model gets worse under paraphrasing" would be
partly confounded with "the question changed."

Two uses:
  1. The 473-row subset where every one of the 8 fine-tuned configurations
     failed on the paraphrased version (highest-value rows to audit).
  2. The full 1,502-row paraphrase subsample, for an overall semantic-drift
     rate to compare the failing-subset rate against.

Runs with 3 independent judges: Qwen2.5-72B-Instruct and Llama-3.3-70B-Instruct
(both bitsandbytes-prequantized -- these do NOT support tensor parallelism in
this vLLM version, "Prequant BitsAndBytes models with tensor parallelism is
not supported. Please try with pipeline parallelism", hence --pp not --tp
below), and openai/gpt-oss-120b (native mxfp4 quantization, no --quantization
flag needed, reusing the exact pipeline_parallel_size=2 / max_model_len=5120
/ trust_remote_code config already proven for it in this repo's own
logs_gptoss_sparql_cot.log).

Run in the vLLM venv, on free GPUs:
    source .venv-vllm/bin/activate
    CUDA_VISIBLE_DEVICES=1,4 python utils/judge_paraphrase_semantics.py \
        --in /tmp/.../paraphrase_merged.csv \
        --out baselines/paraphrase_judge_qwen72b_full.csv \
        --model unsloth/Qwen2.5-72B-Instruct-bnb-4bit --quantization bitsandbytes --pp 2
"""

from __future__ import annotations

import argparse
import re

import pandas as pd
from vllm import LLM, SamplingParams

JUDGE_SYSTEM = (
    "You are auditing a paraphrasing pipeline used to build an NLP benchmark. "
    "You will be shown an ORIGINAL question and a PARAPHRASED rewrite of it. "
    "The paraphrase is supposed to preserve the original question's meaning and "
    "the exact entities and relationship it asks about, varying only surface "
    "wording and syntax. Judge strictly: a correct answer to one must also be a "
    "correct answer to the other, referring to the same entities and the same "
    "relationship being asked about. Superficial rewording is fine; a changed "
    "entity, a changed relation, a narrowed or broadened scope, or an added/"
    "dropped constraint is NOT fine."
)

JUDGE_USER_TEMPLATE = """ORIGINAL: {original}
PARAPHRASE: {paraphrase}

Respond in exactly this format, nothing else:
Verdict: EQUIVALENT or NOT_EQUIVALENT
Reason: <one concise sentence>"""

VERDICT_RE = re.compile(r"Verdict:\s*(EQUIVALENT|NOT_EQUIVALENT)", re.IGNORECASE)
REASON_RE = re.compile(r"Reason:\s*(.+)", re.IGNORECASE)


def build_prompt(tokenizer, original: str, paraphrase: str) -> str:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(original=original, paraphrase=paraphrase)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-judge semantic-equivalence check for paraphrase pairs")
    ap.add_argument("--in", dest="in_csv", required=True, help="CSV with Question (paraphrased) + OriginalQuestion columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit")
    ap.add_argument("--quantization", default="bitsandbytes", help="pass 'none' for natively-quantized models like gpt-oss")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel -- NOT supported for prequant bnb models")
    ap.add_argument("--pp", type=int, default=1, help="pipeline parallel -- use this instead of --tp for bnb models")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=150,
                     help="raise this for native-reasoning models (e.g. gpt-oss) that emit an "
                          "analysis/reasoning trace before the final Verdict line")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    assert "Question" in df.columns and "OriginalQuestion" in df.columns, \
        "expected columns 'Question' (paraphrased) and 'OriginalQuestion'"

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )
    if args.quantization.lower() != "none":
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    prompts = [build_prompt(tokenizer, row.OriginalQuestion, row.Question) for row in df.itertuples()]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling)

    verdicts, reasons = [], []
    for out in outputs:
        text = out.outputs[0].text
        vm = VERDICT_RE.search(text)
        rm = REASON_RE.search(text)
        verdicts.append(vm.group(1).upper() if vm else "UNPARSEABLE")
        reasons.append(rm.group(1).strip() if rm else text.strip()[:200])

    df["judge_verdict"] = verdicts
    df["judge_reason"] = reasons
    df.to_csv(args.out, index=False)

    n = len(df)
    n_equiv = (df["judge_verdict"] == "EQUIVALENT").sum()
    n_not = (df["judge_verdict"] == "NOT_EQUIVALENT").sum()
    n_bad = (df["judge_verdict"] == "UNPARSEABLE").sum()
    print(f"n={n}  EQUIVALENT={n_equiv} ({100*n_equiv/n:.1f}%)  "
          f"NOT_EQUIVALENT={n_not} ({100*n_not/n:.1f}%)  UNPARSEABLE={n_bad}")


if __name__ == "__main__":
    main()
