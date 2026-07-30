#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a paraphrased variant of a TITAN test split, to check whether the
fine-tuned models are doing genuine multi-hop reasoning or just matching the
surface phrasing of the template-generated questions.

Unlike utils/paraphrase.py (built for bracket-templated YAML question sets),
our actual Question columns have NO bracket markup -- the entity name is
just a bare substring (e.g. "...used by the EXOTIC LILY group?"). So the
constraint enforced here is: reword everything BUT keep every entity name
that titan.evaluate.extract_entities() pulls from the reference Path
verbatim, since that name is how the model is expected to seed its
traversal -- change it and we'd be testing something else (whether the
model can find an unknown-alias entity), not phrasing robustness.

Samples a stratified (by Section) subset rather than the full split, since
the point is a robustness CHECK (bootstrap CI over a few hundred/thousand
rows), not a second full benchmark -- the paraphrased CSV is a drop-in
replacement for the original (same Path/Section/... columns) so any
existing eval/serving script can consume it unchanged.

Run (single GPU, e.g. GPU2):
    CUDA_VISIBLE_DEVICES=2 python3 utils/build_paraphrase_eval.py \
        --data datasets/TEMPLATE_DISJOINT/CoT/test_heldout.annotated.csv \
        --sample 1000 --out datasets/TEMPLATE_DISJOINT/CoT/test_heldout.paraphrased.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from titan import evaluate as ET  # noqa: E402
from utils.paraphrase import LLM, LLMConfig  # noqa: E402

SYSTEM_PROMPT = (
    "You are a careful paraphraser for a cyber-threat-intelligence QA dataset. "
    "You reword questions -- different sentence structure, synonyms, reordering -- "
    "while preserving their exact meaning. Output ONLY the single rewritten "
    "question, nothing else: no preamble, no quotes, no numbering."
)


def build_user_prompt(question: str, entities: list) -> str:
    if entities:
        preserve = "; ".join(f'"{e}"' for e in entities)
        constraint = (
            f"The rewrite MUST contain this/these exact substring(s) verbatim, "
            f"unchanged: {preserve}. Everything else may be reworded freely."
        )
    else:
        constraint = "Reword freely; there is no fixed entity name to preserve here."
    return (
        f"Rewrite this question with different phrasing but the identical meaning. "
        f"{constraint}\n\nOriginal question: {question}\n\nRewritten question:"
    )


def clean_output(text: str) -> str:
    text = text.strip().strip('"').strip()
    # drop a leading "Rewritten question:" echo if the model repeats the cue
    text = re.sub(r"^(rewritten question:\s*)", "", text, flags=re.IGNORECASE)
    # keep only the first line -- some models add trailing commentary
    return text.splitlines()[0].strip()


def paraphrase_one(llm: LLM, question: str, entities: list, max_retries: int) -> tuple:
    """Returns (paraphrase_or_None, n_attempts)."""
    user_prompt = build_user_prompt(question, entities)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": user_prompt}]
    for attempt in range(1, max_retries + 1):
        raw = llm.chat(messages, max_new_tokens=160)
        cand = clean_output(raw)
        if not cand or cand.lower() == question.lower():
            continue
        if all(e in cand for e in entities):
            return cand, attempt
    return None, max_retries


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    total = len(df)
    out = (df.groupby("Section", group_keys=False)
             .apply(lambda g: g.sample(max(1, round(n * len(g) / total)),
                                       random_state=seed)))
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    sub = stratified_sample(df, args.sample, args.seed)
    print(f"[INFO] sampled {len(sub)}/{len(df)} rows, "
          f"section distribution:\n{sub['Section'].value_counts()}")

    print(f"[INFO] loading {args.model} ...")
    llm = LLM(LLMConfig(model_name=args.model))

    rows = []
    n_fallback = 0
    for r in tqdm(sub.itertuples(), total=len(sub), desc="paraphrasing"):
        entities = ET.extract_entities(r.Path)
        para, attempts = paraphrase_one(llm, r.Question, entities, args.max_retries)
        rec = r._asdict()
        rec.pop("Index", None)
        rec["OriginalQuestion"] = r.Question
        if para is not None:
            rec["Question"] = para
            rec["Paraphrased"] = True
        else:
            n_fallback += 1
            rec["Paraphrased"] = False  # keep original Question, flagged
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] {len(out_df)} rows -> {args.out} "
          f"({n_fallback} fell back to the original phrasing after "
          f"{args.max_retries} failed retries, {100 * n_fallback / len(out_df):.1f}%)")


if __name__ == "__main__":
    main()
