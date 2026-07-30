#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the TITAN path-planner on a modern backbone (default: Qwen2.5-7B-
Instruct). Same LoRA SFT recipe as train_v2.py (r=16, lr 3e-4, effective
batch 16, early stopping on eval_loss) so results stay comparable to the
Phi-3.5 runs — only the base model and chat template differ.

Runs in the plain training venv (.venv), identical stack to the live
Phi-3.5 CoT/NoCoT runs (transformers 4.57.2 / trl 0.23.0 / unsloth 2025.11.1).

vLLM is deliberately NOT used here. Unsloth's colocated fast_inference=True
path needs a vLLM version new enough for unsloth_zoo's LoRA-worker-manager
API, but vLLM is dropping support for the transformers v4 line (the one
unsloth 2025.11.x/transformers 4.57.2 requires) around the same version —
there is currently no version triple that satisfies both, confirmed by
testing three separate pinned installs, not by assumption. Decomposing
avoids the conflict entirely: train here with plain unsloth (this script),
then serve/evaluate the saved LoRA adapter with vLLM's own native
LoRARequest support in the separate, already-proven .venv-vllm environment
(see serve_lora_vllm.py) — that's genuine vLLM usage, just at inference
time instead of training time.

    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=2 python train_qwen.py \
        --data datasets/TEMPLATE_DISJOINT/CoT_TRAIN \
        --out MODELS/qwen25_7b_titan_cot
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers/trl)

import argparse
import os
from typing import Dict, Optional

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

SYSTEM_PROMPT = (
    "You are a Cybersecurity Assistant. Generate a valid relation path to answer the following "
    "question or instruction."
)


def read_splits(base_dir: str, subsample: Optional[int]) -> DatasetDict:
    out = {}
    for split, fname in (("train", "train_dataset.csv"),
                         ("validation", "val_dataset.csv"),
                         ("test", "test_dataset.csv")):
        path = os.path.join(base_dir, fname)
        df = pd.read_csv(path)
        for col in ("Question", "Path"):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing in {path}")
        df = df.dropna(subset=["Question", "Path"])[["Question", "Path"]]
        if subsample:
            df = df.head(subsample)
        out[split] = Dataset.from_pandas(df.reset_index(drop=True))
    return DatasetDict(out)


def load_model_and_tokenizer(model_name: str, max_seq_length: int, chat_template: str,
                             lora_rank: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def apply_chat_template(ds: DatasetDict, tokenizer) -> DatasetDict:
    def _format(batch: Dict[str, list]) -> Dict[str, list]:
        texts = []
        for q, p in zip(batch["Question"], batch["Path"]):
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": p},
            ]
            texts.append(tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    ds = ds.map(_format, batched=True, num_proc=1)
    print("\n[DEBUG] formatted example:\n", ds["train"][0]["text"][:400], "...\n")
    return ds


def main() -> None:
    p = argparse.ArgumentParser(description="TITAN LoRA SFT on a modern backbone (Qwen2.5-7B default)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--chat-template", default="qwen-2.5")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--train-bsz", type=int, default=8)
    p.add_argument("--eval-bsz", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="200 train examples, 10 steps, no early stopping")
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    use_bf16 = is_bfloat16_supported()

    ds = read_splits(args.data, subsample=200 if args.dry_run else None)
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.seq_len, args.chat_template, args.lora_rank)
    ds = apply_chat_template(ds, tokenizer)

    cfg = SFTConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=10 if args.dry_run else -1,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=1 if args.dry_run else 20,
        eval_strategy="no" if args.dry_run else "epoch",
        save_strategy="no" if args.dry_run else "epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=0 if args.dry_run else 50,
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=not args.dry_run,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.seq_len,
        packing=False,
        dataset_num_proc=1,
    )

    callbacks = [] if args.dry_run else [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=cfg,
        callbacks=callbacks,
    )
    trainer.train()

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        trainer.model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        print(f"[OK] adapters + tokenizer saved to {args.out}")
    else:
        print("[OK] dry run completed without errors")


if __name__ == "__main__":
    main()
