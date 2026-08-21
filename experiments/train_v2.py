#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the TITAN path-planner (LoRA SFT via Unsloth) — updated for
transformers>=4.57 / trl>=0.23 (SFTConfig API). Same recipe as train.py:
Phi-3.5-mini 4-bit, LoRA r=16, lr 3e-4, effective batch 16, early stopping.

Expects a directory with train_dataset.csv / val_dataset.csv / test_dataset.csv,
each containing at least Question and Path columns.

--dry-run trains on a small subset for a few steps to validate the stack.
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers/trl)

import argparse
import os
from typing import Dict

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


def read_splits(base_dir: str, subsample: int | None) -> DatasetDict:
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


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def apply_chat_template(ds: DatasetDict, tokenizer) -> DatasetDict:
    def _format(batch: Dict[str, list]) -> Dict[str, list]:
        texts = []
        for q, p in zip(batch["Question"], batch["Path"]):
            convo = [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": q},
                {"from": "gpt", "value": p},
            ]
            texts.append(tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    ds = ds.map(_format, batched=True, num_proc=1)
    print("\n[DEBUG] formatted example:\n", ds["train"][0]["text"][:400], "...\n")
    return ds


def main() -> None:
    p = argparse.ArgumentParser(description="TITAN LoRA SFT (trl>=0.23 API)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="unsloth/Phi-3.5-mini-instruct")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-bsz", type=int, default=8)
    p.add_argument("--eval-bsz", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="200 examples, 10 steps, no early stopping")
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    use_bf16 = is_bfloat16_supported()

    ds = read_splits(args.data, subsample=200 if args.dry_run else None)
    model, tokenizer = load_model_and_tokenizer(args.model, args.seq_len)
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=cfg,
        callbacks=None if args.dry_run else [EarlyStoppingCallback(early_stopping_patience=3)],
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
