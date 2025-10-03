#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a LoRA SFT model (Phi-3.5-mini-instruct via Unsloth) on TITAN navigation data.

Expected CSVs (default paths):
- SMARTER_COMPLETE_DATASET/train_dataset.csv
- SMARTER_COMPLETE_DATASET/val_dataset.csv
- SMARTER_COMPLETE_DATASET/test_dataset.csv

Each CSV must contain at least the columns:
- "Question" : user prompt
- "Path"     : target sequence (graph navigation path)

Main features:
- Robust dataset loading & validation
- Chat template application (phi-3 mapping)
- LoRA fine-tuning with Unsloth + TRL SFTTrainer
- Early stopping & best model selection
- Saves PEFT adapters + tokenizer
"""

from __future__ import annotations
import argparse
import os
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


# -------------------------
# Global system instruction
# -------------------------
SYSTEM_PROMPT = (
    "You are a Cybersecurity Assistant. Generate a valid relation path to answer the following "
    "question or instruction."
)


# -------------------------
# Dataset utilities
# -------------------------
def _read_split_csvs(base_dir: str) -> DatasetDict:
    """Load train/val/test CSVs from a base directory and return a HF DatasetDict."""
    train_path = os.path.join(base_dir, "train_dataset.csv")
    val_path   = os.path.join(base_dir, "val_dataset.csv")
    test_path  = os.path.join(base_dir, "test_dataset.csv")

    for p in (train_path, val_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Validate columns
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        for col in ("Question", "Path"):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {name} split ({base_dir}).")
        # Drop rows with NaNs in required fields
        before = len(df)
        df.dropna(subset=["Question", "Path"], inplace=True)
        if len(df) < before:
            print(f"[INFO] Dropped {before-len(df)} rows with NaNs in {name} split.")

    return DatasetDict(
        train=Dataset.from_pandas(train_df.reset_index(drop=True)),
        validation=Dataset.from_pandas(val_df.reset_index(drop=True)),
        test=Dataset.from_pandas(test_df.reset_index(drop=True)),
    )


def apply_chat_template(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """
    Convert rows (Question, Path) into a single 'text' field using a chat template:
    system → user → assistant.
    """
    def _format(batch: Dict[str, list]) -> Dict[str, list]:
        texts = []
        for question, path in zip(batch["Question"], batch["Path"]):
            convo = [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human",  "value": question},
                {"from": "gpt",    "value": path},
            ]
            txt = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            texts.append(txt)
        return {"text": texts}

    dataset = dataset.map(_format, batched=True, num_proc=1)
    # Quick sanity print
    print("\n[DEBUG] Example formatted text:\n", dataset["train"][0]["text"][:500], "...\n")
    return dataset


# -------------------------
# Model loading
# -------------------------
def load_model_and_tokenizer(
    model_name: str = "unsloth/Phi-3.5-mini-instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    add_extra_tokens: bool = False,
):
    """
    Load Unsloth model in 4-bit, attach LoRA, and set chat template to 'phi-3'.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto
    )

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Apply Phi-3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    if add_extra_tokens:
        tokenizer.add_tokens(["<SEP>", "<PATH>", "</PATH>"])
        model.resize_token_embeddings(len(tokenizer))

    # Padding setup
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# -------------------------
# Training
# -------------------------
def train(
    dataset_dir: str = "SMARTER_COMPLETE_DATASET",
    output_dir: str = "MODELS/phi_smarter",
    model_name: str = "unsloth/Phi-3.5-mini-instruct",
    learning_rate: float = 3e-4,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    grad_accum_steps: int = 2,
    num_epochs: int = 8,
    max_seq_length: int = 2048,
    seed: int = 42,
):
    # Device + dtype safety
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    use_bf16 = is_bfloat16_supported()
    use_fp16 = not use_bf16

    # Load data
    ds = _read_split_csvs(dataset_dir)

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        add_extra_tokens=False,
    )

    # Build text field with chat template
    ds = apply_chat_template(ds, tokenizer)

    # Trainer
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        report_to="none",  # set to "tensorboard" if you configure it
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # keep consistent with model load
        dataset_num_proc=1,
        packing=False,  # one example per sequence; set True to pack if your data allows
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    trainer.train()

    # Save best adapters + tokenizer (PEFT weights)
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] Saving PEFT adapters and tokenizer...")
    trainer.model.save_pretrained(output_dir)   # saves LoRA adapters
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] Model artifacts saved to: {output_dir}")


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TITAN LoRA SFT model (Phi-3 via Unsloth).")
    p.add_argument("--data", default="SMARTER_COMPLETE_DATASET", help="Folder with train/val/test CSVs")
    p.add_argument("--out",  default="MODELS/phi_smarter", help="Output directory for model artifacts")
    p.add_argument("--model", default="unsloth/Phi-3.5-mini-instruct", help="HF model name")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--train-bsz", type=int, default=8, help="Per-device train batch size")
    p.add_argument("--eval-bsz", type=int, default=8, help="Per-device eval batch size")
    p.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    p.add_argument("--seq-len", type=int, default=2048, help="Max sequence length")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_dir=args.data,
        output_dir=args.out,
        model_name=args.model,
        learning_rate=args.lr,
        train_batch_size=args.train_bsz,
        eval_batch_size=args.eval_bsz,
        grad_accum_steps=args.grad_accum,
        num_epochs=args.epochs,
        max_seq_length=args.seq_len,
        seed=args.seed,
    )
