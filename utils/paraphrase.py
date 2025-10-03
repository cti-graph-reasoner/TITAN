#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate target (Objective) improvements and question variations from YAML templates
using an open-source LLM (e.g., Llama-3.1-8B-Instruct on Hugging Face).

Key fixes vs. original:
- Proper chat usage with tokenizer.apply_chat_template (not passing messages to a text pipeline).
- Correct extraction of generated text (string, not nested dicts).
- Safer retries and validation logic for bracket-preservation and malware-term preservation.
- Clean I/O handling for YAML and CSV.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import transformers
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LLM wrapper (chat with HF models that provide a chat template)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device_map: Union[str, dict] = "auto"   # e.g., "cuda:5" or {"": 0}
    torch_dtype: Optional[torch.dtype] = None  # None -> auto
    trust_remote_code: bool = False
    # Generation defaults
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class LLM:
    """
    Thin wrapper around a HF causal LM with chat template support.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        # Auto dtype if not specified
        dtype = cfg.torch_dtype
        if dtype is None:
            # Prefer bfloat16 if available, else float16, else float32
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            else:
                dtype = torch.float32

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=True,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )

    @torch.inference_mode()
    def chat(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Generate a chat response given a list of messages:
        messages = [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns: generated assistant text (string).
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        temperature = temperature if temperature is not None else self.cfg.temperature
        top_p = top_p if top_p is not None else self.cfg.top_p
        do_sample = do_sample if do_sample is not None else self.cfg.do_sample

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # add the assistant turn
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return generated.strip()


def load_model(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    device_map: Union[str, dict] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> LLM:
    """
    Create an LLM instance with sensible defaults.
    """
    cfg = LLMConfig(
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return LLM(cfg)


# ---------------------------------------------------------------------------
# Text extraction / validation helpers
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[.*?\]")

def extract_bracketed_elements(text: str) -> List[str]:
    """Return all substrings enclosed in square brackets '[...]'."""
    return _BRACKET_RE.findall(text or "")


_MALWARE_TERMS = [
    "ransomware", "spyware", "backdoor", "worm", "trojan", "rat", "in-memory", "steal", "spy"
]

def extract_malware_terms(text: str) -> List[str]:
    """Return malware-related terms present in the text (case-insensitive)."""
    t = (text or "").lower()
    found = [term for term in _MALWARE_TERMS if term in t]
    # Return original case for a nicer output (optional)
    return found


def _lines_that_look_like_questions(text: str) -> List[str]:
    """
    Split text lines and keep those that look like questions or enumerations.
    Heuristics: line contains '?', or starts with 1., 2., etc.
    """
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "?" in line or line[:2].isdigit() or any(line.startswith(f"{i}. ") for i in range(1, 21)):
            # Remove enumeration prefix like "1. " if present
            if ". " in line[:4]:
                line = line.split(". ", 1)[-1].strip()
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def improve_target_description(
    llm: LLM,
    question: str,
    max_retries: int = 10,
) -> Optional[str]:
    """
    Ask the model to produce a single-sentence Objective that preserves:
    - same bracketed spans (if any)
    - no brackets introduced if the question had none
    """
    original_brackets = set(extract_bracketed_elements(question))

    sys = {
        "role": "system",
        "content": (
            "You are a precise rewriting assistant. You will write a single-sentence Objective that "
            "concisely captures the purpose of the user's question. Preserve any bracketed spans exactly as-is. "
            "Do not add brackets if none appear in the question. The Objective must be self-contained."
        ),
    }

    for _ in range(max_retries):
        usr = {
            "role": "user",
            "content": (
                "Write one Objective for the following question. Keep the original meaning. "
                "If the question contains spans in square brackets '[...]', they must appear verbatim in the Objective; "
                "if none appear, the Objective must not contain brackets.\n\n"
                f"Question: {question}"
            ),
        }
        generated = llm.chat([sys, usr])

        # Basic cleanup
        objective = generated.strip().strip('"').strip()
        # Check bracket constraints
        gen_brackets = set(extract_bracketed_elements(objective))

        if original_brackets:
            if gen_brackets == original_brackets:
                return objective
        else:
            if not gen_brackets:
                return objective

    # No valid objective found within retries
    return None


def generate_question_variations(
    llm: LLM,
    question: str,
    num_variations: int = 10,
    max_retries: int = 10,
) -> List[str]:
    """
    Generate paraphrase variations of a question while preserving:
    - bracketed spans (verbatim),
    - malware terms (verbatim, case-insensitive test).
    Returns up to num_variations items.
    """
    original_brackets = set(extract_bracketed_elements(question))
    original_malware = set(extract_malware_terms(question))

    sys = {
        "role": "system",
        "content": (
            "You are a careful paraphraser. You produce diverse question rewrites that preserve critical tokens."
        ),
    }

    valid: List[str] = []
    for _ in range(max_retries):
        usr = {
            "role": "user",
            "content": (
                f"Produce {num_variations} paraphrased questions that preserve the original meaning. "
                "Vary syntax and phrasing; do NOT create multiple-choice questions. "
                "If the original question contains any spans in square brackets '[...]', copy them verbatim, "
                "including brackets, without changes. "
                "If it contains malware-related tokens such as 'ransomware', 'spyware', 'backdoor', 'worm', 'trojan', "
                "'RAT', 'in-memory', 'steal', 'spy', preserve them exactly as written. "
                f"\n\nOriginal question:\n{question}"
            ),
        }
        generated = llm.chat([sys, usr], max_new_tokens=512)
        # Extract candidate lines
        candidates = _lines_that_look_like_questions(generated)

        # Validate each candidate
        for cand in candidates:
            cand_brackets = set(extract_bracketed_elements(cand))
            cand_malware = set(extract_malware_terms(cand))

            if original_brackets and cand_brackets != original_brackets:
                continue
            if not original_brackets and cand_brackets:
                continue
            if cand_malware != original_malware:
                continue

            valid.append(cand)
            if len(valid) >= num_variations:
                break

        if len(valid) >= num_variations:
            break

    return valid[:num_variations]


# ---------------------------------------------------------------------------
# YAML / CSV I/O
# ---------------------------------------------------------------------------

def load_templates(yaml_file: str) -> dict:
    """Load question templates YAML."""
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stringify_variations(variations: Union[str, Sequence[str], None]) -> str:
    """
    Convert variations to a single string for CSV:
    - list/tuple -> joined by '; '
    - str -> unchanged
    - None/other -> empty
    """
    if variations is None:
        return ""
    if isinstance(variations, str):
        return variations
    if isinstance(variations, (list, tuple)):
        return "; ".join(map(str, variations))
    return ""


def save_to_csv(output_file: str, rows: List[Tuple[str, Union[str, List[str]]]]) -> None:
    """
    Write CSV with columns: 'Original Question', 'Variations'
    'Variations' may contain a single improved Objective (string) or list of question rewrites.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Original Question", "Variations"])
        for original, variations in rows:
            w.writerow([original, _stringify_variations(variations)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load model (adjust device_map if you want a specific GPU like "cuda:5")
    llm = load_model(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=None,  # auto-pick based on hardware
    )

    # Load useful templates (expects structure like templates[section] -> list of dicts with 'question' and 'target')
    useful_template_file = "../useful_cot.yaml"
    useful_templates = load_templates(useful_template_file)

    # Collect questions (and associated target templates if needed)
    question_items: List[Tuple[str, Union[str, List[str]]]] = []
    for section in useful_templates.get("templates", {}):
        for entry in tqdm(useful_templates["templates"][section], desc=f"Scanning '{section}' ..."):
            if not isinstance(entry, dict):
                continue
            question_text = entry.get("question")
            # target may be a list; we don't critically need it to improve objectives
            # but we keep for debug/printing if desired:
            target_tpl = entry.get("target", [])
            if isinstance(target_tpl, list) and target_tpl:
                _ = target_tpl[0]

            if question_text:
                question_items.append((question_text, ""))

    # For each question, produce a single improved Objective (fallback if validation fails)
    results: List[Tuple[str, Union[str, List[str]]]] = []
    for original_question, _ in tqdm(question_items, desc="Generating objectives ..."):
        improved = improve_target_description(llm, original_question, max_retries=10)
        if improved is None:
            improved = "No valid Objective generated."
        # Store as a string (single objective) in the CSV 'Variations' column
        results.append((original_question, improved))

    output_file = "target_variations.csv"
    save_to_csv(output_file, results)
    print(f"Questions and targets saved to {output_file}")


if __name__ == "__main__":
    main()
