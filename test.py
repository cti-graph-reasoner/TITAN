#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive TITAN tester:
- Loads a fine-tuned (or base) Unsloth model
- Takes a natural-language CTI question
- Generates a <PATH> ... </PATH> navigation plan
- Parses entities and executes the plan on the graph via graph_algorithm module
"""

from __future__ import annotations
import os
import re
import sys
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
import pandas as pd  # (not strictly needed, but often handy for debugging)
from datasets import Dataset, DatasetDict  # noqa
from transformers import (
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# Import your local graph utilities (make sure the filename is graph_algorithm.py)
import graph_algorithm as GA


# -------------------------
# Constants
# -------------------------
SYSTEM_PROMPT = (
    "You are a Cybersecurity Assistant. Generate a valid relation path to answer the "
    "following question or instruction."
)

DEFAULT_MODEL = "MODELS/phi_smarter"
DEFAULT_NAMES_FILE = "NAMES.txt"
DEFAULT_GRAPH_FILE = "stix_graph_correct.graphml"
DEFAULT_REL_DESC_FILE = "Relationship_Descriptions.txt"

MAX_NEW_TOKENS = 1024
MAX_SEQ_LENGTH = 2048


# -------------------------
# Helpers
# -------------------------
def ensure_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] {label} not found at: {path}")


def load_names(names_file: str) -> set:
    ensure_file(names_file, "Names file")
    with open(names_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in tqdm(f, desc="Loading MITRE ATT&CK names...") if line.strip()}


def load_relationship_description(rel_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build a dict: source -> list[(target, description)]
    Expected lines contain tokens: 'SOURCE: ... , TARGET: ... , DESCRIPTION: ...'
    """
    ensure_file(rel_file, "Relationship description file")
    source_target_rel: Dict[str, List[Tuple[str, str]]] = {}
    with open(rel_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Building Relationship Maps ..."):
        try:
            source = line.split("SOURCE: ", 1)[1].split(",", 1)[0].strip()
            target = line.split("TARGET: ", 1)[1].split(",", 1)[0].strip()
            rel    = line.split("DESCRIPTION: ", 1)[1].strip()
        except Exception:
            # Skip malformed lines
            continue

        source_target_rel.setdefault(source, []).append((target, rel))
    return source_target_rel


def extract_entities_from_output(text: str) -> List[str]:
    """
    Extract one or multiple entities from the generated CoT text.
    This mirrors the patterns used by your CoT prompts.
    """
    try:
        # Single entity pattern
        if "Let's reason step by step, starting from the entity" in text:
            seg = text.split("Let's reason step by step, starting from the entity ", 1)[1]
            entity = seg.split(" (", 1)[0].strip().strip("'")
            return [entity] if entity else []

        # Multi-entity pattern
        if "We are working with the following entities:" in text:
            seg = text.split("We are working with the following entities:", 1)[1]
            # Cut at an obvious next header if present
            for stop_token in ("**Objective**", "**Starting Point", "**Step-by-Step", "**Entity"):
                if stop_token in seg:
                    seg = seg.split(stop_token, 1)[0]
            # Lines like "- 'MalwareA' is a malware"
            entities = []
            for raw in seg.split("-"):
                raw = raw.strip()
                if not raw:
                    continue
                raw = raw.replace(" is a malware", "").replace(" is a tool", "")
                raw = raw.replace(" is an intrusion set", "").replace("'", "")
                if raw:
                    # keep the first token up to newline
                    entities.append(raw.splitlines()[0].strip())
            # Filter empties
            return [e for e in entities if e]
    except Exception:
        pass
    return []


def extract_final_path(text: str) -> str:
    """
    Extract the final <PATH>...</PATH> string.
    Looks for the 'The completed path is:' cue; falls back to searching a tag.
    """
    path_block = None
    if "The completed path is:" in text:
        path_block = text.split("The completed path is:", 1)[1]
    else:
        path_block = text

    # Extract the first <PATH>...</PATH> occurrence
    m = re.search(r"<PATH>.*?</PATH>", path_block, flags=re.DOTALL)
    if not m:
        raise ValueError("No <PATH>...</PATH> found in model output.")
    return m.group(0)


# -------------------------
# Generation stopping on </PATH>
# -------------------------
class StopOnEndPath(StoppingCriteria):
    """Stop generation once the decoded sequence ends with '</PATH>' (after stripping specials)."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.end_tag = "</PATH>"

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return text.strip().endswith(self.end_tag)


# -------------------------
# Model loading
# -------------------------
def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto
    )

    # Apply Phi-3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    # Inference settings
    FastLanguageModel.for_inference(model)  # enables faster inference
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# -------------------------
# Interactive loop
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive TITAN tester.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model or adapter path")
    parser.add_argument("--names", default=DEFAULT_NAMES_FILE, help="NAMES.txt path")
    parser.add_argument("--graph", default=DEFAULT_GRAPH_FILE, help="GraphML path")
    parser.add_argument("--rels",  default=DEFAULT_REL_DESC_FILE, help="Relationship_Descriptions.txt path")
    args = parser.parse_args()

    # Load model/tokenizer
    print(f"[INFO] Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Build stopping criteria & streamer
    stopping = StoppingCriteriaList([StopOnEndPath(tokenizer)])
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Load names, graph, and relationship texts
    names = load_names(args.names)
    print("\n[INFO] Loading graph ...\n")
    GA_graph = GA.load_graph(args.graph)
    rel_map = load_relationship_description(args.rels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    while True:
        try:
            prompt = input("\nINSERT A CTI QUERY (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not prompt or prompt.lower() == "exit":
            print("[INFO] Bye.")
            break

        # Build chat messages
        messages = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": prompt},
        ]

        # Tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # must add for generation
            return_tensors="pt",
        ).to(device)

        # Generate until </PATH>
        print("\n[MODEL OUTPUT]\n")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                stopping_criteria=stopping,
                streamer=streamer,
            )

        # Decode
        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract entities & path
        try:
            entities = extract_entities_from_output(text)
            if entities:
                # snap each entity to the closest known name
                resolved = []
                for ent in entities:
                    close = GA.find_closest_entity(ent, names)
                    print(f"[ENTITY] '{ent}' -> '{close}'")
                    resolved.append(close)
                entities = resolved
            else:
                print("[WARN] No entities detected in output; proceeding without entity constraint.")

            final_path = extract_final_path(text)
            print("\n[FINAL PATH]\n", final_path, "\n")

            # Execute path
            cleaned = GA.extract_path_elements(final_path)
            print("[DEBUG] Cleaned path steps:", cleaned)

            if len(entities) <= 1:
                response, results = GA.follow_graph(GA_graph, entities, cleaned, rel_map)
                print("\n### FINAL RESULTS (single-entity mode) ###\n")
                # results is a set/list depending on your GA implementation
                if isinstance(results, (list, set, tuple)):
                    for i, elem in enumerate(list(results), 1):
                        print(f"{i} - {elem}")
                else:
                    print(results)

                print("\n### COMPLETE REASONING PLAN ###\n")
                print(response)

            else:
                response, results = GA.follow_graph_n_entities(GA_graph, entities, cleaned, rel_map)
                print("\n### FINAL RESULTS (multi-entity mode) ###\n")
                if isinstance(results, dict):
                    for ent, by_type in results.items():
                        print(f"\n[ENTITY] {ent}")
                        for t, items in by_type.items():
                            print(f"  - {t}: {sorted(list(items))}")
                else:
                    print(results)

        except Exception as e:
            print(f"[ERROR] {e}")
            print("[-] No result available.")


if __name__ == "__main__":
    # Make local imports (graph_algorithm.py) resolvable when running from repo root
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    main()
1~#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive TITAN tester:
- Loads a fine-tuned (or base) Unsloth model
- Takes a natural-language CTI question
- Generates a <PATH> ... </PATH> navigation plan
- Parses entities and executes the plan on the graph via graph_algorithm module
"""

from __future__ import annotations
import os
import re
import sys
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
import pandas as pd  # (not strictly needed, but often handy for debugging)
from datasets import Dataset, DatasetDict  # noqa
from transformers import (
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# Import your local graph utilities (make sure the filename is graph_algorithm.py)
import graph_algorithm as GA


# -------------------------
# Constants
# -------------------------
SYSTEM_PROMPT = (
    "You are a Cybersecurity Assistant. Generate a valid relation path to answer the "
    "following question or instruction."
)

DEFAULT_MODEL = "MODELS/phi_smarter"
DEFAULT_NAMES_FILE = "NAMES.txt"
DEFAULT_GRAPH_FILE = "stix_graph_correct.graphml"
DEFAULT_REL_DESC_FILE = "Relationship_Descriptions.txt"

MAX_NEW_TOKENS = 1024
MAX_SEQ_LENGTH = 2048


# -------------------------
# Helpers
# -------------------------
def ensure_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] {label} not found at: {path}")


def load_names(names_file: str) -> set:
    ensure_file(names_file, "Names file")
    with open(names_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in tqdm(f, desc="Loading MITRE ATT&CK names...") if line.strip()}


def load_relationship_description(rel_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build a dict: source -> list[(target, description)]
    Expected lines contain tokens: 'SOURCE: ... , TARGET: ... , DESCRIPTION: ...'
    """
    ensure_file(rel_file, "Relationship description file")
    source_target_rel: Dict[str, List[Tuple[str, str]]] = {}
    with open(rel_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Building Relationship Maps ..."):
        try:
            source = line.split("SOURCE: ", 1)[1].split(",", 1)[0].strip()
            target = line.split("TARGET: ", 1)[1].split(",", 1)[0].strip()
            rel    = line.split("DESCRIPTION: ", 1)[1].strip()
        except Exception:
            # Skip malformed lines
            continue

        source_target_rel.setdefault(source, []).append((target, rel))
    return source_target_rel


def extract_entities_from_output(text: str) -> List[str]:
    """
    Extract one or multiple entities from the generated CoT text.
    This mirrors the patterns used by your CoT prompts.
    """
    try:
        # Single entity pattern
        if "Let's reason step by step, starting from the entity" in text:
            seg = text.split("Let's reason step by step, starting from the entity ", 1)[1]
            entity = seg.split(" (", 1)[0].strip().strip("'")
            return [entity] if entity else []

        # Multi-entity pattern
        if "We are working with the following entities:" in text:
            seg = text.split("We are working with the following entities:", 1)[1]
            # Cut at an obvious next header if present
            for stop_token in ("**Objective**", "**Starting Point", "**Step-by-Step", "**Entity"):
                if stop_token in seg:
                    seg = seg.split(stop_token, 1)[0]
            # Lines like "- 'MalwareA' is a malware"
            entities = []
            for raw in seg.split("-"):
                raw = raw.strip()
                if not raw:
                    continue
                raw = raw.replace(" is a malware", "").replace(" is a tool", "")
                raw = raw.replace(" is an intrusion set", "").replace("'", "")
                if raw:
                    # keep the first token up to newline
                    entities.append(raw.splitlines()[0].strip())
            # Filter empties
            return [e for e in entities if e]
    except Exception:
        pass
    return []


def extract_final_path(text: str) -> str:
    """
    Extract the final <PATH>...</PATH> string.
    Looks for the 'The completed path is:' cue; falls back to searching a tag.
    """
    path_block = None
    if "The completed path is:" in text:
        path_block = text.split("The completed path is:", 1)[1]
    else:
        path_block = text

    # Extract the first <PATH>...</PATH> occurrence
    m = re.search(r"<PATH>.*?</PATH>", path_block, flags=re.DOTALL)
    if not m:
        raise ValueError("No <PATH>...</PATH> found in model output.")
    return m.group(0)


# -------------------------
# Generation stopping on </PATH>
# -------------------------
class StopOnEndPath(StoppingCriteria):
    """Stop generation once the decoded sequence ends with '</PATH>' (after stripping specials)."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.end_tag = "</PATH>"

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return text.strip().endswith(self.end_tag)


# -------------------------
# Model loading
# -------------------------
def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto
    )

    # Apply Phi-3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    # Inference settings
    FastLanguageModel.for_inference(model)  # enables faster inference
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# -------------------------
# Interactive loop
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive TITAN tester.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model or adapter path")
    parser.add_argument("--names", default=DEFAULT_NAMES_FILE, help="NAMES.txt path")
    parser.add_argument("--graph", default=DEFAULT_GRAPH_FILE, help="GraphML path")
    parser.add_argument("--rels",  default=DEFAULT_REL_DESC_FILE, help="Relationship_Descriptions.txt path")
    args = parser.parse_args()

    # Load model/tokenizer
    print(f"[INFO] Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Build stopping criteria & streamer
    stopping = StoppingCriteriaList([StopOnEndPath(tokenizer)])
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Load names, graph, and relationship texts
    names = load_names(args.names)
    print("\n[INFO] Loading graph ...\n")
    GA_graph = GA.load_graph(args.graph)
    rel_map = load_relationship_description(args.rels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    while True:
        try:
            prompt = input("\nINSERT A CTI QUERY (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not prompt or prompt.lower() == "exit":
            print("[INFO] Bye.")
            break

        # Build chat messages
        messages = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": prompt},
        ]

        # Tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # must add for generation
            return_tensors="pt",
        ).to(device)

        # Generate until </PATH>
        print("\n[MODEL OUTPUT]\n")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                stopping_criteria=stopping,
                streamer=streamer,
            )

        # Decode
        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract entities & path
        try:
            entities = extract_entities_from_output(text)
            if entities:
                # snap each entity to the closest known name
                resolved = []
                for ent in entities:
                    close = GA.find_closest_entity(ent, names)
                    print(f"[ENTITY] '{ent}' -> '{close}'")
                    resolved.append(close)
                entities = resolved
            else:
                print("[WARN] No entities detected in output; proceeding without entity constraint.")

            final_path = extract_final_path(text)
            print("\n[FINAL PATH]\n", final_path, "\n")

            # Execute path
            cleaned = GA.extract_path_elements(final_path)
            print("[DEBUG] Cleaned path steps:", cleaned)

            if len(entities) <= 1:
                response, results = GA.follow_graph(GA_graph, entities, cleaned, rel_map)
                print("\n### FINAL RESULTS (single-entity mode) ###\n")
                # results is a set/list depending on your GA implementation
                if isinstance(results, (list, set, tuple)):
                    for i, elem in enumerate(list(results), 1):
                        print(f"{i} - {elem}")
                else:
                    print(results)

                print("\n### COMPLETE REASONING PLAN ###\n")
                print(response)

            else:
                response, results = GA.follow_graph_n_entities(GA_graph, entities, cleaned, rel_map)
                print("\n### FINAL RESULTS (multi-entity mode) ###\n")
                if isinstance(results, dict):
                    for ent, by_type in results.items():
                        print(f"\n[ENTITY] {ent}")
                        for t, items in by_type.items():
                            print(f"  - {t}: {sorted(list(items))}")
                else:
                    print(results)

        except Exception as e:
            print(f"[ERROR] {e}")
            print("[-] No result available.")


if __name__ == "__main__":
    # Make local imports (graph_algorithm.py) resolvable when running from repo root
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    main()
