#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modify targets in YAML templates or JSON datasets using target_variations.csv.

- CSV must have columns: "Original Question", "Variations"
- Input file can be:
  * YAML template with structure:
      templates:
        <section>:
          - question: "..."
            target: ["..."]   # list or string; this script will normalize to string
  * JSON dataset: a list of objects with at least "question". This script will
    add/update a "target" field per item.

Usage examples:
  python modify_target.py --csv target_variations.csv --in utils/useful_cot.yaml --out utils/useful_cot.improved.yaml
  python modify_target.py --csv target_variations.csv --in datasets/CoT/NAVIGATION_DATASET.json --out datasets/CoT/NAVIGATION_DATASET.improved.json

Options:
  --pick {first,longest}   How to pick a target when 'Variations' has multiple items separated by '; '
  --dry-run                Do not write output; just report what would change
  --no-backup              Disable automatic .bak backup of the --out file if it already exists
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Dict, Tuple

try:
    import yaml
except ImportError:  # lazy hint if PyYAML is missing
    yaml = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modify targets using target_variations.csv")
    p.add_argument("--csv", required=True, help="Path to target_variations.csv")
    p.add_argument("--in", dest="inp", required=True, help="Input file (YAML or JSON)")
    p.add_argument("--out", dest="out", required=True, help="Output file (YAML or JSON)")
    p.add_argument(
        "--pick",
        choices=["first", "longest"],
        default="first",
        help="How to pick from multiple variations in CSV (default: first)",
    )
    p.add_argument("--dry-run", action="store_true", help="Only report changes, do not write output")
    p.add_argument("--no-backup", action="store_true", help="Disable .bak backup for existing output")
    return p.parse_args()


def load_variations_map(csv_path: str, pick: str = "first") -> Dict[str, str]:
    """
    Load CSV and build a mapping: question -> best_variation (string).
    'Variations' can be a semicolon-separated list; we pick either the first or the longest.
    """
    mapping: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            q = (row.get("Original Question") or "").strip()
            v = (row.get("Variations") or "").strip()
            if not q or not v:
                continue

            # Split by '; ' when multiple candidates are present
            candidates = [c.strip() for c in v.split(";") if c.strip()]
            if not candidates:
                continue

            if pick == "first":
                best = candidates[0]
            else:  # longest
                best = max(candidates, key=len)

            mapping[q] = best
    return mapping


def is_yaml(path: str) -> bool:
    return path.lower().endswith((".yaml", ".yml"))


def is_json(path: str) -> bool:
    return path.lower().endswith(".json")


def backup_if_needed(out_path: str, no_backup: bool) -> None:
    if no_backup or not os.path.exists(out_path):
        return
    bak = out_path + ".bak"
    shutil.copy2(out_path, bak)
    print(f"[INFO] Created backup: {bak}")


def process_yaml(inp: str, out: str, q2target: Dict[str, str], dry_run: bool, no_backup: bool) -> Tuple[int, int]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please: pip install pyyaml")

    data = yaml.safe_load(open(inp, "r", encoding="utf-8"))
    if not isinstance(data, dict) or "templates" not in data:
        raise ValueError("YAML structure not recognized: expected top-level 'templates' dict.")

    total, updated = 0, 0
    for section, items in (data.get("templates") or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            q = item.get("question")
            if not q:
                continue
            total += 1
            new_target = q2target.get(q)
            if new_target:
                # Normalize to string (your YAML had target as list; we store single improved string)
                prev = item.get("target")
                if prev != new_target:
                    updated += 1
                    if not dry_run:
                        item["target"] = new_target

    print(f"[YAML] Questions scanned: {total} | Updated: {updated}")
    if not dry_run:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        backup_if_needed(out, no_backup)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        print(f"[YAML] Wrote: {out}")
    return total, updated


def process_json(inp: str, out: str, q2target: Dict[str, str], dry_run: bool, no_backup: bool) -> Tuple[int, int]:
    data = json.load(open(inp, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON structure not recognized: expected a list of items.")

    total, updated = 0, 0
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if not q:
            continue
        total += 1
        new_target = q2target.get(q)
        if new_target:
            prev = item.get("target")
            if prev != new_target:
                updated += 1
                if not dry_run:
                    item["target"] = new_target

    print(f"[JSON] Items scanned: {total} | Updated: {updated}")
    if not dry_run:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        backup_if_needed(out, no_backup)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[JSON] Wrote: {out}")
    return total, updated


def main() -> None:
    args = parse_args()
    if not (is_yaml(args.inp) or is_json(args.inp)):
        print("[ERROR] --in must be a .yaml/.yml or .json file", file=sys.stderr)
        sys.exit(2)
    if not (is_yaml(args.out) or is_json(args.out)):
        print("[ERROR] --out must be a .yaml/.yml or .json file", file=sys.stderr)
        sys.exit(2)

    q2target = load_variations_map(args.csv, pick=args.pick)
    if not q2target:
        print("[WARN] No valid rows found in CSV. Nothing to update.")

    print(f"[INFO] Loaded {len(q2target)} question→target mappings from: {args.csv}")

    if is_yaml(args.inp):
        process_yaml(args.inp, args.out, q2target, args.dry_run, args.no_backup)
    elif is_json(args.inp):
        process_json(args.inp, args.out, q2target, args.dry_run, args.no_backup)


if __name__ == "__main__":
    main()
