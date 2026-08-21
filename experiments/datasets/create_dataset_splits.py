#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a Q&A CSV into train/val/test sets, *per section*, using a JSON
"questions-per-section" mapping.

- Reads:
    * CSV with at least a "Question" column
    * JSON mapping: { "<section_name>": [ { "question": "...", ... }, ... ], ... }

- Adds a "Section" column to the CSV by mapping question -> section.
- Performs per-section split with ratios (train/val/test).
- Saves three CSV files in the specified output directory.
- Prints section distributions for each split.

Robustness:
- Validates ratios sum to ~1.0.
- Handles questions not present in the JSON mapping (either drop or assign "Unknown").
- Fallbacks for tiny sections (e.g., 1–2 rows) to avoid train_test_split errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Ensure ratios are within (0,1] and sum approximately to 1."""
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0 or r > 1 for r in ratios):
        raise ValueError("Ratios must be within [0, 1].")
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0. Got {sum(ratios):.6f}.")


def _build_section_mapping(section_json: Dict) -> Dict[str, str]:
    """
    Build a dictionary: question_text -> section_name
    from a JSON structure like: { "secA": [ {"question": "..."} , ...], ... }.
    """
    mapping: Dict[str, str] = {}
    for section, items in section_json.items():
        for item in items:
            q = item.get("question")
            if q:
                mapping[q] = section
    return mapping


def _safe_split_section(
    df_section: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single-section DataFrame into train/val/test with fallbacks for tiny sections.

    Rules of thumb to avoid errors:
    - 0 rows: empty splits
    - 1 row: all to train
    - 2 rows: 1 train, 1 test (val empty)
    - >=3 rows: use sklearn splits with given ratios
    """
    n = len(df_section)
    if n == 0:
        return df_section.iloc[0:0], df_section.iloc[0:0], df_section.iloc[0:0]
    if n == 1:
        return df_section, df_section.iloc[0:0], df_section.iloc[0:0]
    if n == 2:
        # 50/50 split, no val
        train, test = train_test_split(df_section, test_size=0.5, random_state=random_state, shuffle=True)
        return train, df_section.iloc[0:0], test

    # General case (>=3)
    # First split off 'temp' from train portion
    temp_ratio = 1.0 - train_ratio
    train_df, temp_df = train_test_split(df_section, test_size=temp_ratio, random_state=random_state, shuffle=True)

    # Split temp into val/test with the given proportion
    if len(temp_df) == 0:
        return train_df, df_section.iloc[0:0], df_section.iloc[0:0]

    # Allocate test as fraction of (val+test)
    if (val_ratio + test_ratio) == 0:
        # No temp desired; all already in train
        return train_df, df_section.iloc[0:0], df_section.iloc[0:0]

    test_frac = test_ratio / (val_ratio + test_ratio)
    if len(temp_df) == 1:
        # Put single row into validation by default
        return train_df, temp_df, df_section.iloc[0:0]

    val_df, test_df = train_test_split(temp_df, test_size=test_frac, random_state=random_state, shuffle=True)
    return train_df, val_df, test_df


def create_datasets(
    csv_file: str,
    json_file: str,
    output_dir: str = "SMARTER_COMPLETE_DATASET",
    train_ratio: float = 0.8,
    val_ratio: float = 0.05,
    test_ratio: float = 0.15,
    random_state: int = 42,
    drop_unmapped: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create per-section train/val/test splits and save them to CSV.

    Args:
        csv_file: Path to input CSV containing at least a 'Question' column.
        json_file: Path to JSON mapping sections to question items.
        output_dir: Directory where train/val/test CSVs will be saved.
        train_ratio/val_ratio/test_ratio: Split ratios, must sum to 1.0.
        random_state: RNG seed for deterministic splits.
        drop_unmapped: If True, drop rows whose Question is not in JSON mapping.
                       If False, assign Section='Unknown'.

    Returns:
        (train_df, val_df, test_df)
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    # Load inputs
    df = pd.read_csv(csv_file, encoding="utf-8")
    with open(json_file, "r", encoding="utf-8") as f:
        section_json = json.load(f)

    if "Question" not in df.columns:
        raise ValueError("Input CSV must contain a 'Question' column.")

    # Map question -> section
    section_mapping = _build_section_mapping(section_json)
    df["Section"] = df["Question"].map(section_mapping)

    # Handle unmapped questions
    num_unmapped = df["Section"].isna().sum()
    if num_unmapped > 0:
        if drop_unmapped:
            df = df.dropna(subset=["Section"]).reset_index(drop=True)
            print(f"[INFO] Dropped {num_unmapped} rows with no section mapping.")
        else:
            df["Section"] = df["Section"].fillna("Unknown")
            print(f"[INFO] Assigned 'Unknown' to {num_unmapped} rows with no section mapping.")

    # Split per section
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for section in sorted(df["Section"].dropna().unique()):
        section_df = df[df["Section"] == section]
        train_df, val_df, test_df = _safe_split_section(
            section_df, train_ratio, val_ratio, test_ratio, random_state
        )
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    # Combine splits
    train_set = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    val_set = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0]
    test_set = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]

    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train_dataset.csv")
    val_out = os.path.join(output_dir, "val_dataset.csv")
    test_out = os.path.join(output_dir, "test_dataset.csv")

    # Save
    train_set.to_csv(train_out, index=False, encoding="utf-8")
    val_set.to_csv(val_out, index=False, encoding="utf-8")
    test_set.to_csv(test_out, index=False, encoding="utf-8")

    # Print section distributions
    print("\nDistribution of sections:")
    for name, ds in [("Train", train_set), ("Validation", val_set), ("Test", test_set)]:
        if ds.empty:
            print(f"\n{name} Set: (empty)")
            continue
        dist = (ds["Section"].value_counts(normalize=True) * 100).sort_index()
        print(f"\n{name} Set:")
        print(dist.to_string(float_format=lambda x: f"{x:.2f}%"))

    # Summary
    print("\nDataset creation complete:")
    print(f"- Training set:   {len(train_set)} samples -> {train_out}")
    print(f"- Validation set: {len(val_set)} samples -> {val_out}")
    print(f"- Test set:       {len(test_set)} samples -> {test_out}")

    return train_set, val_set, test_set


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-section train/val/test splits from CSV + JSON mapping.")
    p.add_argument("--csv", required=False, default="SMARTER/SMARTER_DATASET.csv", help="Input CSV file path")
    p.add_argument("--json", required=False, default="SMARTER/SMARTER_QUESTION_PER_SECTION.json", help="Input JSON mapping path")
    p.add_argument("--out", required=False, default="SMARTER_COMPLETE_DATASET", help="Output directory")
    p.add_argument("--train", type=float, default=0.80, help="Train ratio")
    p.add_argument("--val", type=float, default=0.05, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.15, help="Test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random state")
    p.add_argument("--keep-unmapped", action="store_true", help="Keep unmapped questions as Section='Unknown'")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_datasets(
        csv_file=args.csv,
        json_file=args.json,
        output_dir=args.out,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_state=args.seed,
        drop_unmapped=not args.keep_unmapped,
    )
