#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a directed NetworkX graph from MITRE ATT&CK STIX JSON files.

Pipeline:
1) Walk a base directory and read all .json STIX bundles.
2) Collect object names (id -> name) and queue relationships.
3) Add attribute edges for non-relationship objects.
4) Post-process all queued relationships, add forward/backward edges.
5) Save the resulting graph to GraphML.

Notes:
- Relationship 'uses' is expanded to 'uses_<target-prefix>'.
- Backward edges are added for 'mitigates'/'detects' and for 'uses' (as 'used_by_<source-prefix>').
- Attribute 'type' becomes 'is_<value>_type' (with '-' replaced by '_').

CLI:
    python build_stix_graph.py --base ../attack-stix-data --out stix_graph_correct.graphml --log-file mitre.txt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
from tqdm import tqdm


# ----------------------------- Data structures ----------------------------- #

@dataclass(frozen=True)
class QueuedRelationship:
    """Minimal representation of a STIX relationship we need to add to the graph."""
    source_ref: str
    target_ref: str
    relationship_type: str


# ------------------------------- Utilities -------------------------------- #

def clean_hyphens_to_underscores(s: str) -> str:
    """Replace '-' with '_' to keep labels GraphML-friendly."""
    return s.replace("-", "_") if "-" in s else s


def to_string_value(value) -> str:
    """
    Normalize STIX attribute values into a single string:
    - list -> first element or 'Not Available' if empty
    - dict -> JSON string
    - str/other -> str(value)
    """
    if isinstance(value, list):
        if len(value) == 0:
            return "Not Available"
        # original behavior: take first item only
        return to_string_value(value[0])
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


# ----------------------------- Core processing ----------------------------- #

class StixGraphBuilder:
    """
    Builder that accumulates nodes/edges from STIX JSON files into a NetworkX DiGraph.
    """

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.g = nx.DiGraph()
        self.id_to_name: Dict[str, str] = {}
        self.relationships: List[QueuedRelationship] = []
        self.log_file = log_file

        # Keys we do not emit as attribute edges
        self._skip_keys = {
            "x_mitre_modified_by_ref", "modified_by_ref", "x_mitre_contents", "id", "source_ref",
            "target_ref", "object_ref", "relationship_type", "x_mitre_version",
            "x_mitre_attack_spec_version", "created_by_ref", "object_marking_refs"
        }

    # ------------------------- Logging helper ------------------------- #

    def _log(self, line: str) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------ STIX file parsing ----------------------- #

    def process_stix_file(self, file_path: str) -> None:
        """Read a STIX JSON bundle and enqueue objects/relationships."""
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {file_path}: {e}")
                return

        objects = data.get("objects", [])
        for obj in objects:
            obj_id = obj.get("id")
            obj_type = obj.get("type")
            if not obj_id or not obj_type:
                continue

            # collect id->name (if present)
            if "name" in obj:
                self.id_to_name[obj_id] = obj["name"]

            # queue relationship or add attributes
            if obj_type == "relationship":
                self._queue_relationship(obj)
            elif "name" in obj:
                self._add_attribute_edges(name=obj["name"], obj=obj)

    def _queue_relationship(self, rel_obj: dict) -> None:
        """Queue a minimal representation of the relationship to post-process later."""
        src = rel_obj.get("source_ref")
        dst = rel_obj.get("target_ref")
        rel = rel_obj.get("relationship_type")

        if not (src and dst and rel):
            return

        # Keep only the fields we need later
        self.relationships.append(QueuedRelationship(source_ref=src, target_ref=dst, relationship_type=rel))

    # -------------------------- Attribute edges ----------------------- #

    def _add_attribute_edges(self, name: str, obj: dict) -> None:
        """
        For a non-relationship object with a 'name', add edges: name -[key]-> value.
        Special case for 'type': 'is_<value>_type' and '_' normalized.
        """
        for key, raw_value in obj.items():
            if key in self._skip_keys:
                continue

            try:
                value = to_string_value(raw_value)
            except Exception as e:
                print(f"[WARN] Failed normalizing value for key={key} on {name}: {e}")
                continue

            edge_label = key
            if key == "type":
                # turn 'attack-pattern' into 'is_attack_pattern_type'
                value = clean_hyphens_to_underscores(value)
                edge_label = f"is_{value}_type"

            self.g.add_edge(name, value, label=edge_label)
            self._log(f"{name} - {edge_label} -> {value}")

    # -------------------------- Relationship edges -------------------- #

    def process_all_relationships(self) -> None:
        """Create graph edges for all queued relationships (with backward edges when needed)."""
        for rel in tqdm(self.relationships, desc="Building INTER Relationships ..."):
            self._add_relationship_edges(rel)

    def _add_relationship_edges(self, r: QueuedRelationship) -> None:
        """
        Add forward edge and useful backward edges for a relationship.
        - 'uses' becomes 'uses_<target-prefix>' and backward 'used_by_<source-prefix>'
        - 'mitigates' backward 'mitigated_by'
        - 'detects' backward 'detected_by'
        - 'attributed-*' backward 'is_responsible'
        """
        src_name = self.id_to_name.get(r.source_ref)
        dst_name = self.id_to_name.get(r.target_ref)
        if not (src_name and dst_name):
            # If names are missing, skip to keep graph readable (or fallback to IDs)
            # src_name = src_name or r.source_ref
            # dst_name = dst_name or r.target_ref
            return

        rel_type = r.relationship_type
        # expand 'uses' with the target object prefix (e.g., attack-pattern, malware, tool)
        if rel_type == "uses":
            target_prefix = r.target_ref.split("--", 1)[0]  # e.g., attack-pattern
            rel_type = f"{rel_type}_{target_prefix}"

        rel_type = clean_hyphens_to_underscores(rel_type)

        # forward edge
        self.g.add_edge(src_name, dst_name, label=rel_type)
        self._log(f"{src_name} - {rel_type} -> {dst_name}")

        # backward edges
        if rel_type == "mitigates":
            inv = "mitigated_by"
            self.g.add_edge(dst_name, src_name, label=inv)
            self._log(f"{dst_name} - {inv} -> {src_name}")

        if rel_type == "detects":
            inv = "detected_by"
            self.g.add_edge(dst_name, src_name, label=inv)
            self._log(f"{dst_name} - {inv} -> {src_name}")

        if rel_type.startswith("attributed"):
            inv = "is_responsible"
            self.g.add_edge(dst_name, src_name, label=inv)
            self._log(f"{dst_name} - {inv} -> {src_name}")

        if rel_type.startswith("uses_"):
            source_prefix = r.source_ref.split("--", 1)[0]  # e.g., intrusion-set, malware, tool, campaign
            inv = f"used_by_{source_prefix}"
            inv = clean_hyphens_to_underscores(inv)
            self.g.add_edge(dst_name, src_name, label=inv)
            self._log(f"{dst_name} - {inv} -> {src_name}")


# ------------------------------ Orchestration ------------------------------ #

def walk_and_process(builder: StixGraphBuilder, base_path: str) -> None:
    """Recursively scan base_path for .json STIX files and process them."""
    for root, _, files in os.walk(base_path):
        print(f"Analyzing {root} ...")
        for file in tqdm(files, desc="Reading JSON files", leave=False):
            if not file.endswith(".json"):
                continue
            builder.process_stix_file(os.path.join(root, file))


def save_graphml(g: nx.DiGraph, out_path: str) -> None:
    """Save the built graph to GraphML."""
    nx.write_graphml(g, out_path)
    print(f"Graph successfully saved to {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a NetworkX graph from MITRE ATT&CK STIX JSON data.")
    ap.add_argument("--base", default="../attack-stix-data", help="Base directory containing STIX JSON files")
    ap.add_argument("--out", default="stix_graph_correct.graphml", help="Output GraphML file")
    ap.add_argument("--log-file", default="mitre.txt", help="Optional edge log file (set empty to disable)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    log_file = args.log_file if args.log_file else None

    # Clean previous log if present
    if log_file and os.path.exists(log_file):
        os.remove(log_file)

    builder = StixGraphBuilder(log_file=log_file)
    walk_and_process(builder, args.base)
    builder.process_all_relationships()
    save_graphml(builder.g, args.out)


if __name__ == "__main__":
    main()
