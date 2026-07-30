#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph navigation helpers over a CTI/MITRE GraphML (NetworkX) graph.

Features:
- Load names for fuzzy matching.
- Get node "type" via edges whose label contains 'type'.
- Filter current nodes by keywords found in their 'description' neighbors
  (supports 'and', 'or', and single keyword).
- Follow a path of steps:
    * relation labels
    * 'is_<X>_type' (start from sources of that relation)
    * 'filter <kw1 and kw2>' / 'filter <kw1 or kw2>'
    * 'mitigated_by' (special-cased like generic relations)
    * 'exec_common <TYPE>' and 'exec_difference <TYPE>' over per-entity results
- Multi-entity traversal that keeps per-entity current node sets across steps.

Return format:
- Human-readable response string summarizing each step
- Aggregated results (per entity -> per node_type -> set(nodes))
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import reduce
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Loading / utilities
# ---------------------------------------------------------------------------

def load_names(names_file: str) -> Set[str]:
    """Load candidate names from a plain-text file (one name per line)."""
    with open(names_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f}


def load_graph(graph_file: str) -> nx.Graph:
    """Load a GraphML file into a NetworkX graph."""
    return nx.read_graphml(graph_file)


def similarity_score(a: str, b: str) -> float:
    """Compute a normalized similarity score between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def find_closest_entity(entity: str, names: Iterable[str], threshold: float = 0.5) -> Optional[str]:
    """Return the closest matching name above a similarity threshold, or None."""
    closest = None
    best = 0.0
    for name in names:
        s = similarity_score(entity, name)
        if s > best and s >= threshold:
            best, closest = s, name
    return closest


def extract_path_elements(path: str) -> List[str]:
    """
    Extract a list of steps from a <PATH>...</PATH> string, using <SEP> as separator if present.
    Returns [] if the format doesn't match.
    """
    m = re.search(r"<PATH>(.*?)</PATH>", path)
    if not m:
        return []
    content = m.group(1)
    return content.split("<SEP>") if "<SEP>" in content else [content]


# ---------------------------------------------------------------------------
# Graph-specific helpers
# ---------------------------------------------------------------------------

def _get_node_type(graph: nx.Graph, node: str) -> Optional[str]:
    """
    Return the 'type' of a node by checking neighbors connected via edges whose label contains 'type'.
    """
    try:
        for nb in graph.neighbors(node):
            label = graph[node][nb].get("label")
            if isinstance(label, str) and "type" in label:
                return nb
        return None
    except Exception as e:
        print(f"[WARN] _get_node_type failed for node={node}: {e}")
        return None


def _get_filtered_nodes(graph: nx.Graph, current_nodes: Iterable[str], keywords: str) -> Set[str]:
    """
    Filter nodes based on the presence of keywords in their 'description' neighbor.
    Supports:
      - "kw1 and kw2": all must be present
      - "kw1 or kw2" : any can be present
      - single keyword
    """
    current = list(current_nodes)
    if not current:
        return set()

    text = keywords.strip().lower()
    if " and " in text:
        conds = [c.strip() for c in text.split(" and ")]
        mode = "and"
    elif " or " in text:
        conds = [c.strip() for c in text.split(" or ")]
        mode = "or"
    else:
        conds = [text]
        mode = "single"

    result: Set[str] = set()
    for node in current:
        try:
            for nb in graph.neighbors(node):
                label = graph[node][nb].get("label")
                if label == "description":
                    desc = str(nb).lower()
                    if mode == "and" and all(c in desc for c in conds):
                        result.add(node)
                    elif mode == "or" and any(c in desc for c in conds):
                        result.add(node)
                    elif mode == "single" and conds[0] in desc:
                        result.add(node)
        except Exception:
            # Skip nodes that raise due to missing attributes
            continue
    return result


def find_type_sources(graph: nx.Graph, relation_label: str) -> List[str]:
    """
    Find all *source* nodes that have an outgoing edge with the given relation label.
    Useful for steps like 'is_<X>_type' when you want to seed the traversal.
    """
    sources: List[str] = []
    for src, _, edge in graph.edges(data=True):
        if edge.get("label") == relation_label:
            sources.append(src)
    return sources


# ---------------------------------------------------------------------------
# Step processing helpers
# ---------------------------------------------------------------------------

def _process_path_step(
    graph: nx.Graph,
    current_nodes: Set[str],
    step_label: str,
    step_index: int,
    entity_results: Dict[Optional[str], Set[str]],
) -> Tuple[str, Set[str]]:
    """
    Process a single step for one entity:
    - If step_label contains '_type', it seeds the next_nodes by finding all sources for that relation.
    - If step_label starts with 'filter ', it filters current_nodes by description keywords.
    - Otherwise, it follows edges whose label == step_label.
    Accumulates results grouped by inferred node type in entity_results.
    """
    response_lines: List[str] = []
    next_nodes: Set[str] = set()

    if "_type" in step_label:
        # Seed from all sources that have such relation label
        next_nodes = set(find_type_sources(graph, step_label))
        response_lines.append(f"[!] Step {step_index}: Seeding from relation '{step_label}'. Found {len(next_nodes)} sources.")
        for n in sorted(next_nodes):
            response_lines.append(f"   - {n}")

    elif step_label.startswith("filter "):
        keywords = step_label.split("filter ", 1)[1].strip()
        filtered = _get_filtered_nodes(graph, current_nodes, keywords)
        next_nodes = filtered
        response_lines.append(f"[!] Step {step_index}: Filtering nodes by keywords '{keywords}'. Kept {len(filtered)} nodes.")
        for n in sorted(filtered):
            response_lines.append(f"   - {n}")

    else:
        # Generic relation traversal
        response_lines.append(f"[!] Step {step_index}: Following relation '{step_label}'.")
        for node in current_nodes:
            try:
                for nb in graph.neighbors(node):
                    if graph[node][nb].get("label") == step_label:
                        next_nodes.add(nb)
                        nb_type = _get_node_type(graph, nb)
                        entity_results.setdefault(nb_type, set()).add(nb)
                        response_lines.append(f"   - {nb} ({nb_type})")
            except Exception:
                continue

    return "\n".join(response_lines) + ("\n" if response_lines else ""), next_nodes


# ---------------------------------------------------------------------------
# Public traversal APIs
# ---------------------------------------------------------------------------

def follow_graph_n_entities(
    graph: nx.Graph,
    entities: Sequence[str],
    path_steps: Sequence[str],
    source_target_rel: Dict[str, List[Tuple[str, str]]],
    debug: bool = False,
) -> Tuple[str, Dict[str, Dict[Optional[str], Set[str]]]]:
    """
    Follow the graph for multiple starting entities, maintaining a separate state per entity.

    Args:
        graph: NetworkX graph.
        entities: List of starting entity names.
        path_steps: List of step labels to follow.
        source_target_rel: Optional mapping {source_node: [(target_node, relation), ...]}.
                           If provided, it is only used to enrich output messages (not required).
        debug: If True, include extra headers.

    Returns:
        response: Human-readable trace of the traversal.
        all_results: {entity -> {node_type -> set(nodes)}} accumulated along the path.
    """
    response_parts: List[str] = []
    # Per-entity current nodes and per-entity results
    current_by_entity: Dict[str, Set[str]] = {e: {e} for e in entities}
    all_results: Dict[str, Dict[Optional[str], Set[str]]] = {e: {} for e in entities}

    if debug:
        response_parts.append(f"[DEBUG] Starting Entities: {entities}")
        response_parts.append(f"[DEBUG] PATH: {path_steps}")

    for i, step in enumerate(path_steps, start=1):
        if step.startswith("select"):
            # Selection step: list current entities (no state change)
            selected = [e for e in entities if e in all_results]
            response_parts.append(f"[!] Step {i}: Selecting entities. Results:")
            response_parts.extend([f"   - {e}" for e in selected])
            continue

        if step.startswith("exec_"):
            # Aggregate operation over entity results: exec_common TYPE / exec_difference TYPE
            op = step.split("exec_", 1)[1].split(" ")[0].strip()
            try:
                op_type = step.split(" ", 1)[1].strip()
            except IndexError:
                op_type = ""
            type_sets = [all_results[e].get(op_type, set()) for e in entities]

            if op == "common":
                combined = reduce(lambda a, b: a & b, type_sets) if type_sets else set()
            elif op == "difference":
                combined = reduce(lambda a, b: a ^ b, type_sets) if type_sets else set()
            else:
                combined = set()
                response_parts.append(f"[ERROR] Unknown operation '{op}'.")

            response_parts.append(f"[!] Step {i}: {op.capitalize()} results for type '{op_type}'.")
            response_parts.extend([f"   - {n}" for n in sorted(combined)])
            continue

        # Normal step: process each entity independently and update its current set
        for e in entities:
            step_response, next_nodes = _process_path_step(
                graph=graph,
                current_nodes=current_by_entity[e],
                step_label=step,
                step_index=i,
                entity_results=all_results[e],
            )
            response_parts.append(step_response)
            # Move forward: next becomes current for next step
            current_by_entity[e] = next_nodes if next_nodes else set()

    return "\n".join(p for p in response_parts if p), all_results


def follow_graph(
    graph: nx.Graph,
    entities: Sequence[str],
    path_steps: Sequence[str],
    source_target_rel: Dict[str, List[Tuple[str, str]]],
    debug: bool = False,
) -> Tuple[str, Set[str]]:
    """
    Single-entity (or blind-start) traversal:
    - If one entity is provided, start from it.
    - If multiple/no entities are provided, try to seed the initial set from the first step.

    Returns:
        response: Human-readable trace.
        current_nodes: Set of nodes at the end of the path.
    """
    # Seed current nodes
    if len(entities) == 1:
        current_nodes: Set[str] = {entities[0]}
    else:
        # Blind-start seeding from the first step
        current_nodes = set()
        if path_steps:
            first = path_steps[0]
            for node in graph.nodes:
                for nb in graph.neighbors(node):
                    if graph[node][nb].get("label") == first:
                        if "_type" not in first:
                            current_nodes.add(nb)
                        else:
                            current_nodes.add(node)

    response_parts: List[str] = []
    for i, step in enumerate(path_steps, start=1):
        if step.startswith("filter "):
            keywords = step.split("filter ", 1)[1].strip()
            filtered = _get_filtered_nodes(graph, current_nodes, keywords)
            current_nodes = filtered
            node_type = _get_node_type(graph, next(iter(filtered), "")) if filtered else None
            response_parts.append(f"[!] Step {i}: Filtering {node_type} nodes by '{keywords}'.")
            response_parts.extend([f"   - {n}" for n in sorted(filtered)])
            continue

        # Relation following
        response_parts.append(f"[!] Step {i}: Following relation '{step}'. Results:")
        next_nodes: Set[str] = set()
        for node in current_nodes:
            for nb in graph.neighbors(node):
                label = graph[node][nb].get("label")
                if "_type" not in step and label == step:
                    next_nodes.add(nb)
                    # Optional enrichment using source_target_rel
                    if node in source_target_rel:
                        rel_list = source_target_rel[node]
                        rel_str = next((rel for (tgt, rel) in rel_list if tgt == nb), None)
                        if rel_str:
                            response_parts.append(f"   - {nb}: {rel_str}")
                        else:
                            response_parts.append(f"   - {nb}")
                    else:
                        response_parts.append(f"   - {nb}")

                if "_type" in step and label == step:
                    # When step is a type relation, keep the source itself
                    next_nodes.add(node)
                    response_parts.append(f"   - {node}")

        current_nodes = next_nodes

    return "\n".join(response_parts), current_nodes
