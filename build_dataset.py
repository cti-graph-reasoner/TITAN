#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Question + Chain-of-Thought (CoT) pairs from:
- YAML templates
- A CTI/MITRE GraphML knowledge graph
- CSV variations for questions/targets (optional)

Outputs:
- Flat JSON list of {"question": ..., "response": ...}
- Per-section JSON grouping
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants / MITRE field descriptions
# ---------------------------------------------------------------------------

DESCRIPTION = False

MITRE_FIELDS: Dict[str, str] = {
    "x_mitre_detection": "Strategies for identifying if a technique has been used by an adversary.",
    "x_mitre_platforms": "List of platforms that apply to the technique.",
    "x_mitre_data_sources": "Sources of information that may be used to identify the action or result of the action being performed.",
    "x_mitre_is_subtechnique": "If true, this attack-pattern is a sub-technique.",
    "x_mitre_system_requirements": "Additional information on requirements the adversary needs to meet or about the state of the system (software, patch level, etc.) that may be required for the technique to work.",
    "x_mitre_tactic_type": "Post-Adversary Device Access, Pre-Adversary Device Access, or Without Adversary Device Access.",
    "x_mitre_permissions_required": "The lowest level of permissions the adversary is required to be operating within to perform the technique on a system.",
    "x_mitre_effective_permissions": "The level of permissions the adversary will attain by performing the technique.",
    "x_mitre_defense_bypassed": "List of defensive tools, methodologies, or processes the technique can bypass.",
    "x_mitre_remote_support": "If true, the technique can be used to execute something on a remote system.",
    "x_mitre_impact_type": "Denotes if the technique can be used for integrity or availability attacks.",
    "x_mitre_aliases": "List of aliases for the given software.",
    "x_mitre_first_seen_citation": "One to many citations for when the Campaign was first reported.",
    "x_mitre_last_seen_citation": "One to many citations for when the Campaign was last reported.",
    "x_mitre_sectors": "List of industry sector(s) an asset may be commonly observed in.",
    "x_mitre_related_assets": "Sector-specific device names or aliases commonly associated with the asset.",
    "attack_pattern": "Attack Patterns describe ways adversaries attempt to compromise targets.",
    "course_of_action": "Actions to prevent or respond to an attack in progress.",
    "malware": "Malicious code/software.",
    "tool": "Legitimate software used by threat actors.",
    "campaign": "Grouping of adversarial behaviors over time against specific targets.",
    "intrusion_set": "Grouped adversarial behaviors/resources believed orchestrated by a single org.",
}

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------

_name_cache: Dict[str, List[str]] = {}
_description_cache: Dict[str, Optional[str]] = {}

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_templates(yaml_file: str) -> dict:
    """Load YAML templates."""
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_graph(graph_file: str) -> nx.Graph:
    """Load GraphML knowledge graph."""
    return nx.read_graphml(graph_file)


def load_multivariated_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV with question variations (optional)."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f"CSV loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except pd.errors.ParserError as e:
        print(f"Error loading CSV file due to parsing issue: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
    except Exception as e:
        print(f"Unexpected error loading CSV file: {e}")
    return None


def load_multivariated_targets(target_file: str) -> Dict[str, List[str]]:
    """Load CSV with target variations: 'Original Question' -> ['var1', 'var2', ...]."""
    targets: Dict[str, List[str]] = {}
    with open(target_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row["Original Question"]
            variations = [v.strip() for v in row["Variations"].split("; ") if v.strip()]
            targets[original] = variations
    return targets


def get_target_for_question(original_question: str, targets_map: Dict[str, List[str]]) -> Optional[List[str]]:
    """Return list of targets associated with a given original question."""
    return targets_map.get(original_question)


def save_generated_questions(output_file: str, items: List[dict]) -> None:
    """Write the flat JSON with all (question, response) pairs."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4)
    print(f"\nFILE SAVED IN {output_file}")


# ---------------------------------------------------------------------------
# Graph helpers (types, names, descriptions, relation mapping)
# ---------------------------------------------------------------------------

def get_names_by_type(graph: nx.Graph, node_type: str) -> List[str]:
    """
    Collect node names assumed to have 'node_type' via an edge convention.
    Fallback tries edges whose label contains 'type'.
    """
    names: List[str] = []

    # Direct convention: edge (source -> target) where target == node_type
    for source, target in graph.edges():
        if str(target).lower() == node_type.lower():
            if source not in names:
                names.append(source)

    # Fallback: edge with label containing 'type'
    if not names:
        for src, dst, data in graph.edges(data=True):
            label = str(data.get("label", "")).lower()
            if "type" in label and str(dst).lower() == node_type.lower():
                if src not in names:
                    names.append(src)

    return names


def get_names_by_type_cached(graph: nx.Graph, node_type: str) -> List[str]:
    """Cached version of get_names_by_type."""
    if node_type in _name_cache:
        return _name_cache[node_type]
    names = get_names_by_type(graph, node_type)
    _name_cache[node_type] = names
    return names


def _get_node_type(graph: nx.Graph, node: str) -> Optional[str]:
    """Return the 'type' of a node by following a neighbor via an edge whose label contains 'type'."""
    try:
        for neighbor in graph.neighbors(node):
            label = graph[node][neighbor].get("label")
            if isinstance(label, str) and "type" in label:
                return neighbor
        return None
    except Exception as e:
        print(f"_get_node_type error on node={node}: {e}")
        return None


def _get_description(graph: nx.Graph, start_node: str, keyword: Optional[str] = None) -> Optional[str]:
    """
    Extract the first sentence of a description linked via edge label == 'description'.
    Cleans parentheses and square brackets.
    """
    try:
        for neighbor in graph.neighbors(start_node):
            label = graph[start_node][neighbor].get("label")
            if label == "description":
                full = str(neighbor)
                cleaned = re.sub(r"\(.*?\)", "", full)
                cleaned = re.sub(r"\[|\]", "", cleaned)
                m = re.search(r"[^.]*\.(?=\s|$)", cleaned)
                return (m.group(0).strip() if m else cleaned) or None
    except Exception:
        pass
    return None


def _get_description_cached(graph: nx.Graph, name: str) -> Optional[str]:
    """Cached description lookup."""
    if name not in _description_cache:
        _description_cache[name] = _get_description(graph, name)
    return _description_cache[name]


def create_relationship_type_mapping(
    graph: nx.Graph,
    output_file: str = "relationship_type_mapping.json"
) -> Dict[str, Dict[str, int]]:
    """
    Build map: relation_label -> {target_type: count}.
    Saves the map to JSON as well.
    """
    rel_map: Dict[str, Dict[str, int]] = {}

    for _, target, data in graph.edges(data=True):
        relationship = data.get("label")
        if not relationship:
            continue

        target_type = _get_node_type(graph, target) or target
        if not target_type:
            continue

        rel_map.setdefault(relationship, {})
        rel_map[relationship][target_type] = rel_map[relationship].get(target_type, 0) + 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rel_map, f, indent=4)

    return rel_map


def get_source_of_relationship(relationship: str, graph: nx.Graph) -> Optional[str]:
    """
    Return a source node for the first edge whose data['label'] == relationship (fallback: data['type']).
    """
    for u, v, data in graph.edges(data=True):
        if data.get("label") == relationship or data.get("type") == relationship:
            return u
    return None


# ---------------------------------------------------------------------------
# CoT prompt builders
# ---------------------------------------------------------------------------

def _initial_phrase_single(graph: nx.Graph, name: Optional[str], entity_type: Optional[str], target: str) -> str:
    desc = _get_description_cached(graph, name) if name else None
    head = "**Task Analysis**\n\n"
    if name:
        head += f"Let's reason step by step, starting from the entity '{name}' (type: {entity_type}). Our goal is: {target}.\n"
        if desc:
            head += f"\n**Entity Description**\n\nDescription: {desc}\n"
    else:
        head += f"Let's reason step by step.\nOur goal is to reach the {target}.\n"
    return head


def _node_type_description(node_type: str) -> Optional[str]:
    return MITRE_FIELDS.get(node_type)


def _starting_point_section(node_type: Optional[str]) -> str:
    descr = _node_type_description(node_type or "")
    start = "\n**Starting Point (START)**\n\n"
    if descr:
        return start + f"To begin, let's focus on nodes of type '{node_type}' to establish our initial scope.\nDescription of '{node_type}': {descr}\n"
    return start + f"To begin, let's focus on nodes of type '{node_type}' to establish our initial scope.\n"


def _path_prompt(
    step: int,
    node_type: Optional[str],
    relationship: str,
    graph: nx.Graph,
    current_node: str,
    last: bool = False,
) -> str:
    """
    Render a single step in the step-by-step reasoning, given the relation and the expected node_type.
    Supports modifiers like 'page_rank N', 'select X', 'filter COND', and 'description'.
    """
    if "page_rank" in relationship:
        number = relationship.split("page_rank ", 1)[1]
        node_type = f"the top {number} {node_type}"

    if node_type and "description" not in relationship:
        node_descr = None  # could be _node_type_description(node_type)
        if last:
            if node_descr:
                return (
                    f"Step {step}: Follow '{relationship}' relationship to reach {node_type}, marking the conclusion of our path.\n"
                    f"Description of {node_type}: {node_descr}\n"
                )
            return f"Step {step}: Follow '{relationship}' relationship to reach {node_type}, marking the conclusion of our path."
        else:
            if "select" in relationship:
                selection = relationship.split("select ", 1)[1]
                return f"Step {step}: Follow '{relationship}' relationship, focusing on {node_type} nodes and selecting '{selection}'."
            if "filter" in relationship:
                args = relationship.split("filter ", 1)[1]
                return f"Step {step}: Follow '{relationship}' relationship, leading us to {node_type} nodes having {args} in their description."
            return f"Step {step}: Follow '{relationship}' relationship, leading us to nodes representing {node_type}."
    elif "description" in relationship:
        return f"Step {step}: Follow '{relationship}' relationship to retrieve the description of the previous node."
    else:
        return (
            f"Step {step}: By following the '{relationship}' relationship, we proceed to the next relevant nodes, "
            "continuing our analysis within the cybersecurity context."
        )


def _final_prompt(final_path: str, graph: nx.Graph, target_node: str, start_node: Optional[str]) -> str:
    desc = _get_description_cached(graph, target_node)
    title = "\n\n**Path Completion**\n\n"
    if start_node and "None" not in str(start_node):
        body = (
            f"By following these steps, represented as {final_path}, "
            f"starting from {start_node} nodes, we have successfully achieved our goal: {target_node}."
        )
    else:
        body = f"By following these steps, represented as {final_path}, we have successfully achieved our goal: {target_node}."

    body += f"\n\n**Final Path**\n\nThe completed path is:\n\n{final_path}"
    if desc:
        body += f"\n\n Description of the final target: {desc}. This path provides insight into the characteristics and relevance of the target node within our cybersecurity context."
    return title + body


def _join_path_tags(path: Sequence[str]) -> str:
    """Render path as <PATH>a<SEP>b</PATH>."""
    if not path:
        return "<PATH></PATH>"
    if len(path) == 1:
        return f"<PATH>{path[0]}</PATH>"
    return "<PATH>" + "<SEP>".join(path) + "</PATH>"


def define_cot_prompt(
    graph: nx.Graph,
    name: Optional[str],
    entity_type: Optional[str],
    target: str,
    path: Sequence[str],
    rel_type_map: Dict[str, Dict[str, int]],
) -> str:
    """CoT for a single entity or none."""
    txt = _initial_phrase_single(graph, name, entity_type, target)
    current_nodes = [name] if name else [None]
    start_node_label = f"'{name}'" if name else None

    for i, rel in enumerate(path):
        for node in current_nodes:
            if i == 0 and node is not None:
                start_type = _get_node_type(graph, node)
                txt += _starting_point_section(start_type)
                txt += "\n\n**Step-by-Step Analysis**\n\n"

            # Most frequent neighbor type for this relation (heuristic)
            neighbor_type = None
            mapping = rel_type_map.get(rel)
            if mapping:
                neighbor_type = max(mapping, key=mapping.get)

            last = (i == len(path) - 1)
            step_line = _path_prompt(i + 1, neighbor_type if not last else target, rel, graph, node or "", last=last)
            txt += f"\n{step_line}"

    final_path = _join_path_tags(path)
    txt += f"\n{_final_prompt(final_path, graph, target, start_node_label)}"
    return txt


def define_cot_prompt_2(
    graph: nx.Graph,
    name_1: str,
    entity_type_1: str,
    name_2: str,
    entity_type_2: str,
    target: str,
    path: Sequence[str],
    rel_type_map: Dict[str, Dict[str, int]],
) -> str:
    """CoT for two entities."""
    header = f"""

**Task Analysis**

Let's reason step by step. We are working with the following entities:
- '{name_1}' is a {entity_type_1}
- '{name_2}' is a {entity_type_2}.
Our goal is : {target}.

**Entity Description**

Description of {name_1}: {_get_description_cached(graph, name_1)}
Description of {name_2}: {_get_description_cached(graph, name_2)}

**Starting Point (START)**

To begin, let's focus on nodes of type {entity_type_2} to establish our initial scope.

**Step-by-Step Analysis**
"""
    txt = header
    current_nodes = [name_2]
    last_neighbor = None

    for i, rel in enumerate(path):
        for node in current_nodes:
            mapping = rel_type_map.get(rel)
            if mapping:
                last_neighbor = max(mapping, key=mapping.get)
            neighbor = last_neighbor

            last = (i == len(path) - 1)
            if rel == "description" and neighbor is not None:
                step_line = _path_prompt(i + 1, neighbor, rel, graph, node, last=last)
            else:
                step_line = _path_prompt(i + 1, (target if last else neighbor), rel, graph, node, last=last)
            txt += f"\n{step_line}"

    final_path = _join_path_tags(path)
    txt += f"\n{_final_prompt(final_path, graph, target, f'\\'{name_2}\\'')}"
    return txt


def define_cot_prompt_n(
    graph: nx.Graph,
    names: Sequence[str],
    entity_types: Sequence[str],
    target: str,
    path: Sequence[str],
    rel_type_map: Dict[str, Dict[str, int]],
) -> str:
    """CoT for N entities."""
    header = f"""

**Task Analysis**

Let's reason step by step. We are working with the following entities:
"""
    for nm, et in zip(names, entity_types):
        header += f"- '{nm}' is a {et}.\n"

    header += f"Our goal is to {target}.\n\n**Entity Descriptions**\n\n"

    for nm, et in zip(names, entity_types):
        header += f"Description of {nm} ({et}): {_get_description_cached(graph, nm)}\n"

    # Starting point guess
    if path and "_type" in path[0]:
        starting_point = path[0].split("is_", 1)[1].split("_type")[0]
    else:
        src = get_source_of_relationship(path[0], graph) if path else None
        starting_point = _get_node_type(graph, src) if src else None

    header += f"\n**Starting Point (START)**\n\nTo begin, let's focus on nodes of type {starting_point} to establish our initial scope."
    txt = header + "\n\n**Step-by-Step Analysis**\n"

    current_nodes = [names[-1]] if names else []
    last_neighbor = None

    for i, rel in enumerate(path):
        for node in current_nodes:
            mapping = rel_type_map.get(rel)
            if mapping:
                last_neighbor = max(mapping, key=mapping.get)
            neighbor = last_neighbor

            last = (i == len(path) - 1)
            if rel == "description" and neighbor is not None:
                step_line = _path_prompt(i + 1, neighbor, rel, graph, node, last=last)
            else:
                step_line = _path_prompt(i + 1, (target if last else neighbor), rel, graph, node, last=last)
            txt += f"\n{step_line}"

    final_path = _join_path_tags(path)
    txt += f"\n{_final_prompt(final_path, graph, target, starting_point)}"
    return txt


# ---------------------------------------------------------------------------
# Placeholder utilities
# ---------------------------------------------------------------------------

def replace_placeholders_once(text: str, placeholders: Sequence[str], values: Sequence[str]) -> str:
    """Sequentially replace placeholders once each (stable ordering)."""
    for ph, val in zip(placeholders, values):
        text = text.replace(ph, val, 1)
    return text


def replace_placeholders_map(text: str, mapping: Dict[str, str]) -> str:
    """Replace placeholders like [EntityType] using a dict mapping."""
    for k, v in mapping.items():
        text = text.replace(f"[{k}]", v)
    return text


# ---------------------------------------------------------------------------
# Template instantiation (1, 2, N entities or none)
# ---------------------------------------------------------------------------

def generate_unique_combinations(names_by_type: Sequence[Sequence[str]], max_samples: int) -> List[Tuple[str, ...]]:
    """
    Generate unique combinations without reusing entities across combinations.
    """
    used: set = set()
    unique: List[Tuple[str, ...]] = []

    for combo in product(*names_by_type):
        if len(set(combo)) == len(combo) and not (used & set(combo)):
            unique.append(combo)
            used.update(combo)
        if len(unique) >= max_samples:
            break

    return unique


def process_combination(
    name_1: str,
    name_2: str,
    template: str,
    entity_type_1: str,
    entity_type_2: str,
    graph: nx.Graph,
    target: str,
    path: Sequence[str],
    rel_map: Dict[str, Dict[str, int]],
) -> Dict[str, str]:
    """Build (question, response) for 2-entity templates."""
    placeholders_q = [f"[{entity_type_1}]", f"[{entity_type_2}]"]
    values_q = [name_1, name_2]
    question = replace_placeholders_once(template, placeholders_q, values_q)

    path_placeholders = [f"<<{entity_type_1}>>", f"<<{entity_type_2}>>"]
    path_values = [name_1, name_2]
    updated_path = json.loads(replace_placeholders_once(json.dumps(path), path_placeholders, path_values))

    response = define_cot_prompt_2(graph, name_1, entity_type_1, name_2, entity_type_2, target, updated_path, rel_map)
    return {"question": question, "response": response}


def process_combination_n(
    names: Sequence[str],
    template: str,
    entity_types: Sequence[str],
    graph: nx.Graph,
    target: str,
    path: Sequence[str],
    rel_map: Dict[str, Dict[str, int]],
) -> Dict[str, str]:
    """Build (question, response) for N-entity templates."""
    question_ph = [f"[{et}]" for et in entity_types]
    question = replace_placeholders_once(template, question_ph, list(names))

    path_ph = [f"<<{et}>>" for et in entity_types]
    updated_path = json.loads(replace_placeholders_once(json.dumps(path), path_ph, list(names)))

    response = define_cot_prompt_n(graph, names, entity_types, target, updated_path, rel_map)
    return {"question": question, "response": response}


def generate_questions_for_type(
    graph: nx.Graph,
    template: str,
    path: Sequence[str],
    target: str,
    rel_type_map: Dict[str, Dict[str, int]],
    sample_percentage: float = 0.001,
    max_samples: int = 50,
) -> List[Dict[str, str]]:
    """
    Instantiate the template according to the number of entity placeholders.
    Pull entity names from the graph, apply text/target variations, and build (question, response).
    """
    pattern = r"\[(.*?)\]"
    types_in_template = re.findall(pattern, template)
    out: List[Dict[str, str]] = []

    mv_questions = load_multivariated_dataframe("TRAINING_FILES/question_variations.csv")
    mv_targets = load_multivariated_targets("TRAINING_FILES/target_variations.csv")

    print(f"OLD TARGET: {target}")
    tgt_list = get_target_for_question(template, mv_targets)
    if tgt_list:
        target = tgt_list[0]
    print(f"NEW TARGET: {target}")

    def random_variation(original: str) -> str:
        if mv_questions is None:
            return original
        row = mv_questions.loc[mv_questions["Original Question"] == original]
        if not row.empty:
            try:
                variations = [v.strip() for v in row.iloc[0]["Variations"].split("; ") if v.strip()]
                return random.choice(variations) if variations else original
            except Exception:
                return original
        return original

    def all_variations(original: str) -> List[str]:
        if mv_questions is None:
            return [original]
        row = mv_questions.loc[mv_questions["Original Question"] == original]
        if not row.empty:
            return [v.strip() for v in row.iloc[0]["Variations"].split("; ") if v.strip()]
        return [original]

    def replace_map(text: str, repl: Dict[str, str]) -> str:
        for etype, name in repl.items():
            text = text.replace(f"[{etype}]", name)
        return text

    # Single-entity templates
    if len(types_in_template) == 1:
        etype = types_in_template[0]
        names = get_names_by_type(graph, etype.lower())
        for nm in names:
            var_tmpl = random_variation(template)
            var_tgt = random_variation(target)
            repl = {etype: nm}
            question = replace_map(var_tmpl, repl)
            final_target = replace_map(var_tgt, repl)
            response = define_cot_prompt(graph, nm, etype, final_target, path, rel_type_map)
            out.append({"question": question, "response": response})

    # Two-entity templates
    elif len(types_in_template) == 2:
        print("Double entity template detected")
        et1, et2 = types_in_template
        clean = lambda x: x.replace("-", "_") if "-" in x else x
        names_1 = get_names_by_type_cached(graph, clean(et1.lower()))
        names_2 = get_names_by_type_cached(graph, clean(et2.lower()))

        combos = list(product(names_1, names_2))
        sample_size = min(int(len(combos) * sample_percentage), max_samples)
        sampled = random.sample(combos, sample_size) if combos and sample_size > 0 else []
        print(f"Generated {len(combos)} combinations. Processing {len(sampled)} of them.")

        # Fixed: argument order when submitting to thread pool
        with ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(
                    process_combination,
                    n1,
                    n2,
                    random_variation(template),   # template
                    et1,
                    et2,
                    graph,
                    random_variation(target),     # target variation
                    path,
                    rel_type_map,
                )
                for (n1, n2) in tqdm(sampled, desc="Scanning sampled combinations...")
            ]
            for fut in futures:
                out.append(fut.result())

    # Three-or-more-entity templates
    elif len(types_in_template) >= 3:
        print(f"Detected {len(types_in_template)} entity types in the template.")
        clean = lambda x: x.replace("-", "_") if "-" in x else x
        etypes = [clean(e.lower()) for e in types_in_template]
        names_by_type = [
            random.sample(get_names_by_type_cached(graph, e), min(len(get_names_by_type_cached(graph, e)), max_samples // max(1, len(etypes))))
            for e in etypes
        ]

        print(f"Sampling names for {len(etypes)} entity types.")
        unique_combos = generate_unique_combinations(names_by_type, max_samples)
        print(f"Generated {len(unique_combos)} unique combinations. Processing {len(unique_combos)} of them.")

        for combo in tqdm(unique_combos, desc=f"Scanning unique combinations ({len(types_in_template)} types)..."):
            repl = dict(zip(types_in_template, combo))
            var_tmpl = random_variation(template)
            var_tgt = random_variation(target)
            question = replace_map(var_tmpl, repl)
            final_target = replace_map(var_tgt, repl)
            out.append(
                process_combination_n(combo, question, types_in_template, graph, final_target, path, rel_type_map)
            )

    # No-entity templates
    else:
        variations = all_variations(template)
        print(f"Template without entities detected. Using all {len(variations)} variations.")
        for var in variations:
            response = define_cot_prompt(graph, None, None, target, path, rel_type_map)
            out.append({"question": var, "response": response})

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_questions(
    template_file: str,
    graph_file: str,
    output_file: str,
    section_output_file: str,
) -> None:
    """Main orchestration: load inputs, generate all Q/A pairs, save outputs."""
    templates = load_templates(template_file)
    graph = load_graph(graph_file)

    # Build the relation->type map once and reuse
    rel_type_map = create_relationship_type_mapping(graph)

    all_items: List[Dict[str, str]] = []
    per_section: Dict[str, List[Dict[str, str]]] = {}

    for section_name, entries in templates.get("templates", {}).items():
        print(f"\n----------------------\nAnalyzing Section: {section_name}")
        section_items: List[Dict[str, str]] = []

        for entry in tqdm(entries, desc="Scanning questions ..."):
            if not isinstance(entry, dict):
                print(f"Skipping entry, unexpected format: {entry}")
                continue

            q_template = entry["question"]
            path = entry["path"]
            target = entry["target"][0]

            # Skip specific degenerate path if needed (kept from original behavior)
            if path == "<PATH>description</PATH>":
                continue

            if section_name == "entity_in_path":
                # Cap number of generated questions for that section
                items = generate_questions_for_type(graph, q_template, path, target, rel_type_map, sample_percentage=0.001)
                if len(items) > 600:
                    items = items[:600]
            else:
                items = generate_questions_for_type(graph, q_template, path, target, rel_type_map)

            all_items.extend(items)
            section_items.extend(items)

        if not section_items:
            print(f"Warning: No questions were generated for section '{section_name}'.")
        per_section[section_name] = section_items

    save_generated_questions(output_file, all_items)
    with open(section_output_file, "w", encoding="utf-8") as f:
        json.dump(per_section, f, indent=4)

    print(f"\nTotal number of questions generated: {len(all_items)}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate QA + CoT dataset from CTI graph and templates.")
    ap.add_argument("--templates", default="useful_cot.yaml", help="YAML template file")
    ap.add_argument("--graph", default="stix_graph_correct.graphml", help="GraphML file")
    ap.add_argument("--out", default="NAVIGATION_DATASET.json", help="Output JSON (flat)")
    ap.add_argument("--out-sections", default="NAVIGATION_QUESTION_PER_SECTION.json", help="Output JSON (per section)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_questions(args.templates, args.graph, args.out, args.out_sections)
