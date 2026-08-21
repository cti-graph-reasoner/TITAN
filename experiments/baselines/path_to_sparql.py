#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechanical compiler: TITAN path DSL -> SPARQL, over the RDF graph produced
by utils/graph_to_rdf.py.

The compiler mirrors titan.evaluate.execute_path() semantics construct by
construct (that function defines the gold answer sets, so equivalence with
it is the correctness criterion — validated by
baselines/validate_sparql_equivalence.py):

  entity seed          VALUES ?x0 { <n/Entity> }
  relation step        ?x_i <r/rel> ?x_{i+1}
  filter <kw>          ?x_i <r/description> ?d ; ?d rdfs:label ?dl ;
                       FILTER(CONTAINS(LCASE(?dl), "kw"))   [and/or supported;
                       keeps the current variable, does not advance]
  is_<X>_type          ?x_i <r/is_<X>_type> ?t   [existence check; seeds ?x0
                       when there is no entity]
  select A B ...       one branch per named entity (variables renamed per
                       branch); with no later exec_* step the branches'
                       final variables are UNIONed (executor: union of
                       branch sets)
  exec_common T /      the executor combines nodes of type T ACCUMULATED
  exec_difference T    along each branch (not just final sets). At compile
                       time the accumulating positions are the relation
                       steps whose target type is T (from
                       rel_target_types.json), plus a select-branch's seed
                       if its own type is T. Each branch's accumulated set
                       compiles to a UNION over those positions projected as
                       ?ans; branches then combine by join (common) or
                       symmetric difference ((A MINUS B) UNION (B MINUS A)).

Known approximation (checked, not assumed, by the validator): a relation
step's target type is taken as the relation's DOMINANT target type in the
KG; relations with mixed target types could diverge from the executor's
per-node typing. Divergences show up as validator mismatches.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional, Sequence
from urllib.parse import quote, unquote

NODE_NS = "http://titan.local/n/"
REL_NS = "http://titan.local/r/"
RDFS_LABEL = "<http://www.w3.org/2000/01/rdf-schema#label>"


def n_uri(name: str) -> str:
    return f"<{NODE_NS}{quote(str(name), safe='')}>"


def uri_to_name(uri: str) -> str:
    for ns in (NODE_NS, REL_NS):
        if uri.startswith(ns):
            return unquote(uri[len(ns):])
    return uri


def r_uri(label: str) -> str:
    return f"<{REL_NS}{quote(str(label), safe='')}>"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _filter_condition(var: str, keywords: str, idx: int) -> tuple[str, str]:
    """Pattern + FILTER for one `filter <keywords>` step (executor semantics:
    'a and b' all in the SAME description; 'a or b' any; else single)."""
    text = keywords.strip().lower()
    if " and " in text:
        conds = [c.strip() for c in text.split(" and ")]
        joiner = " && "
    elif " or " in text:
        conds = [c.strip() for c in text.split(" or ")]
        joiner = " || "
    else:
        conds, joiner = [text], ""
    d, dl = f"?d{idx}", f"?dl{idx}"
    pattern = f"{var} {r_uri('description')} {d} . {d} {RDFS_LABEL} {dl} ."
    cond = joiner.join(f'CONTAINS(LCASE(STR({dl})), "{_esc(c)}")' for c in conds)
    return pattern, f"FILTER({cond})"


class Branch:
    """One traversal branch: seed + compiled pattern + accumulated typed positions."""

    def __init__(self, tag: str, seed_name: Optional[str], seed_type: Optional[str],
                 seeded_by_select: bool):
        self.tag = tag
        self.seed_name = seed_name
        self.seed_type = seed_type
        self.seeded_by_select = seeded_by_select
        self.lines: List[str] = []
        self.var_i = 0
        self.cur = f"?{tag}x0"
        if seed_name is not None:
            self.lines.append(f"VALUES {self.cur} {{ {n_uri(seed_name)} }}")
        # (variable, target_type) for every relation step, in order
        self.rel_positions: List[tuple[str, Optional[str]]] = []
        self.filter_i = 0

    def next_var(self) -> str:
        self.var_i += 1
        return f"?{self.tag}x{self.var_i}"

    def add_relation(self, rel: str, target_type: Optional[str]) -> None:
        nxt = self.next_var()
        self.lines.append(f"{self.cur} {r_uri(rel)} {nxt} .")
        self.cur = nxt
        self.rel_positions.append((nxt, target_type))

    def add_filter(self, keywords: str) -> None:
        self.filter_i += 1
        pat, flt = _filter_condition(self.cur, keywords, self.filter_i)
        self.lines.append(pat)
        self.lines.append(flt)

    def add_type_check(self, type_rel: str, seed_if_empty: bool) -> None:
        if seed_if_empty and not self.lines:
            # blind start: seeding from all sources of is_<X>_type
            self.lines.append(f"{self.cur} {r_uri(type_rel)} ?t{self.var_i} .")
        else:
            self.lines.append(f"{self.cur} {r_uri(type_rel)} ?t{self.var_i}t .")

    def pattern(self) -> str:
        return "\n    ".join(self.lines)

    def acc_projections(self, acc_type: Optional[str]) -> List[str]:
        """WHERE-blocks each binding ?ans to one accumulating position."""
        blocks = []
        for var, ttype in self.rel_positions:
            if acc_type is None or ttype == acc_type:
                blocks.append(f"{{ {self.pattern()} BIND({var} AS ?ans) }}")
        if (acc_type is not None and self.seeded_by_select
                and self.seed_type == acc_type and self.seed_name is not None):
            blocks.append(f"{{ VALUES ?ans {{ {n_uri(self.seed_name)} }} }}")
        return blocks

    def final_projection(self) -> str:
        return f"{{ {self.pattern()} BIND({self.cur} AS ?ans) }}"


def _union(blocks: Sequence[str]) -> str:
    if not blocks:
        return '{ VALUES ?ans { } }'  # provably empty
    return "\n    UNION\n    ".join(blocks)


def compile_path(
    steps: Sequence[str],
    entities: Sequence[str],
    rel_types: dict,
    node_type_fn: Callable[[str], Optional[str]],
    parse_select_fn: Callable[[str], List[str]],
) -> str:
    """Compile one DSL path to a single SPARQL SELECT (DISTINCT ?ans)."""

    def target_type(rel: str) -> Optional[str]:
        info = rel_types.get(rel)
        t = info["target_type"] if info else None
        return None if t in (None, "?") else t

    # --- pass 1: split at select / find exec op ------------------------
    exec_op = exec_type = None
    branches: List[Branch] = []
    if entities:
        branches = [Branch(f"b{i}_", e, node_type_fn(e), False)
                    for i, e in enumerate(entities)]
    else:
        branches = [Branch("b0_", None, None, False)]

    for step in (s.strip() for s in steps):
        if step.startswith("select"):
            args = step[len("select"):].strip()
            names = parse_select_fn(args) if args else []
            if names:
                branches = [Branch(f"b{i}_", nm, node_type_fn(nm), True)
                            for i, nm in enumerate(names)]
            continue
        if step.startswith("exec_"):
            parts = step.split(None, 1)
            exec_op = parts[0][len("exec_"):]
            exec_type = parts[1].strip() if len(parts) > 1 else None
            break  # executor also stops accumulating traversal here
        for br in branches:
            if step.startswith("is_") and step.endswith("_type"):
                # exact type-seed convention, NOT a plain "_type" substring
                # check -- x_mitre_impact_type / x_mitre_tactic_type are
                # ordinary MITRE attribute relations that happen to contain
                # that substring; they must compile as ordinary relation
                # traversal (add_relation), matching the fixed
                # titan.evaluate.execute_path. See
                # experiments/titan_findings.csv, executor_bug.
                br.add_type_check(step, seed_if_empty=br.seed_name is None)
            elif step.startswith("filter "):
                br.add_filter(step[len("filter "):].strip())
            else:
                br.add_relation(step, target_type(step))

    # --- pass 2: assemble ----------------------------------------------
    if exec_op is None:
        body = _union([br.final_projection() for br in branches])
    else:
        per_branch = ["{ " + _union(br.acc_projections(exec_type)) + " }"
                      for br in branches]
        if len(per_branch) == 1:
            body = per_branch[0]
        elif exec_op == "common":
            body = "\n    ".join(per_branch)  # join on ?ans = intersection
        elif exec_op == "difference":       # symmetric, like the executor's ^
            a, b = per_branch[0], per_branch[1]
            body = (f"{{ {a} MINUS {b} }}\n    UNION\n    {{ {b} MINUS {a} }}")
            for extra in per_branch[2:]:     # fold further branches pairwise
                body = (f"{{ {{ {body} }} MINUS {extra} }}\n    UNION\n    "
                        f"{{ {extra} MINUS {{ {body} }} }}")
        else:
            body = '{ VALUES ?ans { } }'

    return f"SELECT DISTINCT ?ans WHERE {{\n    {body}\n}}"


def load_rel_types(path: str = "rel_target_types.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # smoke test on paper examples (types faked where irrelevant)
    rel_types = load_rel_types()
    q = compile_path(
        ["uses_malware", "filter backdoor", "uses_attack_pattern", "mitigated_by"],
        ["Ke3chang"], rel_types, lambda n: "intrusion_set", lambda s: s.split())
    print(q)
    print()
    q2 = compile_path(
        ["is_malware_type", "select FALLCHILL BlackEnergy 3",
         "uses_attack_pattern", "mitigated_by", "exec_common course_of_action"],
        [], rel_types, lambda n: "malware",
        lambda s: ["FALLCHILL", "BlackEnergy 3"])
    print(q2)
