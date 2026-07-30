#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert the TITAN GraphML knowledge graph to RDF (N-Triples), for the
SPARQL baseline (Layer 1 of the DSL-vs-SPARQL comparison).

Mapping — deliberately minimal and bijective so equivalence with the native
graph executor is checkable:
  * every node  -> URI  <http://titan.local/n/<urlencoded name>>
                  plus one rdfs:label triple carrying the original string
  * every edge (u, label, v) -> triple  <n/u> <r/<urlencoded label>> <n/v>
Nothing is dropped, renamed, or literal-ized: description texts and
attribute values stay resource nodes exactly as in the GraphML, so a SPARQL
traversal visits the same node set the executor does. URI -> name is
recoverable by URL-decoding the suffix (no lookup table needed).

Also writes rel_target_types.json: relation label -> most frequent
(source_type, target_type) pair, needed by the path->SPARQL compiler to know
which chain positions produce nodes of a given type (mirrors the executor's
accumulate-by-type semantics at compile time).

Run:
    python3 utils/graph_to_rdf.py --graph stix_graph_correct.graphml \
        --out titan_graph.nt --types rel_target_types.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from collections import Counter
from urllib.parse import quote, unquote

import networkx as nx
from rdflib import Graph as RDFGraph, Literal, URIRef
from rdflib.namespace import RDFS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NODE_NS = "http://titan.local/n/"
REL_NS = "http://titan.local/r/"


def node_uri(name: str) -> URIRef:
    return URIRef(NODE_NS + quote(str(name), safe=""))


def rel_uri(label: str) -> URIRef:
    return URIRef(REL_NS + quote(str(label), safe=""))


def uri_to_name(uri: str) -> str:
    for ns in (NODE_NS, REL_NS):
        if uri.startswith(ns):
            return unquote(uri[len(ns):])
    return uri


def _node_type(g: nx.Graph, node: str):
    try:
        for nb in g.neighbors(node):
            lab = g[node][nb].get("label")
            if isinstance(lab, str) and "type" in lab:
                return nb
    except Exception:
        pass
    return None


def convert(graph_file: str, out_nt: str, out_types: str) -> None:
    print(f"[INFO] loading {graph_file} ...")
    g = nx.read_graphml(graph_file)
    print(f"[INFO] {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")

    rdf = RDFGraph()
    type_sig: dict[str, Counter] = {}
    node_types: dict[str, str | None] = {}

    def ntype(n):
        if n not in node_types:
            node_types[n] = _node_type(g, n)
        return node_types[n]

    for n in g.nodes:
        rdf.add((node_uri(n), RDFS.label, Literal(str(n))))

    for u, v, data in g.edges(data=True):
        label = data.get("label")
        if not label:
            continue
        rdf.add((node_uri(u), rel_uri(label), node_uri(v)))
        type_sig.setdefault(label, Counter())[(ntype(u) or "?", ntype(v) or "?")] += 1

    rdf.serialize(destination=out_nt, format="nt", encoding="utf-8")
    print(f"[OK] {len(rdf)} triples -> {out_nt}")

    rel_types = {}
    for label, counts in type_sig.items():
        (src_t, tgt_t), _ = counts.most_common(1)[0]
        rel_types[label] = {"source_type": src_t, "target_type": tgt_t,
                            "signatures": {f"{s}->{t}": c for (s, t), c in counts.most_common()}}
    with open(out_types, "w", encoding="utf-8") as f:
        json.dump(rel_types, f, indent=1)
    print(f"[OK] {len(rel_types)} relation type signatures -> {out_types}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--graph", default="stix_graph_correct.graphml")
    ap.add_argument("--out", default="titan_graph.nt")
    ap.add_argument("--types", default="rel_target_types.json")
    a = ap.parse_args()
    convert(a.graph, a.out, a.types)
