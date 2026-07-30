#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the paper's results tables directly from the *_eval_report.json files
produced by titan/evaluate.py / score_sparql_predictions.py -- no numbers
retyped by hand.

Emits TWO tables into results_table.tex:

  Table 1 (main, tab:main-results): exact-match only (Path EM for DSL,
  Query EM for SPARQL -- see titan_findings.csv, category metric_addition,
  for why EM and not F1 is the headline number). Columns grouped NoCoT/CoT
  (outer) x DSL/SPARQL (inner); rows grouped Prompted/Fine-tuned via a
  rotated side label instead of per-row text, to save horizontal space.
  The best (max) value in each column is bolded so the strongest model per
  condition is immediately visible.

  Table 2 (tab:by-length-operator): exact match broken down by path length
  (hop count: L1-L3, L4+) and by operator type present in the gold path
  (exec_common/exec_difference/filter/plain/select), one row per model x
  condition. EM is used (not F1) since F1 is NaN whenever the gold answer
  set is empty, which would leave gaps in the smaller operator buckets.

gpt-oss-120b and DeepSeek-R1-70B are placed in the CoT column even though
prompted with the NoCoT-style instruction: both are native reasoning
models that produce an internal reasoning trace before answering regardless
of the instruction wording (Harmony analysis channel / <think> tags).

Missing cells / rows (experiment not run, or still in progress) render as
"--" in Table 1 and are simply omitted (no data to show) in Table 2.

Requires in the paper's preamble: \\usepackage{booktabs,multirow,graphicx}

Run: python3 build_latex_table.py > results_table.tex
"""

from __future__ import annotations

import json
from typing import Optional


def load(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["overall"]
    except FileNotFoundError:
        return None


# (row label, group, dsl_nocot, dsl_cot, sparql_nocot, sparql_cot)
# group: "prompted" or "finetuned"
ROWS = [
    ("Llama-3.3-70B", "prompted",
     "experiments/baselines/llama33_70b_heldout_full_v3_eval_report.json",
     "experiments/baselines/llama33_70b_dsl_cot_heldout_full_eval_report.json",
     "experiments/baselines/llama33_70b_sparql_heldout_full_eval_report.json",
     "experiments/baselines/llama33_70b_sparql_cot_heldout_full_eval_report.json"),
    ("Qwen2.5-7B-Instruct", "prompted",
     "experiments/baselines/qwen25_7b_prompted_dsl_heldout_full_eval_report.json",
     "experiments/baselines/qwen25_7b_prompted_dsl_cot_heldout_full_eval_report.json",
     "experiments/baselines/qwen25_7b_prompted_sparql_heldout_full_eval_report.json",
     "experiments/baselines/qwen25_7b_prompted_sparql_cot_heldout_full_eval_report.json"),
    ("Qwen2.5-72B", "prompted",
     "experiments/baselines/qwen25_72b_heldout_full_v3_eval_report.json",
     "experiments/baselines/qwen25_72b_cot_prompted_heldout_full_eval_report.json",
     "experiments/baselines/qwen25_72b_sparql_heldout_full_eval_report.json",
     "experiments/baselines/qwen25_72b_sparql_cot_heldout_full_eval_report.json"),
    ("GPT-OSS-120B$^\\dagger$", "prompted",
     None,
     "experiments/baselines/gptoss_120b_heldout_full_v3_eval_report.json",
     None,
     "experiments/baselines/gptoss_120b_sparql_cot_heldout_full_eval_report.json"),
    ("DeepSeek-R1-70B$^\\dagger$", "prompted",
     None,
     "experiments/baselines/deepseek_r1_70b_heldout_full_v3_eval_report.json",
     None,
     "experiments/baselines/deepseek_r1_70b_sparql_cot_heldout_full_eval_report.json"),
    ("Phi-3.5-mini (3.8B)", "finetuned",
     "experiments/baselines/phi_titan_nocot_v2_heldout_eval_report.json",
     "experiments/baselines/phi_titan_cot_v2_heldout_eval_report.json",
     "experiments/baselines/phi_titan_sparql_heldout_eval_report.json",
     "experiments/baselines/phi_titan_sparql_cot_heldout_eval_report.json"),
    ("Qwen2.5-7B", "finetuned",
     "experiments/baselines/qwen25_7b_titan_nocot_heldout_eval_report.json",
     "experiments/baselines/qwen25_7b_titan_cot_heldout_eval_report.json",
     "experiments/baselines/qwen25_7b_titan_sparql_heldout_eval_report.json",
     "experiments/baselines/qwen25_7b_titan_sparql_cot_heldout_eval_report.json"),
]

CONDITIONS = [
    ("DSL (NoCoT)", "dsl_nc", False),
    ("SPARQL (NoCoT)", "sparql_nc", True),
    ("DSL (CoT)", "dsl_c", False),
    ("SPARQL (CoT)", "sparql_c", True),
]


def em_of(path: Optional[str], is_sparql: bool) -> Optional[float]:
    d = load(path)
    if d is None:
        return None
    return d.get("query_em" if is_sparql else "path_em")


def load_full(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def em_by_bucket(full_report: Optional[dict], bucket_type: str, bucket_name: str,
                 is_sparql: bool) -> Optional[float]:
    """bucket_type: 'by_length' or 'by_operator'. EM, not F1: chosen as the
    breakdown metric because it has no NaN-on-empty-gold edge case (unlike
    F1) and is the metric this project treats as authoritative (see
    titan_findings.csv, category metric_addition)."""
    if full_report is None:
        return None
    key = "query_em" if is_sparql else "path_em"
    for b in full_report.get(bucket_type, []):
        if b["bucket"] == bucket_name:
            return b.get(key)
    return None


# ---------------------------------------------------------------------------
# Table 1: EM only, bold column max. Single-column width: no side-label
# column (a rotated Prompted/Fine-tuned label costs a whole column that a
# single-column float can't spare) -- group boundaries are instead plain
# \multicolumn header rows ("Prompted" / "Fine-tuned"), and model names are
# shortened where that was the only thing forcing a wider table.
# ---------------------------------------------------------------------------

SHORT_NAME = {
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instr.",
    "Phi-3.5-mini (3.8B)": "Phi-3.5-mini",
}


def build_table1() -> str:
    # em_matrix[row_idx][col_idx] = float or None
    em_matrix = []
    for _, _, dsl_nc, dsl_c, sparql_nc, sparql_c in ROWS:
        em_matrix.append([
            em_of(dsl_nc, False), em_of(sparql_nc, True),
            em_of(dsl_c, False), em_of(sparql_c, True),
        ])
    col_max = [max((row[c] for row in em_matrix if row[c] is not None), default=None)
              for c in range(4)]

    def fmt(v: Optional[float], c: int) -> str:
        if v is None:
            return "--"
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if col_max[c] is not None and abs(v - col_max[c]) < 1e-9 else s

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l cccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{\textbf{NoCoT}} & \multicolumn{2}{c}{\textbf{CoT}} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r"\textbf{Model} & DSL & SPARQL & DSL & SPARQL \\")
    lines.append(r"\midrule")

    groups = [("prompted", "Prompted", 5), ("finetuned", "Fine-tuned", 2)]
    row_idx = 0
    for gi, (gkey, glabel, gsize) in enumerate(groups):
        if gi > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{glabel}}}}} \\\\")
        for _ in range(gsize):
            label, group, *_ = ROWS[row_idx]
            assert group == gkey
            label = SHORT_NAME.get(label, label)
            cells = " & ".join(fmt(em_matrix[row_idx][c], c) for c in range(4))
            lines.append(f"{label} & {cells} \\\\")
            row_idx += 1

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Exact match (Path EM for DSL, Query EM for SPARQL -- exact match "
                r"against the canonical compiled gold SPARQL query) on the template-disjoint, "
                r"leakage-free \emph{test\_heldout} split, grouped by reasoning style (NoCoT: "
                r"direct answer; CoT: reason then answer) and target formalism. Bold marks the "
                r"best value in each column. $\dagger$gpt-oss-120b and DeepSeek-R1-Distill-"
                r"Llama-70B are listed under CoT despite an NoCoT-style prompt instruction: both "
                r"are native reasoning models that produce an internal reasoning trace before "
                r"answering regardless of instruction wording. \emph{--} marks an experiment not "
                r"run or still in progress. A breakdown by path length and operator type is in "
                r"Table~\ref{tab:by-length-operator}.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: EM broken down by path length and by operator type, long format
# (one row per model x condition). EM chosen over F1 as the breakdown metric
# -- see em_by_bucket() docstring.
# ---------------------------------------------------------------------------

LENGTH_BUCKETS = ["L1", "L2", "L3", "L4+"]
OPERATOR_BUCKETS = [
    ("exec_common", "Com"), ("exec_difference", "Diff"), ("filter", "Filt"),
    ("plain", "Plain"), ("select", "Sel"),
]


def build_table2() -> str:
    ncols = len(LENGTH_BUCKETS) + len(OPERATOR_BUCKETS)

    # ---- Pass 1: gather every (model, condition) row that has data --------
    # row = {"model": label, "group": prompted/finetuned, "cond": cond_label,
    #        "cells": [float or None] * ncols}
    rows_data = []
    for label, group, dsl_nc, dsl_c, sparql_nc, sparql_c in ROWS:
        paths = {"dsl_nc": dsl_nc, "dsl_c": dsl_c, "sparql_nc": sparql_nc, "sparql_c": sparql_c}
        for cond_label, key, is_sparql in CONDITIONS:
            d = load_full(paths[key])
            if d is None:
                continue
            len_cells = [em_by_bucket(d, "by_length", b, is_sparql) for b in LENGTH_BUCKETS]
            op_cells = [em_by_bucket(d, "by_operator", b, is_sparql) for b, _ in OPERATOR_BUCKETS]
            rows_data.append({"model": label, "group": group, "cond": cond_label,
                              "cells": len_cells + op_cells})

    # ---- Pass 2: DSL-vs-SPARQL comparison, WITHIN the same model AND the
    # same reasoning style (NoCoT or CoT) -- never across models, never
    # across NoCoT vs CoT. For each (model, style) pair that has both a DSL
    # and a SPARQL row, count which one is the column-max more often; bold
    # that row's CONDITION LABEL only (not the numbers).
    def style_of(cond: str) -> str:
        return "CoT" if "(CoT)" in cond else "NoCoT"

    by_model_style: dict = {}
    for r in rows_data:
        by_model_style.setdefault((r["model"], style_of(r["cond"])), []).append(r)

    winners = set()  # ids of rows_data entries whose condition label gets bolded
    for (_model, _style), pair_rows in by_model_style.items():
        if len(pair_rows) < 2:
            continue  # nothing to compare (only one formalism tested for this style)
        col_max = [max((r["cells"][c] for r in pair_rows if r["cells"][c] is not None),
                       default=None)
                  for c in range(ncols)]
        win_counts = []
        for r in pair_rows:
            wc = sum(1 for c in range(ncols)
                    if r["cells"][c] is not None and col_max[c] is not None
                    and abs(r["cells"][c] - col_max[c]) < 1e-9)
            win_counts.append(wc)
        best = max(win_counts)
        for r, wc in zip(pair_rows, win_counts):
            if wc == best and best > 0:
                winners.add(id(r))

    # ---- Pass 3: emit, grouped Prompted/Fine-tuned (rotated side label) ->
    # model (multirow) -> condition rows, with a rule between models and a
    # lighter rule between the NoCoT and CoT sub-blocks within a model.
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{c l l " + "c" * ncols + "}")
    lines.append(r"\toprule")
    lines.append(f"  &  & & \\multicolumn{{{len(LENGTH_BUCKETS)}}}{{c}}{{\\textbf{{By length}}}} & "
                f"\\multicolumn{{{len(OPERATOR_BUCKETS)}}}{{c}}{{\\textbf{{By operator}}}} \\\\")
    lines.append(f"\\cmidrule(lr){{4-{3 + len(LENGTH_BUCKETS)}}} "
                f"\\cmidrule(lr){{{4 + len(LENGTH_BUCKETS)}-{3 + ncols}}}")
    length_hdr = " & ".join(LENGTH_BUCKETS)
    op_hdr = " & ".join(abbr for _, abbr in OPERATOR_BUCKETS)
    lines.append(r" & \textbf{Model} & \textbf{Cond.} & " + length_hdr + " & " + op_hdr + r" \\")
    lines.append(r"\midrule")

    groups = [("prompted", "Prompted"), ("finetuned", "Fine-tuned")]
    for gi, (gkey, glabel) in enumerate(groups):
        group_models = [(label, group) for label, group, *_ in ROWS if group == gkey]
        # total visible rows in this group, for the rotated label's row-span
        group_row_count = sum(1 for r in rows_data if r["group"] == gkey)
        if group_row_count == 0:
            continue
        if gi > 0:
            lines.append(r"\midrule")

        first_in_group = True
        for label, _ in group_models:
            model_rows = [r for r in rows_data if r["model"] == label]
            if not model_rows:
                continue
            if not first_in_group:
                # starts at column 2 (Model), NOT column 1, so it doesn't cut
                # across the rotated Prompted/Fine-tuned side label's multirow
                lines.append(f"\\cmidrule(lr){{2-{3 + ncols}}}")
            side_prefix = (f"\\multirow{{{group_row_count}}}{{*}}{{\\rotatebox{{90}}"
                          f"{{\\textbf{{{glabel}}}}}}}" if first_in_group else "")
            first_in_group = False

            prev_style = None
            for i, r in enumerate(model_rows):
                style = style_of(r["cond"])
                if prev_style is not None and style != prev_style:
                    # starts at column 3 (Condition): also spares column 2
                    # (Model), whose multirow spans across both styles
                    lines.append(f"\\cmidrule(l){{3-{3 + ncols}}}")
                prev_style = style

                cell_strs = [f"{v:.3f}" if v is not None else "--" for v in r["cells"]]
                cells = " & ".join(cell_strs)
                model_cell = f"\\multirow{{{len(model_rows)}}}{{*}}{{{label}}}" if i == 0 else ""
                cond_cell = f"\\textbf{{{r['cond']}}}" if id(r) in winners else r["cond"]
                prefix = side_prefix if i == 0 else ""
                side_prefix = ""
                lines.append(f"{prefix} & {model_cell} & {cond_cell} & {cells} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Exact match on \emph{test\_heldout}, broken down by path length "
                r"(number of relation hops: L1--L3, L4+ for four or more) and by operator type "
                r"present in the gold path (companion to Table~\ref{tab:main-results}). "
                r"\textbf{Com} = exec\_common (set intersection), \textbf{Diff} = "
                r"exec\_difference (set difference), \textbf{Filt} = filter, \textbf{Plain} = "
                r"plain multi-hop traversal with no filter/set operator, \textbf{Sel} = select "
                r"(multi-entity comparison). EM is used rather than F1 for this breakdown since "
                r"F1 is undefined (NaN) whenever the gold answer set is empty, which would leave "
                r"gaps in some of the smaller operator buckets. \emph{--} marks a bucket with no "
                r"rows for that model/condition. \textbf{Bold} on a condition label marks the "
                r"better-performing formalism (DSL vs.\ SPARQL) \emph{within the same model and "
                r"the same reasoning style} -- e.g.\ Qwen2.5-7B's DSL~(NoCoT) row is compared "
                r"only against its own SPARQL~(NoCoT) row, never against another model's row or "
                r"against its own CoT rows -- based on which is the column-maximum more often "
                r"across the length/operator breakdown.}")
    lines.append(r"\label{tab:by-length-operator}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: paraphrase robustness, ALL 8 fine-tuned configs (2 backbones x 4
# conditions), matched 1502-row subset -- SAME questions' gold answers, only
# the phrasing differs.
# ---------------------------------------------------------------------------

PARAPHRASE_ROWS = [
    ("Qwen2.5-7B", "DSL (NoCoT)", "path_em",
     "experiments/baselines/qwen25_7b_titan_nocot_matched1502_eval_report.json",
     "experiments/baselines/qwen25_7b_titan_nocot_paraphrased_eval_report.json"),
    ("Qwen2.5-7B", "SPARQL (NoCoT)", "query_em",
     "experiments/baselines/qwen25_7b_titan_sparql_matched1502_eval_report.json",
     "experiments/baselines/qwen25_7b_titan_sparql_paraphrased_eval_report.json"),
    ("Qwen2.5-7B", "DSL (CoT)", "path_em",
     "experiments/baselines/qwen25_7b_titan_cot_matched1502_eval_report.json",
     "experiments/baselines/qwen25_7b_titan_cot_paraphrased_eval_report.json"),
    ("Qwen2.5-7B", "SPARQL (CoT)", "query_em",
     "experiments/baselines/qwen25_7b_titan_sparql_cot_matched1502_eval2_report.json",
     "experiments/baselines/qwen25_7b_titan_sparql_cot_paraphrased_eval2_report.json"),
    ("Phi-3.5-mini", "DSL (NoCoT)", "path_em",
     "experiments/baselines/phi_titan_nocot_matched1502_eval_report.json",
     "experiments/baselines/phi_titan_nocot_paraphrased_eval_report.json"),
    ("Phi-3.5-mini", "SPARQL (NoCoT)", "query_em",
     "experiments/baselines/phi_titan_sparql_matched1502_eval_report.json",
     "experiments/baselines/phi_titan_sparql_paraphrased_eval_report.json"),
    ("Phi-3.5-mini", "DSL (CoT)", "path_em",
     "experiments/baselines/phi_titan_cot_matched1502_eval_report.json",
     "experiments/baselines/phi_titan_cot_paraphrased_eval2_report.json"),
    ("Phi-3.5-mini", "SPARQL (CoT)", "query_em",
     "experiments/baselines/phi_titan_sparql_cot_matched1502_eval2_report.json",
     "experiments/baselines/phi_titan_sparql_cot_paraphrased_eval_report.json"),
]


def build_table3() -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l ccc ccc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c}{\textbf{EM}} & \multicolumn{3}{c}{\textbf{F1}} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(r"\textbf{Model} & \textbf{Condition} & Orig. & Para. & $\Delta$ & Orig. & "
                r"Para. & $\Delta$ \\")
    lines.append(r"\midrule")
    prev_model = None
    model_rows = {}
    for model, _, _, _, _ in PARAPHRASE_ROWS:
        model_rows[model] = model_rows.get(model, 0) + 1
    for model, cond_label, em_key, orig_path, para_path in PARAPHRASE_ROWS:
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        first = prev_model != model
        prev_model = model
        model_cell = f"\\multirow{{{model_rows[model]}}}{{*}}{{{model}}}" if first else ""
        orig, para = load(orig_path), load(para_path)
        if orig is None or para is None:
            lines.append(f"{model_cell} & {cond_label} & \\multicolumn{{6}}{{c}}{{-- pending --}} \\\\")
            continue
        em_o, em_p = orig[em_key], para[em_key]
        f1_o, f1_p = orig["entity_f1"], para["entity_f1"]
        lines.append(f"{model_cell} & {cond_label} & {em_o:.3f} & {em_p:.3f} & "
                     f"{em_p - em_o:+.3f} & {f1_o:.3f} & {f1_p:.3f} & {f1_p - f1_o:+.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Paraphrase robustness, all 8 fine-tuned configurations (2 backbones "
                r"$\times$ 4 conditions): same 1502 questions (gold answers unchanged, "
                r"stratified-by-Section subsample of \emph{test\_heldout}), original vs.\ "
                r"LLM-paraphrased wording (entity names constrained to appear verbatim in the "
                r"rewrite). $\Delta$ = paraphrased $-$ original.}")
    lines.append(r"\label{tab:paraphrase-robustness}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: the prompt variants themselves, confronted with results. Every
# system prompt (DSL/SPARQL x NoCoT/CoT) is byte-identical except for its
# final instruction sentence -- that sentence, quoted verbatim, IS the
# variable this table studies. Text copied verbatim from the .replace() calls
# that build *_COT from the base template (not re-derived at build time,
# since importing fewshot_path_baseline.py/sparql_baseline.py here would pull
# in the full training/inference dependency stack for a plain table script):
#   baselines/fewshot_path_baseline.py, SYSTEM_TEMPLATE_COT definition
#   baselines/sparql_baseline.py, SYSTEM_TEMPLATE_COT definition
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = [
    ("DSL", "NoCoT", "dsl_nc", False,
     "Respond with ONLY the path, wrapped in \\texttt{<PATH>...</PATH>}, no explanation."),
    ("DSL", "CoT", "dsl_c", False,
     "First reason step by step about which relations to follow and why, THEN give the "
     "final answer wrapped in \\texttt{<PATH>...</PATH>} at the end of your response."),
    ("SPARQL", "NoCoT", "sparql_nc", True,
     "Respond with ONLY the SPARQL query, no explanation, no markdown code fences."),
    ("SPARQL", "CoT", "sparql_c", True,
     "First reason step by step about which relations to follow and why, THEN give the "
     "final SPARQL query at the end of your response."),
]


def build_table4() -> str:
    paths_by_key = {"dsl_nc": [], "dsl_c": [], "sparql_nc": [], "sparql_c": []}
    for _, _, dsl_nc, dsl_c, sparql_nc, sparql_c in ROWS:
        paths_by_key["dsl_nc"].append(dsl_nc)
        paths_by_key["dsl_c"].append(dsl_c)
        paths_by_key["sparql_nc"].append(sparql_nc)
        paths_by_key["sparql_c"].append(sparql_c)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l p{7.2cm} c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Formalism} & \textbf{Style} & \textbf{Final instruction sentence "
                r"(rest of the prompt -- schema, worked examples, shots -- is byte-identical)} "
                r"& \textbf{Mean EM} & $n$ \\")
    lines.append(r"\midrule")
    for formalism, style, key, is_sparql, sentence in PROMPT_VARIANTS:
        ems = [em_of(p, is_sparql) for p in paths_by_key[key]]
        ems = [e for e in ems if e is not None]
        mean_em = f"{sum(ems) / len(ems):.3f}" if ems else "--"
        lines.append(f"{formalism} & {style} & {sentence} & {mean_em} & {len(ems)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{The prompt variants themselves, confronted with results. Each row's "
                r"instruction sentence is the ONLY part of the system prompt that differs from "
                r"its NoCoT/CoT counterpart for the same formalism -- schema description, worked "
                r"examples, and few-shot examples are byte-identical (see "
                r"experiments/baselines/fewshot\_path\_baseline.py and baselines/sparql\_baseline.py, "
                r"\texttt{SYSTEM\_TEMPLATE}/\texttt{SYSTEM\_TEMPLATE\_COT}). \textbf{Mean EM} is "
                r"the unweighted average exact match across all $n$ models evaluated under that "
                r"exact prompt (Path EM for DSL, Query EM for SPARQL); it mixes models of very "
                r"different scale and training status (prompted vs.\ fine-tuned), so it should be "
                r"read as \emph{which instruction wording tends to help, on average, across "
                r"whatever was tested with it} rather than a controlled per-model comparison -- "
                r"see Table~\ref{tab:main-results} for that.}")
    lines.append(r"\label{tab:prompt-variants}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> None:
    print(r"% Requires in the paper's preamble: \usepackage{booktabs,multirow,graphicx}")
    print(build_table1())
    print()
    print(build_table2())
    print()
    print(build_table3())
    print()
    print(build_table4())


if __name__ == "__main__":
    main()
