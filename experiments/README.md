# Reproducing the TITAN experiments

This folder contains everything needed to regenerate the datasets, retrain the
path-planner models, run the baselines, and reproduce the reported results.

All commands below assume the repository has been installed (`pip install -e .`
from the repository root — see the top-level [README](../README.md)) and are
run **from the repository root**, not from inside `experiments/`.

```
experiments/
├─ datasets/          # dataset generation, splitting, and (most of) the data itself
├─ baselines/          # prompted / fine-tuned / SPARQL / RAG baselines + results
├─ utils/               # graph construction and dataset-building tooling
├─ train.py, train_v2.py, train_qwen.py   # LoRA SFT training scripts
├─ annotate_executability.py, verify_no_leakage.py   # dataset QA / leakage audits
├─ build_latex_table.py   # builds the paper's results tables from eval reports
├─ modify_target.py       # applies paraphrased targets to templates/datasets
├─ rescore_all_sparql.sh  # re-scores every SPARQL baseline prediction file
├─ titan_findings.csv, rel_target_types.json
└─ results/
```

## Data notes

A handful of raw dataset files exceed 50MB and are not tracked in git (see
`.gitignore`); everything else — including all baseline prediction/score
files — is included. The excluded files can be regenerated locally with the
commands below (Step 3), or reconstructed from the smaller `TEMPLATE_DISJOINT`
splits that are included.

---

## 1. Build the TITAN graph

```bash
python experiments/utils/build_graph.py \
  --base ../attack-stix-data \
  --out stix_graph_correct.graphml
```

This requires a local copy of the MITRE ATT&CK STIX JSON bundles. It also
writes `rel_target_types.json` (per-relation dominant target types) via
`experiments/utils/graph_to_rdf.py` if an RDF export is needed:

```bash
python experiments/utils/graph_to_rdf.py \
  --graph stix_graph_correct.graphml \
  --out titan_graph.nt \
  --types experiments/rel_target_types.json
```

## 2. Generate CoT / NoCoT datasets

```bash
python experiments/utils/build_dataset.py \
  --templates experiments/utils/useful_cot.yaml \
  --graph stix_graph_correct.graphml \
  --out experiments/datasets/CoT/NAVIGATION_DATASET.json \
  --out-sections experiments/datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json
```

Re-run against `experiments/datasets/NoCoT/` for the non-reasoning variant.

## 3. Create train/val/test splits

Standard split:

```bash
python experiments/datasets/create_dataset_splits.py \
  --csv experiments/datasets/CoT/NAVIGATION_DATASET.csv \
  --json experiments/datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json \
  --out experiments/datasets/CoT/COMPLETE \
  --train 0.80 --val 0.05 --test 0.15 --seed 42
```

Template-disjoint split (used for the held-out generalization results):

```bash
python experiments/datasets/create_template_disjoint_splits.py \
  --inputs experiments/datasets/CoT/train_dataset.csv \
           experiments/datasets/CoT/val_dataset.csv \
           experiments/datasets/CoT/test_dataset.csv \
  --out experiments/datasets/TEMPLATE_DISJOINT
```

## 4. Train the path-planner (LoRA SFT)

```bash
python experiments/train_v2.py \
  --data experiments/datasets/TEMPLATE_DISJOINT/CoT \
  --out MODELS/phi_titan \
  --model unsloth/Phi-3.5-mini-instruct \
  --lr 3e-4 --train-bsz 8 --eval-bsz 8 --grad-accum 2 \
  --epochs 8 --seq-len 2048 --seed 42
```

`experiments/train_qwen.py` runs the same recipe on a Qwen2.5 backbone; the
dataset directory must contain `train_dataset.csv` / `val_dataset.csv` /
`test_dataset.csv` with at least `Question` and `Path` columns.

## 5. Run the baselines

Each script under `experiments/baselines/` implements one baseline
(few-shot path planning, SPARQL generation, RAG, prompted DSL, …) and writes
predictions plus a scored report:

```bash
python experiments/baselines/score_sparql_predictions.py \
  --pred experiments/baselines/<predictions>.json \
  --data experiments/datasets/TEMPLATE_DISJOINT/CoT/test_heldout.annotated.csv \
  --workers 24 \
  --out experiments/baselines/<predictions>_eval
```

`experiments/rescore_all_sparql.sh` re-runs this scoring step for every
tracked SPARQL prediction file in one pass.

The Graph-RAG baseline lives in its own subfolder,
[`experiments/baselines/graphrag/`](baselines/graphrag/README.md), since it
splits retrieval and generation into separate cluster jobs.

## 6. Validate the splits and build the results tables

```bash
python experiments/verify_no_leakage.py --splits experiments/datasets/TEMPLATE_DISJOINT
python experiments/annotate_executability.py --splits experiments/datasets/TEMPLATE_DISJOINT/CoT
python experiments/build_latex_table.py > experiments/results/results_table.tex
```

## Troubleshooting

- **Missing columns** — rename `question` → `Question` before splitting.
- **GPU unavailable** — training runs on CPU but will be slow.
- **`ModuleNotFoundError: titan`** — run `pip install -e .` from the repository root first.
