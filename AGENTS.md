# AGENTS.md

Guidance for automated coding agents working in this repository.

## Project

TITAN builds a typed, bidirectional knowledge graph from MITRE ATT&CK STIX data,
generates question → `<PATH>…</PATH>` datasets from it, fine-tunes an LLM
(LoRA via Unsloth + TRL) to produce those paths, and executes the generated
paths on the graph to return grounded entities.

## Layout

| Path | Role |
|---|---|
| `utils/build_graph.py` | STIX JSON bundles → GraphML graph |
| `utils/build_dataset.py` | Graph + YAML templates → Q/A JSON (flat and per section) |
| `utils/useful_cot.yaml` | Question templates with `path` and `target` |
| `utils/paraphrase.py` | Optional: target rewording with Llama-3.1-8B-Instruct → `target_variations.csv` (no CLI; paths set in `main`) |
| `modify_target.py` | Applies `target_variations.csv` to YAML templates or JSON datasets |
| `datasets/create_dataset_splits.py` | Per-section train/val/test split |
| `datasets/CoT/`, `datasets/NoCoT/` | Generated splits and evaluation results |
| `train.py` | LoRA SFT training (default base model `unsloth/Phi-3.5-mini-instruct`) |
| `test.py` | Interactive tester: generate a path, execute it on the graph |
| `graph_algorithm.py` | Deterministic path execution over the graph |
| `stix_graph_correct.graphml` | Prebuilt graph |

## Pipeline commands

```bash
python utils/build_graph.py --base ../attack-stix-data --out stix_graph_correct.graphml --log-file mitre.txt
python utils/build_dataset.py --templates utils/useful_cot.yaml --graph stix_graph_correct.graphml \
  --out NAVIGATION_DATASET.json --out-sections NAVIGATION_QUESTION_PER_SECTION.json
python datasets/create_dataset_splits.py --csv <dataset.csv> --json <per_section.json> --out <split_dir> \
  --train 0.80 --val 0.05 --test 0.15 --seed 42
python train.py --data <split_dir> --out MODELS/<name>
python test.py --model MODELS/<name> --names NAMES.txt --graph stix_graph_correct.graphml --rels Relationship_Descriptions.txt
```

`NAMES.txt`, `Relationship_Descriptions.txt` and the STIX data are not in the repository.
Training and `paraphrase.py` need a GPU; graph building, dataset generation and splitting run on CPU.

## Conventions

- Python 3.9+, `from __future__ import annotations`, type hints, module docstring at the top of each script.
- Section separators use `# ----` comment blocks; keep that style.
- Comments are in English and describe what the code does. Do not add changelog-style notes
  ("fixed", "original behavior"), reader-directed advice ("if you want…"), or commented-out code.
- Path format: `<PATH>step1<SEP>step2</PATH>`. Steps are edge labels (e.g. `uses_attack_pattern`),
  `is_<type>_type`, `filter <kw>`, `exec_common <TYPE>`, `exec_difference <TYPE>`.
  Changes to this format must be mirrored in `utils/build_dataset.py`, `graph_algorithm.py` and the YAML templates.
- The chat template mapping (`"human"` / `"gpt"` roles, `from` / `value` keys) and `SYSTEM_PROMPT`
  must stay identical between `train.py` and `test.py`.
- Every relation added to the graph must have its inverse (e.g. `uses_attack_pattern` ↔ `used_by_intrusion_set`).
- Keep seeds fixed (default 42) for splits and training.

## Checks

There is no test suite. After editing, at minimum run:

```bash
python -m py_compile graph_algorithm.py modify_target.py train.py test.py utils/*.py datasets/*.py
```

Do not modify files under `datasets/` or `stix_graph_correct.graphml` by hand; regenerate them with the scripts.
