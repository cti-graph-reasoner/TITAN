# TITAN — End‑to‑End Pipeline (ATT&CK STIX → Graph → Datasets → Train/Val/Test → Training & Test)

TITAN builds a MITRE ATT&CK knowledge graph from STIX bundles, generates QA/navigation datasets (CoT/NoCoT), creates **train/validation/test** splits, and provides training/testing scripts to fine‑tune and evaluate path‑generation models.

<p align="center">
  <img src="images/titan.png" alt="TITAN framework" width="65%">
</p>

## Repository layout
```
TITAN/
├─ datasets/
│  ├─ CoT/
│  ├─ NoCoT/
│  └─ create_dataset_splits.py          # split by section into train/val/test
├─ utils/
│  ├─ build_graph.py                    # STIX → GraphML
│  ├─ build_dataset.py                  # GraphML + YAML templates → dataset JSON (+ per-section JSON)
│  ├─ paraphrase.py                     # optional LLM: target/objective improvement → target_variations.csv
│  └─ useful_cot.yaml                   # question templates with <PATH>...</PATH> + target
├─ graph_algorithm.py                   # graph navigation utilities
├─ train_titan.py                       # LoRA SFT training (Unsloth + TRL)
├─ test_titan.py                        # interactive tester (generate <PATH> and execute on graph)
├─ modify_target.py                     # apply target_variations.csv to YAML/JSON
└─ README.md
```

> Notes
> - `paraphrase.py` is optional and **does not** affect the pipeline unless you apply its output via `modify_target.py`.
> - If your image has a different filename, update the `<img src="images/...">` path accordingly.

---

## Requirements
- Python **3.9+**
- Local MITRE **ATT&CK STIX** JSON bundles (e.g., `../attack-stix-data/`)
- (Optional) GPU for LLM steps (`paraphrase.py`, training)

### Install
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip

pip install networkx pandas pyyaml tqdm scikit-learn
# For training & testing:
pip install torch transformers accelerate datasets trl unsloth
```

---

## 1) Build the graph from STIX
Script: `utils/build_graph.py` → outputs `stix_graph_correct.graphml` (repo root) and optional log.

```bash
python utils/build_graph.py   --base ../attack-stix-data   --out stix_graph_correct.graphml   --log-file mitre.txt
```
> If your version of `build_graph.py` does not accept CLI args, set the paths inside the file or rely on defaults.

---

## 2) Generate datasets (CoT or NoCoT)
Script: `utils/build_dataset.py`  
Inputs:
- `stix_graph_correct.graphml` (from step 1)
- `utils/useful_cot.yaml` (templates with `<PATH>...</PATH>` and `target`)

Typical outputs (CoT):
- `datasets/CoT/NAVIGATION_DATASET.json`  
- `datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json`  

Example (CoT):
```bash
python utils/build_dataset.py   --templates utils/useful_cot.yaml   --graph stix_graph_correct.graphml   --out datasets/CoT/NAVIGATION_DATASET.json   --out datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json
```

If you also want **NoCoT**, re-run with your NoCoT options and output paths in `datasets/NoCoT/` (if supported by your script).

### (Optional) JSON → CSV helper
Some steps prefer CSV. Convert with this one‑liner (no extra script file needed):
```bash
python - <<'PY'
import json, pandas as pd, os
inp="datasets/CoT/NAVIGATION_DATASET.json"; out="datasets/CoT/NAVIGATION_DATASET.csv"
j=json.load(open(inp,"r",encoding="utf-8")); df=pd.DataFrame(j)
if "question" in df.columns: df=df.rename(columns={"question":"Question"})
os.makedirs(os.path.dirname(out), exist_ok=True); df.to_csv(out, index=False, encoding="utf-8")
print("Wrote", out)
PY
```

---

## 3) (Optional) Improve targets with LLM and apply them
`utils/paraphrase.py` can produce **`target_variations.csv`** with refined Objectives/targets.  
This file is **not used automatically** by the pipeline.

To apply it, use **`modify_target.py`** (provided here). It supports both YAML templates and dataset JSON.

### Update YAML templates
```bash
python modify_target.py   --csv target_variations.csv   --in utils/useful_cot.yaml   --out utils/useful_cot.improved.yaml   --pick first
```

### Update a dataset JSON
```bash
python modify_target.py   --csv target_variations.csv   --in datasets/CoT/NAVIGATION_DATASET.json   --out datasets/CoT/NAVIGATION_DATASET.improved.json   --pick longest
```

Flags:
- `--pick {first,longest}`: choose which entry to keep when `Variations` has multiple items separated by `;`
- `--dry-run`: preview changes without writing
- `--no-backup`: disable automatic `.bak` backup if `--out` already exists

---

## 4) Create per‑section train/val/test splits
Script: `datasets/create_dataset_splits.py`  
Inputs:
- CSV with **`Question`** column (e.g., `datasets/CoT/NAVIGATION_DATASET.csv`)
- Per‑section JSON (e.g., `datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json`)

Outputs:
- `datasets/CoT/COMPLETE/train_dataset.csv`
- `datasets/CoT/COMPLETE/val_dataset.csv`
- `datasets/CoT/COMPLETE/test_dataset.csv`

Run:
```bash
python datasets/create_dataset_splits.py   --csv datasets/CoT/NAVIGATION_DATASET.csv   --json datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json   --out datasets/CoT/COMPLETE   --train 0.80 --val 0.05 --test 0.15   --seed 42
```

---

## 5) Train (LoRA SFT, Unsloth + TRL)
Script: `train_titan.py` (provided here). Default expects the **SMARTER** split directory:
```
SMARTER_COMPLETE_DATASET/
  ├─ train_dataset.csv
  ├─ val_dataset.csv
  └─ test_dataset.csv
```

Example:
```bash
python train_titan.py   --data SMARTER_COMPLETE_DATASET   --out MODELS/phi_smarter   --model unsloth/Phi-3.5-mini-instruct   --lr 3e-4 --train-bsz 8 --eval-bsz 8 --grad-accum 2   --epochs 8 --seq-len 2048 --seed 42
```

Key points:
- Saves **LoRA adapters** + tokenizer into `--out`
- If you hit OOM: reduce `--train-bsz` or increase `--grad-accum`

---

## 6) Interactive test
Script: `test_titan.py` (provided here). It loads the trained adapters, generates a `<PATH>...</PATH>` plan, parses entities, and executes the path on the graph via `graph_algorithm.py`.

```bash
python test_titan.py   --model MODELS/phi_smarter   --names NAMES.txt   --graph stix_graph_correct.graphml   --rels Relationship_Descriptions.txt
```

Type a query, for example:
```
what are the different kill chain phases between Carberp and Lucifer?
```

---

## Embed the TITAN image in this README
If your image is at `images/titan.png`, this is already set at the top of the README.  
Alternative Markdown syntax:
```markdown
![TITAN framework](images/titan.png)
```
Centered and resized (HTML):
```html
<p align="center">
  <img src="images/titan.png" alt="TITAN framework" width="600">
</p>
```

---

## Troubleshooting
- **Missing `Question` column**: rename `question` to `Question` before splitting (see JSON→CSV helper).
- **Unmapped questions**: they may be dropped or set to `Unknown`, depending on your splitter logic.
- **Tiny sections**: the splitter handles small groups gracefully (1 row → train; 2 rows → 50/50 train/test).
- **LLM GPU/CPU**: if you don’t have a GPU, training will be slow. For `paraphrase.py`, set `device_map="cpu"` if needed.
- **Paths**: if scripts don’t accept CLI args in your version, configure paths inside the files.

---

## Quick pipeline (CoT)
```bash
# 1) Build graph
python utils/build_graph.py --base ../attack-stix-data --out stix_graph_correct.graphml

# 2) Build dataset (json + per-section)
python utils/build_dataset.py   --templates utils/useful_cot.yaml   --graph stix_graph_correct.graphml   --out datasets/CoT/NAVIGATION_DATASET.json   --out datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json

# 3) (optional) apply LLM targets
python modify_target.py   --csv target_variations.csv   --in datasets/CoT/NAVIGATION_DATASET.json   --out datasets/CoT/NAVIGATION_DATASET.improved.json

# 4) Convert to CSV (if needed)
python - <<'PY'
import json, pandas as pd, os
inp="datasets/CoT/NAVIGATION_DATASET.json"; out="datasets/CoT/NAVIGATION_DATASET.csv"
j=json.load(open(inp,"r",encoding="utf-8")); df=pd.DataFrame(j)
if "question" in df.columns: df=df.rename(columns={"question":"Question"})
os.makedirs(os.path.dirname(out), exist_ok=True); df.to_csv(out, index=False, encoding="utf-8")
print("Wrote", out)
PY

# 5) Split
python datasets/create_dataset_splits.py   --csv datasets/CoT/NAVIGATION_DATASET.csv   --json datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json   --out datasets/CoT/COMPLETE --train 0.80 --val 0.05 --test 0.15

# 6) Train
python train_titan.py --data SMARTER_COMPLETE_DATASET --out MODELS/phi_smarter

# 7) Test
python test_titan.py --model MODELS/phi_smarter --names NAMES.txt --graph stix_graph_correct.graphml --rels Relationship_Descriptions.txt
```
