# TITAN — End-to-End Pipeline (ATT&CK STIX → train/val/test)

This repository builds a MITRE ATT&CK graph from STIX bundles, generates datasets (CoT / NoCoT), and creates **train / validation / test** splits.

## Repository layout

```
TITAN/
├─ datasets/
│  ├─ CoT/
│  ├─ NoCoT/
│  └─ create_dataset_splits.py
├─ utils/
│  ├─ build_graph.py
│  ├─ build_dataset.py
│  ├─ paraphrase.py          # optional (LLM)
│  └─ useful_cot.yaml
├─ graph_algorithm.py
└─ README.md
```

## Requirements
- Python 3.9+
- MITRE **ATT&CK STIX** JSON bundles available locally (e.g., `../attack-stix-data/`)
- (Optional) GPU if using `utils/paraphrase.py`

## Environment & dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip

pip install networkx pandas pyyaml tqdm scikit-learn
# Optional LLM step:
# pip install torch transformers accelerate
```

---

## 1) Build the graph from STIX

Script: `utils/build_graph.py`  
Input: **STIX JSON** directory (e.g., `../attack-stix-data`)  
Output: `stix_graph_correct.graphml` (in the repo root) + optional log

```bash
python utils/build_graph.py   --base ../attack-stix-data   --out stix_graph_correct.graphml   --log-file mitre.txt
```

> If the script doesn't accept CLI args in your version, set paths inside the file or rely on its defaults.

---

## 2) Generate datasets (CoT / NoCoT)

Script: `utils/build_dataset.py`  
Inputs:
- `stix_graph_correct.graphml` (from step 1)
- `utils/useful_cot.yaml` (question templates with `<PATH>...</PATH>` and `target`)

Typical outputs:
- `datasets/CoT/NAVIGATION_DATASET.json`  
- `datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json`  
- (and similarly under `datasets/NoCoT/` if your script supports a NoCoT mode)

Example (CoT):
```bash
python utils/build_dataset.py   --templates utils/useful_cot.yaml   --graph stix_graph_correct.graphml   --out datasets/CoT/NAVIGATION_DATASET.json   --out-sections datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json
```

If you also need a **CSV** (the splitter prefers CSV), convert JSON → CSV:
```python
# save as: scripts/json_to_csv.py
import json, pandas as pd, os
inp = "datasets/CoT/NAVIGATION_DATASET.json"
out = "datasets/CoT/NAVIGATION_DATASET.csv"
j = json.load(open(inp, "r", encoding="utf-8"))
df = pd.DataFrame(j)
if "question" in df.columns:
    df = df.rename(columns={"question": "Question"})
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8")
print("Wrote", out)
```

Run it:
```bash
python scripts/json_to_csv.py
```

> For **NoCoT**, use the same arguments but write outputs under `datasets/NoCoT/` (if supported by your script).

---

## 3) (Optional) Paraphrase / improve targets with an LLM

Script: `utils/paraphrase.py`  
Goal: generate cleaner **Objectives/Targets** and/or question paraphrases while preserving critical tokens (e.g., bracketed spans `[ ... ]`, malware terms).  
Typical output: `target_variations.csv`

```bash
python utils/paraphrase.py
# by default reads utils/useful_cot.yaml and writes target_variations.csv
```

> Skip this step if you don't want to use an LLM (or if you lack a GPU).

---

## 4) Per-section train/val/test split

Script: `datasets/create_dataset_splits.py`  
Inputs:
- CSV with a **`Question`** column (e.g., `datasets/CoT/NAVIGATION_DATASET.csv`)
- JSON **grouped by section** (e.g., `datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json`)

Outputs:
- `datasets/CoT/COMPLETE/train_dataset.csv`
- `datasets/CoT/COMPLETE/val_dataset.csv`
- `datasets/CoT/COMPLETE/test_dataset.csv`

Example (CoT):
```bash
python datasets/create_dataset_splits.py   --csv datasets/CoT/NAVIGATION_DATASET.csv   --json datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json   --out datasets/CoT/COMPLETE   --train 0.80 --val 0.05 --test 0.15   --seed 42
```

Example (NoCoT):
```bash
python datasets/create_dataset_splits.py   --csv datasets/NoCoT/NAVIGATION_DATASET.csv   --json datasets/NoCoT/NAVIGATION_QUESTION_PER_SECTION.json   --out datasets/NoCoT/COMPLETE   --train 0.80 --val 0.05 --test 0.15
```

The script prints the **section distribution** for each split and robustly handles tiny sections.

---

## TL;DR (quick commands)

```bash
# 1) Graph from STIX
python utils/build_graph.py --base ../attack-stix-data --out stix_graph_correct.graphml

# 2) CoT dataset (json + per-section)
python utils/build_dataset.py   --templates utils/useful_cot.yaml   --graph stix_graph_correct.graphml   --out datasets/CoT/NAVIGATION_DATASET.json   --out datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json

# 2b) (optional) JSON → CSV
python - <<'PY'
import json, pandas as pd, os
inp="datasets/CoT/NAVIGATION_DATASET.json"; out="datasets/CoT/NAVIGATION_DATASET.csv"
j=json.load(open(inp,"r",encoding="utf-8")); df=pd.DataFrame(j)
if "question" in df.columns: df=df.rename(columns={"question":"Question"})
os.makedirs(os.path.dirname(out),exist_ok=True); df.to_csv(out,index=False,encoding="utf-8")
print("Wrote", out)
PY

# 3) (optional) LLM for targets/paraphrases
python utils/paraphrase.py

# 4) Split train/val/test
python datasets/create_dataset_splits.py   --csv datasets/CoT/NAVIGATION_DATASET.csv   --json datasets/CoT/NAVIGATION_QUESTION_PER_SECTION.json   --out datasets/CoT/COMPLETE --train 0.80 --val 0.05 --test 0.15
```

---

## Troubleshooting
- **Missing column**: the splitter requires `Question` in the CSV. If you only have `question`, rename it (see step 2b).
- **Unmapped questions**: rows not present in the per-section JSON may be dropped or labeled `Unknown` (depending on your script options).
- **Tiny sections**: the splitter falls back safely (1 row → train; 2 rows → 50/50 train/test).
- **Performance**: dataset generation can be heavy; use `tqdm` progress bars to monitor.
- **LLM**: if no GPU, in `utils/paraphrase.py` set `device_map="cpu"` and reduce token counts.
