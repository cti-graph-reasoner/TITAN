#!/bin/bash
# Run from the repository root.
set -x
source .venv/bin/activate
for f in llama33_70b_sparql_heldout_full qwen25_72b_sparql_heldout_full qwen25_7b_prompted_sparql_heldout_full qwen25_7b_titan_sparql_heldout phi_titan_sparql_heldout phi_titan_sparql_cot_heldout; do
  echo "=== $f ==="
  python experiments/baselines/score_sparql_predictions.py \
    --pred experiments/baselines/${f}.json \
    --data experiments/datasets/TEMPLATE_DISJOINT/CoT/test_heldout.annotated.csv \
    --workers 24 \
    --out experiments/baselines/${f}_eval
done
echo ALL_RESCORE_DONE
