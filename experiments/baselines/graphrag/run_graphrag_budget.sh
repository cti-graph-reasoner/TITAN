#!/bin/bash
# Budget-sensitivity check for the Graph-RAG baseline.
# Same 1,500 gold-executable rows as the k=10 run, retrieved with a 3x larger
# passage budget, so the two scores are paired and differ only in how much of
# the graph the reader is shown.
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0,1

echo "=== $(date +%F\ %H:%M) START retrieval k=30 ==="
$PYTHON_BIN experiments/baselines/graphrag/graphrag_retrieve.py \
   --test experiments/baselines/graphrag/graphrag_budget_subset.csv \
   --top-k 30 --max-context-chars 45000 \
   --out experiments/baselines/graphrag/graphrag_contexts_k30.csv
echo "=== $(date +%F\ %H:%M) DONE retrieval (exit $?) ==="

echo "=== $(date +%F\ %H:%M) START qwen72b nocot k=30 ==="
$PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py \
   --contexts experiments/baselines/graphrag/graphrag_contexts_k30.csv \
   --model unsloth/Qwen2.5-72B-Instruct-bnb-4bit --quantization bitsandbytes \
   --pp 2 --style nocot --max-model-len 16384 --gpu-mem-util 0.92 \
   --out experiments/baselines/graphrag/k30_graphrag_qwen72b_nocot.json
echo "=== $(date +%F\ %H:%M) DONE qwen72b nocot k=30 (exit $?) ==="
echo "=== BUDGET CHECK COMPLETE ==="
