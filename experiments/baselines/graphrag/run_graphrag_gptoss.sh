#!/bin/bash
# Graph-RAG on the 1,000-question smoke-test subset. gpt-oss-120b, CoT only.
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "=== $(date +%H:%M) START gptoss120b cot ==="
$PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py \
   --contexts experiments/baselines/graphrag/graphrag_contexts_s1000.csv \
   --model openai/gpt-oss-120b --tp 2 --style cot --max-new 800 --max-model-len 8192 \
   --gpu-mem-util 0.95 --out experiments/baselines/graphrag/graphrag_gptoss120b_cot.json
echo "=== $(date +%H:%M) DONE gptoss120b cot (exit $?) ==="
