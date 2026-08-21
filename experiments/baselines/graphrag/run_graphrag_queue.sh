#!/bin/bash
# Graph-RAG on the 1,000-question smoke-test subset. Single GPU, 7B model.
PYTHON_BIN="${PYTHON_BIN:-python}"
CTX=experiments/baselines/graphrag/graphrag_contexts_s1000.csv
run () {  # tag model quant tp style
  echo "=== $(date +%H:%M) START $1 $5 ==="
  $PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py --contexts $CTX \
     --model "$2" ${3:+--quantization $3} --tp ${4:-1} --style $5 \
     --out experiments/baselines/graphrag/graphrag_$1_$5.json
  echo "=== $(date +%H:%M) DONE $1 $5 ==="
}
run qwen7b unsloth/Qwen2.5-7B-Instruct-bnb-4bit bitsandbytes 1 nocot
run qwen7b unsloth/Qwen2.5-7B-Instruct-bnb-4bit bitsandbytes 1 cot
