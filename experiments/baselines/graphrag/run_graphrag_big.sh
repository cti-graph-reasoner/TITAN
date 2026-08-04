#!/bin/bash
# Graph-RAG on the 1,000-question smoke-test subset. 70B-class models,
# pipeline-parallel across 2 GPUs.
PYTHON_BIN="${PYTHON_BIN:-python}"
CTX=experiments/baselines/graphrag/graphrag_contexts_s1000.csv
run () {  # tag model quant pp style
  echo "=== $(date +%H:%M) START $1 $5 ==="
  $PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py --contexts $CTX \
     --model "$2" ${3:+--quantization $3} --pp $4 --style $5 \
     --gpu-mem-util 0.90 --out experiments/baselines/graphrag/graphrag_$1_$5.json
  echo "=== $(date +%H:%M) DONE $1 $5 (exit $?) ==="
}
run qwen72b unsloth/Qwen2.5-72B-Instruct-bnb-4bit bitsandbytes 2 nocot
run qwen72b unsloth/Qwen2.5-72B-Instruct-bnb-4bit bitsandbytes 2 cot
run llama70b unsloth/Llama-3.3-70B-Instruct-bnb-4bit bitsandbytes 2 nocot
run llama70b unsloth/Llama-3.3-70B-Instruct-bnb-4bit bitsandbytes 2 cot
