#!/bin/bash
# Graph-RAG on the FULL gold-executable test split (6,772 questions).
# Pair A = GPU 0,1. Llama-3.3-70B, both reasoning styles (pipeline parallel:
# vLLM refuses to tensor-parallelise pre-quantized bitsandbytes checkpoints).
export CUDA_VISIBLE_DEVICES=0,1
PYTHON_BIN="${PYTHON_BIN:-python}"
CTX=experiments/baselines/graphrag/graphrag_contexts_full.csv
run () {  # tag model quant pp style
  echo "=== $(date +%F\ %H:%M) START $1 $5 ==="
  $PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py --contexts $CTX \
     --model "$2" ${3:+--quantization $3} --pp $4 --style $5 \
     --gpu-mem-util 0.90 --out experiments/baselines/graphrag/full_graphrag_$1_$5.json
  echo "=== $(date +%F\ %H:%M) DONE $1 $5 (exit $?) ==="
}
run llama70b unsloth/Llama-3.3-70B-Instruct-bnb-4bit bitsandbytes 2 nocot
run llama70b unsloth/Llama-3.3-70B-Instruct-bnb-4bit bitsandbytes 2 cot
echo "=== $(date +%F\ %H:%M) PAIR A COMPLETE ==="
