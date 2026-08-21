#!/bin/bash
# Graph-RAG on the FULL gold-executable test split (6,772 questions).
# Pair B = GPU 2,6. Shortest cells first so a pipeline error shows up early.
export CUDA_VISIBLE_DEVICES=2,6
PYTHON_BIN="${PYTHON_BIN:-python}"
CTX=experiments/baselines/graphrag/graphrag_contexts_full.csv
run () {  # tag model quant par_flag par_size style extra...
  local tag=$1 model=$2 quant=$3 pflag=$4 psize=$5 style=$6; shift 6
  echo "=== $(date +%F\ %H:%M) START $tag $style ==="
  $PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py --contexts $CTX \
     --model "$model" ${quant:+--quantization $quant} --$pflag $psize --style $style \
     "$@" --out experiments/baselines/graphrag/full_graphrag_${tag}_${style}.json
  echo "=== $(date +%F\ %H:%M) DONE $tag $style (exit $?) ==="
}
run qwen7b     unsloth/Qwen2.5-7B-Instruct-bnb-4bit  bitsandbytes pp 1 nocot --gpu-mem-util 0.90
run qwen7b     unsloth/Qwen2.5-7B-Instruct-bnb-4bit  bitsandbytes pp 1 cot   --gpu-mem-util 0.90
run gptoss120b openai/gpt-oss-120b                   ""           tp 2 cot   --max-new 800 --max-model-len 8192 --gpu-mem-util 0.95
run qwen72b    unsloth/Qwen2.5-72B-Instruct-bnb-4bit bitsandbytes pp 2 nocot --gpu-mem-util 0.90
run qwen72b    unsloth/Qwen2.5-72B-Instruct-bnb-4bit bitsandbytes pp 2 cot   --gpu-mem-util 0.90
echo "=== $(date +%F\ %H:%M) PAIR B COMPLETE ==="
