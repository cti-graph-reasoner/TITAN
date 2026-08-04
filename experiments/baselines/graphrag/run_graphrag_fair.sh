#!/bin/bash
# Fair-comparison Graph-RAG runner (800-row matched subset, NoCoT + few-shot
# for every model). Invoke once per GPU pair, e.g.:
#   run_graphrag_fair.sh 0,1 "qwen72b|unsloth/Qwen2.5-72B-Instruct-bnb-4bit|bitsandbytes|pp|2"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=$1; shift
for spec in "$@"; do
  IFS='|' read -r tag model quant par pv <<< "$spec"
  echo "=== $(date +%F\ %H:%M) START $tag ==="
  Q=""; [ -n "$quant" ] && Q="--quantization $quant"
  $PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py \
     --contexts experiments/baselines/graphrag/graphrag_contexts_fair800.csv \
     --model "$model" $Q --$par $pv \
     --style nocot --fewshot --max-new 1024 \
     --max-model-len 32768 --gpu-mem-util 0.92 \
     --out experiments/baselines/graphrag/fair_graphrag_${tag}_nocot.json
  echo "=== $(date +%F\ %H:%M) DONE $tag (exit $?) ==="
done
echo "=== $(date +%F\ %H:%M) PAIR $CUDA_VISIBLE_DEVICES COMPLETE ==="
