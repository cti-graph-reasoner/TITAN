#!/bin/bash
# Sequential queue, never more than 3 GPUs at once (plus whatever else is in use).
# Each big model needs 3 cards: at 18k-token contexts the KV cache does not fit on 2.
PYTHON_BIN="${PYTHON_BIN:-python}"

wait_free () {   # wait until the given GPUs are all idle
  while true; do
    busy=0
    for g in $(echo $1 | tr ',' ' '); do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      [ "$m" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 60
  done
}

run () {  # tag model quant parflag parval gpus len
  wait_free "$6"
  export CUDA_VISIBLE_DEVICES=$6
  echo "=== $(date +%F\ %H:%M) START $1 on GPU $6 ==="
  Q=""; [ -n "$3" ] && Q="--quantization $3"
  $PYTHON_BIN experiments/baselines/graphrag/graphrag_generate.py \
     --contexts experiments/baselines/graphrag/graphrag_contexts_fair800.csv \
     --model "$2" $Q --$4 $5 \
     --style nocot --fewshot --max-new 1024 \
     --max-model-len $7 --gpu-mem-util 0.92 \
     --out experiments/baselines/graphrag/fair_graphrag_$1_nocot.json
  echo "=== $(date +%F\ %H:%M) DONE $1 (exit $?) ==="
}

run qwen72b    unsloth/Qwen2.5-72B-Instruct-bnb-4bit bitsandbytes pp 3 0,1,6 28672
run gptoss120b openai/gpt-oss-120b                   ""           tp 2 0,1   28672
echo "=== $(date +%F\ %H:%M) QUEUE COMPLETE ==="
