#!/usr/bin/env bash

set -u -o pipefail

TRAIN_PID="$1"
ROOT=/home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
LOG_DIR="$ROOT/logs/contriever_v5"
MODEL_DIR="$ROOT/checkpoint/contriever_v5/best_model"

mkdir -p "$LOG_DIR"
printf '%s waiting for v5 training process %s\n' "$(date -Is)" "$TRAIN_PID" > "$LOG_DIR/retrieval_medqa_orchestrator.log"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 30
done

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
    printf '%s v5 training did not produce a best model; retrieval was not started\n' "$(date -Is)" >> "$LOG_DIR/retrieval_medqa_orchestrator.log"
    exit 1
fi

cd "$ROOT"
printf '%s starting MedQA query retrieval with contriever_v5\n' "$(date -Is)" >> "$LOG_DIR/retrieval_medqa_orchestrator.log"
exec "$PYTHON" -u evaluate_beir.py \
    --model_code contriever_v5 --dataset pubmed --top_k 100 --gpu_id 4 \
    --per_gpu_batch_size 64 --max_length 128 \
    --queries-json results/adv_targeted_results/mirage_medqa_all.queries.json \
    --result_output results/beir_results/mirage_medqa_all-contriever_v5.json \
    > "$LOG_DIR/retrieval_medqa.log" 2>&1
