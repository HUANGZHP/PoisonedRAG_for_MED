#!/usr/bin/env bash

# 保留已启动的 PubMedQA 检索，同时并行启动 MedQA 检索；两组检索各自完成后启动评测。
set -u -o pipefail

PUBMED_PID="$1"
ROOT=/home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
LOG_DIR="$ROOT/logs/contriever_v4"

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
ORCHESTRATOR="$LOG_DIR/parallel_experiment_orchestrator.log"
printf '%s parallel v4 pipeline started; existing PubMedQA PID=%s\n' "$(date -Is)" "$PUBMED_PID" > "$ORCHESTRATOR"

"$PYTHON" -u evaluate_beir.py \
    --model_code contriever_v4 --dataset pubmed --top_k 100 --gpu_id 4 \
    --per_gpu_batch_size 64 --max_length 128 \
    --queries-json results/adv_targeted_results/mirage_medqa_all.queries.json \
    --result_output results/beir_results/mirage_medqa_all-contriever_v4.json \
    > "$LOG_DIR/retrieval_medqa.log" 2>&1 &
MEDQA_PID=$!
printf '%s MedQA retrieval started on GPU 4; PID=%s\n' "$(date -Is)" "$MEDQA_PID" >> "$ORCHESTRATOR"

launch_suite() {
    local label="$1"
    local pid="$2"
    local m="$3"
    local adv_json="$4"
    local ids="$5"
    local retrieval="$6"
    local gpu="$7"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
    if [ ! -s "$retrieval" ]; then
        printf '%s %s retrieval did not produce output\n' "$(date -Is)" "$label" >> "$ORCHESTRATOR"
        return 1
    fi
    printf '%s %s retrieval complete; suite started on GPU %s\n' "$(date -Is)" "$label" "$gpu" >> "$ORCHESTRATOR"
    nohup bash scripts/run_contriever_v4_no_defense_suite.sh \
        "$label" "$m" "$adv_json" "$ids" "$retrieval" "$gpu" \
        > "$LOG_DIR/${label}_no_defense_pipeline.log" 2>&1 < /dev/null &
}

launch_suite pubmedqa_pubmed "$PUBMED_PID" 500 \
    results/adv_targeted_results/mirage_pubmedqa_all.json \
    results/adv_targeted_results/mirage_pubmedqa_all.ids \
    results/beir_results/mirage_pubmedqa_all-contriever_v4.json 5 &
PUB_SUITE_WAITER=$!

launch_suite medqa_pubmed "$MEDQA_PID" 1273 \
    results/adv_targeted_results/mirage_medqa_all.json \
    results/adv_targeted_results/mirage_medqa_all.ids \
    results/beir_results/mirage_medqa_all-contriever_v4.json 6 &
MED_SUITE_WAITER=$!

wait "$PUB_SUITE_WAITER" "$MED_SUITE_WAITER"
printf '%s both v4 experiment suites launched\n' "$(date -Is)" >> "$ORCHESTRATOR"
