#!/usr/bin/env bash

# 生成 Contriever v4 的两份 PubMed 检索结果，再并行启动两组无防御评测。
set -u -o pipefail

ROOT=/home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
LOG_DIR="$ROOT/logs/contriever_v4"

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
ORCHESTRATOR="$LOG_DIR/experiment_orchestrator.log"
printf '%s v4 retrieval and experiment pipeline started\n' "$(date -Is)" > "$ORCHESTRATOR"

run_retrieval() {
    local label="$1"
    local queries="$2"
    local output="$3"
    printf '%s retrieval started: %s\n' "$(date -Is)" "$label" >> "$ORCHESTRATOR"
    "$PYTHON" -u evaluate_beir.py \
        --model_code contriever_v4 --dataset pubmed --top_k 100 --gpu_id 3 \
        --per_gpu_batch_size 64 --max_length 128 --queries-json "$queries" \
        --result_output "$output" > "$LOG_DIR/retrieval_${label}.log" 2>&1
}

run_retrieval pubmedqa \
    results/adv_targeted_results/mirage_pubmedqa_all.queries.json \
    results/beir_results/mirage_pubmedqa_all-contriever_v4.json || exit $?
printf '%s PubMedQA retrieval complete; suite started on GPU 5\n' "$(date -Is)" >> "$ORCHESTRATOR"
nohup bash scripts/run_contriever_v4_no_defense_suite.sh \
    pubmedqa_pubmed 500 \
    results/adv_targeted_results/mirage_pubmedqa_all.json \
    results/adv_targeted_results/mirage_pubmedqa_all.ids \
    results/beir_results/mirage_pubmedqa_all-contriever_v4.json 5 \
    > "$LOG_DIR/pubmedqa_pubmed_no_defense_pipeline.log" 2>&1 < /dev/null &

run_retrieval medqa \
    results/adv_targeted_results/mirage_medqa_all.queries.json \
    results/beir_results/mirage_medqa_all-contriever_v4.json || exit $?
printf '%s MedQA retrieval complete; suite started on GPU 6\n' "$(date -Is)" >> "$ORCHESTRATOR"
nohup bash scripts/run_contriever_v4_no_defense_suite.sh \
    medqa_pubmed 1273 \
    results/adv_targeted_results/mirage_medqa_all.json \
    results/adv_targeted_results/mirage_medqa_all.ids \
    results/beir_results/mirage_medqa_all-contriever_v4.json 6 \
    > "$LOG_DIR/medqa_pubmed_no_defense_pipeline.log" 2>&1 < /dev/null &

printf '%s both v4 suites launched\n' "$(date -Is)" >> "$ORCHESTRATOR"
