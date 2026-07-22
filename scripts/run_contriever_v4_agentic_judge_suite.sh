#!/usr/bin/env bash

set -u -o pipefail

LABEL="$1"
M="$2"
ADV_JSON="$3"
TARGET_IDS="$4"
RETRIEVAL="$5"
GPU="$6"
ROOT=/home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
LOG_DIR="$ROOT/logs/contriever_v4_agentic_judge"

cd "$ROOT"
mkdir -p "$LOG_DIR"
ORCHESTRATOR="$LOG_DIR/${LABEL}_orchestrator.log"

run() {
    local suffix="$1"
    local attack="$2"
    local name="${LABEL}_contriever_v4_gpt41mini_agentic_judge_${suffix}"
    printf '%s starting %s\n' "$(date -Is)" "$name" >> "$ORCHESTRATOR"
    "$PYTHON" -u agentic_main.py \
        --eval_model_code contriever_v4 --eval_dataset pubmed \
        --model_name gpt4.1mini --judge_model_name gpt4.1mini \
        --top_k 5 --attack_method "$attack" --agentic_rag --rag_max_turns 3 \
        --adv_source json --adv_json_path "$ADV_JSON" --target_ids_path "$TARGET_IDS" \
        --retrieval_results_path "$RETRIEVAL" --adv_per_query 5 --M "$M" \
        --repeat_times 1 --gpu_id "$GPU" --name "$name" \
        > "$LOG_DIR/${name}.log" 2>&1
    local status=$?
    printf '%s finished %s status=%s\n' "$(date -Is)" "$name" "$status" >> "$ORCHESTRATOR"
    return "$status"
}

run noattack None || exit $?
run blackbox LM_targeted || exit $?
run hotflip hotflip || exit $?
printf '%s all experiments completed\n' "$(date -Is)" >> "$ORCHESTRATOR"
