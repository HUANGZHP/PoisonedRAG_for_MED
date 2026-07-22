#!/usr/bin/env bash

# 在后台顺序生成 MedQA 的两类负例，并在失败时从头重试一次。
set -u -o pipefail

ROOT=/home/huangzhp53/PoisonedRAG
INPUT=/home/huangzhp53/PubMedQA-and-MedQA/datasets/MedQA-USMLE/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl
BLACKBOX="$ROOT/processed/medqa_blackbox.jsonl"
HOTFLIP="$ROOT/processed/medqa_hotflip.jsonl"
LOG_DIR="$ROOT/logs/contriever_generation"
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
printf '%s MedQA USMLE generation started\n' "$(date -Is)" > "$LOG_DIR/medqa_orchestrator.log"

for attempt in 1 2; do
    rm -f "$BLACKBOX" "$HOTFLIP"
    printf '%s attempt %s: blackbox started\n' "$(date -Is)" "$attempt" >> "$LOG_DIR/medqa_orchestrator.log"
    "$PYTHON" -u scripts/build_blackbox.py \
        --input "$INPUT" --output "$BLACKBOX" --dataset medqa --verbose \
        > "$LOG_DIR/medqa_blackbox.log" 2>&1
    blackbox_status=$?
    if [ "$blackbox_status" -ne 0 ]; then
        printf '%s attempt %s: blackbox failed status=%s\n' "$(date -Is)" "$attempt" "$blackbox_status" >> "$LOG_DIR/medqa_orchestrator.log"
        continue
    fi

    printf '%s attempt %s: hotflip started on GPU 1\n' "$(date -Is)" "$attempt" >> "$LOG_DIR/medqa_orchestrator.log"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u scripts/build_hotflip.py \
        --input "$INPUT" --blackbox-input "$BLACKBOX" --output "$HOTFLIP" \
        --dataset medqa --gpu 0 --verbose > "$LOG_DIR/medqa_hotflip.log" 2>&1
    hotflip_status=$?
    if [ "$hotflip_status" -ne 0 ]; then
        printf '%s attempt %s: hotflip failed status=%s\n' "$(date -Is)" "$attempt" "$hotflip_status" >> "$LOG_DIR/medqa_orchestrator.log"
        continue
    fi

    "$PYTHON" scripts/validate_training_data.py \
        --source "$INPUT" --blackbox "$BLACKBOX" --hotflip "$HOTFLIP" --dataset medqa \
        > "$LOG_DIR/medqa_validation.log" 2>&1
    validation_status=$?
    if [ "$validation_status" -eq 0 ]; then
        printf '%s MedQA USMLE generation completed and validated\n' "$(date -Is)" >> "$LOG_DIR/medqa_orchestrator.log"
        exit 0
    fi
    printf '%s attempt %s: validation failed status=%s; regenerating\n' "$(date -Is)" "$attempt" "$validation_status" >> "$LOG_DIR/medqa_orchestrator.log"
done

printf '%s MedQA USMLE generation failed after two attempts\n' "$(date -Is)" >> "$LOG_DIR/medqa_orchestrator.log"
exit 1
