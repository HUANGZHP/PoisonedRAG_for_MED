#!/usr/bin/env bash

# 顺序完成 PQA-U HotFlip、全量合并校验和 v4 的 1:1 偏好微调。
set -u -o pipefail

ROOT=/home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
WAIT_PID="$1"
INPUT="$ROOT/checkpoint/contriever_v4/input"
LOG_DIR="$ROOT/logs/contriever_v4"

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
printf '%s v4 pipeline waiting for PQA-U blackbox\n' "$(date -Is)" > "$LOG_DIR/orchestrator.log"

while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
if [ ! -s "$INPUT/pqau_blackbox.jsonl" ]; then
    printf '%s PQA-U blackbox failed or produced no output\n' "$(date -Is)" >> "$LOG_DIR/orchestrator.log"
    exit 1
fi

printf '%s PQA-U HotFlip started on GPU 4\n' "$(date -Is)" >> "$LOG_DIR/orchestrator.log"
CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u scripts/build_hotflip.py \
    --input "$INPUT/pqau_sample_source.json" \
    --blackbox-input "$INPUT/pqau_blackbox.jsonl" \
    --output "$INPUT/pqau_hotflip.jsonl" \
    --dataset pubmedqa --gpu 0 --verbose > "$LOG_DIR/pqau_hotflip.log" 2>&1
if [ "$?" -ne 0 ]; then
    printf '%s PQA-U HotFlip failed\n' "$(date -Is)" >> "$LOG_DIR/orchestrator.log"
    exit 1
fi

"$PYTHON" scripts/assemble_pubmedqa_v4.py \
    --source "$INPUT/pubmedqa_l_u_source.json" \
    --pqal-blackbox processed/pubmedqa_blackbox.jsonl \
    --pqau-blackbox "$INPUT/pqau_blackbox.jsonl" \
    --pqal-hotflip processed/pubmedqa_hotflip.jsonl \
    --pqau-hotflip "$INPUT/pqau_hotflip.jsonl" \
    --blackbox-output processed/pubmedqa_v4_blackbox.jsonl \
    --hotflip-output processed/pubmedqa_v4_hotflip.jsonl > "$LOG_DIR/assemble.log" 2>&1
if [ "$?" -ne 0 ]; then
    printf '%s merged PQA-L/PQA-U validation failed\n' "$(date -Is)" >> "$LOG_DIR/orchestrator.log"
    exit 1
fi

"$PYTHON" scripts/validate_training_data.py \
    --source "$INPUT/pubmedqa_l_u_source.json" \
    --blackbox processed/pubmedqa_v4_blackbox.jsonl \
    --hotflip processed/pubmedqa_v4_hotflip.jsonl \
    --dataset pubmedqa > "$LOG_DIR/validation.log" 2>&1
if [ "$?" -ne 0 ]; then
    printf '%s final PQA-L/PQA-U validation failed\n' "$(date -Is)" >> "$LOG_DIR/orchestrator.log"
    exit 1
fi

printf '%s v4 training started from original facebook/contriever on GPU 4\n' "$(date -Is)" >> "$LOG_DIR/orchestrator.log"
CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u -m contriever_stage1.train_replay \
    --pubmed-blackbox-path processed/pubmedqa_v4_blackbox.jsonl \
    --pubmed-hotflip-path processed/pubmedqa_v4_hotflip.jsonl \
    --medqa-blackbox-path checkpoint/contriever_v2/input/medqa_blackbox.jsonl \
    --medqa-hotflip-path checkpoint/contriever_v2/input/medqa_hotflip.jsonl \
    --init-model-path /home/HF_Model/facebook/contriever \
    --output-dir checkpoint/contriever_v4 \
    --epochs 3 --batch-size 16 --pubmed-per-batch 8 --medqa-per-batch 8 \
    --learning-rate 1e-5 --weight-decay 0.01 --temperature 0.05 --gradient-clip 1.0 \
    --validation-size 512 --final-evaluation-size 100 --gpu 0 --mixed-precision \
    > "$LOG_DIR/train.log" 2>&1
status=$?
printf '%s v4 training finished with status=%s\n' "$(date -Is)" "$status" >> "$LOG_DIR/orchestrator.log"
exit "$status"
