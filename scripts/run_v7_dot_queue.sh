#!/usr/bin/env bash
# Resume v7 formal dot/own-5 evaluation safely: one task at a time.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
RESULT_DIR=results/beir_results/formal_dot_v7
QUERY_RESULT_DIR=results/query_results/formal_dot_v7
LOG_DIR=logs/formal_dot_v7
MIN_AVAILABLE_KIB=$((180 * 1024 * 1024))
mkdir -p "$RESULT_DIR" "$QUERY_RESULT_DIR" "$LOG_DIR"

wait_for_memory() {
  local available_kib
  while true; do
    available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if (( available_kib >= MIN_AVAILABLE_KIB )); then
      return
    fi
    echo "$(date '+%F %T') waiting: MemAvailable=${available_kib}KiB, need=${MIN_AVAILABLE_KIB}KiB"
    sleep 300
  done
}

run_retrieval() {
  local gpu=$1 task=$2 queries=$3
  wait_for_memory
  "$PYTHON" -u evaluate_beir.py \
    --model_code contriever_v7 --dataset pubmed --split test --top_k 100 \
    --score_function dot --gpu_id "$gpu" --per_gpu_batch_size 64 --max_length 128 \
    --queries-json "$queries" \
    --result_output "$RESULT_DIR/mirage_${task}_all-contriever_v7-dot.json" \
    > "$LOG_DIR/${task}_retrieval.log" 2>&1
}

run_eval() {
  local gpu=$1 task=$2 attack=$3 adv_json=$4 ids=$5 sample_count=$6
  wait_for_memory
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u main.py \
    --eval_model_code contriever_v7 --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir formal_dot_v7 \
    --attack_method "$attack" --adv_source json --adv_json_path "$adv_json" --target_ids_path "$ids" \
    --retrieval_results_path "$RESULT_DIR/mirage_${task}_all-contriever_v7-dot.json" \
    --score_function dot --adv_per_query 5 --M "$sample_count" --repeat_times 1 \
    --asr_match_mode strict --name "formal_dot_${task}_contriever_v7_${attack}" \
    > "$LOG_DIR/${task}_${attack}.log" 2>&1
}

run_or_skip_eval() {
  local gpu=$1 task=$2 attack=$3 adv_json=$4 ids=$5 sample_count=$6
  local output="$QUERY_RESULT_DIR/formal_dot_${task}_contriever_v7_${attack}.json"
  if test -s "$output"; then
    echo "$(date '+%F %T') reuse completed $output"
  else
    run_eval "$gpu" "$task" "$attack" "$adv_json" "$ids" "$sample_count"
  fi
}

if ! test -s "$RESULT_DIR/mirage_pubmedqa_all-contriever_v7-dot.json.meta.json"; then
  run_retrieval 4 pubmedqa results/adv_targeted_results/mirage_pubmedqa_all.queries.json
fi
if ! test -s "$RESULT_DIR/mirage_medqa_all-contriever_v7-dot.json.meta.json"; then
  run_retrieval 4 medqa results/adv_targeted_results/mirage_medqa_all.queries.json
fi

for attack in None LM_targeted hotflip; do
  run_or_skip_eval 4 pubmedqa "$attack" results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json results/adv_targeted_results/mirage_pubmedqa_all.ids 500
  run_or_skip_eval 4 medqa "$attack" results/adv_targeted_results/mirage_medqa_all.json results/adv_targeted_results/mirage_medqa_all.ids 1273
done
