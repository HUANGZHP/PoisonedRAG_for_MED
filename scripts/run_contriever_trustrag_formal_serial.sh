#!/usr/bin/env bash
# Run original Contriever PubMedQA TrustRAG suites safely: one task at a time.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
V7_PID_FILE=logs/formal_dot_v7/serial_queue.pid
RETRIEVAL_DIR=results/beir_results/formal_dot_contriever
LOG_ROOT=logs/formal_dot_contriever_defenses
MIN_AVAILABLE_KIB=$((180 * 1024 * 1024))
mkdir -p "$RETRIEVAL_DIR" "$LOG_ROOT"

if test -s "$V7_PID_FILE"; then
  v7_pid=$(cat "$V7_PID_FILE")
  while kill -0 "$v7_pid" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for v7 serial queue pid=$v7_pid"
    sleep 120
  done
fi

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

retrieval="$RETRIEVAL_DIR/mirage_pubmedqa_all-contriever-dot.json"
if ! test -s "$retrieval.meta.json"; then
  wait_for_memory
  "$PYTHON" -u evaluate_beir.py \
    --model_code contriever --dataset pubmed --split test --top_k 100 \
    --score_function dot --gpu_id 4 --per_gpu_batch_size 64 --max_length 128 \
    --queries-json results/adv_targeted_results/mirage_pubmedqa_all.queries.json \
    --result_output "$retrieval" \
    > "$LOG_ROOT/retrieval.log" 2>&1
fi
test -s "$retrieval.meta.json"

run_or_skip_eval() {
  local defense=$1 attack=$2
  local query_dir="formal_dot_contriever_${defense}"
  local log_dir="$LOG_ROOT/${defense}"
  local output="results/query_results/${query_dir}/formal_dot_pubmedqa_contriever_${defense}_${attack}.json"
  mkdir -p "$log_dir"
  if test -s "$output"; then
    echo "$(date '+%F %T') reuse completed $output"
    return
  fi
  wait_for_memory
  local cluster_args=()
  if [[ "$defense" == "trustrag_medcluster" ]]; then
    cluster_args=(--medical_semantic_clustering)
  fi
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method "$attack" --adv_source json \
    --adv_json_path results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json \
    --target_ids_path results/adv_targeted_results/mirage_pubmedqa_all.ids \
    --retrieval_results_path "$retrieval" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 \
    --asr_match_mode strict --trustrag_filter "${cluster_args[@]}" \
    --name "formal_dot_pubmedqa_contriever_${defense}_${attack}" \
    > "$log_dir/${attack}.log" 2>&1
}

for defense in trustrag trustrag_medcluster; do
  for attack in None LM_targeted hotflip; do
    run_or_skip_eval "$defense" "$attack"
  done
done
