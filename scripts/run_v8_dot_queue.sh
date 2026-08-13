#!/usr/bin/env bash
# Queue the v8 dot own-5 formal protocol after explicitly supplied GPU jobs finish.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
RESULT_DIR=results/beir_results/formal_dot_v8
LOG_DIR=logs/formal_dot_v8
mkdir -p "$RESULT_DIR" "$LOG_DIR"

for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
done

run_retrieval() {
  local gpu=$1 task=$2 queries=$3
  "$PYTHON" -u evaluate_beir.py \
    --model_code contriever_v8 --dataset pubmed --split test --top_k 100 \
    --score_function dot --gpu_id "$gpu" --per_gpu_batch_size 64 --max_length 128 \
    --queries-json "$queries" \
    --result_output "$RESULT_DIR/mirage_${task}_all-contriever_v8-dot.json" \
    > "$LOG_DIR/${task}_retrieval.log" 2>&1
}

run_retrieval 3 pubmedqa results/adv_targeted_results/mirage_pubmedqa_all.queries.json &
retrieval_pubmed=$!
run_retrieval 5 medqa results/adv_targeted_results/mirage_medqa_all.queries.json &
retrieval_med=$!
wait "$retrieval_pubmed"
wait "$retrieval_med"

test -s "$RESULT_DIR/mirage_pubmedqa_all-contriever_v8-dot.json.meta.json"
test -s "$RESULT_DIR/mirage_medqa_all-contriever_v8-dot.json.meta.json"

run_eval() {
  local gpu=$1 task=$2 attack=$3 adv_json=$4 ids=$5 sample_count=$6
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u main.py \
    --eval_model_code contriever_v8 --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir formal_dot_v8 \
    --attack_method "$attack" --adv_source json --adv_json_path "$adv_json" --target_ids_path "$ids" \
    --retrieval_results_path "$RESULT_DIR/mirage_${task}_all-contriever_v8-dot.json" \
    --score_function dot --adv_per_query 5 --M "$sample_count" --repeat_times 1 \
    --asr_match_mode strict --name "formal_dot_${task}_contriever_v8_${attack}" \
    > "$LOG_DIR/${task}_${attack}.log" 2>&1
}

for attack in None LM_targeted hotflip; do
  run_eval 3 pubmedqa "$attack" results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json results/adv_targeted_results/mirage_pubmedqa_all.ids 500 &
  eval_pubmed=$!
  run_eval 5 medqa "$attack" results/adv_targeted_results/mirage_medqa_all.json results/adv_targeted_results/mirage_medqa_all.ids 1273 &
  eval_med=$!
  wait "$eval_pubmed"
  wait "$eval_med"
done
