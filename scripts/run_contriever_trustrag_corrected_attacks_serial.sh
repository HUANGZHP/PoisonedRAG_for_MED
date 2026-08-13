#!/usr/bin/env bash
# Run the repaired original-Contriever PubMedQA TrustRAG attack matrix safely.
# The clean baselines use the metadata-backed retrieval below, so attacks must
# use that same file rather than the historical dot retrieval.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
ATTACK_SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
TARGET_IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
LOG_ROOT=logs/formal_dot_contriever_defenses_corrected
PID_FILE="$LOG_ROOT/trustrag_corrected_attacks_serial.pid"
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"

validate_inputs() {
  "$PYTHON" -c '
import json
import sys

meta = json.load(open(sys.argv[1]))
assert meta.get("score_function") == "dot", meta
assert meta.get("model_code") == "contriever", meta
source = json.load(open(sys.argv[2]))
rows = list(source.values()) if isinstance(source, dict) else source
assert len(rows) == 500, len(rows)
assert all(
    str(row.get("target_label", "")).strip()
    and len(row.get("adv_texts") or []) == 5
    and all(str(text).strip() for text in row["adv_texts"])
    for row in rows
)
' "$RETRIEVAL.meta.json" "$ATTACK_SOURCE"
}

is_complete_output() {
  local output=$1
  test -s "$output" || return 1
  "$PYTHON" -c '
import json
import sys

payload = json.load(open(sys.argv[1]))
rows = payload[0]["iter_0"]
assert len(rows) == 500, len(rows)
assert len({row["id"] for row in rows}) == 500
' "$output"
}

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

run_eval() {
  local defense=$1
  local attack=$2
  local query_dir="formal_dot_contriever_${defense}_corrected"
  local name="formal_dot_pubmedqa_contriever_${defense}_corrected_${attack}"
  local output="results/query_results/${query_dir}/${name}.json"
  local log_dir="$LOG_ROOT/${defense}"
  local extra_args=()

  mkdir -p "$log_dir"
  if test -e "$output"; then
    if is_complete_output "$output"; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "$(date '+%F %T') refusing to overwrite incomplete output $output" >&2
    return 2
  fi

  if [[ "$defense" == "trustrag_medcluster" ]]; then
    extra_args=(--medical_semantic_clustering)
  fi

  wait_for_memory
  echo "$(date '+%F %T') starting ${defense}/${attack} on GPU4"
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method "$attack" --adv_source json --adv_json_path "$ATTACK_SOURCE" \
    --target_ids_path "$TARGET_IDS" --retrieval_results_path "$RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --trustrag_filter "${extra_args[@]}" --name "$name" \
    > "$log_dir/${attack}.log" 2>&1
  is_complete_output "$output"
  echo "$(date '+%F %T') completed ${defense}/${attack}"
}

validate_inputs
run_eval trustrag LM_targeted
run_eval trustrag_medcluster LM_targeted
run_eval trustrag hotflip
run_eval trustrag_medcluster hotflip
echo "$(date '+%F %T') corrected TrustRAG attack queue complete"
