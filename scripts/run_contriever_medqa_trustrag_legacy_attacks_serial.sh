#!/usr/bin/env bash
# Run the legacy-filter original-Contriever MedQA TrustRAG matrix using the
# verified archival raw-dot retrieval without rebuilding it.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
RETRIEVAL=results/beir_results/mirage_medqa_all-contriever.json
RETRIEVAL_LOG=logs/user_runs_logs/medqa_contriever_retrieval.log
ATTACK_SOURCE=results/adv_targeted_results/mirage_medqa_all.json
TARGET_IDS=results/adv_targeted_results/mirage_medqa_all.ids
LOG_ROOT=logs/formal_dot_contriever_medqa_defenses_legacy
PID_FILE="$LOG_ROOT/trustrag_medqa_legacy_attacks_serial.pid"
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"

validate_inputs() {
  "$PYTHON" -c '
import json
import math
from pathlib import Path

retrieval_path = Path("results/beir_results/mirage_medqa_all-contriever.json")
retrieval_log_path = Path("logs/user_runs_logs/medqa_contriever_retrieval.log")
source_path = Path("results/adv_targeted_results/mirage_medqa_all.json")
ids_path = Path("results/adv_targeted_results/mirage_medqa_all.ids")

for path in (retrieval_path, retrieval_log_path, source_path, ids_path):
    assert path.is_file(), path

retrieval_log = retrieval_log_path.read_text(encoding="utf-8", errors="replace")
assert "model_code='"'"'contriever'"'"'" in retrieval_log
assert "score_function='"'"'dot'"'"'" in retrieval_log
assert "mirage_dataset='"'"'medqa'"'"'" in retrieval_log
assert "result_output='"'"'results/beir_results/mirage_medqa_all-contriever.json'"'"'" in retrieval_log

source = json.load(source_path.open())
rows = list(source.values()) if isinstance(source, dict) else source
source_ids = {str(row["id"]) for row in rows}
target_ids = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
assert len(rows) == 1273, len(rows)
assert len(source_ids) == 1273, len(source_ids)
assert source_ids == target_ids
assert all(
    str(row.get("target_label", "")).strip().upper() in {"A", "B", "C", "D"}
    and len(row.get("adv_texts") or []) == 5
    and all(str(text).strip() for text in row["adv_texts"])
    for row in rows
)

retrieval = json.load(retrieval_path.open())
assert set(map(str, retrieval)) == source_ids
assert all(len(candidates) >= 100 for candidates in retrieval.values())
assert all(
    math.isfinite(float(score))
    for candidates in retrieval.values()
    for score in candidates.values()
)
'
}

is_complete_output() {
  local output=$1
  test -s "$output" || return 1
  "$PYTHON" -c '
import json
import sys

payload = json.load(open(sys.argv[1]))
rows = payload[0]["iter_0"]
assert len(rows) == 1273, len(rows)
assert len({str(row["id"]) for row in rows}) == 1273
assert all(str(row.get("parsed_pred_label", "")).strip().upper() in {"A", "B", "C", "D"} for row in rows)
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
  local query_dir="formal_dot_contriever_medqa_${defense}_legacy"
  local name="formal_dot_medqa_contriever_${defense}_legacy_${attack}"
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
    --score_function dot --adv_per_query 5 --M 1273 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --trustrag_filter "${extra_args[@]}" --name "$name" \
    > "$log_dir/${attack}.log" 2>&1
  is_complete_output "$output"
  echo "$(date '+%F %T') completed ${defense}/${attack}"
}

validate_inputs
run_eval trustrag None
run_eval trustrag_medcluster None
run_eval trustrag LM_targeted
run_eval trustrag_medcluster LM_targeted
run_eval trustrag hotflip
run_eval trustrag_medcluster hotflip
echo "$(date '+%F %T') legacy-filter MedQA TrustRAG attack queue complete"
