#!/usr/bin/env bash
# Run the official-aligned TrustRAG matrix without regenerating retrieval files.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
GPU=4
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))
LOG_ROOT=logs/formal_dot_trustrag_official_aligned_matrix
PID_FILE="$LOG_ROOT/queue.pid"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"

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

validate_inputs() {
  local retrieval=$1
  local source=$2
  local ids=$3
  local expected_count=$4
  local label_kind=$5
  "$PYTHON" - "$retrieval" "$source" "$ids" "$expected_count" "$label_kind" <<'PY'
import json
import math
import sys
from pathlib import Path

retrieval_path, source_path, ids_path, expected_count, label_kind = sys.argv[1:]
expected_count = int(expected_count)
for path in map(Path, (retrieval_path, source_path, ids_path)):
    assert path.is_file(), path
retrieval = json.loads(Path(retrieval_path).read_text())
source = json.loads(Path(source_path).read_text())
rows = list(source.values()) if isinstance(source, dict) else source
ids = {line.strip() for line in Path(ids_path).read_text().splitlines() if line.strip()}
source_ids = {str(row["id"]) for row in rows}
labels = {"A", "B", "C", "D"} if label_kind == "mcq" else {"YES", "NO", "MAYBE"}
assert len(rows) == expected_count, len(rows)
assert source_ids == ids == set(map(str, retrieval)), (len(source_ids), len(ids), len(retrieval))
assert all(
    str(row.get("target_label", "")).strip().upper() in labels
    and len(row.get("adv_texts") or []) == 5
    and all(str(text).strip() for text in row["adv_texts"])
    for row in rows
)
assert all(len(candidates) >= 5 for candidates in retrieval.values())
assert all(math.isfinite(float(score)) for candidates in retrieval.values() for score in candidates.values())
PY
}

is_complete_output() {
  local output=$1
  local expected_count=$2
  local label_kind=$3
  test -s "$output" || return 1
  "$PYTHON" - "$output" "$expected_count" "$label_kind" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected_count = int(sys.argv[2])
labels = {"A", "B", "C", "D"} if sys.argv[3] == "mcq" else {"YES", "NO", "MAYBE"}
rows = payload[0]["iter_0"]
assert len(rows) == expected_count, len(rows)
assert len({str(row["id"]) for row in rows}) == expected_count
assert all(str(row.get("parsed_pred_label", "")).strip().upper() in labels for row in rows)
PY
}

run_eval() {
  local dataset_tag=$1
  local defense=$2
  local attack=$3
  local expected_count=$4
  local label_kind=$5
  local retrieval=$6
  local source=$7
  local ids=$8
  local query_dir="formal_dot_contriever_${dataset_tag}_${defense}_official_aligned"
  local name="formal_dot_${dataset_tag}_contriever_${defense}_official_aligned_${attack}"
  local output="results/query_results/${query_dir}/${name}.json"
  local log_dir="$LOG_ROOT/${dataset_tag}/${defense}"
  local extra_args=()

  mkdir -p "$log_dir"
  if test -e "$output"; then
    if is_complete_output "$output" "$expected_count" "$label_kind"; then
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
  echo "$(date '+%F %T') starting ${dataset_tag}/${defense}/${attack} on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method "$attack" --adv_source json --adv_json_path "$source" \
    --target_ids_path "$ids" --retrieval_results_path "$retrieval" \
    --score_function dot --adv_per_query 5 --M "$expected_count" --repeat_times 1 --seed 12 \
    --asr_match_mode strict --trustrag_filter "${extra_args[@]}" --name "$name" \
    > "$log_dir/${attack}.log" 2>&1
  is_complete_output "$output" "$expected_count" "$label_kind"
  echo "$(date '+%F %T') completed ${dataset_tag}/${defense}/${attack}"
}

PUBMED_RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
PUBMED_SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
PUBMED_IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
MEDQA_RETRIEVAL=results/beir_results/mirage_medqa_all-contriever.json
MEDQA_SOURCE=results/adv_targeted_results/mirage_medqa_all.json
MEDQA_IDS=results/adv_targeted_results/mirage_medqa_all.ids

validate_inputs "$PUBMED_RETRIEVAL" "$PUBMED_SOURCE" "$PUBMED_IDS" 500 yesno
validate_inputs "$MEDQA_RETRIEVAL" "$MEDQA_SOURCE" "$MEDQA_IDS" 1273 mcq

for attack in None LM_targeted hotflip; do
  run_eval pubmedqa trustrag "$attack" 500 yesno "$PUBMED_RETRIEVAL" "$PUBMED_SOURCE" "$PUBMED_IDS"
  run_eval pubmedqa trustrag_medcluster "$attack" 500 yesno "$PUBMED_RETRIEVAL" "$PUBMED_SOURCE" "$PUBMED_IDS"
done
for attack in None LM_targeted hotflip; do
  run_eval medqa trustrag "$attack" 1273 mcq "$MEDQA_RETRIEVAL" "$MEDQA_SOURCE" "$MEDQA_IDS"
  run_eval medqa trustrag_medcluster "$attack" 1273 mcq "$MEDQA_RETRIEVAL" "$MEDQA_SOURCE" "$MEDQA_IDS"
done
echo "$(date '+%F %T') official-aligned TrustRAG medical matrix complete"
