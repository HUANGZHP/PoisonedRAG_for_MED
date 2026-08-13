#!/usr/bin/env bash
# Full clean GPT-5 mini baseline: no attack and every project defense disabled.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
MODEL_NAME=gpt-5-mini
PUB_RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
PUB_SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
PUB_IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
MED_RETRIEVAL=results/beir_results/mirage_medqa_all-contriever.json
MED_RETRIEVAL_LOG=logs/user_runs_logs/medqa_contriever_retrieval.log
MED_SOURCE=results/adv_targeted_results/mirage_medqa_all.json
MED_IDS=results/adv_targeted_results/mirage_medqa_all.ids
RUN_TAG=${BASELINE_RUN_TAG:-}
REQUIRE_NONEMPTY_LABELS=${BASELINE_REQUIRE_NONEMPTY_LABELS:-0}
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "BASELINE_RUN_TAG may contain only letters, numbers, _ and -" >&2
  exit 2
fi
if [[ "$REQUIRE_NONEMPTY_LABELS" != 0 && "$REQUIRE_NONEMPTY_LABELS" != 1 ]]; then
  echo "BASELINE_REQUIRE_NONEMPTY_LABELS must be 0 or 1" >&2
  exit 2
fi
RUN_SUFFIX=${RUN_TAG:+_$RUN_TAG}
LOG_ROOT="logs/formal_dot_gpt5_mini_clean_baseline_pubmedqa_medqa_full_20260810${RUN_SUFFIX}"
PREVIOUS_PUBMED_QUEUE=logs/formal_dot_pubmedqa_medical_kg_original_weight30_trustrag_lmtargeted_full_20260810/queue.pid
PREVIOUS_MEDQA_QUEUE=logs/formal_dot_medqa_medical_kg_original_hardfilter_blackwhite_full_20260810/queue.pid
RUN_NOW=false
RUN_GPU=4
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$LOG_ROOT/queue.pid"
trap 'rm -f "$LOG_ROOT/queue.pid"' EXIT

wait_for_queue() {
  local pid_file=$1
  local label=$2
  if ! test -s "$pid_file"; then
    return
  fi
  local pid
  pid=$(tr -d '[:space:]' < "$pid_file")
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "ignoring invalid predecessor pid file: $pid_file" >&2
    return
  fi
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for ${label} queue pid=${pid} to release its GPU"
    sleep 300
  done
}

validate_inputs() {
  "$PYTHON" - <<'PY'
import json
import math
from pathlib import Path

pub_retrieval = Path("results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json")
pub_metadata = Path(f"{pub_retrieval}.meta.json")
pub_source = Path("results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json")
pub_ids = Path("results/adv_targeted_results/mirage_pubmedqa_all.ids")
med_retrieval = Path("results/beir_results/mirage_medqa_all-contriever.json")
med_retrieval_log = Path("logs/user_runs_logs/medqa_contriever_retrieval.log")
med_source = Path("results/adv_targeted_results/mirage_medqa_all.json")
med_ids = Path("results/adv_targeted_results/mirage_medqa_all.ids")
model_config = Path("model_configs/gpt-5-mini_config.json")

for path in (pub_retrieval, pub_metadata, pub_source, pub_ids, med_retrieval, med_retrieval_log, med_source, med_ids, model_config):
    assert path.is_file() and path.stat().st_size > 0, path

def rows(path):
    data = json.load(path.open(encoding="utf-8"))
    return list(data.values()) if isinstance(data, dict) else data

pub_rows = rows(pub_source)
pub_ids_set = {str(row["id"]) for row in pub_rows}
assert len(pub_rows) == len(pub_ids_set) == 500
assert pub_ids_set == {line.strip() for line in pub_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
assert all(str(row.get("target_label", "")).strip().lower() in {"yes", "no", "maybe"} for row in pub_rows)
metadata = json.load(pub_metadata.open(encoding="utf-8"))
assert metadata.get("model_code") == "contriever", metadata
assert metadata.get("score_function") == "dot", metadata
pub_results = json.load(pub_retrieval.open(encoding="utf-8"))
assert set(map(str, pub_results)) == pub_ids_set
assert all(len(candidates) >= 5 for candidates in pub_results.values())

med_rows = rows(med_source)
med_ids_set = {str(row["id"]) for row in med_rows}
assert len(med_rows) == len(med_ids_set) == 1273
assert med_ids_set == {line.strip() for line in med_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
assert all(str(row.get("target_label", "")).strip().upper() in {"A", "B", "C", "D"} for row in med_rows)
retrieval_log = med_retrieval_log.read_text(encoding="utf-8", errors="replace")
assert "model_code='contriever'" in retrieval_log
assert "score_function='dot'" in retrieval_log
assert "mirage_dataset='medqa'" in retrieval_log
med_results = json.load(med_retrieval.open(encoding="utf-8"))
assert set(map(str, med_results)) == med_ids_set
assert all(len(candidates) >= 5 for candidates in med_results.values())
assert all(math.isfinite(float(score)) for candidates in med_results.values() for score in candidates.values())
print("input validation passed: GPT-5 mini clean/no-defense, PubMedQA=500, MedQA=1273, existing Contriever retrieval")
PY
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

is_complete_output() {
  local output=$1
  local expected_count=$2
  local label_space=$3
  test -s "$output" || return 1
  "$PYTHON" - "$output" "$expected_count" "$label_space" "$REQUIRE_NONEMPTY_LABELS" <<'PY'
import json
import sys

path, expected_count, label_space, require_nonempty = sys.argv[1:]
rows = json.load(open(path, encoding="utf-8"))[0]["iter_0"]
expected_count = int(expected_count)
allowed = {item.strip().lower() for item in label_space.split(",")}
assert len(rows) == expected_count
assert len({str(row["id"]) for row in rows}) == expected_count
assert all(str(row.get("parsed_pred_label") or "").strip().lower() in allowed for row in rows)
assert all(row.get("medical_kg_filter_enabled") is False for row in rows)
assert all(row.get("trustrag_filter_enabled") is False for row in rows)
assert all(row.get("medical_semantic_clustering_enabled") is False for row in rows)
assert all(row.get("judge_enabled") is False for row in rows)
if require_nonempty == "1":
    assert all(str(row.get("output_poison") or "").strip() for row in rows)
    assert all(str(row.get("parsed_pred_label") or "").strip() for row in rows)
PY
}

run_pubmedqa() {
  local query_dir="formal_dot_gpt5_mini_clean_baseline_pubmedqa_medqa_full_20260810${RUN_SUFFIX}"
  local name="formal_dot_pubmedqa_contriever_gpt5_mini_clean_no_defense_full_20260810${RUN_SUFFIX}"
  local output="results/query_results/${query_dir}/${name}.json"
  local log="$LOG_ROOT/pubmedqa_clean.log"
  if test -e "$output"; then
    if is_complete_output "$output" 500 yes,no,maybe,; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "refusing to overwrite incomplete output: $output" >&2
    return 2
  fi
  wait_for_memory
  echo "$(date '+%F %T') starting PubMedQA GPT-5 mini clean/no-defense on GPU${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name "$MODEL_NAME" \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method None --adv_source json --adv_json_path "$PUB_SOURCE" \
    --target_ids_path "$PUB_IDS" --retrieval_results_path "$PUB_RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --no_medical_kg_filter \
    --name "$name" > "$log" 2>&1
  is_complete_output "$output" 500 yes,no,maybe,
  echo "$(date '+%F %T') completed PubMedQA GPT-5 mini clean/no-defense"
}

run_medqa() {
  local query_dir="formal_dot_gpt5_mini_clean_baseline_pubmedqa_medqa_full_20260810${RUN_SUFFIX}"
  local name="formal_dot_medqa_contriever_gpt5_mini_clean_no_defense_full_20260810${RUN_SUFFIX}"
  local output="results/query_results/${query_dir}/${name}.json"
  local log="$LOG_ROOT/medqa_clean.log"
  if test -e "$output"; then
    if is_complete_output "$output" 1273 a,b,c,d,; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "refusing to overwrite incomplete output: $output" >&2
    return 2
  fi
  wait_for_memory
  echo "$(date '+%F %T') starting MedQA GPT-5 mini clean/no-defense on GPU${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name "$MODEL_NAME" \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method None --adv_source json --adv_json_path "$MED_SOURCE" \
    --target_ids_path "$MED_IDS" --retrieval_results_path "$MED_RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 1273 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --no_medical_kg_filter \
    --name "$name" > "$log" 2>&1
  is_complete_output "$output" 1273 a,b,c,d,
  echo "$(date '+%F %T') completed MedQA GPT-5 mini clean/no-defense"
}

if [[ "${1:-}" == "--preflight" && $# -eq 1 ]]; then
  validate_inputs
  echo "$(date '+%F %T') preflight complete"
  exit 0
elif [[ "${1:-}" == "--run-now" && $# -eq 1 ]]; then
  RUN_NOW=true
  RUN_GPU=7
  MIN_AVAILABLE_KIB=$((160 * 1024 * 1024))
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--preflight|--run-now]" >&2
  exit 2
fi

if [[ "$RUN_NOW" == false ]]; then
  wait_for_queue "$PREVIOUS_PUBMED_QUEUE" "PubMedQA KG+TrustRAG"
  wait_for_queue "$PREVIOUS_MEDQA_QUEUE" "MedQA KG hard-filter"
else
  echo "$(date '+%F %T') priority run: skipping predecessor waits; GPU${RUN_GPU}, minimum available memory=${MIN_AVAILABLE_KIB}KiB"
fi
validate_inputs
run_pubmedqa
run_medqa
echo "$(date '+%F %T') queue complete"
