#!/usr/bin/env bash
# Original Contriever raw-dot evaluation queue.  Reuses verified archival
# retrieval results and waits for safe host memory before each GPU run.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
PUB_RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
PUB_SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
PUB_IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
MED_RETRIEVAL=results/beir_results/mirage_medqa_all-contriever.json
MED_RETRIEVAL_LOG=logs/user_runs_logs/medqa_contriever_retrieval.log
MED_SOURCE=results/adv_targeted_results/mirage_medqa_all.json
MED_IDS=results/adv_targeted_results/mirage_medqa_all.ids
LOG_ROOT=logs/formal_dot_original_contriever_pubmedqa_clean_medqa_trustrag_n60_20260803
PID_FILE="$LOG_ROOT/queue.pid"
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

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

for path in (pub_retrieval, pub_metadata, pub_source, pub_ids,
             med_retrieval, med_retrieval_log, med_source, med_ids):
    assert path.is_file(), path

def rows(path):
    data = json.load(path.open(encoding="utf-8"))
    return list(data.values()) if isinstance(data, dict) else data

pub_rows = rows(pub_source)
pub_id_set = {str(row["id"]) for row in pub_rows}
assert len(pub_rows) == len(pub_id_set) == 500
assert pub_id_set == {line.strip() for line in pub_ids.read_text().splitlines() if line.strip()}
assert all(str(row.get("target_label", "")).strip().lower() in {"yes", "no", "maybe"} for row in pub_rows)

metadata = json.load(pub_metadata.open(encoding="utf-8"))
assert metadata.get("score_function") == "dot", metadata
assert metadata.get("model_code") == "contriever", metadata
pub_results = json.load(pub_retrieval.open(encoding="utf-8"))
assert set(map(str, pub_results)) == pub_id_set
assert all(len(candidates) >= 5 for candidates in pub_results.values())

med_rows = rows(med_source)
med_id_set = {str(row["id"]) for row in med_rows}
assert len(med_rows) == len(med_id_set) == 1273
assert med_id_set == {line.strip() for line in med_ids.read_text().splitlines() if line.strip()}
assert all(
    str(row.get("target_label", "")).strip().upper() in {"A", "B", "C", "D"}
    and len(row.get("adv_texts") or []) == 5
    and all(str(text).strip() for text in row["adv_texts"])
    for row in med_rows
)
selected = med_rows[:60]
assert len({str(row["id"]) for row in selected}) == 60

retrieval_log = med_retrieval_log.read_text(encoding="utf-8", errors="replace")
assert "model_code='contriever'" in retrieval_log
assert "score_function='dot'" in retrieval_log
assert "mirage_dataset='medqa'" in retrieval_log
assert "result_output='results/beir_results/mirage_medqa_all-contriever.json'" in retrieval_log
med_results = json.load(med_retrieval.open(encoding="utf-8"))
assert set(map(str, med_results)) == med_id_set
assert all(len(candidates) >= 100 for candidates in med_results.values())
assert all(math.isfinite(float(score)) for candidates in med_results.values() for score in candidates.values())

print("input validation passed: PubMedQA=500; MedQA n=60 uses first 60 rows in the fixed source order")
PY
}

is_complete_output() {
  local output=$1
  local expected_count=$2
  local label_space=$3
  test -s "$output" || return 1
  "$PYTHON" - "$output" "$expected_count" "$label_space" <<'PY'
import json
import sys

path, expected_count, label_space = sys.argv[1:]
rows = json.load(open(path, encoding="utf-8"))[0]["iter_0"]
expected_count = int(expected_count)
allowed = {item.strip() for item in label_space.split(",")}
assert len(rows) == expected_count, len(rows)
assert len({str(row["id"]) for row in rows}) == expected_count
assert all(str(row.get("parsed_pred_label", "")).strip().lower() in allowed for row in rows)
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

run_pubmedqa_clean() {
  local query_dir=formal_dot_contriever_pubmedqa_clean_20260803
  local name=formal_dot_pubmedqa_contriever_clean_20260803_None
  local output="results/query_results/${query_dir}/${name}.json"
  local log="$LOG_ROOT/pubmedqa_clean.log"

  if test -e "$output"; then
    if is_complete_output "$output" 500 yes,no,maybe; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "refusing to overwrite incomplete output: $output" >&2
    return 2
  fi

  wait_for_memory
  echo "$(date '+%F %T') starting PubMedQA clean/no-defense on GPU4"
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method None --adv_source json --adv_json_path "$PUB_SOURCE" \
    --target_ids_path "$PUB_IDS" --retrieval_results_path "$PUB_RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --name "$name" > "$log" 2>&1
  is_complete_output "$output" 500 yes,no,maybe
  echo "$(date '+%F %T') completed PubMedQA clean/no-defense"
}

run_medqa_trustrag_attack() {
  local attack=$1
  local query_dir=formal_dot_contriever_medqa_trustrag_n60_20260803
  local name="formal_dot_medqa_contriever_trustrag_n60_20260803_${attack}"
  local output="results/query_results/${query_dir}/${name}.json"
  local log="$LOG_ROOT/medqa_trustrag_${attack}.log"

  if test -e "$output"; then
    if is_complete_output "$output" 60 a,b,c,d; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "refusing to overwrite incomplete output: $output" >&2
    return 2
  fi

  wait_for_memory
  echo "$(date '+%F %T') starting MedQA n=60 TrustRAG/${attack} on GPU4"
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method "$attack" --adv_source json --adv_json_path "$MED_SOURCE" \
    --target_ids_path "$MED_IDS" --retrieval_results_path "$MED_RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 60 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --trustrag_filter --name "$name" > "$log" 2>&1
  is_complete_output "$output" 60 a,b,c,d
  echo "$(date '+%F %T') completed MedQA n=60 TrustRAG/${attack}"
}

validate_inputs
run_pubmedqa_clean
run_medqa_trustrag_attack LM_targeted
run_medqa_trustrag_attack hotflip
echo "$(date '+%F %T') queue complete"
