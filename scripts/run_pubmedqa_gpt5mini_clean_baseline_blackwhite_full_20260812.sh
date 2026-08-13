#!/usr/bin/env bash
# PubMedQA black-box/white-box GPT-5 mini baseline: attacks enabled, every defense disabled.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
MODEL_NAME=gpt-5-mini
RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
QUERY_DIR=formal_dot_pubmedqa_gpt5_mini_clean_baseline_blackwhite_full_20260812
LOG_ROOT=logs/formal_dot_pubmedqa_gpt5_mini_clean_baseline_blackwhite_full_20260812
RUN_GPU=2
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$LOG_ROOT/queue.pid"
trap 'rm -f "$LOG_ROOT/queue.pid"' EXIT

validate_inputs() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

retrieval_path = Path("results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json")
metadata_path = Path(f"{retrieval_path}.meta.json")
source_path = Path("results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json")
ids_path = Path("results/adv_targeted_results/mirage_pubmedqa_all.ids")
config_path = Path("model_configs/gpt-5-mini_config.json")
for path in (retrieval_path, metadata_path, source_path, ids_path, config_path):
    assert path.is_file() and path.stat().st_size > 0, path

rows = json.load(source_path.open(encoding="utf-8"))
if isinstance(rows, dict):
    rows = list(rows.values())
ids = [str(row["id"]) for row in rows]
assert len(rows) == len(set(ids)) == 500
assert set(ids) == {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
assert all(str(row.get("target_label", "")).strip().lower() in {"yes", "no", "maybe"} for row in rows)
assert all(len(row.get("adv_texts") or []) == 5 for row in rows)

metadata = json.load(metadata_path.open(encoding="utf-8"))
assert metadata.get("score_function") == "dot", metadata
assert metadata.get("model_code") == "contriever", metadata
retrieval = json.load(retrieval_path.open(encoding="utf-8"))
assert set(map(str, retrieval)) == set(ids)
assert all(len(candidates) >= 5 for candidates in retrieval.values())
print("input validation passed: PubMedQA=500, gpt-5-mini, Contriever/raw-dot, own-5 attack, all defenses disabled")
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
  local attack=$2
  test -s "$output" || return 1
  "$PYTHON" - "$output" "$attack" <<'PY'
import json
import sys

path, attack = sys.argv[1:]
rows = json.load(open(path, encoding="utf-8"))[0]["iter_0"]
assert len(rows) == 500
assert len({str(row["id"]) for row in rows}) == 500
assert all(str(row.get("parsed_pred_label") or "").strip().lower() in {"yes", "no", "maybe"} for row in rows)
assert all(str(row.get("output_poison") or "").strip() for row in rows)
assert all(row.get("medical_kg_filter_enabled") is False for row in rows)
assert all(row.get("trustrag_filter_enabled") is False for row in rows)
assert all(row.get("medical_semantic_clustering_enabled") is False for row in rows)
assert all(row.get("judge_enabled") is False for row in rows)
assert attack in {"LM_targeted", "hotflip"}
PY
}

run_experiment() {
  local attack=$1
  local variant=$2
  local name="formal_dot_pubmedqa_contriever_gpt5_mini_clean_baseline_${variant}_full_20260812"
  local output="results/query_results/${QUERY_DIR}/${name}.json"
  local log="$LOG_ROOT/${variant}.log"
  if test -e "$output"; then
    if is_complete_output "$output" "$attack"; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "refusing to overwrite incomplete output: $output" >&2
    return 2
  fi
  wait_for_memory
  echo "$(date '+%F %T') starting PubMedQA GPT-5 mini clean-baseline/${variant} on GPU${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name "$MODEL_NAME" \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$QUERY_DIR" \
    --attack_method "$attack" --adv_source json --adv_json_path "$SOURCE" \
    --target_ids_path "$IDS" --retrieval_results_path "$RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --no_medical_kg_filter \
    --name "$name" > "$log" 2>&1
  is_complete_output "$output" "$attack"
  echo "$(date '+%F %T') completed PubMedQA GPT-5 mini clean-baseline/${variant}"
}

if [[ "${1:-}" == "--preflight" && $# -eq 1 ]]; then
  validate_inputs
  echo "$(date '+%F %T') preflight complete"
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--preflight]" >&2
  exit 2
fi

validate_inputs
run_experiment LM_targeted blackbox_lm_targeted
run_experiment hotflip whitebox_hotflip
echo "$(date '+%F %T') queue complete"
