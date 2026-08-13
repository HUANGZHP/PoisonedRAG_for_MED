#!/usr/bin/env bash
# Fair PubMedQA clean baseline: same r3 inputs/settings as GPT-5 mini, only the answer model changes.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
MODEL_NAME=gpt4.1mini
RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
QUERY_DIR=formal_dot_gpt41mini_clean_baseline_pubmedqa_full_20260812
LOG_ROOT=logs/formal_dot_gpt41mini_clean_baseline_pubmedqa_full_20260812
RUN_GPU=3
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
config_path = Path("model_configs/gpt4.1mini_config.json")
for path in (retrieval_path, metadata_path, source_path, ids_path, config_path):
    assert path.is_file() and path.stat().st_size > 0, path

rows = json.load(source_path.open(encoding="utf-8"))
if isinstance(rows, dict):
    rows = list(rows.values())
ids = [str(row["id"]) for row in rows]
assert len(rows) == len(set(ids)) == 500
assert set(ids) == {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
assert all(str(row.get("target_label", "")).strip().lower() in {"yes", "no", "maybe"} for row in rows)

metadata = json.load(metadata_path.open(encoding="utf-8"))
assert metadata.get("score_function") == "dot", metadata
assert metadata.get("model_code") == "contriever", metadata
retrieval = json.load(retrieval_path.open(encoding="utf-8"))
assert set(map(str, retrieval)) == set(ids)
assert all(len(candidates) >= 5 for candidates in retrieval.values())
print("input validation passed: PubMedQA=500, gpt4.1mini, exact GPT-5 mini r3 raw-dot inputs, no attack, all defenses disabled")
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
  test -s "$output" || return 1
  "$PYTHON" - "$output" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))[0]["iter_0"]
assert len(rows) == 500
assert len({str(row["id"]) for row in rows}) == 500
assert all(str(row.get("parsed_pred_label") or "").strip().lower() in {"yes", "no", "maybe"} for row in rows)
assert all(str(row.get("output_poison") or "").strip() for row in rows)
assert all(row.get("medical_kg_filter_enabled") is False for row in rows)
assert all(row.get("trustrag_filter_enabled") is False for row in rows)
assert all(row.get("medical_semantic_clustering_enabled") is False for row in rows)
assert all(row.get("judge_enabled") is False for row in rows)
PY
}

OUTPUT="results/query_results/${QUERY_DIR}/formal_dot_pubmedqa_contriever_gpt41mini_clean_no_defense_aligned_full_20260812.json"
LOG="$LOG_ROOT/pubmedqa_clean.log"

if [[ "${1:-}" == "--preflight" && $# -eq 1 ]]; then
  validate_inputs
  echo "$(date '+%F %T') preflight complete"
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--preflight]" >&2
  exit 2
fi

if test -e "$OUTPUT"; then
  if is_complete_output "$OUTPUT"; then
    echo "$(date '+%F %T') reuse completed $OUTPUT"
    exit 0
  fi
  echo "refusing to overwrite incomplete output: $OUTPUT" >&2
  exit 2
fi

validate_inputs
wait_for_memory
echo "$(date '+%F %T') starting aligned PubMedQA gpt4.1mini clean baseline on GPU${RUN_GPU}"
CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON" -u main.py \
  --eval_model_code contriever --eval_dataset pubmed --model_name "$MODEL_NAME" \
  --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$QUERY_DIR" \
  --attack_method None --adv_source json --adv_json_path "$SOURCE" \
  --target_ids_path "$IDS" --retrieval_results_path "$RETRIEVAL" \
  --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
  --asr_match_mode strict --no_medical_kg_filter \
  --name formal_dot_pubmedqa_contriever_gpt41mini_clean_no_defense_aligned_full_20260812 > "$LOG" 2>&1
is_complete_output "$OUTPUT"
echo "$(date '+%F %T') completed aligned PubMedQA gpt4.1mini clean baseline"
