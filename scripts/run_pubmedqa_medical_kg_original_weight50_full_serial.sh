#!/usr/bin/env bash
# Full PubMedQA re-run with an equal retrieval/safety weighting.  The old
# 0.20 run remains untouched; this queue writes to distinct result names.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
KG_ARTIFACT=datasets/medical_kg/bios_v3_preferred_sample_20260803
LOG_ROOT=logs/formal_dot_pubmedqa_medical_kg_original_weight50_full_20260804
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
artifact_dir = Path("datasets/medical_kg/bios_v3_preferred_sample_20260803")
for path in (retrieval_path, metadata_path, source_path, ids_path):
    assert path.is_file(), path
for name in ("metadata.json", "concepts.json", "relationships.json", "edges.npz", "concept_embeddings.npy", "relationship_embeddings.npy"):
    assert (artifact_dir / name).is_file(), artifact_dir / name
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
assert all(len(candidates) >= 10 for candidates in retrieval.values())
print("input validation passed: PubMedQA=500, Contriever/raw-dot, own-5, KG weight=0.50")
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
  local expects_trustrag=$2
  test -s "$output" || return 1
  "$PYTHON" - "$output" "$expects_trustrag" <<'PY'
import json
import sys

path, expects_trustrag = sys.argv[1:]
rows = json.load(open(path, encoding="utf-8"))[0]["iter_0"]
assert len(rows) == 500
assert len({str(row["id"]) for row in rows}) == 500
# Empty predictions are strict failures in ASR/accuracy, but do not make a
# 500-row evaluation incomplete or justify overwriting it.
assert all(str(row.get("parsed_pred_label") or "").strip().lower() in {"", "yes", "no", "maybe"} for row in rows)
assert all(row.get("medical_kg_filter_enabled") is True for row in rows)
assert all(row.get("medical_kg_filter_applied") is True for row in rows)
assert all(row.get("medical_kg_mode") == "original" for row in rows)
assert all(abs(float(row.get("medical_kg_rerank_weight", -1.0)) - 0.50) < 1e-9 for row in rows)
assert all(bool(row.get("trustrag_filter_enabled")) == (expects_trustrag == "true") for row in rows)
PY
}

run_experiment() {
  local variant=$1
  local trustrag_flag=$2
  local query_dir=formal_dot_pubmedqa_medical_kg_original_weight50_full_20260804
  local name="formal_dot_pubmedqa_contriever_medical_kg_original_weight50_full_20260804_${variant}"
  local output="results/query_results/${query_dir}/${name}.json"
  local log="$LOG_ROOT/${variant}.log"
  if test -e "$output"; then
    if is_complete_output "$output" "$trustrag_flag"; then
      echo "$(date '+%F %T') reuse completed $output"
      return
    fi
    echo "refusing to overwrite incomplete output: $output" >&2
    return 2
  fi
  wait_for_memory
  echo "$(date '+%F %T') starting PubMedQA full medical-KG/${variant} on GPU4, weight=0.50"
  local extra_args=()
  if [[ "$trustrag_flag" == "true" ]]; then
    extra_args+=(--trustrag_filter)
  fi
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$query_dir" \
    --attack_method LM_targeted --adv_source json --adv_json_path "$SOURCE" \
    --target_ids_path "$IDS" --retrieval_results_path "$RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --medical_kg_filter --medical_kg_mode original \
    --medical_kg_artifact_dir "$KG_ARTIFACT" --medical_kg_candidate_k 10 \
    --medical_kg_rerank_weight 0.50 --name "$name" "${extra_args[@]}" > "$log" 2>&1
  is_complete_output "$output" "$trustrag_flag"
  echo "$(date '+%F %T') completed PubMedQA full medical-KG/${variant}, weight=0.50"
}

validate_inputs
run_experiment kg_only false
run_experiment kg_plus_trustrag true
echo "$(date '+%F %T') queue complete"
