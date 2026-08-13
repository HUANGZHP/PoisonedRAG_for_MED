#!/usr/bin/env bash
# PubMedQA complete black-box/white-box comparison with clinical-priority BIOS KG.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
RETRIEVAL=results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json
SOURCE=results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json
IDS=results/adv_targeted_results/mirage_pubmedqa_all.ids
KG_ARTIFACT=datasets/medical_kg/bios_v3_clinical_priority_20260812
SHARED_CACHE=results/medical_kg_caches/mirage_pubmedqa_all_gpt41mini_contriever_triplets.jsonl
QUERY_DIR=formal_dot_pubmedqa_clinical_priority_kg_original_hardfilter_blackwhite_full_20260812
LOG_ROOT=logs/formal_dot_pubmedqa_clinical_priority_kg_original_hardfilter_blackwhite_full_20260812
RUN_GPU=0
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
artifact_dir = Path("datasets/medical_kg/bios_v3_clinical_priority_20260812")
cache_path = Path("results/medical_kg_caches/mirage_pubmedqa_all_gpt41mini_contriever_triplets.jsonl")

for path in (retrieval_path, metadata_path, source_path, ids_path, cache_path):
    assert path.is_file() and path.stat().st_size > 0, path
for name in ("metadata.json", "concepts.json", "relationships.json", "edges.npz", "concept_embeddings.npy", "relationship_embeddings.npy"):
    assert (artifact_dir / name).is_file(), artifact_dir / name

artifact = json.load((artifact_dir / "metadata.json").open(encoding="utf-8"))
assert artifact.get("sampling_policy") == "clinical-priority", artifact
assert artifact.get("non_strict_relations") == ["associated with"], artifact
assert int(artifact.get("edge_count", 0)) > 0, artifact
assert int(artifact.get("effective_edge_counts", {}).get("is a", 0)) <= 150_000, artifact

rows = json.load(source_path.open(encoding="utf-8"))
if isinstance(rows, dict):
    rows = list(rows.values())
ids = [str(row["id"]) for row in rows]
assert len(rows) == len(set(ids)) == 500
assert set(ids) == {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
assert all(str(row.get("target_label", "")).strip().lower() in {"yes", "no", "maybe"} for row in rows)
assert all(len(row.get("adv_texts") or []) == 5 for row in rows)

retrieval_metadata = json.load(metadata_path.open(encoding="utf-8"))
assert retrieval_metadata.get("score_function") == "dot", retrieval_metadata
assert retrieval_metadata.get("model_code") == "contriever", retrieval_metadata
retrieval = json.load(retrieval_path.open(encoding="utf-8"))
assert set(map(str, retrieval)) == set(ids)
assert all(len(candidates) >= 10 for candidates in retrieval.values())
print("input validation passed: PubMedQA=500, gpt4.1mini, Contriever/raw-dot, new clinical-priority KG, original hard-filter=1:0, no TrustRAG")
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
assert all(row.get("medical_kg_filter_enabled") is True for row in rows)
assert all(row.get("medical_kg_filter_applied") is True for row in rows)
assert all(row.get("medical_kg_mode") == "original" for row in rows)
assert all(row.get("medical_kg_decision_mode") == "hard_filter" for row in rows)
assert all(abs(float(row.get("medical_kg_hard_filter_threshold", -1.0)) - 1.0) < 1e-9 for row in rows)
assert all(row.get("trustrag_filter_enabled") is False for row in rows)
assert all(row.get("medical_semantic_clustering_enabled") is False for row in rows)
assert all(row.get("medical_kg_ignored_triplets") is not None for row in rows)
assert attack in {"LM_targeted", "hotflip"}
PY
}

run_experiment() {
  local attack=$1
  local variant=$2
  local name="formal_dot_pubmedqa_contriever_clinical_priority_kg_original_hardfilter_${variant}_full_20260812"
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
  echo "$(date '+%F %T') starting PubMedQA clinical-priority KG hard-filter/${variant} on GPU${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON" -u main.py \
    --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini \
    --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir "$QUERY_DIR" \
    --attack_method "$attack" --adv_source json --adv_json_path "$SOURCE" \
    --target_ids_path "$IDS" --retrieval_results_path "$RETRIEVAL" \
    --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --seed 12 \
    --asr_match_mode strict --medical_kg_filter --medical_kg_mode original \
    --medical_kg_decision_mode hard_filter \
    --medical_kg_hard_filter_threshold 1.0 --medical_kg_rerank_weight 1.0 \
    --medical_kg_artifact_dir "$KG_ARTIFACT" --medical_kg_candidate_k 10 \
    --medical_kg_triplet_cache_path "$SHARED_CACHE" \
    --medical_kg_triplet_cache_namespace triplet_schema_v1 \
    --name "$name" > "$log" 2>&1
  is_complete_output "$output" "$attack"
  echo "$(date '+%F %T') completed PubMedQA clinical-priority KG hard-filter/${variant}"
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
