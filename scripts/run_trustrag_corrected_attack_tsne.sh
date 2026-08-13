#!/usr/bin/env bash
# Wait for the audited corrected TrustRAG attacks, then create T-SNE plots.
set -euo pipefail

cd /home/huangzhp53/PoisonedRAG

PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
LOG_DIR=logs/trustrag_corrected_attack_tsne
PID_FILE="$LOG_DIR/queue.pid"
MIN_AVAILABLE_KIB=$((220 * 1024 * 1024))
RESULTS=(
  results/query_results/formal_dot_contriever_trustrag_medcluster_corrected/formal_dot_pubmedqa_contriever_trustrag_medcluster_corrected_LM_targeted.json
  results/query_results/formal_dot_contriever_trustrag_corrected/formal_dot_pubmedqa_contriever_trustrag_corrected_hotflip.json
  results/query_results/formal_dot_contriever_trustrag_medcluster_corrected/formal_dot_pubmedqa_contriever_trustrag_medcluster_corrected_hotflip.json
)

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$PID_FILE"

complete_audited_result() {
  local path=$1
  test -s "$path" || return 1
  "$PYTHON" -c '
import json
import sys

payload = json.load(open(sys.argv[1]))
rows = payload[0]["iter_0"]
required = {
    "own5_adv", "pre_filter_adv", "post_trustrag_adv",
    "post_medical_cluster_adv", "trustrag_removed_adv_count",
    "medical_semantic_clustering_removed_adv_count", "final_adv_count",
}
assert len(rows) == 500
assert len({row["id"] for row in rows}) == 500
assert all(required.issubset(row) for row in rows)
' "$path"
}

while true; do
  ready=1
  for result in "${RESULTS[@]}"; do
    if ! complete_audited_result "$result"; then
      ready=0
      break
    fi
  done
  if (( ready )); then
    break
  fi
  echo "$(date '+%F %T') waiting for all three audited attack outputs"
  sleep 300
done

while true; do
  available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available_kib >= MIN_AVAILABLE_KIB )); then
    break
  fi
  echo "$(date '+%F %T') waiting: MemAvailable=${available_kib}KiB, need=${MIN_AVAILABLE_KIB}KiB"
  sleep 300
done

echo "$(date '+%F %T') starting audited TrustRAG attack T-SNE on GPU6"
CUDA_VISIBLE_DEVICES=6 "$PYTHON" -u scripts/visualize_trustrag_attack_tsne.py \
  --gpu 0 --per-category 3 --normal-k 10 --seed 12 --max-length 512 \
  --result-json "${RESULTS[0]}" \
  --result-json "${RESULTS[1]}" \
  --result-json "${RESULTS[2]}" \
  > "$LOG_DIR/visualize.log" 2>&1
echo "$(date '+%F %T') T-SNE visualization completed"
