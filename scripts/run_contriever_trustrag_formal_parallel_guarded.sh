#!/usr/bin/env bash
set -euo pipefail
cd /home/huangzhp53/PoisonedRAG
PYTHON=/home/huangzhp53/miniconda3/envs/PoisonedRAG/bin/python
ROOT=logs/formal_dot_contriever_defenses
RETR=results/beir_results/mirage_pubmedqa_all-contriever.json
START=$((280*1024*1024))
FALLBACK=$((220*1024*1024))
mkdir -p "$ROOT"

log(){ echo "$(date '+%F %T') $*"; }
mem(){ awk '/MemAvailable:/ {print $2}' /proc/meminfo; }
gpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v n="$1" '$1+0==n {gsub(/ /,"",$2);print $2;exit}'; }
wait_start(){
  while true; do
    a=$(mem); g1=$(gpu 1); g2=$(gpu 2)
    if (( a>=START && g1<=4096 && g2<=4096 )); then return; fi
    log "waiting: MemAvailable=$a KiB GPU1=$g1 MiB GPU2=$g2 MiB"
    sleep 120
  done
}
wait_serial(){ while (( $(mem)<FALLBACK )); do log "waiting for serial memory"; sleep 120; done; }
out(){ printf 'results/query_results/formal_dot_contriever_%s/formal_dot_pubmedqa_contriever_%s_%s.json' "$1" "$1" "$2"; }
run(){
  d=$1; attack=$2; physical=$3; o=$(out "$d" "$attack")
  test -s "$o" && return
  mkdir -p "$ROOT/$d"
  extra=
  test "$d" = trustrag_medcluster && extra=--medical_semantic_clustering
  log "starting $d/$attack GPU$physical"
  CUDA_VISIBLE_DEVICES=$physical "$PYTHON" -u main.py --eval_model_code contriever --eval_dataset pubmed --model_name gpt4.1mini --judge_model_name None --top_k 5 --gpu_id 0 --query_results_dir formal_dot_contriever_${d} --attack_method "$attack" --adv_source json --adv_json_path results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json --target_ids_path results/adv_targeted_results/mirage_pubmedqa_all.ids --retrieval_results_path "$RETR" --score_function dot --adv_per_query 5 --M 500 --repeat_times 1 --asr_match_mode strict --trustrag_filter $extra --name formal_dot_pubmedqa_contriever_${d}_${attack} > "$ROOT/$d/$attack.log" 2>&1
}
serial(){
  log "falling back to serial for $1"
  wait_serial; run trustrag "$1" 1
  wait_serial; run trustrag_medcluster "$1" 2
}
pair(){
  attack=$1; aout=$(out trustrag "$attack"); bout=$(out trustrag_medcluster "$attack")
  if test -s "$aout" && test -s "$bout"; then return; fi
  wait_start
  run trustrag "$attack" 1 & p1=$!
  run trustrag_medcluster "$attack" 2 & p2=$!
  fallback=0
  while kill -0 "$p1" 2>/dev/null || kill -0 "$p2" 2>/dev/null; do
    a=$(mem)
    if (( a<FALLBACK )); then
      log "memory guard tripped ($a KiB); ending pair and using serial"
      kill -TERM "$p1" "$p2" 2>/dev/null || true
      fallback=1
      break
    fi
    sleep 60
  done
  wait "$p1" || true
  wait "$p2" || true
  if (( fallback )); then serial "$attack"; fi
  test -s "$aout" && test -s "$bout"
  log "completed $attack"
}
for attack in None LM_targeted hotflip; do pair "$attack"; done
log "guarded parallel suites complete"
