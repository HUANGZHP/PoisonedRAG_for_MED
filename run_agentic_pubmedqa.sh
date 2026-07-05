#!/bin/bash
# 等待检索结果文件生成
while [ ! -f /home/huangzhp53/PoisonedRAG/results/beir_results/pubmed_contriever_mirage_pubmedqa_n60.json ]; do
  sleep 30
done
sleep 60  # 确保文件写入完成

cd /home/huangzhp53/PoisonedRAG

nohup python -u agentic_main.py \
  --eval_model_code contriever \
  --eval_dataset pubmed \
  --model_name gpt4.1mini \
  --top_k 5 \
  --attack_method None \
  --agentic_rag \
  --judge_model_name gpt4.1mini \
  --rag_max_turns 3 \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_contriever_mirage_pubmedqa_n60.json \
  --M 60 \
  --name pubmed_contriever_gpt41mini_agentic_judge_norid \
  > logs/user_runs_logs/pubmed_contriever_gpt41mini_agentic_judge_norid.out 2>&1 &

echo "Agentic RAG PID: $!"
