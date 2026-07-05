#!/bin/bash
# 等待 n60 实验全部结束
while ps aux | grep -q "[a]gentic_main.*mirage_medqa_n60"; do
  sleep 60
done
sleep 30

cd /home/huangzhp53/PoisonedRAG

echo "===== n60 全部完成，启动全量 MedQA (1273) ====="

# 无攻击 (GPU 4)
nohup python -u agentic_main.py \
  --eval_model_code contriever --eval_dataset pubmed \
  --model_name gpt4.1mini --top_k 5 --attack_method None \
  --agentic_rag --judge_model_name gpt4.1mini --rag_max_turns 3 \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_medqa_all.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_all.ids \
  --retrieval_results_path results/beir_results/mirage_medqa_all-contriever.json \
  --M 1273 --gpu_id 4 \
  --name pubmed_contriever_gpt41mini_agentic_noattack_full \
  > logs/user_runs_logs/pubmed_gpt41mini_agentic_noattack_full.out 2>&1 &
echo "无攻击 full PID: $!"

# 黑盒 (GPU 5)
nohup python -u agentic_main.py \
  --eval_model_code contriever --eval_dataset pubmed \
  --model_name gpt4.1mini --top_k 5 --attack_method LM_targeted \
  --agentic_rag --judge_model_name gpt4.1mini --rag_max_turns 3 \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_medqa_all.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_all.ids \
  --retrieval_results_path results/beir_results/mirage_medqa_all-contriever.json \
  --adv_per_query 5 --M 1273 --gpu_id 5 \
  --name pubmed_contriever_gpt41mini_agentic_blackbox_full \
  > logs/user_runs_logs/pubmed_gpt41mini_agentic_blackbox_full.out 2>&1 &
echo "黑盒 full PID: $!"

# 白盒 (GPU 6)
nohup python -u agentic_main.py \
  --eval_model_code contriever --eval_dataset pubmed \
  --model_name gpt4.1mini --top_k 5 --attack_method hotflip \
  --agentic_rag --judge_model_name gpt4.1mini --rag_max_turns 3 \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_medqa_all.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_all.ids \
  --retrieval_results_path results/beir_results/mirage_medqa_all-contriever.json \
  --adv_per_query 5 --M 1273 --gpu_id 6 \
  --name pubmed_contriever_gpt41mini_agentic_whitebox_full \
  > logs/user_runs_logs/pubmed_gpt41mini_agentic_whitebox_full.out 2>&1 &
echo "白盒 full PID: $!"

echo "全部已提交"
