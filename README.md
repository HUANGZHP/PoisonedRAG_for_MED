# PoisonedRAG_for_MED
该项目用于评估RAG知识库投毒在医疗方面的影响

## 1. 环境与依赖

```bash
pip install -r requirements.txt
conda activate PoisonedRAG
```

在线模型（如 gpt4）请配置：

```bash
export OPENAI_API_KEY="<your_key>"
export OPENAI_BASE_URL="<your_base_url>"
```

## 2. 组合速览

- Corpus: pubmed / statpearls / textbooks（本地 MedRAG chunk/index），hotpotqa / nq / msmarco（BEIR，自动下载）
- Query 来源: MIRAGE 导出、BEIR 内置、或自定义 queries json
- Retriever: medcpt / contriever / dpr / ance / bm25
- LLM: model_configs/<name>_config.json（如 gpt4、gpt5mini、llama31_8b_instruct）

## 3. 从头到尾流程

### 3.1 准备 queries / adv_json

**A. MIRAGE 导出（推荐用于 MedRAG corpus）**

```bash
python -u gen_mirage_queries.py \
  --dataset pubmedqa \
  --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60
```

产物：
- results/adv_targeted_results/mirage_pubmedqa_n60.json
- results/adv_targeted_results/mirage_pubmedqa_n60.ids
- results/adv_targeted_results/mirage_pubmedqa_n60.queries.json

**B. BEIR 内置 queries（如 hotpotqa / nq / msmarco）**

无需生成，直接进入检索步骤。

**C. 自定义 queries**

准备 adv json（最小字段）：

```json
{
  "q0": {
    "id": "q0",
    "question": "Your question here",
    "incorrect answer": "B",
    "adv_texts": []
  }
}
```

说明：
- 使用 `--adv_source corpus` 时需要 `incorrect answer`
- 使用 `--adv_source json` 时需要 `adv_texts` 非空

### 3.2 生成检索结果（retrieval_results_path）

模板：

```bash
python -u evaluate_beir.py \
  --model_code <medcpt|contriever|dpr|ance|bm25> \
  --dataset <pubmed|statpearls|textbooks|hotpotqa|nq|msmarco> \
  --split test \
  --queries-json <path/to/queries.json> \
  --result_output <path/to/retrieval.json> \
  --top_k 100 \
  --gpu_id 0
```

说明：
- 使用 MIRAGE 或自定义 queries 时，务必传 `--queries-json`
- medcpt 会优先使用本地 MedCPT index，可加 `--use_faiss_gpu True`
 - 若使用 BEIR 内置 queries，可省略 `--queries-json`

### 3.3 运行 main（No Attack / LM_targeted / hotflip）

模板（No Attack）：

```bash
python -u main.py \
  --eval_model_code <retriever> \
  --eval_dataset <corpus> \
  --split test \
  --query_results_dir user_runs \
  --model_name <llm_config_name> \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --attack_method None \
  --retrieval_results_path <retrieval.json> \
  --M <N> \
  --seed 12 \
  --asr_match_mode loose \
  --name <run_name>
```

模板（攻击）：

```bash
python -u main.py \
  --eval_model_code <retriever> \
  --eval_dataset <corpus> \
  --split test \
  --query_results_dir user_runs \
  --model_name <llm_config_name> \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --attack_method <LM_targeted|hotflip> \
  --adv_source <corpus|json> \
  --adv_json_path <adv.json> \
  --target_ids_path <ids.txt> \
  --retrieval_results_path <retrieval.json> \
  --adv_per_query 5 \
  --score_function dot \
  --repeat_times 1 \
  --M <N> \
  --seed 12 \
  --asr_match_mode loose \
  --name <run_name>
```

说明：
- 没有 ids 文件时，删掉 `--target_ids_path`
- 使用 `--adv_source corpus` 时必须提供 `--adv_json_path` 与 `--retrieval_results_path`

### 3.4 可选：生成攻击性文本 adv_json（BEIR + qrels）

作用：当没有现成的 adv_json 时，用 BEIR 的 queries + qrels 自动生成“错误答案 + 攻击文本”，供 `--adv_source json` 的攻击流程直接使用。

```bash
python -u gen_adv.py \
  --eval_model_code contriever \
  --eval_dataset hotpotqa \
  --split test \
  --model_name gpt4 \
  --adv_per_query 5 \
  --data_num 60 \
  --save_path results/adv_targeted_results
```

输出：results/adv_targeted_results/hotpotqa.json

## 4. 组合命令示例

### 示例 A：pubmed + MIRAGE pubmedqa + medcpt + gpt4

```bash
python -u gen_mirage_queries.py \
  --dataset pubmedqa \
  --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60

python -u evaluate_beir.py \
  --model_code medcpt \
  --dataset pubmed \
  --split test \
  --queries-json results/adv_targeted_results/mirage_pubmedqa_n60.queries.json \
  --result_output results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --top_k 100 \
  --use_faiss_gpu True \
  --gpu_id 0

python -u main.py \
  --eval_model_code medcpt \
  --eval_dataset pubmed \
  --split test \
  --query_results_dir user_runs \
  --model_name gpt4 \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --attack_method None \
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --adv_per_query 5 \
  --score_function dot \
  --repeat_times 1 \
  --M 60 \
  --seed 12 \
  --asr_match_mode loose \
  --name pubmed_pubmedqa_medcpt_gpt4_noattack

python -u main.py \
  --eval_model_code medcpt \
  --eval_dataset pubmed \
  --split test \
  --query_results_dir user_runs \
  --model_name gpt4 \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --attack_method LM_targeted \
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --adv_per_query 5 \
  --score_function dot \
  --repeat_times 1 \
  --M 60 \
  --seed 12 \
  --asr_match_mode loose \
  --name pubmed_pubmedqa_medcpt_gpt4_blackbox
```

### 示例 B：textbooks + MIRAGE medqa + medcpt + llama31_8b_instruct

```bash
python -u gen_mirage_queries.py \
  --dataset medqa \
  --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_medqa_n60

python -u evaluate_beir.py \
  --model_code medcpt \
  --dataset textbooks \
  --split test \
  --queries-json results/adv_targeted_results/mirage_medqa_n60.queries.json \
  --result_output results/beir_results/textbooks_medcpt_mirage_medqa_n60.json \
  --top_k 100 \
  --use_faiss_gpu True \
  --gpu_id 0

python -u main.py \
  --eval_model_code medcpt \
  --eval_dataset textbooks \
  --split test \
  --query_results_dir user_runs \
  --model_name llama31_8b_instruct \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --attack_method hotflip \
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_medqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_n60.ids \
  --retrieval_results_path results/beir_results/textbooks_medcpt_mirage_medqa_n60.json \
  --adv_per_query 5 \
  --score_function dot \
  --repeat_times 1 \
  --M 60 \
  --seed 12 \
  --asr_match_mode loose \
  --name textbooks_medqa_medcpt_llama31_whitebox
```

### 示例 C：hotpotqa + 内置 queries + contriever + gpt4

```bash
python -u evaluate_beir.py \
  --model_code contriever \
  --dataset hotpotqa \
  --split test \
  --result_output results/beir_results/hotpotqa_contriever.json \
  --top_k 100 \
  --gpu_id 0

python -u gen_adv.py \
  --eval_model_code contriever \
  --eval_dataset hotpotqa \
  --split test \
  --model_name gpt4 \
  --adv_per_query 5 \
  --data_num 60 \
  --save_path results/adv_targeted_results

python -u main.py \
  --eval_model_code contriever \
  --eval_dataset hotpotqa \
  --split test \
  --query_results_dir user_runs \
  --model_name gpt4 \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --attack_method LM_targeted \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/hotpotqa.json \
  --retrieval_results_path results/beir_results/hotpotqa_contriever.json \
  --adv_per_query 5 \
  --score_function dot \
  --repeat_times 1 \
  --M 60 \
  --seed 12 \
  --asr_match_mode loose \
  --name hotpotqa_contriever_gpt4_blackbox
```

## 5. 输出位置

- adv / ids / queries: results/adv_targeted_results/
- 检索结果: results/beir_results/
- 评测结果: results/query_results/<query_results_dir>/
- 日志: logs/<query_results_dir>_logs/


