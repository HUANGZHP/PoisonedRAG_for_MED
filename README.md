# PoisonedRAG_for_MED

该项目用于评估检索增强生成（RAG）在医疗场景下受对抗注入（知识库投毒）影响的程度与鲁棒性。支持无攻击、黑盒（LM_targeted）、白盒（hotflip）三种攻击方法以及可选裁判（Judge）过滤。

## 1. 环境与依赖

```bash
conda create -n PoisonedRAG_for_Med python=3.10 -y
conda activate PoisonedRAG_for_Med
pip install -r requirements.txt
```

在线模型（如 gpt4）请配置：

```bash
export OPENAI_API_KEY="<your_key>"
export OPENAI_BASE_URL="<your_base_url>"
```

## 2. 数据与模型下载

### 2.1 MedRAG 语料库（pubmed / statpearls / textbooks）

三个语料库来自 [MedRAG 项目](https://github.com/Teddy-XiongGZ/MedRAG)，需下载 pre-chunked 文档与 FAISS 索引后放入 `datasets/<名称>/`。

**方式一：通过 MedRAG 脚本下载**
```bash
git clone https://github.com/Teddy-XiongGZ/MedRAG.git /tmp/MedRAG
cd /tmp/MedRAG
python -c "
from medrag.utils import download_dataset
download_dataset('pubmed', db_dir='/path/to/PoisonedRAG/datasets')
"
```

**方式二：手动下载后解压**
从 [MedRAG 发布页](https://github.com/Teddy-XiongGZ/MedRAG#corpora) 获取语料包，解压至：
```
datasets/pubmed/
├── chunk/          # 分块文档（.jsonl）
├── index/          # FAISS / BM25 索引
```

> 注意：pubmed 语料约 60GB，下载解压耗时较长。

### 2.2 BEIR 数据集（hotpotqa / nq / msmarco）

无需手动下载。运行 `evaluate_beir.py` 或 `main.py` 时自动下载并解压至 `datasets/<名称>/`。

### 2.3 MIRAGE 评测集

已内置 `MIRAGE/benchmark.json`，包含 pubmedqa / medqa / medmcqa / mmlu / bioasq 五个子集。

### 2.4 检索器模型

| 检索器 | 获取方式 |
|--------|----------|
| MedCPT | `python download_medcpt_models.py` |
| Contriever | 首次运行自动从 HuggingFace 加载 |
| DPR / ANCE / BM25 | 同上，自动加载 |

```bash
# 下载 MedCPT Query Encoder（~1GB）
python download_medcpt_models.py

# 设置本地路径
export MEDCPT_QUERY_ENCODER_PATH=models/ncbi/MedCPT-Query-Encoder
```

### 2.5 LLM 模型

配置文件位于 `model_configs/`。在线模型需配 API key，本地模型（llama 等）需提前缓存：

```bash
export HF_MODEL_ROOT=/path/to/cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')"
```

## 3. 组合速览

- Corpus: pubmed / statpearls / textbooks（MedRAG），hotpotqa / nq / msmarco（BEIR）
- Query 来源: MIRAGE 导出 / BEIR 内置 / 自定义
- Retriever: medcpt / contriever / dpr / ance / bm25
- LLM: model_configs/<name>_config.json

## 4. 完整流程

### 4.1 准备 queries

```bash
# MIRAGE 导出（推荐用于 MedRAG corpus）
python -u gen_mirage_queries.py   --dataset pubmedqa --limit 60   --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60
```

产物：`*.json` / `*.ids` / `*.queries.json`

BEIR 内置 queries 无需生成；自定义 queries 需准备 adv_json。

### 4.2 生成检索结果

```bash
python -u evaluate_beir.py   --model_code <medcpt|contriever|dpr|ance|bm25>   --dataset <pubmed|statpearls|textbooks|hotpotqa|nq|msmarco>   --split test   --queries-json <path/to/queries.json>   --result_output <path/to/retrieval.json>   --top_k 100   --gpu_id 0
```

### 4.3 运行 main

```bash
# 无攻击
python -u main.py   --eval_dataset <corpus> --eval_model_code <retriever>   --model_name <llm> --top_k 5 --attack_method None   --retrieval_results_path <retrieval.json>   --M <N> --name <run_name>

# 攻击（黑/白盒）
python -u main.py   --eval_dataset <corpus> --eval_model_code <retriever>   --model_name <llm> --top_k 5   --attack_method <LM_targeted|hotflip>   --adv_source <corpus|json>   --adv_json_path <adv.json> --target_ids_path <ids.txt>   --retrieval_results_path <retrieval.json>   --adv_per_query 5 --M <N> --name <run_name>
```

### 4.4 可选：生成攻击文本

```bash
python -u gen_adv.py   --eval_model_code contriever --eval_dataset hotpotqa   --model_name gpt4 --adv_per_query 5 --data_num 60   --save_path results/adv_targeted_results
```
## 5. 组合示例

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

## 6. 输出位置

- adv / ids / queries: results/adv_targeted_results/
- 检索结果: results/beir_results/
- 评测结果: results/query_results/<query_results_dir>/
- 日志: logs/<query_results_dir>_logs/


