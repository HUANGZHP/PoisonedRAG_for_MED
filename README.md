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

代码通过 `--eval_model_code` 选择检索器，通过 `model_configs/` 配置 LLM（见 2.5）。

#### 2.4.1 MedCPT（推荐，需手动下载）

```bash
# 下载 Query Encoder（~1GB）到 models/ncbi/MedCPT-Query-Encoder/
python download_medcpt_models.py

# 可选：同时下载 Article Encoder（构建索引用）
python download_medcpt_models.py --include-article-encoder

# 设置环境变量（非必需，自动发现路径）
export MEDCPT_QUERY_ENCODER_PATH=models/ncbi/MedCPT-Query-Encoder
```

下载完成后，目录结构应为：
```
models/ncbi/MedCPT-Query-Encoder/
├── config.json
├── pytorch_model.bin
└── tokenizer.json
```

MedCPT 还依赖本地 FAISS 索引（见 2.1 语料库下载），运行 `evaluate_beir.py` 时会自动从 `datasets/<名称>/index/` 加载。

#### 2.4.2 Contriever / DPR / ANCE（自动加载）

首次运行 `evaluate_beir.py` 或 `main.py` 时会自动从 HuggingFace 下载。如需本地缓存：

```bash
# 设置缓存目录（可选）
export HF_MODEL_ROOT=/path/to/model_cache

# 提前下载
python -c "
from src.utils import load_models
model, c_model, tokenizer, get_emb = load_models('contriever')
"
```

#### 2.4.3 BM25（依赖 pyserini）

首次运行时自动从索引加载。需确保语料库目录下有 BM25 索引（见 2.1）。

#### 2.4.4 验证检索器是否正常

```bash
python -c "
from src.utils import load_models
model, c_model, tokenizer, get_emb = load_models('medcpt')
print('MedCPT loaded successfully:', type(model).__name__)
"
```

### 2.5 LLM 模型

LLM 通过 JSON 配置文件驱动，位于 `model_configs/`。系统根据 `--model_name <名称>` 自动加载 `model_configs/<名称>_config.json`。

#### 2.5.1 配置文件格式

```json
{
  "model_info": {
    "provider": "gpt",           // gpt（在线）/ local（本地）
    "name": "gpt-4"              // 模型名称
  },
  "api_key_info": {
    "base_url": "https://api.chatanywhere.tech/v1",  // 在线 API 地址
    "api_keys": ["YOUR_API_KEY"],  // API key 列表
    "api_key_use": 0               // 当前使用的 key 索引
  },
  "params": {
    "temperature": 0.1,
    "seed": 100,
    "gpus": [],
    "max_output_tokens": 300
  }
}
```

#### 2.5.2 在线模型（如 GPT-4 / GPT-4.1）

**步骤 1：配置 API key**

方法 A — 环境变量（推荐，对已有配置文件生效）：
```bash
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.chatanywhere.tech/v1"
```

方法 B — 直接编辑 `model_configs/gpt4_config.json`，将 `api_keys` 中的 `YOUR_API_KEY` 替换为真实的 key。

**步骤 2：新增一个在线模型**

复制现有配置并修改：
```bash
# 以 gpt4_config.json 为模板
cp model_configs/gpt4_config.json model_configs/gpt4mini_config.json
```
编辑 `model_configs/gpt4mini_config.json`，将 `model_info.name` 改为实际的模型名（如 `gpt-4-mini`）。

**步骤 3：使用**
```bash
python main.py --model_name gpt4mini ...
```

#### 2.5.3 本地模型（如 Llama-3.1-8B-Instruct）

**步骤 1：从 HuggingFace 下载模型**

```bash
# 方式一：自动下载（首次运行自动缓存）
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
"

# 方式二：指定缓存目录
export HF_HOME=/path/to/hf_cache
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct
```

> 注意：使用 Llama 模型需要先通过 HuggingFace 授权（https://huggingface.co/meta-llama），并确保已下载所需模型。

**步骤 2：检查配置文件（以 Llama-7B 为例）**

`model_configs/llama7b_config.json` 示例：
```json
{
  "model_info": {
    "provider": "local",
    "name": "meta-llama/Llama-2-7b-chat-hf",
    "model_cls": "LlamaForCausalLM",
    "tokenizer_cls": "LlamaTokenizer"
  },
  "api_key_info": null,
  "params": {
    "temperature": 0.1,
    "max_output_tokens": 300,
    "gpus": [0],
    "load_in_8bit": false,
    "load_in_4bit": false
  }
}
```

关键字段说明：
- `provider`: 必须设为 `"local"`
- `name`: HuggingFace 上的模型名称或本地路径
- `gpus`: GPU 编号列表，`[]` 表示 CPU
- `load_in_8bit` / `load_in_4bit`: 量化选项，显存不足时可开启

**步骤 3：新增本地模型**

```bash
cp model_configs/llama7b_config.json model_configs/qwen2_config.json
```
编辑 `model_configs/qwen2_config.json`，将 `name` 改为 `"Qwen/Qwen2-7B-Instruct"`，并调整 `gpus` / 量化参数。

**步骤 4：使用**
```bash
# Llama-7B
python main.py --model_name llama7b ...

# Llama-13B
python main.py --model_name llama13b ...
```

#### 2.5.4 验证 LLM 是否正常

```bash
python -c "
from src.models import create_model
llm = create_model('model_configs/gpt4_config.json')
response = llm.query('Hello, say hi!')
print('Response:', response)
"
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
# ===== 1. 导出 MIRAGE queries =====
python -u gen_mirage_queries.py \
  --dataset pubmedqa \           # 可选: pubmedqa / medqa / medmcqa / mmlu / bioasq
  --limit 60 \                   # ← 导出问题数量，可改为 10 / 30 / 100
  --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60

# ===== 2. MedCPT 检索 =====
python -u evaluate_beir.py \
  --model_code medcpt \          # ← 可选: contriever / dpr / ance / bm25
  --dataset pubmed \             # ← 可选: statpearls / textbooks
  --split test \
  --queries-json results/adv_targeted_results/mirage_pubmedqa_n60.queries.json \
  --result_output results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --top_k 100 \
  --use_faiss_gpu True \         # ← FAISS GPU 加速，无 GPU 时设为 False
  --gpu_id 0

# ===== 3. 无攻击 =====
python -u main.py \
  --eval_model_code medcpt \
  --eval_dataset pubmed \
  --model_name gpt4 \            # ← 可选: gpt5mini / llama7b / llama13b 等
  --top_k 5 \                    # ← 每问题取前 k 篇文档
  --attack_method None \
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --M 60 \                       # ← 评估问题数，应与 limit 一致
  --name pubmed_pubmedqa_medcpt_gpt4_noattack

# ===== 4. 黑盒攻击 =====
python -u main.py \
  --eval_model_code medcpt \
  --eval_dataset pubmed \
  --model_name gpt4 \
  --top_k 5 \
  --attack_method LM_targeted \  # ← 可选: LM_targeted / hotflip
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --adv_per_query 5 \            # ← 每问题注入攻击文本数
  --M 60 \
  --name pubmed_pubmedqa_medcpt_gpt4_blackbox
```

### 示例 B：textbooks + MIRAGE medqa + medcpt + llama7b

```bash
# ===== 1. 导出 MIRAGE queries =====
python -u gen_mirage_queries.py \
  --dataset medqa \               # ← 可选: pubmedqa / bioasq / mmlu / medmcqa
  --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_medqa_n60

# ===== 2. MedCPT 检索 =====
python -u evaluate_beir.py \
  --model_code medcpt \
  --dataset textbooks \           # ← 语料库，可选: pubmed / statpearls
  --queries-json results/adv_targeted_results/mirage_medqa_n60.queries.json \
  --result_output results/beir_results/textbooks_medcpt_mirage_medqa_n60.json \
  --top_k 100 \
  --use_faiss_gpu True \
  --gpu_id 0

# ===== 3. 白盒攻击（hotflip） =====
python -u main.py \
  --eval_model_code medcpt \
  --eval_dataset textbooks \
  --model_name llama7b \              # ← 也可换为 llama13b
  --top_k 5 \
  --attack_method hotflip \            # ← 白盒攻击
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_medqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_n60.ids \
  --retrieval_results_path results/beir_results/textbooks_medcpt_mirage_medqa_n60.json \
  --adv_per_query 5 \
  --M 60 \
  --name textbooks_medqa_medcpt_llama7b_whitebox
```

### 示例 C：hotpotqa + 内置 queries + contriever + gpt4

```bash
# ===== 1. Contriever 检索（hotpotqa 为 BEIR 数据集，自动下载） =====
python -u evaluate_beir.py \
  --model_code contriever \       # ← 可选: dpr / ance / bm25
  --dataset hotpotqa \            # ← 可选: nq / msmarco
  --result_output results/beir_results/hotpotqa_contriever.json \
  --top_k 100 \
  --gpu_id 0

# ===== 2. 用 gpt4 生成攻击文本（需要 qrels，BEIR 内置） =====
python -u gen_adv.py \
  --eval_model_code contriever \
  --eval_dataset hotpotqa \
  --model_name gpt4 \             # ← 生成攻击文本用的 LLM
  --adv_per_query 5 \
  --data_num 60 \                 # ← 生成攻击文本的问题数
  --save_path results/adv_targeted_results

# ===== 3. 黑盒攻击 =====
python -u main.py \
  --eval_model_code contriever \
  --eval_dataset hotpotqa \
  --model_name gpt4 \
  --top_k 5 \
  --attack_method LM_targeted \
  --adv_source json \             # ← json: 使用预生成的攻击文本
  --adv_json_path results/adv_targeted_results/hotpotqa.json \
  --retrieval_results_path results/beir_results/hotpotqa_contriever.json \
  --adv_per_query 5 \
  --M 60 \
  --name hotpotqa_contriever_gpt4_blackbox
```

## 6. 输出位置

- adv / ids / queries: results/adv_targeted_results/
- 检索结果: results/beir_results/
- 评测结果: results/query_results/<query_results_dir>/
- 日志: logs/<query_results_dir>_logs/


