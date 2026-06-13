# PoisonedRAG_for_MED

该项目用于评估检索增强生成（RAG）在医疗场景下受对抗注入（知识库投毒）影响的程度与鲁棒性。

## 核心功能

| 功能 | 说明 |
|------|------|
| **无攻击** | 标准 RAG，不注入对抗文本 |
| **黑盒攻击** (LM_targeted) | 用 LLM 生成伪科学文章注入检索结果前列 |
| **白盒攻击** (hotflip) | 利用检索器梯度逐 token 替换字符 |
| **裁判过滤** (Judge) | 额外 LLM 检测对抗文本并拦截 |
| **Agentic RAG** (Search-o1) | 多轮「搜索→推理→再搜索」循环 |
| **Reason-in-Docs** (RiD) | 对检索文档逐篇分析，提取有用信息 |
| **RiD 防御模式** | 严格相关性过滤，丢弃不相关/误导性文档 |

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

代码通过 `--eval_model_code` 选择检索器，支持以下选项：
- **medcpt**: 医学专用稠密检索（推荐，需手动下载模型和索引）
- **contriever**: 通用稠密检索（自动从 HuggingFace 加载）
- **contriever-msmarco**: MS MARCO 微调版 Contriever
- **contriever-chinese**: 中文场景优化的 Contriever
- **ance**: Approximate Nearest Neighbor Negative Contrastive Estimation
- **dpr**: Dense Passage Retrieval
- **bm25**: 稀疏检索（依赖 pyserini）

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
- Query 来源: MIRAGE 导出 / BEIR 内置 / 自定义（adv_json）
- Retriever: medcpt / contriever / dpr / ance / bm25
- LLM: model_configs/<name>_config.json
- **Agentic RAG**: 支持 `--agentic_rag` / `--reason_in_docs` / `--rid_defense`
- **裁判过滤**: 支持 `--judge_model_name` / `--judge_model_config_path`

## 4. 完整流程

### 4.1 准备 queries

#### 4.1.1 MIRAGE 导出（推荐用于 MedRAG corpus）

```bash
python -u gen_mirage_queries.py --dataset pubmedqa --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60
```

产物：`*.json` / `*.ids` / `*.queries.json`

#### 4.1.2 医学课程选择题解析（生成对抗文本的 query 来源）

```bash
# 解析 病程单选题-DeepSeek.docx → adv_json + ids + queries
python parse_course_queries.py
```

产物：
- `results/adv_targeted_results/course_exam.json`（直接可用作 `--adv_json_path`）
- `results/adv_targeted_results/course_exam.ids`
- `results/adv_targeted_results/course_exam.queries.json`

### 4.2 生成检索结果

```bash
python -u evaluate_beir.py \
  --model_code <medcpt|contriever|dpr|ance|bm25> \
  --dataset <pubmed|statpearls|textbooks|hotpotqa|nq|msmarco> \
  --split test \
  --queries-json <path/to/queries.json> \
  --result_output <path/to/retrieval.json> \
  --top_k 100 --gpu_id 0
```

### 4.3 生成攻击文本

#### 4.3.1 标准 BEIR 数据集（用于 hotpotqa / nq / msmarco）

```bash
python gen_adv.py \
  --eval_model_code contriever \
  --eval_dataset hotpotqa \
  --model_name gpt4 \
  --adv_per_query 5 --data_num 60 \
  --save_path results/adv_targeted_results
```

#### 4.3.2 MCQ 定向攻击（用于 MIRAGE medqa 等选择题子集）

将完整题干 + 选项输入 LLM，明确指向某个错误选项：

```bash
python gen_adv_for_mcq.py \
  --model_name gpt4.1mini \
  --benchmark_path MIRAGE/benchmark.json \
  --mirage_dataset medqa \
  --concurrency 10 \
  --output results/adv_targeted_results/mirage_medqa_mcq.json
```

### 4.4 运行 main（标准攻击评测）

```bash
# 无攻击
python -u main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 --attack_method None \
  --retrieval_results_path <retrieval.json> \
  --M <N> --name <run_name>

# 攻击（黑/白盒）
python -u main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 \
  --attack_method <LM_targeted|hotflip> \
  --adv_source <corpus|json> \
  --adv_json_path <adv.json> --target_ids_path <ids.txt> \
  --retrieval_results_path <retrieval.json> \
  --adv_per_query 5 --M <N> --name <run_name>

# 攻击 + 裁判过滤
python -u main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 \
  --attack_method LM_targeted \
  --judge_model_name gpt4.1mini \              # ← 启用裁判
  --adv_source json \
  --adv_json_path <adv.json> \
  --target_ids_path <ids.txt> \
  --retrieval_results_path <retrieval.json> \
  --adv_per_query 5 --M <N> --name <run_name>
```

### 4.5 Agentic RAG（Search-o1 风格）

支持多轮「搜索→推理→再搜索」循环，LLM 通过特殊标记 `<|begin_search_query|>...<|end_search_query|>` 自主触发检索。

```bash
# Agentic RAG（无攻击，Reason-in-Docs + 裁判过滤）
python -u agentic_main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 \
  --attack_method None \
  --agentic_rag \                               # ← 启用 Agentic RAG 循环
  --reason_in_docs \                             # ← 启用 RiD 逐篇分析
  --rag_max_turns 3 \                            # ← 最大搜索轮次
  --rag_verbose \                                # ← 打印详细过程
  --judge_model_name gpt4.1mini \                # ← 启用裁判过滤
  --retrieval_results_path <retrieval.json> \
  --name <run_name>

# Agentic RAG + 防御模式（严格相关性过滤）
python -u agentic_main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 \
  --attack_method None \
  --agentic_rag \
  --reason_in_docs \
  --rid_defense \                                # ← 防御模式
  --rag_max_turns 3 \
  --retrieval_results_path <retrieval.json> \
  --name <run_name>

# 仅 Reason-in-Docs（无 Agentic 循环）
python -u agentic_main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 \
  --attack_method None \
  --reason_in_docs \                             # ← 只启用 RiD
  --rid_defense \
  --retrieval_results_path <retrieval.json> \
  --name <run_name>

# Agentic RAG + 攻击（黑盒）
python -u agentic_main.py \
  --eval_dataset <corpus> --eval_model_code <retriever> \
  --model_name <llm> --top_k 5 \
  --attack_method LM_targeted \
  --agentic_rag --reason_in_docs \
  --judge_model_name gpt4.1mini \
  --adv_source json \
  --adv_json_path <adv.json> \
  --target_ids_path <ids.txt> \
  --retrieval_results_path <retrieval.json> \
  --adv_per_query 5 --M <N> --name <run_name>
```

> **参数说明**: `agentic_main.py` 完全兼容 `main.py` 的所有参数（`--adv_source`、`--adv_json_path`、`--target_ids_path` 等），同时新增以下参数：
> - `--agentic_rag`：启用搜索-推理循环
> - `--reason_in_docs`：启用 Reason-in-Documents 模块
> - `--rid_defense`：启用严格相关性过滤（防御模式）
> - `--rag_max_turns`：最大搜索轮次（默认 3）
> - `--rag_verbose`：打印详细过程
> - `--judge_model_name`：裁判 LLM 名称
> - `--judge_model_config_path`：裁判 LLM 配置路径
## 5. 组合示例

### 示例 A：pubmed + MIRAGE pubmedqa + medcpt + gpt4（标准流程）

```bash
# ===== 1. 导出 MIRAGE queries =====
python -u gen_mirage_queries.py --dataset pubmedqa --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60

# ===== 2. MedCPT 检索 =====
python -u evaluate_beir.py --model_code medcpt --dataset pubmed \
  --queries-json results/adv_targeted_results/mirage_pubmedqa_n60.queries.json \
  --result_output results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --top_k 100 --use_faiss_gpu True --gpu_id 0

# ===== 3. 无攻击 =====
python -u main.py \
  --eval_model_code medcpt --eval_dataset pubmed \
  --model_name gpt4 --top_k 5 --attack_method None \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --M 60 --name pubmed_pubmedqa_medcpt_gpt4_noattack

# ===== 4. 黑盒攻击 =====
python -u main.py \
  --eval_model_code medcpt --eval_dataset pubmed \
  --model_name gpt4 --top_k 5 --attack_method LM_targeted \
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --adv_per_query 5 --M 60 \
  --name pubmed_pubmedqa_medcpt_gpt4_blackbox

# ===== 5. 黑盒攻击 + 裁判过滤 =====
python -u main.py \
  --eval_model_code medcpt --eval_dataset pubmed \
  --model_name gpt4 --top_k 5 --attack_method LM_targeted \
  --judge_model_name gpt4.1mini \                    # ← 启用裁判 LLM
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --adv_per_query 5 --M 60 \
  --name pubmed_pubmedqa_medcpt_gpt4_judge
```

### 示例 B：textbooks + MIRAGE medqa + medcpt + llama7b（白盒）

```bash
# ===== 1. 导出 MIRAGE queries =====
python -u gen_mirage_queries.py --dataset medqa --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_medqa_n60

# ===== 2. MedCPT 检索 =====
python -u evaluate_beir.py --model_code medcpt --dataset textbooks \
  --queries-json results/adv_targeted_results/mirage_medqa_n60.queries.json \
  --result_output results/beir_results/textbooks_medcpt_mirage_medqa_n60.json \
  --top_k 100 --use_faiss_gpu True --gpu_id 0

# ===== 3. 白盒攻击 =====
python -u main.py \
  --eval_model_code medcpt --eval_dataset textbooks \
  --model_name llama7b --top_k 5 --attack_method hotflip \
  --adv_source corpus \
  --adv_json_path results/adv_targeted_results/mirage_medqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_n60.ids \
  --retrieval_results_path results/beir_results/textbooks_medcpt_mirage_medqa_n60.json \
  --adv_per_query 5 --M 60 \
  --name textbooks_medqa_medcpt_llama7b_whitebox
```

### 示例 C：hotpotqa + contriever + gpt4（BEIR 数据集，内置 queries）

```bash
# ===== 1. Contriever 检索（hotpotqa 自动下载） =====
python -u evaluate_beir.py --model_code contriever --dataset hotpotqa \
  --result_output results/beir_results/hotpotqa_contriever.json \
  --top_k 100 --gpu_id 0

# ===== 2. 用 gpt4 生成攻击文本 =====
python -u gen_adv.py \
  --eval_model_code contriever --eval_dataset hotpotqa \
  --model_name gpt4 --adv_per_query 5 --data_num 60 \
  --save_path results/adv_targeted_results

# ===== 3. 黑盒攻击 =====
python -u main.py \
  --eval_model_code contriever --eval_dataset hotpotqa \
  --model_name gpt4 --top_k 5 --attack_method LM_targeted \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/hotpotqa.json \
  --retrieval_results_path results/beir_results/hotpotqa_contriever.json \
  --adv_per_query 5 --M 60 \
  --name hotpotqa_contriever_gpt4_blackbox
```

### 示例 D：MCQ 定向攻击 + medqa + gpt4.1mini

```bash
# ===== 1. 生成 MCQ 对抗文本（题干+选项整体输入） =====
python gen_adv_for_mcq.py \
  --model_name gpt4.1mini \
  --benchmark_path MIRAGE/benchmark.json \
  --mirage_dataset medqa \
  --concurrency 10 \
  --output results/adv_targeted_results/mirage_medqa_mcq.json

# ===== 2. Contriever 检索 =====
python -u evaluate_beir.py --model_code contriever --dataset pubmed \
  --queries-json results/adv_targeted_results/mirage_medqa_mcq.queries.json \
  --result_output results/beir_results/pubmed_contriever_medqa_mcq.json \
  --top_k 100 --gpu_id 0

# ===== 3. 黑盒攻击 =====
python -u main.py \
  --eval_model_code contriever --eval_dataset pubmed \
  --model_name gpt4.1mini --top_k 5 \
  --attack_method LM_targeted --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_medqa_mcq.json \
  --target_ids_path results/adv_targeted_results/mirage_medqa_mcq.ids \
  --retrieval_results_path results/beir_results/pubmed_contriever_medqa_mcq.json \
  --adv_per_query 5 --M <N> \
  --name pubmed_medqa_gpt41mini_blackbox_mcq
```

### 示例 F：使用裁判模型进行过滤（Judge Defense）

裁判 LLM 参数适用于 `main.py` 和 `agentic_main.py`：

```bash
# main.py + 裁判
python -u main.py \
  --eval_model_code contriever --eval_dataset pubmed \
  --model_name gpt4 --top_k 5 --attack_method LM_targeted \
  --judge_model_name gpt4.1mini \                 # ← 裁判模型名称
  --judge_model_config_path model_configs/gpt4.1mini_config.json \  # ← 或指定配置路径
  --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --M 60 --name pubmed_judge_test

# agentic_main.py + 裁判 + RiD 防御
python -u agentic_main.py \
  --eval_model_code contriever --eval_dataset pubmed \
  --model_name gpt4 --top_k 5 --attack_method LM_targeted \
  --judge_model_name gpt4.1mini \
  --agentic_rag --reason_in_docs --rid_defense \
  --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_n60.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_n60.ids \
  --retrieval_results_path results/beir_results/pubmed_medcpt_mirage_pubmedqa_n60.json \
  --adv_per_query 5 --M 60 --name pubmed_agentic_judge_defense
```

## 6. 输出位置

- adv / ids / queries: results/adv_targeted_results/
- 检索结果: results/beir_results/
- 评测结果: results/query_results/<query_results_dir>/
- 日志: logs/<query_results_dir>_logs/


