# PoisonedRAG_for_MED

该项目用于评估检索增强生成（RAG）在医疗场景下受对抗注入（知识库投毒）影响的程度与鲁棒性。

## 核心功能

| 功能 | 说明 |
|------|------|
| **无攻击** | 标准 RAG，不注入对抗文本 |
| **黑盒攻击** (LM_targeted) | 用 LLM 生成伪科学文章注入检索结果前列 |
| **白盒攻击** (hotflip) | 利用检索器梯度逐 token 替换字符 |
| **裁判过滤** (Judge) | 额外 LLM 检测对抗文本并拦截 |
| **防御前后 F1 对比** | 启用法官时同时报告 pre-defense 和 post-defense 的注入成功率 |
| **8-bit 量化加载** | Qwen3/Llama 等本地模型支持 load_in_8bit / load_in_4bit，节省显存 |
| **Agentic RAG** (Search-o1) | 多轮「搜索→推理→再搜索」循环 |
| **Reason-in-Docs** (RiD) | 对检索文档逐篇分析，提取有用信息 |
| **RiD 防御模式** | 严格相关性过滤，丢弃不相关/误导性文档 |
| **BIOS 医学知识图谱防御** | 用 LLM 三元组、MedCPT 映射与 BIOS 图谱边验证为候选文档计风险并重排 |

## 1. 环境与依赖

```bash
conda create -n PoisonedRAG python=3.10 -y
conda activate PoisonedRAG
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
- **contriever_v1**: 以 PubMedQA 微调的 Contriever
- **contriever_v2**: 以 v1 为初始化、再以 MedQA 微调的 Contriever
- **contriever_v3**: PubMedQA 与 MedQA 1:1 replay 微调版
- **contriever_v4**: PQA-U 与 MedQA 1:1 微调版
- **contriever_v5**: 仅以 MedQA 训练的微调版
- **contriever_v6**: 剔除 MIRAGE PubMedQA 题目后，以 PQA-U 与 MedQA 1:1 微调的 v4 对照版
- **contriever-msmarco**: MS MARCO 微调版 Contriever
- **contriever-chinese**: 中文场景优化的 Contriever
- **contriever-chinese_v1**: 使用 v8 的 PQA-U/MedQA 8:8 replay 训练配方，在 `contriever-chinese` 基础模型上微调的独立语言迁移实验（训练数据为英文医学 QA）
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

微调版本默认加载 `checkpoint/contriever_v1/best_model` 或 `checkpoint/contriever_v2/best_model`；也可分别用 `CONTRIEVER_V1_PATH`、`CONTRIEVER_V2_PATH` 指定其他路径。

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


### 2.6 核心配置

- **Corpus**: pubmed / statpearls / textbooks（MedRAG），hotpotqa / nq / msmarco（BEIR）
- **Query** 来源: MIRAGE 导出 / BEIR 内置 / 自定义（adv_json）
- **Retriever**: medcpt / contriever / dpr / ance / bm25
- **LLM**: model_configs/<name>_config.json


## 3. 完整流程

### 3.1 导出 MIRAGE queries

```bash
python -u gen_mirage_queries.py --dataset pubmedqa --limit 60 \
  --output_prefix results/adv_targeted_results/mirage_pubmedqa_n60
```

产物：`*.json`（对抗文本标签）/ `*.ids`（query id）/ `*.queries.json`（检索用）

### 3.1.1 生成 Contriever 偏好微调数据

仅使用 PQA-L 的 `official_pqal/ori_pqal.json` 与 MedQA USMLE train。先设置 API key 和数据根目录：

```bash
export OPENAI_API_KEY="<your_api_key>"  # 不要提交 key
DATA_ROOT=/path/to/PubMedQA-and-MedQA
PQA_L="$DATA_ROOT/datasets/PubMedQA/official_pqal/ori_pqal.json"
MEDQA_USMLE="$DATA_ROOT/datasets/MedQA-USMLE/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl"
```

LLM 只生成后缀 `I`：Black-box 为 `query + I`，HotFlip 复用同一 I、只优化负例前缀；query 和 positive 不会改写。

```bash
# PubMedQA：Black-box → HotFlip → 校验
python -u scripts/build_blackbox.py --input "$PQA_L" --output processed/pubmedqa_blackbox.jsonl --dataset pubmedqa
CUDA_VISIBLE_DEVICES=1 python -u scripts/build_hotflip.py --input "$PQA_L" --blackbox-input processed/pubmedqa_blackbox.jsonl --output processed/pubmedqa_hotflip.jsonl --dataset pubmedqa --gpu 0
python scripts/validate_training_data.py --source "$PQA_L" --blackbox processed/pubmedqa_blackbox.jsonl --hotflip processed/pubmedqa_hotflip.jsonl --dataset pubmedqa

# MedQA：Black-box → HotFlip → 校验
python -u scripts/build_blackbox.py --input "$MEDQA_USMLE" --output processed/medqa_blackbox.jsonl --dataset medqa
CUDA_VISIBLE_DEVICES=1 python -u scripts/build_hotflip.py --input "$MEDQA_USMLE" --blackbox-input processed/medqa_blackbox.jsonl --output processed/medqa_hotflip.jsonl --dataset medqa --gpu 0
python scripts/validate_training_data.py --source "$MEDQA_USMLE" --blackbox processed/medqa_blackbox.jsonl --hotflip processed/medqa_hotflip.jsonl --dataset medqa
```

校验会检查来源、空文本、重复、`positive == negative`、两文件配对与 attack type；成功时输出 `checks: passed`。输出为 JSONL：

```json
{"query": "...", "positive": "...", "negative": "...", "attack_type": "blackbox"}
```

### 3.2 生成检索结果

```bash
python -u evaluate_beir.py \
  --model_code <medcpt|contriever|contriever_v1|contriever_v2|contriever_v3|contriever_v4|contriever_v5|contriever_v6|dpr|ance|bm25> \
  --dataset <pubmed|statpearls|textbooks|hotpotqa|nq|msmarco> \
  --split test --top_k 100 --score_function dot \
  --queries-json <path/to/queries.json> \
  --result_output <path/to/retrieval-dot.json>
```

experiment_config.py 中的 score_function 控制新实验的相似度，默认值为 dot（与原论文一致）；cosine 实验须显式指定 cos_sim。预检索和攻击评测必须使用同一个值；余弦预检索会额外生成对应的 .meta.json，评测会校验该文件，拒绝混用不同评分方式的候选分数。

### 3.3 生成攻击文本

```bash
# 黑盒攻击文本（LM_targeted）
python gen_adv.py \
  --eval_model_code contriever --eval_dataset <dataset> \
  --model_name gpt4 --adv_per_query 5 --data_num 60

# MCQ 定向攻击（题干+选项整体输入）
python gen_adv_for_mcq.py \
  --model_name gpt4.1mini --benchmark_path MIRAGE/benchmark.json \
  --mirage_dataset medqa --concurrency 10
```

### 3.4 实验方法

每个 query 的评测过程如下：先读取预生成的检索结果和该题专属的 5 篇对抗文本，按同一相似度定义与正常文档混排；随后执行可选防御，最后由生成模型回答。实验不在运行时重新生成黑盒攻击文本；HotFlip 会针对当前检索器生成前缀。

```text
Query + 预生成检索结果 + 预生成攻击文本
                    ↓
            按分数混排候选文档
                    ↓
  BIOS KG 风险重排 / TrustRAG / 医学语义聚类 / Judge / RiD（按配置）
                    ↓
            Agentic 或普通 RAG 生成
                    ↓
             严格 ASR 与检索 F1
```

攻击方法：

| 方法 | 配置值 | 实验含义 |
|------|------|------|
| 无攻击 | `attack_method=None` | 不注入恶意文本，用于正常能力基线。 |
| 黑盒 | `attack_method="LM_targeted"` | 注入 LLM 生成的、面向目标 query 的恶意上下文。 |
| 白盒 | `attack_method="hotflip"` | 使用与黑盒相同的恶意主体文本，并通过 HotFlip 优化前缀，使其更容易被目标检索器召回。 |

防御方法：

| 组件 | 开关 | 作用与边界 |
|------|------|------|
| Judge | `judge_model_name` | 识别乱码、提示注入、答非所问等可疑上下文。动态 Agentic 模式下，它在每轮 TrustRAG 之后、RiD 之前检查原始证据；若不足 `top_k`，只回退到 TrustRAG/医学聚类已保留的池。 |
| Agentic RAG | `agentic_rag=True` | 默认每个搜索标签都会触发新检索：本地 BM25 全库候选 → 当前检索器按 dot/cos 重排 → 防御 → 继续推理。`--no-agentic-live-retrieval` 仅用于复现旧版反复注入同一 top-k 的历史行为。 |
| Reason-in-Docs | `reason_in_docs=True` | 逐篇提取与问题有关的信息，降低长上下文噪声；本身不是恶意过滤。动态 Agentic 中它位于 TrustRAG/Judge 之后。 |
| RiD 严格模式 | `rid_defense=True` | 仅在 `reason_in_docs=True` 时生效；额外删除无关、误导或证据不足文档，可能同时误删有效医学证据。 |
| TrustRAG | `trustrag_filter=True` | 单轮 RAG 时执行聚类过滤和内部知识/冲突消解/最终回答三阶段；与 Agentic 同开时，作为每轮原始证据的簇过滤器，最终回答由 Agentic 生成。 |
| 医学语义聚类 | `medical_semantic_clustering=True` | 用 MedCPT Article Encoder 对候选医学文档聚类，额外过滤高凝聚的可疑模板簇；可与 TrustRAG 叠加。 |
| BIOS 医学知识图谱 | `medical_kg_filter=True` | **仅普通单轮 RAG（`entrypoint="standard"`）已接入，默认开启。** 先从 reserve 候选池（默认 10 篇）抽取医学三元组，再按“关系 → 合法实体端点 → 图中真实边”验证，以风险分重排为最终 `top_k`。默认 `original`：任一未验证三元组即高风险；`conservative`：低置信度 `unknown` 不扣分。该组件使用的公开 BIOS-v3 工件是可复现的近似变体，不应与论文未公开的精简图快照混称。 |


#### AgenticRAG + TrustRAG + Judge 的组合顺序

当 `agentic_rag=True`、`trustrag_filter=True` 且 Judge 已配置时，Agentic 是外层控制器。每次模型输出搜索标签后，入口按如下顺序处理该轮，而不是将三个开关当作互斥分支：

```text
Agentic 生成 search query
  → 本地 BM25 从全库取 agentic_retrieval_candidate_k 个候选（默认 100）
  → 当前评测检索器按 score_function=dot/cos_sim 重排
  → 混入该原始题目自己的 5 篇攻击文本，并按同一 search query 重新打分
  → 截取 reserve candidate_k（默认 10）
  → TrustRAG（→ 可选医学语义聚类）
  → LLM-as-Judge
  → 保留原始排序中的 top_k（默认 5）
  → 可选 Reason-in-Documents
  → 注入 Agentic 下一轮推理
```

Judge 的证据下限回退只恢复其输入的 **post-TrustRAG/post-cluster** 候选，绝不会重新加入被 TrustRAG 或医学聚类删除的文档。每题的结果 JSON 会记录每轮生成的 query、候选文档 ID/分数、每层过滤统计及最终保留集合；攻击文本始终限定为该题 own-5，绝不跨题混入。

动态稠密检索采用显式的两阶段实现：BM25 负责全库候选生成，当前 Contriever/DPR/ANCE 等编码器负责对这些候选按本实验的 dot 或 cosine 定义重排。它不是原始问题离线 top-100 JSON 的伪复用，也不应在未匹配候选生成方式时与“离线全库稠密检索”结果混称为同一检索口径。动态模式要求目标语料有本地 BM25 索引；缺失时程序会报错，而不会悄悄退回静态 top-k。

TrustRAG 的三阶段冲突作答与 Agentic 的最终作答不能同时充当最终回答器。因此二者同开时，TrustRAG 只负责每轮候选过滤；Agentic 在完成搜索后输出最终答案。`rag_max_turns` 用尽时，系统会额外请求一次不允许继续搜索的最终回答，避免把搜索标签本身当作预测结果。

### 3.5 集中配置运行

**所有评测、攻击与防御变量均在 [`experiment_config.py`](experiment_config.py) 中统一设置。** 每个字段已有中文注释、可选值和常用范围；不再需要编写多行命令。

完成配置编辑后，唯一的启动命令是：

```bash
python run_experiment_from_config.py
```

配置字段按以下几组组织：

| 配置组 | 代表字段 | 用途 |
|------|------|------|
| 运行入口 | `entrypoint` | `"agentic"` 或 `"standard"`。 |
| 检索 | `eval_model_code`、`eval_dataset`、`retrieval_results_path`、`top_k`、`gpu_id` | 选择模型、语料、检索文件和 GPU。 |
| 攻击与规模 | `attack_method`、`adv_json_path`、`target_ids_path`、`M`、`repeat_times` | 选择攻击、评测 query 和重复轮数。 |
| LLM 与 Judge | `model_name`、`judge_model_name` | 选择回答模型与恶意文本裁判；Judge 写 `"None"` 即关闭。 |
| Agentic / RiD | `agentic_rag`、`agentic_live_retrieval`、`agentic_retrieval_candidate_k`、`agentic_dense_batch_size`、`agentic_doc_cache_size`、`reason_in_docs`、`rid_defense`、`rag_max_turns` | 控制动态多轮检索、有界向量缓存与逐篇文档处理。 |
| TrustRAG / 医学聚类 | `trustrag_filter`、`medical_semantic_clustering` 及其路径/阈值 | 选择 TrustRAG 和医学语义聚类。 |
| BIOS KG 防御（标准入口） | `medical_kg_filter`、`medical_kg_mode`、`medical_kg_decision_mode`、`medical_kg_hard_filter_threshold`、`medical_kg_artifact_dir`、`medical_kg_candidate_k`、`medical_kg_rerank_weight`、`medical_kg_match_threshold`、`medical_kg_encoder_path`、`medical_kg_device`、`medical_kg_batch_size`、`medical_kg_max_chars`、`medical_kg_max_triplets`、`medical_kg_ner_config_path`、`medical_kg_triplet_cache_path`、`medical_kg_triplet_cache_namespace` | 默认启用 `original` 二值风险和 `rerank`；可显式选择直接硬剔除。 |

为确保不同实验可直接比较，以下设置已在程序中固定，不在配置文件暴露：对抗文本始终从预生成 JSON 读取、ASR 始终使用严格匹配、标注真值上下文始终关闭。

常用配置组合：

```python
# 无防御基线
"judge_model_name": "None",
"agentic_rag": False,
"reason_in_docs": False,
"trustrag_filter": False,
"medical_semantic_clustering": False,
"medical_kg_filter": False,

# Agentic + Judge，未启用 RiD
"judge_model_name": "gpt4.1mini",
"agentic_rag": True,
"reason_in_docs": False,
"rid_defense": False,

# TrustRAG + 医学语义聚类
"trustrag_filter": True,
"medical_semantic_clustering": True,
```

#### BIOS 医学知识图谱防御（标准单轮 RAG）

`main.py` 的默认值与 `experiment_config.py` 都已设置为启用 KG 防御和 `original` 模式。要让集中配置实际使用此组件，先将运行入口设为普通单轮 RAG：

```python
"entrypoint": "standard",
"medical_kg_filter": True,
"medical_kg_mode": "original",
"medical_kg_artifact_dir": "datasets/medical_kg/bios_v3_english_pt_full_20260813",
```

默认工件为公开 BIOS v3 的全量英文 preferred-term 临床关系图：454,746 个概念、1,109,060 条有效边，13 类医学验证关系均不做配额或随机抽样。它仍是对论文“去同义词节点”预处理的可复现近似，不等同于论文未公开的精简图快照。

其中，`original` 与论文的段落级二值判定保持一致：一篇文档只要存在未被 BIOS 图谱验证的三元组，风险即为 1；`conservative` 则把低相似度映射得到的 `unknown` 保持中性，以降低公开图谱覆盖不完整造成的误删。KG 首先重排 `medical_kg_candidate_k` 篇候选（默认 10），然后只把最终 `top_k` 篇交给后续 TrustRAG 或回答模型；它不会重新生成检索文件。

#### 工作流程、模式与边界

医疗 KG 的处理顺序固定如下；它操作的是已有检索结果，不会重建索引、重跑检索，也不会改写对抗文本。

```text
raw-dot 检索 top-10
  → LLM 从每篇文档最多 6,000 个字符中抽取至多 10 个（实体 1，关系，实体 2）
  → MedCPT 映射关系与该关系允许的实体端点
  → 查询 BIOS 是否存在对应真实边
  → 计算文档级 KG 风险
  → rerank 选出 top-5，或 hard_filter 直接删除高风险文档
  → （可选）原版 TrustRAG → 回答模型
```

| 项目 | `original` | `conservative` |
|---|---|---|
| 三元组映射 | 按原文式严格边验证 | 额外使用 `medical_kg_match_threshold` 判断低置信映射 |
| 风险计算 | 任一三元组为 `invalid` 或 `unknown`，文档风险即为 1 | `invalid / (valid + invalid)`；`unknown` 中性，若没有可判定三元组则风险为 0 |
| 适用性 | 与论文段落级筛查口径最接近 | 用于公开 BIOS 覆盖不完整时降低误删 |

#### 当前风险分的精确定义

风险分是**文档级**的，不把问题文本直接代入公式；问题只在前一步决定哪些文档会进入候选池。对每个抽取出的三元组，系统先匹配关系，再只在该关系允许的头、尾实体集合中用 MedCPT 找最相近实体，最后检查映射后的边是否真实存在于 BIOS。

设一篇文档的三元组审计计数为 `valid`、`invalid`、`unknown`、`ignored`，则：

\[
R_{\mathrm{original}}(d)=
\begin{cases}
1,&N_{\mathrm{invalid}}+N_{\mathrm{unknown}}>0\\
0,&\text{其他情况}
\end{cases}
\]

\[
R_{\mathrm{conservative}}(d)=
\begin{cases}
\dfrac{N_{\mathrm{invalid}}}{N_{\mathrm{valid}}+N_{\mathrm{invalid}}},&N_{\mathrm{valid}}+N_{\mathrm{invalid}}>0\\
0,&\text{其他情况}
\end{cases}
\]

因此，默认 `original` 是“一票否决”：哪怕一条三元组无法验证，文档风险就是 1。`conservative` 不会因为低置信或知识图谱未覆盖而惩罚 `unknown`；例如 1 条 `valid`、1 条 `invalid` 的风险为 0.5，只有 `unknown` 时风险为 0。当前全量 BIOS 工件的 `non_strict_relations=[]`，所以没有被忽略的关系；历史临床优先工件才会将 `associated with` 标为 `ignored`。

默认 `rerank` 先对当前候选池的检索分数做 min-max 归一化，再计算：

\[
S(d)=(1-w)\widetilde{S}_{\mathrm{retrieval}}(d)+w(1-R(d))
\]

其中默认 \(w=0.80\)，即检索分数 : KG 安全分数 = 2:8。若设置 `medical_kg_decision_mode="hard_filter"`，则不使用该混合分：满足 \(R(d)\geq\texttt{medical\_kg\_hard\_filter\_threshold}\) 的文档会被直接删除（默认阈值 1.0），余下文档保持原检索排序，且不会为了凑满 `top_k` 把已删文档放回。

此前讨论过的“KG 支持度 + 冲突度 + 候选异常模式”的连续风险公式已经移除，当前代码没有使用该公式，也没有对候选文档间的重复模式额外加分。

KG 仅接入 `entrypoint="standard"`。和 TrustRAG 同开时，KG 先从 top-10 选出最终候选，原版 TrustRAG 随后处理这些候选；二者都不会把已被前一层删除的文档重新补回。医疗语义聚类是独立模块，若开启则在 TrustRAG 之后运行。

常用配置：

```python
# KG 重排，安全分 : 检索分 = 3:7，并叠加原版 TrustRAG
"entrypoint": "standard",
"medical_kg_filter": True,
"medical_kg_mode": "original",
"medical_kg_decision_mode": "rerank",
"medical_kg_rerank_weight": 0.30,
"trustrag_filter": True,

# 1:0：识别为高风险即删除，不启用 TrustRAG
"medical_kg_filter": True,
"medical_kg_mode": "original",
"medical_kg_decision_mode": "hard_filter",
"medical_kg_hard_filter_threshold": 1.0,
"trustrag_filter": False,

# 无防御基线：必须显式关闭默认开启的 KG
"medical_kg_filter": False,
"trustrag_filter": False,
"medical_semantic_clustering": False,
"judge_model_name": "None",
```

#### 直接剔除已识别恶意文档

默认 `medical_kg_decision_mode="rerank"` 只会降低高风险文档排序。若实验需要“安全 : 检索 = 1:0”的直接剔除口径，使用：

```bash
--medical_kg_decision_mode hard_filter \
--medical_kg_hard_filter_threshold 1.0
```

此时 `original` 二值规则标为 `kg_risk=1` 的候选会在回答前直接删除；剩余文档保持原始检索顺序，**不会**为了凑足 `top_k` 把已删候选放回。因而最终上下文可能少于 5 篇，甚至为空；结果会记录 `medical_kg_hard_filtered_count` 和每篇文档的 `hard_filtered` 标记。

公开 BIOS-v3 的构建输入位于 `/home/Dataset/BIOS_v3/`。当前默认工件为 `datasets/medical_kg/bios_v3_english_pt_full_20260813`：保留英文 preferred-term 节点之间、医学三元组验证使用的全部 13 类关系边，**不做关系配额、reservoir 抽样或低优先级关系排除**；`associated with` 也按严格关系处理（本版源数据中没有可用的英文 PT 边）。仅保留英文 PT 是论文“去同义词节点”预处理的可复现近似，不等同于随机抽样。旧的 `bios_v3_clinical_priority_20260812` 与 `bios_v3_preferred_sample_20260803` 均保留作历史对照，结果不得与全量工件混称。三者都不是论文未公开的原始精简图快照；若取得论文规范图，应以 `scripts/build_bios_refined_kg.py --triples-json <ground_truth.json>` 重建并在配置中替换路径。

临时关闭默认 KG 防御可在直接运行时加：

```bash
python main.py --no_medical_kg_filter ...
```

三元组缓存会在抽取每篇候选文档后追加写入 `results/medical_kg_caches/`，因此每组完成后均保留可复用缓存。缓存键由文档文本、抽取长度、最大三元组数和命名空间生成；内容只有文档哈希与三元组，不含问题、答案、金标、目标标签或排序分数。相同数据集的“KG-only”和“KG + TrustRAG”可安全复用该缓存；若更换抽取模型、提示词或输出格式，须修改 `medical_kg_triplet_cache_namespace`。

结果 JSON 中每题都会记录 `medical_kg_filter_enabled`、`medical_kg_filter_applied`、`medical_kg_mode`、`medical_kg_decision_mode`、`medical_kg_rerank_weight`、有效/无效/未知/`ignored` 三元组计数、`medical_kg_document_audits`，以及直接剔除时的 `medical_kg_hard_filtered_count`。这些字段用于核对防御是否真正生效；比较实验时应同时固定检索文件、攻击文本、`top_k`、ASR 严格匹配口径和 KG 工件路径。

### 3.6 微调医学对抗鲁棒检索器 `contriever_v1` / `contriever_v2`

`contriever_v1` 在官方 `facebook/contriever` 上以 PubMedQA 训练；`contriever_v2` 以 v1 为初始化继续使用 MedQA 训练。二者均不改变模型结构、Mean Pooling 或 L2 Normalize，并将同一 Query 的两类攻击负例严格合并为一个训练样本：

```
{query, positive, blackbox_negative, hotflip_negative}
```

训练使用 Multiple-Negatives InfoNCE（temperature=0.05）：每个 Query 的正例为 `positive`，候选负例同时包含本条样本的 Black-box / HotFlip negative，以及 batch 内其余样本的正例和负例。因此不要把两个 JSONL 当作独立数据集训练。

#### 训练数据要求

仅传入下列两份 JSONL。每行都必须是 `query`、`positive`、`negative`、`attack_type` 四个字段；两份文件的 Query 与 positive 必须一一对应。

```
processed/pubmedqa_blackbox.jsonl  # attack_type 为 blackbox
processed/pubmedqa_hotflip.jsonl   # attack_type 为 hotflip
```

训练启动前会自动拒绝空文本、`positive == negative`、重复 Query、攻击类型错误、缺失配对，以及两文件 positive 不一致的情况。

#### 运行微调

```bash
conda activate PoisonedRAG

# 若本地已缓存 facebook/contriever，可把 --local-model-path 改为该目录；
# 未提供时会从 HuggingFace 的 facebook/contriever 加载。
CUDA_VISIBLE_DEVICES=0 python -m contriever_stage1.train \
  --blackbox-path processed/pubmedqa_blackbox.jsonl \
  --hotflip-path processed/pubmedqa_hotflip.jsonl \
  --model-name facebook/contriever \
  --output-dir checkpoint/contriever_v1 \
  --epochs 5 \
  --batch-size 16 \
  --max-length 512 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --temperature 0.05 \
  --gradient-clip 1.0 \
  --mixed-precision
```

可选地使用本地模型缓存：

```bash
CUDA_VISIBLE_DEVICES=0 python -m contriever_stage1.train \
  --local-model-path /path/to/facebook/contriever \
  --output-dir checkpoint/contriever_v1
```

每个 epoch 都会记录训练损失、正例/两类负例的平均相似度、两种 gap、`Positive > Negative` 准确率，并保存 `checkpoint/contriever_v1/epoch*/`。`best_model/` 由验证损失、平均 gap 和两种准确率共同决定。`final_evaluation.json` 给出微调前后在固定随机抽取 100 条训练样本上的对比；它用于训练过程诊断，不应视为独立测试集结果。

#### `contriever_v6`：无 MIRAGE PubMedQA 泄漏对照

v4 的 PQA-L 组成包含 MIRAGE PubMedQA 的评测题，因此不能用于检验未见题上的防御泛化。v6 保留 v4 的 9,174 条 PQA-U 样本，并从原始 PQA-U 以固定随机种子补入 1,000 条样本；`scripts/prepare_contriever_v6_pubmedqa.py` 会在生成和合并阶段强制拒绝与 MIRAGE PubMedQA query 重合的记录。v6 的 PubMedQA/MedQA 训练规模仍为 10,174:10,174，且训练超参数、原始 Contriever 初始化和 8:8 replay batch 与 v4 一致。

准备完成后，必须同时满足以下条件才可训练或评测：

- `checkpoint/contriever_v6/input/leakage_manifest.json` 中 `mirage_overlap` 为 `0`；
- 黑盒与 HotFlip JSONL 通过 `scripts/validate_training_data.py`；
- 正式检索结果由 `evaluate_beir.py --score_function cos_sim` 生成，并存在匹配的 `.meta.json`；
- 正式攻击评测采用每题 own-5 攻击文本和严格指定 `target_label` 的 ASR。

`contriever_v6` 默认加载 `checkpoint/contriever_v6/best_model`；若存放在其他位置，可设置 `CONTRIEVER_V6_PATH`。

#### 使用微调后的检索器

`contriever_v1` 与 `contriever_v2` 默认分别查找 `checkpoint/contriever_v1/best_model`、`checkpoint/contriever_v2/best_model`。若模型存放在其他位置，先显式指定路径：

```bash
export CONTRIEVER_V1_PATH=/absolute/path/to/checkpoint/contriever_v1/best_model
export CONTRIEVER_V2_PATH=/absolute/path/to/checkpoint/contriever_v2/best_model
```

先用该权重生成检索结果：

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate_beir.py \
  --model_code contriever_v2 \
  --dataset pubmed --split test --top_k 100 --score_function cos_sim \
  --queries-json results/adv_targeted_results/mirage_pubmedqa_all.queries.json \
  --result_output results/beir_results/mirage_pubmedqa_all-contriever_v2-cos.json
```

随后可与任意攻击模式一起评测：

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --eval_model_code contriever_v2 --eval_dataset pubmed \
  --model_name gpt4.1mini --judge_model_name None --top_k 5 \
  --attack_method hotflip --adv_source json \
  --adv_json_path results/adv_targeted_results/mirage_pubmedqa_all.json \
  --target_ids_path results/adv_targeted_results/mirage_pubmedqa_all.ids \
  --retrieval_results_path results/beir_results/mirage_pubmedqa_all-contriever_v2-cos.json \
  --score_function cos_sim \
  --adv_per_query 5 --M 500 --repeat_times 1 \
  --name pubmed_contriever_v2_gpt41mini_hotflip
```

## 4. 查看运行结果

运行日志输出至 `logs/user_runs_logs/<name>.out`，最终结果示例：

```
ASR: [0.88]
ASR Mean: 0.88

Ret (pre-defense): [[5, 4, 5, ...], ...]
Precision mean (pre-defense): 0.95
Recall mean (pre-defense): 0.95
F1 mean (pre-defense): 0.95

Ret (post-defense): [[0, 1, 0, ...], ...]
Precision mean (post-defense): 0.11
Recall mean (post-defense): 0.11
F1 mean (post-defense): 0.11

Ending...
```

### 指标说明

| 指标 | 含义 |
|------|------|
| **ASR Mean** | 攻击成功率。MCQ 仅在输出攻击源指定的错误选项时计为成功；不是任意答错率。 |
| **ASR** | 逐轮次的 ASR 列表（每轮 M 个 query） |
| **F1 (pre-defense)** | 裁判过滤**前**本题专属攻击文本的检索成功率（仅当启用 judge 时显示） |
| **F1 (post-defense)** | 裁判过滤**后**的对抗文本残留率 |
| **Precision / Recall** | 对抗文本在 top-K 检索结果中的精确率 / 召回率 |

### 快速查看结果

```bash
# 查看最终 ASR 和 F1
grep -aE "ASR Mean|F1 mean" logs/user_runs_logs/qwen3_whitebox_v5.out

# 查看逐轮进度
grep -a "Target Question" logs/user_runs_logs/qwen3_whitebox_v5.out | tail -5

# 查看完整流程（从启动到结束）
grep -aE "Namespace|Using|Doing|ASR Mean|F1 mean|Ending" logs/user_runs_logs/*.out
```

### 结果文件

| 路径 | 内容 |
|------|------|
| `logs/user_runs_logs/<name>.out` | 完整运行日志 |
| `results/query_results/<dir>/<name>.json` | 逐 query 的详细结果（含 model output、parsed label 等） |

## 5. 输出位置

| 类型 | 路径 | 格式 |
|------|------|------|
| 对抗文本 / ids / queries | `results/adv_targeted_results/` | `.json` / `.ids` / `.queries.json` |
| 检索结果 | `results/beir_results/` | `.json` |
| 评测结果（逐 query 详情） | `results/query_results/<dir>/<name>.json` | `.json` |
| 运行日志 | `logs/user_runs_logs/<name>.out` | 文本 |
