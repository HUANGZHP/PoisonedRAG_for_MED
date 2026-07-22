# Retriever Preference Fine-tuning (Contriever) — Triplet Data Construction

为 Contriever 模型构建 triplet 训练数据（query, positive, negative），支持两种攻击方式生成的 hard negatives。

## 目录结构

```
processed/                     # 输出目录（自动创建）
    pubmedqa_blackbox.jsonl
    pubmedqa_hotflip.jsonl
    medqa_blackbox.jsonl
    medqa_hotflip.jsonl

scripts/
    __init__.py
    utils.py                   # 通用工具（数据加载、实体匹配、质量校验）
    build_blackbox.py          # Black-box Attack 负样本生成
    build_hotflip.py           # HotFlip Attack 负样本生成
```

## 依赖

无额外依赖，仅需 Python 标准库 + `numpy`：

```bash
pip install numpy
```

## 数据来源

仓库 [PubMedQA-and-MedQA](https://github.com/HUANGZHP/PubMedQA-and-MedQA) 需要先克隆到本地：

```bash
git clone https://github.com/HUANGZHP/PubMedQA-and-MedQA.git /home/huangzhp53/PubMedQA-and-MedQA
```

## 运行方式

### 1. Black-box Attack

```bash
# PubMedQA
python scripts/build_blackbox.py \
    --input /home/huangzhp53/PubMedQA-and-MedQA/datasets/PubMedQA/official_pqal/pqal_fold0/train_set.json \
    --output processed/pubmedqa_blackbox.jsonl \
    --dataset pubmedqa

# MedQA
python scripts/build_blackbox.py \
    --input /home/huangzhp53/PubMedQA-and-MedQA/datasets/MedQA-USMLE/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl \
    --output processed/medqa_blackbox.jsonl \
    --dataset medqa
```

### 2. HotFlip Attack

```bash
# PubMedQA
python scripts/build_hotflip.py \
    --input /home/huangzhp53/PubMedQA-and-MedQA/datasets/PubMedQA/official_pqal/pqal_fold0/train_set.json \
    --output processed/pubmedqa_hotflip.jsonl \
    --dataset pubmedqa

# MedQA
python scripts/build_hotflip.py \
    --input /home/huangzhp53/PubMedQA-and-MedQA/datasets/MedQA-USMLE/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl \
    --output processed/medqa_hotflip.jsonl \
    --dataset medqa
```

## 输出格式

每行一个 JSON 对象：

```json
{
    "query": "What is the first-line treatment of...",
    "positive": "The recommended first-line treatment is...",
    "negative": "The recommended first-line treatment is...",
    "attack_type": "blackbox"
}
```

## 攻击策略说明

### Black-box Attack

- 识别 positive 文本中的医学实体（疾病、药物、治疗方案、生物标志物、诊断方法等）
- 替换 30-60% 的实体为误导性替代项
- 可选追加误导性结论句
- 结果文本保持医学语言风格，但语义错误

### HotFlip Attack

- 仅修改 1-3 个关键医学 token（优先替换疾病名、药物名、生物标志物）
- 其余文本完全保持不变
- 产生极高的词汇重叠率（lexical overlap），适合作为 hard negative

## 质量校验

生成前自动检查：

- query / positive / negative 非空
- negative ≠ positive
- 长度差距 ≤ ±20%

不符合条件的样本自动跳过。

## 可复现性

固定随机种子 `random.seed(42)` + `numpy.random.seed(42)`。
