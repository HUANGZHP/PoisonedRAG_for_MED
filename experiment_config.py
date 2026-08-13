"""集中实验配置：修改本文件后，运行 python run_experiment_from_config.py。"""

CONFIG = {
    # 选择执行入口："agentic" 为 Agentic RAG；"standard" 为普通单轮 RAG。
    "entrypoint": "agentic",

    # 检索器：contriever、contriever_v1/v2/v3/v4/v5、medcpt、bm25、dpr、ance 等。
    "eval_model_code": "contriever_v4",
    # 语料库：pubmed、statpearls、textbooks，或 BEIR 的 nq、hotpotqa、msmarco。
    "eval_dataset": "pubmed",
    # 数据切分：通常为 "test"；支持数据集时也可使用 "train" 或 "dev"。
    "split": "test",
    # 预生成检索结果的 JSON 路径；必须由下方 score_function 同样的设置生成，dot 文件建议以 -dot.json 结尾；cosine 仅在显式指定时使用 -cos.json。
    "retrieval_results_path": "results/beir_results/mirage_pubmedqa_all-contriever_v4-dot.json",
    # 最终传入生成模型的文档数；正整数，常用 3、5、10。
    "top_k": 5,
    # 单卡 GPU 编号；整数，例如 0、1、2。
    "gpu_id": 0,
    # 可选多卡编号，逗号分隔，例如 "0,1"；留空则只使用 gpu_id。
    "gpu_ids": "",
    # 检索分数："dot"（默认，与原论文一致）或 "cos_sim"（需显式指定）。修改后必须重建上方 retrieval_results_path。
    "score_function": "dot",

    # 主生成模型名称；需对应 model_configs/<名称>_config.json，例如 gpt4.1mini。
    "model_name": "gpt4.1mini",
    # 主模型配置文件；None 自动使用 model_name 对应文件，也可写自定义 JSON 路径。
    "model_config_path": None,
    # Judge 模型名称；"None" 关闭 Judge，其他值需对应 model_configs/<名称>_config.json。
    "judge_model_name": "gpt4.1mini",
    # Judge 配置文件；留空自动按 judge_model_name 寻找，也可写自定义 JSON 路径。
    "judge_model_config_path": "",
    # 攻击：None（无攻击）、"LM_targeted"（黑盒）、"hotflip"（白盒）。
    "attack_method": "LM_targeted",
    # 对抗文本 JSON 路径；无攻击时仍可作为评测 query 的来源。
    "adv_json_path": "results/adv_targeted_results/mirage_pubmedqa_all.json",
    # 要评测的 query ID 文件；留空则使用 adv_json_path 中的全部 query。
    "target_ids_path": "results/adv_targeted_results/mirage_pubmedqa_all.ids",
    # 每个 query 注入的对抗文本数；正整数，通常为 5。
    "adv_per_query": 5,
    # 单轮评测 query 数；必须不大于目标数据总数，例如 PubMedQA 为 500、MedQA 为 1273。
    "M": 500,
    # 重复评测轮数；正整数，通常为 1。
    "repeat_times": 1,
    # 随机种子；任意整数，用于可复现抽样和攻击。
    "seed": 12,
    # 本次实验名称；会用于结果和日志文件名，应避免与已有实验重名。
    "name": "configured_experiment",
    # 结果子目录名称；通常保持 "main"。
    "query_results_dir": "main",

    # 是否启用 Agentic RAG；True/False，仅 entrypoint="agentic" 时生效。
    "agentic_rag": True,
    # 是否启用 Reason-in-Docs；True/False，仅 entrypoint="agentic" 时生效。
    "reason_in_docs": False,
    # 是否启用 RiD 严格相关性防御；True/False，仅 reason_in_docs=True 时生效（Agentic 模式也需同时开启 reason_in_docs）。
    "rid_defense": False,
    # Agentic 最大推理/检索轮数；正整数，常用 1、2、3。
    "rag_max_turns": 3,
    # 是否在日志打印 Agentic 中间步骤；True/False。
    "rag_verbose": False,
    # 是否让每个 Agentic 搜索标签触发新的全库候选检索；默认 True。False 仅用于复现旧版“反复注入同一 top-k”的历史行为。
    "agentic_live_retrieval": True,
    # 每轮先由本地 BM25 全库索引召回的候选数；稠密检索器会按本次 dot/cos 配置重排这些候选，必须不小于防御 reserve candidate_k。
    "agentic_retrieval_candidate_k": 100,
    # 动态稠密重排的文档编码批大小；显存紧张时优先调小。
    "agentic_dense_batch_size": 16,
    # 动态重排的 CPU 文档向量 LRU 缓存上限；0 关闭缓存，避免长任务无界增长。
    "agentic_doc_cache_size": 4096,

    # 是否启用原版 TrustRAG；普通 RAG 时执行过滤和三阶段冲突消解；与 Agentic 同开时仅在每轮原始证据上执行过滤，最终回答由 Agentic 生成。
    "trustrag_filter": False,
    # TrustRAG 通用语义编码器本地目录；默认使用已有 Contriever，也可改为本地 SimCSE 目录。
    "trustrag_encoder_path": "/home/HF_Model/facebook/contriever",

    # 是否额外启用 MedCPT 医学语义聚类；True/False，可单独使用或与 TrustRAG 叠加。
    "medical_semantic_clustering": False,
    # 医学聚类参与过滤的候选数；正整数且不小于 top_k，常用 10 或 20。
    "medical_cluster_candidate_k": 10,
    # MedCPT Article Encoder 的本地目录；用于医学文档语义向量。
    "medical_cluster_encoder_path": "/home/HF_Model/ncbi/MedCPT-Article-Encoder",
    # 医学编码批大小；正整数，显存不足时可改为 1、2、4。
    "medical_cluster_batch_size": 8,
    # 医学编码最大 token 数；正整数，常用 128、256、512。
    "medical_cluster_max_length": 512,
    # 判定高凝聚可疑簇的平均余弦相似度阈值；0~1，默认 0.88。
    "medical_cluster_similarity_threshold": 0.88,

    # BIOS+MedCPT 医学知识图谱防御（仅标准单轮入口 entrypoint="standard" 生效；Agentic 入口尚未接入此模块）。
    # 默认启用。若需关闭，设为 False。
    "medical_kg_filter": True,
    # original：原文二值规则，任一未验证三元组即将文档视为高风险；conservative：低置信 unknown 不扣分。
    "medical_kg_mode": "original",
    # 默认由公开 BIOS v3 的英文 PT 术语构建：13 类医学验证关系的全部边均保留，不做关系配额或抽样；仍不等同于论文未公开的原始精简图快照。
    "medical_kg_artifact_dir": "datasets/medical_kg/bios_v3_english_pt_full_20260813",
    # KG 重排前读取的候选文档数；重排后仍输出 top_k 篇。
    "medical_kg_candidate_k": 10,
    # 最终排序中 KG 安全分的权重，范围 0~1；0.80 即检索分数 : 安全分数 = 2:8。
    "medical_kg_rerank_weight": 0.80,
    # 仅 conservative 模式使用的实体/关系最低余弦相似度；original 模式不使用阈值。
    "medical_kg_match_threshold": 0.45,
    # rerank 按风险重排；hard_filter 直接剔除达到阈值的候选，且不再保留最低 top_k 篇。
    "medical_kg_decision_mode": "rerank",
    # hard_filter 的剔除阈值；original 二值风险设为 1.0 即直接剔除所有已识别恶意文档。
    "medical_kg_hard_filter_threshold": 1.0,
    # 与原文对齐的 MedCPT Query Encoder。
    "medical_kg_encoder_path": "/home/HF_Model/ncbi/MedCPT-Query-Encoder",
    # KG 向量计算设备；默认 CPU，避免占用主实验 GPU。
    "medical_kg_device": "cpu",
    # KG MedCPT 编码批大小。
    "medical_kg_batch_size": 32,
    # 每篇候选文档交给 LLM 三元组抽取器的最大字符数。
    "medical_kg_max_chars": 6000,
    # 每篇候选文档最多验证的显式医学三元组数。
    "medical_kg_max_triplets": 10,
    # 独立的零样本三元组抽取模型配置；留空时复用评测 LLM。
    "medical_kg_ner_config_path": "",
    # 三元组抽取缓存 JSONL；留空时按对抗文本来源和检索器自动放在 results/medical_kg_caches/。
    # 缓存仅含文档哈希与抽取三元组，不含题目、答案、目标标签、金标或排序分数。
    "medical_kg_triplet_cache_path": "",
    # 变更抽取模型、提示词或输出格式时修改该名称，以避免复用旧抽取结果。
    "medical_kg_triplet_cache_namespace": "triplet_schema_v1",
}
