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
    # 预生成检索结果的 JSON 路径；留空时按默认命名自动寻找。
    "retrieval_results_path": "results/beir_results/mirage_pubmedqa_all-contriever_v4.json",
    # 最终传入生成模型的文档数；正整数，常用 3、5、10。
    "top_k": 5,
    # 单卡 GPU 编号；整数，例如 0、1、2。
    "gpu_id": 0,
    # 可选多卡编号，逗号分隔，例如 "0,1"；留空则只使用 gpu_id。
    "gpu_ids": "",
    # 检索分数："dot"（默认）或 "cos_sim"。
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

    # 是否启用原版 TrustRAG；True 时固定执行 ROUGE-L 门控、KMeans 过滤和三阶段冲突消解。
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
}
