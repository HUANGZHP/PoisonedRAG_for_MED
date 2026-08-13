# 更新记录

## 2026-07-23：评测口径与候选池修正

### 已完成

- MCQ 攻击成功率（ASR）统一为严格口径：仅当模型回答攻击预先指定的错误选项时计为成功；普通答错不再计入 ASR。
- 每道题的候选池只注入其自身对应的 5 篇攻击文档，不再混入其它题目的攻击文档。
- 标准检索流程和 Agentic 流程均使用上述候选池与严格 ASR 口径。
- 正常文档检索、攻击文档排序与 HotFlip 优化统一采用 cosine 相似度。
- 实验配置中的 score_function 默认值设为 cos_sim；预检索结果会写入评分方式元信息，若与当前配置不一致会拒绝混用，需按当前评分方式重新生成预检索结果。

### 已验证

- 已完成语法检查。
- 已完成评测补丁一致性检查。
- 已完成 cosine 评分元信息的最小化验证。
- 本条目不包含旧 dot-product 结果或旧候选池口径的实验指标；相关实验需按本次修正后重新运行。

### 已有重跑结果（历史协议，仅供追溯）

以下数值来自已有 `rerun` / `redo` 结果文件，均为旧 `dot-product + 全局攻击池` 协议。严格 ASR 已依据攻击源 JSON 的指定错误标签重新计算；结果文件内旧的 `target_label` 字段不作为依据。攻击检索 F1 为结果文件中本题 `injected_adv` 占 top-5 的比例，未计入跨题攻击文档，因此不能与本次修正后的正式协议混用。

#### MedQA + PubMed（N=1,273，无防御）

| 模型 | 攻击 | 严格 ASR | 正确率 | 本题攻击检索 F1 | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| contriever_v2 | 无攻击 | — | 79.97% | — | 已完成 |
| contriever_v2 | 黑盒 | 80.75% | 17.36% | 71.55% | 已完成 |
| contriever_v2 | HotFlip | 84.68% | 13.90% | 85.77% | 已完成 |
| contriever_v3 | 无攻击 | — | 80.36% | — | 已完成 |
| contriever_v3 | 黑盒 | 91.04% | 8.41% | 89.00% | 已完成 |
| contriever_v3 | HotFlip | 84.76% | 13.59% | 85.81% | 已完成 |
| contriever_v4 | 无攻击 | — | 80.44% | — | 已完成 |
| contriever_v4 | 黑盒 | 82.72% | 15.87% | 69.05% | 已完成 |
| contriever_v4 | HotFlip | — | — | — | 旧协议任务仍在运行，暂不记录 |

#### PubMedQA + PubMed（N=500，无防御）

有效攻击源为 `mirage_pubmedqa_all_gpt41mini.json`（500 题 × 5 篇，均非空）。`mirage_pubmedqa_all.json` 的攻击文本全空，相关攻击结果已排除。

| 模型 | 无攻击正确率 | 黑盒：严格 ASR / 正确率 / F1 | HotFlip：严格 ASR / 正确率 / F1 |
| --- | ---: | --- | --- |
| contriever | 72.20% | 88.40% / 6.20% / 94.00% | 89.80% / 6.80% / 96.60% |
| contriever_v1 | 71.80% | 11.80% / 72.00% / 0.28% | 13.60% / 70.00% / 3.48% |
| contriever_v2 | 14.80% | 7.80% / 14.60% / 0.00% | 11.40% / 14.20% / 1.16% |
| contriever_v3 | 70.20% | 10.80% / 70.00% / 0.08% | 12.20% / 69.60% / 1.84% |
| contriever_v4 | 61.80% | 11.20% / 62.20% / 0.00% | 15.20% / 59.00% / 3.52% |

### 本次修正后正式协议的结果

暂无。新协议要求 cosine 预检索、每题仅注入自身 5 篇攻击文档，并以严格指定错误选项计算 MCQ ASR；历史表中的数值不得作为该协议的对照结果。

## 2026-07-23：重跑请求澄清（未执行）

- 用户要求的是报告已有重跑实验结果，并非启动新的重跑。
- 此前误启动 `contriever_v2 + MedQA + PubMed` 的 cosine 预检索；在用户澄清后已停止。
- 停止时仍处于 PubMed 语料加载阶段，未产生预检索文件或任何实验结果。
- 未执行无攻击、黑盒或 HotFlip 评测；本条不产生新的 ASR、检索或回答指标。
- 后续仅在用户明确要求重跑时再启动新任务。

## 2026-07-23：v6 无 MIRAGE 泄漏对照与正式重跑（进行中）

- 用户已授权训练并运行新的正式实验。
- v6 将复用 v4 的原始 Contriever 初始化、3 epoch、每 batch 8 条 PubMedQA + 8 条 MedQA 的 1:1 replay 配方。
- 旧 v4 的 1,000 条 PQA-L 训练题包含 MIRAGE PubMedQA 的全部 500 道评测题，不能用于无泄漏对照；v6 保留 9,174 条已核验无 MIRAGE 重合的 PQA-U 样本，并从原始 PQA-U 补入 1,000 条经 MIRAGE query 过滤的新样本，总数仍为 10,174。
- 正式评测将使用 cosine 检索、元信息校验、每题仅注入自身 5 篇攻击文本与严格指定标签 ASR；将重跑 contriever_v1、v3、v4、v6 的 PubMedQA 与 MedQA 无攻击、黑盒和 HotFlip。
- 本条目尚未产生任何新指标。

### 当前状态

- 已生成 `checkpoint/contriever_v6/input/pubmedqa_v6_source.json`：10,174 条唯一 PQA-U query，与 MIRAGE PubMedQA 的 500 题重合为 0。
- 已启动 1,000 条替代 PQA-U 样本的黑盒负例生成；完成后将生成对应 HotFlip 负例、合并并校验训练数据，再启动 v6 训练。
- 已启动 contriever_v1、contriever_v3、contriever_v4 在 PubMedQA 与 MedQA 上的 6 个 cosine 预检索任务（`top_k=100`、`max_length=128`）；每项将生成独立结果与 `.meta.json`。
- v6 的 1,000 条替代 PQA-U 黑盒负例已生成完成；完整配对校验将在对应 HotFlip 负例生成后执行。
- 为避免影响 GPU 0–3 的其他用户任务，已撤销尚未产生输出的 v1/v3 预检索启动，仅保留空闲 GPU 5、6 上的 v4 两项预检索；其余任务将按资源空闲情况继续。
- 已启动 v6 替代 PQA-U 样本的 HotFlip 负例生成（GPU 5）；完成后将执行完整配对校验并训练 v6。
- 已排入受 `set -e` 保护的后续链：HotFlip 成功后合并 v6 PQA-U 训练文件、执行 `validate_training_data.py`，并以 v4 的原始 1:1 replay 参数训练 `checkpoint/contriever_v6`；若 HotFlip、合并或校验失败，训练不会启动。
- 初次 HotFlip 因脚本内遗留的默认 CUDA:0 张量与物理 GPU 5 不一致而在第 1 题前失败，未写出输出；已改用仓库既有的 `CUDA_VISIBLE_DEVICES=5` 与脚本内 `--gpu 0` 调用方式重启，当前生成正常。后续链已绑定到本次重启的进程。
- v6 的 1,000 条 HotFlip 负例已完成；PQA-U 黑盒/HotFlip 合并后通过 `validate_training_data.py` 校验，确认 10,174 条训练 query 与 MIRAGE PubMedQA 的 500 题保持 0 重合。
- v6 已按 v4 的原始 3 epoch、8:8 PubMedQA/MedQA replay 配方完成训练，`checkpoint/contriever_v6/best_model` 已保存。尚未产生任何正式攻击评测指标。
- 已排入 `scripts/run_v6_formal_queue.sh`：等待当前 v4 cosine 预检索释放 GPU 5、6 后，先生成 v6 的 PubMedQA/MedQA cosine 检索结果与 `.meta.json`，再依次运行无攻击、LM_targeted、HotFlip 的正式评测。每次评测均使用 own-5、严格 ASR、`score_function=cos_sim` 和相应有效攻击源。
- README 已补充 v6 的无 MIRAGE 泄漏构成、必经校验与正式评测要求；未修改任何模型或实验配置。

### v6 正式协议进度（2026-07-24）

- v6 的 PubMedQA 与 MedQA cosine 预检索均已完成，并各自生成匹配的 `.meta.json`。
- 无攻击与 LM_targeted 已完成：PubMedQA 的 LM_targeted 本题攻击检索 F1 为 0.00、严格 ASR 为 16%；MedQA 的 LM_targeted 本题攻击检索 F1 为 76%、严格 ASR 为 84%。这些是 v6 的正式协议中间结果，HotFlip 尚在运行，尚不可作为完整攻击对比结论。
- HotFlip 正在运行：PubMedQA 约 67%，MedQA 约 18%。
- contriever_v4 的两套 cosine 预检索已完成并带元信息；contriever_v1/v3 的正式预检索与评测尚未启动。

### 原始 Contriever cosine 无攻击基线（2026-07-24，进行中）

- 用户要求以原始 Contriever 在与 v6 相同的正式 cosine 协议下补跑 PubMedQA 无攻击基线，用于区分模型退化与旧 dot 口径差异。
- 已在空闲 GPU 4 启动：先生成 （top_k=100、max_length=128）及匹配 ，成功后自动用相同提示词、解析器与  运行 500 题回答评测。
- 结果将写入  与 ；不注入攻击文本，不影响 GPU 5、6 的 v6 HotFlip 队列。
- 路径补充：预检索文件为 results/beir_results/formal_cosine/mirage_pubmedqa_all-contriever-cos.json 及其 .meta.json；回答结果目录为 results/query_results/formal_cosine_baseline/；评测参数为 attack_method=None。
- v6 HotFlip：PubMedQA 的 500/500 已完成并已写入结果；MedQA 当前为 1,237/1,273（约 97%），仍在运行。
- 原始 Contriever 的 PubMedQA cosine 无攻击基线正在 GPU 4 生成检索文件；编码阶段已到第 48/478 个语料块，尚未进入回答评测。

### v6 dot own-5 诊断轮次（2026-07-24，已排队）

- 用户要求 v6 追加一轮 dot 相似度实验。该轮保持 own-5、每题 5 篇攻击文档、严格 target_label ASR、同一攻击源、同一提示词与解析器；仅将 score_function 改为 dot。
- 队列等待正在运行的 v6 MedQA cosine HotFlip 结束后使用 GPU 5、6；将依次完成 PubMedQA/MedQA 的 dot 检索（top_k=100）、无攻击、LM_targeted 和 HotFlip。
- dot 轮次仅用于诊断 cosine 与 dot 的影响，不与正式 cosine 结果混合作为协议内横向结论。
- v6 cosine 正式轮次已完成：PubMedQA 无攻击/LM_targeted/HotFlip 的准确率分别为 47.2/47.2/46.0%，严格 ASR 为 15.2/15.8/16.6%；攻击文档实际进入上下文的题数为 0/500、0/500、10/500。MedQA 对应准确率为 79.5/14.9/14.4%，严格 ASR 为 7.8/83.7/84.4%；攻击文档实际进入上下文的题数为 0/1273、1124/1273、1245/1273。PubMedQA 的约 32% 输出为未解析的弃答，单列报告，不把其误作攻击成功。
- 原始 Contriever cosine 基线仍处于检索阶段，尚无可比回答指标；v6 dot own-5 队列已开始两项检索，也尚未产生回答指标。

### v7 规划：MedQA own-5 独立错误证据 + dot（2026-07-24，待实现/排队）

- 用户要求保持 v6 的原始 Contriever 初始化、10,174:10,174 的 8:8 replay、3 epoch 等高层配方，仅将 MedQA 训练负例替换为每个训练题 own-5 的独立错误医学证据，并使用 raw dot 相似度训练 v7。
- 现有 v6 数据接口仅有两条负例且训练时固定 L2 normalize；v7 需要最小扩展以读取每题五条负例、将五条均纳入 InfoNCE 候选，并在 dot 模式下关闭 normalize。不会改动现有 v6 路径或结果。
- MedQA 训练集为 10,174 题，因此需生成并校验 50,870 条独立错误医学证据；将严格核验其 query 与 MIRAGE MedQA 评测题无重合。为避免与正在运行的 dot 评测争用 API/GPU，后续任务应在该队列结束后启动。
- 口径更正：用户明确 v7 每个训练题仍严格使用一条黑盒负例和一条白盒（HotFlip）负例；此前 own-5 被误解为训练时五负例，现已撤销。own-5 仅用于正式评测的每题攻击注入。
- v7 将保留 v6 的 PubMedQA 黑盒/白盒训练数据和 8:8 replay；MedQA 仅替换为每题一条独立错误医学证据黑盒负例及其对应的一条 dot HotFlip 白盒负例。训练目标仍为 P、NB、NH 三路 InfoNCE，只在 dot 模式关闭 L2 normalize。
- 因此 MedQA 生成量为 10,174 条黑盒加 10,174 条白盒，而非 50,870 条；预估生成、白盒优化、校验和训练合计约 7 至 9 小时（不含排队与后续完整评测）。
- 用户决定：此后所有新实验默认使用 raw dot-product；cosine 仅在明确指定的历史对照或消融中使用。已将 main.py、agentic_main.py、evaluate_beir.py 与 experiment_config.py 的默认 score_function 改为 dot。正在运行的 cosine/dot 队列及既有结果保持原样。
- 用户授权：v7 训练完成并通过数据/数值校验后，自动启动完整 dot 正式评测。评测将生成 PubMedQA 与 MedQA top-100 dot 检索文件及元信息，然后依次运行无攻击、LM_targeted、HotFlip；保持 own-5、严格 target_label ASR 与有效攻击源。
- 远程连接已恢复。用户追加第二阶段：在当前队列及 v7 的无防御 dot 正式轮次全部完成后，对 v7 与原始 Contriever 分别运行 PubMedQA、MedQA 的全防御评测。每个模型/数据集均运行无攻击、LM_targeted、HotFlip；启用 Judge（gpt4.1mini）、原版 TrustRAG 与医学语义聚类，明确不启用 RID。该阶段仍使用 dot、own-5 与严格 target_label ASR；原始 Contriever 将在该阶段另行生成对应的 dot 检索文件。

### 2026-07-24 队列最终结果（2026-07-27 汇总）

- v6 cosine 与 dot own-5 两轮均已完整结束；原始 Contriever 的 PubMedQA cosine 无攻击基线亦已结束，当前无相关后台实验进程。
- v6 cosine：PubMedQA 无攻击/LM_targeted/HotFlip 准确率为 47.2/47.2/46.0%，严格 ASR 为 15.2/15.8/16.6%，own-5 攻击检索 F1 为不适用/0.00/0.48%。MedQA 对应为准确率 79.5/14.9/14.4%，严格 ASR 7.8/83.7/84.4%，F1 不适用/76.37/90.09%。
- v6 dot：PubMedQA 无攻击/LM_targeted/HotFlip 准确率为 59.4/59.6/55.6%，严格 ASR 为 12.4/12.2/17.2%，own-5 F1 为不适用/0.00/3.24%。MedQA 对应为准确率 78.7/7.6/11.8%，严格 ASR 7.3/91.8/87.1%，F1 不适用/93.86/97.44%。
- 原始 Contriever PubMedQA cosine 无攻击基线：总体准确率 75.0%，严格 target-label 命中率 8.6%，未解析/弃答率 11.2%。尚未运行原始 Contriever 的 MedQA cosine 或任一原始 Contriever dot 基线。
- v4 仅完成 cosine 预检索；v1/v3 未启动；v7 和后续全防御阶段尚未启动。
- v7 未启动的原因已核实：此前仅记录了方案与后续评测意图，未实际创建 MedQA 独立黑盒数据、dot 训练分支或自动队列；不是 GPU、API 或服务器故障。该执行遗漏现已确认，后续须先完成实现和校验再启动。
- v7 已于 2026-07-27 启动完整训练链：已从旧 MedQA 配对文件重建并核验 10,174 题的 MedQA-USMLE 训练源；当前正在生成独立黑盒错误医学证据。后续受 set -e 保护，依次执行 dot HotFlip、训练数据校验、去除黑盒 query 前缀以得到独立黑盒训练文本，以及 3 epoch 的 8:8 dot replay 训练至 checkpoint/contriever_v7。
- 为使 v7 使用 raw dot 训练，contriever_stage1/evaluate.py 已支持通过 CONTRIEVER_TRAIN_SCORE_FUNCTION=dot 关闭训练时 L2 normalize；默认仍保持 cos_sim，既有 v6 结果不受影响。
- v8 已排队：等待 v7 的 MedQA 黑盒与 HotFlip 数据文件完成，并等待 v7 训练释放 GPU 5 后，用完全相同的数据、初始化、8:8 replay、3 epoch、学习率和 seed 训练 checkpoint/contriever_v8；唯一差异为 CONTRIEVER_TRAIN_SCORE_FUNCTION=cos_sim，从而保留 L2 向量归一化。

- 2026-07-27: v7 dot training started on GPU 5 after 10,174 MedQA blackbox and HotFlip negatives passed validation; v8 queue restored and waits for v7 success before cosine-normalized training on identical data.

- 2026-07-27: User clarified that all training negatives must retain Q+I. Stopped the short-lived v7 I-only run before any checkpoint was produced; restarted v7 dot training with raw MedQA blackbox Q+I and HotFlip Q+I inputs. v8 queue now waits on this replacement v7 process.

- 2026-07-27: Correct Q+I v7 and L2-normalized v8 training both completed (3 epochs). Started v7 dot/own-5 formal retrieval and strict-ASR evaluation queue: PubMedQA and MedQA, None/LM_targeted/HotFlip.

- 2026-07-27: Restarted v7 dot/own-5 formal queue with unambiguous formal_dot_v7 paths and contriever_v7-dot retrieval filenames; the prior 11-second queue was stopped before it wrote any result file.

- 2026-07-27: Started v8 dot/own-5 formal retrieval and strict-ASR evaluation in parallel with v7. It uses independent formal_dot_v8 outputs and GPUs 3/5; protocol is identical to v7.

- 2026-07-28: v8 dot/own-5 formal evaluation completed: PubMedQA accuracy/strict-ASR/own5-F1 None=65.6/10.2/N-A, LM=65.6/9.6/0.00, HotFlip=60.6/15.0/4.04; MedQA None=79.18/7.78/N-A, LM=8.80/90.42/88.30, HotFlip=12.33/86.25/95.43. v7 PubMedQA retrieval completed, but the concurrent v7 MedQA retrieval process was system-killed and set -e stopped its queue before any v7 answer evaluation. No formal queues remain running. Host memory pressure is high (swap full); future retrieval/evaluation jobs must run one at a time.

- 2026-07-28: Resumed incomplete v7 dot/own-5 protocol with a memory-gated serial queue. It reuses the completed PubMedQA retrieval, runs only one task at a time, and waits for at least 180 GiB MemAvailable before each remaining retrieval/evaluation.

- 2026-07-28: Fixed TrustRAG answer-contract bug. The final TrustRAG prompt now preserves task format (PubMedQA: exactly yes/no; MCQ: exactly one valid option letter). Empty responses in any TrustRAG stage now raise RuntimeError and abort evaluation. py_compile and AST-level mock tests passed without API calls.

- 2026-07-28: Queued repaired original-Contriever PubMedQA TrustRAG and TrustRAG+medical-semantic-clustering formal dot/own-5 suites. The queue waits for the v7 serial queue, reuses no old retrieval, generates a metadata-backed original Contriever dot retrieval, and runs every remaining retrieval/evaluation task one at a time with a 180 GiB MemAvailable gate.

- 2026-07-28: Fixed TrustRAG answer-contract bug. TrustRAG final response now receives the dataset-specific format requirement (PubMedQA: exactly yes/no; MCQ: exactly one valid option letter). Any empty response in its internal, consolidation, or final stage now raises RuntimeError and aborts evaluation instead of being recorded as an unanswered item. Syntax and mock no-API tests passed.

- 2026-07-28: Fixed medical-semantic-clustering no-op. Both TrustRAG variants now filter a shared candidate_k pool (default 10) before the final top-5 truncation; this gives the MedCPT filter room to remove a suspicious cluster while preserving a fair same-candidate-pool comparison.
2026-07-28: User authorized guarded parallel original-Contriever PubMedQA TrustRAG vs TrustRAG+medical-clustering rerun. Added isolated GPU 1/2 execution after v7, 280GiB start gate and 220GiB fallback-to-serial memory guard.
2026-07-28: Guarded parallel TrustRAG queue launched successfully; it waits for v7 before fresh original-Contriever retrieval and paired GPU 1/2 evaluations.
2026-07-28: Per user confirmation, TrustRAG rerun reuses historical original-Contriever PubMedQA retrieval results/beir_results/mirage_pubmedqa_all-contriever.json as dot. Removed v7/retrieval wait; guarded GPU1/2 paired evaluation may run alongside v7.
2026-07-28: Fixed guarded TrustRAG queue shell variable concatenation before model launch; restarted paired evaluation queue.

2026-07-28: Corrected F1, PubMedQA label, and TrustRAG evidence-integrity issues.
- Retrieval F1 now explicitly returns 0 when both precision and recall are 0; prior huge positive/negative F1 values were uninitialized NumPy output, not model behavior.
- PubMedQA now uses the closed yes/no/maybe label space in both main entry points. Prompts require exactly one declared label and parsing/strict matching use the same declared label space.
- TrustRAG now removes a candidate cluster only when it is both lexically triggered and high-cohesion, preserves original rank, and refuses any deletion that would leave fewer than top_k documents.
- TrustRAG final answering now verifies the retained raw documents, rather than relying only on compressed consolidation plus internal knowledge. The agentic entry point now uses the same candidate-pool and label rules.
- Python syntax checks, no-API mock regression tests, and both-entry-point label-contract imports passed for F1 zero handling, case-insensitive PubMedQA yes/no/maybe parsing, retained-evidence floor, and final raw-evidence verification. Existing TrustRAG jobs were started before these edits and remain pre-fix results; no job was stopped or restarted by this code change.

2026-07-28: Follow-up hardening for label, filtering, and evidence floors.
- Agentic RAG and Reason-in-Documents now receive the same PubMedQA yes/no/maybe closed label space as direct and TrustRAG answering; their final prompts no longer hard-code binary yes/no.
- Judge filtering now evaluates the reserve candidate pool and restores the original ranked candidates whenever fewer than top_k would remain. The Agentic outer path avoids a redundant second Judge pass; the standalone Agentic Judge helper also has a non-empty fallback.
- TrustRAG now requires the lexical duplicate evidence to occur inside the same high-cohesion cluster selected for deletion, with at least three members and a repeated template. A lexical match in another cluster cannot trigger deletion.
- Closed-label parsing now reads raw output and accepts only one declared token after control-wrapper removal. Text such as a label plus punctuation or explanation is unparsed.
- The TrustRAG final raw-evidence appendix is explicitly delimited as untrusted quoted data and may not be followed as instructions.
- Expanded syntax and no-API regression tests passed for F1 zero handling, strict labels, PubMedQA maybe, cluster-local filtering, minimum-evidence floors, Agentic/RiD label forwarding, and TrustRAG raw-evidence verification. No running job was stopped or restarted.

2026-07-28: Implemented the AgenticRAG + TrustRAG + LLM-as-Judge composition (no experiment started or stopped).

- `run_agentic_rag` now accepts backward-compatible per-round retrieval and raw-evidence filtering callbacks. A generated `<|begin_search_query|>…<|end_search_query|>` therefore triggers a fresh retrieval rather than reinjecting the initial static top-k; `--no-agentic-live-retrieval` remains the explicit legacy-reproduction path.
- `agentic_main.py` now uses the order `live retrieval → own-5 re-scoring/mixed ranking → TrustRAG → optional medical semantic clustering → Judge → optional RiD → Agentic continuation`. Judge operates after TrustRAG on raw documents and its top-k fallback restores only the post-TrustRAG/post-cluster candidate pool.
- Live dense Agentic retrieval uses the local BM25 index for corpus-wide candidate generation (default 100), then reranks those candidates with the configured evaluation encoder and `score_function`. Its bounded CPU LRU document-embedding cache defaults to 4,096 entries; result JSON records backend, generated query, document ID/score candidates, and filter statistics for every round.
- With `agentic_rag` and `trustrag_filter` both enabled, Agentic owns the final answer; TrustRAG's three-stage conflict answer is intentionally not used as a second competing answerer. At the search-turn cap, Agentic receives one final no-more-search answer request instead of returning a search tag as the prediction.
- Added no-API regression tests for two live retrieval rounds, legacy static compatibility, turn-cap final answering, Judge rank preservation, and Judge fallback never reintroducing a TrustRAG-removed document. All five tests and syntax checks passed.
- Corrected the Agentic BM25 attack scorer typo `math.og` → `math.log`; BM25 own-5 scoring can now run in the same live composition path.


2026-07-29: Started corrected original-Contriever PubMedQA TrustRAG clean-baseline rerun after identifying the prior binary-label historical outputs.
- The rerun first generates a fresh original-Contriever dot retrieval file with matching `.meta.json`, then runs only attack_method=None serially: TrustRAG followed by TrustRAG + medical semantic clustering.
- It uses the corrected PubMedQA yes/no/maybe closed-label contract, reserve candidate pool before filtering, dot scoring, top_k=5, and the valid gpt41mini attack-source payload only as the evaluation query source. No attack text is injected.
- Runs are isolated to GPU 4 and serial to avoid the prior memory-pressure failure. Earlier 64.0% / 64.8% results remain pre-fix historical observations and are not overwritten.

2026-07-29: Queued corrected v8 dot/own-5 full formal rerun.
- Reuses the existing metadata-verified v8 dot top-100 retrieval files for PubMedQA and MedQA; both report score_function=dot and the matching MIRAGE query files.
- Writes fresh results to results/query_results/formal_dot_v8_corrected/ without overwriting the historical v8 outputs. The suite runs PubMedQA and MedQA, each with None, LM_targeted, and HotFlip, using current closed-label parsing, own-5 injection, strict ASR, and valid attack sources.
- The queue waits for the current corrected original-Contriever TrustRAG clean-baseline process to finish, then runs one evaluation at a time on GPU 5 to avoid concurrent corpus loading and memory pressure.

2026-07-29: Adjusted the corrected v8 rerun to bounded two-task parallelism at user request.
- The prior v8 shell was only waiting and had written no result, so it was replaced without interrupting any evaluation.
- Current live tasks are original-Contriever dot retrieval for the corrected TrustRAG baseline on GPU 4 and v8 PubMedQA None evaluation on GPU 5. v8 retains one-at-a-time execution for its remaining five evaluations; no third corpus-loading task will be launched.
- At launch, MemAvailable was about 300 GiB and remained about 287 GiB after both tasks began; the previous memory-leak incident is therefore mitigated by a strict maximum of two simultaneous actual tasks and separate GPUs.

2026-07-29: Added an original-Contriever t-SNE diagnostic, queued with a memory gate.
- `scripts/visualize_original_contriever_tsne.py` deterministically samples five PubMedQA and five MedQA queries, then uses the original Contriever to embed each query, its original-Contriever top-5 normal chunks, five actual Q+I black-box texts, and five regenerated original-Contriever dot HotFlip texts.
- It fits one raw-vector t-SNE per benchmark across the five cases and exports five labeled local plots plus a five-case montage, raw vectors, point metadata (norm, dot, cosine, clean rank, mixed rank), selected attack inputs, and a reproducibility manifest.
- The visualization run uses the historical original-Contriever dot retrieval files requested by the user and waits for MemAvailable >= 280 GiB before using isolated GPU 6. It does not call an LLM or modify any evaluation result.

2026-07-29: Completed original-Contriever t-SNE diagnostic for five PubMedQA and five MedQA queries.
- Outputs are in results/embedding_viz/original_contriever_dot_tsne/: ten case plots, five-case montages, point CSVs, raw vectors, the selected black-box source, regenerated white-box own-5 texts, and manifest.json.
- Across the 25 own-5 texts per benchmark, HotFlip median mixed rank among the 15 local documents was 3 on both PubMedQA and MedQA (top-5: 20/25 and 23/25); black-box medians were 7 and 8 (top-5: 5/25 and 2/25); normal top-5 documents had median mixed rank 13 in both. This is a local original-Contriever dot diagnostic, not a task-answer ASR result.
- HotFlip also raised median vector L2 norm relative to normal chunks (PubMedQA 1.815 vs 1.586; MedQA 2.029 vs 1.662), while its median cosine was lower than black-box (PubMedQA 0.698 vs 0.791; MedQA 0.833 vs 0.952). The visualization therefore supports a dot-score/norm contribution in addition to semantic alignment.

2026-07-29: Started contriever-chinese_v1 training with the v8 recipe.
- Uses the same four validated v8 replay inputs (10,174 PubMedQA black-box, 10,174 PubMedQA HotFlip, 10,174 MedQA black-box Q+I, and 10,174 MedQA HotFlip Q+I), 8:8 replay, 3 epochs, seed 42, max_length 512, learning rate 1e-5, and weight_decay 0.01.
- Initializes from local /home/HF_Model/facebook/contriever-base-chinese and retains v8 training-time cosine/L2 vector normalization via CONTRIEVER_TRAIN_SCORE_FUNCTION=cos_sim. Output target is checkpoint/contriever-chinese_v1 on GPU 6.
- The first file-path invocation exited before model loading because the project package was not importable; it was immediately restarted through python -m contriever_stage1.train_replay. No checkpoint or evaluation result was overwritten.

2026-07-29: Completed contriever-chinese_v1 training and integrated the new model code.
- Training ended normally at 16:57:49. Epoch 3 was selected as best_model with a dual-domain replay selection score of 1.719473; epoch1/epoch2/epoch3 and root run_config.json/final_evaluation.json are present under checkpoint/contriever-chinese_v1.
- best_model contains the model weights, config, tokenizer, and selection.json. final_evaluation.json is a fixed sampled training-data diagnostic, not a formal PubMedQA/MedQA clean-accuracy or ASR result.
- Added contriever-chinese_v1 path resolution (including the local facebook Chinese base path) and CLI support in main.py, agentic_main.py, evaluate_beir.py, and gen_adv.py; README now identifies it as an English-medical-data language-transfer experiment using the v8 replay recipe.
- Python syntax and model-code mapping checks passed. No retrieval generation, attack generation, or answer evaluation was started.

2026-07-29: Completed contriever-chinese_v1 retrieval for the CSCO corpus + 75 course colorectal single-choice queries.
- Used datasets/csco_colorectal_2026/chunk (66 CSCO guideline chunks) and results/adv_targeted_results/course_colorectal_75.queries.json (75 aligned course:0000–course:0074 queries).
- Wrote a new independent raw-dot retrieval file to results/beir_results/formal_dot_contriever_chinese_v1/course_colorectal_75-csco_colorectal_2026-contriever-chinese_v1-dot.json with its matching .meta.json; model=contriever-chinese_v1, max_length=512, requested top_k=100.
- Verification passed: all 75 query IDs are covered, every query has all 66 available CSCO chunks, all scores are finite, and metadata matches dot/chinese_v1/csco/512/top100.
- No answer generation, attack injection/generation, or defense evaluation was started.

2026-07-29: Completed the contriever-chinese_v1 CSCO + 75 course colorectal single-choice no-defense baseline.
- Used the new metadata-verified dot retrieval file, course_colorectal_75.json/.ids, gpt4.1mini, top_k=5, M=75, repeat_times=1, and seed=12. No attack text was injected; Judge, TrustRAG, and medical semantic clustering were all disabled.
- Result: 75/75 parsed answers, 58/75 correct by parsed_pred_label == correct_option (clean accuracy 77.33%), 0 unparsed. Every question retained five retrieved CSCO chunks.
- The terminal ASR Mean=12.00% is only the fraction of clean predictions that happened to equal the attack source target_label (9/75); it is not an attack-success result and must not be reported as clean accuracy.
- Output: results/query_results/formal_csco_chinese_v1/csco_course75_contriever_chinese_v1_gpt41mini_noattack_dot.json; log: logs/contriever_chinese_v1/csco_course75_noattack_dot.log.

2026-07-29: Started no-defense black-box and white-box attacks for contriever-chinese_v1 on CSCO + 75 course colorectal MCQs.
- Both use the metadata-verified v1 raw-dot retrieval, the valid 75-query own-5 source, gpt4.1mini, top_k=5, strict MCQ target_label ASR, and no Judge/TrustRAG/medical clustering.
- Black-box LM_targeted completed before white-box launch; results are isolated under formal_csco_chinese_v1 and will be recorded with separately computed clean accuracy.
- White-box HotFlip is running serially on GPU 2 as 15 batches of 5 questions so each batch is saved after completion; it dynamically generates v1-specific own-5 texts from the current source rather than reusing historical Chinese-model attacks.

2026-07-29: Completed contriever-chinese_v1 CSCO + 75 MCQ no-defense black-box and white-box attack evaluations.
- Black-box LM_targeted: all 75 aligned queries parsed; strict target-label ASR=65/75=86.67%; actual accuracy by parsed_pred_label==correct_option=8/75=10.67%; mean own-5 attack documents retained in final top-5=3.5067 (retrieval F1=0.7013).
- White-box HotFlip: all 15 saved batches cover the same 75 unique queries; strict target-label ASR=72/75=96.00%; actual accuracy=3/75=4.00%; mean own-5 attack documents retained in final top-5=4.2800 (retrieval F1=0.8560).
- Both runs have 0 unparsed answers, all five final contexts per query, and no Judge/TrustRAG/medical-clustering defense. For both methods injected_adv count is bounded 0..5; source/query/retrieval ID sets match exactly. HotFlip texts were generated dynamically with contriever-chinese_v1 raw-dot gradients, not reused from historical models.
- Compared with the matching no-attack baseline (58/75=77.33%), black-box and white-box reduce actual accuracy by 66.67pp and 73.33pp respectively. Results: results/query_results/formal_csco_chinese_v1/csco_course75_contriever_chinese_v1_gpt41mini_{blackbox,hotflip}_dot.json.

2026-07-29: Completed corrected original-Contriever PubMedQA TrustRAG clean baselines (500 questions, no attack injection).
- TrustRAG: 367/500 correct (73.40%), 500/500 parsed under the corrected yes/no/maybe contract. The filter removed 9 documents across 3 queries.
- TrustRAG + medical semantic clustering: 378/500 correct (75.60%), 500/500 parsed; observed +2.20pp over TrustRAG. The medical filter actually removed only 2 documents in 1 query, so this one-run difference is not sufficient evidence to causally attribute the gain to clustering.
- Both runs use the corrected raw-dot top-5 pipeline, gpt4.1mini, and no Judge or attack text. Their terminal ASR fields (42/500 and 36/500 target-label matches respectively) are not attack-success metrics in no-attack runs and must not be reported as accuracy.
- Outputs: results/query_results/formal_dot_contriever_trustrag_corrected/formal_dot_pubmedqa_contriever_trustrag_corrected_None.json and results/query_results/formal_dot_contriever_trustrag_medcluster_corrected/formal_dot_pubmedqa_contriever_trustrag_medcluster_corrected_None.json.

2026-07-30: Queued the corrected original-Contriever PubMedQA TrustRAG attack matrix after confirming the clean-only rerun had no repaired attack outputs.
- The four evaluations are ordinary TrustRAG and TrustRAG + medical semantic clustering, each with LM_targeted black-box and HotFlip white-box attacks; they write only to new `_corrected` result directories and do not overwrite 7/28 historical files.
- Every task reuses the same metadata-verified fresh original-Contriever raw-dot retrieval as the corrected clean baselines, uses the valid 500 x 5 gpt41mini attack source, own-5 injection, strict target_label ASR, and the repaired PubMedQA yes/no/maybe contract.
- The queue performs source/meta validation and requires each finished output to contain exactly 500 unique IDs. It runs one GPU4 task at a time and waits for MemAvailable >= 220 GiB before each start; this serial policy is chosen because swap is full and the prior paired HotFlip run hit the memory guard.

2026-07-30: Added audit-only per-query attack-retention fields for the corrected TrustRAG attack outputs, so T-SNE case selection can prove an attack was actually filtered rather than merely absent from the reserve pool.
- New result fields record the number of own-5 attack texts in the pre-filter candidate pool, after original TrustRAG, after medical semantic clustering, each defense's removed-attack count, and the final retained count.
- This changes no encoder, score, ranking, filtering, prompt, or answer logic. The already-running first ordinary TrustRAG black-box process has its loaded code and remains unaffected; the following three queued evaluations will write the extra audit fields.

2026-07-30: Expanded the same audit-only TrustRAG output instrumentation with the exact own-5 texts and the attack-text subsets before filtering, after original TrustRAG, and after medical clustering.
- This makes each plotted point traceable to its actual retrieval/defense state without regenerating white-box text or inferring removal from a count. It still does not change the experiment algorithm or the in-memory first task; subsequent queued tasks will include it.

2026-07-30: Added an audited corrected-TrustRAG attack T-SNE diagnostic and queued it behind the remaining formal attack outputs.
- It will select up to three strong cases with own-5 attacks demonstrably entering the reserve pool, fully removed by the enabled filters, and a correct non-target answer; and up to three strong failures with a surviving reserve-pool attack and an exact target_label hit.
- Every selected case is plotted with query, the same ten normal dot-retrieval candidates, and the actual stored own-5 attack texts. Paired panels use original-Contriever embeddings and the TrustRAG CLS embedding space, with point states for not-in-pool, removed by TrustRAG, removed by medical clustering, reserve-retained, and final top-5.
- The visualization waits for the three subsequently audited attack outputs and MemAvailable >= 220 GiB, then runs alone on GPU6. It makes no LLM/API calls and cannot change any answer-evaluation result.

2026-07-30: Superseded the waiting corrected-TrustRAG T-SNE post-processing queue at the user's clarification; it was a sleeping visualization-only process and was stopped without touching the active attack evaluation.

2026-07-30: Started a separate historical original-Contriever T-SNE diagnostic from completed data.
- It selects five attack-target-hit and five attack-target-not-hit cases for each of PubMedQA and MedQA, requiring the completed historical black-box and HotFlip runs to agree with the valid attack-source target_label.
- The diagrams use historical original dot retrieval normal chunks, current valid Q+I black-box source texts, and reproducibly regenerated original-Contriever HotFlip own-5 texts. They are explicitly labeled as raw-dot/global-pool historical diagnostics, not byte-for-byte replays or formal own-5 results.
- MedQA has no completed original-Contriever TrustRAG result, so its two panels are attack-outcome groups rather than defense-success/failure groups. No LLM/API call is made; the diagnostic uses isolated GPU6 and the active GPU4 evaluation remains unchanged.

2026-07-30: Completed the historical original-Contriever outcome T-SNE diagnostic.
- Generated four five-case montages and 20 individual case images under results/embedding_viz/original_contriever_historical_outcomes_tsne/: PubMedQA target hit/not hit and MedQA target hit/not hit.
- Selection joins each result to the valid attack-source target_label and requires the completed historical black-box and HotFlip predictions to agree with the group. The fixed IDs are recorded in manifest.json and points.csv/raw vectors are saved for audit.
- Each case shows query, five historical normal dot-retrieval chunks, five Q+I black-box source texts, and five reproducibly regenerated original-Contriever HotFlip texts. The attack-text points are not asserted to be byte-for-byte historical replay inputs.
- This is explicitly a raw-dot/global-candidate-pool historical diagnostic. It is not a formal own-5/cosine result and does not establish modern defense effectiveness; in particular, no completed original-Contriever MedQA TrustRAG run exists.

2026-07-30: Completed the corrected original-Contriever PubMedQA TrustRAG attack matrix.
- The serial GPU4 queue finished naturally: ordinary TrustRAG and TrustRAG + medical semantic clustering, each under LM_targeted and HotFlip. No process was manually stopped or restarted.
- All four attack outputs passed completion checks: 500 rows, 500 unique IDs, exact ID-set alignment to the valid 500 x 5 gpt41mini attack source, and 500/500 parsed answers in the closed yes/no/maybe label space. Strict ASR below is recomputed against the attack-source target_label, not inferred from ordinary errors.
- Ordinary TrustRAG: LM_targeted accuracy 184/500=36.80%, strict ASR 179/500=35.80%, final own-5 F1 45.00% (1,125 retained attack texts); HotFlip accuracy 33/500=6.60%, strict ASR 349/500=69.80%, final own-5 F1 91.36% (2,284 retained attack texts).
- TrustRAG + medical semantic clustering: LM_targeted accuracy 180/500=36.00%, strict ASR 190/500=38.00%, final own-5 F1 44.32% (1,108 retained attack texts); HotFlip accuracy 32/500=6.40%, strict ASR 353/500=70.60%, final own-5 F1 91.24% (2,281 retained attack texts).
- The preceding corrected clean baselines remain: ordinary TrustRAG 367/500=73.40% accuracy and TrustRAG + medical clustering 378/500=75.60%, both fully parsed. Their target-label coincidences are clean diagnostics, not ASR.
- Audit fields were added only after the first ordinary TrustRAG LM_targeted process had loaded code, so that one completed output has no pre/post-filter attack audit fields. The later three outputs record five nonempty own-5 texts per query and their filter states. For LM_targeted + medical clustering, original TrustRAG removed 1,177 reserve-pool attack texts across 246 queries and medical clustering removed a further 22 across 5 queries; HotFlip filtering was minimal (ordinary: 3 texts across 1 query; with medical clustering: 3 plus 5 texts across one query each).

2026-07-30: Queued corrected original-Contriever MedQA TrustRAG evaluation at user request.
- Added scripts/run_contriever_medqa_trustrag_corrected_attacks_serial.sh; it runs one GPU4 evaluation at a time with a 220 GiB MemAvailable start gate and refuses to overwrite incomplete outputs.
- The six-run matrix is ordinary TrustRAG and TrustRAG + medical semantic clustering, each under None, LM_targeted, and HotFlip; it uses raw dot, own-5 injection, strict MCQ target_label ASR, gpt4.1mini, and the current closed option parsing.
- It reuses the archival original-Contriever MedQA retrieval results/beir_results/mirage_medqa_all-contriever.json rather than regenerating retrieval. The archival generation log verifies original Contriever, PubMed corpus, MIRAGE MedQA, raw dot, top-100, and max_length=128; the queue validates 1,273 source/retrieval/ID alignment, five nonempty own-5 texts per query, legal A/B/C/D target labels, and finite retrieval scores before starting.
- Outputs and logs are isolated under formal_dot_contriever_medqa_*_corrected and logs/formal_dot_contriever_medqa_defenses_corrected respectively.

2026-07-30: User authorized an exact legacy TrustRAG filter rollback and a clean MedQA restart.
- Terminated the active corrected MedQA queue process and its single no-attack evaluation child before any result JSON was written; existing files were preserved and no completed output was overwritten.
- Restored the legacy TrustRAG KMeans+ROUGE selection behavior in src/trustrag_filter.py. The compatibility min_keep argument remains accepted by current callers but is intentionally ignored, so the legacy filter may retain fewer than top_k documents as authorized.
- Kept the current evaluation integrity rules unchanged: raw dot, own-5, strict MCQ target_label ASR, A/B/C/D closed parsing, current answer-format contract, nonempty-response failures, and F1=0 when precision=recall=0. Only the TrustRAG document-selection behavior was rolled back.
- Added scripts/run_contriever_medqa_trustrag_legacy_attacks_serial.sh to rerun the complete six-run matrix in new legacy result/log directories. Python compilation, a no-API synthetic test proving legacy filtering can return fewer than top_k, and the five existing no-API regression tests passed.

2026-07-31: Stopped the legacy MedQA TrustRAG queue and aligned the implementation to the official TrustRAG repository.
- At the user's instruction, terminated the legacy queue parent and its active LM_targeted child before an LM result JSON was written; the two completed clean legacy result files were preserved and no completed output was overwritten.
- Replaced the TrustRAG core with the official `princeton-nlp/sup-simcse-bert-base-uncased` CLS encoder path, stemmed `rouge_score` ROUGE-L, and the official KMeans/cluster ordering and n-gram deletion branches. `rouge-score==0.1.2` is installed in the PoisonedRAG environment.
- Restored the official GPT conflict-query prompts exactly for all three stages. For this closed-label evaluation wrapper only, the required PubMedQA/MedQA output format is appended after the official final prompt; with no wrapper instruction, the three prompts compare exactly to the official source.
- Fixed the MedQA integration defect: option text is now included in each TrustRAG stage, rather than asking for an option letter without presenting the choices.
- Pure TrustRAG and TrustRAG+medical-clustering now filter only the original retrieval top-k (five), matching the official implementation; the medical clustering remains an explicitly extra post-filter and does not expand TrustRAG to top-10.
- No evaluation has been started after this alignment. Compilation, five existing no-API regression tests, an official-function filter comparison, and a three-stage prompt comparison passed.
- The official SimCSE weights are not yet present locally: direct Hugging Face and mirror LFS transfers stalled without payload, so the code deliberately has no Contriever fallback. A future run must first materialize the exact model at `/home/HF_Model/princeton-nlp/sup-simcse-bert-base-uncased`.

2026-07-31: Resolved the official TrustRAG SimCSE weight dependency without substituting another encoder.
- Hugging Face direct and hf-mirror.com TLS/LFS transfers reset or timed out; hf-mirror.net supported HTTP range requests and was used only after retrieving the official Git-LFS pointer.
- The pointer requires `pytorch_model.bin` size 437,998,343 bytes and SHA-256 `2834abc1b9961124fd0b3134c8cf0bc1144b5fc3a4f339d16d6b6c16d25c2004`.
- Downloaded to an isolated staging directory with resumable eight-way ranges, verified both exact size and SHA-256, then moved the complete checkpoint to `/home/HF_Model/princeton-nlp/sup-simcse-bert-base-uncased`.
- The final directory contains the official config, tokenizer files, vocabulary, and verified PyTorch checkpoint. Local `AutoTokenizer`/`AutoModel` loading succeeds (`bert`, hidden size 768, vocab 30,522).
- A CPU-only TrustRAGOriginalFilter smoke test completed successfully (ROUGE trigger and KMeans branch); no LLM call, retrieval, attack generation, or evaluation was started.

2026-07-31: Removed the medical-semantic-clustering retention floor at the user's instruction.
- MedicalSemanticClusterFilter and its shared embedding helper no longer require five documents to remain after deleting a high-cohesion suspicious cluster; the same removal behavior now applies in both main.py and agentic_main.py.
- Official TrustRAG behavior, raw-dot retrieval, own-5 injection, strict ASR, and answer evaluation are unchanged.
- A no-model synthetic check now removes a three-document high-cohesion cluster from five candidates and retains two; compilation and all five existing no-API regression tests passed. No experiment was started.

2026-07-31: Started the official-aligned TrustRAG medical-defense evaluation matrix at the user's request.
- Added scripts/run_trustrag_official_aligned_medical_matrix.sh, a single-GPU4 serial queue that uses existing raw-dot retrieval files and the validated own-5 attack sources; it never regenerates retrieval and refuses to overwrite incomplete outputs.
- The matrix contains PubMedQA and MedQA, each with ordinary TrustRAG and TrustRAG plus medical semantic clustering, under None, LM_targeted, and HotFlip: twelve evaluations total. Outputs/logs use new `official_aligned` names.
- Preflight verifies source/retrieval/ID-set equality, five nonempty own-5 texts per query, finite retrieval scores, and each dataset's closed label space. The queue is live; its first task is PubMedQA ordinary TrustRAG with no attack.

2026-07-31: Progress checkpoint for the official-aligned TrustRAG medical-defense matrix.
- Completed PubMedQA ordinary TrustRAG/no attack, PubMedQA TrustRAG+medical clustering/no attack, and PubMedQA ordinary TrustRAG/LM_targeted; each produced a new 500-row result JSON in its isolated official_aligned directory.
- The fourth task, PubMedQA TrustRAG+medical clustering/LM_targeted, is active. The serial queue has no reported error and will continue with the remaining PubMedQA HotFlip pair and all six MedQA evaluations.

2026-07-31: First completed results from the official-aligned TrustRAG medical-defense matrix (PubMedQA, 500 questions).
- Ordinary TrustRAG/no attack: 267/500 accuracy = 53.40%; all responses parsed. TrustRAG removed 308 retrieved contexts across the run.
- TrustRAG plus medical semantic clustering/no attack: 267/500 accuracy = 53.40%; all responses parsed. TrustRAG removed 308 contexts and medical clustering removed 27 additional contexts, with no net accuracy change in this run.
- Ordinary TrustRAG/LM_targeted: 241/500 accuracy = 48.20%; strict target-label ASR = 89/500 = 17.80%; final own-5 retrieval F1 = 2.76% (69 retained attack texts out of 2,146 entering the defense pool). The clean target-label coincidences are diagnostics only and are not reported as ASR.

2026-07-31: Additional completed results from the official-aligned TrustRAG medical-defense matrix (PubMedQA, 500 questions).
- TrustRAG plus medical semantic clustering/LM_targeted: 242/500 accuracy = 48.40%; strict target-label ASR = 94/500 = 18.80%; final own-5 retrieval F1 = 2.52% (63 retained attack texts out of 2,146 entering the defense pool). TrustRAG removed 2,077 attack texts and medical clustering removed 6 more; all responses parsed.
- Ordinary TrustRAG/HotFlip: 227/500 accuracy = 45.40%; strict target-label ASR = 110/500 = 22.00%; final own-5 retrieval F1 = 10.84% (271 retained attack texts out of 2,286 entering the defense pool). All responses parsed.

2026-07-31: Downloaded BIOS v3 and extracted it for the planned biomedical knowledge-graph guard.
- Source archives remain in datasets/BIOS_v3; verified extracted data is at /home/Dataset/BIOS_v3 (Concepts 2,469,974,447 bytes; Relations 5,297,938,217 bytes; Semtypes 548,305,792 bytes; 7.8G total). Each 7z archive passed its integrity test before extraction.

2026-07-31: Restored gen_adv_csco.py from the parent of deletion commit 12a497c at the user’s request.
- The restored root-level file matches historical blob 85492a84dab4e26cd50487c470c7ad388dee510d and passes Python syntax compilation; it remains untracked, with no commit created.

2026-08-03: Added CSCO course-context support at the user's request.
- The LLM-facing question now prepends a `clinical_context` patient-course field in main.py and agentic_main.py for ordinary RAG, TrustRAG, judge filtering, Reason-in-Docs, and Agentic RAG; retrieval and existing retrieval files still use the original question only.
- gen_adv_csco.py now includes `clinical_context` in the adversarial-text generation prompt whenever it is present.
- Result rows record whether clinical context was used and its character count. Syntax compilation and synthetic prompt tests passed.
- The current course_colorectal_75.json has 75/75 rows without `clinical_context`; the original 病程单选题-DeepSeek.docx is not present on the server, so no clinical text was fabricated or backfilled.

2026-08-03: Recovered the available CSCO course source and populated LLM clinical-context inputs.
- `病程单选题-DeepSeek.docx` was found in the project root (33,256 bytes). It is ignored by `*.docx` and has no Git blob/history, so no Git checkout was needed or possible.
- Parsed 15 case-summary headings and strictly matched all 75 Word questions to the existing course_colorectal_75.json order before updating it.
- Added a nonempty `clinical_context` to all 75 rows (15 distinct case summaries, five questions each), preserving each original question, options, labels, and existing adversarial texts.
- Saved the original JSON as results/adv_targeted_results/course_colorectal_75.pre_clinical_context_20260803.json (SHA-256 before update: 6df03bbba0ef01b87edef1c7961481b3914f162d0d4e09ffd3361bc04bbdc7ea). A prompt test confirms the current evaluation prompt contains both the matched course summary and question.

## 2026-08-03：MedCPT 本地索引与长度保护修复

- 修复 evaluate_beir.py 中错误置于 apply_tokenizer_max_length 末尾的无条件异常；无内置 query 的报错已移回 query fallback 分支。
- MedCPT 现在直接使用已加载的评测语料做命中 ID 校验，不再额外构造并重复常驻整套 PubMed 语料；本地 FAISS 索引可独立加载与检索。
- --max_length 被限制在 BERT/MedCPT 支持的 1..512；MedCPT 与 DPR fallback 都会显式传递该长度，且不再篡改模型的位置编码配置。
- MedCPT 本地索引失败时默认明确中止，不再静默退回 BEIR 全库重编码；只有显式传入 --allow-beir-fallback 才会退回。
- 已通过 py_compile 与 diff whitespace 检查；未启动正式检索或回答评测，未产生或覆盖实验结果。

## 2026-08-03：MedCPT 原生索引单题验证

- 初次单题验证发现 evaluate_beir.py 仍会在 MedCPT 前预加载完整 PubMed 分片；可用内存降至约 155 GiB，已主动停止该验证，未生成输出。
- 随后将 MedCPT 原生索引路径改为直接从 FAISS metadata 返回文档 ID；仅在显式允许 BEIR fallback 时加载完整语料，并修正该无语料路径严格只返回 top_k 篇。
- 使用本地 MedCPT Query Encoder、PubMed FAISS 索引、max_length=512、top_k=5、单条自定义 query 重新验证。原生路径正常完成，未发生 fallback，也未出现 50687 vs 512 错误。
- 结果包含 1 个 query 和恰好 5 篇文档；逐篇回查均与 datasets/pubmed/chunk 中的原始文档 ID 一致。
- 验证输出：results/beir_results/verification/medcpt_pubmed_single_query_max512.json 及同名 metadata；这不是正式实验结果。

## 2026-08-03：原始 Contriever 补充评测队列（已启动，等待资源）

- 用户明确要求：复用现有 raw-dot 检索文件，运行 PubMedQA 全样本（无攻击、无防御），并运行 MedQA 的原版 TrustRAG 黑盒 LM_targeted 与白盒 HotFlip（固定 n=60）。
- 已新增独立串行脚本 scripts/run_original_contriever_pubmedqa_clean_medqa_trustrag_n60_serial.sh；其先校验 PubMedQA 500 题 metadata（contriever + dot）及 MedQA 归档检索日志/输入，拒绝覆盖任何不完整结果。
- MedQA 的 n=60 定义为 mirage_medqa_all.json 通过固定 target-id 筛选后的前 60 条源文件顺序，seed=12、repeat=1、own-5、严格标签 ASR；未重建检索文件。
- 队列于 2026-08-03 14:14 启动，使用 GPU4 串行执行。启动时 MemAvailable 约 192 GiB，低于 220 GiB 安全阈值，因此当前仅等待，不占用 GPU；内存达阈值后才会开始第一项。

## 2026-08-03：BIOS 知识图谱三元组验证与风险重排（已落地，尚未用于正式评测）

- 依据 Alber et al.（Nature Medicine, 2025）及其公开代码新增 src/medical_kg_filter.py：零样本 LLM 将文本抽为 origin/relation/target；MedCPT Query Encoder 分别映射三部分；关系先匹配、实体再限制在该关系的合法端点集合中匹配，最后查询图中真实边。
- 新增 main.py 的 --medical_kg_filter 开关。KG 在 top-10 候选池内重排为最终 top-5；若叠加 TrustRAG，KG 先筛选至 5 篇，原版 TrustRAG 仍只接收 5 篇，避免将该组合误称为原版 TrustRAG。
- original 模式遵循原文：任一未验证三元组即使该文档风险为 1；conservative 模式将低语义匹配置信度的 unknown 视为中性，只有已可靠映射但图中不连通的三元组计入风险。结果逐题保存全部三元组、匹配概念/关系、相似度、风险及排序变化。
- 新增 scripts/build_bios_refined_kg.py，可从提供的规范 ground-truth JSON 构建工件，或从 /home/Dataset/BIOS_v3 的 Concepts/Relations 生成确定性的关系均衡 BIOS-v3 变体；元数据会明确标注后者并非论文未公开的原始精简图快照。
- 用原作者公开仓库的 data/test_graph.json 在 datasets/medical_kg/official_demo_smoke_20260803 完成端到端冒烟验证（14 concepts、5 relations、15 edges、MedCPT 768 维向量）；已验证 ibuprofen--may treat--stomach pain 可映射至有效 BIOS 边，ibuprofen--may treat--stomach ulcers 被拒绝。

### BIOS v3 公开数据重建状态

- 2026-08-03：医学 KG 的原文对齐模式名称已统一为 `original`；命令行参数、结果字段和内部接口不再使用旧名称。
- 2026-08-03：标准单轮入口的医学 KG 防御默认启用，默认模式为 `original`，默认工件指向已完成的 BIOS-v3 关系均衡工件；新增 `--no_medical_kg_filter` 可显式关闭。experiment_config.py 已补齐所有医学 KG 参数及其说明。
- 2026-08-03：README 已补充 BIOS 医学知识图谱防御的工作顺序、`original`/`conservative` 差异、默认参数、标准入口边界、关闭方式和公开 BIOS-v3 工件的适用范围。

- 对公开 BIOS v3 的 English PT（preferred term）严格预检表明，保留论文列出的 13 类关系时，边数超过论文的 416,302 阈值；构建器按默认安全策略中止，而非暗中截断。

## 2026-08-03：PubMedQA 医学 KG 全样本评测队列（已排队）

- 用户明确要求在 PubMedQA 全 500 题上比较医学 KG 单独启用与医学 KG 叠加原版 TrustRAG；均使用原始 Contriever、既有 raw-dot 检索、own-5 注入和黑盒 LM_targeted，不重建检索文件。
- 新增串行脚本 `scripts/run_pubmedqa_medical_kg_original_full_serial.sh`：先等待正在执行的 MedQA n=60 TrustRAG/HotFlip 队列结束，再按 KG-only、KG+TrustRAG 的顺序运行，且拒绝覆盖不完整结果。
- KG 固定使用 `original` 模式与 `datasets/medical_kg/bios_v3_preferred_sample_20260803` 工件；每组完成校验 500 个唯一题目、闭集标签、KG 确实启用/应用及 TrustRAG 开关状态后才标记完成。
- 2026-08-03 15:14：前序 MedQA n=60 白盒队列已完成，KG-only 组已开始；KG+TrustRAG 组将在其完整输出通过校验后自动接续。

## 2026-08-03：MedQA 医学 KG 全样本评测队列（已排队）

- 用户要求按 PubMedQA KG 实验的同一口径补跑 MedQA 全 1273 题；固定原始 Contriever、既有 raw-dot 检索、own-5 注入、黑盒 LM_targeted 和 KG `original`，分别比较 KG-only 与 KG+原版 TrustRAG。
- 新增串行脚本 `scripts/run_medqa_medical_kg_original_full_serial.sh`。它先完整校验 MedQA 1273 题输入、历史 raw-dot 检索日志和 KG 工件，再等待当前 PubMedQA KG 队列完成后运行两组，不重建检索文件且拒绝覆盖不完整输出。

## 2026-08-03：医学 KG 三元组缓存（已落地）

- `LLMTripletExtractor` 现以“截断后的文档文本、抽取长度、最大三元组数和命名空间”的 SHA-256 作为缓存键，逐篇追加到 `results/medical_kg_caches/*.jsonl`；每组结束后缓存都保留，并可在后续同源 KG-only / KG+TrustRAG 评测中读取。
- 缓存只包含不可逆的文档内容哈希和抽取三元组；不写入问题、答案、金标、攻击目标、检索分数、排序或最终预测，因此不会将标签或评测结论从一个实验泄漏到另一个实验。
- 新增 `medical_kg_triplet_cache_path` 与 `medical_kg_triplet_cache_namespace` 配置/命令行参数；后者用于在更换三元组抽取模型、提示词或输出格式时隔离旧缓存。已通过医学 KG 三项单元检查、配置语法和 whitespace 检查。
- 已启动的 PubMedQA KG-only 进程在代码更新前加载，保持原参数不被中途改变；其后的任务将自动使用缓存。

## 2026-08-04：PubMedQA 与 MedQA KG 实验并行化（已获用户许可）

- MedQA 队列此前仅在等待 PubMedQA 队列，尚未启动任何 MedQA 评测。按用户许可，已将其改为立即使用物理 GPU1；运行中的 PubMedQA KG-only 保持在物理 GPU4，未被中断或修改。
- 同一数据集内的 KG-only 与 KG+原版 TrustRAG 仍串行，以避免写同一个三元组缓存；PubMedQA 与 MedQA 的缓存路径由各自对抗文本来源自动区分。两队列仅共享只读输入和 API 服务，不共享结果、标签或缓存写入。
- 2026-08-04 14:46：MedQA KG-only 已完成输入与工件校验并启动；该进程通过 `CUDA_VISIBLE_DEVICES=1` 使用物理 GPU1，PubMedQA KG-only 继续使用物理 GPU4。

## 2026-08-04：KG 安全分权重调整

- 根据用户要求，集中实验配置将 `medical_kg_rerank_weight` 从 0.20 调整为 0.50，使归一化检索分数与 `(1 - KG 风险)` 安全分数按 5:5 合成。该参数只影响此后通过配置启动的新实验。
- 当前执行中的 MedQA KG-only 已在旧参数 0.20 下启动，保持不变，避免中途改变实验口径；0.50 仍是风险重排而非论文式硬过滤。

## 2026-08-04：PubMedQA KG 5:5 全样本重跑（已启动）

- 用户要求重跑 PubMedQA。旧的 0.20 KG-only 结果保留在原目录；新增独立队列以 `medical_kg_rerank_weight=0.50` 跑全 500 题的 KG-only 与 KG+原版 TrustRAG，复用既有 Contriever/raw-dot 检索及 own-5 黑盒攻击文本，不重建检索文件。
- 新队列的完成校验允许空的 `parsed_pred_label`：空输出仍按严格 ASR/准确率规则计错，但 500 条完整逐题记录视作完整输出，禁止因该单题重新覆盖结果。逐题结果新增保存实际 `medical_kg_rerank_weight`，确保两组可审计比较。

## 2026-08-06：PubMedQA KG 8:2 黑盒/白盒无 TrustRAG 对照（已启动）

- 用户要求将 KG 安全分 : 检索分数提高到 8:2。集中配置的 `medical_kg_rerank_weight` 相应更新为 `0.80`；已经启动的实验均以其命令行参数运行，不受此变更影响。
- 新增独立串行队列，固定原始 Contriever、既有 raw-dot 检索、PubMedQA 全 500 题、own-5、KG `original`，依次运行黑盒 `LM_targeted` 和白盒 `hotflip`；两组均显式不启用 TrustRAG，使用独立结果/日志目录，绝不覆盖此前 5:5 结果。
- 两组显式复用 `mirage_pubmedqa_all_gpt41mini_contriever_triplets.jsonl`。该缓存仅按文档哈希复用三元组；HotFlip 新生成且未命中的文档将追加自身三元组，不包含题目、标签、排序或预测，因而不造成评测泄漏。
- 2026-08-06 11:23：黑盒 `LM_targeted` 已在物理 GPU4 启动；通过 500 题输入、raw-dot 检索、BIOS 工件和既有缓存的预检。该组完成后，队列将以相同的 8:2 参数启动白盒 `hotflip`。

## 2026-08-06：问题相关的连续医学 KG 风险（已实现，默认关闭）

- 按用户给定公式新增可选 `medical_kg_risk_model="evidence_weighted"`：`R(d,q)=0.45(1-C_KG)+0.40*C_conflict+0.15*A_pattern`，其中 `C_KG=0.45*E_entity+0.35*P_path+0.20*T_type`；两个权重组均在运行前校验和为 1。
- `E_entity` 由问题（含临床上下文）与文档三元组实体的精确包含或 MedCPT 语义相关度计算；`P_path` 由 BIOS 受限端点的有效关系边计算；`T_type` 由关系端点角色约束计算。每项及最终 0～1 风险均写入逐文档审计结果。
- `C_conflict` 仅在文档显式否定某关系、且 BIOS 以高置信正向边支持相同实体对时为 1；图谱查不到、低置信映射和覆盖不足均保持 0，绝不以“没有证据”推断为医学冲突。当前未接入额外的结构化指南证据源。
- `A_pattern` 在同一 reserve pool 内结合重复实体—关系—实体签名和高于阈值的文本同质性；其只提供连续风险，不调用或替代现有医疗语义聚类过滤。
- 旧 `binary` 风险仍为默认，原有 `original` / `conservative` 规则及当前运行任务的风险和排序不变。新增 CLI、集中配置、README 和三项单测覆盖连续支持、显式冲突、未知覆盖非冲突以及候选模式分。

## 2026-08-06：PubMedQA 连续风险 8:2 黑盒/白盒无 TrustRAG 对照（已启动并等待）

- 用户要求以刚实现的 `evidence_weighted` 风险公式重跑上一轮 PubMedQA 对照。新队列固定完全相同的原始 Contriever、raw-dot、全 500 题、own-5、KG `original`、安全:检索=8:2、无 TrustRAG，并依次运行黑盒 `LM_targeted` 与白盒 `hotflip`；唯一实验变量是风险模型。
- 队列会在现有二值风险黑白盒队列结束后才开始，避免两个进程并发追加同一份三元组缓存；随后显式复用该缓存。结果目录、日志目录和完成校验均与二值风险结果隔离，且校验 `medical_kg_risk_model="evidence_weighted"`。
- 2026-08-06 15:22：连续风险队列已启动，检测到二值风险队列仍在执行，现处于安全等待状态；不会抢占 GPU 或读取/写入正在更新的缓存。

## 2026-08-07：PubMedQA 1:0 二值风险直接剔除黑盒/白盒对照（已启动并等待）

- 新增 `medical_kg_decision_mode="hard_filter"` 与 `medical_kg_hard_filter_threshold`。在 `original` + `binary` 下阈值为 1.0 时，任何已识别恶意文档会直接移出候选池，保留文档按原始检索分排序，且不再执行“至少保留 5 篇”的回填。
- 用户要求以该 1:0 口径重跑 PubMedQA 全 500 题、原始 Contriever/raw-dot、own-5、无 TrustRAG 的黑盒 `LM_targeted` 与白盒 `hotflip`。新队列将等待连续风险队列完整结束后再运行，共享安全的三元组缓存但不并发写入；输出审计会记录直接剔除数量及每篇文档的 `hard_filtered` 标记。
- 2026-08-07 14:34：直接剔除队列已启动，正在等待连续风险队列结束；未抢占正在运行的 GPU 或缓存文件。

## 2026-08-10：撤销连续医学 KG 风险公式

- 按用户要求删除 `evidence_weighted` 连续风险公式，以及对应的 KG 支持度、显式冲突、候选模式参数、命令行入口、集中配置、README 说明、单元测试和专用实验队列脚本。
- 标准 KG 防御保留原有 `original` / `conservative` 二值判定和风险重排；`medical_kg_decision_mode="hard_filter"` 的 1:0 直接剔除功能继续保留。
- 已完成的连续公式实验输出和日志保留为不可变历史记录，便于追溯既有结论；它们不再能通过当前代码或脚本重新运行。
- 直接剔除队列脚本已取消对连续风险队列的等待依赖，并移除了已废弃的风险模型参数校验。

## 2026-08-10：新增 KG 3:7 + TrustRAG 与 MedQA 1:0 对照（已排队）

- PubMedQA/PoisonedRAG：沿用原始 Contriever、既有 raw-dot 检索、own-5、严格 ASR 和文档哈希三元组缓存，运行 `LM_targeted` 黑盒攻击；医学 KG 设为 `original`，安全分:检索分=3:7（`medical_kg_rerank_weight=0.30`），并开启原版 TrustRAG。
- MedQA：沿用归档的 Contriever/raw-dot 检索、own-5、严格 ASR 和 MedQA 独立三元组缓存，运行 `LM_targeted` 黑盒与 `hotflip` 白盒攻击；医学 KG 设为 `original` + `hard_filter`，阈值为 1.0（安全:检索=1:0），不启用 TrustRAG。
- 两队列使用不同数据集的缓存和不同 GPU（PubMedQA: GPU4；MedQA: GPU7），不共享可写缓存、不覆盖历史输出；每一组完成后先进行 500/1273 题完整性和配置校验。

## 2026-08-10：新增 GPT-5 mini 模型配置

- 新建 `model_configs/gpt-5-mini_config.json`，可通过 `--model_name gpt-5-mini` 自动加载；模型提供商保持 GPT，模型名设为 `gpt-5-mini`。
- 连接信息以不回显的方式沿用 `gpt4.1mini_config.json` 的 API Key 与 Base URL；未读取、打印或记录任何凭据值。
- 2026-08-10：确认项目没有引用旧文件名后，按用户要求删除旧的 `model_configs/gpt5mini_config.json`，仅保留可由 `--model_name gpt-5-mini` 自动加载的配置。

## 2026-08-10：GPT-5 mini 无攻击、无防御基线（已排队）

- 新增全样本串行队列：PubMedQA 500 题、MedQA 1273 题，模型为 `gpt-5-mini`，固定原始 Contriever 和既有检索文件；不注入对抗文本。
- 每次运行显式传入 `--no_medical_kg_filter`，且不传 TrustRAG、医疗语义聚类或 Judge 开关；完成校验要求四项防御字段均为 False，避免默认 KG 防御造成基线污染。
- 队列会先等待当前 PubMedQA KG+TrustRAG 与 MedQA KG 直接剔除队列完成，再在 GPU4 上依次运行两套基线；避免 GPU、主机内存和 API 资源争抢，也不覆盖任何历史结果。

## 2026-08-10：README 医疗 KG 文档更新

- 扩充 README 的 BIOS 医学知识图谱章节：补充原始检索候选到 KG/TrustRAG/回答的固定顺序、`original` 与 `conservative` 的二值风险差异、标准入口边界和连续公式已移除的状态。
- 添加 3:7 KG+TrustRAG、1:0 直接剔除和无防御基线的集中配置示例；补充三元组缓存的安全复用边界及结果审计字段，便于实验复现与配置核验。

## 2026-08-10：GPT-5 mini 基线优先执行

- 按用户要求，将尚未开始、仅等待前序队列的 GPT-5 mini 清洁基线提前。旧等待进程未启动 `main.py`、未写入结果，安全终止后以同一独立结果目录重启。
- 基线新增 `--run-now`：跳过前序队列等待，改用 GPU7；基线不启用 KG、TrustRAG、医学语义聚类或 Judge，资源需求较低，主机可用内存阈值从 220 GiB 下调至 160 GiB。PubMedQA 与 MedQA 仍在该队列内串行运行。
- 当前 KG 实验继续保持原参数和各自的 GPU；基线与其不共享可写 KG 缓存，也不覆盖任何历史结果。

## 2026-08-11：API 额度中断后的实验暂停与基线核验

- PubMedQA 的 KG 3:7 + TrustRAG 黑盒在第 78/500 题首次 API 余额不足时因 TrustRAG 强制阶段返回空文本而中止；前 77 题未形成完整结果 JSON，不能作为正式指标。
- 按用户要求暂停 MedQA KG 1:0 黑盒：队列脚本和 `main.py` 子进程均处于 stopped 状态，最后成功处理第 517 题，已进入第 518 题；其后的 HotFlip 不会自动启动。
- GPT-5 mini 的无攻击、无防御基线已写出全量 JSON，但存在空响应。按 `output_poison` 非空过滤后：PubMedQA 保留 342/500，19 题正确，准确率 5.56%；MedQA 保留 468/1273，0 题正确，准确率 0.00%。这些结果仍应标注为 API/模型输出异常下的非正式诊断结果。

## 2026-08-11：GPT-5 mini 无防御基线重跑（r2）

- 按用户要求重跑 PubMedQA 500 题和 MedQA 1273 题的 GPT-5 mini 清洁基线；结果与日志写入独立 `r2` 目录，不覆盖上一轮异常输出。
- 基线脚本支持 `BASELINE_RUN_TAG` 以隔离重跑目录，并通过 `BASELINE_REQUIRE_NONEMPTY_LABELS=1` 要求每题原始响应与闭集预测标签都非空；若 API 再次产生空响应，输出会被完整性校验拒绝，不会被当作成功实验。
- 已暂停的 MedQA KG 黑盒及其后续 HotFlip 保持停止状态，重跑仅启动 GPT-5 mini 无攻击、无防御基线。

## 2026-08-11：MedQA KG 1:0 黑白盒重启（r2）

- 按用户要求，先前停在第 518/1273 题的 MedQA KG 1:0 黑盒队列已被明确终止；其未产出完整结果 JSON，不能继续或作为正式结果使用。
- 新建独立 r2 队列，从头依次运行黑盒 `LM_targeted` 与白盒 `hotflip`。实验口径不变：MedQA 全 1273 题、原始 Contriever/raw-dot、own-5、严格 ASR、医学 KG `original` + `hard_filter` 阈值 1.0、不开启 TrustRAG；沿用仅含文档哈希与三元组的 MedQA 缓存。
- r2 使用独立结果和日志目录，在物理 GPU4 启动黑盒阶段；GPT-5 mini 清洁基线继续在 GPU7 上运行。r2 完整性校验进一步要求每题均有非空原始回答且可解析为 A–D，空 API 输出不会被误判为完成。

## 2026-08-12：GPT-5 mini 空响应修复与无防御基线重跑（r3）

- r2 的 PubMedQA 清洁基线虽完成 500 题，但有 169 题空输出，严格完整性校验拒绝该产物；其后 MedQA 未启动。空输出在题目序号和提示词长度上均匀分布，日志也无 API 异常，原因定位为项目适配器仍用 `max_tokens=300`，未为 GPT-5 reasoning 模型配置推理强度或记录完成原因。
- 依据官方 GPT-5 Chat Completions 约定，适配器只对 `gpt-5*` 改用 `max_completion_tokens`（其上限包含 reasoning token），并传入 `reasoning_effort`；保留其他模型的旧 `max_tokens` 分支，确保运行中的 gpt4.1-mini MedQA 实验不受影响。
- `gpt-5-mini` 配置设为 `max_output_tokens=2048`、`reasoning_effort=minimal`，空文本时最多重试两次，重试预算为 8192。日志会审计空输出的尝试序号、完成原因、总 completion token 与 reasoning token，不回显提示词或凭据。
- 在 3 个 r2 中曾为空的真实 PubMedQA 提示词上校验通过：3/3 均返回非空、合法的 `yes/no/maybe`。按用户要求启动独立 r3 队列：PubMedQA 500 题完成并通过非空校验后，再运行 MedQA 1273 题；使用 GPU7，与 GPU4 上的 MedQA KG 1:0 黑盒实验并行，输出和日志不覆盖 r1/r2。

## 2026-08-12：BIOS KG 临床关系优先抽样（已完成）

- 新增可复现的 `clinical-priority` 构建策略与版本化目标工件 `datasets/medical_kg/bios_v3_clinical_priority_20260812`，不覆盖历史关系均衡工件 `bios_v3_preferred_sample_20260803`，因此不影响正在运行的 MedQA r2 旧图实验。
- 高优先级关系全量保留：`may treat`、`may be treated by`、`contraindication`、`has adverse effect`、`is adverse effect of`、`interacts with`、`may diagnose`、`may be diagnosed by` 与 `differential diagnosis`（BIOS 原始标签 `ddx`）。中优先级 `may cause` / `may be caused by` 上限各 100,000，`is a` 上限 150,000；低优先级 `associated with` 配额为 0，且不参与严格风险判定。
- 从 `/home/Dataset/BIOS_v3/` 实际筛得 1,184,444 个英文 PT 术语、817,438 条采样候选边。写入时按规范化文本三元组去重，最终工件为 173,845 个概念、643,635 条有效唯一边；除 `is a`（628,698 条中确定性抽取 150,000 条）外，中优先级两类关系的英文 PT 边数均低于配额，故实际均全量保留；原始文件中 `associated with` 的英文 PT 可用边为 0。
- KG 审计新增 `ignored` 状态与 `medical_kg_ignored_triplets`：新工件元数据中的 `non_strict_relations=["associated with"]` 会使这类抽取断言保留在审计中、但不因图中缺边被判为 `unknown` 或 `invalid`。旧工件缺少该元数据，因此其行为不变。
- 新工件已完成构建并可加载：`sampling_policy="clinical-priority"`、12 种严格关系、`non_strict_relations=["associated with"]`；元数据同时记录采样前计数、确定性配额、写入前候选数和去重后的逐关系有效边数。`raw_edge_cap` 为 `null`，避免把仅供旧策略使用的 416,302 上限误解为本策略限制。
- 已通过静态编译、工件加载/计数一致性验证和手动单元验证：覆盖原有图验证、原始模式 1:0 硬过滤、非严格关系中性处理，以及高全保留/中配额/低排除的确定性抽样。服务器环境未安装 pytest，故直接调用全部测试函数完成验证。
- `main.py` 与 `experiment_config.py` 的后续默认工件均已切换到该新版本；正在运行的 MedQA r2 通过命令行显式指定旧路径，仍保持旧工件、不受切换影响。

## 2026-08-12：PubMedQA 新临床优先 KG 1:0 对照（已启动）

- 按用户要求启动独立 PubMedQA 全量队列：`gpt4.1mini`、原始 Contriever/raw-dot、500 题、own-5、严格 ASR、`bios_v3_clinical_priority_20260812`、`original + hard_filter=1.0`（严格删除）、不启用 TrustRAG 或医疗语义聚类。
- 队列依次运行 LM_targeted 黑盒与 HotFlip 白盒，使用 GPU0；结果目录和日志目录均带 `clinical_priority_kg...20260812`，不覆盖旧图 1:0 结果。三元组缓存复用 `mirage_pubmedqa_all_gpt41mini_contriever_triplets.jsonl`；其只按文档哈希保存抽取三元组，不含 KG 判定或答案，故跨 KG 工件安全复用，且当前没有其他 PubMedQA 进程写该缓存。
- 启动前已核验 500 个题目/对抗文本/检索候选、raw-dot 元数据、新工件的 `clinical-priority` 元数据与 `associated with` 非严格标记；结果完整性校验要求 500 个唯一题目、非空合法 yes/no/maybe 输出、KG 确实应用、严格删除阈值为 1.0，且 TrustRAG/医疗聚类均关闭。

## 2026-08-12：GPT-5 mini 无攻击无防御基线 r3（已完成）

- PubMedQA 500/500 与 MedQA 1273/1273 均完成，所有原始输出与闭集标签非空且合法；防御项（KG、TrustRAG、医疗语义聚类、Judge）均确认为关闭。
- 按闭集预测与金标计算，PubMedQA 准确率为 349/500 = 69.80%，MedQA 准确率为 1024/1273 = 80.44%。日志中的“ASR”只是无攻击设置下恰好输出 `target_label` 的比例，不是攻击成功率，不能用作正式 ASR。

## 2026-08-12：PubMedQA 新临床优先 KG 1:0 的 GPT-5 mini 黑白盒对照（已启动）

- 按用户要求，以与正在运行的 gpt4.1-mini PubMedQA 队列一致的实验口径，新增 `gpt-5-mini` 的 LM_targeted 黑盒与 HotFlip 白盒队列：原始 Contriever/raw-dot、500 题、own-5、严格 ASR、`bios_v3_clinical_priority_20260812`、`original + hard_filter=1.0`、不开启 TrustRAG 或医疗语义聚类。
- 使用独立 GPU2、独立结果/日志目录，以及独立三元组缓存 `mirage_pubmedqa_all_gpt5_mini_contriever_triplets.jsonl` 和命名空间 `triplet_schema_v1_gpt5_mini`；因此不会与 gpt4.1-mini 的 PubMedQA 缓存或旧图 MedQA 缓存并发写入。启动前已通过全量输入、KG 元数据和输出路径预检。

## 2026-08-12：GPT-5 mini 黑白盒任务修正为无防御基线（已启动）

- 用户澄清仅需要 GPT-5 mini 的黑盒/白盒**无防御基线**。此前误启动的新 KG 严格删除队列在仍处于语料加载阶段即停止，未生成任何结果 JSON；gpt4.1-mini 新 KG 队列和 MedQA 队列均未受影响。该错误队列的独立空缓存文件保留，不覆盖或删除任何数据。
- 已启动正确的独立队列：PubMedQA 500 题、原始 Contriever/raw-dot、own-5、严格 ASR、`gpt-5-mini`，顺序运行 LM_targeted 黑盒和 HotFlip 白盒；显式 `--no_medical_kg_filter`，且 TrustRAG、医疗语义聚类、Judge 均关闭。输出校验要求每题非空合法 yes/no/maybe 标签及四项防御全部关闭。
- 当前 gpt4.1-mini 与 GPT-5 mini 的既有无攻击结果不能据此判断前者模型能力更强：历史 gpt4.1-mini PubMedQA 使用的是另一份检索/对抗输入路径及旧版解析适配器，65/500 题未解析为闭集标签；GPT-5 mini r3 使用的是当前 raw-dot 输入、修复后的 GPT-5 适配器和强制非空校验。应在相同检索文件、问题集、提示词、标签解析器和无防御开关下重跑才可作模型比较。

## 2026-08-12：历史 gpt4.1-mini 无攻击无防御基线准确率核验

- 历史全样本基线（原始 Contriever/dot、top-5、无攻击、Judge 关闭）为：PubMedQA 361/500 = 72.20%，MedQA 1022/1273 = 80.28%。
- PubMedQA 的原始回答均非空，但历史标签解析器有 65/500 题未生成闭集标签；若只在 435 个已解析标签上计算，准确率为 361/435 = 82.99%。正式跨模型对比应优先使用全样本 72.20%，并同时报告解析缺失。MedQA 1273 个标签均可解析，无该问题。

## 2026-08-12：gpt4.1-mini 与 GPT-5 mini 历史 PubMedQA 基线可比性核验

- 两次结果覆盖相同的 500 个 PubMedQA 问题，500 个金标答案也完全一致；因此数据集本身不是差异来源。
- 但历史 gpt4.1-mini 运行使用 `results/beir_results/mirage_pubmedqa_all-contriever.json`，GPT-5 mini r3 使用 `results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json`。逐题比较显示 top-5 与 top-10 候选在 500/500 个问题上均不完全相同，继而最终 `input_prompt` 也在 500/500 个问题上不相同；这是最主要的非模型变量。
- 两份问题源中的问题文本 500/500 相同，但对抗文本 500/500 不同。无攻击基线不会注入对抗文本，故该项不是主因；它仍表明两次运行并未固定同一输入工件。历史命令还以 `model_name='palm2'` 加载 `gpt4.1mini_config.json`，使用旧标签解析器与 loose ASR；GPT-5 mini r3 使用当前 `--model_name gpt-5-mini` 入口、重试与强制非空标签校验。历史 gpt4.1-mini 有 65 个空标签，GPT-5 mini 为 0。
- 在双方均有标签的 435 题上，预测恰好一致的只有 326 题；因 prompt 不同，这 109 个差异不能归因于模型本身。公平结论需要以同一 raw-dot 检索、相同提示词/解析代码、同一无防御开关，仅替换模型后重跑。

## 2026-08-12：gpt4.1-mini 对齐 GPT-5 mini r3 的 PubMedQA 无攻击基线（已启动）

- 按用户要求启动新的公平模型对照：`gpt4.1mini` 使用与 GPT-5 mini r3 完全相同的 500 个目标 id、`mirage_pubmedqa_all_gpt41mini.json` 问题源、`formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json` 检索文件、Contriever/raw-dot、top-5、seed=12、当前代码、严格标签口径、无攻击以及 KG/TrustRAG/医疗聚类/Judge 全部关闭；唯一有意变量为回答模型。
- 任务使用空闲 GPU3、独立输出和日志目录，不读写 KG 三元组缓存，不影响正在运行的 gpt4.1-mini KG 队列、GPT-5 mini 黑白盒无防御队列或 MedQA 队列。完成校验强制 500 个唯一题、所有回答和 yes/no/maybe 标签非空、四项防御均为关闭。

## 2026-08-12：PubMedQA 历史与当前检索文件核验

- 历史文件为 `results/beir_results/mirage_pubmedqa_all-contriever.json`（5 月生成）；当前正式文件为 `results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json`（7 月生成）。两者声明的配方相同：Contriever、dot 相似度、相同的 500 题查询 JSON、`max_length=128`、每题 top-100。
- 历史文件未保留可核验的 sidecar 元数据；当前正式文件带有上述配置元数据，因而更可复现。两个文件均有 500 个题目、每题 100 篇候选，但实际排序显著不同：top-1 仅 360/500（72%）相同；top-5 平均交集 2.198/5（仅 4 个题目集合完全相同，顺序完全相同为 0）；top-10 平均交集 4.226/10，top-100 平均交集 44.11/100。
- 因此不是“新旧文件仅改了路径/名称”。在不改变问题、模型、相似度度量和名义长度/候选数的前提下，检索语料切分/索引、文档集合或检索实现版本至少有一项已变化；仅凭现有旧工件不能进一步唯一确定原因。当前正式实验统一使用带元数据的 `formal_dot_contriever` 文件；旧文件只用于复核历史结果，二者不应混作公平模型对照的输入。

## 2026-08-13：默认 BIOS KG 改为全量无抽样（构建预检中）

- 构建器的默认策略已从 `clinical-priority` 改为 `no-sampling`。新策略保留公开 BIOS v3 中英文 preferred-term 节点之间、当前医学三元组验证使用的全部 13 类关系边：不设置关系配额、不作 reservoir 抽样、不再排除 `associated with`；仍保留英文 PT 筛选，作为论文“去同义词节点”预处理的可复现近似。
- 新工件将写到独立路径 `datasets/medical_kg/bios_v3_english_pt_full_20260813`，并已改为 `main.py`、`experiment_config.py` 和 README 的默认值。历史 `bios_v3_clinical_priority_20260812` / `bios_v3_preferred_sample_20260803` 及所有通过命令行显式指定它们的在运行实验不受影响。
- `--max-edges` 现为可选的“只中止、不抽样”安全保护；默认不设上限，防止全量模式因旧的 416,302 条遗留上限而提前失败。新增测试覆盖全量模式必须保留每一条所选关系边；旧临床优先测试和新增全量测试均已在服务器的 PoisonedRAG 环境通过。
- 对 `/home/Dataset/BIOS_v3/` 的只读 dry-run 已完成：13 类关系的源边为 1,296,136 条，文本规范化去重后为 1,109,060 条有效边、454,746 个概念；旧临床优先工件为 643,635 条边、173,845 个概念。源数据内 `associated with` 的英文 PT 边实际为 0，因此全量策略不会凭空补造该关系。
- 完整工件已于 2026-08-13 构建完成：使用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0`）、MedCPT Query Encoder、batch size 128，输出到独立目录 `datasets/medical_kg/bios_v3_english_pt_full_20260813`。完整性校验通过：`sampling_policy="no-sampling"`、`concept_count=454746`、`edge_count=1109060`、`raw_edge_cap=null`，六个加载必需工件均存在。旧工件未被覆盖，GPU 0 / GPU 4 实验未被该构建占用。
- 相关功能代码、测试与可复现实验脚本已于 2026-08-13 提交并推送到 GitHub `main`：提交 `4423233`（`Add reproducible medical RAG defenses and full BIOS KG`）。提交明确排除含凭据的 `model_configs/`、运行日志/缓存/数据工件、外部参考仓库副本和备份文件。
