# Purpose: Implements src/attack_for_understanding.py in the PoisonedRAG project.

# 从 sentence_transformers 导入 SentenceTransformer（用于检测模型类型）
from sentence_transformers import SentenceTransformer
# 导入 PyTorch
import torch
# 导入随机数模块
import random
# 导入 tqdm 进度条
from tqdm import tqdm
# 从 src.utils 导入 load_json 函数
from src.utils import load_json
# 导入 JSON 模块
import json
# 导入操作系统模块
import os


class GradientStorage:
    """
    该对象用于存储指定 PyTorch 模块输出的中间梯度。
    通常用于获取嵌入层的梯度，以进行 HotFlip 攻击。
    """
    def __init__(self, module):
        # 初始化存储梯度的变量为 None
        self._stored_gradient = None
        # 在模块上注册一个后向钩子，钩子函数会在反向传播时被调用
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        # 钩子函数：将模块输出的梯度（grad_out[0]）存储起来
        self._stored_gradient = grad_out[0]

    def get(self):
        # 返回存储的梯度
        return self._stored_gradient


def get_embeddings(model):
    """返回模型的词嵌入模块（word_embeddings）"""
    # 对于 SentenceTransformer 模型，需要访问其内部的 auto_model
    if isinstance(model, SentenceTransformer):
        # SentenceTransformer 的第一个子模块通常是 transformer 模型
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        # 对于普通 HuggingFace 模型，直接访问 embeddings.word_embeddings
        embeddings = model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """
    根据平均梯度选择 top-k 个候选 token 用于替换。
    - averaged_grad: 形状 (embedding_dim,) 的梯度向量
    - embedding_matrix: 形状 (vocab_size, embedding_dim) 的词嵌入矩阵
    - increase_loss: 如果为 True，选择使损失增大的 token；否则选择使损失减小的 token
    - num_candidates: 返回的候选 token 数量
    - filter: 可选的惩罚项（形状与 embedding_matrix 相同），减去该值以排除某些 token
    """
    with torch.no_grad():
        # 计算梯度与每个词向量的点积，得到每个 token 的得分
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,   # embedding_matrix：形状为 (vocab_size, embedding_dim)，是检索器的词嵌入矩阵，每一行对应一个 token 的嵌入向量。
            averaged_grad   # averaged_grad：形状为 (embedding_dim,)，是损失函数关于当前 token 位置嵌入向量的梯度。
        )   # 这个点积值可以理解为：如果将当前 token 替换为该候选 token，损失函数的变化量（在线性近似下）
        # 如果提供了 filter，则减去 filter（通常用于屏蔽特殊 token）
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        # 如果 increase_loss 为 False，则取负，即选择使损失减小的 token
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        # 取 top-k 个得分最高的 token id
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)
    return top_k_ids


class Attacker():
    """
    攻击器类，根据指定的攻击方法生成恶意文本。
    支持两种方法：
        - 'LM_targeted': 黑盒攻击，直接拼接目标问题与预先生成的 I 部分。
        - 'hotflip': 白盒攻击，通过梯度优化使 S 部分与目标问题更相似。
    """
    def __init__(self, args, **kwargs):
        # 保存命令行参数
        self.args = args
        # 攻击方法名称
        self.attack_method = args.attack_method
        # 每个目标问题的恶意文本数量（N）
        self.adv_per_query = args.adv_per_query

        # 从 kwargs 中获取检索器相关组件
        self.model = kwargs.get('model', None)          # 问题编码器
        self.c_model = kwargs.get('c_model', None)      # 文本编码器
        self.tokenizer = kwargs.get('tokenizer', None)  # 分词器
        self.get_emb = kwargs.get('get_emb', None)      # 获取 embedding 的函数

        # 如果攻击方法是 hotflip，则初始化相关超参数
        if args.attack_method == 'hotflip':
            self.max_seq_length = kwargs.get('max_seq_length', 128)
            self.pad_to_max_length = kwargs.get('pad_to_max_length', True)
            self.per_gpu_eval_batch_size = kwargs.get('per_gpu_eval_batch_size', 64)
            self.num_adv_passage_tokens = kwargs.get('num_adv_passage_tokens', 30)  # S 部分的初始长度
            self.num_cand = kwargs.get('num_cand', 100)      # 每次优化的候选 token 数
            self.num_iter = kwargs.get('num_iter', 30)       # 优化迭代次数
            self.gold_init = kwargs.get('gold_init', True)   # 是否用目标问题初始化 S
            self.early_stop = kwargs.get('early_stop', False) # 是否提前停止

        # 加载预先生成的 I 部分和错误答案（与 main.py 相同文件）
        self.all_adv_texts = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')

    def get_attack(self, target_queries) -> list:
        """
        为每个目标问题生成一组恶意文本。
        参数 target_queries: 列表，每个元素是包含 'query' 等信息的字典。
        返回值: adv_text_groups，是一个列表，每个元素是一个子列表，包含 N 个恶意文本。
        """
        adv_text_groups = []
        # 黑盒攻击：LM_targeted（直接拼接）
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                # 获取目标问题文本
                question = target_queries[i]['query']
                # 获取该问题的 ID
                id = target_queries[i]['id']
                # 从预先生成的数据中取出该问题的 I 部分（adv_texts），并截取前 adv_per_query 个
                adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]
                # 构造 S 部分：目标问题 + 句点
                adv_text_a = question + "."
                # 拼接 S 和每个 I，得到完整的恶意文本 P = S ⊕ I
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)

        # 白盒攻击：hotflip（梯度优化）
        elif self.attack_method == 'hotflip':
            adv_text_groups = self.hotflip(target_queries)
        else:
            raise NotImplementedError
        return adv_text_groups

    def hotflip(self, target_queries, adv_b=None, **kwargs) -> list:
        """
        使用 HotFlip 方法为每个目标问题优化生成恶意文本。
        target_queries: 列表，每个元素包含 'query', 'top1_score', 'id'。
        返回: adv_text_groups，与 get_attack 格式相同。
        """
        device = 'cuda'
        print('Doing HotFlip attack!')
        adv_text_groups = []

        # 遍历每个目标问题
        for query_score in tqdm(target_queries):
            query = query_score['query']               # 目标问题文本
            top1_score = query_score['top1_score']     # top-1 检索结果的相似度分数
            id = query_score['id']                     # 问题 ID
            # 获取预先生成的 I 部分列表
            adv_texts_b = self.all_adv_texts[id]['adv_texts']

            # 当前问题的恶意文本列表（将包含 N 个优化后的 P）
            adv_texts = []
            # 对每个 I 部分独立优化（每个 I 生成一个 P）
            for j in range(self.adv_per_query):
                # 将 I 部分 tokenize，得到 token id 列表
                adv_b = adv_texts_b[j]
                adv_b = self.tokenizer(adv_b, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']

                # 初始化 S 部分（adv_a）
                if self.gold_init:
                    # 使用目标问题作为 S 的初始值（tokenize）
                    adv_a = query
                    adv_a = self.tokenizer(adv_a, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']
                else:
                    # 使用 [MASK] 符号填充固定长度作为初始 S
                    adv_a = [self.tokenizer.mask_token_id] * self.num_adv_passage_tokens

                # 获取文本编码器的词嵌入层，并创建梯度存储对象
                embeddings = get_embeddings(self.c_model)
                embedding_gradient = GradientStorage(embeddings)

                # 将 S 和 I 的 token 列表拼接，得到初始恶意文本的 token ids
                adv_passage = adv_a + adv_b
                # 转换为 PyTorch tensor，并添加 batch 维度
                adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
                # 创建 attention mask 和 token_type_ids（全 1 和全 0）
                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

                # 对目标问题编码，获取其 embedding（固定不变）
                q_sent = self.tokenizer(query, max_length=self.max_seq_length, truncation=True,
                                        padding="max_length" if self.pad_to_max_length else False, return_tensors="pt")
                q_sent = {key: value.cuda() for key, value in q_sent.items()}
                q_emb = self.get_emb(self.model, q_sent).detach()

                # ----- 开始迭代优化 S 部分 -----
                for it_ in range(self.num_iter):
                    grad = None
                    # 清空文本编码器的梯度
                    self.c_model.zero_grad()

                    # 构建当前恶意文本的输入字典
                    p_sent = {
                        'input_ids': adv_passage_ids,
                        'attention_mask': adv_passage_attention,
                        'token_type_ids': adv_passage_token_type
                    }
                    # 获取恶意文本的 embedding
                    p_emb = self.get_emb(self.c_model, p_sent)

                    # 计算恶意文本与目标问题的相似度（作为损失）
                    if self.args.score_function == 'dot':
                        sim = torch.mm(p_emb, q_emb.T)           # 点积，形状 (1,1)
                    elif self.args.score_function == 'cos_sim':
                        sim = torch.cosine_similarity(p_emb, q_emb)  # 余弦相似度
                    else:
                        raise KeyError

                    loss = sim.mean()   # 标量损失，我们要最大化它

                    # 如果启用 early stop，且当前相似度已经超过 top1_score + 0.1，则停止优化
                    if self.early_stop and sim.item() > top1_score + 0.1:
                        break

                    # 反向传播，计算梯度
                    loss.backward()

                    # 获取嵌入层存储的梯度（形状 (batch, seq_len, emb_dim)）
                    temp_grad = embedding_gradient.get()
                    if grad is None:
                        # 对所有 token 的梯度求和，得到 (seq_len, emb_dim)
                        grad = temp_grad.sum(dim=0)
                    else:
                        grad += temp_grad.sum(dim=0)

                    # 随机选择 S 部分中的一个位置进行替换（只修改 S，不修改 I）
                    # adv_a 的长度可能小于整个序列长度，这里 token_to_flip 是相对于整个序列的索引
                    token_to_flip = random.randrange(len(adv_a))

                    # 调用 hotflip_attack，获取候选 token id
                    candidates = hotflip_attack(grad[token_to_flip],
                                                embeddings.weight,
                                                increase_loss=True,
                                                num_candidates=self.num_cand,
                                                filter=None)

                    # 计算当前损失值（作为 baseline）
                    current_score = 0
                    candidate_scores = torch.zeros(self.num_cand, device=device)

                    temp_score = loss.sum().cpu().item()
                    current_score += temp_score

                    # 对每个候选 token，评估替换后的损失
                    for i, candidate in enumerate(candidates):
                        # 克隆当前 token 序列
                        temp_adv_passage = adv_passage_ids.clone()
                        # 替换指定位置的 token
                        temp_adv_passage[:, token_to_flip] = candidate
                        temp_p_sent = {
                            'input_ids': temp_adv_passage,
                            'attention_mask': adv_passage_attention,
                            'token_type_ids': adv_passage_token_type
                        }
                        temp_p_emb = self.get_emb(self.c_model, temp_p_sent)
                        with torch.no_grad():
                            if self.args.score_function == 'dot':
                                temp_sim = torch.mm(temp_p_emb, q_emb.T)
                            elif self.args.score_function == 'cos_sim':
                                temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                            else:
                                raise KeyError
                            can_loss = temp_sim.mean()
                            temp_score = can_loss.sum().cpu().item()
                            candidate_scores[i] += temp_score

                    # 如果存在候选 token 使损失大于当前值，则选择最优的进行替换
                    if (candidate_scores > current_score).any():
                        best_candidate_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    else:
                        # 如果没有更好的候选，则跳过本次迭代
                        continue

                # 优化结束后，将最终的 token ids 解码为文本
                adv_text = self.tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                adv_texts.append(adv_text)

            # 当前目标问题的 N 个恶意文本添加进总结果
            adv_text_groups.append(adv_texts)

        return adv_text_groups