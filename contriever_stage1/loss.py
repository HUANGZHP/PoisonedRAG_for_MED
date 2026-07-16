"""同时包含两类攻击负例的 Multiple-Negatives InfoNCE 损失。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass
class LossOutput:
    """损失及用于日志、验证的相似度统计。"""

    loss: torch.Tensor
    positive_score: torch.Tensor
    blackbox_score: torch.Tensor
    hotflip_score: torch.Tensor
    positive_blackbox_gap: torch.Tensor
    positive_hotflip_gap: torch.Tensor


def multiple_negatives_info_nce(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    blackbox_embeddings: torch.Tensor,
    hotflip_embeddings: torch.Tensor,
    temperature: float,
) -> LossOutput:
    """计算 P 为正例、NB/NH 及 batch 内全部文本为负例的 InfoNCE。"""
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    batch_size = query_embeddings.shape[0]
    candidates = torch.cat((positive_embeddings, blackbox_embeddings, hotflip_embeddings), dim=0)
    logits = query_embeddings @ candidates.transpose(0, 1) / temperature
    labels = torch.arange(batch_size, device=query_embeddings.device)
    loss = functional.cross_entropy(logits, labels)
    positive_score = (query_embeddings * positive_embeddings).sum(dim=-1)
    blackbox_score = (query_embeddings * blackbox_embeddings).sum(dim=-1)
    hotflip_score = (query_embeddings * hotflip_embeddings).sum(dim=-1)
    return LossOutput(
        loss=loss,
        positive_score=positive_score,
        blackbox_score=blackbox_score,
        hotflip_score=hotflip_score,
        positive_blackbox_gap=positive_score - blackbox_score,
        positive_hotflip_gap=positive_score - hotflip_score,
    )
