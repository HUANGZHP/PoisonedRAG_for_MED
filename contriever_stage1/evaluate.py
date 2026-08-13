"""Contriever 微调前后和逐 epoch 的检索偏好评估。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Dict, Iterable, List, Sequence

import torch

from .dataset import PreferenceCollator, PreferenceExample
from .loss import LossOutput, multiple_negatives_info_nce
from .utils import move_batch_to_device


@dataclass
class EvaluationMetrics:
    """正例、两种负例及 InfoNCE 的聚合指标。"""

    loss: float
    positive_score: float
    blackbox_score: float
    hotflip_score: float
    positive_blackbox_gap: float
    positive_hotflip_gap: float
    positive_blackbox_accuracy: float
    positive_hotflip_accuracy: float
    average_gap: float
    count: int

    def to_dict(self) -> Dict[str, float | int]:
        """转换为可保存的字典。"""
        return asdict(self)


def _forward_loss(model: torch.nn.Module, batch: Dict[str, Dict[str, torch.Tensor]], temperature: float) -> LossOutput:
    """用官方 Mean Pooling 与 L2 Normalize 获取四组嵌入并计算损失。"""
    query_embeddings = model(**batch["query"], normalize=os.environ.get("CONTRIEVER_TRAIN_SCORE_FUNCTION", "cos_sim") == "cos_sim")
    positive_embeddings = model(**batch["positive"], normalize=os.environ.get("CONTRIEVER_TRAIN_SCORE_FUNCTION", "cos_sim") == "cos_sim")
    blackbox_embeddings = model(**batch["blackbox_negative"], normalize=os.environ.get("CONTRIEVER_TRAIN_SCORE_FUNCTION", "cos_sim") == "cos_sim")
    hotflip_embeddings = model(**batch["hotflip_negative"], normalize=os.environ.get("CONTRIEVER_TRAIN_SCORE_FUNCTION", "cos_sim") == "cos_sim")
    return multiple_negatives_info_nce(
        query_embeddings, positive_embeddings, blackbox_embeddings, hotflip_embeddings, temperature
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, Dict[str, torch.Tensor]]],
    device: torch.device,
    temperature: float,
    mixed_precision: bool,
) -> EvaluationMetrics:
    """在给定样本上计算验证损失、得分、准确率与平均 gap。"""
    model.eval()
    sums = {key: 0.0 for key in ("loss", "positive", "blackbox", "hotflip", "gap_blackbox", "gap_hotflip", "acc_blackbox", "acc_hotflip")}
    total = 0
    use_amp = mixed_precision and device.type == "cuda"
    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = _forward_loss(model, batch, temperature)
        size = output.positive_score.shape[0]
        total += size
        sums["loss"] += float(output.loss.detach()) * size
        sums["positive"] += float(output.positive_score.sum())
        sums["blackbox"] += float(output.blackbox_score.sum())
        sums["hotflip"] += float(output.hotflip_score.sum())
        sums["gap_blackbox"] += float(output.positive_blackbox_gap.sum())
        sums["gap_hotflip"] += float(output.positive_hotflip_gap.sum())
        sums["acc_blackbox"] += float((output.positive_score > output.blackbox_score).sum())
        sums["acc_hotflip"] += float((output.positive_score > output.hotflip_score).sum())
    if total == 0:
        raise ValueError("验证 DataLoader 为空")
    gap_blackbox = sums["gap_blackbox"] / total
    gap_hotflip = sums["gap_hotflip"] / total
    return EvaluationMetrics(
        loss=sums["loss"] / total,
        positive_score=sums["positive"] / total,
        blackbox_score=sums["blackbox"] / total,
        hotflip_score=sums["hotflip"] / total,
        positive_blackbox_gap=gap_blackbox,
        positive_hotflip_gap=gap_hotflip,
        positive_blackbox_accuracy=sums["acc_blackbox"] / total,
        positive_hotflip_accuracy=sums["acc_hotflip"] / total,
        average_gap=(gap_blackbox + gap_hotflip) / 2,
        count=total,
    )


def forward_loss(model: torch.nn.Module, batch: Dict[str, Dict[str, torch.Tensor]], temperature: float) -> LossOutput:
    """向训练循环公开四路前向和 InfoNCE 计算。"""
    return _forward_loss(model, batch, temperature)


@torch.no_grad()
def sample_query_scores(
    model: torch.nn.Module,
    examples: Sequence[PreferenceExample],
    collator: PreferenceCollator,
    device: torch.device,
    mixed_precision: bool,
) -> List[Dict[str, float | str]]:
    """返回随机验证样本的 query、正例和两种负例相似度，供逐 epoch 日志输出。"""
    model.eval()
    rows: List[Dict[str, float | str]] = []
    use_amp = mixed_precision and device.type == "cuda"
    for example in examples:
        batch = move_batch_to_device(collator([example]), device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = _forward_loss(model, batch, temperature=1.0)
        rows.append(
            {
                "query": example.query,
                "positive_score": float(output.positive_score.item()),
                "blackbox_score": float(output.blackbox_score.item()),
                "hotflip_score": float(output.hotflip_score.item()),
            }
        )
    return rows
