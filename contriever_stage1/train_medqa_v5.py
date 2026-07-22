"""仅使用 MedQA-USMLE 攻击三元组训练原始 Contriever v5。"""
from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from contriever_stage1.dataset import PreferenceCollator, PreferenceExample, PubMedPreferenceDataset, load_merged_examples
from contriever_stage1.evaluate import EvaluationMetrics, evaluate_model, forward_loss, sample_query_scores
from contriever_stage1.train import _make_loader, _train_epoch
from contriever_stage1.utils import move_batch_to_device, save_json, set_seed
from src.contriever_src.contriever import Contriever


LOGGER = logging.getLogger("contriever_v5")


def parse_args() -> argparse.Namespace:
    """解析仅 MedQA-USMLE 的数据、训练、验证与保存参数。"""
    parser = argparse.ArgumentParser(description="Contriever v5：MedQA-USMLE only")
    parser.add_argument("--blackbox-path", default="checkpoint/contriever_v2/input/medqa_blackbox.jsonl")
    parser.add_argument("--hotflip-path", default="checkpoint/contriever_v2/input/medqa_hotflip.jsonl")
    parser.add_argument("--init-model-path", default="/home/HF_Model/facebook/contriever")
    parser.add_argument("--output-dir", default="checkpoint/contriever_v5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def split_by_query(
    examples: Sequence[PreferenceExample], validation_fraction: float, test_fraction: float, seed: int
) -> Tuple[List[PreferenceExample], List[PreferenceExample], List[PreferenceExample]]:
    """按唯一 query 固定随机划分训练、验证与最终测试数据。"""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("验证与测试比例必须在 0 和 1 之间")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("验证与测试比例之和必须小于 1")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    validation_count = round(count * validation_fraction)
    test_count = round(count * test_fraction)
    test_examples = shuffled[:test_count]
    validation_examples = shuffled[test_count:test_count + validation_count]
    train_examples = shuffled[test_count + validation_count:]
    if min(len(train_examples), len(validation_examples), len(test_examples)) == 0:
        raise ValueError("query-disjoint 划分后存在空集合")
    all_queries = [item.query for group in (train_examples, validation_examples, test_examples) for item in group]
    if len(all_queries) != len(set(all_queries)):
        raise ValueError("query-disjoint 划分失败：同一 query 出现在多个集合")
    return train_examples, validation_examples, test_examples


def selection_score(metrics: EvaluationMetrics) -> float:
    """将低验证损失、高 gap 与两类攻击准确率组合为检查点选择分数。"""
    accuracy = (metrics.positive_blackbox_accuracy + metrics.positive_hotflip_accuracy) / 2
    return -metrics.loss + metrics.average_gap + accuracy


def log_metrics(stage: str, epoch: int, metrics: EvaluationMetrics) -> None:
    """记录损失、相似度、gap 与两类攻击准确率。"""
    LOGGER.info(
        "%s epoch=%d loss=%.6f pos=%.6f blackbox=%.6f hotflip=%.6f "
        "gap_b=%.6f gap_h=%.6f acc_b=%.4f acc_h=%.4f",
        stage, epoch, metrics.loss, metrics.positive_score, metrics.blackbox_score,
        metrics.hotflip_score, metrics.positive_blackbox_gap, metrics.positive_hotflip_gap,
        metrics.positive_blackbox_accuracy, metrics.positive_hotflip_accuracy,
    )


def save_epoch(
    model: Contriever, tokenizer: AutoTokenizer, optimizer: AdamW, output_dir: Path,
    epoch: int, validation: EvaluationMetrics, score: float,
) -> None:
    """保存当前 epoch 的官方模型、tokenizer、优化器与验证指标。"""
    epoch_dir = output_dir / f"epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    torch.save({"epoch": epoch, "optimizer": optimizer.state_dict()}, epoch_dir / "training_state.pt")
    save_json(epoch_dir / "metrics.json", {"validation": validation.to_dict(), "selection_score": score})


def save_best(model: Contriever, tokenizer: AutoTokenizer, output_dir: Path, epoch: int, metrics: EvaluationMetrics, score: float) -> None:
    """在不改动其他版本 checkpoint 的前提下更新 v5 最佳模型。"""
    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    save_json(best_dir / "selection.json", {"epoch": epoch, "validation": metrics.to_dict(), "selection_score": score})


def main() -> None:
    """完成 MedQA 数据检查、query-disjoint 训练、验证、保存与最终测试。"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(args.seed)
    if "medqa" not in args.blackbox_path.lower() or "medqa" not in args.hotflip_path.lower():
        raise ValueError("v5 仅接受 MedQA-USMLE 的 blackbox 与 hotflip 文件")
    model_path = Path(args.init_model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到原始 Contriever 初始化模型：{model_path}")

    examples = load_merged_examples(args.blackbox_path, args.hotflip_path)
    train_examples, validation_examples, test_examples = split_by_query(
        examples, args.validation_fraction, args.test_fraction, args.seed
    )
    LOGGER.info(
        "MedQA-USMLE 数据检查通过：total=%d train=%d validation=%d test=%d，未读取 PubMedQA",
        len(examples), len(train_examples), len(validation_examples), len(test_examples),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    collator = PreferenceCollator(tokenizer, args.max_length)
    train_loader = _make_loader(
        PubMedPreferenceDataset(train_examples), collator, args.batch_size, True, args.num_workers
    )
    validation_loader = _make_loader(
        PubMedPreferenceDataset(validation_examples), collator, args.batch_size, False, args.num_workers
    )
    test_loader = _make_loader(
        PubMedPreferenceDataset(test_examples), collator, args.batch_size, False, args.num_workers
    )
    first_batch = next(iter(validation_loader))
    if set(first_batch) != {"query", "positive", "blackbox_negative", "hotflip_negative"}:
        raise ValueError("DataLoader 未返回完整 Q/P/NB/NH batch")

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = Contriever.from_pretrained(str(model_path), local_files_only=True).to(device)
    if getattr(model.config, "pooling", "average") not in {"average", "avg"}:
        raise ValueError("Contriever 未启用官方 Mean Pooling")
    with torch.no_grad():
        probe = forward_loss(model, move_batch_to_device(first_batch, device), args.temperature)
    if not torch.isfinite(probe.loss):
        raise ValueError("模型前向或 InfoNCE 损失异常")
    LOGGER.info("query-disjoint DataLoader、官方 pooling/L2 Normalize 和 InfoNCE 冒烟验证通过")

    before = evaluate_model(model, test_loader, device, args.temperature, args.mixed_precision)
    log_metrics("before_test", 0, before)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    best_score = float("-inf")
    history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train = _train_epoch(model, train_loader, optimizer, scaler, device, args.temperature, args.gradient_clip, args.mixed_precision)
        validation = evaluate_model(model, validation_loader, device, args.temperature, args.mixed_precision)
        score = selection_score(validation)
        log_metrics("train", epoch, train)
        log_metrics("validation", epoch, validation)
        for preview in sample_query_scores(model, validation_examples[:args.preview_count], collator, device, args.mixed_precision):
            LOGGER.info("validation_sample epoch=%d query=%s pos=%.6f blackbox=%.6f hotflip=%.6f", epoch, preview["query"], preview["positive_score"], preview["blackbox_score"], preview["hotflip_score"])
        save_epoch(model, tokenizer, optimizer, output_dir, epoch, validation, score)
        history.append({"epoch": epoch, "train": train.to_dict(), "validation": validation.to_dict(), "selection_score": score})
        if score > best_score:
            best_score = score
            save_best(model, tokenizer, output_dir, epoch, validation, score)
            LOGGER.info("epoch=%d 更新 best_model，选择分数=%.6f", epoch, score)

    after = evaluate_model(model, test_loader, device, args.temperature, args.mixed_precision)
    log_metrics("after_test", args.epochs, after)
    save_json(output_dir / "final_evaluation.json", {
        "initial_model": str(model_path), "dataset": "MedQA-USMLE train only", "query_disjoint": True,
        "counts": {"total": len(examples), "train": len(train_examples), "validation": len(validation_examples), "test": len(test_examples)},
        "before": before.to_dict(), "after": after.to_dict(), "epochs": history,
    })
    save_json(output_dir / "run_config.json", vars(args))
    LOGGER.info("v5 训练完成，结果已保存至 %s", output_dir)


if __name__ == "__main__":
    main()
