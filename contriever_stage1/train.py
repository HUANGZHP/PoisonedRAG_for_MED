"""使用 PubMedQA 的 blackbox 与 HotFlip 负例微调官方 facebook/contriever。"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .dataset import PreferenceCollator, PreferenceExample, PubMedPreferenceDataset, load_merged_examples
from .evaluate import EvaluationMetrics, evaluate_model, forward_loss, sample_query_scores
from src.contriever_src.contriever import Contriever
from .utils import move_batch_to_device, save_json, set_seed


LOGGER = logging.getLogger("contriever_stage1")


def parse_args() -> argparse.Namespace:
    """解析所有数据、模型、训练、验证和保存参数。"""
    parser = argparse.ArgumentParser(description="PubMedQA 对抗鲁棒 Contriever 第一阶段微调")
    parser.add_argument("--blackbox-path", default="processed/pubmedqa_blackbox.jsonl")
    parser.add_argument("--hotflip-path", default="processed/pubmedqa_hotflip.jsonl")
    parser.add_argument("--model-name", default="facebook/contriever")
    parser.add_argument("--local-model-path", default="")
    parser.add_argument("--output-dir", default="checkpoint/contriever_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--validation-preview-count", type=int, default=3)
    parser.add_argument("--final-evaluation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _device(gpu: int) -> torch.device:
    """返回优先使用指定 GPU 的计算设备。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def _resolve_model_source(model_name: str, local_model_path: str) -> str:
    """优先使用本地缓存的官方 facebook/contriever，避免服务器联网下载。"""
    if local_model_path:
        local_path = Path(local_model_path)
        if local_path.is_dir():
            return str(local_path)
    return model_name


def _sample_examples(examples: List[PreferenceExample], count: int, seed: int) -> List[PreferenceExample]:
    """以固定种子从训练数据中无放回抽取验证或最终评估样本。"""
    if count <= 0:
        raise ValueError("抽样数量必须为正数")
    return random.Random(seed).sample(examples, min(count, len(examples)))


def _make_loader(
    dataset: PubMedPreferenceDataset | Subset[PreferenceExample],
    collator: PreferenceCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Dict[str, Dict[str, torch.Tensor]]]:
    """构造保留四路文本字段的 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=False,
    )


def _save_checkpoint(
    model: Contriever,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    output_dir: Path,
    epoch: int,
    metrics: EvaluationMetrics,
    selection_score: float,
) -> None:
    """保存每个 epoch 的官方模型、tokenizer、优化器与验证元数据。"""
    epoch_dir = output_dir / f"epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    torch.save({"epoch": epoch, "optimizer": optimizer.state_dict()}, epoch_dir / "training_state.pt")
    save_json(epoch_dir / "metrics.json", {"validation": metrics.to_dict(), "selection_score": selection_score})


def _save_best_model(
    model: Contriever,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    epoch: int,
    metrics: EvaluationMetrics,
    selection_score: float,
) -> None:
    """依据验证损失、平均 gap 和两类准确率共同保存最佳模型。"""
    best_dir = output_dir / "best_model"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    save_json(best_dir / "selection.json", {"epoch": epoch, "validation": metrics.to_dict(), "selection_score": selection_score})


def _selection_score(metrics: EvaluationMetrics) -> float:
    """将低验证损失、高平均 gap 与两类高准确率组合为最大化分数。"""
    mean_accuracy = (metrics.positive_blackbox_accuracy + metrics.positive_hotflip_accuracy) / 2
    return -metrics.loss + metrics.average_gap + mean_accuracy


def _train_epoch(
    model: Contriever,
    dataloader: DataLoader[Dict[str, Dict[str, torch.Tensor]]],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    temperature: float,
    gradient_clip: float,
    mixed_precision: bool,
) -> EvaluationMetrics:
    """完成一个 epoch 的四路 InfoNCE 训练并汇总训练指标。"""
    model.train()
    sums = {key: 0.0 for key in ("loss", "positive", "blackbox", "hotflip", "gap_blackbox", "gap_hotflip", "acc_blackbox", "acc_hotflip")}
    total = 0
    use_amp = mixed_precision and device.type == "cuda"
    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = forward_loss(model, batch, temperature)
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        size = output.positive_score.shape[0]
        total += size
        sums["loss"] += float(output.loss.detach()) * size
        sums["positive"] += float(output.positive_score.detach().sum())
        sums["blackbox"] += float(output.blackbox_score.detach().sum())
        sums["hotflip"] += float(output.hotflip_score.detach().sum())
        sums["gap_blackbox"] += float(output.positive_blackbox_gap.detach().sum())
        sums["gap_hotflip"] += float(output.positive_hotflip_gap.detach().sum())
        sums["acc_blackbox"] += float((output.positive_score.detach() > output.blackbox_score.detach()).sum())
        sums["acc_hotflip"] += float((output.positive_score.detach() > output.hotflip_score.detach()).sum())
    if total == 0:
        raise ValueError("训练 DataLoader 为空")
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


def _log_metrics(stage: str, epoch: int, metrics: EvaluationMetrics) -> None:
    """按实验要求输出每个阶段的核心相似度、gap 与准确率。"""
    LOGGER.info(
        "%s epoch=%d loss=%.6f pos=%.6f blackbox=%.6f hotflip=%.6f "
        "pos-blackbox=%.6f pos-hotflip=%.6f acc_blackbox=%.4f acc_hotflip=%.4f avg_gap=%.6f",
        stage, epoch, metrics.loss, metrics.positive_score, metrics.blackbox_score, metrics.hotflip_score,
        metrics.positive_blackbox_gap, metrics.positive_hotflip_gap,
        metrics.positive_blackbox_accuracy, metrics.positive_hotflip_accuracy, metrics.average_gap,
    )


def main() -> None:
    """依序检查数据、验证前向与损失、训练、保存 checkpoint 并输出最终评估。"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_merged_examples(args.blackbox_path, args.hotflip_path)
    LOGGER.info("数据检查通过：%d 个 Query 均匹配两类负例", len(examples))
    dataset = PubMedPreferenceDataset(examples)
    model_source = _resolve_model_source(args.model_name, args.local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=Path(model_source).is_dir())
    collator = PreferenceCollator(tokenizer, args.max_length)
    train_loader = _make_loader(dataset, collator, args.batch_size, True, args.num_workers)
    first_batch = next(iter(_make_loader(dataset, collator, args.batch_size, False, args.num_workers)))
    if set(first_batch) != {"query", "positive", "blackbox_negative", "hotflip_negative"}:
        raise ValueError("DataLoader 未返回完整四路 batch")

    device = _device(args.gpu)
    model = Contriever.from_pretrained(model_source, local_files_only=Path(model_source).is_dir()).to(device)
    if getattr(model.config, "pooling", "average") not in {"average", "avg"}:
        raise ValueError("facebook/contriever 未启用官方 Mean Pooling")
    with torch.no_grad():
        probe = move_batch_to_device(first_batch, device)
        probe_output = forward_loss(model, probe, args.temperature)
    if not torch.isfinite(probe_output.loss):
        raise ValueError("模型前向或 InfoNCE 损失异常")
    LOGGER.info("DataLoader、官方 Mean Pooling/L2 Normalize 前向及 InfoNCE 冒烟验证通过")

    final_examples = _sample_examples(examples, args.final_evaluation_size, args.seed)
    final_loader = _make_loader(PubMedPreferenceDataset(final_examples), collator, args.batch_size, False, args.num_workers)
    before_metrics = evaluate_model(model, final_loader, device, args.temperature, args.mixed_precision)
    _log_metrics("before_finetuning", 0, before_metrics)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = float("-inf")
    history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scaler, device, args.temperature, args.gradient_clip, args.mixed_precision
        )
        validation_examples = _sample_examples(examples, args.validation_size, args.seed + epoch)
        validation_loader = _make_loader(
            PubMedPreferenceDataset(validation_examples), collator, args.batch_size, False, args.num_workers
        )
        validation_metrics = evaluate_model(model, validation_loader, device, args.temperature, args.mixed_precision)
        preview_examples = validation_examples[: min(args.validation_preview_count, len(validation_examples))]
        for row in sample_query_scores(model, preview_examples, collator, device, args.mixed_precision):
            LOGGER.info(
                "validation_sample epoch=%d query=%s positive=%.6f blackbox=%.6f hotflip=%.6f",
                epoch, row["query"], row["positive_score"], row["blackbox_score"], row["hotflip_score"],
            )
        selection_score = _selection_score(validation_metrics)
        _log_metrics("train", epoch, train_metrics)
        _log_metrics("validation", epoch, validation_metrics)
        _save_checkpoint(model, tokenizer, optimizer, output_dir, epoch, validation_metrics, selection_score)
        history.append({"epoch": epoch, "train": train_metrics.to_dict(), "validation": validation_metrics.to_dict(), "selection_score": selection_score})
        if selection_score > best_score:
            best_score = selection_score
            _save_best_model(model, tokenizer, output_dir, epoch, validation_metrics, selection_score)
            LOGGER.info("epoch=%d 更新 best_model，联合选择分数=%.6f", epoch, selection_score)

    after_metrics = evaluate_model(model, final_loader, device, args.temperature, args.mixed_precision)
    _log_metrics("after_finetuning", args.epochs, after_metrics)
    save_json(
        output_dir / "final_evaluation.json",
        {
            "model": args.model_name,
            "model_source": model_source,
            "dataset": "PubMedQA only",
            "final_evaluation_samples": len(final_examples),
            "before_finetuning": before_metrics.to_dict(),
            "after_finetuning": after_metrics.to_dict(),
            "epochs": history,
        },
    )
    save_json(output_dir / "run_config.json", vars(args))
    LOGGER.info("训练完成，最终评估已保存至 %s", output_dir / "final_evaluation.json")


if __name__ == "__main__":
    main()
