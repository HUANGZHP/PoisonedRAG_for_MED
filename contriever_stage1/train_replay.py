"""以固定比例的 PubMedQA replay 和 MedQA 训练 Contriever v3。"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from transformers import AutoTokenizer

from contriever_stage1.dataset import PreferenceCollator, PreferenceExample, PubMedPreferenceDataset, load_merged_examples
from contriever_stage1.evaluate import EvaluationMetrics, evaluate_model, forward_loss
from contriever_stage1.utils import move_batch_to_device, save_json, set_seed
from src.contriever_src.contriever import Contriever


LOGGER = logging.getLogger("contriever_v3")


class BalancedReplayBatchSampler(Sampler[List[int]]):
    """每个 batch 固定抽取 MedQA 与 PubMedQA，以 PubMedQA 作为 replay。"""

    def __init__(
        self,
        pubmed_count: int,
        medqa_count: int,
        pubmed_per_batch: int,
        medqa_per_batch: int,
        steps: int,
        seed: int,
    ) -> None:
        """保存两个数据域的索引范围、批次比例和本 epoch 的训练步数。"""
        if min(pubmed_count, medqa_count, pubmed_per_batch, medqa_per_batch, steps) <= 0:
            raise ValueError("replay sampler 的样本数、批次大小和步数必须为正数")
        self.pubmed_count = pubmed_count
        self.medqa_count = medqa_count
        self.pubmed_per_batch = pubmed_per_batch
        self.medqa_per_batch = medqa_per_batch
        self.steps = steps
        self.seed = seed

    def __len__(self) -> int:
        """返回该 epoch 的固定训练步数。"""
        return self.steps

    def __iter__(self) -> Iterator[List[int]]:
        """循环洗牌两个域的索引，确保 MedQA 主数据被连续覆盖。"""
        generator = random.Random(self.seed)
        pubmed_order = list(range(self.pubmed_count))
        medqa_order = list(range(self.medqa_count))
        generator.shuffle(pubmed_order)
        generator.shuffle(medqa_order)
        pubmed_pos, medqa_pos = 0, 0

        def take(order: List[int], position: int, count: int) -> tuple[List[int], int]:
            chosen: List[int] = []
            while len(chosen) < count:
                if position == len(order):
                    generator.shuffle(order)
                    position = 0
                remaining = min(count - len(chosen), len(order) - position)
                chosen.extend(order[position:position + remaining])
                position += remaining
            return chosen, position

        for _ in range(self.steps):
            pubmed_indices, pubmed_pos = take(pubmed_order, pubmed_pos, self.pubmed_per_batch)
            medqa_indices, medqa_pos = take(medqa_order, medqa_pos, self.medqa_per_batch)
            yield pubmed_indices + [self.pubmed_count + index for index in medqa_indices]


def parse_args() -> argparse.Namespace:
    """解析 replay 训练、验证、模型和保存参数。"""
    parser = argparse.ArgumentParser(description="Contriever v3：MedQA 主训练与 PubMedQA replay")
    parser.add_argument("--pubmed-blackbox-path", default="processed/pubmedqa_blackbox.jsonl")
    parser.add_argument("--pubmed-hotflip-path", default="processed/pubmedqa_hotflip.jsonl")
    parser.add_argument("--medqa-blackbox-path", default="checkpoint/contriever_v2/input/medqa_blackbox.jsonl")
    parser.add_argument("--medqa-hotflip-path", default="checkpoint/contriever_v2/input/medqa_hotflip.jsonl")
    parser.add_argument("--init-model-path", default="/home/HF_Model/facebook/contriever")
    parser.add_argument("--output-dir", default="checkpoint/contriever_v3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pubmed-per-batch", type=int, default=8)
    parser.add_argument("--medqa-per-batch", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--final-evaluation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def device_for(gpu: int) -> torch.device:
    """返回可用时的指定 CUDA 设备，否则使用 CPU。"""
    return torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")


def sample_examples(examples: Sequence[PreferenceExample], count: int, seed: int) -> List[PreferenceExample]:
    """以固定种子无放回抽取指定数量的验证样本。"""
    return random.Random(seed).sample(list(examples), min(count, len(examples)))


def make_eval_loader(
    examples: Sequence[PreferenceExample], collator: PreferenceCollator, batch_size: int, workers: int
) -> DataLoader[Dict[str, Dict[str, torch.Tensor]]]:
    """构建保持 Q/P/NB/NH 四路字段的评估 DataLoader。"""
    return DataLoader(
        PubMedPreferenceDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def aggregate_metrics(sums: Dict[str, float], total: int) -> EvaluationMetrics:
    """将训练循环累计量转换为统一的评估指标。"""
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


def train_epoch(
    model: Contriever,
    loader: DataLoader[Dict[str, Dict[str, torch.Tensor]]],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    temperature: float,
    gradient_clip: float,
    mixed_precision: bool,
) -> EvaluationMetrics:
    """完成一个固定比例 replay epoch，并返回四路相似度统计。"""
    model.train()
    sums = {key: 0.0 for key in ("loss", "positive", "blackbox", "hotflip", "gap_blackbox", "gap_hotflip", "acc_blackbox", "acc_hotflip")}
    total = 0
    use_amp = mixed_precision and device.type == "cuda"
    for raw_batch in loader:
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
    return aggregate_metrics(sums, total)


def selection_score(pubmed: EvaluationMetrics, medqa: EvaluationMetrics) -> float:
    """以两个域同等权重组合低损失、高 gap 与准确率，选择最佳 checkpoint。"""
    def score(metrics: EvaluationMetrics) -> float:
        accuracy = (metrics.positive_blackbox_accuracy + metrics.positive_hotflip_accuracy) / 2
        return -metrics.loss + metrics.average_gap + accuracy

    return (score(pubmed) + score(medqa)) / 2


def save_checkpoint(
    model: Contriever,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    output_dir: Path,
    epoch: int,
    metrics: Dict[str, EvaluationMetrics],
    score: float,
    best: bool,
) -> None:
    """保存每个 epoch 或当前最佳模型，并写入双域验证指标。"""
    directory = output_dir / ("best_model" if best else f"epoch{epoch}")
    if best and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    if not best:
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict()}, directory / "training_state.pt")
    save_json(directory / ("selection.json" if best else "metrics.json"), {
        "epoch": epoch,
        "selection_score": score,
        "validation": {name: metric.to_dict() for name, metric in metrics.items()},
    })


def log_metrics(prefix: str, epoch: int, metrics: EvaluationMetrics) -> None:
    """输出一个数据域的损失、相似度、gap 与准确率。"""
    LOGGER.info(
        "%s epoch=%d loss=%.6f pos=%.6f blackbox=%.6f hotflip=%.6f gap_b=%.6f gap_h=%.6f acc_b=%.4f acc_h=%.4f",
        prefix, epoch, metrics.loss, metrics.positive_score, metrics.blackbox_score, metrics.hotflip_score,
        metrics.positive_blackbox_gap, metrics.positive_hotflip_gap,
        metrics.positive_blackbox_accuracy, metrics.positive_hotflip_accuracy,
    )


def main() -> None:
    """执行原始 Contriever 初始化、固定比例 replay、双域验证与 v3 checkpoint 保存。"""
    args = parse_args()
    if args.batch_size != args.pubmed_per_batch + args.medqa_per_batch:
        raise ValueError("batch-size 必须等于 pubmed-per-batch 与 medqa-per-batch 之和")
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pubmed_examples = load_merged_examples(args.pubmed_blackbox_path, args.pubmed_hotflip_path)
    medqa_examples = load_merged_examples(args.medqa_blackbox_path, args.medqa_hotflip_path)
    LOGGER.info("数据检查通过：PubMedQA=%d，MedQA=%d", len(pubmed_examples), len(medqa_examples))

    init_path = Path(args.init_model_path)
    if not init_path.is_dir():
        raise FileNotFoundError(f"找不到原始 Contriever 初始化模型：{init_path}")
    device = device_for(args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=True)
    collator = PreferenceCollator(tokenizer, args.max_length)
    model = Contriever.from_pretrained(init_path, local_files_only=True).to(device)
    if getattr(model.config, "pooling", "average") not in {"average", "avg"}:
        raise ValueError("Contriever 未启用官方 Mean Pooling")

    combined = ConcatDataset([PubMedPreferenceDataset(pubmed_examples), PubMedPreferenceDataset(medqa_examples)])
    steps_per_epoch = args.steps_per_epoch or ((len(medqa_examples) + args.medqa_per_batch - 1) // args.medqa_per_batch)
    probe_sampler = BalancedReplayBatchSampler(len(pubmed_examples), len(medqa_examples), args.pubmed_per_batch, args.medqa_per_batch, 1, args.seed)
    probe_batch = next(iter(DataLoader(combined, batch_sampler=probe_sampler, collate_fn=collator)))
    with torch.no_grad():
        probe = move_batch_to_device(probe_batch, device)
        if not torch.isfinite(forward_loss(model, probe, args.temperature).loss):
            raise ValueError("模型前向或 InfoNCE 损失异常")
    LOGGER.info("固定 1:1 replay batch、官方 pooling 与 InfoNCE 冒烟验证通过；每 epoch=%d 步", steps_per_epoch)

    final_pubmed = sample_examples(pubmed_examples, args.final_evaluation_size, args.seed)
    final_medqa = sample_examples(medqa_examples, args.final_evaluation_size, args.seed)
    before = {
        "pubmedqa": evaluate_model(model, make_eval_loader(final_pubmed, collator, args.batch_size, args.num_workers), device, args.temperature, args.mixed_precision),
        "medqa": evaluate_model(model, make_eval_loader(final_medqa, collator, args.batch_size, args.num_workers), device, args.temperature, args.mixed_precision),
    }
    log_metrics("before_pubmedqa", 0, before["pubmedqa"])
    log_metrics("before_medqa", 0, before["medqa"])

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        sampler = BalancedReplayBatchSampler(
            len(pubmed_examples), len(medqa_examples), args.pubmed_per_batch,
            args.medqa_per_batch, steps_per_epoch, args.seed + epoch,
        )
        train_loader = DataLoader(combined, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collator)
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, args.temperature, args.gradient_clip, args.mixed_precision)
        validation = {
            "pubmedqa": evaluate_model(model, make_eval_loader(sample_examples(pubmed_examples, args.validation_size, args.seed + epoch), collator, args.batch_size, args.num_workers), device, args.temperature, args.mixed_precision),
            "medqa": evaluate_model(model, make_eval_loader(sample_examples(medqa_examples, args.validation_size, args.seed + epoch), collator, args.batch_size, args.num_workers), device, args.temperature, args.mixed_precision),
        }
        score = selection_score(validation["pubmedqa"], validation["medqa"])
        log_metrics("train_mixed", epoch, train_metrics)
        log_metrics("validation_pubmedqa", epoch, validation["pubmedqa"])
        log_metrics("validation_medqa", epoch, validation["medqa"])
        save_checkpoint(model, tokenizer, optimizer, output_dir, epoch, validation, score, best=False)
        history.append({"epoch": epoch, "train": train_metrics.to_dict(), "validation": {name: metric.to_dict() for name, metric in validation.items()}, "selection_score": score})
        if score > best_score:
            best_score = score
            save_checkpoint(model, tokenizer, optimizer, output_dir, epoch, validation, score, best=True)
            LOGGER.info("epoch=%d 更新 best_model，双域选择分数=%.6f", epoch, score)

    after = {
        "pubmedqa": evaluate_model(model, make_eval_loader(final_pubmed, collator, args.batch_size, args.num_workers), device, args.temperature, args.mixed_precision),
        "medqa": evaluate_model(model, make_eval_loader(final_medqa, collator, args.batch_size, args.num_workers), device, args.temperature, args.mixed_precision),
    }
    log_metrics("after_pubmedqa", args.epochs, after["pubmedqa"])
    log_metrics("after_medqa", args.epochs, after["medqa"])
    save_json(output_dir / "final_evaluation.json", {
        "initial_model": str(init_path),
        "training": "1:1 MedQA/PubMedQA balanced replay",
        "before": {name: metric.to_dict() for name, metric in before.items()},
        "after": {name: metric.to_dict() for name, metric in after.items()},
        "epochs": history,
    })
    save_json(output_dir / "run_config.json", vars(args))
    LOGGER.info("v3 训练完成，结果已保存至 %s", output_dir)


if __name__ == "__main__":
    main()
