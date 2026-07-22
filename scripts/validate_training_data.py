"""校验由同一源数据构建的 Black-box 与 HotFlip 训练数据。"""

import argparse
import json
from pathlib import Path
from typing import Any

from utils import load_medqa, load_pubmedqa, normalize_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并拒绝空行以外的非对象记录。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(row)
    return rows


def validate_one(
    rows: list[dict[str, Any]], source_pairs: set[tuple[str, str]], attack_type: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """校验单个攻击文件，返回以 query/positive 为键的映射。"""
    expected_fields = {"query", "positive", "negative", "attack_type"}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    triplets: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != expected_fields:
            raise ValueError(f"第 {index} 条字段不符合要求")
        query = normalize_text(row["query"])
        positive = normalize_text(row["positive"])
        negative = normalize_text(row["negative"])
        if not query or not positive or not negative:
            raise ValueError(f"第 {index} 条存在空 query、positive 或 negative")
        if row["attack_type"] != attack_type:
            raise ValueError(f"第 {index} 条 attack_type 错误")
        if positive == negative:
            raise ValueError(f"第 {index} 条 positive 与 negative 相同")
        pair = (query, positive)
        triplet = (query, positive, negative)
        if pair in by_pair:
            raise ValueError(f"第 {index} 条出现重复 query/positive")
        if triplet in triplets:
            raise ValueError(f"第 {index} 条出现重复训练样本")
        by_pair[pair] = row
        triplets.add(triplet)
    if set(by_pair) != source_pairs:
        missing = len(source_pairs - set(by_pair))
        unexpected = len(set(by_pair) - source_pairs)
        raise ValueError(f"输出与源数据不一致：缺失 {missing} 条，额外 {unexpected} 条")
    return by_pair


def validate(source: Path, blackbox: Path, hotflip: Path, dataset: str) -> None:
    """校验源数据、两份攻击数据及其逐条配对关系。"""
    if dataset == "medqa" and "MedQA-USMLE" not in str(source):
        raise ValueError("MedQA 源路径不是 USMLE 数据集")
    source_rows = load_medqa(str(source)) if dataset == "medqa" else load_pubmedqa(str(source))
    source_pairs = {
        (normalize_text(row["query"]), normalize_text(row["positive"])) for row in source_rows
    }
    if len(source_pairs) != len(source_rows):
        raise ValueError("源数据存在重复 query/positive，不能满足一条 query 对应一个样本")
    blackbox_map = validate_one(load_jsonl(blackbox), source_pairs, "blackbox")
    hotflip_map = validate_one(load_jsonl(hotflip), source_pairs, "hotflip")
    if set(blackbox_map) != set(hotflip_map):
        raise ValueError("Black-box 与 HotFlip 的 query/positive 未完全匹配")
    same_negative = sum(
        normalize_text(blackbox_map[key]["negative"])
        == normalize_text(hotflip_map[key]["negative"])
        for key in source_pairs
    )
    if same_negative:
        raise ValueError(f"有 {same_negative} 条样本的两种 negative 相同")
    print(
        json.dumps(
            {
                "original_samples": len(source_rows),
                "blackbox_samples": len(blackbox_map),
                "hotflip_samples": len(hotflip_map),
                "source": "USMLE train only" if dataset == "medqa" else "PubMedQA",
                "checks": "passed",
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    """解析参数并执行校验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--blackbox", required=True, type=Path)
    parser.add_argument("--hotflip", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["pubmedqa", "medqa"])
    args = parser.parse_args()
    validate(args.source, args.blackbox, args.hotflip, args.dataset)


if __name__ == "__main__":
    main()
