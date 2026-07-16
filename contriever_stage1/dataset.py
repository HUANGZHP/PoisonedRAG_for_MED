"""PubMedQA 对抗检索微调数据的读取、匹配与批处理。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class PreferenceExample:
    """一个 query、正例与两类攻击负例组成的训练样本。"""

    query: str
    positive: str
    blackbox_negative: str
    hotflip_negative: str


def _read_jsonl(path: Path, expected_attack_type: str) -> List[Dict[str, str]]:
    """读取并逐行校验单个攻击类型的 JSONL 文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    rows: List[Dict[str, str]] = []
    expected_fields = {"query", "positive", "negative", "attack_type"}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if set(row) != expected_fields:
                raise ValueError(f"{path} 第 {line_no} 行字段不符合要求")
            if row["attack_type"] != expected_attack_type:
                raise ValueError(f"{path} 第 {line_no} 行 attack_type 错误")
            normalized = {key: " ".join(str(row[key]).split()) for key in ("query", "positive", "negative")}
            if any(not value for value in normalized.values()):
                raise ValueError(f"{path} 第 {line_no} 行存在空文本")
            if normalized["positive"] == normalized["negative"]:
                raise ValueError(f"{path} 第 {line_no} 行 positive 与 negative 相同")
            rows.append({**normalized, "attack_type": expected_attack_type})
    return rows


def load_merged_examples(blackbox_path: str, hotflip_path: str) -> List[PreferenceExample]:
    """按 query 严格合并两类攻击文件，并在异常时阻止后续训练。"""
    blackbox_rows = _read_jsonl(Path(blackbox_path), "blackbox")
    hotflip_rows = _read_jsonl(Path(hotflip_path), "hotflip")
    blackbox_by_query = {row["query"]: row for row in blackbox_rows}
    hotflip_by_query = {row["query"]: row for row in hotflip_rows}
    if len(blackbox_by_query) != len(blackbox_rows):
        raise ValueError("blackbox 文件存在重复 Query")
    if len(hotflip_by_query) != len(hotflip_rows):
        raise ValueError("hotflip 文件存在重复 Query")
    blackbox_queries, hotflip_queries = set(blackbox_by_query), set(hotflip_by_query)
    if blackbox_queries != hotflip_queries:
        missing_hotflip = sorted(blackbox_queries - hotflip_queries)
        missing_blackbox = sorted(hotflip_queries - blackbox_queries)
        raise ValueError(
            f"两文件 Query 不可完全匹配：缺少 hotflip={len(missing_hotflip)}，"
            f"缺少 blackbox={len(missing_blackbox)}"
        )
    examples: List[PreferenceExample] = []
    for row in blackbox_rows:
        query = row["query"]
        paired = hotflip_by_query[query]
        if row["positive"] != paired["positive"]:
            raise ValueError(f"Query 的 positive 不一致：{query[:120]}")
        examples.append(
            PreferenceExample(
                query=query,
                positive=row["positive"],
                blackbox_negative=row["negative"],
                hotflip_negative=paired["negative"],
            )
        )
    if not examples:
        raise ValueError("合并后训练数据为空")
    return examples


class PubMedPreferenceDataset(Dataset[PreferenceExample]):
    """每个元素同时包含正例、blackbox 负例和 HotFlip 负例。"""

    def __init__(self, examples: Sequence[PreferenceExample]) -> None:
        """保存已通过匹配校验的样本。"""
        self.examples = list(examples)

    def __len__(self) -> int:
        """返回训练样本数。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferenceExample:
        """返回指定样本。"""
        return self.examples[index]


class PreferenceCollator:
    """使用同一 Contriever tokenizer 分别编码四种文本字段。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        """初始化 tokenizer 与最大序列长度。"""
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """对一组文本执行动态 padding、截断和张量化。"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value for key, value in encoded.items()}

    def __call__(self, batch: Sequence[PreferenceExample]) -> Dict[str, Dict[str, torch.Tensor]]:
        """构造一个同时含有 Q、P、NB、NH 的完整 batch。"""
        return {
            "query": self._encode([item.query for item in batch]),
            "positive": self._encode([item.positive for item in batch]),
            "blackbox_negative": self._encode([item.blackbox_negative for item in batch]),
            "hotflip_negative": self._encode([item.hotflip_negative for item in batch]),
        }
