"""实验配置文件加载工具。"""

from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Any, Dict, Iterable


def load_experiment_config(path: str, valid_keys: Iterable[str] | None = None) -> Dict[str, Any]:
    """读取 Python 配置文件中的 ``CONFIG`` 字典，并可按 argparse 参数名过滤。"""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    config = runpy.run_path(str(config_path)).get("CONFIG")
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须定义字典 CONFIG：{config_path}")
    if valid_keys is None:
        return dict(config)
    allowed = set(valid_keys)
    return {key: value for key, value in config.items() if key in allowed}


def validate_retrieval_score_function(results_path: str, expected_score_function: str) -> None:
    """校验预检索结果与当前实验使用同一种相似度定义。

    余弦检索结果必须带有由 ``evaluate_beir.py`` 写出的元信息文件，避免把
    旧 dot-product 结果与在线计算的余弦攻击分数混合排序。历史 dot 结果在
    没有元信息时仍可读取，以保持旧实验的兼容性。
    """

    expected = str(expected_score_function).strip()
    metadata_path = Path(f"{results_path}.meta.json")
    if not metadata_path.is_file():
        if expected == "cos_sim":
            raise FileNotFoundError(
                "余弦实验缺少预检索元信息："
                f"{metadata_path}。请先用 evaluate_beir.py --score_function cos_sim "
                "重建 retrieval_results_path。"
            )
        return

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    actual = str(metadata.get("score_function", "")).strip()
    if actual != expected:
        raise ValueError(
            "预检索结果的相似度与当前实验不一致："
            f"results={actual!r}, experiment={expected!r}, path={results_path}。"
        )
