"""训练、验证与结果保存的共用工具。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    """将四组 tokenizer 输出移动到指定设备。"""
    return {name: {key: value.to(device, non_blocking=True) for key, value in encoded.items()} for name, encoded in batch.items()}


def save_json(path: Path, content: Dict[str, Any]) -> None:
    """以 UTF-8 JSON 格式保存实验元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
