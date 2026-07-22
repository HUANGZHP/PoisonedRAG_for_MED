"""实验配置文件加载工具。"""

from __future__ import annotations

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
