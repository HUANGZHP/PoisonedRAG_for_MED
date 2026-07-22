"""从集中配置文件启动标准或 Agentic 实验。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.config_utils import load_experiment_config


def main() -> None:
    """读取配置入口并转交至对应的实验脚本。"""

    parser = argparse.ArgumentParser(description="从实验配置启动评测。")
    parser.add_argument("--config", default="experiment_config.py", help="集中实验配置文件路径。")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    entrypoint = str(config.get("entrypoint", "agentic")).strip().lower()
    scripts = {"agentic": "agentic_main.py", "standard": "main.py"}
    if entrypoint not in scripts:
        raise ValueError("CONFIG['entrypoint'] 只能是 'agentic' 或 'standard'。")
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, scripts[entrypoint], "--config", str(Path(args.config).resolve())], cwd=root, check=True)


if __name__ == "__main__":
    main()
