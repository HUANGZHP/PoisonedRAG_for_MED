"""合并 PQA-L 与 PQA-U 的攻击数据，并对 Contriever v4 输入作严格校验。"""

import argparse
import json
from pathlib import Path
from typing import Any

from utils import load_pubmedqa, normalize_text

FIELDS = {"query", "positive", "negative", "attack_type"}


def read_jsonl(path: Path, expected_attack: str) -> list[dict[str, str]]:
    """读取并校验单个攻击 JSONL 文件的行级字段。"""
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row: dict[str, str] = json.loads(line)
        if set(row) != FIELDS or row.get("attack_type") != expected_attack:
            raise ValueError(f"{path}:{line_number} 字段或 attack_type 错误")
        if any(not normalize_text(row.get(field, "")) for field in ("query", "positive", "negative")):
            raise ValueError(f"{path}:{line_number} 存在空文本")
        if normalize_text(row["positive"]) == normalize_text(row["negative"]):
            raise ValueError(f"{path}:{line_number} positive 与 negative 相同")
        rows.append(row)
    return rows


def to_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """按规范化 query/positive 构建映射，并拒绝重复样本。"""
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (normalize_text(row["query"]), normalize_text(row["positive"]))
        if key in result:
            raise ValueError("攻击数据中存在重复 query/positive")
        result[key] = row
    return result


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    """以 JSONL 格式写出合并后的训练数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """按照源文件顺序合并数据，确保两个攻击版本只在 negative 上不同。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--pqal-blackbox", required=True, type=Path)
    parser.add_argument("--pqau-blackbox", required=True, type=Path)
    parser.add_argument("--pqal-hotflip", required=True, type=Path)
    parser.add_argument("--pqau-hotflip", required=True, type=Path)
    parser.add_argument("--blackbox-output", required=True, type=Path)
    parser.add_argument("--hotflip-output", required=True, type=Path)
    args = parser.parse_args()

    source_rows = load_pubmedqa(str(args.source))
    source_keys = [(normalize_text(row["query"]), normalize_text(row["positive"])) for row in source_rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("合并源数据存在重复 query/positive")
    if len({key[0] for key in source_keys}) != len(source_keys):
        raise ValueError("合并源数据存在重复 query")

    blackbox = to_map(read_jsonl(args.pqal_blackbox, "blackbox") + read_jsonl(args.pqau_blackbox, "blackbox"))
    hotflip = to_map(read_jsonl(args.pqal_hotflip, "hotflip") + read_jsonl(args.pqau_hotflip, "hotflip"))
    expected = set(source_keys)
    if set(blackbox) != expected or set(hotflip) != expected:
        raise ValueError("攻击数据与合并源数据无法一一匹配")

    blackbox_rows = [blackbox[key] for key in source_keys]
    hotflip_rows = [hotflip[key] for key in source_keys]
    same_negative = sum(
        normalize_text(blackbox[key]["negative"]) == normalize_text(hotflip[key]["negative"])
        for key in source_keys
    )
    if same_negative:
        raise ValueError(f"有 {same_negative} 条样本的两种 negative 相同")
    write_jsonl(args.blackbox_output, blackbox_rows)
    write_jsonl(args.hotflip_output, hotflip_rows)
    print(json.dumps({"samples": len(source_rows), "checks": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
