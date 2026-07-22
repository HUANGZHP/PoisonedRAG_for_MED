"""构建用于 Contriever v4 的 PQA-L 与 PQA-U 固定抽样源数据。"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def normalized(value: Any) -> str:
    """将文本规整为单空格分隔的字符串。"""
    return " ".join(str(value or "").split())


def canonical_record(
    question: str,
    contexts: list[str],
    long_answer: str,
    source: str,
) -> dict[str, Any]:
    """转换为既有 PubMedQA 加载器支持的统一记录格式。"""
    return {
        "QUESTION": question,
        "CONTEXTS": contexts,
        "LONG_ANSWER": long_answer,
        "SOURCE_SUBSET": source,
    }


def load_pqal(path: Path) -> dict[str, dict[str, Any]]:
    """读取完整 PQA-L，并统一字段格式。"""
    raw: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, Any]] = {}
    for pubid, row in raw.items():
        question = normalized(row.get("QUESTION"))
        contexts = [normalized(text) for text in row.get("CONTEXTS", []) if normalized(text)]
        if not question or not contexts:
            raise ValueError(f"PQA-L 样本 {pubid} 缺少问题或上下文")
        output[f"pqal_{pubid}"] = canonical_record(
            question, contexts, normalized(row.get("LONG_ANSWER")), "PQA-L"
        )
    return output


def sample_pqau(
    path: Path,
    size: int,
    seed: int,
    excluded_questions: set[str],
) -> dict[str, dict[str, Any]]:
    """从 PQA-U parquet 固定随机抽取无重复问题的样本。"""
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = table.to_pylist()
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    output: dict[str, dict[str, Any]] = {}
    seen_questions = set(excluded_questions)
    for index in indices:
        row = rows[index]
        question = normalized(row.get("question"))
        context = row.get("context") or {}
        contexts = [normalized(text) for text in context.get("contexts", []) if normalized(text)]
        if not question or not contexts or question in seen_questions:
            continue
        pubid = str(row["pubid"])
        output[f"pqau_{pubid}"] = canonical_record(
            question, contexts, normalized(row.get("long_answer")), "PQA-U"
        )
        seen_questions.add(question)
        if len(output) == size:
            break
    if len(output) != size:
        raise RuntimeError(f"PQA-U 可用样本不足：需要 {size}，实际 {len(output)}")
    return output


def main() -> None:
    """解析参数、构建合并源文件并写出可复现实验清单。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pqal", required=True, type=Path)
    parser.add_argument("--pqau", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pqau-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pqau-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pqal = load_pqal(args.pqal)
    pqau = sample_pqau(
        args.pqau,
        size=args.pqau_size,
        seed=args.seed,
        excluded_questions={normalized(row["QUESTION"]) for row in pqal.values()},
    )
    merged = {**pqal, **pqau}
    questions = [normalized(row["QUESTION"]) for row in merged.values()]
    if len(questions) != len(set(questions)):
        raise RuntimeError("合并数据存在重复 query")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    args.pqau_output.parent.mkdir(parents=True, exist_ok=True)
    args.pqau_output.write_text(json.dumps(pqau, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "pqal_samples": len(pqal),
        "pqau_samples": len(pqau),
        "total_samples": len(merged),
        "pqau_random_seed": args.seed,
        "source": "PQA-L full + PQA-U random sample only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
