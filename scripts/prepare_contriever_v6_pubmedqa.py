#!/usr/bin/env python3
"""Prepare a leakage-free PubMedQA component for the Contriever v6 control.

v4 combines 9,174 PQA-U examples with 1,000 PQA-L examples.  The latter
contains every MIRAGE PubMedQA evaluation question.  This utility retains the
verified PQA-U component and deterministically replaces the PQA-L component
with unused PQA-U examples, rejecting every normalized MIRAGE question.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def normalize_question(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mirage_questions(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        benchmark = json.load(handle)
    return {
        normalize_question(row["question"])
        for row in benchmark["pubmedqa"].values()
        if normalize_question(row.get("question"))
    }


def to_source_record(row: Any) -> dict[str, Any]:
    context = row["context"]
    if hasattr(context, "tolist"):
        context = context.tolist()
    if not isinstance(context, list):
        context = [str(context)]
    return {
        "QUESTION": str(row["question"]),
        "CONTEXTS": [str(item) for item in context],
        "LONG_ANSWER": str(row["long_answer"]),
        "SOURCE_SUBSET": "PQA-U",
    }


def prepare(args: argparse.Namespace) -> None:
    import pandas as pd

    blocked = mirage_questions(Path(args.mirage_benchmark))
    with Path(args.existing_pqau_source).open(encoding="utf-8") as handle:
        existing = json.load(handle)
    if not isinstance(existing, dict):
        raise TypeError("existing PQA-U source must be a JSON object")
    existing_questions = {normalize_question(row.get("QUESTION")) for row in existing.values()}
    if blocked & existing_questions:
        raise ValueError("existing PQA-U source unexpectedly overlaps MIRAGE PubMedQA")

    frame = pd.read_parquet(args.pqau_parquet)
    required = {"pubid", "question", "context", "long_answer"}
    if missing := required - set(frame.columns):
        raise ValueError(f"PQA-U parquet is missing columns: {sorted(missing)}")
    candidates = []
    for _, row in frame.iterrows():
        question = normalize_question(row["question"])
        if not question or question in blocked or question in existing_questions:
            continue
        candidates.append(row)
    if len(candidates) < args.replacement_count:
        raise RuntimeError(f"only {len(candidates)} safe PQA-U candidates for {args.replacement_count} replacements")
    selected = random.Random(args.seed).sample(candidates, args.replacement_count)

    replacements: dict[str, dict[str, Any]] = {}
    for row in selected:
        key = f"pqau_v6_{row['pubid']}"
        if key in replacements:
            raise ValueError(f"duplicate PQA-U pubid: {row['pubid']}")
        replacements[key] = to_source_record(row)
    selected_questions = {normalize_question(row["QUESTION"]) for row in replacements.values()}
    if len(selected_questions) != len(replacements) or blocked & selected_questions or existing_questions & selected_questions:
        raise ValueError("replacement query validation failed")

    combined = {**existing, **replacements}
    if len(combined) != len(existing) + len(replacements):
        raise ValueError("combined source key collision")
    Path(args.output_replacement_source).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_replacement_source).open("w", encoding="utf-8") as handle:
        json.dump(replacements, handle, ensure_ascii=False, indent=2)
    with Path(args.output_combined_source).open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, indent=2)
    manifest = {
        "seed": args.seed,
        "mirage_pubmedqa_questions": len(blocked),
        "retained_pqau_examples": len(existing),
        "replacement_pqau_examples": len(replacements),
        "combined_examples": len(combined),
        "mirage_overlap": 0,
    }
    with Path(args.output_manifest).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))


def merge(args: argparse.Namespace) -> None:
    blocked = mirage_questions(Path(args.mirage_benchmark))
    base_blackbox, base_hotflip = read_jsonl(Path(args.base_blackbox)), read_jsonl(Path(args.base_hotflip))
    replacement_blackbox = read_jsonl(Path(args.replacement_blackbox))
    replacement_hotflip = read_jsonl(Path(args.replacement_hotflip))
    if len(replacement_blackbox) != len(replacement_hotflip):
        raise ValueError("replacement blackbox/hotflip counts differ")
    for name, rows in (("base blackbox", base_blackbox), ("base hotflip", base_hotflip), ("replacement blackbox", replacement_blackbox), ("replacement hotflip", replacement_hotflip)):
        if any(not normalize_question(row.get("query")) for row in rows):
            raise ValueError(f"empty query in {name}")
        if any(normalize_question(row.get("query")) in blocked for row in rows):
            raise ValueError(f"MIRAGE query leaked into {name}")
    blackbox, hotflip = base_blackbox + replacement_blackbox, base_hotflip + replacement_hotflip
    if len(blackbox) != args.expected_total or len(hotflip) != args.expected_total:
        raise ValueError(f"expected {args.expected_total} rows, got blackbox={len(blackbox)}, hotflip={len(hotflip)}")
    black_pairs = [(normalize_question(row.get("query")), str(row.get("positive", "")).strip()) for row in blackbox]
    hot_pairs = [(normalize_question(row.get("query")), str(row.get("positive", "")).strip()) for row in hotflip]
    if len(set(black_pairs)) != len(black_pairs) or set(black_pairs) != set(hot_pairs):
        raise ValueError("merged blackbox/hotflip pairs are not one-to-one")
    write_jsonl(Path(args.output_blackbox), blackbox)
    write_jsonl(Path(args.output_hotflip), hotflip)
    print(json.dumps({"combined_examples": len(blackbox), "mirage_overlap": 0}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--pqau-parquet", required=True)
    prepare_parser.add_argument("--existing-pqau-source", required=True)
    prepare_parser.add_argument("--mirage-benchmark", default="MIRAGE/benchmark.json")
    prepare_parser.add_argument("--replacement-count", type=int, default=1000)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--output-replacement-source", required=True)
    prepare_parser.add_argument("--output-combined-source", required=True)
    prepare_parser.add_argument("--output-manifest", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--base-blackbox", required=True)
    merge_parser.add_argument("--base-hotflip", required=True)
    merge_parser.add_argument("--replacement-blackbox", required=True)
    merge_parser.add_argument("--replacement-hotflip", required=True)
    merge_parser.add_argument("--mirage-benchmark", default="MIRAGE/benchmark.json")
    merge_parser.add_argument("--expected-total", type=int, default=10174)
    merge_parser.add_argument("--output-blackbox", required=True)
    merge_parser.add_argument("--output-hotflip", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    else:
        merge(arguments)
