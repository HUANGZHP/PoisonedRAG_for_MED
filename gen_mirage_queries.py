import argparse
import json
import os
from typing import Dict


def _load_benchmark(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark json at {path}")
    return data


def _normalize_answer(text: str) -> str:
    return str(text or "").strip().lower()


def _flip_pubmedqa_answer(answer_text: str) -> str:
    ans = _normalize_answer(answer_text)
    if ans == "yes":
        return "no"
    if ans == "no":
        return "yes"
    if ans == "maybe":
        return "no"
    return "no"


def _pick_incorrect_multiple_choice(item: Dict, answer_key: str) -> str:
    """For multi-choice datasets, pick a wrong option letter (different key)."""
    options = item.get("options", {})
    if not isinstance(options, dict):
        return ""
    keys = [str(k).strip() for k in sorted(options.keys()) if str(k).strip()]
    if len(keys) < 2:
        return ""
    wrong_key = answer_key
    for k in keys:
        if k != answer_key:
            wrong_key = k
            break
    return str(wrong_key).strip()


def _resolve_correct_answer(item: Dict) -> str:
    answer_key = str(item.get("answer", "")).strip()
    options = item.get("options", {})
    if isinstance(options, dict) and answer_key in options:
        return str(options[answer_key]).strip()
    return str(item.get("answer", "")).strip()


def _resolve_correct_option(item: Dict) -> str:
    answer_key = str(item.get("answer", "")).strip()
    options = item.get("options", {})
    if isinstance(options, dict) and answer_key in options:
        return answer_key
    return ""


def _detect_question_type(item: Dict) -> str:
    options = item.get("options", {})
    answer_key = str(item.get("answer", "")).strip()
    if isinstance(options, dict) and answer_key in options:
        return "mcq"
    ans_norm = _normalize_answer(answer_key)
    if ans_norm in ["yes", "no", "maybe"]:
        return "yesno"
    return "freeform"


def _get_incorrect_answer(item: Dict, dataset: str, question_type: str) -> str:
    answer_key = str(item.get("answer", "")).strip()
    if question_type == "yesno":
        correct = _resolve_correct_answer(item)
        return _flip_pubmedqa_answer(correct)
    if question_type == "mcq":
        return _pick_incorrect_multiple_choice(item, answer_key)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MIRAGE PubMedQA to PoisonedRAG query format")
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=os.path.join("MIRAGE", "benchmark.json"),
        help="Path to MIRAGE benchmark.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmedqa",
        help="MIRAGE subset name to export (e.g., pubmedqa)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of questions to export (<=0 means all)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Output prefix for json/ids/queries files",
    )
    args = parser.parse_args()

    dataset = (args.dataset or "").strip().lower()
    if not dataset:
        raise ValueError("--dataset cannot be empty")

    benchmark = _load_benchmark(args.benchmark_path)
    if dataset not in benchmark:
        raise KeyError(f"Dataset '{dataset}' not found in {args.benchmark_path}")

    subset = benchmark[dataset]
    if not isinstance(subset, dict):
        raise ValueError(f"Dataset '{dataset}' has invalid format in {args.benchmark_path}")

    limit = int(args.limit)
    ids = sorted(subset.keys())
    if limit > 0:
        ids = ids[:limit]

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        size_label = f"n{len(ids)}" if limit > 0 else "all"
        output_prefix = os.path.join("results", "adv_targeted_results", f"mirage_{dataset}_{size_label}")

    adv_json = {}
    queries = {}
    ids_lines = []

    for qid in ids:
        item = subset.get(qid, {})
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        question_type = _detect_question_type(item)
        correct = _resolve_correct_answer(item)
        correct_option = _resolve_correct_option(item)
        incorrect = _get_incorrect_answer(item, dataset, question_type)
        query_id = f"{dataset}:{qid}"
        adv_json[query_id] = {
            "id": query_id,
            "question": question,
            "correct answer": correct,
            "incorrect answer": incorrect,
            "question_type": question_type,
            "options": item.get("options", {}) if isinstance(item.get("options", {}), dict) else {},
            "correct_option": correct_option,
            "adv_texts": [],
        }
        queries[query_id] = question
        ids_lines.append(query_id)

    if not adv_json:
        raise ValueError("No valid questions found to export")

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    with open(output_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(adv_json, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(output_prefix + ".ids", "w", encoding="utf-8") as f:
        for qid in ids_lines:
            f.write(qid + "\n")

    with open(output_prefix + ".queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Exported {len(ids_lines)} questions to {output_prefix}.*")


if __name__ == "__main__":
    main()
