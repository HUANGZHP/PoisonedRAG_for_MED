"""构建黑盒 Contriever 三元组：P_blackbox = query + I。

LLM 只生成固定的对抗后缀 I；query 是未修改的 S 部分，因此负例严格
遵循仓库中的 LM_targeted 攻击形式。
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_utils import call_llm, build_blackbox_prompt
from utils import load_medqa, load_pubmedqa, normalize_text, setup_logger

LOGGER = logging.getLogger(__name__)
CONCURRENCY = 5


def _load(path, dataset):
    return load_pubmedqa(path) if dataset == "pubmedqa" else load_medqa(path)


def _compose(query, suffix_i):
    return f"{normalize_text(query)}. {normalize_text(suffix_i)}"


def _generate(sample):
    query, positive = sample["query"], sample["positive"]
    prompt = build_blackbox_prompt(query, positive)
    for _ in range(5):
        suffix_i = normalize_text(call_llm(
            prompt=prompt,
            system_msg="You generate only the adversarial medical suffix I. Output no labels or commentary.",
            temperature=0.8,
            max_tokens=1024,
        ) or "")
        negative = _compose(query, suffix_i)
        if suffix_i and normalize_text(negative) != normalize_text(positive):
            return {"query": query, "positive": positive, "negative": negative,
                    "attack_type": "blackbox"}
    return None


def build(input_path, output_path, dataset, limit=0):
    samples = _load(input_path, dataset)
    if limit:
        samples = samples[:limit]
    pairs = [(s["query"], s["positive"]) for s in samples]
    if len(set(pairs)) != len(pairs):
        raise ValueError("source contains duplicate query/positive pairs")
    results, failures = {}, []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        pending = {executor.submit(_generate, sample): idx for idx, sample in enumerate(samples)}
        for future in as_completed(pending):
            idx = pending[future]
            item = future.result()
            if item is None:
                failures.append(idx)
            else:
                results[idx] = item
            done = len(results) + len(failures)
            if done % 50 == 0 or done == len(samples):
                LOGGER.info("blackbox %d/%d", done, len(samples))
    if failures:
        raise RuntimeError(f"blackbox generation failed for {len(failures)} samples")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for idx in range(len(samples)):
            out.write(json.dumps(results[idx], ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True, choices=["pubmedqa", "medqa"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logger(args.verbose)
    build(args.input, args.output, args.dataset, args.limit)


if __name__ == "__main__":
    main()
