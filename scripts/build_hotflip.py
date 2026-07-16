"""使用与黑盒数据相同的固定 I 构建 HotFlip Contriever 三元组。"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.attack import Attacker
from src.utils import load_models
from utils import load_medqa, load_pubmedqa, normalize_text, setup_logger

LOGGER = logging.getLogger(__name__)


def _load(path, dataset):
    return load_pubmedqa(path) if dataset == "pubmedqa" else load_medqa(path)


def _suffix_i(query, negative):
    prefix = f"{normalize_text(query)}. "
    text = normalize_text(negative)
    if not text.startswith(prefix):
        raise ValueError("blackbox negative does not have the expected query + I form")
    suffix = text[len(prefix):].strip()
    if not suffix:
        raise ValueError("blackbox I is empty")
    return suffix


def build(input_path, blackbox_path, output_path, dataset, limit=0, gpu=0):
    samples = _load(input_path, dataset)
    if limit:
        samples = samples[:limit]
    blackbox = [json.loads(line) for line in open(blackbox_path, encoding="utf-8") if line.strip()]
    black_by_pair = {(x["query"], x["positive"]): x for x in blackbox}
    if len(black_by_pair) != len(blackbox):
        raise ValueError("blackbox output has duplicate query/positive pairs")
    if len(samples) != len(blackbox):
        raise ValueError("blackbox output count does not match source count")

    adv_map, targets = {}, []
    for sample in samples:
        key = (sample["query"], sample["positive"])
        row = black_by_pair.get(key)
        if row is None or row.get("attack_type") != "blackbox":
            raise ValueError("missing or invalid matching blackbox sample")
        adv_map[sample["id"]] = {"adv_texts": [_suffix_i(sample["query"], row["negative"])]}
        targets.append({"id": sample["id"], "query": sample["query"], "top1_score": 0.0})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as tmp:
        json.dump(adv_map, tmp, ensure_ascii=False)
        tmp_path = tmp.name
    try:
        device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        model, c_model, tokenizer, get_emb = load_models("contriever")
        model.to(device).eval()
        c_model.to(device).eval()
        args = SimpleNamespace(attack_method="hotflip", adv_per_query=1, eval_dataset="training",
                               adv_json_path=tmp_path, adv_source="json", score_function="dot")
        attacker = Attacker(args, model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb,
                            max_seq_length=128, pad_to_max_length=True, num_adv_passage_tokens=30,
                            num_cand=30, num_iter=15, gold_init=True, early_stop=False)
        groups = attacker.hotflip(targets)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if len(groups) != len(samples) or any(len(group) != 1 for group in groups):
        raise RuntimeError("HotFlip did not return exactly one negative per source sample")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for sample, group in zip(samples, groups):
            negative = normalize_text(group[0])
            if not negative or negative == normalize_text(sample["positive"]):
                raise RuntimeError("invalid HotFlip negative")
            out.write(json.dumps({"query": sample["query"], "positive": sample["positive"],
                                  "negative": negative, "attack_type": "hotflip"}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--blackbox-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True, choices=["pubmedqa", "medqa"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logger(args.verbose)
    build(args.input, args.blackbox_input, args.output, args.dataset, args.limit, args.gpu)


if __name__ == "__main__":
    main()
