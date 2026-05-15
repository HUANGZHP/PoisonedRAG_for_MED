#!/usr/bin/env python3

"""Unified experiment runner.

This file merges previous run entrypoints into a single script:
1) legacy-main: old batch behavior that calls main.py directly.
2) mirage-blackbox: large-scale MIRAGE attack/no-attack pipeline.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs_path(path: str) -> str:
    p = (path or "").strip()
    if os.path.isabs(p):
        return p
    return os.path.join(_repo_root(), p)


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_gpu_ids(raw_gpu_ids: str) -> List[int]:
    out = []
    for part in (raw_gpu_ids or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            gid = int(token)
        except ValueError:
            continue
        if gid < 0 or gid in out:
            continue
        out.append(gid)
    return out


@dataclass
class GpuInfo:
    index: int
    free_mb: int
    util: int


def _detect_available_gpus(min_free_mem_gb: int, max_gpus: int) -> List[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        print(f"Warning: failed to query GPUs via nvidia-smi: {exc}")
        return []

    min_free_mb = int(min_free_mem_gb) * 1024
    rows: List[GpuInfo] = []
    for line in output.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(parts[1])
            util = int(parts[2])
        except ValueError:
            continue
        if free_mb >= min_free_mb:
            rows.append(GpuInfo(index=idx, free_mb=free_mb, util=util))

    rows.sort(key=lambda x: (-x.free_mb, x.util, x.index))
    return [r.index for r in rows[: max(1, int(max_gpus))]]


def _load_ids(ids_path: str) -> List[str]:
    ids = []
    with open(ids_path, "r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if qid:
                ids.append(qid)
    return ids


def _save_ids(ids_path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(ids_path) or ".", exist_ok=True)
    with open(ids_path, "w", encoding="utf-8") as f:
        for qid in ids:
            f.write(qid + "\n")


def _filter_json_by_ids(json_path: str, ids: List[str]) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict root in {json_path}")

    filtered = {}
    for qid in ids:
        if qid in data:
            filtered[qid] = data[qid]

    missing = [qid for qid in ids if qid not in filtered]
    if missing:
        raise ValueError(f"{json_path} missing ids, first few: {missing[:5]}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _compute_metrics(result_path: str, top_k: int, adv_per_query: int) -> Dict:
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid result json format: {result_path}")

    asr_flags = []
    ret_list = []
    total_rows = 0

    for item in data:
        if not isinstance(item, dict) or not item:
            continue
        iter_key = next(iter(item.keys()))
        rows = item.get(iter_key, [])
        if not isinstance(rows, list):
            continue

        for row in rows:
            pred = str(row.get("output_poison", "")).strip().lower()
            wrong = str(row.get("incorrect_answer", row.get("incorrect answer", ""))).strip().lower()
            asr_flags.append(1 if (wrong and wrong in pred) else 0)
            ret_list.append(len(row.get("injected_adv") or []))
            total_rows += 1

    n = total_rows
    asr_mean = float(sum(asr_flags)) / float(n) if n > 0 else 0.0
    ret_total = int(sum(ret_list))
    precision = float(ret_total) / float(n * top_k) if n > 0 and top_k > 0 else 0.0
    recall = float(ret_total) / float(n * adv_per_query) if n > 0 and adv_per_query > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "n": n,
        "asr_mean": asr_mean,
        "ret_total": ret_total,
        "ret_list": ret_list,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _run_legacy_main_once(test_params: Dict) -> None:
    log_file, log_name = _legacy_get_log_name(test_params)
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--eval_model_code",
        str(test_params["eval_model_code"]),
        "--eval_dataset",
        str(test_params["eval_dataset"]),
        "--split",
        str(test_params["split"]),
        "--query_results_dir",
        str(test_params["query_results_dir"]),
        "--model_name",
        str(test_params["model_name"]),
        "--top_k",
        str(test_params["top_k"]),
        "--use_truth",
        str(test_params["use_truth"]),
        "--gpu_id",
        str(test_params["gpu_id"]),
        "--attack_method",
        str(test_params["attack_method"]),
        "--adv_per_query",
        str(test_params["adv_per_query"]),
        "--score_function",
        str(test_params["score_function"]),
        "--repeat_times",
        str(test_params["repeat_times"]),
        "--M",
        str(test_params["M"]),
        "--seed",
        str(test_params["seed"]),
        "--name",
        str(log_name),
    ]

    run_in_background = bool(test_params.get("run_in_background", False))
    with open(log_file, "w", encoding="utf-8") as fout:
        if run_in_background:
            if os.name == "nt":
                subprocess.Popen(
                    cmd,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT, start_new_session=True)
        else:
            subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, check=False)


def _legacy_get_log_name(test_params: Dict):
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params["use_truth"] in [True, "True"]:
        log_name = (
            f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-"
            f"{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
        )
    else:
        log_name = (
            f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-"
            f"{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
        )

    if test_params["attack_method"] not in [None, "None"]:
        log_name += (
            f"-adv-{test_params['attack_method']}-{test_params['score_function']}-"
            f"{test_params['adv_per_query']}-{test_params['top_k']}"
        )

    if test_params["note"] is not None:
        log_name = test_params["note"]

    return f"logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name


def run_legacy_main(args) -> None:
    params = {
        "eval_model_code": args.eval_model_code,
        "eval_dataset": "pubmed",
        "split": args.split,
        "query_results_dir": args.query_results_dir,
        "model_name": args.model_name,
        "use_truth": args.use_truth,
        "top_k": args.top_k,
        "gpu_id": args.gpu_id,
        "attack_method": args.attack_method,
        "adv_per_query": args.adv_per_query,
        "score_function": args.score_function,
        "repeat_times": args.repeat_times,
        "M": args.M,
        "seed": args.seed,
        "note": args.note,
        "run_in_background": args.run_in_background,
    }

    for dataset in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        params["eval_dataset"] = dataset
        _run_legacy_main_once(params)


def run_mirage_blackbox(args) -> None:
    repo = _repo_root()

    if args.M <= 0:
        raise ValueError("--M must be positive")
    if args.selected_questions <= 0:
        raise ValueError("--selected_questions must be positive")

    selected_questions = int(args.selected_questions)
    if selected_questions % args.M != 0:
        selected_questions = (selected_questions // args.M) * args.M
        if selected_questions <= 0:
            raise ValueError("selected_questions is too small after aligning with M")
        print(f"Adjusted selected_questions to {selected_questions} to be divisible by M={args.M}")

    explicit_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if explicit_gpu_ids:
        picked_gpu_ids = explicit_gpu_ids
    else:
        picked_gpu_ids = _detect_available_gpus(int(args.min_free_mem_gb), int(args.max_gpus))
    if not picked_gpu_ids:
        picked_gpu_ids = [0]

    gpu_ids_str = ",".join(str(x) for x in picked_gpu_ids)
    primary_gpu = picked_gpu_ids[0]
    print(f"Using GPUs: {picked_gpu_ids} (primary={primary_gpu})")

    adv_json_rel = f"results/adv_targeted_results/{args.tag}.json"
    ids_rel = f"results/adv_targeted_results/{args.tag}.ids"
    ids_path = _abs_path(ids_rel)

    gen_cmd = [
        sys.executable,
        "-u",
        "gen_adv.py",
        "--eval_model_code",
        args.eval_model_code,
        "--eval_dataset",
        args.eval_dataset,
        "--adv_per_query",
        str(args.adv_per_query),
        "--data_num",
        str(selected_questions),
        "--seed",
        str(args.seed),
        "--retrieval_topn",
        str(args.retrieval_topn),
        "--docs_per_adv",
        str(args.docs_per_adv),
        "--output_file",
        adv_json_rel,
        "--ids_file",
        ids_rel,
    ]
    _run(gen_cmd)

    selected_ids = _load_ids(ids_path)
    if len(selected_ids) < args.M:
        raise ValueError(f"Selected ids too few ({len(selected_ids)}) for M={args.M}")

    usable_n = (len(selected_ids) // args.M) * args.M
    if usable_n != len(selected_ids):
        selected_ids = selected_ids[:usable_n]
        _save_ids(ids_path, selected_ids)
        print(f"Trimmed id list to {usable_n} for exact M-divisible evaluation")

    if args.repeat_times <= 0:
        repeat_times = len(selected_ids) // args.M
    else:
        repeat_times = min(int(args.repeat_times), len(selected_ids) // args.M)
    if repeat_times <= 0:
        raise ValueError("repeat_times resolved to 0; check selected ids and M")

    effective_n = repeat_times * args.M
    if effective_n < len(selected_ids):
        selected_ids = selected_ids[:effective_n]
        _save_ids(ids_path, selected_ids)
        print(f"Trimmed id list to first {effective_n} ids to match repeat_times={repeat_times}")

    attack_name = f"{args.tag}_attack"
    noattack_name = f"{args.tag}_noattack"

    common = [
        "--eval_model_code",
        args.eval_model_code,
        "--eval_dataset",
        args.eval_dataset,
        "--split",
        args.split,
        "--query_results_dir",
        args.query_results_dir,
        "--model_name",
        args.model_name,
        "--top_k",
        str(args.top_k),
        "--use_truth",
        "False",
        "--gpu_id",
        str(primary_gpu),
        "--gpu_ids",
        gpu_ids_str,
        "--adv_source",
        "json",
        "--adv_json_path",
        adv_json_rel,
        "--target_ids_path",
        ids_rel,
        "--adv_per_query",
        str(args.adv_per_query),
        "--score_function",
        args.score_function,
        "--repeat_times",
        str(repeat_times),
        "--M",
        str(args.M),
        "--seed",
        str(args.seed),
    ]

    attack_cmd = [
        sys.executable,
        "-u",
        "main.py",
        *common,
        "--attack_method",
        "LM_targeted",
        "--name",
        attack_name,
    ]
    _run(attack_cmd)

    noattack_cmd = [
        sys.executable,
        "-u",
        "main.py",
        *common,
        "--attack_method",
        "None",
        "--name",
        noattack_name,
    ]
    _run(noattack_cmd)

    attack_res_path = os.path.join(repo, "results", "query_results", args.query_results_dir, f"{attack_name}.json")
    noattack_res_path = os.path.join(repo, "results", "query_results", args.query_results_dir, f"{noattack_name}.json")

    attack_metrics = _compute_metrics(attack_res_path, top_k=args.top_k, adv_per_query=args.adv_per_query)
    noattack_metrics = _compute_metrics(noattack_res_path, top_k=args.top_k, adv_per_query=args.adv_per_query)

    delta = {
        "asr": attack_metrics["asr_mean"] - noattack_metrics["asr_mean"],
        "ret_total": attack_metrics["ret_total"] - noattack_metrics["ret_total"],
        "precision": attack_metrics["precision"] - noattack_metrics["precision"],
        "recall": attack_metrics["recall"] - noattack_metrics["recall"],
        "f1": attack_metrics["f1"] - noattack_metrics["f1"],
    }

    report = {
        "config": {
            "eval_model_code": args.eval_model_code,
            "eval_dataset": args.eval_dataset,
            "model_name": args.model_name,
            "query_results_dir": args.query_results_dir,
            "selected_questions": selected_questions,
            "effective_questions": effective_n,
            "adv_per_query": args.adv_per_query,
            "top_k": args.top_k,
            "score_function": args.score_function,
            "M": args.M,
            "repeat_times": repeat_times,
            "seed": args.seed,
            "gpu_ids": picked_gpu_ids,
            "adv_json": adv_json_rel,
            "ids_file": ids_rel,
            "attack_result": os.path.relpath(attack_res_path, repo),
            "noattack_result": os.path.relpath(noattack_res_path, repo),
        },
        "attack": attack_metrics,
        "noattack": noattack_metrics,
        "delta_attack_minus_noattack": delta,
    }

    report_rel = os.path.join("results", "query_results", args.query_results_dir, f"{args.tag}_comparison.json")
    report_path = _abs_path(report_rel)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\n=== Comparison Summary ===")
    print(f"Questions evaluated: {effective_n}")
    print(f"ASR attack/no-attack: {attack_metrics['asr_mean']:.4f} / {noattack_metrics['asr_mean']:.4f}")
    print(f"Ret total attack/no-attack: {attack_metrics['ret_total']} / {noattack_metrics['ret_total']}")
    print(f"Precision attack/no-attack: {attack_metrics['precision']:.4f} / {noattack_metrics['precision']:.4f}")
    print(f"Comparison report: {report_rel}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified runner for PoisonedRAG pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    legacy = subparsers.add_parser("legacy-main", help="Run legacy batch main.py experiments")
    legacy.add_argument("--datasets", type=str, default="pubmed,statpearls,textbooks")
    legacy.add_argument("--eval_model_code", type=str, default="medcpt")
    legacy.add_argument("--split", type=str, default="test")
    legacy.add_argument("--query_results_dir", type=str, default="main")
    legacy.add_argument("--model_name", type=str, default="gpt4")
    legacy.add_argument("--use_truth", type=str, default="False")
    legacy.add_argument("--top_k", type=int, default=5)
    legacy.add_argument("--gpu_id", type=int, default=0)
    legacy.add_argument("--attack_method", type=str, default="LM_targeted")
    legacy.add_argument("--adv_per_query", type=int, default=5)
    legacy.add_argument("--score_function", type=str, default="dot", choices=["dot", "cos_sim"])
    legacy.add_argument("--repeat_times", type=int, default=10)
    legacy.add_argument("--M", type=int, default=10)
    legacy.add_argument("--seed", type=int, default=12)
    legacy.add_argument("--note", type=str, default=None)
    legacy.add_argument("--run_in_background", action="store_true")

    mirage = subparsers.add_parser("mirage-blackbox", help="Run large-scale MIRAGE black-box experiment")
    mirage.add_argument("--eval_model_code", type=str, default="medcpt", choices=["medcpt"])
    mirage.add_argument("--eval_dataset", type=str, default="pubmed", choices=["pubmed"])
    mirage.add_argument("--split", type=str, default="test")
    mirage.add_argument("--model_name", type=str, default="gpt4")
    mirage.add_argument("--query_results_dir", type=str, default="large")
    mirage.add_argument("--selected_questions", type=int, default=300)
    mirage.add_argument("--adv_per_query", type=int, default=5)
    mirage.add_argument("--retrieval_topn", type=int, default=80)
    mirage.add_argument("--docs_per_adv", type=int, default=2)
    mirage.add_argument("--top_k", type=int, default=5)
    mirage.add_argument("--score_function", type=str, default="dot", choices=["dot", "cos_sim"])
    mirage.add_argument("--M", type=int, default=30)
    mirage.add_argument("--repeat_times", type=int, default=0)
    mirage.add_argument("--seed", type=int, default=12)
    mirage.add_argument("--gpu_ids", type=str, default="")
    mirage.add_argument("--max_gpus", type=int, default=4)
    mirage.add_argument("--min_free_mem_gb", type=int, default=20)
    mirage.add_argument("--tag", type=str, default="mirage_pubmed_medcpt_blackbox_large")

    return parser


def _run_default_legacy_batch() -> None:
    params = {
        "eval_model_code": "medcpt",
        "eval_dataset": "pubmed",
        "split": "test",
        "query_results_dir": "main",
        "model_name": "gpt4",
        "use_truth": "False",
        "top_k": 5,
        "gpu_id": 3,
        "attack_method": "LM_targeted",
        "adv_per_query": 5,
        "score_function": "dot",
        "repeat_times": 10,
        "M": 10,
        "seed": 12,
        "note": None,
        "run_in_background": False,
    }
    for dataset in ["pubmed", "statpearls", "textbooks"]:
        params["eval_dataset"] = dataset
        _run_legacy_main_once(params)


def main() -> None:
    if len(sys.argv) == 1:
        print("No subcommand provided, running legacy default batch.")
        _run_default_legacy_batch()
        return

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "legacy-main":
        run_legacy_main(args)
    elif args.command == "mirage-blackbox":
        run_mirage_blackbox(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()