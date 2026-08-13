#!/usr/bin/env python3
"""T-SNE diagnostics for fixed historical original-Contriever outcome cases.

The four groups are selected from completed *historical* raw-dot/global-pool
experiments.  A group means attack target hit or target not hit in both the
historical black-box and HotFlip evaluations.  The points themselves use the
current valid black-box source and reproducibly regenerated original-Contriever
HotFlip texts, so the figure never claims a byte-for-byte replay of an old run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag_corpus import load_medrag_corpus_subset


GROUPS = {
    "pubmedqa_attack_target_hit": [
        "pubmedqa:22532370",
        "pubmedqa:18537964",
        "pubmedqa:25756710",
        "pubmedqa:23497210",
        "pubmedqa:25986020",
    ],
    "pubmedqa_attack_target_not_hit": [
        "pubmedqa:23621776",
        "pubmedqa:28359277",
        "pubmedqa:26209118",
        "pubmedqa:19836806",
        "pubmedqa:7497757",
    ],
    "medqa_attack_target_hit": [
        "medqa:0601",
        "medqa:0679",
        "medqa:0428",
        "medqa:0530",
        "medqa:0340",
    ],
    "medqa_attack_target_not_hit": [
        "medqa:0017",
        "medqa:0897",
        "medqa:1112",
        "medqa:1068",
        "medqa:0956",
    ],
}

SPECS = {
    "pubmedqa": {
        "dataset": "pubmed",
        "retrieval": "results/beir_results/mirage_pubmedqa_all-contriever.json",
        "source": "results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json",
        "blackbox": "results/query_results/main/pubmedqa_pubmed_contriever_gpt41mini_redo_blackbox.json",
        "hotflip": "results/query_results/main/pubmedqa_pubmed_contriever_gpt41mini_redo_hotflip.json",
    },
    "medqa": {
        # MedQA queries in this project retrieve from the same PubMed/MedRAG
        # chunk corpus as the historical original-Contriever run.
        "dataset": "pubmed",
        "retrieval": "results/beir_results/mirage_medqa_all-contriever.json",
        "source": "results/adv_targeted_results/mirage_medqa_all.json",
        "blackbox": "results/query_results/user_runs/medqa_full_gpt4.1mini_LMtargeted_gpu0_v2.json",
        "hotflip": "results/query_results/user_runs/medqa_full_gpt4.1mini_hotflip_gpu1_v2.json",
    },
}


def load_base_module():
    path = ROOT / "scripts" / "visualize_original_contriever_tsne.py"
    spec = importlib.util.spec_from_file_location("_original_tsne", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        rows = payload[0]["iter_0"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected result structure: {path}") from exc
    return {str(row["id"]): row for row in rows}


def label(value: object, dataset_key: str) -> str:
    text = str(value or "").strip()
    if dataset_key == "medqa":
        return text.upper()
    normalized = " ".join(text.lower().split())
    matches = [candidate for candidate in ("yes", "no", "maybe") if re.search(rf"(?<![a-z]){candidate}(?![a-z])", normalized)]
    return matches[0] if len(matches) == 1 else ""


def status_from_group(group_name: str) -> bool:
    return group_name.endswith("_attack_target_hit")


def attach_normal_contexts(selected: list[dict], dataset: str, retrieval: dict) -> None:
    requested = []
    for row in selected:
        ranking = retrieval.get(row["query_id"])
        if not isinstance(ranking, dict) or len(ranking) < 5:
            raise ValueError(f"Missing five normal retrieval documents for {row['query_id']}")
        row["normal_ids"] = list(ranking)[:5]
        row["top1_score"] = float(next(iter(ranking.values())))
        requested.extend(row["normal_ids"])
    corpus = load_medrag_corpus_subset(dataset, requested)
    for row in selected:
        normal = []
        for rank, doc_id in enumerate(row["normal_ids"], start=1):
            document = corpus.get(doc_id)
            text = " ".join(str((document or {}).get("text") or "").split())
            if not text:
                raise KeyError(f"Missing historical retrieved document {doc_id} for {row['query_id']}")
            normal.append({"doc_id": doc_id, "text": text, "retrieval_rank": rank})
        row["normal"] = normal


def prepare_selected(group_name: str, spec: dict) -> tuple[list[dict], list[dict]]:
    result_root = ROOT
    source = json.loads((result_root / spec["source"]).read_text(encoding="utf-8"))
    retrieval = json.loads((result_root / spec["retrieval"]).read_text(encoding="utf-8"))
    blackbox = load_rows(result_root / spec["blackbox"])
    hotflip = load_rows(result_root / spec["hotflip"])
    dataset_key = group_name.split("_")[0]
    expected_hit = status_from_group(group_name)
    selected = []
    audit_rows = []
    for query_id in GROUPS[group_name]:
        if query_id not in source or query_id not in retrieval or query_id not in blackbox or query_id not in hotflip:
            raise KeyError(f"Historical input alignment failed for {query_id}")
        source_row = source[query_id]
        target = label(source_row.get("target_label"), dataset_key)
        blackbox_prediction = label(blackbox[query_id].get("parsed_pred_label"), dataset_key)
        hotflip_prediction = label(hotflip[query_id].get("parsed_pred_label"), dataset_key)
        if not target or not blackbox_prediction or not hotflip_prediction:
            raise ValueError(f"Unparseable historical answer for {query_id}")
        blackbox_hit = blackbox_prediction == target
        hotflip_hit = hotflip_prediction == target
        if blackbox_hit != expected_hit or hotflip_hit != expected_hit:
            raise ValueError(
                f"{query_id} no longer satisfies {group_name}: "
                f"blackbox={blackbox_prediction}, hotflip={hotflip_prediction}, target={target}"
            )
        attacks = [" ".join(str(text).split()) for text in source_row.get("adv_texts", [])[:5]]
        if len(attacks) != 5 or any(not text for text in attacks):
            raise ValueError(f"Source has no complete own-5 black-box text group for {query_id}")
        task = group_name.replace("_", " ") + " · historical"
        selected.append(
            {
                "task": task,
                "query_id": query_id,
                "question": " ".join(str(source_row.get("question") or "").split()),
                "blackbox_i": attacks,
                "normal_ids": [],
                "top1_score": 0.0,
                "group": group_name,
            }
        )
        audit_rows.append(
            {
                "query_id": query_id,
                "source_target_label": target,
                "blackbox_prediction": blackbox_prediction,
                "hotflip_prediction": hotflip_prediction,
                "historical_blackbox_final_injected": len(blackbox[query_id].get("injected_adv", [])),
                "historical_hotflip_final_injected": len(hotflip[query_id].get("injected_adv", [])),
                "result_target_label_ignored": True,
            }
        )
    attach_normal_contexts(selected, spec["dataset"], retrieval)
    return selected, audit_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output-dir", default="results/embedding_viz/original_contriever_historical_outcomes_tsne")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base_module()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = {}
    audit: dict[str, list[dict]] = {}
    all_attacks: dict[str, dict] = {}
    for group_name in GROUPS:
        dataset_key = group_name.split("_")[0]
        selected, audit_rows = prepare_selected(group_name, SPECS[dataset_key])
        grouped[group_name] = selected
        audit[group_name] = audit_rows
        source = json.loads((ROOT / SPECS[dataset_key]["source"]).read_text(encoding="utf-8"))
        all_attacks.update(source)

    selected_all = [row for rows in grouped.values() for row in rows]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer, get_emb = base.load_models("contriever")
    model.to(device).eval()
    # This records the exact reproducibly regenerated HotFlip texts used for
    # the points. It is intentionally not represented as a historical replay.
    regenerated_hotflip = base.generate_hotflip(
        model, tokenizer, get_emb, all_attacks, selected_all, output_dir, args.seed
    )

    manifest = {
        "model_code": "contriever",
        "score_function": "dot",
        "seed": args.seed,
        "normal_k": 5,
        "historical_protocol": "raw dot / global candidate pool; diagnostic only, not formal own-5 evidence",
        "outcome_definition": "source target_label hit or not hit in both completed historical black-box and HotFlip result files",
        "text_policy": "black-box points use current valid source Q+I; HotFlip points are regenerated with original Contriever, not byte-for-byte historical replay",
        "groups": audit,
        "inputs": SPECS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for dataset_key in ("pubmedqa", "medqa"):
        rows = grouped[f"{dataset_key}_attack_target_hit"] + grouped[f"{dataset_key}_attack_target_not_hit"]
        records = base.make_records(rows, regenerated_hotflip)
        vectors = base.encode(model, tokenizer, get_emb, [record["text"] for record in records], device, max_length=128)
        base.add_scores(records, vectors)
        coords = base.tsne_coordinates(vectors, args.seed + (0 if dataset_key == "pubmedqa" else 1))
        for record, coord in zip(records, coords):
            record["tsne_x"], record["tsne_y"] = float(coord[0]), float(coord[1])
        np.savez_compressed(output_dir / f"{dataset_key}_raw_vectors.npz", vectors=vectors)
        base.write_csv(records, output_dir / f"{dataset_key}_points.csv")
        for group_name in (f"{dataset_key}_attack_target_hit", f"{dataset_key}_attack_target_not_hit"):
            paths = []
            for row in grouped[group_name]:
                path = output_dir / f"{group_name}_{row['query_id'].replace(':', '_')}.png"
                base.draw_case(records, coords, row["task"], row["query_id"], path)
                paths.append(path)
            base.make_montage(paths, output_dir / f"{group_name}_five_cases.png")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
