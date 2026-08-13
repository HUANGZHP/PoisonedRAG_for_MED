#!/usr/bin/env python3
"""Create reproducible local t-SNE diagnostics for original Contriever attacks.

Each selected MIRAGE query contributes its query vector, original-Contriever
top-5 normal documents, five Q+I black-box texts, and five HotFlip white-box
texts.  One joint projection is fit per benchmark, then exported as five
case-level plots so local locations are comparable inside that benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attack import Attacker
from src.medrag_corpus import load_medrag_corpus_subset
from src.utils import load_models


COLORS = {
    "query": (25, 25, 35),
    "normal": (48, 108, 196),
    "blackbox": (202, 63, 63),
    "hotflip": (226, 133, 37),
    "grid": (205, 210, 218),
    "line": (177, 183, 193),
    "text": (30, 34, 42),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--normal-k", type=int, default=5)
    parser.add_argument("--output-dir", default="results/embedding_viz/original_contriever_dot_tsne")
    parser.add_argument(
        "--pubmed-retrieval",
        default="results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json",
    )
    parser.add_argument(
        "--medqa-retrieval",
        default="results/beir_results/formal_dot_contriever/mirage_medqa_all-contriever-dot.json",
    )
    return parser.parse_args()


def compact(text: object) -> str:
    return " ".join(str(text or "").split())


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_ids(retrieval: dict, attacks: dict, count: int, seed: int) -> list[str]:
    candidates = [qid for qid, docs in retrieval.items() if qid in attacks and isinstance(docs, dict) and docs]
    if len(candidates) < count:
        raise ValueError(f"Need {count} eligible query ids, found {len(candidates)}")
    return random.Random(seed).sample(sorted(candidates), count)


def encode(model, tokenizer, get_emb, texts: list[str], device: torch.device, max_length: int = 128) -> np.ndarray:
    vectors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            vectors.append(get_emb(model, inputs).detach().cpu())
    return torch.cat(vectors, dim=0).numpy().astype(np.float32)


def generate_hotflip(model, tokenizer, get_emb, attacks: dict, selected: list[dict], out_dir: Path, seed: int) -> dict:
    """Regenerate all own-5 white-box texts under the exact original Contriever."""
    tmp_adv = out_dir / "selected_blackbox_source.json"
    with tmp_adv.open("w", encoding="utf-8") as handle:
        json.dump({row["query_id"]: attacks[row["query_id"]] for row in selected}, handle, ensure_ascii=False, indent=2)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = SimpleNamespace(
        attack_method="hotflip",
        adv_per_query=5,
        eval_dataset="pubmed",
        adv_json_path=str(tmp_adv),
        adv_source="json",
        score_function="dot",
    )
    attacker = Attacker(
        args,
        model=model,
        c_model=model,
        tokenizer=tokenizer,
        get_emb=get_emb,
        max_seq_length=128,
        pad_to_max_length=True,
        num_adv_passage_tokens=30,
        num_cand=30,
        num_iter=15,
        gold_init=True,
        early_stop=False,
    )
    targets = [{"id": row["query_id"], "query": row["question"], "top1_score": row["top1_score"]} for row in selected]
    groups = attacker.hotflip(targets)
    if len(groups) != len(selected) or any(len(group) != 5 for group in groups):
        raise RuntimeError("HotFlip did not produce five texts for every selected query")
    output = {row["query_id"]: [compact(text) for text in group] for row, group in zip(selected, groups)}
    with (out_dir / "whitebox_hotflip_own5.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    return output


def prepare_task(task: str, retrieval_path: str, attack_path: str, normal_k: int, samples: int, seed: int):
    retrieval = load_json(retrieval_path)
    attacks = load_json(attack_path)
    query_ids = choose_ids(retrieval, attacks, samples, seed)
    selected = []
    requested_doc_ids = []
    for qid in query_ids:
        row = attacks[qid]
        ranking = retrieval[qid]
        normal_ids = list(ranking)[:normal_k]
        requested_doc_ids.extend(normal_ids)
        selected.append(
            {
                "task": task,
                "query_id": qid,
                "question": compact(row["question"]),
                "blackbox_i": [compact(text) for text in row["adv_texts"][:5]],
                "normal_ids": normal_ids,
                "top1_score": float(next(iter(ranking.values()))),
            }
        )
    corpus = load_medrag_corpus_subset("pubmed", requested_doc_ids)
    for row in selected:
        normal = []
        for rank, doc_id in enumerate(row["normal_ids"], start=1):
            doc = corpus.get(doc_id)
            if doc is None or not compact(doc.get("text")):
                raise KeyError(f"Missing retrieved document {doc_id} for {row['query_id']}")
            normal.append({"doc_id": doc_id, "text": compact(doc["text"]), "retrieval_rank": rank})
        row["normal"] = normal
    return selected, attacks


def make_records(selected: list[dict], whitebox: dict) -> list[dict]:
    records = []
    for row in selected:
        qid = row["query_id"]
        records.append(
            {
                "task": row["task"], "query_id": qid, "kind": "query", "ordinal": 0,
                "label": "Q", "text": row["question"], "doc_id": "", "retrieval_rank": "",
            }
        )
        for normal in row["normal"]:
            records.append(
                {
                    "task": row["task"], "query_id": qid, "kind": "normal", "ordinal": normal["retrieval_rank"],
                    "label": f"N{normal['retrieval_rank']}", "text": normal["text"], "doc_id": normal["doc_id"],
                    "retrieval_rank": normal["retrieval_rank"],
                }
            )
        for index, evidence in enumerate(row["blackbox_i"], start=1):
            # Match LM_targeted's actual retrieval input: Q + period + I.
            records.append(
                {
                    "task": row["task"], "query_id": qid, "kind": "blackbox", "ordinal": index,
                    "label": f"B{index}", "text": f"{row['question']}. {evidence}", "doc_id": "", "retrieval_rank": "",
                }
            )
        for index, text in enumerate(whitebox[qid], start=1):
            records.append(
                {
                    "task": row["task"], "query_id": qid, "kind": "hotflip", "ordinal": index,
                    "label": f"W{index}", "text": text, "doc_id": "", "retrieval_rank": "",
                }
            )
    return records


def add_scores(records: list[dict], vectors: np.ndarray):
    by_query: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        record["vector_l2"] = float(np.linalg.norm(vectors[index]))
        by_query.setdefault(record["query_id"], []).append(index)
    for indices in by_query.values():
        q_index = next(index for index in indices if records[index]["kind"] == "query")
        q_vector = vectors[q_index]
        doc_indices = [index for index in indices if index != q_index]
        dots = {index: float(np.dot(q_vector, vectors[index])) for index in doc_indices}
        order = sorted(doc_indices, key=lambda index: dots[index], reverse=True)
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        q_norm = float(np.linalg.norm(q_vector))
        for index in indices:
            if index == q_index:
                records[index].update({"dot_to_query": "", "cosine_to_query": "", "mixed_rank_15": ""})
                continue
            denom = max(q_norm * float(np.linalg.norm(vectors[index])), 1e-12)
            records[index].update(
                {
                    "dot_to_query": dots[index],
                    "cosine_to_query": float(dots[index] / denom),
                    "mixed_rank_15": ranks[index],
                }
            )


def tsne_coordinates(vectors: np.ndarray, seed: int) -> np.ndarray:
    n_samples, dimension = vectors.shape
    components = min(50, dimension, n_samples - 1)
    reduced = PCA(n_components=components, random_state=seed).fit_transform(vectors)
    perplexity = min(15, max(5, (n_samples - 1) / 3))
    return TSNE(
        n_components=2,
        metric="euclidean",
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        max_iter=2000,
        random_state=seed,
    ).fit_transform(reduced)


def get_font(size: int):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, kind: str, radius: int):
    color = COLORS[kind]
    if kind == "query":
        pts = []
        for step in range(10):
            angle = np.pi / 2 + step * np.pi / 5
            r = radius if step % 2 == 0 else radius * 0.42
            pts.append((x + r * np.cos(angle), y - r * np.sin(angle)))
        draw.polygon(pts, fill=color)
    elif kind == "normal":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif kind == "blackbox":
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)
    else:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_case(records: list[dict], coords: np.ndarray, task: str, query_id: str, output_path: Path):
    chosen = [(record, coords[index]) for index, record in enumerate(records) if record["query_id"] == query_id]
    width, height = 1120, 860
    margin_left, margin_right, margin_top, margin_bottom = 105, 55, 130, 115
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = get_font(26), get_font(17), get_font(14)
    points = np.array([coord for _, coord in chosen], dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    low -= span * 0.18
    high += span * 0.18

    def project(coord):
        x = margin_left + (coord[0] - low[0]) / (high[0] - low[0]) * (width - margin_left - margin_right)
        y = height - margin_bottom - (coord[1] - low[1]) / (high[1] - low[1]) * (height - margin_top - margin_bottom)
        return float(x), float(y)

    question = next(record["text"] for record, _ in chosen if record["kind"] == "query")
    draw.text((margin_left, 24), f"{task.upper()} · {query_id}", fill=COLORS["text"], font=title_font)
    for line_index, line in enumerate(textwrap.wrap(question, width=115)[:2]):
        draw.text((margin_left, 58 + line_index * 21), line, fill=COLORS["text"], font=body_font)
    for grid in np.linspace(0, 1, 5):
        x = margin_left + grid * (width - margin_left - margin_right)
        y = margin_top + grid * (height - margin_top - margin_bottom)
        draw.line((x, margin_top, x, height - margin_bottom), fill=COLORS["grid"], width=1)
        draw.line((margin_left, y, width - margin_right, y), fill=COLORS["grid"], width=1)
    query_xy = next(project(coord) for record, coord in chosen if record["kind"] == "query")
    for record, coord in chosen:
        if record["kind"] != "query":
            draw.line((*query_xy, *project(coord)), fill=COLORS["line"], width=1)
    for record, coord in chosen:
        x, y = project(coord)
        draw_marker(draw, x, y, record["kind"], 10 if record["kind"] == "query" else 8)
        draw.text((x + 10, y - 8), record["label"], fill=COLORS["text"], font=small_font)
    legend = [("Q query", "query"), ("N normal top-5", "normal"), ("B black-box Q+I", "blackbox"), ("W white-box HotFlip", "hotflip")]
    start_x = margin_left
    for label, kind in legend:
        draw_marker(draw, start_x + 8, height - 64, kind, 7)
        draw.text((start_x + 21, height - 73), label, fill=COLORS["text"], font=small_font)
        start_x += 205 if kind != "blackbox" else 228
    draw.text((margin_left, height - 36), "Joint t-SNE fitted across the five sampled queries of this benchmark; axes are arbitrary.", fill=COLORS["text"], font=small_font)
    image.save(output_path)


def make_montage(paths: list[Path], output_path: Path):
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w, thumb_h = 550, 422
    canvas = Image.new("RGB", (thumb_w * 3, thumb_h * 2), (245, 247, 250))
    for index, image in enumerate(images):
        image.thumbnail((thumb_w, thumb_h))
        x = (index % 3) * thumb_w + (thumb_w - image.width) // 2
        y = (index // 3) * thumb_h + (thumb_h - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(output_path)


def write_csv(records: list[dict], output_path: Path):
    fieldnames = [
        "task", "query_id", "kind", "ordinal", "label", "doc_id", "retrieval_rank", "mixed_rank_15",
        "vector_l2", "dot_to_query", "cosine_to_query", "tsne_x", "tsne_y", "text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.pubmed_retrieval, args.medqa_retrieval):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required retrieval file is not ready: {path}")

    pubmed, pubmed_attacks = prepare_task(
        "pubmedqa", args.pubmed_retrieval, "results/adv_targeted_results/mirage_pubmedqa_all_gpt41mini.json",
        args.normal_k, args.samples, args.seed,
    )
    medqa, medqa_attacks = prepare_task(
        "medqa", args.medqa_retrieval, "results/adv_targeted_results/mirage_medqa_all.json",
        args.normal_k, args.samples, args.seed + 1,
    )
    selected = pubmed + medqa
    all_attacks = dict(pubmed_attacks)
    all_attacks.update(medqa_attacks)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer, get_emb = load_models("contriever")
    model.to(device).eval()
    whitebox = generate_hotflip(model, tokenizer, get_emb, all_attacks, selected, out_dir, args.seed)

    manifest = {
        "model_code": "contriever",
        "score_function": "dot",
        "seed": args.seed,
        "normal_k": args.normal_k,
        "pubmedqa_ids": [row["query_id"] for row in pubmed],
        "medqa_ids": [row["query_id"] for row in medqa],
        "pubmed_retrieval": args.pubmed_retrieval,
        "medqa_retrieval": args.medqa_retrieval,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for task, rows in (("pubmedqa", pubmed), ("medqa", medqa)):
        records = make_records(rows, whitebox)
        vectors = encode(model, tokenizer, get_emb, [record["text"] for record in records], device, max_length=128)
        add_scores(records, vectors)
        coords = tsne_coordinates(vectors, args.seed)
        for record, coord in zip(records, coords):
            record["tsne_x"], record["tsne_y"] = float(coord[0]), float(coord[1])
        np.savez_compressed(out_dir / f"{task}_raw_vectors.npz", vectors=vectors)
        write_csv(records, out_dir / f"{task}_points.csv")
        paths = []
        for row in rows:
            path = out_dir / f"{task}_{row['query_id'].replace(':', '_')}.png"
            draw_case(records, coords, task, row["query_id"], path)
            paths.append(path)
        make_montage(paths, out_dir / f"{task}_five_cases.png")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
