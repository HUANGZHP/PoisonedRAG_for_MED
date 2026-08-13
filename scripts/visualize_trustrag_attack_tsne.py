#!/usr/bin/env python3
"""Visualize audited corrected TrustRAG attack cases with paired t-SNE plots.

The script deliberately refuses result files without the attack-retention audit
fields.  It therefore does not infer that a missing final attack was filtered:
each displayed success has an own-5 attack text recorded in the reserve pool
and no surviving attack text after the enabled filters.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag_corpus import load_medrag_corpus_subset
from src.utils import load_models


ANSWER_LABELS = ("yes", "no", "maybe")
AUDIT_FIELDS = (
    "own5_adv",
    "pre_filter_adv",
    "post_trustrag_adv",
    "post_medical_cluster_adv",
    "trustrag_removed_adv_count",
    "medical_semantic_clustering_removed_adv_count",
    "final_adv_count",
)

COLORS = {
    "query": (35, 39, 48),
    "normal": (58, 112, 190),
    "not_candidate": (150, 156, 166),
    "removed_trustrag": (43, 148, 100),
    "removed_medical": (126, 87, 194),
    "retained_reserve": (221, 136, 41),
    "retained_final": (202, 66, 61),
    "grid": (216, 221, 228),
    "line": (188, 194, 204),
    "text": (31, 35, 43),
}


@dataclass
class Case:
    category: str
    defense: str
    attack_method: str
    result_path: str
    query_id: str
    question: str
    answer: str
    target_label: str
    prediction: str
    own5_adv: list[str]
    pre_filter_adv: list[str]
    post_trustrag_adv: list[str]
    post_medical_cluster_adv: list[str]
    injected_adv: list[str]
    trustrag_removed_adv_count: int
    medical_removed_adv_count: int
    final_adv_count: int
    normal_ids: list[str] = field(default_factory=list)
    normal_texts: list[str] = field(default_factory=list)


@dataclass
class Point:
    case_index: int
    query_id: str
    category: str
    defense: str
    attack_method: str
    kind: str
    ordinal: int
    label: str
    state: str
    text: str
    doc_id: str = ""
    raw_x: float = 0.0
    raw_y: float = 0.0
    filter_x: float = 0.0
    filter_y: float = 0.0
    raw_l2: float = 0.0
    filter_l2: float = 0.0


def compact(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_label(value: object) -> str:
    text = compact(value).casefold()
    matches = [label for label in ANSWER_LABELS if re.search(rf"(?<![a-z]){label}(?![a-z])", text)]
    return matches[0] if len(matches) == 1 else ""


def load_result_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        rows = payload[0]["iter_0"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected result payload in {path}") from exc
    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError(f"Expected exactly 500 rows in {path}, found {len(rows) if isinstance(rows, list) else type(rows)}")
    if len({str(row.get("id", "")) for row in rows}) != 500:
        raise ValueError(f"Expected 500 unique IDs in {path}")
    missing = [field for field in AUDIT_FIELDS if any(field not in row for row in rows)]
    if missing:
        raise ValueError(
            f"{path} does not contain the required audit fields: {sorted(set(missing))}. "
            "Do not infer filtering from injected_adv alone."
        )
    return rows


def infer_defense(path: Path) -> str:
    return "trustrag_medcluster" if "trustrag_medcluster" in str(path) else "trustrag"


def infer_attack(path: Path) -> str:
    name = path.name
    if name.endswith("_LM_targeted.json"):
        return "LM_targeted"
    if name.endswith("_hotflip.json"):
        return "hotflip"
    return "unknown"


def make_case(row: dict, path: Path, category: str) -> Case:
    answer = parse_label(row.get("answer"))
    prediction = parse_label(row.get("parsed_pred_label"))
    target = parse_label(row.get("target_label"))
    if not answer or not prediction or not target:
        raise ValueError(f"Unparseable label in selected record {row.get('id')} from {path}")
    return Case(
        category=category,
        defense=infer_defense(path),
        attack_method=infer_attack(path),
        result_path=str(path),
        query_id=str(row["id"]),
        question=compact(row.get("question")),
        answer=answer,
        target_label=target,
        prediction=prediction,
        own5_adv=[compact(text) for text in row["own5_adv"]],
        pre_filter_adv=[compact(text) for text in row["pre_filter_adv"]],
        post_trustrag_adv=[compact(text) for text in row["post_trustrag_adv"]],
        post_medical_cluster_adv=[compact(text) for text in row["post_medical_cluster_adv"]],
        injected_adv=[compact(text) for text in row.get("injected_adv", [])],
        trustrag_removed_adv_count=int(row["trustrag_removed_adv_count"]),
        medical_removed_adv_count=int(row["medical_semantic_clustering_removed_adv_count"]),
        final_adv_count=int(row["final_adv_count"]),
    )


def rank_key(case: Case) -> tuple:
    return (
        -(case.trustrag_removed_adv_count + case.medical_removed_adv_count),
        -len(case.pre_filter_adv),
        case.defense,
        case.attack_method,
        case.query_id,
    )


def select_cases(paths: Iterable[Path], per_category: int) -> tuple[list[Case], dict]:
    successful: list[Case] = []
    failed: list[Case] = []
    audit = {"files": {}, "success_candidates": 0, "failure_candidates": 0}
    for path in paths:
        rows = load_result_rows(path)
        file_audit = {"rows": len(rows), "success_candidates": 0, "failure_candidates": 0}
        for row in rows:
            answer = parse_label(row.get("answer"))
            prediction = parse_label(row.get("parsed_pred_label"))
            target = parse_label(row.get("target_label"))
            pre = [compact(text) for text in row["pre_filter_adv"]]
            post = [compact(text) for text in row["post_medical_cluster_adv"]]
            # Strong positive case: an injected own-5 item entered the reserve
            # pool, all of those items were removed by enabled filters, and the
            # final answer is correct rather than the attack target.
            defense_success = bool(pre) and not post and prediction == answer and prediction != target
            # Strong negative case: an attack survived filtering and the model
            # exactly emitted the attack's requested target label.
            defense_failure = bool(pre) and bool(post) and prediction == target
            if defense_success:
                successful.append(make_case(row, path, "filtered_and_correct"))
                file_audit["success_candidates"] += 1
            if defense_failure:
                failed.append(make_case(row, path, "attack_target_hit"))
                file_audit["failure_candidates"] += 1
        audit["files"][str(path)] = file_audit
        audit["success_candidates"] += file_audit["success_candidates"]
        audit["failure_candidates"] += file_audit["failure_candidates"]

    successful.sort(key=rank_key)
    failed.sort(key=rank_key)
    chosen: list[Case] = []
    seen: set[tuple[str, str]] = set()
    for source in (successful, failed):
        category_count = 0
        for case in source:
            identity = (case.category, case.query_id)
            if identity in seen:
                continue
            chosen.append(case)
            seen.add(identity)
            category_count += 1
            if category_count >= per_category:
                break
    audit["selected_success"] = sum(case.category == "filtered_and_correct" for case in chosen)
    audit["selected_failure"] = sum(case.category == "attack_target_hit" for case in chosen)
    return chosen, audit


def load_normal_contexts(cases: list[Case], retrieval_path: Path, normal_k: int) -> None:
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    requested_ids: list[str] = []
    for case in cases:
        ranking = retrieval.get(case.query_id)
        if not isinstance(ranking, dict) or len(ranking) < normal_k:
            raise ValueError(f"Need {normal_k} retrieved normal documents for {case.query_id}")
        case.normal_ids = list(ranking)[:normal_k]
        requested_ids.extend(case.normal_ids)
    corpus = load_medrag_corpus_subset("pubmed", requested_ids)
    for case in cases:
        case.normal_texts = []
        for doc_id in case.normal_ids:
            doc = corpus.get(doc_id)
            if doc is None or not compact(doc.get("text")):
                raise KeyError(f"Missing retrieved PubMed context {doc_id} for {case.query_id}")
            case.normal_texts.append(compact(doc["text"]))


def membership_state(text: str, case: Case) -> str:
    if text not in case.pre_filter_adv:
        return "not_candidate"
    if text not in case.post_trustrag_adv:
        return "removed_trustrag"
    if text not in case.post_medical_cluster_adv:
        return "removed_medical"
    if text in case.injected_adv:
        return "retained_final"
    return "retained_reserve"


def make_points(cases: list[Case]) -> list[Point]:
    points: list[Point] = []
    for case_index, case in enumerate(cases):
        common = dict(
            case_index=case_index,
            query_id=case.query_id,
            category=case.category,
            defense=case.defense,
            attack_method=case.attack_method,
        )
        points.append(Point(**common, kind="query", ordinal=0, label="Q", state="query", text=case.question))
        for ordinal, (doc_id, text) in enumerate(zip(case.normal_ids, case.normal_texts), start=1):
            points.append(
                Point(**common, kind="normal", ordinal=ordinal, label=f"N{ordinal}", state="normal", text=text, doc_id=doc_id)
            )
        for ordinal, text in enumerate(case.own5_adv, start=1):
            points.append(
                Point(
                    **common,
                    kind="attack",
                    ordinal=ordinal,
                    label=f"A{ordinal}",
                    state=membership_state(text, case),
                    text=text,
                )
            )
    return points


def encode_contriever(texts: list[str], device: torch.device, max_length: int) -> np.ndarray:
    model, _, tokenizer, get_emb = load_models("contriever")
    model.to(device).eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            vectors.append(get_emb(model, inputs).detach().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(vectors, dim=0).numpy().astype(np.float32)


def encode_trustrag_cls(texts: list[str], device: torch.device, max_length: int) -> np.ndarray:
    model_path = "/home/HF_Model/facebook/contriever"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            batch = texts[start : start + 8]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            vectors.append(outputs.hidden_states[-1][:, 0, :].detach().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(vectors, dim=0).numpy().astype(np.float32)


def tsne(vectors: np.ndarray, seed: int) -> np.ndarray:
    count, dim = vectors.shape
    if count < 6:
        raise ValueError(f"Need at least six points for t-SNE, got {count}")
    components = min(50, dim, count - 1)
    reduced = PCA(n_components=components, random_state=seed).fit_transform(vectors)
    perplexity = min(20, max(5, (count - 1) / 3))
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


def project(coords: np.ndarray, x0: int, y0: int, width: int, height: int) -> list[tuple[float, float]]:
    low, high = coords.min(axis=0), coords.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    low -= span * 0.15
    high += span * 0.15
    return [
        (
            x0 + (point[0] - low[0]) / (high[0] - low[0]) * width,
            y0 + height - (point[1] - low[1]) / (high[1] - low[1]) * height,
        )
        for point in coords
    ]


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, point: Point, radius: int) -> None:
    color = COLORS[point.state]
    if point.kind == "query":
        vertices = []
        for step in range(10):
            angle = np.pi / 2 + step * np.pi / 5
            distance = radius if step % 2 == 0 else radius * 0.45
            vertices.append((x + distance * np.cos(angle), y - distance * np.sin(angle)))
        draw.polygon(vertices, fill=color)
    elif point.kind == "normal":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif point.state == "not_candidate":
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], outline=color, width=2)
    elif point.state == "removed_trustrag":
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)
    elif point.state == "removed_medical":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    elif point.state == "retained_reserve":
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)
    else:
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255))


def draw_panel(
    draw: ImageDraw.ImageDraw,
    points: list[Point],
    coords: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    label_font,
) -> None:
    draw.text((x0, y0 - 28), title, fill=COLORS["text"], font=label_font)
    for frac in np.linspace(0, 1, 5):
        x = x0 + frac * width
        y = y0 + frac * height
        draw.line((x, y0, x, y0 + height), fill=COLORS["grid"], width=1)
        draw.line((x0, y, x0 + width, y), fill=COLORS["grid"], width=1)
    xy = project(coords, x0, y0, width, height)
    query_xy = xy[0]
    for point, (x, y) in zip(points[1:], xy[1:]):
        draw.line((*query_xy, x, y), fill=COLORS["line"], width=1)
    for point, (x, y) in zip(points, xy):
        draw_marker(draw, x, y, point, 9 if point.kind == "query" else 7)
        draw.text((x + 9, y - 8), point.label, fill=COLORS["text"], font=label_font)


def draw_case(case: Case, points: list[Point], output_path: Path) -> None:
    width, height = 1840, 920
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = get_font(25), get_font(16), get_font(14)
    outcome = "filtered attack + correct answer" if case.category == "filtered_and_correct" else "attack target hit"
    draw.text((70, 24), f"{outcome} · {case.defense} · {case.attack_method} · {case.query_id}", fill=COLORS["text"], font=title_font)
    question = textwrap.wrap(case.question, width=180)[:2]
    for index, line in enumerate(question):
        draw.text((70, 60 + 20 * index), line, fill=COLORS["text"], font=body_font)
    info = (
        f"candidate attacks {len(case.pre_filter_adv)} → TrustRAG {len(case.post_trustrag_adv)} "
        f"→ medical {len(case.post_medical_cluster_adv)} → final top-5 {case.final_adv_count}"
        f"   |   gold={case.answer}, target={case.target_label}, prediction={case.prediction}"
    )
    draw.text((70, 108), info, fill=COLORS["text"], font=body_font)
    case_points = points
    raw = np.asarray([[point.raw_x, point.raw_y] for point in case_points], dtype=np.float32)
    filt = np.asarray([[point.filter_x, point.filter_y] for point in case_points], dtype=np.float32)
    draw_panel(draw, case_points, raw, 70, 180, 780, 580, "Original Contriever embedding t-SNE", small_font)
    draw_panel(draw, case_points, filt, 980, 180, 780, 580, "TrustRAG CLS embedding t-SNE", small_font)
    legend = [
        ("Q query", "query"),
        ("N normal candidate", "normal"),
        ("A not in candidate", "not_candidate"),
        ("A removed by TrustRAG", "removed_trustrag"),
        ("A removed by medical clustering", "removed_medical"),
        ("A retained reserve", "retained_reserve"),
        ("A final top-5", "retained_final"),
    ]
    start_x = 70
    for label, state in legend:
        marker_point = Point(0, "", "", "", "", "attack", 0, "", state, "")
        if state == "query":
            marker_point.kind, marker_point.state = "query", "query"
        elif state == "normal":
            marker_point.kind, marker_point.state = "normal", "normal"
        draw_marker(draw, start_x + 8, 815, marker_point, 7)
        draw.text((start_x + 20, 805), label, fill=COLORS["text"], font=small_font)
        start_x += 215 if state not in {"removed_trustrag", "removed_medical"} else 245
    draw.text((70, 860), "Both t-SNE fits are joint across the selected cases; axes are arbitrary. Labels N1–N10 and A1–A5 are local to this case.", fill=COLORS["text"], font=small_font)
    image.save(output_path)


def make_montage(paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    columns = 2
    thumb_w, thumb_h = 900, 450
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (246, 248, 251))
    for index, image in enumerate(images):
        image.thumbnail((thumb_w, thumb_h))
        x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
        y = (index // columns) * thumb_h + (thumb_h - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(output_path)


def write_csv(points: list[Point], output_path: Path) -> None:
    fields = [field for field in Point.__dataclass_fields__]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(point.__dict__ for point in points)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-json",
        action="append",
        default=[],
        help="Corrected attack result JSON. Repeat for each audited result file.",
    )
    parser.add_argument(
        "--retrieval-json",
        default="results/beir_results/formal_dot_contriever/mirage_pubmedqa_all-contriever-dot.json",
    )
    parser.add_argument("--output-dir", default="results/embedding_viz/trustrag_corrected_attack_tsne")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--normal-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.result_json:
        raise ValueError("Pass one or more audited corrected attack JSON files with --result-json")
    paths = [Path(path) for path in args.result_json]
    for path in paths + [Path(args.retrieval_json)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    cases, audit = select_cases(paths, args.per_category)
    if not cases:
        raise RuntimeError("No strict filtered-and-correct or attack-target-hit cases were found; no plot was fabricated.")
    load_normal_contexts(cases, Path(args.retrieval_json), args.normal_k)
    points = make_points(cases)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    texts = [point.text for point in points]
    raw_vectors = encode_contriever(texts, device, args.max_length)
    filter_vectors = encode_trustrag_cls(texts, device, args.max_length)
    raw_coords = tsne(raw_vectors, args.seed)
    filter_coords = tsne(filter_vectors, args.seed + 1)
    for point, raw, filt, raw_vector, filter_vector in zip(points, raw_coords, filter_coords, raw_vectors, filter_vectors):
        point.raw_x, point.raw_y = float(raw[0]), float(raw[1])
        point.filter_x, point.filter_y = float(filt[0]), float(filt[1])
        point.raw_l2 = float(np.linalg.norm(raw_vector))
        point.filter_l2 = float(np.linalg.norm(filter_vector))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "contriever_raw_vectors.npz", vectors=raw_vectors)
    np.savez_compressed(output_dir / "trustrag_cls_vectors.npz", vectors=filter_vectors)
    write_csv(points, output_dir / "points.csv")
    images = []
    for index, case in enumerate(cases):
        path = output_dir / f"{case.category}_{case.defense}_{case.attack_method}_{case.query_id.replace(':', '_')}.png"
        draw_case(case, [point for point in points if point.case_index == index], path)
        images.append(path)
    make_montage(images, output_dir / "selected_cases_montage.png")
    manifest = {
        "model_code": "contriever",
        "retrieval_score_function": "dot",
        "retrieval_json": args.retrieval_json,
        "seed": args.seed,
        "normal_k": args.normal_k,
        "embedding_max_length": args.max_length,
        "device": str(device),
        "selection_audit": audit,
        "cases": [
            {
                "category": case.category,
                "defense": case.defense,
                "attack_method": case.attack_method,
                "result_path": case.result_path,
                "query_id": case.query_id,
                "gold_label": case.answer,
                "target_label": case.target_label,
                "prediction": case.prediction,
                "pre_filter_adv_count": len(case.pre_filter_adv),
                "post_trustrag_adv_count": len(case.post_trustrag_adv),
                "post_medical_cluster_adv_count": len(case.post_medical_cluster_adv),
                "final_adv_count": case.final_adv_count,
            }
            for case in cases
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
