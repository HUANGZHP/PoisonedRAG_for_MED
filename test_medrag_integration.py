# Purpose: Implements test_medrag_integration.py in the PoisonedRAG project.

import argparse
import logging
import os

from src.medrag_corpus import MedCorpus
from medrag_retriever import MedCPTRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for MedRAG corpus + MedCPT retriever integration")
    parser.add_argument("--base-dir", type=str, default="/home/huangzhp53/PoisonedRAG/datasets")
    parser.add_argument("--sources", type=str, default="textbooks", help="Comma-separated sources")
    parser.add_argument("--query", type=str, default="What is the function of the kidney?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--medcpt-query-encoder-path", type=str, default="")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU ids for multi-GPU FAISS")
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars while loading corpus",
    )
    return parser.parse_args()


def parse_gpu_ids(raw_gpu_ids: str):
    gpu_ids = []
    for part in (raw_gpu_ids or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError:
            continue
        if gpu_id < 0 or gpu_id in gpu_ids:
            continue
        gpu_ids.append(gpu_id)
    return gpu_ids


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    if requested_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in requested_gpu_ids)
        faiss_gpu_devices = list(range(len(requested_gpu_ids)))
        logging.info("Using multi-GPU with CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        faiss_gpu_devices = [0]
        logging.info("Using single GPU with CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    corpus = MedCorpus(
        base_dir=args.base_dir,
        sources=sources,
        max_docs_per_source=args.max_docs_per_source,
        show_progress=args.show_progress,
    )

    print(f"Loaded documents: {len(corpus)}")
    if len(corpus) == 0:
        raise RuntimeError("Corpus is empty. Please verify chunk directories and source names.")

    medcpt_query_encoder_path = args.medcpt_query_encoder_path.strip() or None
    retriever = MedCPTRetriever(
        corpus=corpus,
        index_base_dir=args.base_dir,
        query_encoder_name=medcpt_query_encoder_path,
        faiss_gpu_devices=faiss_gpu_devices,
    )
    hits = retriever.retrieve(args.query, k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Top-{args.top_k} hits: {len(hits)}")
    for i, hit in enumerate(hits, start=1):
        preview = hit["text"][:160].replace("\n", " ")
        print(
            f"[{i}] id={hit['id']} source={hit['source']} score={hit['score']:.4f} "
            f"title={hit.get('title', '')} text={preview}"
        )


if __name__ == "__main__":
    main()
