# Purpose: Implements src/utils.py in the PoisonedRAG project.

import sys, os
import logging
import json
import csv
import re
import importlib.util
from pathlib import Path
from typing import Dict, Tuple, Iterable, Any, Optional

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
except ImportError:
    util = None
    GenericDataLoader = None
from src.medrag_corpus import (
    load_medrag_corpus,
    MEDRAG_DATASETS,
    DEFAULT_MEDRAG_DATA_DIR,
    DEFAULT_MEDRAG_ROOT,
    DEFAULT_MEDRAG_DB_DIR,
)
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from src.contriever_src.contriever import Contriever
except ImportError:
    Contriever = None


def _pick_existing_path(*candidates: str) -> Optional[str]:
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            return p
    return None


HF_MODEL_ROOT = os.environ.get("HF_MODEL_ROOT", "/home/HF_Model")

model_code_to_qmodel_name = {
    "contriever": _pick_existing_path(
        os.path.join(HF_MODEL_ROOT, "facebook", "contriever"),
        os.path.join(HF_MODEL_ROOT, "contriever"),
    ) or "facebook/contriever",
    "contriever-msmarco": _pick_existing_path(
        os.path.join(HF_MODEL_ROOT, "facebook", "contriever-msmarco"),
        os.path.join(HF_MODEL_ROOT, "contriever-msmarco"),
    ) or "facebook/contriever-msmarco",
    "contriever-chinese": _pick_existing_path(
        os.path.join(HF_MODEL_ROOT, "aqweteddy", "contriever-base-chinese"),
        os.path.join(HF_MODEL_ROOT, "contriever-base-chinese"),
        os.path.join(os.path.dirname(__file__), "..", "models", "aqweteddy", "contriever-base-chinese"),
    ) or "aqweteddy/contriever-base-chinese",
    "ance": _pick_existing_path(
        os.path.join(HF_MODEL_ROOT, "sentence-transformers", "msmarco-roberta-base-ance-firstp"),
        os.path.join(HF_MODEL_ROOT, "msmarco-roberta-base-ance-firstp"),
    ) or "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "dpr": "facebook/dpr-question_encoder-single-nq-base",
    "medcpt": _pick_existing_path(
        os.path.join(HF_MODEL_ROOT, "ncbi", "MedCPT-Query-Encoder"),
        os.path.join(HF_MODEL_ROOT, "MedCPT-Query-Encoder"),
    ) or "ncbi/MedCPT-Query-Encoder",
}

model_code_to_cmodel_name = {
    "contriever": model_code_to_qmodel_name["contriever"],
    "contriever-msmarco": model_code_to_qmodel_name["contriever-msmarco"],
    "contriever-chinese": model_code_to_qmodel_name["contriever-chinese"],
    "ance": model_code_to_qmodel_name["ance"],
    "dpr": "facebook/dpr-ctx_encoder-single-nq-base",
    "medcpt": _pick_existing_path(
        os.path.join(HF_MODEL_ROOT, "ncbi", "MedCPT-Article-Encoder"),
        os.path.join(HF_MODEL_ROOT, "MedCPT-Article-Encoder"),
    ) or "ncbi/MedCPT-Article-Encoder",
    "bm25": "bm25"
}

# 预存本地 Index 路径，供检索器实例化时读取
LOCAL_INDEX_PATHS = {
    "pumbed_bm25": "/home/Dataset/PoisonedRAG/datasets/pubmed/index/bm25",
    "pumbed_medcpt": "/home/Dataset/PoisonedRAG/datasets/pubmed/index/ncbi/MedCPT-Article-Encoder",
    "pubmed_bm25": "/home/Dataset/PoisonedRAG/datasets/pubmed/index/bm25",
    "pubmed_medcpt": "/home/Dataset/PoisonedRAG/datasets/pubmed/index/ncbi/MedCPT-Article-Encoder",
    "statpearls_bm25": "/home/Dataset/PoisonedRAG/datasets/statpearls/index/bm25",
    "statpearls_medcpt": "/home/Dataset/PoisonedRAG/datasets/statpearls/index/ncbi/MedCPT-Article-Encoder",
    "textbooks_bm25": "/home/Dataset/PoisonedRAG/datasets/textbooks/index/bm25",
    "textbooks_medcpt": "/home/Dataset/PoisonedRAG/datasets/textbooks/index/ncbi/MedCPT-Article-Encoder",
}

def contriever_get_emb(model, input):
    return model(**input)

def contriever_chinese_get_emb(model, input):
    """Average pooling for Chinese Contriever (BertRetriver with pooling='avg')."""
    output = model(**input)
    last_hidden = output.last_hidden_state
    attention_mask = input["attention_mask"]
    last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
    emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return emb

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]


def medcpt_get_emb(model, input):
    output = model(**input)
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    # Fallback for models without pooler: use [CLS] token representation.
    return output.last_hidden_state[:, 0]


def get_local_index_path(dataset: str, model_code: str) -> Optional[str]:
    dataset_norm = MEDRAG_DATASETS.get((dataset or "").lower(), (dataset or "").lower())
    key = f"{dataset_norm}_{model_code}"
    if key in LOCAL_INDEX_PATHS and os.path.exists(LOCAL_INDEX_PATHS[key]):
        return LOCAL_INDEX_PATHS[key]

    env_key = f"POISONEDRAG_{dataset_norm.upper()}_{model_code.upper()}_INDEX"
    env_path = os.environ.get(env_key)
    if env_path and os.path.exists(env_path):
        return env_path

    default = Path(__file__).resolve().parents[1] / "datasets" / dataset_norm / "index"
    if model_code == "bm25":
        candidate = default / "bm25"
    else:
        candidate = default / "ncbi" / "MedCPT-Article-Encoder"
    if candidate.exists():
        return str(candidate)

    return None


_MEDRAG_UTILS_MODULE = None


def _load_medrag_utils_module():
    global _MEDRAG_UTILS_MODULE
    if _MEDRAG_UTILS_MODULE is not None:
        return _MEDRAG_UTILS_MODULE

    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    env_root = os.environ.get("MEDRAG_ROOT", DEFAULT_MEDRAG_ROOT)
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        repo_root.parent / "MedRAG",
        repo_root / "MedRAG",
        Path.home() / "MedRAG",
    ])

    for root in candidates:
        utils_path = root / "src" / "utils.py"
        if not utils_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("medrag_external_utils", str(utils_path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logging.warning("Failed to import MedRAG utils from %s: %s", utils_path, exc)
            continue
        _MEDRAG_UTILS_MODULE = module
        return module

    return None


class _LocalBM25RetrievalSystem:
    def __init__(self, index_dir: str):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_dir)
        self.chunk_lookup = _ChunkLookup.from_index_dir(index_dir)

    def retrieve(self, question: str, k: int = 32, id_only: bool = True):
        hits = self.searcher.search(question, k=k)
        docs = []
        for h in hits:
            doc_id = str(h.docid)
            try:
                doc = self.searcher.doc(h.docid)
                if doc is not None and getattr(doc, "raw", None):
                    raw_obj = json.loads(doc.raw())
                    if isinstance(raw_obj, dict) and raw_obj.get("id"):
                        doc_id = str(raw_obj["id"])
            except Exception:
                pass

            # If index uses source_index style id, map back to original chunk id when possible.
            if self.chunk_lookup is not None and "_" in doc_id:
                head, tail = doc_id.rsplit("_", 1)
                if tail.isdigit():
                    mapped = self.chunk_lookup.get_doc_id(head, int(tail))
                    if mapped:
                        doc_id = mapped

            docs.append({"id": doc_id})
        scores = [float(h.score) for h in hits]
        return docs, scores


class _LocalMedCPTRetrievalSystem:
    def __init__(self, index_dir: str, query_encoder_name: str):
        import faiss
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Transformer, Pooling

        self.index = faiss.read_index(str(Path(index_dir) / "faiss.index"))
        self._gpu_res = None
        if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
            try:
                self._gpu_res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self.index, co)
                logging.info("Local MedCPT FAISS index moved to GPU (device=0, float16=True)")
            except Exception as exc:
                logging.warning("Failed to move local MedCPT FAISS index to GPU, fallback CPU: %s", exc)
        metadata_path = Path(index_dir) / "metadatas.jsonl"
        self.metadatas = [
            json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.chunk_lookup = _ChunkLookup.from_index_dir(index_dir)

        transformer = Transformer(query_encoder_name)
        pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(modules=[transformer, pooling], device=device)
        self.encoder.eval()

    def retrieve(self, question: str, k: int = 32, id_only: bool = True):
        emb = self.encoder.encode([question], show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        scores, indices = self.index.search(emb, k=k)

        docs = []
        out_scores = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            doc_id = f"{meta['source']}_{meta['index']}"
            if self.chunk_lookup is not None:
                mapped = self.chunk_lookup.get_doc_id(str(meta["source"]), int(meta["index"]))
                if mapped:
                    doc_id = mapped
            docs.append({"id": doc_id})
            out_scores.append(float(scores[0][rank]))
        return docs, out_scores


class _ChunkLookup:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        self._cache: Dict[str, list] = {}

    @staticmethod
    def from_index_dir(index_dir: str):
        idx_path = Path(index_dir)
        for parent in [idx_path] + list(idx_path.parents):
            candidate = parent / "chunk"
            if candidate.exists() and candidate.is_dir():
                return _ChunkLookup(candidate)
        return None

    def _load_source(self, source: str):
        if source in self._cache:
            return self._cache[source]
        fpath = self.chunk_dir / f"{source}.jsonl"
        if not fpath.exists():
            self._cache[source] = []
            return self._cache[source]
        rows = []
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({})
        self._cache[source] = rows
        return rows

    def get_doc_id(self, source: str, index: int) -> Optional[str]:
        rows = self._load_source(source)
        if index < 0 or index >= len(rows):
            return None
        row = rows[index]
        if isinstance(row, dict) and row.get("id"):
            return str(row["id"])
        return None


def get_medrag_retrieval_system(model_code: str, dataset: str):
    if model_code not in ["bm25", "medcpt"]:
        return None

    module = _load_medrag_utils_module()
    if module is None:
        logging.warning("MedRAG utils.py not found. Set MEDRAG_ROOT to enable local-index retrieval.")

    retriever_name_map = {
        "bm25": "BM25",
        "medcpt": "MedCPT",
    }
    corpus_name_map = {
        "pubmed": "PubMed",
        "statpearls": "StatPearls",
        "textbooks": "Textbooks",
    }
    dataset_norm = MEDRAG_DATASETS.get((dataset or "").lower(), (dataset or "").lower())
    if dataset_norm not in corpus_name_map:
        return None

    # Prefer PoisonedRAG local datasets directory if it has MedRAG-style corpus chunks.
    repo_root = Path(__file__).resolve().parents[1]
    db_candidates = []
    env_db_dir = os.environ.get("MEDRAG_DB_DIR", DEFAULT_MEDRAG_DB_DIR)
    if env_db_dir:
        db_candidates.append(Path(env_db_dir))
    db_candidates.append(repo_root / "datasets")

    if module is not None:
        for db_dir in db_candidates:
            chunk_dir = db_dir / dataset_norm / "chunk"
            if not chunk_dir.exists():
                continue
            try:
                return module.RetrievalSystem(
                    retriever_name=retriever_name_map[model_code],
                    corpus_name=corpus_name_map[dataset_norm],
                    db_dir=str(db_dir),
                    HNSW=False,
                    cache=False,
                )
            except Exception as exc:
                logging.warning("Failed to init MedRAG RetrievalSystem at %s: %s", db_dir, exc)
                break

    # Fallback: use local index directly (no chunk dir required).
    local_index = get_local_index_path(dataset_norm, model_code)
    if local_index:
        try:
            if model_code == "bm25":
                return _LocalBM25RetrievalSystem(local_index)
            return _LocalMedCPTRetrievalSystem(local_index, model_code_to_qmodel_name["medcpt"])
        except Exception as exc:
            logging.warning("Failed to init local %s index at %s: %s", model_code, local_index, exc)

    logging.warning("No MedRAG chunk directory or local index found for dataset=%s, model=%s", dataset_norm, model_code)
    return None

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if model_code == "contriever-chinese":
        # Chinese Contriever: use AutoModel (BertRetriver) + custom avg pooling
        model_name = model_code_to_qmodel_name[model_code]
        model = AutoModel.from_pretrained(model_name)
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        get_emb = contriever_chinese_get_emb
    elif 'contriever' in model_code:
        if Contriever is None:
            raise ImportError("Contriever dependencies are missing. Please install required packages.")
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for ANCE models. Please install it.")
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    elif 'medcpt' in model_code:
        model = AutoModel.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = AutoModel.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = medcpt_get_emb
    elif 'dpr' in model_code:
        model = AutoModel.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = AutoModel.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb

def _is_medrag_dataset(name: str) -> bool:
    return (name or "").strip().lower() in MEDRAG_DATASETS

def _normalize_medrag_dataset(name: str) -> str:
    name = (name or "").strip().lower()
    return MEDRAG_DATASETS.get(name, name)

def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _candidate_dataset_dirs(dataset: str, repo_root: Path, medrag_root: Optional[str] = None) -> Iterable[Path]:
    dataset_norm = _normalize_medrag_dataset(dataset)

    env_root = os.environ.get("MEDRAG_DATA_DIR", DEFAULT_MEDRAG_DATA_DIR)
    if env_root:
        p = Path(env_root)
        yield p if p.is_dir() else p.parent

    if medrag_root:
        p = Path(medrag_root)
        yield p if p.is_dir() else p.parent
        yield p / "data" / dataset_norm
        yield p / dataset_norm

    yield repo_root / "datasets" / dataset
    yield repo_root / "datasets" / dataset_norm
    yield repo_root / "data" / dataset
    yield repo_root / "data" / dataset_norm
    yield repo_root / "MedRAG" / "data" / dataset_norm
    yield repo_root / "MedRAG" / "data" / dataset
    yield repo_root / "medrag" / "data" / dataset_norm
    yield repo_root / "medrag" / "data" / dataset

def _candidate_query_paths(root: Path, split: str) -> Iterable[Path]:
    yield root / "queries.jsonl"
    yield root / "queries.json"
    yield root / "queries" / f"{split}.jsonl"
    yield root / "queries" / f"{split}.json"
    yield root / f"{split}_queries.jsonl"
    yield root / f"{split}_queries.json"

def _candidate_qrels_paths(root: Path, split: str) -> Iterable[Path]:
    yield root / "qrels" / f"{split}.tsv"
    yield root / "qrels" / f"{split}.jsonl"
    yield root / "qrels" / f"{split}.json"
    yield root / f"qrels_{split}.tsv"
    yield root / f"qrels_{split}.jsonl"
    yield root / f"qrels_{split}.json"

def _extract_query(obj: Dict[str, Any], fallback_id: int) -> Tuple[str, str]:
    qid = obj.get("_id") or obj.get("id") or obj.get("qid") or obj.get("query_id") or obj.get("question_id")
    if qid is None:
        qid = str(fallback_id)
    text = obj.get("text") or obj.get("query") or obj.get("question") or obj.get("title")
    if text is None:
        text = ""
    return str(qid), str(text)

def _load_queries(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    qid, text = _extract_query(obj, idx)
                    queries[qid] = text
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "queries" in data:
            data = data["queries"]
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    qid, text = _extract_query(v, k)
                    queries[qid] = text
                else:
                    queries[str(k)] = str(v)
        elif isinstance(data, list):
            for idx, obj in enumerate(data):
                if isinstance(obj, dict):
                    qid, text = _extract_query(obj, idx)
                    queries[qid] = text
                else:
                    queries[str(idx)] = str(obj)
    return queries

def _insert_qrel(qrels: Dict[str, Dict[str, float]], qid: Any, docid: Any, score: Any) -> None:
    if qid is None or docid is None:
        return
    try:
        score_val = float(score) if score is not None else 1.0
    except (TypeError, ValueError):
        score_val = 1.0
    qid = str(qid)
    docid = str(docid)
    qrels.setdefault(qid, {})[docid] = score_val

def _load_qrels_json(data: Any) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    if isinstance(data, dict) and "qrels" in data:
        data = data["qrels"]
    if isinstance(data, dict) and data:
        if all(isinstance(v, dict) for v in data.values()):
            for qid, docs in data.items():
                for docid, score in docs.items():
                    _insert_qrel(qrels, qid, docid, score)
            return qrels
        if all(isinstance(v, list) for v in data.values()):
            for qid, docs in data.items():
                for docid in docs:
                    _insert_qrel(qrels, qid, docid, 1.0)
            return qrels
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                qid = obj.get("query-id") or obj.get("query_id") or obj.get("qid")
                docid = obj.get("corpus-id") or obj.get("doc_id") or obj.get("docid") or obj.get("corpus_id")
                score = obj.get("score") or obj.get("relevance") or obj.get("rel")
                _insert_qrel(qrels, qid, docid, score)
    return qrels

def _load_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    if path.suffix.lower() in [".json", ".jsonl"]:
        if path.suffix.lower() == ".jsonl":
            qrels: Dict[str, Dict[str, float]] = {}
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    partial = _load_qrels_json(obj)
                    for qid, docs in partial.items():
                        qrels.setdefault(qid, {}).update(docs)
            return qrels
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _load_qrels_json(data)

    qrels: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        first = next(reader, None)
        if not first:
            return qrels
        header = [c.lower() for c in first]
        has_header = any(h in {"query-id", "query_id", "qid"} for h in header)
        if has_header:
            col_q = header.index("query-id") if "query-id" in header else header.index("query_id") if "query_id" in header else header.index("qid")
            col_d = header.index("corpus-id") if "corpus-id" in header else header.index("corpus_id") if "corpus_id" in header else header.index("docid") if "docid" in header else header.index("doc_id")
            col_s = header.index("score") if "score" in header else None
            for row in reader:
                if len(row) <= max(col_q, col_d):
                    continue
                score = row[col_s] if col_s is not None and col_s < len(row) else 1.0
                _insert_qrel(qrels, row[col_q], row[col_d], score)
        else:
            rows = [first] + list(reader)
            for row in rows:
                if len(row) < 2:
                    continue
                score = row[2] if len(row) > 2 else 1.0
                _insert_qrel(qrels, row[0], row[1], score)
    return qrels

def _resolve_medrag_query_qrels_paths(dataset: str, split: str, medrag_root: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
    repo_root = Path(__file__).resolve().parents[1]
    for root in _candidate_dataset_dirs(dataset, repo_root, medrag_root):
        if not root.exists():
            continue
        if root.is_file():
            root = root.parent
        q_path = _find_first_existing(_candidate_query_paths(root, split))
        r_path = _find_first_existing(_candidate_qrels_paths(root, split))
        if q_path or r_path:
            return q_path, r_path
    return None, None

def load_medrag_datasets(
    dataset: str,
    split: str = "test",
    require_queries: bool = True,
    require_qrels: bool = True,
    medrag_root: Optional[str] = None,
):
    corpus = load_medrag_corpus(dataset, medrag_root=medrag_root)
    q_path, r_path = _resolve_medrag_query_qrels_paths(dataset, split, medrag_root)

    queries = _load_queries(q_path) if q_path else {}
    qrels = _load_qrels(r_path) if r_path else {}

    if require_queries and not queries:
        raise FileNotFoundError(f"Queries not found for MedRAG dataset={dataset}, split={split}.")
    if require_qrels and not qrels:
        raise FileNotFoundError(f"Qrels not found for MedRAG dataset={dataset}, split={split}.")

    return corpus, queries, qrels

def load_beir_datasets(
    dataset: str,
    split: str = "test",
    require_queries: bool = True,
    require_qrels: bool = True,
):
    if _is_medrag_dataset(dataset):
        return load_medrag_datasets(
            dataset=dataset,
            split=split,
            require_queries=require_queries,
            require_qrels=require_qrels,
        )

    if dataset == "msmarco" and split == "test":
        split = "train"
    if util is None or GenericDataLoader is None:
        raise ImportError(
            "beir is required for non-MedRAG datasets. Please install beir: pip install beir"
        )
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

    if require_queries and not queries:
        raise FileNotFoundError(f"Queries not found for BEIR dataset={dataset}, split={split}.")
    if require_qrels and not qrels:
        raise FileNotFoundError(f"Qrels not found for BEIR dataset={dataset}, split={split}.")

    return corpus, queries, qrels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, ensure_ascii=False, indent=4)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def normalize_for_match(s):
    """Normalize free-form text before loose containment matching."""
    text = clean_str(s)
    if not text:
        return ""

    text = text.replace("`", " ")
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<\|[^\n]*?\|>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def contains_normalized_target(target, output):
    """Loose match on normalized token sequences."""
    norm_target = normalize_for_match(target)
    norm_output = normalize_for_match(output)
    if not norm_target or not norm_output:
        return False
    return norm_target in norm_output

def extract_binary_label(s):
    """Extract a strict yes/no label from model output.

    Returns:
        "yes" | "no" | None
    """
    text = clean_str(s)
    if not text:
        return None

    # Remove common wrappers and reasoning blocks before parsing label token.
    text = text.replace("`", " ").strip()
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<\|[^\n]*?\|>', ' ', text)
    text = re.sub(r'^[\s"\'\[\(\{>*#:\-]+', '', text)
    text = text.strip()

    m = re.match(r'^(yes|no)\b', text)
    if m:
        return m.group(1)

    # Some models put the final short answer at the end after reasoning.
    m = re.search(r'\b(yes|no)\b[\s\.!?"\']*$', text)
    if m:
        return m.group(1)

    return None


def extract_choice_label(s, valid_choices=None):
    """Extract a single multiple-choice label (e.g., A/B/C/D)."""
    text = clean_str(s)
    if not text:
        return None

    if valid_choices is None:
        valid_choices = ["a", "b", "c", "d"]
    valid_set = {str(c).strip().lower() for c in valid_choices if str(c).strip()}
    if not valid_set:
        return None

    text = text.replace("`", " ").strip()
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<\|[^\n]*?\|>', ' ', text)
    text = re.sub(r'^[\s"\'\[\(\{>*#:\-]+', '', text)
    text = text.strip()

    m = re.match(r'^([a-z])\b', text)
    if m and m.group(1) in valid_set:
        return m.group(1).upper()

    m = re.search(r'\b([a-z])\b[\s\.!?"\']*$', text)
    if m and m.group(1) in valid_set:
        return m.group(1).upper()

    return None

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores
