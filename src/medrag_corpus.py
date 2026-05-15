# Purpose: Implements src/medrag_corpus.py in the PoisonedRAG project.

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

DEFAULT_MEDRAG_DATA_DIR = "/home/huangzhp53/PoisonedRAG/datasets"
DEFAULT_MEDRAG_ROOT = "/home/huangzhp53/MedRAG"
DEFAULT_MEDRAG_DB_DIR = "/home/huangzhp53/PoisonedRAG/datasets"

# Apply project defaults only when users have not configured these variables.
os.environ.setdefault("MEDRAG_DATA_DIR", DEFAULT_MEDRAG_DATA_DIR)
os.environ.setdefault("MEDRAG_ROOT", DEFAULT_MEDRAG_ROOT)
os.environ.setdefault("MEDRAG_DB_DIR", DEFAULT_MEDRAG_DB_DIR)

MEDRAG_DATASETS = {
    "pumbed": "pubmed",
    "pubmed": "pubmed",
    "statpearls": "statpearls",
    "textbooks": "textbooks",
}

_FORMAT_PRIORITY = [".jsonl", ".parquet", ".txt", ".json"]
_DEFAULT_SOURCES = ("pubmed", "statpearls", "textbooks")
LOGGER = logging.getLogger(__name__)


def _normalize_dataset(name: str) -> str:
    name = (name or "").strip().lower()
    if name not in MEDRAG_DATASETS:
        raise ValueError(f"Unsupported MedRAG dataset: {name}")
    return MEDRAG_DATASETS[name]


def _iter_base_dirs(repo_root: Path, medrag_root: Optional[Path] = None) -> Iterable[Path]:
    env_path = os.environ.get("MEDRAG_DATA_DIR")
    if env_path:
        yield Path(env_path)

    if medrag_root is not None:
        yield medrag_root

    yield repo_root / "datasets"
    yield repo_root / "data"
    yield repo_root / "MedRAG"
    yield repo_root / "medrag"


def _iter_dataset_dirs(base: Path, dataset: str) -> Iterable[Path]:
    # If the base path already points to the dataset directory.
    if base.name.lower() == dataset and base.is_dir():
        yield base

    # If the base path is the chunk directory itself.
    if base.name.lower() == "chunk" and base.parent.name.lower() == dataset and base.is_dir():
        yield base.parent

    yield base / dataset
    yield base / "datasets" / dataset
    yield base / "data" / dataset


def _resolve_dataset_dir(dataset: str, repo_root: Path, medrag_root: Optional[Path] = None) -> Optional[Path]:
    dataset = _normalize_dataset(dataset)
    for base in _iter_base_dirs(repo_root, medrag_root):
        if not base.exists():
            continue
        for cand in _iter_dataset_dirs(base, dataset):
            if cand.exists() and cand.is_dir():
                return cand
    return None


def _iter_candidate_files(base: Path, dataset: str) -> Iterable[Path]:
    yield base / "data" / dataset / "corpus.jsonl"
    yield base / "data" / dataset / "corpus.json"
    yield base / "data" / f"{dataset}.jsonl"
    yield base / "data" / f"{dataset}.json"
    yield base / "data" / "corpus" / f"{dataset}.jsonl"
    yield base / "data" / "corpus" / f"{dataset}.json"
    yield base / "corpus" / f"{dataset}.jsonl"
    yield base / "corpus" / f"{dataset}.json"
    yield base / "datasets" / dataset / "corpus.jsonl"
    yield base / "datasets" / dataset / "corpus.json"
    yield base / "datasets" / f"{dataset}.jsonl"
    yield base / "datasets" / f"{dataset}.json"


def _resolve_corpus_path(dataset: str, repo_root: Path, medrag_root: Optional[Path] = None) -> Path:
    dataset = _normalize_dataset(dataset)

    for base in _iter_base_dirs(repo_root, medrag_root):
        if not base.exists():
            continue
        if base.is_file():
            return base

        for cand in _iter_candidate_files(base, dataset):
            if cand.exists() and cand.is_file():
                return cand

        for pat in (f"**/*{dataset}*.jsonl", f"**/*{dataset}*.json"):
            for p in base.rglob(pat):
                if p.is_file() and "chunk" not in p.parts:
                    return p

    raise FileNotFoundError(
        f"Cannot find MedRAG corpus file for dataset={dataset}. "
        f"Try setting MEDRAG_DATA_DIR to the corpus file or MedRAG root dir."
    )


def _detect_chunk_files(chunk_dir: Path) -> Tuple[str, List[Path], Dict[str, List[Path]]]:
    grouped: Dict[str, List[Path]] = {ext: [] for ext in _FORMAT_PRIORITY}

    for p in sorted(chunk_dir.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in grouped:
            grouped[ext].append(p)

    for ext in _FORMAT_PRIORITY:
        if grouped[ext]:
            return ext, grouped[ext], grouped

    return "", [], grouped


def _iter_docs_from_json(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        yield from _expand_json_obj(data)


def _iter_docs_from_txt(path: Path) -> Iterable[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield {"id": path.stem, "title": path.stem, "text": text}


def _iter_docs_from_parquet(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas/pyarrow is required for parquet chunks") from exc

    df = pd.read_parquet(path)
    for row in df.to_dict(orient="records"):
        if isinstance(row, dict):
            yield row


def _iter_docs_from_chunk_file(path: Path, ext: str) -> Iterable[Dict[str, Any]]:
    if ext in [".jsonl", ".json"]:
        yield from _iter_docs_from_json(path)
    elif ext == ".txt":
        yield from _iter_docs_from_txt(path)
    elif ext == ".parquet":
        yield from _iter_docs_from_parquet(path)


def _expand_json_obj(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
            else:
                yield {"text": obj}
    elif isinstance(data, dict):
        if "corpus" in data:
            yield from _expand_json_obj(data["corpus"])
            return
        if "documents" in data:
            yield from _expand_json_obj(data["documents"])
            return
        if "docs" in data:
            yield from _expand_json_obj(data["docs"])
            return
        for k, v in data.items():
            if isinstance(v, dict):
                obj = dict(v)
                obj.setdefault("id", k)
                yield obj
            else:
                yield {"id": k, "text": v}


def _first_non_empty(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in obj and obj[k] is not None and obj[k] != "":
            return obj[k]
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(map(str, value))
    return str(value)


def _extract_doc(obj: Dict[str, Any], fallback_id: int) -> Tuple[str, Dict[str, str]]:
    doc_id = _first_non_empty(obj, ["id", "_id", "doc_id", "document_id", "pmid", "PMID", "uid", "pid"])
    if doc_id is None:
        doc_id = str(fallback_id)

    title = _first_non_empty(obj, ["title", "paper_title", "book_title", "chapter_title", "name"])
    if title is None:
        title = ""

    # For pubmed chunks, 'contents' is often title+body and aligns better with indexing text.
    text = _first_non_empty(obj, ["text", "abstract", "contents", "content", "body", "article", "paragraph"])
    if text is None and "sections" in obj and isinstance(obj["sections"], list):
        text = _normalize_text([s.get("text", s) if isinstance(s, dict) else s for s in obj["sections"]])
    if text is None and "sentences" in obj and isinstance(obj["sentences"], list):
        text = _normalize_text(obj["sentences"])

    return str(doc_id), {"title": _normalize_text(title), "text": _normalize_text(text)}


def _load_from_chunk_dir(dataset: str, chunk_dir: Path) -> Dict[str, Dict[str, str]]:
    ext, files, all_grouped = _detect_chunk_files(chunk_dir)
    if not files:
        raise FileNotFoundError(f"No supported chunk files found in {chunk_dir}")

    available_exts = [k for k, v in all_grouped.items() if v]
    if len(available_exts) > 1:
        logging.info(
            "Chunk dir %s has multiple formats %s; choose %s by priority",
            chunk_dir,
            available_exts,
            ext,
        )

    corpus: Dict[str, Dict[str, str]] = {}
    file_pbar = tqdm(files, desc=f"Loading {dataset} chunks", unit="file")

    for fp in file_pbar:
        try:
            for idx, obj in enumerate(_iter_docs_from_chunk_file(fp, ext)):
                if not isinstance(obj, dict):
                    continue
                doc_id, doc = _extract_doc(obj, idx)
                if doc_id in corpus:
                    # Keep deterministic unique IDs if rare collisions happen across chunk files.
                    base = doc_id
                    n = 1
                    while f"{base}:{n}" in corpus:
                        n += 1
                    doc_id = f"{base}:{n}"
                corpus[doc_id] = doc
        except Exception as exc:
            logging.warning("Failed reading chunk file %s: %s", fp, exc)

    return corpus


def load_medrag_corpus(dataset: str, medrag_root: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Return BEIR-compatible corpus format:
    {doc_id: {"title": "...", "text": "..."}, ...}

    Adapter behavior for current PoisonedRAG datasets format:
    - Preferred: load all shard files under <dataset>/chunk/ (e.g., pubmed23nXXXX.jsonl)
    - Fallback: legacy single corpus file discovery
    """
    dataset = _normalize_dataset(dataset)
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(medrag_root) if medrag_root else None

    dataset_dir = _resolve_dataset_dir(dataset, repo_root, root)
    if dataset_dir is not None:
        chunk_dir = dataset_dir / "chunk"
        if chunk_dir.exists() and chunk_dir.is_dir():
            logging.info("Loading MedRAG corpus shards from %s", chunk_dir)
            corpus = _load_from_chunk_dir(dataset, chunk_dir)
            logging.info("Loaded %d documents for dataset=%s from chunk shards", len(corpus), dataset)
            return corpus

    # Fallback to legacy monolithic corpus file layout.
    path = _resolve_corpus_path(dataset, repo_root, root)
    logging.info("Loading MedRAG corpus from %s", path)

    corpus: Dict[str, Dict[str, str]] = {}
    for idx, obj in enumerate(_iter_docs_from_json(path)):
        doc_id, doc = _extract_doc(obj, idx)
        corpus[doc_id] = doc

    logging.info("Loaded %d documents for dataset=%s from single corpus file", len(corpus), dataset)
    return corpus


class MedCorpus:
    """
    Load pre-chunked MedRAG corpora from
    <base_dir>/<source>/chunk with automatic format detection.

    Output document format:
    {
        "id": str,
        "text": str,
        "title": str,
        "source": str,
    }
    """

    def __init__(
        self,
        base_dir: str,
        sources: Sequence[str] = _DEFAULT_SOURCES,
        max_docs_per_source: Optional[int] = None,
        show_progress: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.sources = [str(s).strip().lower() for s in sources]
        self.max_docs_per_source = max_docs_per_source
        self.show_progress = show_progress

        self.documents: List[Dict[str, str]] = []
        self._doc_by_id: Dict[str, Dict[str, str]] = {}
        self._source_doc_ids: Dict[str, List[str]] = {source: [] for source in self.sources}

        for source in self.sources:
            self._load_source(source)

    def _load_source(self, source_name: str):
        chunk_dir = self.base_dir / source_name / "chunk"
        if not chunk_dir.exists() or not chunk_dir.is_dir():
            LOGGER.warning("Skip source '%s': chunk dir not found at %s", source_name, chunk_dir)
            return

        selected_ext, files, available = _detect_chunk_files(chunk_dir)
        if not files:
            LOGGER.warning("Skip source '%s': no chunk files found in %s", source_name, chunk_dir)
            return

        available_formats = [ext for ext, items in available.items() if items]
        if len(available_formats) > 1:
            LOGGER.info(
                "Source '%s' has multiple formats %s; selecting '%s' by priority rule",
                source_name,
                available_formats,
                selected_ext,
            )

        loaded = 0
        pbar = tqdm(
            files,
            desc=f"Loading {source_name} {selected_ext}",
            disable=not self.show_progress,
            unit="file",
        )

        for file_path in pbar:
            try:
                for _, raw_obj in enumerate(_iter_docs_from_chunk_file(file_path, selected_ext)):
                    if self.max_docs_per_source is not None and loaded >= self.max_docs_per_source:
                        break
                    if not isinstance(raw_obj, dict):
                        continue

                    fallback_id = f"{file_path.stem}_{loaded}"
                    doc_id, doc = _extract_doc(raw_obj, fallback_id)
                    text = (doc.get("text") or "").strip()
                    if not text:
                        continue

                    self._append_document(
                        {
                            "id": str(doc_id),
                            "text": text,
                            "title": (doc.get("title") or "").strip(),
                            "source": source_name,
                        }
                    )
                    loaded += 1
            except Exception as exc:
                LOGGER.warning("Failed reading chunk file %s: %s", file_path, exc)

            if self.max_docs_per_source is not None and loaded >= self.max_docs_per_source:
                LOGGER.info(
                    "Reached max_docs_per_source=%s for '%s'",
                    self.max_docs_per_source,
                    source_name,
                )
                break

        LOGGER.info("Loaded %d docs from source '%s'", loaded, source_name)

    def _append_document(self, doc: Dict[str, str]):
        doc_id = doc["id"]
        if doc_id in self._doc_by_id:
            base_id = f"{doc['source']}:{doc_id}"
            candidate = base_id
            dup_idx = 1
            while candidate in self._doc_by_id:
                dup_idx += 1
                candidate = f"{base_id}:{dup_idx}"
            doc["id"] = candidate
            doc_id = candidate

        self.documents.append(doc)
        self._doc_by_id[doc_id] = doc
        self._source_doc_ids.setdefault(doc["source"], []).append(doc_id)

    def get_texts(self) -> List[str]:
        return [doc["text"] for doc in self.documents]

    def get_metadata(self) -> List[Dict[str, str]]:
        return [dict(doc) for doc in self.documents]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, str]]:
        return self._doc_by_id.get(str(doc_id))

    def get_source_doc_ids(self, source_name: str) -> List[str]:
        return list(self._source_doc_ids.get(source_name, []))

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.documents[idx]
