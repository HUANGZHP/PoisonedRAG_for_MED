# Purpose: Implements medrag_retriever.py in the PoisonedRAG project.

import json
import logging
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.medrag_corpus import MedCorpus

LOGGER = logging.getLogger(__name__)

try:
    import faiss
except Exception as exc:  # pragma: no cover - import error is raised at runtime usage
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


@dataclass
class _SourceIndex:
    source: str
    index_dir: Path
    index_file: Path
    index: Any
    id_map: Optional[Dict[int, str]]
    metadata_accessor: Optional["_JsonlRowAccessor"]


class _JsonlRowAccessor:
    """
    Memory-efficient random access to jsonl rows via byte offsets.
    Useful when metadata file is very large.
    """

    def __init__(self, jsonl_path: Path):
        self.path = jsonl_path
        self.offsets = array("Q")
        self._fh = None

        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= len(self.offsets):
            return None

        if self._fh is None or self._fh.closed:
            self._fh = self.path.open("rb")

        self._fh.seek(self.offsets[idx])
        line = self._fh.readline()
        if not line:
            return None

        try:
            return json.loads(line.decode("utf-8"))
        except Exception:
            return None

    def close(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __del__(self):  # pragma: no cover - destructor path is non-deterministic
        self.close()


class MedCPTRetriever:
    """
    Load MedCPT FAISS index from prebuilt MedRAG index folders and retrieve with
    ncbi/MedCPT-Query-Encoder.
    """

    def __init__(
        self,
        corpus: Optional[MedCorpus] = None,
        index_base_dir: Optional[str] = None,
        sources: Optional[Sequence[str]] = None,
        query_encoder_name: Optional[str] = None,
        max_length: int = 512,
        device: Optional[str] = None,
        use_mmap: bool = True,
        normalize_query: bool = False,
        topk_per_source_multiplier: int = 4,
        fp16: bool = False,
        use_faiss_gpu: Optional[bool] = None,
        faiss_gpu_device: int = 0,
        faiss_gpu_devices: Optional[Sequence[int]] = None,
        faiss_gpu_use_float16: bool = True,
    ):
        if faiss is None:
            raise ImportError(f"faiss is required but unavailable: {_FAISS_IMPORT_ERROR}")

        if corpus is None and not index_base_dir:
            raise ValueError("index_base_dir is required when no MedCorpus is supplied.")
        if corpus is None and not sources:
            raise ValueError("sources is required when no MedCorpus is supplied.")

        self.corpus = corpus
        self.index_base_dir = (
            Path(index_base_dir)
            if index_base_dir
            else Path(corpus.base_dir)  # type: ignore[union-attr]
        )
        self.sources = (
            [str(source).strip().lower() for source in sources if str(source).strip()]
            if sources is not None
            else list(corpus.sources)  # type: ignore[union-attr]
        )
        self.query_encoder_name = self._resolve_query_encoder_name(query_encoder_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mmap = use_mmap
        self.normalize_query = normalize_query
        self.topk_per_source_multiplier = max(1, int(topk_per_source_multiplier))
        self.faiss_gpu_device = int(faiss_gpu_device)
        self.faiss_gpu_devices = self._resolve_faiss_gpu_devices(faiss_gpu_devices, self.faiss_gpu_device)
        self.faiss_gpu_device = self.faiss_gpu_devices[0]
        self.faiss_gpu_use_float16 = bool(faiss_gpu_use_float16)
        self._gpu_resources: List[Any] = []
        self._partial_source_warned: set[str] = set()
        self._loaded_docs_per_source: Dict[str, int] = (
            {
                src: len(getattr(self.corpus, "_source_doc_ids", {}).get(src, []))
                for src in self.sources
            }
            if self.corpus is not None
            else {}
        )

        faiss_gpu_available = hasattr(faiss, "StandardGpuResources")
        if use_faiss_gpu is None:
            self.use_faiss_gpu = self.device.startswith("cuda") and faiss_gpu_available
        else:
            self.use_faiss_gpu = bool(use_faiss_gpu)

        if self.use_faiss_gpu and not faiss_gpu_available:
            LOGGER.warning("use_faiss_gpu=True but current faiss build has no GPU support; fallback to CPU index search")
            self.use_faiss_gpu = False

        if self.use_faiss_gpu:
            LOGGER.info("FAISS GPU devices configured: %s", self.faiss_gpu_devices)

        self.tokenizer = AutoTokenizer.from_pretrained(self.query_encoder_name)
        self.query_encoder = AutoModel.from_pretrained(self.query_encoder_name)
        self.query_encoder.to(self.device)
        self.query_encoder.eval()
        self.max_length = self._validate_max_length(max_length)
        if fp16 and self.device.startswith("cuda"):
            self.query_encoder.half()

        self.source_indices: Dict[str, _SourceIndex] = {}
        self._load_all_source_indices()

        if not self.source_indices:
            raise FileNotFoundError(
                "No MedCPT index was loaded. Check <base_dir>/<source>/index directories "
                "and ensure FAISS files (.index/.faiss) exist."
            )

    def _resolve_query_encoder_name(self, query_encoder_name: Optional[str]) -> str:
        if query_encoder_name:
            candidate = Path(query_encoder_name)
            if candidate.exists() and candidate.is_dir():
                LOGGER.info("Using local MedCPT query encoder: %s", candidate)
                return str(candidate)
            return query_encoder_name

        env_local = os.environ.get("MEDCPT_QUERY_ENCODER_PATH", "").strip()
        if env_local:
            env_path = Path(env_local)
            if env_path.exists() and env_path.is_dir() and (env_path / "config.json").exists():
                LOGGER.info("Using MEDCPT_QUERY_ENCODER_PATH: %s", env_path)
                return str(env_path)

        workspace_root = Path(__file__).resolve().parent
        local_candidates = [
            workspace_root / "models" / "ncbi" / "MedCPT-Query-Encoder",
            workspace_root / "models" / "medcpt" / "query-encoder",
            workspace_root / "datasets" / "pubmed" / "index" / "ncbi" / "MedCPT-Query-Encoder",
        ]

        hf_root = os.environ.get("HF_MODEL_ROOT", "").strip()
        if hf_root:
            local_candidates.extend(
                [
                    Path(hf_root) / "ncbi" / "MedCPT-Query-Encoder",
                    Path(hf_root) / "MedCPT-Query-Encoder",
                ]
            )

        for path in local_candidates:
            if path.exists() and path.is_dir() and (path / "config.json").exists():
                LOGGER.info("Using discovered local MedCPT query encoder: %s", path)
                return str(path)

        return "ncbi/MedCPT-Query-Encoder"

    def _validate_max_length(self, requested: int) -> int:
        try:
            requested = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid MedCPT max_length={requested!r}") from exc

        if requested <= 0:
            raise ValueError(f"MedCPT max_length must be positive, got {requested}.")

        limits = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            limits.append(tokenizer_limit)
        config_limit = getattr(getattr(self.query_encoder, "config", None), "max_position_embeddings", None)
        if isinstance(config_limit, int) and config_limit > 0:
            limits.append(config_limit)

        supported = min(limits) if limits else 512
        if requested > supported:
            raise ValueError(
                f"MedCPT max_length={requested} exceeds the loaded encoder limit={supported}."
            )
        return requested

    def _resolve_faiss_gpu_devices(
        self,
        faiss_gpu_devices: Optional[Sequence[int]],
        default_device: int,
    ) -> List[int]:
        parsed: List[int] = []

        devices: Optional[Sequence[Any]] = faiss_gpu_devices  # type: ignore[assignment]
        if devices is None:
            env_devices = os.environ.get("FAISS_GPU_DEVICES", "").strip()
            if env_devices:
                devices = [part.strip() for part in env_devices.split(",") if part.strip()]

        if devices is None:
            return [int(default_device)]

        for item in devices:
            try:
                dev = int(item)
            except Exception:
                continue
            if dev < 0 or dev in parsed:
                continue
            parsed.append(dev)

        if not parsed:
            parsed.append(int(default_device))
        return parsed

    def _load_all_source_indices(self):
        for source in self.sources:
            try:
                src_idx = self._load_source_index(source)
            except FileNotFoundError as exc:
                LOGGER.warning("Skip source '%s': %s", source, exc)
                continue
            except Exception as exc:
                LOGGER.warning("Failed loading source index '%s': %s", source, exc)
                continue

            self.source_indices[source] = src_idx
            LOGGER.info("Loaded MedCPT index for %s from %s", source, src_idx.index_file)

    def _load_source_index(self, source: str) -> _SourceIndex:
        source_index_root = self.index_base_dir / source / "index"
        if not source_index_root.exists() or not source_index_root.is_dir():
            raise FileNotFoundError(f"index dir not found at {source_index_root}")

        index_file = self._find_faiss_index_file(source_index_root)
        if index_file is None:
            raise FileNotFoundError(f"no .index/.faiss file under {source_index_root}")

        index_dir = index_file.parent
        index_obj = self._read_faiss_index(index_file)

        id_map = self._load_id_map(index_dir)
        metadata_accessor = self._load_metadata_accessor(index_dir)

        # For unrecognized layouts we keep extension points through these hooks.
        # Add new loaders in _load_id_map/_load_metadata_accessor if new formats appear.
        return _SourceIndex(
            source=source,
            index_dir=index_dir,
            index_file=index_file,
            index=index_obj,
            id_map=id_map,
            metadata_accessor=metadata_accessor,
        )

    def _find_faiss_index_file(self, root: Path) -> Optional[Path]:
        candidates = sorted(
            [
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".index", ".faiss"}
            ]
        )
        if not candidates:
            return None

        def rank(path: Path) -> Tuple[int, str]:
            path_str = str(path).lower()
            priority = 2
            if "medcpt" in path_str:
                priority = 0
            elif "article-encoder" in path_str:
                priority = 1
            return priority, path_str

        candidates.sort(key=rank)
        return candidates[0]

    def _read_faiss_index(self, index_file: Path):
        mmap_flag = getattr(faiss, "IO_FLAG_MMAP", 0)
        if self.use_mmap and mmap_flag:
            cpu_index = faiss.read_index(str(index_file), mmap_flag)
        else:
            cpu_index = faiss.read_index(str(index_file))

        return self._maybe_move_index_to_gpu(cpu_index, index_file)

    def _maybe_move_index_to_gpu(self, cpu_index: Any, index_file: Path):
        if not self.use_faiss_gpu:
            return cpu_index

        if not torch.cuda.is_available():
            LOGGER.warning("FAISS GPU requested but CUDA is unavailable; fallback to CPU index search")
            return cpu_index

        visible_gpu_count = int(torch.cuda.device_count())
        active_gpu_ids = [gpu_id for gpu_id in self.faiss_gpu_devices if 0 <= gpu_id < visible_gpu_count]
        if not active_gpu_ids:
            LOGGER.warning(
                "No valid FAISS GPU devices in %s under visible cuda device_count=%d; fallback to CPU index search",
                self.faiss_gpu_devices,
                visible_gpu_count,
            )
            return cpu_index

        # Avoid expensive OOM attempts for extremely large indices (e.g. PubMed).
        # The on-disk file size is compressed and under-estimates real GPU memory needed.
        # Estimate from index dimensions: ntotal * d * 4 bytes (float32) + 50% overhead.
        try:
            total_mem = sum(int(torch.cuda.get_device_properties(gpu_id).total_memory) for gpu_id in active_gpu_ids)
            est_dim = getattr(cpu_index, "d", 768)
            est_ntotal = int(getattr(cpu_index, "ntotal", 0))
            est_mem_needed = float(est_ntotal) * float(est_dim) * 4.0 * 1.5  # float32 + overhead
            est_mem_ratio = est_mem_needed / float(total_mem) if total_mem > 0 else 99.0
            safety_ratio = 0.70 if len(active_gpu_ids) == 1 else 0.85
            if est_mem_ratio >= safety_ratio:
                LOGGER.warning(
                    "Skip moving FAISS index to GPU: estimated GPU memory needed=%.2f GB "
                    "(ntotal=%d, d=%d) exceeds %.0f%% of available %.2f GB on devices=%s; "
                    "fallback to CPU index search",
                    est_mem_needed / (1024 ** 3),
                    est_ntotal,
                    est_dim,
                    safety_ratio * 100,
                    total_mem / (1024 ** 3),
                    active_gpu_ids,
                )
                return cpu_index
        except Exception as exc:
            LOGGER.warning("Unable to pre-check GPU memory for %s: %s", index_file, exc)

        try:
            if len(active_gpu_ids) == 1:
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = self.faiss_gpu_use_float16
                gpu_index = faiss.index_cpu_to_gpu(res, active_gpu_ids[0], cpu_index, co)
                self._gpu_resources.append(res)
                LOGGER.info(
                    "Moved FAISS index to single GPU device=%d (float16=%s): %s",
                    active_gpu_ids[0],
                    self.faiss_gpu_use_float16,
                    index_file,
                )
                return gpu_index

            if hasattr(faiss, "index_cpu_to_gpu_multiple_py") and hasattr(faiss, "GpuMultipleClonerOptions"):
                resources = [faiss.StandardGpuResources() for _ in active_gpu_ids]
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = self.faiss_gpu_use_float16
                co.shard = True
                gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, cpu_index, co=co, gpus=active_gpu_ids)
                self._gpu_resources.extend(resources)
                LOGGER.info(
                    "Moved FAISS index to multi-GPU shard devices=%s (float16=%s): %s",
                    active_gpu_ids,
                    self.faiss_gpu_use_float16,
                    index_file,
                )
                return gpu_index

            LOGGER.warning(
                "Current FAISS build lacks multi-GPU python API; fallback to single GPU device=%d",
                active_gpu_ids[0],
            )
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.faiss_gpu_use_float16
            gpu_index = faiss.index_cpu_to_gpu(res, active_gpu_ids[0], cpu_index, co)
            self._gpu_resources.append(res)
            return gpu_index
        except Exception as exc:
            LOGGER.warning("Failed to move FAISS index to GPU for %s: %s; fallback to CPU index search", index_file, exc)
            return cpu_index

    def _load_id_map(self, index_dir: Path) -> Optional[Dict[int, str]]:
        candidates = sorted(
            [
                p
                for p in index_dir.glob("*")
                if p.is_file()
                and p.suffix.lower() in {".json", ".npy"}
                and any(x in p.stem.lower() for x in ["id", "doc", "map", "mapping"])
            ]
        )

        for path in candidates:
            try:
                if path.suffix.lower() == ".json":
                    with path.open("r", encoding="utf-8") as f:
                        obj = json.load(f)
                    mapped = self._json_to_id_map(obj)
                else:
                    arr = np.load(path, allow_pickle=True, mmap_mode="r")
                    mapped = {i: str(v) for i, v in enumerate(arr.tolist())}

                if mapped:
                    LOGGER.info("Loaded id map from %s (%d entries)", path, len(mapped))
                    return mapped
            except Exception as exc:
                LOGGER.warning("Failed loading id map %s: %s", path, exc)

        return None

    @staticmethod
    def _json_to_id_map(obj: Any) -> Dict[int, str]:
        if isinstance(obj, list):
            return {i: str(v) for i, v in enumerate(obj)}
        if isinstance(obj, dict):
            out: Dict[int, str] = {}
            for k, v in obj.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out
        return {}

    def _load_metadata_accessor(self, index_dir: Path) -> Optional[_JsonlRowAccessor]:
        # Common MedRAG naming.
        metadata_candidates = [
            index_dir / "metadatas.jsonl",
            index_dir / "metadata.jsonl",
            index_dir / "metadatas.json",
            index_dir / "metadata.json",
        ]

        for path in metadata_candidates:
            if not path.exists() or not path.is_file():
                continue

            if path.suffix.lower() == ".jsonl":
                try:
                    accessor = _JsonlRowAccessor(path)
                    LOGGER.info("Loaded metadata accessor from %s (%d rows)", path, len(accessor))
                    return accessor
                except Exception as exc:
                    LOGGER.warning("Failed loading metadata jsonl %s: %s", path, exc)
                    return None

            # JSON metadata fallback (loads whole object).
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                tmp_jsonl = index_dir / ".tmp_metadata_cache.jsonl"
                with tmp_jsonl.open("w", encoding="utf-8") as fw:
                    if isinstance(data, list):
                        for row in data:
                            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    elif isinstance(data, dict):
                        for _, row in data.items():
                            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                accessor = _JsonlRowAccessor(tmp_jsonl)
                LOGGER.warning("Converted JSON metadata to temporary jsonl cache: %s", tmp_jsonl)
                return accessor
            except Exception as exc:
                LOGGER.warning("Failed loading metadata json %s: %s", path, exc)
                return None

        return None

    def _encode_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.query_encoder(**inputs)
            # MedCPT indexing path in this repo uses CLS pooling; keep query side consistent.
            emb = outputs.last_hidden_state[:, 0, :]

        emb = emb.float().cpu().numpy().astype(np.float32)
        if self.normalize_query:
            faiss.normalize_L2(emb)
        return emb

    def _doc_id_from_metadata(self, source: str, idx: int, meta: Optional[Dict[str, Any]]) -> str:
        if isinstance(meta, dict):
            for key in ["id", "doc_id", "document_id", "uid"]:
                if meta.get(key) is not None:
                    return str(meta[key])

            source_part = meta.get("source", source)
            local_idx = meta.get("index", idx)
            return f"{source_part}_{local_idx}"

        return f"{source}_{idx}"

    def _resolve_doc_id(self, src: _SourceIndex, faiss_row_idx: int) -> str:
        if src.id_map is not None and faiss_row_idx in src.id_map:
            return src.id_map[faiss_row_idx]

        if src.metadata_accessor is not None:
            meta = src.metadata_accessor.get(faiss_row_idx)
            return self._doc_id_from_metadata(src.source, faiss_row_idx, meta)

        # If no mapping file exists, this is our best fallback.
        return f"{src.source}_{faiss_row_idx}"

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        query_emb = self._encode_query(query)
        merged: List[Dict[str, Any]] = []

        for source, src in self.source_indices.items():
            index_ntotal = int(src.index.ntotal)
            if index_ntotal <= 0:
                continue

            loaded_docs = int(self._loaded_docs_per_source.get(source, index_ntotal))
            if (
                loaded_docs > 0
                and loaded_docs < index_ntotal
                and source not in self._partial_source_warned
            ):
                LOGGER.warning(
                    "Source '%s' corpus is partially loaded (%d docs) while FAISS index has %d vectors; "
                    "top hits outside loaded subset will be dropped, which can hurt relevance. "
                    "Increase --max-docs-per-source or remove the cap for accurate results.",
                    source,
                    loaded_docs,
                    index_ntotal,
                )
                self._partial_source_warned.add(source)

            local_k = min(index_ntotal, max(k, k * self.topk_per_source_multiplier))
            scores, indices = src.index.search(query_emb, local_k)

            for rank, row_idx in enumerate(indices[0].tolist()):
                if row_idx < 0:
                    continue

                score = float(scores[0][rank])
                doc_id = self._resolve_doc_id(src, int(row_idx))
                doc = self.corpus.get_document_by_id(doc_id) if self.corpus is not None else None

                if doc is None and self.corpus is not None:
                    # If corpus ID format changed, try source-prefixed fallback.
                    prefixed = f"{source}:{doc_id}"
                    doc = self.corpus.get_document_by_id(prefixed)

                if doc is None and self.corpus is not None:
                    continue

                merged.append(
                    {
                        "text": doc["text"] if doc is not None else "",
                        "score": score,
                        "id": doc["id"] if doc is not None else doc_id,
                        "title": doc.get("title", "") if doc is not None else "",
                        "source": doc.get("source", source) if doc is not None else source,
                    }
                )

        # Deduplicate by doc id and keep highest score.
        best_by_id: Dict[str, Dict[str, Any]] = {}
        for hit in merged:
            prev = best_by_id.get(hit["id"])
            if prev is None or hit["score"] > prev["score"]:
                best_by_id[hit["id"]] = hit

        ranked = sorted(best_by_id.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        # Compatibility alias for legacy code that used retriever.search(query, k).
        return self.retrieve(query, k)
