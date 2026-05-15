# Purpose: Implements evaluate_beir.py in the PoisonedRAG project.

import logging
import os
import json
import torch
import transformers
from typing import Dict
from tqdm import tqdm

from beir import LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from src.contriever_src.contriever import Contriever
from src.contriever_src.beir_utils import DenseEncoderModel
from src.utils import load_beir_datasets, get_medrag_retrieval_system
from src.medrag_corpus import MedCorpus
from medrag_retriever import MedCPTRetriever

import argparse
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--model_code', type=str, default="contriever", choices=["contriever", "dpr", "ance", "bm25", "medcpt"])
parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--dataset', type=str, default="pubmed", choices=["pubmed", "statpearls", "textbooks"], help='MedRAG dataset to evaluate')
parser.add_argument('--split', type=str, default='test')

parser.add_argument('--result_output', default="results/beir_results/debug.json", type=str)

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--gpu_ids', type=str, default='', help='Comma-separated GPU ids for multi-GPU FAISS (e.g. "0,1,2,3")')
parser.add_argument('--use_faiss_gpu', action=argparse.BooleanOptionalAction, default=False, help='Move MedCPT FAISS index to GPU when available')
parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--medcpt_query_encoder_path', type=str, default='', help='Local path for MedCPT query encoder')
parser.add_argument('--show-progress', action=argparse.BooleanOptionalAction, default=True, help='Show progress bars for corpus/query loops')
parser.add_argument('--query', type=str, default='', help='Single query text used when dataset has no built-in queries')
parser.add_argument('--queries-json', type=str, default='', help='Path to custom queries json when dataset has no built-in queries')
parser.add_argument('--prefer-mirage-queries', action=argparse.BooleanOptionalAction, default=True, help='Prefer loading queries from MIRAGE/benchmark.json')
parser.add_argument('--mirage-benchmark-path', type=str, default=os.path.join(os.getcwd(), 'MIRAGE', 'benchmark.json'), help='Path to MIRAGE benchmark.json')
parser.add_argument('--mirage-dataset', type=str, default='auto', help='MIRAGE subset name (auto/pubmedqa/medqa/medmcqa/mmlu/bioasq/all)')

args = parser.parse_args()

from src.utils import model_code_to_cmodel_name, model_code_to_qmodel_name


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


def _normalize_query_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _load_queries_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: Dict[str, str] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                q = _normalize_query_text(v.get("query") or v.get("text") or v.get("question"))
            else:
                q = _normalize_query_text(v)
            if q:
                queries[str(k)] = q
        return queries

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                qid = _normalize_query_text(item.get("id")) or f"q{i}"
                q = _normalize_query_text(item.get("query") or item.get("text") or item.get("question"))
            else:
                qid = f"q{i}"
                q = _normalize_query_text(item)
            if q:
                queries[str(qid)] = q
        return queries

    raise ValueError(f"Unsupported queries json format in {path}")


def _build_fallback_queries(args) -> Dict[str, str]:
    single_query = _normalize_query_text(args.query)
    if single_query:
        return {"q0": single_query}

    query_json_path = _normalize_query_text(args.queries_json)
    if query_json_path:
        queries = _load_queries_from_json(query_json_path)
        if queries:
            return queries
        raise ValueError(f"No valid queries found in --queries-json file: {query_json_path}")

    raise FileNotFoundError(
        "Dataset has no built-in queries. Please provide --query \"...\" or --queries-json /path/to/queries.json."
    )


def _default_mirage_dataset_for(dataset: str) -> str:
    ds = (dataset or "").strip().lower()
    if ds == "pubmed":
        return "pubmedqa"
    return "all"


def _load_queries_from_mirage(benchmark_path: str, mirage_dataset: str, eval_dataset: str) -> Dict[str, str]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if not isinstance(benchmark, dict) or not benchmark:
        raise ValueError(f"Invalid MIRAGE benchmark file: {benchmark_path}")

    selected = (mirage_dataset or "auto").strip().lower()
    if selected == "auto":
        selected = _default_mirage_dataset_for(eval_dataset)

    available = {str(k).lower(): k for k in benchmark.keys()}

    if selected == "all":
        selected_keys = [available[k] for k in sorted(available.keys())]
    else:
        if selected not in available:
            raise ValueError(
                f"MIRAGE dataset '{selected}' not found in {benchmark_path}. Available: {sorted(available.keys())}"
            )
        selected_keys = [available[selected]]

    queries: Dict[str, str] = {}
    for subset in selected_keys:
        subset_obj = benchmark.get(subset, {})
        if not isinstance(subset_obj, dict):
            continue
        for qid, item in subset_obj.items():
            if isinstance(item, dict):
                text = _normalize_query_text(item.get("question") or item.get("query") or item.get("text"))
            else:
                text = _normalize_query_text(item)
            if not text:
                continue
            queries[f"{subset}:{qid}"] = text

    return queries


def resolve_dpr_class():
    try:
        from beir.retrieval.models import DPR
        return DPR
    except Exception:
        pass
    try:
        from beir.retrieval.models.dpr import DPR
        return DPR
    except Exception:
        return None


def resolve_sentencebert_class():
    try:
        from beir.retrieval.models import SentenceBERT
        return SentenceBERT
    except Exception:
        pass
    try:
        from beir.retrieval.models.sentence_bert import SentenceBERT
        return SentenceBERT
    except Exception:
        return None


def resolve_bm25_class():
    try:
        from beir.retrieval.search.lexical import BM25Search as BM25
        return BM25
    except Exception:
        return None


def retrieve_with_medrag_system(corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], medrag_system, top_k: int, show_progress: bool = True):
    results = {}
    query_iter = queries.items()
    if show_progress:
        query_iter = tqdm(query_iter, total=len(queries), desc="Retrieving", unit="query")
    for query_id, query_text in query_iter:
        docs, scores = medrag_system.retrieve(query_text, k=max(top_k, 2000), id_only=True)
        query_results = {}
        for doc, score in zip(docs, scores):
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                continue
            # Keep ids that exist in PoisonedRAG corpus to avoid downstream KeyError.
            if doc_id in corpus:
                query_results[doc_id] = float(score)
        results[str(query_id)] = query_results
    return results


def retrieve_with_medcpt_retriever(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    retriever: MedCPTRetriever,
    top_k: int,
    show_progress: bool = True,
):
    results = {}
    query_iter = queries.items()
    if show_progress:
        query_iter = tqdm(query_iter, total=len(queries), desc="Retrieving", unit="query")
    for query_id, query_text in query_iter:
        hits = retriever.retrieve(query_text, k=max(top_k, 2000))
        query_results = {}
        for hit in hits:
            doc_id = str(hit.get("id", ""))
            if doc_id and doc_id in corpus:
                query_results[doc_id] = float(hit.get("score", 0.0))
        results[str(query_id)] = query_results
    return results

def compress(results):
    if not results:
        return {}
    k_old = 0
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

logging.info(args)

visible_faiss_gpu_ids = None
requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
if requested_gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in requested_gpu_ids)
    visible_faiss_gpu_ids = list(range(len(requested_gpu_ids)))
    logging.info("Using multi-GPU with CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    visible_faiss_gpu_ids = [0]
    logging.info("Using single GPU with CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


logging.info("Loading dataset...")
corpus, _, _ = load_beir_datasets(args.dataset, args.split, require_queries=False, require_qrels=False)
queries = {}

if args.prefer_mirage_queries:
    try:
        mirage_queries = _load_queries_from_mirage(
            benchmark_path=args.mirage_benchmark_path,
            mirage_dataset=args.mirage_dataset,
            eval_dataset=args.dataset,
        )
        if mirage_queries:
            queries = mirage_queries
            logging.info(
                "Loaded %d queries from MIRAGE benchmark (%s, subset=%s)",
                len(queries),
                args.mirage_benchmark_path,
                args.mirage_dataset,
            )
        else:
            logging.warning("MIRAGE benchmark loaded but yielded 0 queries")
    except Exception as exc:
        logging.warning("Failed loading MIRAGE queries: %s", exc)

if not queries:
    try:
        _, built_in_queries, _ = load_beir_datasets(args.dataset, args.split, require_queries=True, require_qrels=False)
        queries = built_in_queries
        logging.info("Loaded %d built-in dataset queries", len(queries))
    except FileNotFoundError as exc:
        err = str(exc)
        if "Queries not found for MedRAG dataset" not in err:
            raise

        logging.warning("%s", err)
        logging.warning("Falling back to user-provided queries via --query/--queries-json.")
        queries = _build_fallback_queries(args)
        logging.info("Loaded %d fallback queries", len(queries))

# grp: If you want to use other datasets, you could prepare your dataset as the format of beir, then load it here.

results = None
if args.model_code == "medcpt":
    try:
        medcpt_query_encoder_path = args.medcpt_query_encoder_path.strip() or os.environ.get("MEDCPT_QUERY_ENCODER_PATH", "").strip() or None
        med_corpus = MedCorpus(
            base_dir=os.path.join(os.getcwd(), "datasets"),
            sources=[args.dataset],
            show_progress=args.show_progress,
        )
        medcpt_retriever = MedCPTRetriever(
            corpus=med_corpus,
            query_encoder_name=medcpt_query_encoder_path,
            faiss_gpu_devices=visible_faiss_gpu_ids,
            use_faiss_gpu=args.use_faiss_gpu,
            use_mmap=True,
        )
        logging.info("Using MedCPTRetriever on local prebuilt MedCPT index for %s", args.dataset)
        results = retrieve_with_medcpt_retriever(corpus, queries, medcpt_retriever, args.top_k, show_progress=args.show_progress)
    except Exception as exc:
        logging.warning("MedCPTRetriever init/retrieval failed, fallback to BEIR path: %s", exc)
elif args.model_code == "bm25":
    medrag_system = get_medrag_retrieval_system(args.model_code, args.dataset)
    if medrag_system is not None:
        logging.info("Using MedRAG local index for %s on %s", args.model_code, args.dataset)
        results = retrieve_with_medrag_system(corpus, queries, medrag_system, args.top_k, show_progress=args.show_progress)

if results is None:
    logging.info("Loading BEIR retriever model...")
    if args.model_code == 'contriever':
        encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code])
        encoder = encoder.to(device)
        tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
        model = DRES(
            DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer),
            batch_size=args.per_gpu_batch_size,
            show_progress_bar=args.show_progress,
        )
    elif args.model_code in ['dpr', 'medcpt']:
        DPR = resolve_dpr_class()
        if DPR is None:
            raise ImportError("DPR is not available in current beir version. Try: pip install beir==1.0.1")
        model = DRES(
            DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])),
            batch_size=args.per_gpu_batch_size,
            corpus_chunk_size=5000,
            show_progress_bar=args.show_progress,
        )
    elif args.model_code == 'ance':
        SentenceBERT = resolve_sentencebert_class()
        if SentenceBERT is None:
            raise ImportError("SentenceBERT class is not available in current beir version.")
        model = DRES(
            SentenceBERT(model_code_to_cmodel_name[args.model_code]),
            batch_size=args.per_gpu_batch_size,
            show_progress_bar=args.show_progress,
        )
    elif args.model_code == 'bm25':
        BM25 = resolve_bm25_class()
        if BM25 is None:
            raise ImportError("BM25Search is not available in current beir version. Please use MedRAG local index or upgrade beir.")
        model = BM25(index_name=f"beir_{args.dataset}", hostname=os.environ.get("BM25_HOSTNAME", "localhost"), initialize=True)
    else:
        raise NotImplementedError

    logging.info("Retriever class: %s", model.__class__.__name__)
    retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[args.top_k])
    results = retriever.retrieve(corpus, queries)
                                            
logging.info("Printing results to %s"%(args.result_output))
sub_results = compress(results)

with open(args.result_output, 'w', encoding='utf-8') as f:
    json.dump(sub_results, f)
