#!/usr/bin/env python3
"""Build a compact, MedCPT-indexed BIOS graph artifact for PoisonedRAG.

The Nature Medicine reference implementation consumes an already refined BIOS
ground-truth graph.  Its exact graph snapshot is not published with the public
demo repository, so this builder has two explicit routes:

* ``--triples-json``: build from a supplied canonical/refined ground-truth
  JSON.  This is the route to use if the authors' exact artifact becomes
  available.
* ``--bios-concepts`` + ``--bios-relations``: derive a deterministic English
  preferred-term BIOS v3 artifact.  The default ``no-sampling`` policy keeps
  every edge for the 13 clinical relation labels used by the verifier.  It
  deliberately performs no relation quota, random sampling, or selective
  removal; the only retained source-side refinement is the paper-aligned
  exclusion of non-English and synonym terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medical_kg_filter import (
    DEFAULT_MAX_LENGTH,
    MedCPTTextEncoder,
    MedicalTriplet,
    write_bios_artifact,
)


# The paper's supplementary incompatible-relation table enumerates these 13
# clinical relation labels.  BIOS v3 uses ``ddx`` for differential diagnosis.
ORIGINAL_ALIGNED_RELATIONS: Tuple[str, ...] = (
    "associated with",
    "contraindication",
    "ddx",
    "has adverse effect",
    "interacts with",
    "is a",
    "is adverse effect of",
    "may be caused by",
    "may cause",
    "may be diagnosed by",
    "may diagnose",
    "may treat",
    "may be treated by",
)

# Relation-aware clinical policy for the reproduced BIOS-v3 artifact. ``None``
# means retain every English preferred-term edge; an integer means deterministic
# reservoir sampling up to that cap; zero means exclude from the strict graph.
# The two may-cause relations currently fall below their 100k cap, so they are
# retained in full; the very broad ``is a`` hierarchy is capped to prevent it
# from dominating the graph. ``associated with`` is intentionally not strict.
CLINICAL_PRIORITY_RELATION_QUOTAS: Mapping[str, int | None] = {
    # High priority: all retained.
    "may treat": None,
    "may be treated by": None,
    "contraindication": None,
    "has adverse effect": None,
    "is adverse effect of": None,
    "interacts with": None,
    "may diagnose": None,
    "may be diagnosed by": None,
    "ddx": None,
    # Middle priority: capped only when the source relation is very broad.
    "may cause": 100_000,
    "may be caused by": 100_000,
    "is a": 150_000,
    # Low priority: excluded rather than being treated as an invalid claim.
    "associated with": 0,
}

# ``no-sampling`` is the default formal policy.  The two sampling policies are
# retained only to reproduce historical ablations; neither is used by default.
SAMPLING_POLICIES = ("no-sampling", "clinical-priority", "relation-balanced")


def _canonical_relation(value: str) -> str:
    value = " ".join(value.strip().split()).casefold()
    return "differential diagnosis" if value == "ddx" else value


def _artifact_edge_key(triplet: MedicalTriplet) -> Tuple[str, str, str]:
    """Match the text-normalisation and edge de-duplication used by the writer."""

    clean = triplet.cleaned()
    return tuple(
        " ".join(str(value).replace("_", " ").strip().split()).casefold()
        for value in (clean.origin, clean.relationship, clean.target)
    )


def _effective_edge_counts(triples: Sequence[MedicalTriplet]) -> Dict[str, int]:
    """Count unique text edges that will actually be materialised in the artifact."""

    seen = set()
    counts: Dict[str, int] = {}
    for triplet in triples:
        key = _artifact_edge_key(triplet)
        if key in seen:
            continue
        seen.add(key)
        counts[triplet.cleaned().relationship] = counts.get(triplet.cleaned().relationship, 0) + 1
    return dict(sorted(counts.items()))


def _load_json_triples(path: Path) -> List[MedicalTriplet]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--triples-json must contain a JSON list of triples.")
    triples: List[MedicalTriplet] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        triplet = MedicalTriplet(
            str(row.get("origin", "")),
            str(row.get("relationship", row.get("relation", ""))),
            str(row.get("target", "")),
        ).cleaned()
        if all((triplet.origin, triplet.relationship, triplet.target)):
            triples.append(triplet)
    if not triples:
        raise ValueError(f"No valid triples found in {path}.")
    return triples


def _derive_preferred_term_bios_edges(
    concepts_path: Path,
    relations_path: Path,
    relation_allowlist: Sequence[str],
    max_edges: int | None,
    seed: int,
    sampling_policy: str,
    clinical_priority_quotas: Mapping[str, int | None] | None = None,
) -> Tuple[List[MedicalTriplet], Dict[str, object]]:
    """Derive an English preferred-term BIOS graph from the raw v3 dump.

    The paper states that it removes nodes labelled as synonyms.  ``tty=PT`` is
    BIOS's preferred-term label, whereas ``ET`` entries are alternate terms;
    retaining only English PT terms is therefore the closest reproducible raw
    transformation available in the public BIOS v3 release.  The authors'
    exact historical refined graph is not publicly distributed.

    ``no-sampling`` is the default: every BIOS edge whose endpoints are English
    preferred terms and whose relation lies in ``relation_allowlist`` is kept.
    No cap, reservoir sampling, or low-priority relation exclusion is applied.
    ``clinical-priority`` and ``relation-balanced`` remain available only for
    reproducing historical ablations.
    """

    if max_edges is not None and max_edges < 1:
        raise ValueError("--max-edges must be positive when supplied.")
    if sampling_policy == "relation-balanced" and max_edges is None:
        raise ValueError("--sampling-policy relation-balanced requires an explicit --max-edges.")
    if sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(f"Unknown sampling policy: {sampling_policy!r}")
    allowed = {_canonical_relation(value) for value in relation_allowlist}
    if not allowed:
        raise ValueError("At least one relationship must be selected.")
    terms: Dict[str, str] = {}
    with concepts_path.open(encoding="utf-8") as handle:
        header = next(handle, "").rstrip("\n").split("\t")
        if header != ["cid", "tid", "str", "tty", "lang"]:
            raise ValueError(f"Unexpected BIOS concepts header: {header!r}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                continue
            _, tid, text, term_type, language = fields
            if term_type == "PT" and language == "EN" and text.strip() and tid not in terms:
                terms[tid] = text.strip()
            if line_number % 10_000_000 == 0:
                print(f"scanned BIOS concepts: {line_number:,} rows; English PT terms={len(terms):,}", flush=True)
    if not terms:
        raise ValueError("No English BIOS preferred terms were found.")

    rng = random.Random(seed)
    if sampling_policy == "clinical-priority":
        priority_quotas = clinical_priority_quotas or CLINICAL_PRIORITY_RELATION_QUOTAS
        quotas = {
            _canonical_relation(relation): quota
            for relation, quota in priority_quotas.items()
            if _canonical_relation(relation) in allowed
        }
        missing = sorted(allowed - set(quotas))
        if missing:
            raise ValueError(
                "Clinical-priority policy has no quota for requested relations: " + ", ".join(missing)
            )
        reservoirs: Dict[str, List[MedicalTriplet]] = {relation: [] for relation in quotas}
        per_relation = None
    elif sampling_policy == "relation-balanced":
        quotas = {}
        assert max_edges is not None
        per_relation = max(1, max_edges // len(allowed))
        reservoirs = {relation: [] for relation in allowed}
    else:
        # Full-graph mode has no per-relation reservoir.  Edges are appended in
        # source order and are never sampled or discarded by a quota.
        quotas = {}
        per_relation = None
        reservoirs = {}
    seen_by_relation: Dict[str, int] = {relation: 0 for relation in allowed}
    all_triples: List[MedicalTriplet] = []

    with relations_path.open(encoding="utf-8") as handle:
        header = next(handle, "").rstrip("\n").split("\t")
        if header[:5] != ["head.cid", "head.tid", "relation", "tail.cid", "tail.tid"]:
            raise ValueError(f"Unexpected BIOS relations header: {header!r}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                continue
            _, head_tid, relation, _, tail_tid = fields
            relation = _canonical_relation(relation)
            if relation not in allowed or head_tid not in terms or tail_tid not in terms:
                continue
            triplet = MedicalTriplet(terms[head_tid], relation, terms[tail_tid]).cleaned()
            seen_by_relation[relation] += 1
            if sampling_policy == "clinical-priority":
                quota = quotas[relation]
                if quota is None:
                    reservoirs[relation].append(triplet)
                elif quota > 0:
                    bucket = reservoirs[relation]
                    if len(bucket) < quota:
                        bucket.append(triplet)
                    else:
                        replacement = rng.randrange(seen_by_relation[relation])
                        if replacement < quota:
                            bucket[replacement] = triplet
            elif sampling_policy == "relation-balanced":
                bucket = reservoirs[relation]
                assert per_relation is not None
                if len(bucket) < per_relation:
                    bucket.append(triplet)
                else:
                    replacement = rng.randrange(seen_by_relation[relation])
                    if replacement < per_relation:
                        bucket[replacement] = triplet
            else:
                all_triples.append(triplet)
                if max_edges is not None and len(all_triples) > max_edges:
                    raise RuntimeError(
                        "Preferred-term BIOS graph exceeds explicit --max-edges=%d. "
                        "No sampling has been performed; omit --max-edges to retain the complete graph, "
                        "or increase this safety guard after review."
                        % max_edges
                    )
            if line_number % 10_000_000 == 0:
                print(f"scanned BIOS relations: {line_number:,} rows", flush=True)

    triples = all_triples if sampling_policy == "no-sampling" else [
        edge for relation in sorted(reservoirs) for edge in reservoirs[relation]
    ]
    if not triples:
        raise ValueError("No selected BIOS clinical edges had English PT endpoints.")
    selected_by_relation = (
        {relation: len(reservoirs[relation]) for relation in sorted(reservoirs)}
        if sampling_policy != "no-sampling"
        else {relation: seen_by_relation[relation] for relation in sorted(seen_by_relation)}
    )
    selection = {
        "sampling_policy": sampling_policy,
        "source_english_pt_edge_counts": dict(sorted(seen_by_relation.items())),
        "selected_edge_counts": selected_by_relation,
        "relation_quotas": (
            {relation: quotas[relation] for relation in sorted(quotas)}
            if sampling_policy == "clinical-priority"
            else None
        ),
    }
    print(
        "English PT terms=%d; retained triples=%d; selection=%s"
        % (len(terms), len(triples), json.dumps(selection, sort_keys=True)),
        flush=True,
    )
    return triples, selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--triples-json", type=Path, help="Canonical refined graph JSON list.")
    source.add_argument("--bios-relations", type=Path, help="Raw BIOS v3 Relations*.txt.")
    parser.add_argument("--bios-concepts", type=Path, help="Raw BIOS v3 Concepts*.txt (required with --bios-relations).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="/home/HF_Model/ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--device", default="cpu", help="Embedding device, for example cpu or cuda:1.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional abort-only safety guard. Omit (default) to retain every selected raw BIOS edge; never samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate and count the selected triples without encoding or writing an artifact.")
    parser.add_argument(
        "--sampling-policy",
        choices=SAMPLING_POLICIES,
        default="no-sampling",
        help="no-sampling (default) retains all selected BIOS edges; other choices reproduce historical sampled ablations.",
    )
    parser.add_argument(
        "--allow-relation-balanced-sample",
        action="store_true",
        help="Deprecated compatibility alias for --sampling-policy relation-balanced.",
    )
    parser.add_argument(
        "--relations",
        nargs="*",
        default=list(ORIGINAL_ALIGNED_RELATIONS),
        help="BIOS relationship labels to retain for raw-data derivation; default is all 13 labels used by the verifier.",
    )
    parser.add_argument("--force", action="store_true", help="Permit replacing a complete existing artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bios_relations is not None and args.bios_concepts is None:
        raise ValueError("--bios-concepts is required with --bios-relations.")
    output = args.output_dir
    existing = [output / name for name in ("metadata.json", "concepts.json", "relationships.json", "edges.npz")]
    if any(path.exists() for path in existing) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite an existing KG artifact at {output}. Use --force only after review."
        )
    if args.allow_relation_balanced_sample:
        if args.sampling_policy != "no-sampling":
            raise ValueError("--allow-relation-balanced-sample cannot be combined with an explicit sampling policy.")
        args.sampling_policy = "relation-balanced"
    if args.triples_json is not None:
        triples = _load_json_triples(args.triples_json)
        source_description = f"canonical graph supplied at {args.triples_json}"
        refinement = "external_canonical_ground_truth"
        selection: Dict[str, object] = {
            "sampling_policy": "external-canonical",
            "source_english_pt_edge_counts": None,
            "selected_edge_counts": None,
            "relation_quotas": None,
        }
    else:
        triples, selection = _derive_preferred_term_bios_edges(
            args.bios_concepts,
            args.bios_relations,
            args.relations,
            args.max_edges,
            args.seed,
            args.sampling_policy,
        )
        if args.sampling_policy == "clinical-priority":
            source_description = f"BIOS v3 English PT clinical-priority graph from {args.bios_relations}"
            refinement = "bios_v3_english_preferred_terms_clinical_priority_sample_not_original_unpublished_snapshot"
        elif args.sampling_policy == "relation-balanced":
            source_description = f"BIOS v3 English PT relation-balanced sample from {args.bios_relations}"
            refinement = "bios_v3_english_preferred_terms_relation_balanced_sample_not_original_unpublished_snapshot"
        else:
            source_description = f"BIOS v3 full English preferred-term clinical-relation graph from {args.bios_relations}"
            refinement = "bios_v3_english_preferred_terms_all_selected_relations_no_sampling_not_original_unpublished_snapshot"

    selection["effective_edge_counts"] = _effective_edge_counts(triples)
    selection["selected_triples_before_dedup"] = len(triples)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "source_description": source_description,
                    "refinement": refinement,
                    "selected_triples": len(triples),
                    "selected_relations": sorted({triplet.relationship for triplet in triples}),
                    "selected_concepts": len({term for triplet in triples for term in (triplet.origin, triplet.target)}),
                    "selection": selection,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    encoder = MedCPTTextEncoder(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    metadata = write_bios_artifact(
        triples,
        output,
        embedder=encoder,
        source_description=source_description,
        embedding_max_length=args.max_length,
        extra_metadata={
            "reference_algorithm": "Alber et al., Nature Medicine 2025; MedCPT + refined BIOS triplet verification",
            "refinement": refinement,
            "relations_requested": [_canonical_relation(value) for value in args.relations],
            "sampling_policy": selection["sampling_policy"],
            "source_english_pt_edge_counts": selection["source_english_pt_edge_counts"],
            "selected_edge_counts": selection["selected_edge_counts"],
            "effective_edge_counts": selection["effective_edge_counts"],
            "selected_triples_before_dedup": selection["selected_triples_before_dedup"],
            "relation_quotas": selection["relation_quotas"],
            "strict_relations": (
                sorted(
                    _canonical_relation(relation)
                    for relation, quota in CLINICAL_PRIORITY_RELATION_QUOTAS.items()
                    if quota != 0
                )
                if args.triples_json is None and args.sampling_policy == "clinical-priority"
                else sorted({_canonical_relation(relation) for relation in args.relations})
                if args.triples_json is None and args.sampling_policy == "no-sampling"
                else None
            ),
            "non_strict_relations": ["associated with"] if args.triples_json is None and args.sampling_policy == "clinical-priority" else [],
            # --max-edges is intentionally only a legacy guard. The clinical
            # policy is governed by its explicit per-relation quotas instead.
            "raw_edge_cap": args.max_edges if args.bios_relations is not None and args.sampling_policy == "no-sampling" else None,
            "relation_balanced_sample": args.sampling_policy == "relation-balanced" if args.bios_relations is not None else False,
            "seed": args.seed if args.bios_relations is not None else None,
        },
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
