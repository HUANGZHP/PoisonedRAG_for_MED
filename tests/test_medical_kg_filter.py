import json

import numpy as np

from src.medical_kg_filter import (
    BIOSKnowledgeGraph,
    LLMTripletExtractor,
    MedicalKGRiskReranker,
    MedicalTriplet,
    write_bios_artifact,
)


def _embed(texts):
    table = {
        "ibuprofen": [1.0, 0.0, 0.0],
        "stomach pain": [0.0, 1.0, 0.0],
        "stomach ulcer": [0.0, 0.0, 1.0],
        "aspirin": [0.0, 0.0, -1.0],
        "may treat": [1.0, 1.0, 0.0],
        "mystery": [-1.0, 0.0, 0.0],
    }
    return np.asarray([table.get(str(text).casefold(), table["mystery"]) for text in texts], dtype=np.float32)


def _graph():
    concepts = ["ibuprofen", "stomach pain", "stomach ulcer", "aspirin"]
    relations = ["may treat"]
    return BIOSKnowledgeGraph(
        concepts=concepts,
        relationships=relations,
        origins=np.asarray([0, 3], dtype=np.int32),
        relation_ids=np.asarray([0, 0], dtype=np.int32),
        targets=np.asarray([1, 2], dtype=np.int32),
        concept_embeddings=_embed(concepts),
        relationship_embeddings=_embed(relations),
        embedder=_embed,
    )


def test_original_style_edge_verification_and_unknown():
    graph = _graph()
    valid = graph.verify(MedicalTriplet("ibuprofen", "may treat", "stomach pain"), original_mode=True)
    invalid = graph.verify(MedicalTriplet("ibuprofen", "may treat", "stomach ulcer"), original_mode=True)
    unknown = graph.verify(
        MedicalTriplet("mystery", "may treat", "stomach pain"),
        original_mode=False,
        match_threshold=0.45,
    )
    assert valid.state == "valid"
    assert valid.valid_phrase == "ibuprofen may treat stomach pain"
    assert invalid.state == "invalid"
    assert unknown.state == "unknown"


class _StaticExtractor:
    def extract(self, text):
        if "poison" in text:
            return [MedicalTriplet("ibuprofen", "may treat", "stomach ulcer")]
        return [MedicalTriplet("ibuprofen", "may treat", "stomach pain")]


def test_risk_reranking_preserves_unknown_neutrality():
    reranker = MedicalKGRiskReranker(
        _graph(),
        _StaticExtractor(),
        mode="conservative",
        rerank_weight=0.60,
        match_threshold=0.45,
    )
    ranked, stats = reranker.rerank(
        [
            {"context": "poisoned assertion", "score": 0.90},
            {"context": "verified assertion", "score": 0.80},
        ],
        final_top_k=1,
    )
    assert ranked[0]["context"] == "verified assertion"
    assert stats.invalid_triplets == 1
    assert stats.valid_triplets == 1
    assert stats.risky_document_count == 1


def test_binary_hard_filter_removes_identified_malicious_documents_without_topk_fallback():
    reranker = MedicalKGRiskReranker(
        _graph(),
        _StaticExtractor(),
        mode="original",
        rerank_weight=1.0,
        decision_mode="hard_filter",
        hard_filter_threshold=1.0,
    )
    ranked, stats = reranker.rerank(
        [
            {"context": "poisoned assertion", "score": 0.90},
            {"context": "verified assertion", "score": 0.80},
        ],
        final_top_k=5,
    )
    assert [item["context"] for item in ranked] == ["verified assertion"]
    assert stats.selected_count == 1
    assert stats.hard_filtered_count == 1
    audit_by_original_rank = {audit["original_rank"]: audit for audit in stats.document_audits}
    assert audit_by_original_rank[1]["hard_filtered"] is True
    assert audit_by_original_rank[1]["final_rank"] is None


def test_non_strict_relation_is_audited_without_becoming_original_mode_risk():
    class _AssociationExtractor:
        def extract(self, text):
            return [MedicalTriplet("ibuprofen", "associated with", "stomach ulcer")]

    reranker = MedicalKGRiskReranker(
        _graph(),
        _AssociationExtractor(),
        mode="original",
        decision_mode="hard_filter",
        hard_filter_threshold=1.0,
        non_strict_relationships=("associated with",),
    )
    ranked, stats = reranker.rerank(
        [{"context": "non-causal association", "score": 0.90}],
        final_top_k=1,
    )
    assert [item["context"] for item in ranked] == ["non-causal association"]
    assert stats.ignored_triplets == 1
    assert stats.risky_document_count == 0
    assert stats.document_audits[0]["ignored_triplet_count"] == 1
    assert stats.document_audits[0]["triplets"][0]["state"] == "ignored"


def test_llm_triplet_json_parser_and_artifact_writer(tmp_path):
    class _LLM:
        calls = 0

        def query(self, _):
            self.calls += 1
            return json.dumps([{"origin": "ibuprofen", "relationship": "may treat", "target": "stomach pain"}])

    llm = _LLM()
    cache_path = tmp_path / "triplets.jsonl"
    extractor = LLMTripletExtractor(llm, max_chars=256, max_triplets=2, cache_path=cache_path)
    assert extractor.extract("x") == [MedicalTriplet("ibuprofen", "may treat", "stomach pain")]
    assert extractor.extract("x") == [MedicalTriplet("ibuprofen", "may treat", "stomach pain")]
    assert llm.calls == 1
    reloaded = LLMTripletExtractor(_LLM(), max_chars=256, max_triplets=2, cache_path=cache_path)
    assert reloaded.extract("x") == [MedicalTriplet("ibuprofen", "may treat", "stomach pain")]
    assert reloaded.cache_hits == 1
    assert cache_path.read_text(encoding="utf-8").count("\n") == 1
    metadata = write_bios_artifact(
        [MedicalTriplet("ibuprofen", "may treat", "stomach pain")],
        tmp_path / "artifact",
        embedder=_embed,
        source_description="unit-test",
    )
    assert metadata["concept_count"] == 2
    assert metadata["edge_count"] == 1
    assert (tmp_path / "artifact" / "edges.npz").is_file()
