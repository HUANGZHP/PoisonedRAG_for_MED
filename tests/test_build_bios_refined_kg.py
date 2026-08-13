import importlib.util
from pathlib import Path


_LOCAL_EVIDENCE_SCRIPT = Path(__file__).with_name("build_bios_refined_kg.evidence.py")
_SCRIPT = _LOCAL_EVIDENCE_SCRIPT if _LOCAL_EVIDENCE_SCRIPT.is_file() else Path(__file__).parents[1] / "scripts" / "build_bios_refined_kg.py"
_SPEC = importlib.util.spec_from_file_location("build_bios_refined_kg_evidence", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_clinical_priority_keeps_high_relation_caps_mid_and_excludes_low(tmp_path):
    concepts = tmp_path / "concepts.tsv"
    concepts.write_text(
        "cid\ttid\tstr\ttty\tlang\n"
        "1\tt1\tdrug a\tPT\tEN\n"
        "2\tt2\tdisease a\tPT\tEN\n"
        "3\tt3\tdisease b\tPT\tEN\n"
        "4\tt4\tclass a\tPT\tEN\n",
        encoding="utf-8",
    )
    relations = tmp_path / "relations.tsv"
    relations.write_text(
        "head.cid\thead.tid\trelation\ttail.cid\ttail.tid\n"
        "1\tt1\tmay treat\t2\tt2\n"
        "1\tt1\tmay treat\t3\tt3\n"
        "2\tt2\tis a\t4\tt4\n"
        "3\tt3\tis a\t4\tt4\n"
        "1\tt1\tassociated with\t2\tt2\n",
        encoding="utf-8",
    )

    triples, selection = _MODULE._derive_preferred_term_bios_edges(
        concepts,
        relations,
        ("may treat", "is a", "associated with"),
        max_edges=99,
        seed=0,
        sampling_policy="clinical-priority",
        clinical_priority_quotas={"may treat": None, "is a": 1, "associated with": 0},
    )

    assert selection["selected_edge_counts"] == {
        "associated with": 0,
        "is a": 1,
        "may treat": 2,
    }
    assert selection["source_english_pt_edge_counts"] == {
        "associated with": 1,
        "is a": 2,
        "may treat": 2,
    }
    assert {(edge.relationship, edge.origin, edge.target) for edge in triples if edge.relationship == "may treat"} == {
        ("may treat", "drug a", "disease a"),
        ("may treat", "drug a", "disease b"),
    }
    assert sum(edge.relationship == "is a" for edge in triples) == 1
    assert all(edge.relationship != "associated with" for edge in triples)
    assert _MODULE._effective_edge_counts(triples) == {"is a": 1, "may treat": 2}


def test_no_sampling_keeps_every_selected_relation_without_a_quota(tmp_path):
    concepts = tmp_path / "concepts.tsv"
    concepts.write_text(
        "cid\ttid\tstr\ttty\tlang\n"
        "1\tt1\tdrug a\tPT\tEN\n"
        "2\tt2\tdisease a\tPT\tEN\n"
        "3\tt3\tdisease b\tPT\tEN\n"
        "4\tt4\tclass a\tPT\tEN\n",
        encoding="utf-8",
    )
    relations = tmp_path / "relations.tsv"
    relations.write_text(
        "head.cid\thead.tid\trelation\ttail.cid\ttail.tid\n"
        "1\tt1\tmay treat\t2\tt2\n"
        "1\tt1\tmay treat\t3\tt3\n"
        "2\tt2\tis a\t4\tt4\n"
        "3\tt3\tis a\t4\tt4\n"
        "1\tt1\tassociated with\t2\tt2\n",
        encoding="utf-8",
    )

    triples, selection = _MODULE._derive_preferred_term_bios_edges(
        concepts,
        relations,
        ("may treat", "is a", "associated with"),
        max_edges=None,
        seed=0,
        sampling_policy="no-sampling",
    )

    assert len(triples) == 5
    assert selection["source_english_pt_edge_counts"] == {
        "associated with": 1,
        "is a": 2,
        "may treat": 2,
    }
    assert selection["selected_edge_counts"] == selection["source_english_pt_edge_counts"]
    assert _MODULE._effective_edge_counts(triples) == selection["source_english_pt_edge_counts"]
