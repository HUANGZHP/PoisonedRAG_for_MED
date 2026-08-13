import unittest

from agentic_main import _apply_candidate_defenses
from src.trustrag_filter import TrustRAGOriginalStats


class FakeTrustRAG:
    def __init__(self, remove):
        self.remove = set(remove)
        self.calls = []

    def filter(self, documents, min_keep):
        self.calls.append((list(documents), min_keep))
        retained = [doc for doc in documents if doc not in self.remove]
        return retained, TrustRAGOriginalStats(True, True, len(documents) - len(retained))


class FakeJudge:
    def __init__(self, malicious):
        self.malicious = set(malicious)
        self.calls = []

    def query(self, prompt):
        self.calls.append(prompt)
        return "yes" if any(f"Context:\n{doc}" in prompt for doc in self.malicious) else "no"


class AgenticDefensePipelineTest(unittest.TestCase):
    def test_judge_fallback_never_reintroduces_trustrag_removed_documents(self):
        trustrag = FakeTrustRAG(remove={"poison"})
        judge = FakeJudge(malicious={"b", "c", "d"})

        retained, info = _apply_candidate_defenses(
            "question",
            ["a", "poison", "b", "c", "d", "e"],
            candidate_k=6,
            top_k=3,
            trustrag_filter=trustrag,
            medical_cluster_filter=None,
            judge_llm=judge,
            adv_text_set={"poison"},
        )

        self.assertEqual(trustrag.calls, [(["a", "poison", "b", "c", "d", "e"], 3)])
        self.assertTrue(info["judge_retention_fallback"])
        self.assertEqual(retained, ["a", "b", "c"])
        self.assertNotIn("poison", info["pre_judge_candidates"])
        self.assertNotIn("poison", retained)
        self.assertEqual(info["judge_filtered_count"], 0)

    def test_judge_keeps_relative_rank_when_its_filter_has_enough_documents(self):
        trustrag = FakeTrustRAG(remove={"poison"})
        judge = FakeJudge(malicious={"b"})

        retained, info = _apply_candidate_defenses(
            "question",
            ["a", "poison", "b", "c", "d", "e"],
            candidate_k=6,
            top_k=3,
            trustrag_filter=trustrag,
            medical_cluster_filter=None,
            judge_llm=judge,
            adv_text_set={"poison"},
        )

        self.assertFalse(info["judge_retention_fallback"])
        self.assertEqual(retained, ["a", "c", "d"])
        self.assertEqual(info["judge_filtered_count"], 1)
        self.assertNotIn("poison", retained)


if __name__ == "__main__":
    unittest.main()
