import unittest

from src.search_o1_integration import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    run_agentic_rag,
)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def query(self, prompt):
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("unexpected extra LLM call")
        return self._responses.pop(0)


def search_tag(query):
    return f"{BEGIN_SEARCH_QUERY}{query}{END_SEARCH_QUERY}"


class AgenticLiveRetrievalTest(unittest.TestCase):
    def test_live_retrieval_and_filter_run_for_each_requested_query(self):
        llm = FakeLLM([search_tag("first query"), search_tag("second query"), "yes"])
        retrieved = []
        filtered = []

        def retrieve(query, candidate_k):
            retrieved.append((query, candidate_k))
            return [f"raw:{query}", "raw:other"]

        def apply_filter(query, documents):
            filtered.append((query, list(documents)))
            return [f"safe:{query}"]

        answer, survived, trace = run_agentic_rag(
            llm=llm,
            question="medical question",
            topk_contents=["legacy static document"],
            question_type="yesno",
            answer_labels=("yes", "no", "maybe"),
            max_turns=3,
            round_retrieve=retrieve,
            round_filter=apply_filter,
            candidate_k=10,
            return_trace=True,
        )

        self.assertEqual(answer, "yes")
        self.assertEqual(survived, 0)
        self.assertEqual(retrieved, [("first query", 10), ("second query", 10)])
        self.assertEqual(
            filtered,
            [
                ("first query", ["raw:first query", "raw:other"]),
                ("second query", ["raw:second query", "raw:other"]),
            ],
        )
        self.assertEqual(len(trace["rounds"]), 2)
        self.assertIn("safe:first query", llm.prompts[1])
        self.assertIn("safe:second query", llm.prompts[2])
        self.assertNotIn("legacy static document", llm.prompts[1])

    def test_turn_limit_requests_a_final_answer_instead_of_returning_search_tag(self):
        llm = FakeLLM([search_tag("q1"), search_tag("q2"), "no"])
        calls = []

        def retrieve(query, candidate_k):
            calls.append((query, candidate_k))
            return [query]

        answer, _, trace = run_agentic_rag(
            llm=llm,
            question="question",
            topk_contents=[],
            question_type="yesno",
            answer_labels=("yes", "no", "maybe"),
            max_turns=2,
            round_retrieve=retrieve,
            candidate_k=5,
            return_trace=True,
        )

        self.assertEqual(answer, "no")
        self.assertEqual(calls, [("q1", 5), ("q2", 5)])
        self.assertEqual(len(trace["rounds"]), 2)
        self.assertIn("Do not request another search", llm.prompts[-1])

    def test_legacy_call_keeps_static_topk_when_no_callback_is_supplied(self):
        llm = FakeLLM([search_tag("ignored query"), "maybe"])

        answer, _, trace = run_agentic_rag(
            llm=llm,
            question="question",
            topk_contents=["legacy evidence"],
            question_type="yesno",
            answer_labels=("yes", "no", "maybe"),
            max_turns=2,
            return_trace=True,
        )

        self.assertEqual(answer, "maybe")
        self.assertFalse(trace["live_retrieval"])
        self.assertIn("legacy evidence", llm.prompts[1])


if __name__ == "__main__":
    unittest.main()
