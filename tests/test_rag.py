import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from app.rag import (
    DATA_FILE,
    KoreanGrammarRAG,
    NO_ANSWER,
    NO_TOPIK_RESULTS,
    RetrievalPlan,
)


ENTRY = {
    "id": "1",
    "sense_id": "2",
    "sense_number": 1,
    "grammar": "-는 반면에",
    "definition_en": "while; whereas",
    "definition_ko": "두 사실을 대조함을 나타내는 표현.",
    "usage": "",
    "examples": ["도시는 복잡한 반면에 시골은 조용하다."],
    "related": [],
    "source": "Korean Basic Dictionary",
}

TOPIK_RESULT = {
    "retrieval_type": "lexical",
    "match_type": "candidate_occurrence",
    "grammar_query": "-는 반면에",
    "match_count": 1,
    "exam_number": 60,
    "section": "reading",
    "question_start": 13,
    "question_end": 13,
    "question_numbers": [13],
    "pages": [9],
    "source_file": "topik_60.pdf",
    "chunk_type": "question",
    "text": "TOPIK text must stay out of the explanation prompt.",
}


def make_rag() -> KoreanGrammarRAG:
    rag = KoreanGrammarRAG.__new__(KoreanGrammarRAG)
    rag.entries = [ENTRY]
    rag.entry_lookup = {rag._make_entry_key(ENTRY): ENTRY}
    rag.grammar_lookup = {ENTRY["grammar"]: [ENTRY]}
    rag.vectorstore = Mock()
    rag.vectorstore.similarity_search_with_score.side_effect = AssertionError(
        "exact lookup should not use the vector store"
    )
    rag.planner = Mock()
    rag.planner.invoke.side_effect = AssertionError(
        "explicit grammar should not use the planner"
    )
    rag.relevance_validator = Mock()
    rag.relevance_validator.invoke.side_effect = AssertionError(
        "exact lookup should not use relevance validation"
    )
    rag.llm = Mock()
    rag.topik = Mock()
    rag.topik.units = [TOPIK_RESULT]
    rag.topik.lexical_search.return_value = [TOPIK_RESULT]
    return rag


class DeterministicRoutingTests(TestCase):
    def test_explicit_definition_uses_dictionary_only(self):
        plan = make_rag()._create_fast_plan("What does -는 반면에 mean?")

        self.assertTrue(plan.use_dictionary)
        self.assertFalse(plan.use_topik)
        self.assertEqual(plan.grammar_forms, ["-는 반면에"])

    def test_topik_occurrence_uses_lexical_only(self):
        rag = make_rag()
        plan = rag._create_fast_plan("Show me TOPIK occurrences of -는 반면에.")

        self.assertFalse(plan.use_dictionary)
        self.assertTrue(plan.use_topik)
        self.assertEqual(plan.topik_grammar_forms, ["-는 반면에"])

        result = rag.ask("Show me TOPIK occurrences of -는 반면에.")
        self.assertIsNone(result["explanation"])
        self.assertEqual(len(result["topik_occurrences"]), 1)
        self.assertIsNone(result["topik_message"])
        rag.planner.invoke.assert_not_called()
        rag.llm.invoke.assert_not_called()

    def test_combined_request_uses_one_grounded_generation_call(self):
        rag = make_rag()
        rag.llm.invoke.return_value = SimpleNamespace(content="Grounded answer")

        result = rag.ask(
            "Explain -는 반면에 and show me TOPIK occurrences."
        )

        self.assertEqual(result["explanation"], "Grounded answer")
        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(len(result["topik_occurrences"]), 1)
        self.assertIsNone(result["topik_message"])
        rag.planner.invoke.assert_not_called()
        rag.relevance_validator.invoke.assert_not_called()
        rag.llm.invoke.assert_called_once()
        prompt = rag.llm.invoke.call_args.args[0]
        self.assertNotIn(TOPIK_RESULT["text"], prompt)
        self.assertNotIn("Relevant TOPIK occurrences are shown below", prompt)
        self.assertIn("Do not mention\nTOPIK", prompt)

    def test_combined_request_reports_no_topik_matches_deterministically(self):
        rag = make_rag()
        rag.topik.lexical_search.return_value = []
        rag.llm.invoke.return_value = SimpleNamespace(content="Grounded answer")

        result = rag.ask(
            "Explain -는 반면에 and show me TOPIK occurrences."
        )

        self.assertEqual(result["explanation"], "Grounded answer")
        self.assertEqual(result["topik_occurrences"], [])
        self.assertEqual(result["topik_message"], NO_TOPIK_RESULTS)

    def test_grammar_only_answer_has_no_topik_state(self):
        rag = make_rag()
        grammar_entry = {**ENTRY, "grammar": "-지만"}
        rag.entries = [grammar_entry]
        rag.entry_lookup = {rag._make_entry_key(grammar_entry): grammar_entry}
        rag.grammar_lookup = {"-지만": [grammar_entry]}
        rag.llm.invoke.return_value = SimpleNamespace(content="Grounded answer")

        result = rag.ask("What does -지만 mean?")

        self.assertEqual(result["explanation"], "Grounded answer")
        self.assertEqual(result["topik_occurrences"], [])
        self.assertIsNone(result["topik_message"])
        rag.topik.lexical_search.assert_not_called()

    def test_korean_suffix_after_known_form_is_trimmed(self):
        rag = make_rag()
        plan = rag._create_fast_plan("토픽에서 -는 반면에 예문을 보여 줘.")
        self.assertEqual(plan.topik_grammar_forms, ["-는 반면에"])

        plan = rag._create_fast_plan("-는 반면에와 다른 표현을 비교해 줘.")
        self.assertEqual(plan.grammar_forms, ["-는 반면에"])

    def test_topic_and_function_queries_still_use_llm_planning(self):
        rag = make_rag()
        self.assertIsNone(
            rag._create_fast_plan("Find TOPIK questions about environmental pollution.")
        )
        self.assertIsNone(
            rag._create_fast_plan("What grammar can I use to express a guess?")
        )

    def test_exact_candidate_skips_semantic_retrieval_and_validation(self):
        rag = make_rag()
        candidates = rag._retrieve_dictionary_candidates(
            RetrievalPlan(grammar_forms=["-는 반면에"])
        )
        selected = rag._validate_dictionary_relevance("question", candidates)

        self.assertEqual(selected, candidates)
        rag.vectorstore.similarity_search_with_score.assert_not_called()
        rag.relevance_validator.invoke.assert_not_called()

    def test_no_evidence_abstains(self):
        rag = make_rag()
        rag.grammar_lookup = {}
        rag.vectorstore.similarity_search_with_score.side_effect = None
        rag.vectorstore.similarity_search_with_score.return_value = []

        result = rag.ask("What does -없는문법 mean?")

        self.assertEqual(result["explanation"], NO_ANSWER)
        self.assertEqual(result["entries"], [])


class ResultHelpersTests(TestCase):
    def test_semantic_candidates_are_diversified_by_dictionary_relationships(self):
        rag = make_rag()
        variants = [
            {
                **ENTRY,
                "id": str(index),
                "grammar": grammar,
                "definition_en": definition,
                "definition_ko": definition,
                "related": [
                    {"grammar": related, "type": "참고어"}
                    for related in related_forms
                ],
            }
            for index, (grammar, definition, related_forms) in enumerate(
                [
                    (
                        "-ㄹ 듯",
                        "An expression used to guess that something said might be similar to the preceding situation.",
                        ["-을 듯"],
                    ),
                    (
                        "-을 듯",
                        "An expression used to guess that something said might be similar to the preceding situation.",
                        ["-ㄹ 듯"],
                    ),
                    (
                        "-ㄹ 듯하다",
                        "An expression used to guess the content of the preceding statement.",
                        [],
                    ),
                    ("-을지요", "guess family two", ["-ㄹ지요"]),
                    ("-ㄹ 것 같다", "guess family three", ["-을 것 같다"]),
                ],
                start=1,
            )
        ]
        rag.grammar_lookup = {}
        for entry in variants:
            rag.grammar_lookup.setdefault(entry["grammar"], []).append(entry)
        candidates = [
            {"entry": entry, "retrieval_type": "semantic", "distance": index / 10}
            for index, entry in enumerate(variants, start=1)
        ]

        diversified = rag._diversify_semantic_candidates(candidates)

        self.assertEqual(
            [item["entry"]["grammar"] for item in diversified],
            ["-ㄹ 듯", "-을지요", "-ㄹ 것 같다"],
        )
        self.assertEqual(
            diversified[0]["family_variants"],
            ["-ㄹ 듯", "-ㄹ 듯하다", "-을 듯"],
        )

    def test_topik_deduplication_preserves_first_result(self):
        rag = make_rag()
        duplicate = dict(TOPIK_RESULT, distance=0.2)
        self.assertEqual(
            rag._dedupe_topik_results([TOPIK_RESULT, duplicate]), [TOPIK_RESULT]
        )

    def test_page_normalization(self):
        self.assertEqual(KoreanGrammarRAG._normalize_pages("9, 10,9"), [9, 10])

    def test_local_dictionary_contains_reference_form(self):
        if not DATA_FILE.exists():
            self.skipTest("Dictionary data has not been prepared.")

        with DATA_FILE.open(encoding="utf-8") as file:
            entries = json.load(file)
        matches = [entry for entry in entries if entry.get("grammar") == "-는 반면에"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["source"], "Korean Basic Dictionary")
