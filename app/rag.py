import json
import logging
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.topik_retriever import TopikRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data/processed/grammar_clean.json"
CHROMA_DIR = PROJECT_ROOT / "data/chroma"
COLLECTION_NAME = "korean_grammar"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_MODEL_NAME = "gpt-5.6-luna"

RETRIEVAL_K = 5
OPEN_ENDED_RETRIEVAL_K = 25
MAX_DICTIONARY_CANDIDATES = 10
MAX_TOPIK_RESULTS = 5
SEMANTIC_DISTANCE_THRESHOLD = 0.45

NO_ANSWER = (
    "The available Korean grammar sources do not contain enough relevant "
    "information to answer this question reliably."
)
NO_TOPIK_RESULTS = "No matching TOPIK questions found."

EXPLANATION_MARKERS = (
    "explain",
    "explanation",
    "what does",
    "what is",
    "definition",
    "meaning",
    "mean?",
    "difference",
    "compare",
    "when should",
    "how do i use",
    "how to use",
    "why is",
    "why does",
    "설명",
    "뜻",
    "의미",
    "차이",
    "비교",
    "사용법",
)

logger = logging.getLogger(__name__)


class RetrievalPlan(BaseModel):
    search_queries: list[str] = Field(default_factory=list)
    grammar_forms: list[str] = Field(default_factory=list)
    use_dictionary: bool = True
    use_topik: bool = False
    topik_queries: list[str] = Field(default_factory=list)
    topik_grammar_forms: list[str] = Field(default_factory=list)


class RelevanceDecision(BaseModel):
    answerable: bool
    relevant_entry_numbers: list[int] = Field(default_factory=list)


class KoreanGrammarRAG:
    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        with DATA_FILE.open(encoding="utf-8") as file:
            self.entries = json.load(file)

        self.entry_lookup = {
            self._make_entry_key(entry): entry for entry in self.entries
        }
        grammar_lookup: defaultdict[str, list[dict]] = defaultdict(list)
        for entry in self.entries:
            grammar = (entry.get("grammar") or "").strip()
            if grammar:
                grammar_lookup[grammar].append(entry)
        self.grammar_lookup = dict(grammar_lookup)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        # Both collections share one CPU embedding model.
        self.topik = TopikRetriever(embeddings=self.embeddings)

        self.llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME))
        self.planner = self.llm.with_structured_output(RetrievalPlan)
        self.relevance_validator = self.llm.with_structured_output(
            RelevanceDecision
        )

    @staticmethod
    def _make_entry_key(entry: dict) -> str:
        return (
            f"{entry.get('id', '')}:"
            f"{entry.get('sense_id', '')}:"
            f"{entry.get('sense_number', '')}"
        )

    @staticmethod
    def _make_document_key(document: Any) -> str:
        metadata = document.metadata
        return (
            f"{metadata.get('entry_id', '')}:"
            f"{metadata.get('sense_id', '')}:"
            f"{metadata.get('sense_number', '')}"
        )

    @staticmethod
    def _extract_explicit_grammar_forms(question: str) -> list[str]:
        """Extract forms written in dictionary notation, such as '-는 반면에'."""
        pattern = (
            r"(?<!\S)-[가-힣ㄱ-ㅎㅏ-ㅣ()]+"
            r"(?:\s+[가-힣ㄱ-ㅎㅏ-ㅣ()]+)*"
        )
        return list(
            dict.fromkeys(match.strip() for match in re.findall(pattern, question))
        )

    def _resolve_explicit_grammar_forms(self, question: str) -> list[str]:
        """Trim Korean prose accidentally captured after a known grammar form."""
        resolved = []
        for candidate in self._extract_explicit_grammar_forms(question):
            if candidate in self.grammar_lookup:
                match = candidate
            else:
                known_prefixes = [
                    grammar
                    for grammar in self.grammar_lookup
                    if candidate.startswith(grammar)
                ]
                match = max(known_prefixes, key=len) if known_prefixes else candidate
            if match not in resolved:
                resolved.append(match)
        return resolved

    @staticmethod
    def _mentions_topik(question: str) -> bool:
        return "topik" in question.lower() or "토픽" in question

    @staticmethod
    def _wants_explanation(question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in EXPLANATION_MARKERS)

    def _create_fast_plan(self, question: str) -> RetrievalPlan | None:
        """Route explicit grammar requests without an LLM planning call."""
        explicit_forms = self._resolve_explicit_grammar_forms(question)
        if not explicit_forms:
            return None

        use_topik = self._mentions_topik(question)
        use_dictionary = not use_topik or self._wants_explanation(question)
        return RetrievalPlan(
            grammar_forms=explicit_forms if use_dictionary else [],
            use_dictionary=use_dictionary,
            use_topik=use_topik,
            topik_grammar_forms=explicit_forms if use_topik else [],
        )

    def _create_plan(self, question: str) -> RetrievalPlan:
        prompt = """
You plan retrieval for a Korean grammar assistant with two independent sources.

Korean Basic Dictionary is authoritative for definitions, meanings, functions,
usage rules, comparisons, and finding grammar that expresses a function.

TOPIK is authentic exam evidence only. Use it for literal occurrences of a
grammar form or semantic retrieval of questions about a topic. It is not an
authority for grammar facts.

Rules:
- Put semantic dictionary searches in search_queries.
- Put canonical, explicitly named forms in grammar_forms.
- Put Korean topic searches for TOPIK in topik_queries; translate an English
  topic into useful Korean search terms.
- Put literal TOPIK grammar searches in topik_grammar_forms.
- Do not create a TOPIK topic query for a literal grammar occurrence search.
- Enable both sources only when both are needed.
- Do not answer the user.
- Return at most 5 dictionary queries/forms, 3 TOPIK queries, and 5 TOPIK forms.

Examples:
"What grammar can I use to express a guess?"
  -> dictionary semantic retrieval
"Find TOPIK questions about environmental pollution."
  -> use_topik=true, use_dictionary=false,
     topik_queries=["환경 오염 수질 오염"]
""".strip()
        return self.planner.invoke([("system", prompt), ("user", question)])

    def _stabilize_plan(
        self, question: str, plan: RetrievalPlan
    ) -> RetrievalPlan:
        explicit_forms = self._resolve_explicit_grammar_forms(question)
        mentions_topik = self._mentions_topik(question)
        wants_explanation = self._wants_explanation(question)

        for grammar in explicit_forms:
            if grammar not in plan.grammar_forms:
                plan.grammar_forms.append(grammar)

        if mentions_topik:
            plan.use_topik = True
        if wants_explanation:
            plan.use_dictionary = True

        if wants_explanation:
            for grammar in plan.topik_grammar_forms:
                if grammar not in plan.grammar_forms:
                    plan.grammar_forms.append(grammar)

        if mentions_topik:
            for grammar in explicit_forms or plan.grammar_forms:
                if grammar not in plan.topik_grammar_forms:
                    plan.topik_grammar_forms.append(grammar)

        if plan.use_topik and not plan.topik_grammar_forms and not plan.topik_queries:
            plan.topik_queries = [question]
        return plan

    def _find_exact_grammar(self, grammar_forms: list[str]) -> list[dict]:
        results = []
        for grammar in dict.fromkeys(
            form.strip() for form in grammar_forms if form.strip()
        ):
            results.extend(
                {
                    "entry": entry,
                    "retrieval_type": "exact",
                    "distance": None,
                }
                for entry in self.grammar_lookup.get(grammar, [])
            )
        return results

    @staticmethod
    def _normalize_definition(value: str) -> str:
        return re.sub(r"[^a-z0-9가-힣]+", "", (value or "").lower())

    @staticmethod
    def _attachment_core(grammar: str) -> str:
        """Remove a leading Korean attachment allomorph for family comparison."""
        value = re.sub(r"\s+", " ", grammar.lstrip("-").strip())
        if " " in value:
            prefix, core = value.split(" ", 1)
            if prefix in {"ㄴ", "은", "는", "ㄹ", "을", "ㅁ", "음", "아", "어"}:
                return core.replace(" ", "")
        return re.sub(r"^(?:ㄴ|은|는|ㄹ|을|ㅁ|음|던)", "", value).replace(" ", "")

    def _family_forms(self, entry: dict) -> set[str]:
        """Find attachment variants supported by dictionary cross-references."""
        grammar = (entry.get("grammar") or "").strip()
        family = {grammar} if grammar else set()
        grammar_core = self._attachment_core(grammar)
        definitions = {
            self._normalize_definition(entry.get("definition_en", "")),
            self._normalize_definition(entry.get("definition_ko", "")),
        }
        definitions.discard("")

        for related in entry.get("related") or []:
            if not isinstance(related, dict):
                continue
            related_form = (related.get("grammar") or "").strip()
            if not related_form:
                continue
            related_entries = self.grammar_lookup.get(related_form, [])
            related_definitions = {
                self._normalize_definition(candidate.get(field, ""))
                for candidate in related_entries
                for field in ("definition_en", "definition_ko")
            }
            same_attachment_family = (
                grammar_core
                and grammar_core == self._attachment_core(related_form)
            )
            if definitions.intersection(related_definitions) or same_attachment_family:
                family.add(related_form)
        return family

    def _near_equivalent_family(self, left: dict, right: dict) -> bool:
        left_core = self._attachment_core(left.get("grammar", ""))
        right_core = self._attachment_core(right.get("grammar", ""))
        if not left_core or not right_core:
            return False
        if not (left_core.startswith(right_core) or right_core.startswith(left_core)):
            return False

        similarities = []
        for field in ("definition_en", "definition_ko"):
            left_definition = self._normalize_definition(left.get(field, ""))
            right_definition = self._normalize_definition(right.get(field, ""))
            if left_definition and right_definition:
                similarities.append(
                    SequenceMatcher(None, left_definition, right_definition).ratio()
                )
        return bool(similarities) and max(similarities) >= 0.65

    def _diversify_semantic_candidates(self, candidates: list[dict]) -> list[dict]:
        """Keep the best-ranked entry from each dictionary-backed family."""
        families: list[dict] = []
        for candidate in candidates:
            forms = self._family_forms(candidate["entry"])
            matching = next(
                (
                    family
                    for family in families
                    if family["forms"].intersection(forms)
                    or self._near_equivalent_family(
                        family["representative"]["entry"], candidate["entry"]
                    )
                ),
                None,
            )
            if matching is None:
                families.append({"representative": candidate, "forms": set(forms)})
            else:
                matching["forms"].update(forms)

        diversified = []
        for family in families[:MAX_DICTIONARY_CANDIDATES]:
            representative = dict(family["representative"])
            representative["family_variants"] = sorted(family["forms"])
            diversified.append(representative)
        return diversified

    def _retrieve_dictionary_candidates(self, plan: RetrievalPlan) -> list[dict]:
        if not plan.use_dictionary:
            return []

        combined = {
            self._make_entry_key(result["entry"]): result
            for result in self._find_exact_grammar(plan.grammar_forms)
        }
        matched_forms = {
            result["entry"].get("grammar") for result in combined.values()
        }

        # Exact forms do not need a redundant embedding search. Unmatched forms
        # still get a semantic fallback for notation or normalization variants.
        queries = [
            *plan.search_queries,
            *(form for form in plan.grammar_forms if form not in matched_forms),
        ]
        queries = list(
            dict.fromkeys(query.strip() for query in queries if query.strip())
        )

        open_ended = bool(plan.search_queries) and not plan.grammar_forms
        retrieval_k = OPEN_ENDED_RETRIEVAL_K if open_ended else RETRIEVAL_K
        for query in queries:
            for document, distance in self.vectorstore.similarity_search_with_score(
                "query: " + query, k=retrieval_k
            ):
                if distance > SEMANTIC_DISTANCE_THRESHOLD:
                    continue
                key = self._make_document_key(document)
                entry = self.entry_lookup.get(key)
                if entry is None:
                    continue

                existing = combined.get(key)
                if existing and existing["retrieval_type"] == "exact":
                    continue
                if existing is None or distance < existing["distance"]:
                    combined[key] = {
                        "entry": entry,
                        "retrieval_type": "semantic",
                        "distance": float(distance),
                    }

        results = list(combined.values())
        results.sort(
            key=lambda item: (
                item["retrieval_type"] != "exact",
                item["distance"] if item["distance"] is not None else 0,
            )
        )
        if open_ended:
            return self._diversify_semantic_candidates(results)
        return results[:MAX_DICTIONARY_CANDIDATES]

    @staticmethod
    def _format_validation_entry(number: int, result: dict) -> str:
        entry = result["entry"]
        return f"""
ENTRY {number}
Grammar: {entry.get("grammar", "")}
English definition: {entry.get("definition_en", "")}
Korean definition: {entry.get("definition_ko", "")}
Usage: {entry.get("usage", "")}
Related: {entry.get("related", "")}
""".strip()

    def _validate_dictionary_relevance(
        self, question: str, candidates: list[dict]
    ) -> list[dict]:
        if not candidates:
            return []

        # An exact dictionary match is deterministic and must not be discarded
        # by a relevance model.
        exact_results = [
            item for item in candidates if item.get("retrieval_type") == "exact"
        ]
        semantic_results = [
            item for item in candidates if item.get("retrieval_type") != "exact"
        ]
        if not semantic_results:
            return exact_results

        evidence = "\n\n".join(
            self._format_validation_entry(index, candidate)
            for index, candidate in enumerate(semantic_results, start=1)
        )
        prompt = """
Validate retrieved Korean Basic Dictionary evidence using only the records
provided. Exact requested entries have already been retained separately.
Select only additional semantic entries that materially help answer the user.
Superficial word overlap is insufficient, related forms may be selected when
useful, and selecting zero records is valid. Entry numbers are 1-based.
""".strip()
        decision = self.relevance_validator.invoke(
            [
                ("system", prompt),
                (
                    "user",
                    f"QUESTION:\n{question}\n\nADDITIONAL DICTIONARY EVIDENCE:\n{evidence}",
                ),
            ]
        )
        if not decision.answerable:
            return exact_results

        valid_indexes = {
            number - 1
            for number in decision.relevant_entry_numbers
            if 1 <= number <= len(semantic_results)
        }
        return exact_results + [
            candidate
            for index, candidate in enumerate(semantic_results)
            if index in valid_indexes
        ]

    @staticmethod
    def _topik_result_key(item: dict) -> tuple:
        return (
            item.get("exam_number"),
            item.get("question_start"),
            item.get("question_end"),
            item.get("source_file"),
        )

    def _dedupe_topik_results(self, results: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for item in results:
            key = self._topik_result_key(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _retrieve_topik(
        self, plan: RetrievalPlan, dictionary_results: list[dict]
    ) -> list[dict]:
        if not plan.use_topik:
            return []

        combined = {}
        grammar_forms = list(
            dict.fromkeys(
                grammar.strip()
                for grammar in plan.topik_grammar_forms
                if grammar.strip()
            )
        )

        # A functional grammar query may use validated dictionary hits to turn
        # the request into concrete lexical occurrence searches.
        if not grammar_forms:
            for result in dictionary_results[:3]:
                grammar = (result["entry"].get("grammar") or "").strip()
                if grammar and grammar not in grammar_forms:
                    grammar_forms.append(grammar)

        # Literal matching is stronger and cheaper for a named grammar form.
        for grammar_form in grammar_forms:
            for item in self.topik.lexical_search(
                grammar_form, k=MAX_TOPIK_RESULTS
            ):
                combined[self._topik_result_key(item)] = item

        # Semantic TOPIK retrieval is reserved for topic searches.
        if not grammar_forms:
            queries = list(
                dict.fromkeys(
                    query.strip() for query in plan.topik_queries if query.strip()
                )
            )
            for query in queries:
                for item in self.topik.semantic_search(query, k=MAX_TOPIK_RESULTS):
                    key = self._topik_result_key(item)
                    existing = combined.get(key)
                    if existing and existing.get("retrieval_type") == "lexical":
                        continue
                    combined[key] = item

        results = self._dedupe_topik_results(list(combined.values()))
        results.sort(
            key=lambda item: (
                item.get("retrieval_type") != "lexical",
                item.get("distance", 999),
            )
        )
        return results[:MAX_TOPIK_RESULTS]

    @staticmethod
    def _normalize_pages(pages: Any) -> list[int]:
        if isinstance(pages, list):
            values = pages
        elif isinstance(pages, str):
            values = pages.split(",")
        elif isinstance(pages, int):
            values = [pages]
        else:
            return []

        result = []
        for value in values:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page not in result:
                result.append(page)
        return result

    @staticmethod
    def _serialize_entry(result: dict) -> dict:
        entry = dict(result["entry"])
        entry["retrieval_type"] = result["retrieval_type"]
        entry["distance"] = result["distance"]
        if result.get("family_variants"):
            entry["family_variants"] = result["family_variants"]
        return entry

    def _serialize_topik(self, result: dict) -> dict:
        item = dict(result)
        item["pages"] = self._normalize_pages(item.get("pages"))
        return item

    def _build_sources(
        self, dictionary_results: list[dict], topik_results: list[dict]
    ) -> list[dict]:
        sources = []
        seen_dictionary = set()
        for result in dictionary_results:
            entry = result["entry"]
            key = self._make_entry_key(entry)
            if key in seen_dictionary:
                continue
            seen_dictionary.add(key)
            sources.append(
                {
                    "source_type": "dictionary",
                    "name": "Korean Basic Dictionary",
                    "grammar": entry.get("grammar"),
                    "entry_id": entry.get("id"),
                    "sense_id": entry.get("sense_id"),
                }
            )

        seen_topik = set()
        for result in topik_results:
            key = self._topik_result_key(result)
            if key in seen_topik:
                continue
            seen_topik.add(key)
            sources.append(
                {
                    "source_type": "topik",
                    "name": f"TOPIK {result['exam_number']}",
                    "exam_number": result["exam_number"],
                    "question_start": result["question_start"],
                    "question_end": result["question_end"],
                    "pages": self._normalize_pages(result.get("pages")),
                    "source_file": result["source_file"],
                }
            )
        return sources

    @staticmethod
    def _format_dictionary_evidence(result: dict) -> str:
        entry = result["entry"]
        examples = entry.get("examples", [])
        if isinstance(examples, list):
            examples_text = "\n".join(f"- {example}" for example in examples)
        else:
            examples_text = str(examples)
        related = entry.get("related", [])
        if isinstance(related, list):
            related_text = ", ".join(
                item.get("grammar", "")
                for item in related
                if isinstance(item, dict) and item.get("grammar")
            )
        else:
            related_text = str(related)
        return f"""
GRAMMAR: {entry.get("grammar", "")}
English definition: {entry.get("definition_en", "")}
Korean definition: {entry.get("definition_ko", "")}
Usage: {entry.get("usage", "")}
Retrieved examples:
{examples_text}
Related grammar: {related_text}
""".strip()

    def _generate_explanation(
        self,
        question: str,
        dictionary_results: list[dict],
    ) -> str:
        dictionary_evidence = "\n\n---\n\n".join(
            self._format_dictionary_evidence(result)
            for result in dictionary_results
        )
        prompt = f"""
You are a Korean grammar assistant. Explain grammar using only the Korean Basic
Dictionary evidence below; do not use memorized grammar knowledge or invent
examples. You may paraphrase the records in clear English and translate their
examples when useful.

Answer only the grammar-explanation part of the user's request. Do not mention
TOPIK, exam questions, retrieval results, evidence availability, result counts,
or content displayed elsewhere in the interface.

If the dictionary evidence is insufficient, respond with:
"{NO_ANSWER}"

Do not add a Sources section. Answer primarily in English unless another
language was requested.

USER QUESTION:
{question}

KOREAN BASIC DICTIONARY EVIDENCE:
{dictionary_evidence or "[none]"}
""".strip()
        return self.llm.invoke(prompt).content

    def ask(self, question: str, explain: bool = True) -> dict:
        started = perf_counter()
        timings: dict[str, float] = {}

        step = perf_counter()
        plan = self._create_fast_plan(question)
        route = "deterministic"
        if plan is None:
            route = "llm"
            plan = self._stabilize_plan(question, self._create_plan(question))
        timings["planning"] = perf_counter() - step

        step = perf_counter()
        dictionary_candidates = self._retrieve_dictionary_candidates(plan)
        timings["dictionary_retrieval"] = perf_counter() - step

        step = perf_counter()
        dictionary_results = self._validate_dictionary_relevance(
            question, dictionary_candidates
        )
        timings["dictionary_validation"] = perf_counter() - step

        step = perf_counter()
        topik_results = self._dedupe_topik_results(
            self._retrieve_topik(plan, dictionary_results)
        )
        topik_message = (
            NO_TOPIK_RESULTS if plan.use_topik and not topik_results else None
        )
        timings["topik_retrieval"] = perf_counter() - step
        timings["generation"] = 0.0

        if not dictionary_results and not topik_results:
            result = {
                "question": question,
                "explanation": NO_ANSWER if explain and plan.use_dictionary else None,
                "entries": [],
                "topik_occurrences": [],
                "topik_message": topik_message,
                "sources": [],
            }
        else:
            entries = [self._serialize_entry(item) for item in dictionary_results]
            topik_occurrences = [
                self._serialize_topik(item) for item in topik_results
            ]
            explanation = None
            step = perf_counter()
            if explain and dictionary_results:
                explanation = self._generate_explanation(
                    question, dictionary_results
                )
            timings["generation"] = perf_counter() - step
            result = {
                "question": question,
                "explanation": explanation,
                "entries": entries,
                "topik_occurrences": topik_occurrences,
                "topik_message": topik_message,
                "sources": self._build_sources(dictionary_results, topik_results),
            }

        timings["total"] = perf_counter() - started
        logger.debug(
            "RAG timings route=%s %s",
            route,
            " ".join(
                f"{name}={seconds:.3f}s" for name, seconds in timings.items()
            ),
        )
        return result
