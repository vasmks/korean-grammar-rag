import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data/topik/processed/topik_retrieval_units.json"
CHROMA_DIR = PROJECT_ROOT / "data/chroma_topik"
COLLECTION_NAME = "topik_examples"

DEFAULT_K = 5
DEFAULT_DISTANCE_THRESHOLD = 0.40


class TopikRetriever:
    def __init__(self, embeddings: Any) -> None:
        with DATA_FILE.open(encoding="utf-8") as file:
            units = json.load(file)

        self.units = [
            unit
            for unit in units
            if unit.get(
                "indexable",
                unit.get("content_type") != "visual_or_sparse",
            )
        ]
        # PDF text normalization is deterministic, so do it once at startup.
        self._normalized_units = [
            (unit, self._normalize_text(unit.get("text", "")))
            for unit in self.units
        ]
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))

    @classmethod
    def _normalize_grammar_form(cls, grammar_form: str) -> str:
        return cls._normalize_text(grammar_form.strip().lstrip("-"))

    def lexical_search(
        self, grammar_form: str, k: int = DEFAULT_K
    ) -> list[dict[str, Any]]:
        target = self._normalize_grammar_form(grammar_form)
        if not target:
            return []

        results = []
        for unit, normalized_text in self._normalized_units:
            if target not in normalized_text:
                continue
            results.append(
                {
                    "retrieval_type": "lexical",
                    "match_type": "candidate_occurrence",
                    "grammar_query": grammar_form,
                    "match_count": normalized_text.count(target),
                    "exam_number": unit["exam_number"],
                    "section": unit["section"],
                    "question_start": unit["question_start"],
                    "question_end": unit["question_end"],
                    "question_numbers": unit["question_numbers"],
                    "pages": unit["pages"],
                    "source_file": unit["source_file"],
                    "chunk_type": unit["chunk_type"],
                    "text": unit["text"],
                }
            )

        results.sort(
            key=lambda item: (
                -item["match_count"],
                item["exam_number"],
                item["question_start"],
            )
        )
        return results[:k]

    def semantic_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        raw_results = self.vectorstore.similarity_search_with_score(
            "query: " + query, k=k
        )
        results = []
        for document, distance in raw_results:
            if distance > distance_threshold:
                continue
            metadata = document.metadata
            text = document.page_content
            if text.startswith("passage:"):
                text = text[len("passage:") :].strip()
            results.append(
                {
                    "retrieval_type": "semantic",
                    "distance": float(distance),
                    "exam_number": metadata["exam_number"],
                    "section": metadata["section"],
                    "question_start": metadata["question_start"],
                    "question_end": metadata["question_end"],
                    "pages": metadata["pages"],
                    "source_file": metadata["source_file"],
                    "chunk_type": metadata["chunk_type"],
                    "text": text,
                }
            )
        return results
