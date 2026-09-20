"""Run a small behavior-focused retrieval evaluation.

Semantic cases call the configured OpenAI model for planning and validation.
Answers are disabled so this script evaluates retrieval rather than prose.
"""

import sys
from dataclasses import dataclass
from time import perf_counter

from app.rag import KoreanGrammarRAG


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    needs_dictionary: bool
    needs_topik: bool


CASES = (
    EvaluationCase("What does -은 듯 mean?", True, False),
    EvaluationCase("What does -는 반면에 mean?", True, False),
    EvaluationCase("Show me TOPIK occurrences of -는 반면에.", False, True),
    EvaluationCase(
        "Explain -는 반면에 and show me TOPIK occurrences.", True, True
    ),
    EvaluationCase(
        "Find TOPIK questions about environmental pollution.", False, True
    ),
    EvaluationCase("What grammar can I use to express a guess?", True, False),
    EvaluationCase("Who is the president of France?", False, False),
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rag = KoreanGrammarRAG()
    passed = 0

    for case in CASES:
        started = perf_counter()
        result = rag.ask(case.question, explain=False)
        elapsed = perf_counter() - started
        has_dictionary = bool(result["entries"])
        has_topik = bool(result["topik_occurrences"])
        ok = (
            has_dictionary == case.needs_dictionary
            and has_topik == case.needs_topik
        )
        if case.question == "What does -은 듯 mean?":
            ok = ok and any(
                entry.get("grammar") == "-은 듯" for entry in result["entries"]
            )
        if case.question == "What grammar can I use to express a guess?":
            family_keys = {
                tuple(entry.get("family_variants") or [entry.get("grammar")])
                for entry in result["entries"]
            }
            ok = ok and len(family_keys) >= 3
        passed += ok
        print(
            f"{'PASS' if ok else 'FAIL'} {elapsed:6.2f}s | "
            f"dictionary={len(result['entries'])} "
            f"topik={len(result['topik_occurrences'])} | {case.question}"
        )
        if result["entries"]:
            print(
                "  grammar: "
                + ", ".join(entry.get("grammar", "") for entry in result["entries"])
            )

    print(f"\nPassed {passed}/{len(CASES)} cases.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
