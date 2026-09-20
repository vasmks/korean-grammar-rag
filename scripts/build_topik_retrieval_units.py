"""Build one TOPIK retrieval unit per logical question."""

from __future__ import annotations

import json
from pathlib import Path


INPUT_FILE = Path("data/topik/processed/topik_reading_groups.json")
OUTPUT_FILE = Path("data/topik/processed/topik_retrieval_units.json")
CONTEXTS_FILE = Path("data/topik/processed/topik_shared_contexts.json")


def join_blocks(blocks: list[dict]) -> str:
    return "\n".join(block["text"].strip() for block in blocks if block["text"].strip())


def position(item: dict) -> tuple[int, float]:
    return int(item["page"]), float(item["y"])


def anchor_position(anchor: dict) -> tuple[int, float]:
    return int(anchor["page"]), float(anchor["bbox"][1])


def preview_regions(
    start: tuple[int, float], end: tuple[int, float]
) -> list[dict]:
    """Represent a vertical PDF span without rasterizing it."""
    start_page, start_y = start
    end_page, end_y = end
    if end_page < start_page or (end_page == start_page and end_y <= start_y):
        end_page, end_y = start_page, start_y + 24

    return [
        {
            "page_number": page,
            "start_y": round(start_y if page == start_page else 0.0, 2),
            "end_y": round(end_y, 2) if page == end_page else None,
        }
        for page in range(start_page, end_page + 1)
    ]


def question_start_order(blocks: list[dict], anchor: dict) -> int:
    """Include same-row stem blocks that sort immediately before an anchor."""
    start_order = int(anchor["block_order"])
    anchor_y = float(anchor["bbox"][1])
    anchor_page = int(anchor["page"])
    for block in reversed(blocks):
        if int(block["order"]) >= start_order:
            continue
        if int(block["page"]) != anchor_page:
            break
        if float(block["bbox"][3]) >= anchor_y - 6:
            start_order = int(block["order"])
            continue
        break
    return start_order


def _layout_gap_candidates(
    blocks: list[dict], start_order: int, end_order: int
) -> list[tuple[float, int]]:
    segment = [
        block
        for block in blocks
        if start_order <= int(block["order"]) < end_order
    ]
    candidates: list[tuple[float, int]] = []
    for current, following in zip(segment, segment[1:]):
        following_order = int(following["order"])
        if following_order >= end_order:
            continue
        if int(current["page"]) != int(following["page"]):
            candidates.append((1000.0, following_order))
            continue
        gap = float(following["bbox"][1]) - float(current["bbox"][3])
        if gap >= 18.0:
            candidates.append((gap, following_order))
    return candidates


def _select_gap_orders(
    candidates: list[tuple[float, int]], boundary_count: int
) -> list[int]:
    """Return ordered, clearly dominant layout cuts or no inference."""
    if boundary_count <= 0 or len(candidates) < boundary_count:
        return []
    ranked = sorted(candidates, reverse=True)
    selected = ranked[:boundary_count]
    remaining = ranked[boundary_count:]
    weakest_selected = selected[-1][0]
    strongest_remaining = remaining[0][0] if remaining else 0.0
    if strongest_remaining and weakest_selected < strongest_remaining * 1.45:
        return []
    return sorted(order for _, order in selected)


def infer_missing_question_starts(
    scope: dict,
    expected: list[int],
    confirmed_starts: dict[int, int],
) -> dict[int, int]:
    """Infer clearly separated missing runs, including shared-scope edges."""
    inferred: dict[int, int] = {}
    index = 0
    while index < len(expected):
        if expected[index] in confirmed_starts:
            index += 1
            continue

        run_start = index
        while index < len(expected) and expected[index] not in confirmed_starts:
            index += 1
        missing = expected[run_start:index]

        if run_start > 0 and index < len(expected):
            previous_number = expected[run_start - 1]
            following_number = expected[index]
            candidates = _layout_gap_candidates(
                scope["blocks"],
                confirmed_starts[previous_number],
                confirmed_starts[following_number],
            )
        elif (
            run_start == 0
            and index < len(expected)
            and scope.get("structure_type") == "shared_context"
            and scope.get("shared_context_blocks")
        ):
            # The parser conservatively puts all pre-anchor blocks into shared
            # context. A clear gap can separate the actual stimulus from one
            # or more missing leading sibling questions.
            first_context_order = min(
                int(block["order"])
                for block in scope["shared_context_blocks"]
            )
            candidates = _layout_gap_candidates(
                scope["blocks"],
                first_context_order,
                confirmed_starts[expected[index]],
            )
        elif (
            run_start > 0
            and index == len(expected)
            and scope.get("structure_type") == "shared_context"
        ):
            # At the trailing edge the scope boundary supplies the outer bound
            # that a following sibling anchor would normally provide.
            final_order = max(
                int(block["order"]) for block in scope["blocks"]
            ) + 1
            candidates = _layout_gap_candidates(
                scope["blocks"],
                confirmed_starts[expected[run_start - 1]],
                final_order,
            )
        else:
            continue

        selected_orders = _select_gap_orders(candidates, len(missing))
        if selected_orders:
            inferred.update(zip(missing, selected_orders))
    return inferred


def question_blocks(
    scope: dict,
    start_order: int | None,
    end_order: int | None,
) -> list[dict]:
    blocks = scope["blocks"]
    if start_order is None:
        # A missing OCR anchor is uncertain. Preserve the complete source scope
        # so required text or visual context is never silently discarded.
        return blocks
    context_orders = {
        int(block["order"]) for block in scope.get("shared_context_blocks", [])
    }
    return [
        block
        for block in blocks
        if int(block["order"]) >= start_order
        and (end_order is None or int(block["order"]) < end_order)
        and int(block["order"]) not in context_orders
    ]


def make_shared_context(
    scope: dict, context_blocks: list[dict] | None = None
) -> dict | None:
    if scope.get("structure_type") != "shared_context":
        return None
    context_id = scope["instruction_id"].replace("instruction", "context")
    blocks = (
        context_blocks
        if context_blocks is not None
        else scope.get("shared_context_blocks") or []
    )
    return {
        "shared_context_id": context_id,
        "source_type": "topik",
        "exam_number": scope["exam_number"],
        "section": scope["section"],
        "instruction_id": scope["instruction_id"],
        "question_numbers": scope["question_numbers"],
        "pages": sorted({block["page"] for block in blocks}),
        "source_file": scope["source_file"],
        "extraction_sources": sorted(
            {block["extraction_source"] for block in blocks}
        ),
        "structure_confidence": scope["structure_confidence"],
        "text": join_blocks(blocks),
        "blocks": blocks,
    }


def build_units(scopes: list[dict]) -> tuple[list[dict], list[dict]]:
    units: list[dict] = []
    contexts: list[dict] = []

    for scope in scopes:
        anchors = {
            int(anchor["question_number"]): anchor
            for anchor in scope.get("question_anchors", [])
        }
        expected = [int(number) for number in scope["question_numbers"]]
        confirmed_starts = {
            number: question_start_order(scope["blocks"], anchor)
            for number, anchor in anchors.items()
        }
        inferred_starts = infer_missing_question_starts(
            scope, expected, confirmed_starts
        )
        question_starts = {**confirmed_starts, **inferred_starts}
        context_blocks = scope.get("shared_context_blocks") or []
        if expected and expected[0] in inferred_starts:
            first_question_order = inferred_starts[expected[0]]
            context_blocks = [
                block
                for block in context_blocks
                if int(block["order"]) < first_question_order
            ]
        context = make_shared_context(scope, context_blocks)
        if context is not None:
            contexts.append(context)
        context_id = context["shared_context_id"] if context else None
        context_text = context["text"] if context else ""
        known_after = {
            number: next(
                (
                    anchors[candidate]
                    for candidate in expected
                    if candidate > number and candidate in anchors
                ),
                None,
            )
            for number in expected
        }

        scope_start = position(scope["scope_start"])
        scope_end = position(scope["scope_end"])
        for number in expected:
            anchor = anchors.get(number)
            next_anchor = known_after[number]
            next_start_order = next(
                (
                    question_starts[candidate]
                    for candidate in expected
                    if candidate > number and candidate in question_starts
                ),
                None,
            )
            selected_blocks = question_blocks(
                {**scope, "shared_context_blocks": context_blocks},
                question_starts.get(number),
                next_start_order,
            )
            question_text = join_blocks(selected_blocks)
            next_expected = number + 1 if number < expected[-1] else None
            uncertain_boundary = (
                next_expected is not None and next_expected not in anchors
            )
            confidence = (
                "low"
                if anchor is None
                or uncertain_boundary
                or (
                    scope.get("structure_type") == "shared_context"
                    and not context_text
                )
                else "high"
            )

            if scope.get("structure_type") == "shared_context":
                preview_start = scope_start
            elif anchor is not None:
                anchor_page, anchor_y = anchor_position(anchor)
                preview_start = (anchor_page, max(0.0, anchor_y - 6))
            else:
                preview_start = scope_start

            preview_end = (
                anchor_position(next_anchor) if next_anchor is not None else scope_end
            )
            regions = preview_regions(preview_start, preview_end)
            parts = [scope.get("instruction", "")]
            if context_text:
                parts.append(context_text)
            parts.append(question_text)
            retrieval_text = "\n".join(part.strip() for part in parts if part.strip())

            units.append(
                {
                    "source_type": "topik",
                    "chunk_type": "question",
                    "exam_number": scope["exam_number"],
                    "section": scope["section"],
                    "question_numbers": [number],
                    "question_start": number,
                    "question_end": number,
                    "instruction_id": scope["instruction_id"],
                    "instruction_range": scope["instruction_range"],
                    "instruction": scope.get("instruction", ""),
                    "shared_context_id": context_id,
                    "shared_context_text": context_text,
                    "question_text": question_text,
                    "pages": sorted({region["page_number"] for region in regions}),
                    "source_file": scope["source_file"],
                    "extraction_sources": scope.get("extraction_sources", []),
                    "content_type": scope["content_type"],
                    "indexable": scope["content_type"] != "visual_or_sparse",
                    "structure_confidence": confidence,
                    "preview_regions": regions,
                    "text": retrieval_text,
                }
            )

    return units, contexts


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as file:
        scopes = json.load(file)

    units, contexts = build_units(scopes)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(units, file, ensure_ascii=False, indent=2)
    with CONTEXTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(contexts, file, ensure_ascii=False, indent=2)

    print(f"Instruction scopes : {len(scopes)}")
    print(f"Shared contexts    : {len(contexts)}")
    print(f"Question units     : {len(units)}")
    print(f"Indexable          : {sum(unit['indexable'] for unit in units)}")
    print(f"Visual/sparse      : {sum(not unit['indexable'] for unit in units)}")
    print(
        "Low confidence     : "
        f"{sum(unit['structure_confidence'] == 'low' for unit in units)}"
    )
    print(f"Saved to {OUTPUT_FILE} and {CONTEXTS_FILE}")


if __name__ == "__main__":
    main()
