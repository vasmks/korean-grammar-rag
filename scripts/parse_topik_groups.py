"""Parse extracted TOPIK reading lines into coordinate-bearing instruction scopes."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from scripts.topik_ranges import parse_instruction_range
except ModuleNotFoundError:  # Support direct script execution
    from topik_ranges import parse_instruction_range


INPUT_FILE = Path("data/topik/processed/topik_reading_pages.json")
OUTPUT_FILE = Path("data/topik/processed/topik_reading_groups.json")

QUESTION_PREFIX_RE = re.compile(r"^\s*([0-9Oo]{1,2})\s*[.．]")
QUESTION_ONLY_RE = re.compile(r"^\s*([0-9Oo]{1,2})\s*$")


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def normalize_question_token(token: str) -> int | None:
    """Normalize a narrowly scoped OCR number token such as ``3O.``."""
    normalized = token.translate(str.maketrans({"O": "0", "o": "0"}))
    if not normalized.isdigit():
        return None
    number = int(normalized)
    return number if 1 <= number <= 50 else None


def question_anchor(text: str) -> int | None:
    """Return a leading question number without interpreting numbers in prose."""
    match = QUESTION_PREFIX_RE.match(text) or QUESTION_ONLY_RE.match(text)
    if match is None:
        return None
    return normalize_question_token(match.group(1))


def is_noise_block(block: dict) -> bool:
    value = compact_text(block.get("text", ""))
    if not value:
        return True

    y0 = float(block["bbox"][1])
    if re.fullmatch(r"\d{1,3}", value) and y0 > 760:
        return True
    if "TOPIKⅡ읽기" in value or "TOPIKII읽기" in value:
        return True
    if "한국어능력시험" in value and "읽기" in value and "교시" in value:
        return True
    return False


def flatten_blocks(records: list[dict]) -> list[dict]:
    """Give native and OCR lines the same ordered, coordinate-bearing shape."""
    blocks: list[dict] = []
    for record in sorted(records, key=lambda item: int(item["page"])):
        page = int(record["page"])
        method = record.get("extraction_method", "unknown")
        page_lines = sorted(
            record.get("lines") or [],
            key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])),
        )
        for line in page_lines:
            bbox = line.get("bbox")
            text = str(line.get("text", "")).strip()
            if not text or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            block = {
                "page": page,
                "text": text,
                "bbox": [round(float(value), 2) for value in bbox],
                "extraction_source": method,
                "confidence": line.get("confidence"),
            }
            if not is_noise_block(block):
                blocks.append(block)

    for order, block in enumerate(blocks):
        block["order"] = order
    return blocks


def same_row(left: dict, right: dict, tolerance: float = 7.0) -> bool:
    if left["page"] != right["page"]:
        return False
    left_middle = (left["bbox"][1] + left["bbox"][3]) / 2
    right_middle = (right["bbox"][1] + right["bbox"][3]) / 2
    return abs(left_middle - right_middle) <= tolerance


def locate_instruction_headers(blocks: list[dict]) -> list[dict]:
    headers: list[dict] = []
    seen_ranges: set[tuple[int, int]] = set()
    for index, block in enumerate(blocks):
        question_range = parse_instruction_range(block["text"])
        if question_range is None:
            continue
        question_start, question_end = question_range

        # EasyOCR can return the instruction and its [x~y] suffix as separate
        # boxes. Pull all same-row boxes into the instruction scope header.
        row_indices = [index]
        cursor = index - 1
        while cursor >= 0 and same_row(blocks[cursor], block):
            row_indices.append(cursor)
            cursor -= 1
        cursor = index + 1
        while cursor < len(blocks) and same_row(blocks[cursor], block):
            row_indices.append(cursor)
            cursor += 1

        range_key = (question_start, question_end)
        if range_key in seen_ranges:
            continue
        seen_ranges.add(range_key)
        headers.append(
            {
                "question_start": question_start,
                "question_end": question_end,
                "header_index": index,
                "scope_start_index": min(row_indices),
                "row_indices": sorted(row_indices),
            }
        )

    return sorted(headers, key=lambda item: item["scope_start_index"])


def join_blocks(blocks: list[dict]) -> str:
    return "\n".join(block["text"].strip() for block in blocks if block["text"].strip())


def has_shared_context_instruction(instruction: str) -> bool:
    compact = compact_text(instruction)
    return "물음에답하십시오" in compact or "물음에답하시오" in compact


def classify_content(question_start: int, question_end: int, text: str) -> str:
    """Classify short visual or sparse question groups for indexing."""
    if question_start >= 5 and question_end <= 12:
        if len(re.findall(r"[가-힣]", text)) < 250:
            return "visual_or_sparse"
    return "text"


def parse_exam(records: list[dict]) -> list[dict]:
    blocks = flatten_blocks(records)
    headers = locate_instruction_headers(blocks)
    if not records or not headers:
        return []

    first_record = records[0]
    exam_number = int(first_record["exam_number"])
    source_file = first_record["source_file"]
    scopes: list[dict] = []

    for header_position, header in enumerate(headers):
        scope_end = (
            headers[header_position + 1]["scope_start_index"]
            if header_position + 1 < len(headers)
            else len(blocks)
        )
        scope_blocks = blocks[header["scope_start_index"] : scope_end]
        next_scope_block = blocks[scope_end] if scope_end < len(blocks) else None
        last_scope_block = scope_blocks[-1]
        if (
            next_scope_block is not None
            and next_scope_block["page"] == last_scope_block["page"]
        ):
            scope_end_position = {
                "page": next_scope_block["page"],
                "y": next_scope_block["bbox"][1],
            }
        else:
            scope_end_position = {
                "page": last_scope_block["page"],
                "y": last_scope_block["bbox"][3] + 10,
            }
        header_orders = {
            blocks[index]["order"] for index in header["row_indices"]
        }
        instruction_blocks = [
            block for block in scope_blocks if block["order"] in header_orders
        ]
        instruction_blocks.sort(key=lambda item: (item["page"], item["bbox"][0]))
        instruction = " ".join(block["text"] for block in instruction_blocks).strip()

        expected = list(
            range(header["question_start"], header["question_end"] + 1)
        )
        anchors: dict[int, dict] = {}
        for block in scope_blocks:
            number = question_anchor(block["text"])
            if number in expected and number not in anchors:
                anchors[number] = {
                    "question_number": number,
                    "page": block["page"],
                    "bbox": block["bbox"],
                    "block_order": block["order"],
                    "text": block["text"],
                }

        first_anchor_order = min(
            (item["block_order"] for item in anchors.values()),
            default=None,
        )
        context_blocks = [
            block
            for block in scope_blocks
            if block["order"] not in header_orders
            and (first_anchor_order is None or block["order"] < first_anchor_order)
        ]
        shared_by_instruction = has_shared_context_instruction(instruction)
        shared_by_layout = bool(context_blocks)
        context_text = join_blocks(context_blocks) if shared_by_instruction else ""
        missing_anchors = sorted(set(expected) - set(anchors))
        structure_confidence = "high"
        if missing_anchors or (shared_by_instruction and not context_text):
            structure_confidence = "low"

        text = join_blocks(scope_blocks)
        instruction_id = (
            f"topik_{exam_number}_instruction_"
            f"{header['question_start']}_{header['question_end']}"
        )
        scopes.append(
            {
                "source_type": "topik",
                "chunk_type": "instruction_scope",
                "exam_number": exam_number,
                "section": "reading",
                "instruction_id": instruction_id,
                "instruction_range": [
                    header["question_start"],
                    header["question_end"],
                ],
                "question_start": header["question_start"],
                "question_end": header["question_end"],
                "question_numbers": expected,
                "pages": sorted({block["page"] for block in scope_blocks}),
                "source_file": source_file,
                "scope_start": {
                    "page": scope_blocks[0]["page"],
                    "y": scope_blocks[0]["bbox"][1],
                },
                "scope_end": scope_end_position,
                "instruction": instruction,
                "structure_type": (
                    "shared_context" if shared_by_instruction else "independent"
                ),
                "structure_signals": {
                    "shared_instruction_language": shared_by_instruction,
                    "pre_question_content": shared_by_layout,
                    "pre_question_block_count": len(context_blocks),
                },
                "structure_confidence": structure_confidence,
                "missing_question_anchors": missing_anchors,
                "question_anchors": [anchors[number] for number in sorted(anchors)],
                "shared_context_text": context_text,
                "shared_context_blocks": context_blocks if shared_by_instruction else [],
                "blocks": scope_blocks,
                "extraction_sources": sorted(
                    {block["extraction_source"] for block in scope_blocks}
                ),
                "content_type": classify_content(
                    header["question_start"], header["question_end"], text
                ),
                "text": text,
            }
        )

    return scopes


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as file:
        page_records = json.load(file)

    exams: defaultdict[int, list[dict]] = defaultdict(list)
    for record in page_records:
        exams[int(record["exam_number"])].append(record)

    all_scopes: list[dict] = []
    for exam_number in sorted(exams):
        scopes = parse_exam(exams[exam_number])
        all_scopes.extend(scopes)
        shared = sum(scope["structure_type"] == "shared_context" for scope in scopes)
        low = sum(scope["structure_confidence"] == "low" for scope in scopes)
        print(
            f"TOPIK {exam_number}: {len(scopes)} instruction scopes "
            f"({shared} shared-context, {low} low-confidence)"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(all_scopes, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_scopes)} instruction scopes to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
