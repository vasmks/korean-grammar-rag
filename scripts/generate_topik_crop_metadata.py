import json
import re
from pathlib import Path

import pymupdf

try:
    from scripts.build_topik_retrieval_units import (
        infer_missing_question_starts,
        question_start_order,
    )
except ModuleNotFoundError:  # Support: python scripts/generate_topik_crop_metadata.py
    from build_topik_retrieval_units import (
        infer_missing_question_starts,
        question_start_order,
    )


UNITS_FILE = Path(
    "data/topik/processed/topik_retrieval_units.json"
)

PDF_DIR = Path(
    "data/topik/raw"
)

OUTPUT_FILE = Path(
    "data/topik/processed/topik_crop_metadata.json"
)
PAGES_FILE = Path(
    "data/topik/processed/topik_reading_pages.json"
)
GROUPS_FILE = Path(
    "data/topik/processed/topik_reading_groups.json"
)

# Populated only for OCR pages. Native PDFs continue to use PyMuPDF lines.
OCR_PAGE_LINES: dict[tuple[str, int], list[dict]] = {}


# Crop configuration

HORIZONTAL_PADDING = 16
TOP_PADDING = 8
BOTTOM_PADDING = 10
MIN_INFERRED_VERTICAL_GAP = 24
MAX_VISUAL_CUT_INK_RATIO = 0.02


# General helpers


def make_key(
    exam_number: int,
    question_start: int,
    question_end: int,
) -> str:

    return (
        f"{exam_number}:"
        f"{question_start}:"
        f"{question_end}"
    )


def get_pages(
    unit: dict,
) -> list[int]:

    pages = unit.get(
        "pages",
        [],
    )

    if isinstance(
        pages,
        list,
    ):
        return [
            int(page)
            for page in pages
        ]

    if isinstance(
        pages,
        str,
    ):
        return [
            int(value.strip())
            for value in pages.split(",")
            if value.strip()
        ]

    if isinstance(
        pages,
        int,
    ):
        return [pages]

    return []


def compact_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        "",
        text,
    )


# PDF text-line extraction


def get_lines(
    page,
) -> list[dict]:

    source_file = Path(page.parent.name).name
    cached_lines = OCR_PAGE_LINES.get(
        (source_file, page.number + 1)
    )
    if cached_lines is not None:
        return [
            {
                "text": item["text"],
                "rect": pymupdf.Rect(*item["bbox"]),
            }
            for item in cached_lines
            if item.get("text") and len(item.get("bbox", [])) == 4
        ]

    page_dict = page.get_text(
        "dict"
    )

    results = []

    for block in page_dict.get(
        "blocks",
        [],
    ):

        if block.get(
            "type"
        ) != 0:
            continue

        for line in block.get(
            "lines",
            [],
        ):

            spans = line.get(
                "spans",
                [],
            )

            if not spans:
                continue

            text = "".join(
                span.get(
                    "text",
                    "",
                )
                for span in spans
            ).strip()

            if not text:
                continue

            x0 = min(
                span["bbox"][0]
                for span in spans
            )

            y0 = min(
                span["bbox"][1]
                for span in spans
            )

            x1 = max(
                span["bbox"][2]
                for span in spans
            )

            y1 = max(
                span["bbox"][3]
                for span in spans
            )

            results.append(
                {
                    "text": text,
                    "rect": pymupdf.Rect(
                        x0,
                        y0,
                        x1,
                        y1,
                    ),
                }
            )

    results.sort(
        key=lambda item: (
            item["rect"].y0,
            item["rect"].x0,
        )
    )

    return results


def preview_regions(
    start: tuple[int, float],
    end: tuple[int, float],
) -> list[dict]:
    """Represent a refined visual span without changing its retrieval unit."""
    start_page, start_y = start
    end_page, end_y = end
    if end_page < start_page or (end_page == start_page and end_y <= start_y):
        return []
    return [
        {
            "page_number": page_number,
            "start_y": round(start_y if page_number == start_page else 0.0, 2),
            "end_y": round(end_y, 2) if page_number == end_page else None,
        }
        for page_number in range(start_page, end_page + 1)
    ]


def _block_position(block: dict, edge: int) -> tuple[int, float]:
    return int(block["page"]), float(block["bbox"][edge])


def _anchor_position(anchor: dict) -> tuple[int, float]:
    return int(anchor["page"]), float(anchor["bbox"][1])


def _candidate_layout_gaps(
    scope: dict,
    previous_anchor: dict,
    next_anchor: dict,
) -> list[tuple[float, int, float, float]]:
    """Find strong whitespace gaps inside two confirmed question anchors."""
    lower = _anchor_position(previous_anchor)
    upper = _anchor_position(next_anchor)
    blocks = [
        block
        for block in scope.get("blocks", [])
        if len(block.get("bbox", [])) == 4
        and _block_position(block, 1) >= lower
        and _block_position(block, 1) < upper
    ]
    blocks.sort(key=lambda block: int(block.get("order", 0)))

    heights = [
        float(block["bbox"][3]) - float(block["bbox"][1])
        for block in blocks
    ]
    typical_height = sorted(heights)[len(heights) // 2] if heights else 0.0
    minimum_gap = max(MIN_INFERRED_VERTICAL_GAP, typical_height * 1.5)

    gaps = []
    for left, right in zip(blocks, blocks[1:]):
        if int(left["page"]) != int(right["page"]):
            continue
        gap_start = float(left["bbox"][3])
        gap_end = float(right["bbox"][1])
        gap_size = gap_end - gap_start
        if gap_size >= minimum_gap:
            gaps.append((gap_size, int(left["page"]), gap_start, gap_end))
    return gaps


def find_visual_whitespace_cut(
    page,
    gap_start: float,
    gap_end: float,
) -> float | None:
    """Choose a blank raster row, accounting for graphics absent from OCR."""
    clip = pymupdf.Rect(
        page.rect.x0 + 32,
        gap_start + 2,
        page.rect.x1 - 32,
        gap_end - 2,
    ) & page.rect
    if clip.width <= 0 or clip.height < 4:
        return None

    pixmap = page.get_pixmap(
        colorspace=pymupdf.csGRAY,
        alpha=False,
        clip=clip,
    )
    if pixmap.width <= 0 or pixmap.height <= 0:
        return None

    samples = pixmap.samples
    row_ratios = []
    for row in range(pixmap.height):
        offset = row * pixmap.stride
        pixels = samples[offset:offset + pixmap.width]
        row_ratios.append(
            sum(pixel < 220 for pixel in pixels) / pixmap.width
        )

    window_radius = min(2, max(0, pixmap.height // 4))
    midpoint = (pixmap.height - 1) / 2
    candidates = []
    for row in range(pixmap.height):
        first = max(0, row - window_radius)
        last = min(pixmap.height, row + window_radius + 1)
        ink_ratio = sum(row_ratios[first:last]) / (last - first)
        candidates.append((ink_ratio, abs(row - midpoint), row))
    ink_ratio, _, row = min(candidates)
    if ink_ratio > MAX_VISUAL_CUT_INK_RATIO:
        return None
    return round(clip.y0 + row * clip.height / pixmap.height, 2)


def infer_independent_visual_starts(
    scope: dict,
    doc,
) -> dict[int, tuple[int, float]]:
    """Infer only missing independent-question starts between known anchors."""
    if scope.get("structure_type") != "independent":
        return {}

    expected = [int(number) for number in scope.get("question_numbers", [])]
    anchors = {
        int(anchor["question_number"]): anchor
        for anchor in scope.get("question_anchors", [])
    }
    inferred = {}
    index = 0
    while index < len(expected):
        if expected[index] in anchors:
            index += 1
            continue
        first_missing = index
        while index < len(expected) and expected[index] not in anchors:
            index += 1
        missing = expected[first_missing:index]

        # Two real neighboring anchors are required. One-sided guesses remain
        # conservative because they could otherwise cut a question graphic.
        if first_missing == 0 or index >= len(expected):
            continue
        previous_number = expected[first_missing - 1]
        next_number = expected[index]
        previous_anchor = anchors.get(previous_number)
        next_anchor = anchors.get(next_number)
        if previous_anchor is None or next_anchor is None:
            continue

        candidates = _candidate_layout_gaps(
            scope,
            previous_anchor,
            next_anchor,
        )
        resolved = []
        for gap_size, page_number, gap_start, gap_end in candidates:
            if not 1 <= page_number <= len(doc):
                continue
            cut_y = find_visual_whitespace_cut(
                doc[page_number - 1],
                gap_start,
                gap_end,
            )
            if cut_y is not None:
                resolved.append((gap_size, page_number, cut_y))

        if len(resolved) < len(missing):
            continue
        strongest = sorted(resolved, reverse=True)[:len(missing)]
        strongest.sort(key=lambda item: (item[1], item[2]))
        for question_number, (_, page_number, cut_y) in zip(missing, strongest):
            inferred[question_number] = (page_number, cut_y)
    return inferred


def refine_visual_preview_regions(
    unit: dict,
    inferred_starts: dict[int, tuple[int, float]],
) -> list[dict]:
    """Apply inferred starts to crops only; leave retrieval metadata untouched."""
    regions = unit.get("preview_regions") or []
    if not regions:
        return []
    question_number = int(unit["question_start"])
    first = regions[0]
    last = regions[-1]
    start = inferred_starts.get(
        question_number,
        (int(first["page_number"]), float(first.get("start_y") or 0.0)),
    )
    next_start = inferred_starts.get(question_number + 1)
    if next_start is not None:
        end = next_start
    else:
        raw_end = last.get("end_y")
        if raw_end is None:
            return regions
        end = (int(last["page_number"]), float(raw_end))
    return preview_regions(start, end) or regions


def infer_shared_visual_starts(
    scope: dict,
    doc,
) -> dict[int, tuple[int, float]]:
    """Verify inferred starts for shared question groups against the rendered page."""
    if scope.get("structure_type") != "shared_context":
        return {}

    expected = [int(number) for number in scope.get("question_numbers", [])]
    anchors = {
        int(anchor["question_number"]): anchor
        for anchor in scope.get("question_anchors", [])
    }
    confirmed_starts = {
        number: question_start_order(scope.get("blocks", []), anchor)
        for number, anchor in anchors.items()
    }
    inferred_orders = infer_missing_question_starts(
        scope, expected, confirmed_starts
    )
    blocks_by_order = {
        int(block["order"]): block
        for block in scope.get("blocks", [])
        if len(block.get("bbox", [])) == 4
    }
    inferred = {}
    for number, order in inferred_orders.items():
        block = blocks_by_order.get(order)
        previous = blocks_by_order.get(order - 1)
        if block is None or previous is None:
            continue
        page_number = int(block["page"])
        if not 1 <= page_number <= len(doc):
            continue
        if int(previous["page"]) != page_number:
            inferred[number] = (
                page_number,
                max(0.0, float(block["bbox"][1]) - 6),
            )
            continue
        cut_y = find_visual_whitespace_cut(
            doc[page_number - 1],
            float(previous["bbox"][3]),
            float(block["bbox"][1]),
        )
        if cut_y is not None:
            inferred[number] = (page_number, cut_y)
    return inferred


def compose_shared_visual_preview_regions(
    scope: dict,
    unit: dict,
    inferred_starts: dict[int, tuple[int, float]] | None = None,
) -> list[dict]:
    """Separate a shared stimulus from the target sibling question."""
    if (
        scope.get("structure_type") != "shared_context"
        or unit.get("shared_context_id") is None
        or not scope.get("shared_context_blocks")
    ):
        return unit.get("preview_regions") or []

    expected = [int(number) for number in scope.get("question_numbers", [])]
    question_number = int(unit["question_start"])
    anchors = {
        int(anchor["question_number"]): anchor
        for anchor in scope.get("question_anchors", [])
    }
    if question_number not in expected or not expected:
        return unit.get("preview_regions") or []
    target_index = expected.index(question_number)

    def visual_anchor_start(anchor: dict) -> tuple[int, float]:
        return (
            int(anchor["page"]),
            max(0.0, float(anchor["bbox"][1]) - 6),
        )

    question_starts = {
        number: visual_anchor_start(anchor)
        for number, anchor in anchors.items()
    }
    question_starts.update(inferred_starts or {})
    if expected[0] not in question_starts or question_number not in question_starts:
        return unit.get("preview_regions") or []

    context_start = (
        int(scope["scope_start"]["page"]),
        float(scope["scope_start"]["y"]),
    )
    first_question_start = question_starts[expected[0]]
    target_start = question_starts[question_number]

    if target_index + 1 < len(expected):
        target_end = question_starts.get(expected[target_index + 1])
        if target_end is None:
            return unit.get("preview_regions") or []
    else:
        target_end = (
            int(scope["scope_end"]["page"]),
            float(scope["scope_end"]["y"]),
        )

    context_regions = preview_regions(context_start, first_question_start)
    question_regions = preview_regions(target_start, target_end)
    if not context_regions or not question_regions:
        return unit.get("preview_regions") or []
    for region in context_regions:
        region["role"] = "shared_context"
    for region in question_regions:
        region["role"] = "question"
    return context_regions + question_regions


# Header / footer filtering


def is_noise_line(
    text: str,
) -> bool:

    value = compact_text(
        text
    )

    if not value:
        return True

    # Printed page number.
    if re.fullmatch(
        r"\d+",
        value,
    ):
        return True

    if (
        "TOPIKⅡ읽기"
        in value
    ):
        return True

    if (
        "TOPIKII읽기"
        in value
    ):
        return True

    if (
        "한국어능력시험"
        in value
        and "읽기"
        in value
    ):
        return True

    return False


def get_meaningful_lines(
    page,
) -> list[dict]:

    return [
        line
        for line in get_lines(
            page
        )
        if not is_noise_line(
            line["text"]
        )
    ]


# Question detection


def question_pattern(
    question_number: int,
):

    return re.compile(
        rf"^\s*"
        rf"{question_number}"
        rf"\s*(?:[\.．]|$)"
    )


def find_question_line(
    page,
    question_number: int,
):

    pattern = question_pattern(
        question_number
    )

    matches = []

    for line in get_lines(
        page
    ):

        if pattern.search(
            line["text"]
        ):
            matches.append(
                line
            )

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            item["rect"].y0,
            item["rect"].x0,
        )
    )

    return matches[0]


def find_section_lines(
    page,
) -> list[dict]:

    return [
        line
        for line in get_lines(
            page
        )
        if "※" in line["text"]
        or re.search(r"\[\s*\d+\s*[～~\-–]\s*\d+\s*\]", line["text"])
    ]


def find_group_start_line(
    page,
    question_start: int,
    question_end: int,
):

    start = str(
        question_start
    )

    end = str(
        question_end
    )

    for line in find_section_lines(
        page
    ):

        value = compact_text(
            line["text"]
        )

        if (
            start in value
            and end in value
        ):
            return line

    return None


# Retrieval-unit start


def find_unit_start(
    doc,
    unit: dict,
):

    pages = get_pages(
        unit
    )

    if not pages:
        return None

    question_start = unit[
        "question_start"
    ]

    question_end = unit[
        "question_end"
    ]

    # Any multi-question unit starts at its section header. This also supports
    # OCR groups where punctuation after individual question numbers was lost.
    if question_start != question_end:

        for page_number in pages:

            page = doc[
                page_number - 1
            ]

            line = find_group_start_line(
                page,
                question_start,
                question_end,
            )

            if line is not None:

                return (
                    page_number,
                    line["rect"].y0,
                )

    # Individual question.
    for page_number in pages:

        page = doc[
            page_number - 1
        ]

        line = find_question_line(
            page,
            question_start,
        )

        if line is not None:

            return (
                page_number,
                line["rect"].y0,
            )

    return None


# End boundary


def find_next_boundary(
    page,
    current_y: float,
    next_question: int,
):

    candidates = []

    next_line = find_question_line(
        page,
        next_question,
    )

    if next_line is not None:

        y = next_line[
            "rect"
        ].y0

        if y > current_y + 5:

            candidates.append(
                y
            )

    # A new ※ section may begin before the next
    # numbered question.
    for line in find_section_lines(
        page
    ):

        y = line[
            "rect"
        ].y0

        if y > current_y + 10:

            candidates.append(
                y
            )

    if not candidates:
        return None

    return min(
        candidates
    )


# Crop coordinate calculation


def get_crop_content_lines(
    page,
    start_y: float,
    boundary_y: float | None,
) -> list[dict]:

    results = []

    for line in get_lines(
        page
    ):

        rect = line[
            "rect"
        ]

        if rect.y1 < start_y:
            continue

        if (
            boundary_y is not None
            and rect.y0 >= boundary_y
        ):
            continue

        if is_noise_line(
            line["text"]
        ):
            continue

        results.append(
            line
        )

    return results


def calculate_bbox(
    page,
    start_y: float,
    boundary_y: float | None,
) -> list[float] | None:

    lines = get_crop_content_lines(
        page,
        start_y,
        boundary_y,
    )

    if not lines:
        return None

    # Horizontal coordinates come from actual
    # question content.

    x0 = min(
        line["rect"].x0
        for line in lines
    )

    x1 = max(
        line["rect"].x1
        for line in lines
    )

    # Vertical coordinates.

    y0 = min(
        line["rect"].y0
        for line in lines
    )

    y1 = max(
        line["rect"].y1
        for line in lines
    )

    x0 = max(
        page.rect.x0,
        x0 - HORIZONTAL_PADDING,
    )

    x1 = min(
        page.rect.x1,
        x1 + HORIZONTAL_PADDING,
    )

    y0 = max(
        page.rect.y0,
        y0 - TOP_PADDING,
    )

    y1 = min(
        page.rect.y1,
        y1 + BOTTOM_PADDING,
    )

    # Do not cross into next question.
    if boundary_y is not None:

        y1 = min(
            y1,
            boundary_y - 2,
        )

    if (
        x1 <= x0
        or y1 <= y0
    ):
        return None

    return [
        round(
            x0,
            2,
        ),
        round(
            y0,
            2,
        ),
        round(
            x1,
            2,
        ),
        round(
            y1,
            2,
        ),
    ]


def calculate_structural_bbox(
    page,
    start_y: float,
    end_y: float | None,
) -> list[float] | None:
    """Convert a vertical page region into a page-wide PDF crop."""
    lines = get_crop_content_lines(page, start_y, end_y)
    if not lines:
        return None

    x0 = max(page.rect.x0, page.rect.x0 + 32)
    x1 = min(page.rect.x1, page.rect.x1 - 32)
    y0 = max(page.rect.y0, start_y - TOP_PADDING)
    if end_y is None:
        y1 = min(
            page.rect.y1,
            max(line["rect"].y1 for line in lines) + BOTTOM_PADDING,
        )
    else:
        y1 = min(page.rect.y1, end_y - 2)

    if x1 <= x0 or y1 <= y0:
        return None
    return [round(value, 2) for value in (x0, y0, x1, y1)]


def generate_structural_metadata(doc, unit: dict) -> list[dict]:
    crops = []
    regions = unit.get("_visual_preview_regions", unit.get("preview_regions"))
    for region in regions or []:
        page_number = int(region["page_number"])
        if not 1 <= page_number <= len(doc):
            continue
        page = doc[page_number - 1]
        start_y = float(region.get("start_y") or page.rect.y0)
        raw_end_y = region.get("end_y")
        end_y = float(raw_end_y) if raw_end_y is not None else None
        bbox = calculate_structural_bbox(page, start_y, end_y)
        if bbox is None:
            continue
        crop = {
            "page_number": page_number,
            "page_index": page_number - 1,
            "bbox": bbox,
        }
        if region.get("role"):
            crop["role"] = region["role"]
        crops.append(crop)
    return crops


# Metadata generation for one retrieval unit


def generate_unit_metadata(
    doc,
    unit: dict,
) -> list[dict]:

    if unit.get("preview_regions"):
        return generate_structural_metadata(doc, unit)

    pages = get_pages(
        unit
    )

    if not pages:
        return []

    start_location = find_unit_start(
        doc,
        unit,
    )

    if start_location is None:
        return []

    (
        start_page,
        start_y,
    ) = start_location

    next_question = (
        unit[
            "question_end"
        ]
        + 1
    )

    last_page = max(
        pages
    )

    crops = []

    for page_number in range(
        start_page,
        last_page + 1,
    ):

        page = doc[
            page_number - 1
        ]

        # First page.

        if (
            page_number
            == start_page
        ):

            page_start_y = (
                start_y
            )

        # Possible continuation page.

        else:

            meaningful_lines = (
                get_meaningful_lines(
                    page
                )
            )

            if not meaningful_lines:
                continue

            first_line = (
                meaningful_lines[0]
            )

            next_question_line = (
                find_question_line(
                    page,
                    next_question,
                )
            )

            # Example:
            #
            # Q10 metadata may contain pages [22, 23],
            # but if page 23 starts directly with Q11,
            # page 23 does not belong to Q10.
            if (
                next_question_line
                is not None
                and next_question_line[
                    "rect"
                ].y0
                <= first_line[
                    "rect"
                ].y0 + 5
            ):
                break

            page_start_y = (
                first_line[
                    "rect"
                ].y0
            )

        # Locate the next question / section.

        boundary = find_next_boundary(
            page,
            page_start_y,
            next_question,
        )

        # Calculate metadata only.
        # No PNG is created.

        bbox = calculate_bbox(
            page,
            page_start_y,
            boundary,
        )

        if bbox is None:
            continue

        crops.append(
            {
                # Human-readable PDF page number.
                "page_number":
                    page_number,

                # Convenient for PyMuPDF:
                # doc[page_index]
                "page_index":
                    page_number - 1,

                # PDF coordinates, NOT image pixels.
                "bbox":
                    bbox,
            }
        )

        # The following question starts on this page,
        # therefore this unit is complete.
        if boundary is not None:
            break

    return crops


# Main


def main():

    OCR_PAGE_LINES.clear()
    if PAGES_FILE.exists():
        with open(PAGES_FILE, "r", encoding="utf-8") as f:
            page_records = json.load(f)
        for record in page_records:
            if record.get("extraction_method") == "ocr":
                OCR_PAGE_LINES[
                    (record["source_file"], int(record["page"]))
                ] = record.get("lines") or []

    with open(
        UNITS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        units = json.load(f)

    scopes_by_instruction = {}
    if GROUPS_FILE.exists():
        with open(GROUPS_FILE, "r", encoding="utf-8") as f:
            scopes = json.load(f)
        scopes_by_instruction = {
            (int(scope["exam_number"]), scope["instruction_id"]): scope
            for scope in scopes
        }

    documents = {}

    metadata = {}

    successful = 0
    failed = 0
    total_crops = 0
    inferred_starts_by_instruction = {}

    for unit in units:

        source_file = unit[
            "source_file"
        ]

        pdf_path = (
            PDF_DIR
            / source_file
        )

        if not pdf_path.exists():

            print(
                f"Missing PDF: "
                f"{pdf_path}"
            )

            failed += 1
            continue

        if (
            source_file
            not in documents
        ):

            documents[
                source_file
            ] = pymupdf.open(
                pdf_path
            )

        doc = documents[
            source_file
        ]

        scope_key = (
            int(unit["exam_number"]),
            unit.get("instruction_id"),
        )
        scope = scopes_by_instruction.get(scope_key)
        visual_unit = unit
        if scope is not None and unit.get("shared_context_id") is not None:
            if scope_key not in inferred_starts_by_instruction:
                inferred_starts_by_instruction[scope_key] = (
                    infer_shared_visual_starts(scope, doc)
                )
            visual_unit = dict(unit)
            visual_unit["_visual_preview_regions"] = (
                compose_shared_visual_preview_regions(
                    scope,
                    unit,
                    inferred_starts_by_instruction[scope_key],
                )
            )
        elif (
            scope is not None
            and scope_key not in inferred_starts_by_instruction
        ):
            inferred_starts_by_instruction[scope_key] = (
                infer_independent_visual_starts(scope, doc)
            )
        if unit.get("shared_context_id") is None:
            inferred_starts = inferred_starts_by_instruction.get(scope_key, {})
            if inferred_starts:
                visual_unit = dict(unit)
                visual_unit["_visual_preview_regions"] = (
                    refine_visual_preview_regions(unit, inferred_starts)
                )

        crops = generate_unit_metadata(
            doc,
            visual_unit,
        )

        key = make_key(
            unit[
                "exam_number"
            ],
            unit[
                "question_start"
            ],
            unit[
                "question_end"
            ],
        )

        if not crops:

            print(
                f"FAILED: TOPIK "
                f"{unit['exam_number']} "
                f"Q{unit['question_start']}"
                f"-{unit['question_end']}"
            )

            failed += 1
            continue

        actual_pages = [
            crop[
                "page_number"
            ]
            for crop in crops
        ]

        metadata[
            key
        ] = {
            "exam_number":
                unit[
                    "exam_number"
                ],

            "question_start":
                unit[
                    "question_start"
                ],

            "question_end":
                unit[
                    "question_end"
                ],

            "question_numbers":
                unit.get(
                    "question_numbers",
                    list(
                        range(
                            unit[
                                "question_start"
                            ],
                            unit[
                                "question_end"
                            ]
                            + 1,
                        )
                    ),
                ),

            "source_file":
                source_file,

            "chunk_type":
                unit.get(
                    "chunk_type"
                ),

            "instruction_id":
                unit.get(
                    "instruction_id"
                ),

            "shared_context_id":
                unit.get(
                    "shared_context_id"
                ),

            "structure_confidence":
                unit.get(
                    "structure_confidence"
                ),

            # These are the pages actually containing
            # this retrieval unit, rather than the broad
            # page range inherited from its original group.
            "pages":
                actual_pages,

            "crops":
                crops,
        }

        successful += 1
        total_crops += len(
            crops
        )

    for doc in (
        documents.values()
    ):
        doc.close()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print(
        "TOPIK CROP METADATA COMPLETE"
    )
    print("=" * 60)

    print(
        f"Units      : "
        f"{len(units)}"
    )

    print(
        f"Successful : "
        f"{successful}"
    )

    print(
        f"Failed     : "
        f"{failed}"
    )

    print(
        f"Crop boxes : "
        f"{total_crops}"
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
