"""Extract configured TOPIK reading pages, using cached OCR when necessary."""

from __future__ import annotations

import json

import pymupdf

try:
    from scripts.topik_page_extractor import (
        TopikPageExtractor,
        assess_native_structure,
    )
    from scripts.topik_sources import PROJECT_ROOT, load_sources
except ModuleNotFoundError:  # Support: python scripts/extract_topik_reading.py
    from topik_page_extractor import (
        TopikPageExtractor,
        assess_native_structure,
    )
    from topik_sources import PROJECT_ROOT, load_sources


PDF_DIR = PROJECT_ROOT / "data/topik/raw"
OUTPUT_FILE = PROJECT_ROOT / "data/topik/processed/topik_reading_pages.json"
DIAGNOSTICS_FILE = (
    PROJECT_ROOT / "data/topik/processed/topik_extraction_diagnostics.json"
)


def extract_sources() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    diagnostics: list[str] = []

    for source in load_sources():
        pdf_path = PDF_DIR / source.file
        if not pdf_path.exists():
            diagnostics.append(f"TOPIK {source.exam_number}: missing {pdf_path}")
            continue

        document = pymupdf.open(pdf_path)
        start_page, end_page = source.reading_pages
        if end_page > len(document):
            diagnostics.append(
                f"TOPIK {source.exam_number}: configured page {end_page} exceeds "
                f"the {len(document)}-page PDF"
            )
            document.close()
            continue

        print(
            f"Processing TOPIK {source.exam_number}: pages "
            f"{start_page}-{end_page} ({source.extraction})"
        )
        extractor = TopikPageExtractor(pdf_path, extraction=source.extraction)
        method_counts: dict[str, int] = {}
        page_numbers = list(range(start_page, end_page + 1))
        extracted_pages: list[dict] = []

        if source.extraction == "auto":
            native_candidates: list[dict] = []
            for page_number in page_numbers:
                try:
                    native_candidates.append(
                        extractor.extract_native_candidate(
                            document[page_number - 1], page_number
                        )
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    diagnostics.append(
                        f"TOPIK {source.exam_number} page {page_number}: "
                        f"native candidate failed ({error})"
                    )

            assessment = assess_native_structure(native_candidates)
            selected = (
                "native"
                if len(native_candidates) == len(page_numbers)
                and assessment["accepted"]
                else "ocr"
            )
            print(
                "  Auto selection: "
                f"{selected} (ranges={assessment['instruction_ranges']}, "
                f"coverage={assessment['covered_questions']}, "
                f"anchors={assessment['unique_question_anchors']}, "
                f"progression={assessment['anchor_progression_ratio']:.2f})"
            )
            if selected == "native":
                extracted_pages = native_candidates
            else:
                for page_number in page_numbers:
                    try:
                        extracted_pages.append(
                            extractor.extract_ocr_page(
                                document[page_number - 1], page_number
                            )
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        diagnostics.append(
                            f"TOPIK {source.exam_number} page {page_number}: "
                            f"OCR fallback failed ({error})"
                        )
        else:
            for page_number in page_numbers:
                try:
                    extracted_pages.append(
                        extractor.extract_page(document[page_number - 1], page_number)
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    diagnostics.append(
                        f"TOPIK {source.exam_number} page {page_number}: "
                        f"extraction failed ({error})"
                    )

        for extracted in extracted_pages:
            page_number = int(extracted["page"])
            try:
                method = extracted["extraction_method"]
                method_counts[method] = method_counts.get(method, 0) + 1
                if not extracted["text"].strip():
                    diagnostics.append(
                        f"TOPIK {source.exam_number} page {page_number}: "
                        f"empty {method} text"
                    )
                    continue
                records.append(
                    {
                        "source_type": "topik",
                        "exam_number": source.exam_number,
                        "section": "reading",
                        "source_file": source.file,
                        **extracted,
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                diagnostics.append(
                    f"TOPIK {source.exam_number} page {page_number}: "
                    f"invalid extraction record ({error})"
                )

        document.close()
        counts = ", ".join(f"{key}={value}" for key, value in method_counts.items())
        print(f"  Extracted {sum(method_counts.values())} pages ({counts})")

    return records, diagnostics


def main() -> None:
    records, diagnostics = extract_sources()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    summaries = []
    for source in load_sources():
        source_records = [
            record for record in records if record["exam_number"] == source.exam_number
        ]
        methods: dict[str, int] = {}
        for record in source_records:
            method = record["extraction_method"]
            methods[method] = methods.get(method, 0) + 1
        summaries.append(
            {
                "exam_number": source.exam_number,
                "source_file": source.file,
                "configured_pages": list(source.reading_pages),
                "pages_extracted": len(source_records),
                "methods": methods,
            }
        )
    with DIAGNOSTICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {"sources": summaries, "warnings": diagnostics},
            file,
            ensure_ascii=False,
            indent=2,
        )

    for message in diagnostics:
        print(f"WARNING: {message}")
    print(f"Saved {len(records)} reading pages to: {OUTPUT_FILE}")
    print(f"Saved extraction diagnostics to: {DIAGNOSTICS_FILE}")
    if diagnostics:
        print(f"Completed with {len(diagnostics)} warning(s).")


if __name__ == "__main__":
    main()
