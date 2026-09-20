"""Run the existing TOPIK ingestion pipeline from one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.topik_sources import DEFAULT_MANIFEST, PROJECT_ROOT, load_sources


RAW_DIR = PROJECT_ROOT / "data/topik/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/topik/processed"

PIPELINE = (
    ("Extracting reading pages", "scripts.extract_topik_reading"),
    ("Parsing question structure", "scripts.parse_topik_groups"),
    ("Building retrieval units", "scripts.build_topik_retrieval_units"),
    ("Building TOPIK Chroma index", "scripts.build_topik_chroma"),
    ("Generating preview metadata", "scripts.generate_topik_crop_metadata"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one configured TOPIK exam and rebuild TOPIK data."
    )
    parser.add_argument("--exam", type=int, required=True, help="TOPIK exam number")
    parser.add_argument("--file", help="PDF filename in data/topik/raw/")
    parser.add_argument(
        "--reading-pages",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="inclusive PDF page range containing the reading section",
    )
    parser.add_argument(
        "--extraction",
        choices=("auto", "native", "ocr"),
        help="text extraction mode (default: auto for a new source)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate current generated data without rebuilding it",
    )
    return parser.parse_args()


def configure_source(args: argparse.Namespace) -> None:
    with DEFAULT_MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    sources = manifest.get("sources", [])
    existing = next(
        (item for item in sources if int(item["exam_number"]) == args.exam), None
    )
    metadata_supplied = any(
        value is not None
        for value in (args.file, args.reading_pages, args.extraction)
    )
    changed = False

    if existing is None and not (args.file and args.reading_pages):
        raise SystemExit(
            f"TOPIK {args.exam} is not in {DEFAULT_MANIFEST}. "
            "Provide --file and --reading-pages to add it."
        )

    if existing is None:
        existing = {
            "exam_number": args.exam,
            "file": args.file,
            "reading_pages": args.reading_pages,
            "extraction": args.extraction or "auto",
        }
        sources.append(existing)
        changed = True
    elif metadata_supplied:
        if args.file is not None:
            existing["file"] = args.file
        if args.reading_pages is not None:
            existing["reading_pages"] = args.reading_pages
        if args.extraction is not None:
            existing["extraction"] = args.extraction
        changed = True

    filename = str(existing["file"])
    if Path(filename).name != filename:
        raise SystemExit("--file must be a filename, not a path.")
    pdf_path = RAW_DIR / filename
    if not pdf_path.is_file():
        raise SystemExit(f"Source PDF not found: {pdf_path}")

    if existing.get("enabled") is False:
        raise SystemExit(f"TOPIK {args.exam} is disabled in {DEFAULT_MANIFEST}.")

    if changed:
        manifest["sources"] = sorted(
            sources, key=lambda item: int(item["exam_number"])
        )
        temporary_manifest = DEFAULT_MANIFEST.with_suffix(".json.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        try:
            load_sources(temporary_manifest)
            temporary_manifest.replace(DEFAULT_MANIFEST)
        finally:
            temporary_manifest.unlink(missing_ok=True)

    # Apply the same validation used by every pipeline stage.
    enabled = {source.exam_number for source in load_sources()}
    if args.exam not in enabled:
        raise SystemExit(f"TOPIK {args.exam} is not enabled in the source manifest.")


def run_module(label: str, module: str) -> None:
    print(f"\n{label}...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def load_json(filename: str):
    with (PROCESSED_DIR / filename).open(encoding="utf-8") as file:
        return json.load(file)


def print_summary(exam_number: int) -> None:
    pages = [
        item
        for item in load_json("topik_reading_pages.json")
        if int(item["exam_number"]) == exam_number
    ]
    units = [
        item
        for item in load_json("topik_retrieval_units.json")
        if int(item["exam_number"]) == exam_number
    ]
    contexts = [
        item
        for item in load_json("topik_shared_contexts.json")
        if int(item["exam_number"]) == exam_number
    ]
    metadata = load_json("topik_crop_metadata.json")
    previews = [
        item
        for item in metadata.values()
        if int(item["exam_number"]) == exam_number
    ]

    methods = Counter(item["extraction_method"] for item in pages)
    extraction = ", ".join(
        f"{method.upper()} ({count} pages)" for method, count in sorted(methods.items())
    ) or "none"
    low_confidence = sorted(
        int(item["question_start"])
        for item in units
        if item.get("structure_confidence") == "low"
    )
    low_label = ", ".join(f"Q{number}" for number in low_confidence) or "none"

    print(f"\nTOPIK {exam_number}")
    print(f"Extraction: {extraction}")
    print(f"Questions: {len(units)}/50")
    print(f"Shared contexts: {len(contexts)}")
    print(f"Low confidence: {low_label}")
    print(f"Indexable: {sum(bool(item.get('indexable')) for item in units)}")
    print(f"Preview records: {len(previews)}")
    print("Validation: PASS")


def main() -> int:
    args = parse_args()
    configure_source(args)

    if not args.validate_only:
        for label, module in PIPELINE:
            run_module(label, module)

    run_module("Validating TOPIK data", "scripts.validate_topik_question_mapping")
    print_summary(args.exam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
