"""Validate TOPIK retrieval units and lazy PDF crop metadata."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

try:
    from scripts.topik_sources import load_sources
except ModuleNotFoundError:
    from topik_sources import load_sources


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNITS_FILE = PROJECT_ROOT / "data/topik/processed/topik_retrieval_units.json"
METADATA_FILE = PROJECT_ROOT / "data/topik/processed/topik_crop_metadata.json"
CONTEXTS_FILE = PROJECT_ROOT / "data/topik/processed/topik_shared_contexts.json"
PDF_DIR = PROJECT_ROOT / "data/topik/raw"


def unit_key(item: dict) -> str:
    return (
        f"{item['exam_number']}:"
        f"{item['question_start']}:"
        f"{item['question_end']}"
    )


def validate_question_coverage(units: list[dict]) -> list[str]:
    errors = []
    coverage: defaultdict[int, defaultdict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for unit in units:
        key = unit_key(unit)
        if (
            unit.get("chunk_type") != "question"
            or unit.get("question_start") != unit.get("question_end")
            or unit.get("question_numbers") != [unit.get("question_start")]
        ):
            errors.append(f"{key}: retrieval unit is not one logical question")
        for question in unit["question_numbers"]:
            coverage[unit["exam_number"]][question].append(key)

    enabled_exams = {source.exam_number for source in load_sources()}
    unit_exams = set(coverage)
    for exam_number in sorted(enabled_exams - unit_exams):
        errors.append(f"Enabled TOPIK {exam_number} has no retrieval units")
    for exam_number in sorted(unit_exams - enabled_exams):
        errors.append(f"TOPIK {exam_number} is not enabled in the source manifest")

    for exam_number, questions in sorted(coverage.items()):
        missing = [number for number in range(1, 51) if number not in questions]
        duplicates = {
            number: keys for number, keys in questions.items() if len(keys) > 1
        }
        print(
            f"TOPIK {exam_number}: 50-question coverage "
            f"missing={missing or 'none'} duplicates={duplicates or 'none'}"
        )
        if missing:
            errors.append(f"TOPIK {exam_number} is missing questions {missing}")
        if duplicates:
            errors.append(f"TOPIK {exam_number} has duplicate question mappings")
    return errors


def validate_shared_contexts(
    units: list[dict], contexts: list[dict]
) -> list[str]:
    errors = []
    context_counts = Counter(item.get("shared_context_id") for item in contexts)
    duplicate_ids = sorted(
        context_id
        for context_id, count in context_counts.items()
        if context_id is not None and count > 1
    )
    if duplicate_ids:
        errors.append(f"Duplicate shared-context IDs: {duplicate_ids}")

    context_by_id = {
        item["shared_context_id"]: item
        for item in contexts
        if item.get("shared_context_id")
    }
    references: defaultdict[str, list[int]] = defaultdict(list)
    for unit in units:
        context_id = unit.get("shared_context_id")
        if context_id is None:
            continue
        references[context_id].append(unit["question_start"])
        context = context_by_id.get(context_id)
        if context is None:
            errors.append(f"{unit_key(unit)}: unknown shared context {context_id}")
            continue
        if unit.get("shared_context_text") != context.get("text"):
            errors.append(f"{unit_key(unit)}: attached shared context text differs")
        if context.get("source_file") != unit.get("source_file"):
            errors.append(f"{unit_key(unit)}: shared context source differs")

    orphaned = sorted(set(context_by_id) - set(references))
    if orphaned:
        errors.append(f"Orphaned shared contexts: {orphaned}")
    for context_id, questions in references.items():
        context = context_by_id.get(context_id)
        if context and sorted(questions) != sorted(context.get("question_numbers", [])):
            errors.append(f"{context_id}: referenced questions do not match context")

    print(
        f"Shared contexts: {len(contexts)} total, "
        f"{len(references)} referenced, {len(orphaned)} orphaned"
    )
    return errors


def validate_crop_metadata(
    units: list[dict], metadata: dict[str, dict]
) -> list[str]:
    errors = []
    indexed_units = [
        unit
        for unit in units
        if unit.get(
            "indexable",
            unit.get("content_type") != "visual_or_sparse",
        )
    ]
    expected = {unit_key(unit): unit for unit in units}

    missing = sorted(set(expected) - set(metadata))
    orphaned = sorted(set(metadata) - set(expected))
    if missing:
        errors.append(f"Missing crop metadata for {len(missing)} units: {missing[:10]}")
    if orphaned:
        errors.append(f"Crop metadata has {len(orphaned)} orphaned units: {orphaned[:10]}")

    documents = {}
    try:
        for key, item in metadata.items():
            source_file = item.get("source_file")
            pdf_path = PDF_DIR / source_file if source_file else None
            if pdf_path is None or not pdf_path.exists():
                errors.append(f"{key}: source PDF is missing ({source_file})")
                continue

            expected_unit = expected.get(key)
            if expected_unit and source_file != expected_unit.get("source_file"):
                errors.append(f"{key}: crop and retrieval sources do not match")

            document = documents.get(source_file)
            if document is None:
                document = pymupdf.open(pdf_path)
                documents[source_file] = document

            crops = item.get("crops") or []
            if not crops:
                errors.append(f"{key}: no crops")
                continue

            for index, crop in enumerate(crops):
                page_index = crop.get("page_index")
                bbox = crop.get("bbox")
                if not isinstance(page_index, int) or not 0 <= page_index < len(document):
                    errors.append(f"{key} crop {index}: invalid page index {page_index}")
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"{key} crop {index}: invalid bounding box")
                    continue
                try:
                    clip = pymupdf.Rect(*bbox)
                except (TypeError, ValueError):
                    errors.append(f"{key} crop {index}: non-numeric bounding box")
                    continue
                page_rect = document[page_index].rect
                if clip.is_empty or not page_rect.contains(clip):
                    errors.append(
                        f"{key} crop {index}: bounding box is outside PDF page"
                    )
    finally:
        for document in documents.values():
            document.close()

    print(
        f"Retrieval units: {len(units)} total, {len(indexed_units)} indexed; "
        f"crop records: {len(metadata)}"
    )
    return errors


def main() -> int:
    with UNITS_FILE.open(encoding="utf-8") as file:
        units = json.load(file)
    with METADATA_FILE.open(encoding="utf-8") as file:
        metadata = json.load(file)
    with CONTEXTS_FILE.open(encoding="utf-8") as file:
        contexts = json.load(file)

    key_counts = Counter(unit_key(unit) for unit in units)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    errors = []
    if duplicate_keys:
        errors.append(f"Duplicate retrieval-unit keys: {duplicate_keys}")
    errors.extend(validate_question_coverage(units))
    errors.extend(validate_shared_contexts(units, contexts))
    errors.extend(validate_crop_metadata(units, metadata))

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("TOPIK retrieval and crop metadata validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
