"""Shared TOPIK source-manifest loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/topik/sources.json"


@dataclass(frozen=True)
class TopikSource:
    exam_number: int
    file: str
    reading_pages: tuple[int, int]
    extraction: str = "auto"
    enabled: bool = True


def load_sources(path: Path = DEFAULT_MANIFEST) -> list[TopikSource]:
    """Load enabled sources, rejecting ambiguous or unsafe configuration."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    raw_sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(raw_sources, list):
        raise ValueError("TOPIK manifest must contain a 'sources' list.")

    sources: list[TopikSource] = []
    seen_exams: set[int] = set()
    seen_files: set[str] = set()
    for index, item in enumerate(raw_sources, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"TOPIK source {index} must be an object.")

        exam_number = int(item["exam_number"])
        filename = str(item["file"]).strip()
        pages = item.get("reading_pages")
        extraction = str(item.get("extraction", "auto")).lower()
        enabled = bool(item.get("enabled", True))

        if not filename or Path(filename).name != filename:
            raise ValueError(f"TOPIK {exam_number}: 'file' must be a filename.")
        if (
            not isinstance(pages, list)
            or len(pages) != 2
            or not all(isinstance(page, int) for page in pages)
            or pages[0] < 1
            or pages[1] < pages[0]
        ):
            raise ValueError(
                f"TOPIK {exam_number}: 'reading_pages' must be [start, end]."
            )
        if extraction not in {"auto", "native", "ocr"}:
            raise ValueError(
                f"TOPIK {exam_number}: extraction must be auto, native, or ocr."
            )
        if exam_number in seen_exams or filename in seen_files:
            raise ValueError(f"Duplicate TOPIK source: {exam_number} / {filename}")

        seen_exams.add(exam_number)
        seen_files.add(filename)
        if enabled:
            sources.append(
                TopikSource(
                    exam_number=exam_number,
                    file=filename,
                    reading_pages=(pages[0], pages[1]),
                    extraction=extraction,
                    enabled=enabled,
                )
            )

    return sources
