"""Native-PDF extraction with an optional, cached EasyOCR fallback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pymupdf

try:
    from scripts.topik_ranges import parse_instruction_range
except ModuleNotFoundError:  # Support direct script execution
    from topik_ranges import parse_instruction_range


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_CACHE_DIR = PROJECT_ROOT / "data/topik/processed/ocr_cache"
OCR_MODEL_DIR = PROJECT_ROOT / "data/topik/ocr_models"
OCR_DPI = 200
OCR_ENGINE = "easyocr-1.7.2"

QUESTION_ANCHOR_RE = re.compile(r"^\s*([0-9Oo]{1,2})\s*[.．]")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_is_usable(text: str) -> bool:
    """Reject empty/image-only layers while tolerating ordinary PDF artifacts."""
    cleaned = clean_text(text)
    korean_count = len(re.findall(r"[가-힣]", cleaned))
    return len(cleaned) >= 50 and korean_count >= 10


def _question_anchor(text: str) -> int | None:
    match = QUESTION_ANCHOR_RE.match(text)
    if match is None:
        return None
    token = match.group(1).translate(str.maketrans({"O": "0", "o": "0"}))
    if not token.isdigit():
        return None
    number = int(token)
    return number if 1 <= number <= 50 else None


def assess_native_structure(records: list[dict]) -> dict:
    """Return a small, explainable structural check for an auto candidate."""
    page_count = len(records)
    usable_pages = sum(text_is_usable(record.get("text", "")) for record in records)
    ranges: list[tuple[int, int]] = []
    anchors: list[int] = []

    for record in records:
        for line in record.get("lines") or []:
            text = str(line.get("text", ""))
            question_range = parse_instruction_range(text)
            if question_range is not None:
                ranges.append(question_range)
            anchor = _question_anchor(text)
            if anchor is not None:
                anchors.append(anchor)

    covered_questions = {
        number
        for start, end in ranges
        for number in range(start, end + 1)
    }
    unique_anchors = set(anchors)
    anchors_in_ranges = unique_anchors & covered_questions
    progression_pairs = list(zip(anchors, anchors[1:]))
    plausible_steps = sum(
        current < following and following - current <= 3
        for current, following in progression_pairs
    )
    progression_ratio = (
        plausible_steps / len(progression_pairs) if progression_pairs else 0.0
    )

    minimum_headers = 1 if page_count <= 2 else max(2, math.ceil(page_count / 4))
    minimum_coverage = 2 if page_count <= 2 else page_count
    usable_ratio = usable_pages / page_count if page_count else 0.0
    anchor_coverage_ratio = (
        len(anchors_in_ranges) / len(covered_questions)
        if covered_questions
        else 0.0
    )
    accepted = all(
        (
            page_count > 0,
            usable_ratio >= 0.8,
            len(ranges) >= minimum_headers,
            len(covered_questions) >= minimum_coverage,
            anchor_coverage_ratio >= 0.65,
            progression_ratio >= 0.8,
        )
    )
    return {
        "accepted": accepted,
        "page_count": page_count,
        "usable_pages": usable_pages,
        "instruction_ranges": len(ranges),
        "covered_questions": len(covered_questions),
        "unique_question_anchors": len(unique_anchors),
        "anchor_coverage_ratio": round(anchor_coverage_ratio, 4),
        "anchor_progression_ratio": round(progression_ratio, 4),
    }


def choose_auto_extraction(records: list[dict]) -> str:
    """Select native only when its parser-facing structure is plausible."""
    return "native" if assess_native_structure(records)["accepted"] else "ocr"


def native_lines(page: pymupdf.Page) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text or not spans:
                continue
            lines.append(
                {
                    "text": text,
                    "bbox": [
                        min(span["bbox"][0] for span in spans),
                        min(span["bbox"][1] for span in spans),
                        max(span["bbox"][2] for span in spans),
                        max(span["bbox"][3] for span in spans),
                    ],
                    "confidence": 1.0,
                }
            )
    lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return lines


def pdf_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EasyOcrEngine:
    """Lazily create one CPU reader and reuse it for the preprocessing run."""

    def __init__(self) -> None:
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as error:
                raise RuntimeError(
                    "OCR is required for this PDF. Install project requirements "
                    "and place/download the EasyOCR Korean models first."
                ) from error

            OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                ["ko", "en"],
                gpu=False,
                model_storage_directory=str(OCR_MODEL_DIR),
                verbose=False,
            )
        return self._reader

    def extract(self, page: pymupdf.Page) -> tuple[str, list[dict[str, Any]], dict]:
        scale = OCR_DPI / 72
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )

        import numpy as np

        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        raw_results = self._get_reader().readtext(image, detail=1, paragraph=False)

        lines = []
        for box, text, confidence in raw_results:
            cleaned = clean_text(str(text))
            if not cleaned:
                continue
            xs = [float(point[0]) / scale for point in box]
            ys = [float(point[1]) / scale for point in box]
            lines.append(
                {
                    "text": cleaned,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": round(float(confidence), 4),
                }
            )

        lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        text = clean_text("\n".join(line["text"] for line in lines))
        confidences = [line["confidence"] for line in lines]
        diagnostics = {
            "ocr_line_count": len(lines),
            "average_confidence": round(sum(confidences) / len(confidences), 4)
            if confidences
            else 0.0,
        }
        return text, lines, diagnostics


class TopikPageExtractor:
    def __init__(self, pdf_path: Path, extraction: str = "auto") -> None:
        self.pdf_path = pdf_path
        self.extraction = extraction
        self.fingerprint = pdf_fingerprint(pdf_path)
        self.ocr = EasyOcrEngine()

    def _cache_path(self, page_number: int) -> Path:
        return OCR_CACHE_DIR / f"{self.pdf_path.stem}_p{page_number:03}.json"

    def _read_cache(self, page_number: int) -> dict | None:
        path = self._cache_path(page_number)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as file:
                cached = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        if (
            cached.get("pdf_sha256") != self.fingerprint
            or cached.get("ocr_engine") != OCR_ENGINE
            or cached.get("ocr_dpi") != OCR_DPI
        ):
            return None
        record = cached.get("page_record")
        if not isinstance(record, dict) or not record.get("text", "").strip():
            return None
        return record

    def _write_cache(self, page_number: int, record: dict) -> None:
        OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdf_sha256": self.fingerprint,
            "ocr_engine": OCR_ENGINE,
            "ocr_dpi": OCR_DPI,
            "page_record": record,
        }
        with self._cache_path(page_number).open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def extract_native_candidate(self, page: pymupdf.Page, page_number: int) -> dict:
        raw_text = page.get_text("text")
        text = clean_text(raw_text)
        lines = native_lines(page)
        return {
            "page": page_number,
            "extraction_method": "native",
            "text": text,
            "lines": lines,
            "diagnostics": {
                "native_text_chars": len(text),
                "text_chars": len(text),
                "korean_chars": len(re.findall(r"[가-힣]", text)),
            },
        }

    def extract_ocr_page(self, page: pymupdf.Page, page_number: int) -> dict:
        cached = self._read_cache(page_number)
        if cached is not None:
            return cached

        raw_native_text = page.get_text("text")
        text, lines, diagnostics = self.ocr.extract(page)
        record = {
            "page": page_number,
            "extraction_method": "ocr",
            "text": text,
            "lines": lines,
            "diagnostics": {
                **diagnostics,
                "native_text_chars": len(clean_text(raw_native_text)),
                "text_chars": len(text),
                "korean_chars": len(re.findall(r"[가-힣]", text)),
            },
        }
        if text.strip():
            self._write_cache(page_number, record)
        return record

    def extract_page(self, page: pymupdf.Page, page_number: int) -> dict:
        if self.extraction == "ocr":
            return self.extract_ocr_page(page, page_number)

        native = self.extract_native_candidate(page, page_number)
        if self.extraction == "native":
            if not text_is_usable(native["text"]):
                native["extraction_method"] = "native_unusable"
            return native
        if choose_auto_extraction([native]) == "native":
            return native
        return self.extract_ocr_page(page, page_number)
