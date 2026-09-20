import json
from functools import lru_cache
from pathlib import Path

import pymupdf
from fastapi import HTTPException
from fastapi.responses import Response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_FILE = PROJECT_ROOT / "data/topik/processed/topik_crop_metadata.json"
PDF_DIR = PROJECT_ROOT / "data/topik/raw"
RENDER_SCALE = 2.0


class TopikPreviewService:
    def __init__(self) -> None:
        if not METADATA_FILE.exists():
            self.metadata = {}
            return

        with METADATA_FILE.open(encoding="utf-8") as file:
            self.metadata = json.load(file)

    @staticmethod
    def make_key(exam_number: int, question_start: int, question_end: int) -> str:
        return f"{exam_number}:{question_start}:{question_end}"

    def get_metadata(
        self, exam_number: int, question_start: int, question_end: int
    ) -> dict:
        key = self.make_key(exam_number, question_start, question_end)
        item = self.metadata.get(key)
        if item is None:
            raise HTTPException(
                status_code=404, detail="TOPIK crop metadata not found."
            )
        return item

    @lru_cache(maxsize=64)
    def render(
        self,
        exam_number: int,
        question_start: int,
        question_end: int,
        crop_index: int = 0,
    ) -> bytes:
        """Render and cache a requested crop; no PNG is stored on disk."""
        item = self.get_metadata(exam_number, question_start, question_end)
        crops = item.get("crops", [])
        if crop_index < 0 or crop_index >= len(crops):
            raise HTTPException(status_code=404, detail="TOPIK crop not found.")

        crop = crops[crop_index]
        pdf_path = PDF_DIR / item["source_file"]
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="TOPIK PDF not found.")

        page_index = crop["page_index"]
        document = pymupdf.open(pdf_path)
        try:
            if page_index < 0 or page_index >= len(document):
                raise HTTPException(
                    status_code=500, detail="Invalid PDF page index."
                )
            page = document[page_index]
            clip = pymupdf.Rect(*crop["bbox"])
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
                clip=clip,
                alpha=False,
            )
            return pixmap.tobytes("png")
        finally:
            document.close()


topik_preview_service = TopikPreviewService()


def make_preview_response(
    exam_number: int,
    question_start: int,
    question_end: int,
    crop_index: int = 0,
) -> Response:
    image_bytes = topik_preview_service.render(
        exam_number, question_start, question_end, crop_index
    )
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
