import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import KoreanGrammarRAG
from app.topik_preview import make_preview_response, topik_preview_service


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"
INDEX_FILE = STATIC_DIR / "index.html"
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1, description="Question about Korean grammar or TOPIK content."
    )
    explain: bool = Field(
        default=True,
        description="Generate an explanation grounded in dictionary evidence.",
    )


class AskResponse(BaseModel):
    question: str
    explanation: str | None
    entries: list[dict[str, Any]]
    topik_occurrences: list[dict[str, Any]]
    topik_message: str | None = None
    sources: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Korean Grammar RAG")
    app.state.rag = KoreanGrammarRAG()
    logger.info("Korean Grammar RAG ready")
    yield
    logger.info("Shutting down Korean Grammar RAG")


app = FastAPI(
    title="Korean Grammar RAG",
    description=(
        "A grounded Korean grammar assistant using the Korean Basic Dictionary "
        "and authentic TOPIK reading material."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html was not found.")
    return FileResponse(INDEX_FILE)


@app.get("/health", tags=["System"])
def health(request: Request) -> dict[str, Any]:
    rag: KoreanGrammarRAG = request.app.state.rag
    return {
        "status": "ok",
        "dictionary_entries": len(rag.entries),
        "topik_documents": len(rag.topik.units),
    }


def add_topik_preview_urls(result: dict) -> dict:
    """Attach lazy-render URLs without coupling retrieval to the web layer."""
    for occurrence in result.get("topik_occurrences", []):
        exam_number = occurrence.get("exam_number")
        question_start = occurrence.get("question_start")
        question_end = occurrence.get("question_end")
        if None in (exam_number, question_start, question_end):
            continue
        occurrence["preview_url"] = (
            f"/topik/preview/{exam_number}/{question_start}/{question_end}"
        )
        key = topik_preview_service.make_key(
            int(exam_number), int(question_start), int(question_end)
        )
        metadata = topik_preview_service.metadata.get(key, {})
        crop_count = len(metadata.get("crops", []))
        occurrence["preview_urls"] = [
            occurrence["preview_url"]
            if crop_index == 0
            else f"{occurrence['preview_url']}?crop_index={crop_index}"
            for crop_index in range(crop_count)
        ] or [occurrence["preview_url"]]
    return result


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(payload: AskRequest, request: Request) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rag: KoreanGrammarRAG = request.app.state.rag
    try:
        result = add_topik_preview_urls(
            rag.ask(question=question, explain=payload.explain)
        )
        return AskResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail="The RAG request failed.") from exc


@app.get(
    "/topik/preview/{exam_number}/{question_start}/{question_end}",
    tags=["TOPIK"],
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Rendered crop from the original TOPIK PDF.",
        },
        404: {"description": "TOPIK crop metadata or PDF not found."},
    },
)
def topik_preview(
    exam_number: int,
    question_start: int,
    question_end: int,
    crop_index: int = Query(
        default=0,
        ge=0,
        description="Crop number for retrieval units spanning multiple PDF pages.",
    ),
):
    return make_preview_response(
        exam_number=exam_number,
        question_start=question_start,
        question_end=question_end,
        crop_index=crop_index,
    )
