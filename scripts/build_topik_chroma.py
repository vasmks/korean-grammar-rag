"""Build the TOPIK question vector index."""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


INPUT_FILE = Path("data/topik/processed/topik_retrieval_units.json")
CHROMA_DIR = Path("data/chroma_topik")
COLLECTION_NAME = "topik_examples"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def make_document(unit: dict) -> Document:
    question_start = unit["question_start"]
    question_end = unit["question_end"]
    question_label = (
        f"Question {question_start}"
        if question_start == question_end
        else f"Questions {question_start}-{question_end}"
    )

    # E5 expects this prefix for indexed passages.
    page_content = (
        "passage:\n"
        f"TOPIK {unit['exam_number']} reading exam\n"
        f"{question_label}\n"
        f"{unit['text']}"
    )
    metadata = {
        "source_type": "topik",
        "exam_number": unit["exam_number"],
        "section": unit["section"],
        "question_start": question_start,
        "question_end": question_end,
        "pages": ",".join(str(page) for page in unit["pages"]),
        "source_file": unit["source_file"],
        "chunk_type": unit["chunk_type"],
        "instruction_id": unit.get("instruction_id", ""),
        "shared_context_id": unit.get("shared_context_id") or "",
        "structure_confidence": unit.get("structure_confidence", ""),
    }
    return Document(page_content=page_content, metadata=metadata)


def make_id(unit: dict) -> str:
    return (
        f"topik_{unit['exam_number']}_"
        f"{unit['question_start']}_{unit['question_end']}"
    )


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as file:
        units = json.load(file)

    usable_units = [
        unit
        for unit in units
        if unit.get(
            "indexable", unit.get("content_type") != "visual_or_sparse"
        )
    ]
    print(f"Loaded {len(units)} retrieval units.")
    print(f"Usable text units : {len(usable_units)}")
    print(f"Skipped visual    : {len(units) - len(usable_units)}")

    if CHROMA_DIR.exists():
        print(f"Removing old vector store: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Embedding TOPIK documents...")
    vectorstore.add_documents(
        documents=[make_document(unit) for unit in usable_units],
        ids=[make_id(unit) for unit in usable_units],
    )

    print(f"Stored documents : {vectorstore._collection.count()}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Directory        : {CHROMA_DIR}")


if __name__ == "__main__":
    main()
